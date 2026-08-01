# MLB Prediction Model — Complete Technical Documentation

Audience: an LLM (or engineer) with no prior context on this codebase, needing a complete
mental model of how every piece works, why it exists, and how it rolls up into a final
score/prop prediction. Nothing is intentionally omitted — this includes dead ends,
reverted experiments, and investigated-but-never-built ideas, because this project's own
practice is to keep that history visible rather than erase it.

All paths are relative to the repo root: `/Users/brettscully/Desktop/sports-models/mlb`.

---

## 0. One-paragraph summary

For a single plate appearance (PA), the model estimates a batter's and a pitcher's
true-talent rate for each of ~16 mutually-exclusive outcomes (strikeout, walk, single,
double, triple, home run, sac fly, etc.) using Marcel-style shrinkage of multi-season
Statcast/Baseball-Reference-style data, applies a batter-only "contact quality" correction
from Statcast expected-stats (xBACON, barrel rate, bat speed, sprint speed), combines
batter/pitcher/league rates via an odds-ratio ("log5"-style") formula, multiplies in ~7
independently-validated contextual factors (base-out state, platoon, park, weather, times
through the order, catcher framing + umpire + defense), renormalizes to a valid probability
distribution, and samples an outcome. A `TransitionTable` (built from real historical
base-out-state transition frequencies) converts that PA outcome into runs scored and the
next base-out state. A full game is simulated as a sequence of half-innings until 27 outs
(9 innings) using real/projected lineups, real/predictive bullpen usage, and real/forecast
weather. Monte Carlo (100s–1000s of trials) over full games produces the final score
distribution, win probability, and player props.

**Current validated full-stack performance** (oracle backtest, `validate_game_simulator.py`,
n=597-600 real 2024-2025 games, 200 trials/game — see §9 for exact protocol): **total score
MAE 3.409, margin MAE 3.445, straight-up (SU) win/loss accuracy 60.5%** at the ORIGINAL
n=597/200-trial protocol. **IMPORTANT CAVEAT (§11.7): a later re-validation at a tightened
protocol (n=995 games/500 trials, bootstrap CI on Brier score) found this figure's own
noise floor is wide enough that it should not be treated as a precise fact** — the same
re-validation measured **total MAE 3.402, margin MAE 3.524, SU 59.0%, Brier 0.2343** for
the identical current-production code, and found that bat speed and pulled-air rate (two
of the three factors below) do NOT show a statistically significant full-stack effect
under the more rigorous test, despite being kept on the strength of an SU improvement at
the original protocol. See §11.7 before treating any SU/MAE figure in this document as
more precise than "roughly this, ±1-2 points." See §11 for the full ledger of signals
investigated this session, including several that did NOT clear the bar and were reverted.

---

## 1. Architecture overview

```
raw MLB Stats API + Statcast  (src/ingest/fetch.py)
        │
        ▼
per-PA table (src/ingest/build_pa_table.py)          per-pitch table (src/ingest/build_pitch_table.py)
        │                                                     │  [investigated, NOT wired into production — §6]
        ▼
walk-forward true-talent rate engine (src/models/true_talent.py)
        │
        ▼
batter-only contact-quality correction (src/models/expected_stats.py)
        │
        ▼
odds-ratio matchup combine: batter × pitcher × league (src/models/matchup.py)
        │
        ▼
contextual multipliers, applied per-outcome-category (7 independent modules — §4):
  state · platoon · park · weather · times-through-order · catcher/umpire · defense
        │
        ▼
renormalize → single-PA outcome distribution (src/models/game_simulator.py:combine_matchup_distribution)
        │
        ▼
sample outcome → TransitionTable (src/models/base_out_transitions.py) → runs + next base-out state
        │
        ▼
GameSimulator: repeat per PA until 3 outs × 2 halves × 9 innings (src/models/game_simulator.py)
        │
        ▼
Monte Carlo aggregation → score distribution / win prob / props
        │
        ├── validate_game_simulator.py       (oracle backtest — real historical bullpen sequence)
        ├── validate_predictive_bullpen.py   (honest backtest — predicted bullpen usage)
        ├── validate_prop_calibration.py     (are per-prop probabilities calibrated?)
        └── props.py + pipeline/*.py         (live daily generation)
```

Every contextual factor and every batter-quality signal in this system was independently
validated via a **leakage-free methodology** before being wired in: build the signal from
data available strictly *before* the target season/game, predict the *real* target-season
outcome, and check whether it improves prediction *beyond what the current production
signal already captures* (an incremental-R² test, not a standalone correlation). Then, for
anything genuinely touching the simulator's outcome distribution, a **full-stack isolated
A/B test** (same seed, same n≈597-600 real games, one temporarily-disabled copy of the
production file vs. the real one) is the final arbiter — total MAE, margin MAE, and SU
accuracy (weighted most heavily) must show genuine net improvement, or the change is
reverted. This discipline is why §11 lists as many reverted ideas as kept ones.

---

## 2. Data ingestion

### 2.1 `src/utils/paths.py`
Central path constants (`DATA_RAW`, `DATA_PROCESSED`, etc.) used by every other module —
no business logic.

### 2.2 `src/ingest/fetch.py`
Wraps the MLB Stats API and Baseball Savant/Statcast CSV endpoints. Key functions:
- `fetch_schedule_season(season)` — full season schedule/results; capped at "today," cannot
  reach future dates.
- `fetch_schedule_day(date)` — used instead of the season fetcher when a **future** date's
  probable pitchers/lineups are needed (the season fetcher structurally can't return
  future-dated games).
- `fetch_statcast_season(season)` — full pitch-by-pitch Statcast data for a season.
- `verify_and_backfill_statcast(season)` — a cross-source completeness check specifically
  added because "a silent partial-fetch that raises no exception once cost the sibling NFL
  project two whole missing seasons" (per `daily_update.py`'s docstring) — compares fetched
  row counts/date coverage against the schedule and re-fetches gaps.
- `fetch_sprint_speed_seasons(seasons)` — Baseball Savant's season-aggregate Sprint Speed
  leaderboard (a biometric not derivable from per-pitch data).
- Various other Statcast leaderboard fetchers (used by `expected_stats.py`'s bat-speed/
  fastball-stuff loaders, `spray.py`'s pull-rate builder, etc.)

### 2.3 `src/ingest/build_pa_table.py`
Builds the per-plate-appearance table (`pa_table_*.parquet`) that is the foundation for
**every** downstream rate estimate. One row per completed PA.

**`PA_COLUMNS`**: game_pk, season, game_date, at_bat_number, inning, inning_topbot, batter,
pitcher, catcher, stand (batter hand), p_throws (pitcher hand), balls, strikes, outs_when_up,
on_1b/on_2b/on_3b (runner identity), events (raw Statcast event string), description,
home_team, away_team, and derived columns below.

**`OUTCOME_CATEGORIES` / `EVENT_TO_CATEGORY`** — maps Statcast's raw `events` string to one
of the ~16 mutually-exclusive PA outcome categories used everywhere downstream: `strikeout`,
`walk`, `hit_by_pitch`, `single`, `double`, `triple`, `home_run`, `field_out`, `double_play`,
`triple_play`, `fielders_choice`, `field_error`, `sac_fly`, `sac_bunt`, `catcher_interf`,
`intent_walk` (this last one split out from ordinary `walk` specifically because it's a
*strategic* decision, not a pitch-outcome, and is modeled separately in
`game_simulator.py` — see §7).

A subtle, explicitly-avoided bug: grouping to find "the last row of at-bat X" must use
`.first()`/`.last()` semantics correctly — an earlier draft used `.nth(0)` in a context
where it silently returned the wrong row under certain group-key orderings; fixed and now
used consistently and deliberately (e.g. `player_game_snapshot`'s "rate as of first PA in
game" convention explicitly relies on a correct `.nth(0)` after sorting by
`at_bat_number`).

### 2.4 `src/ingest/build_pitch_table.py` — feeds the (currently unwired) pitch-level investigation
Builds a per-**pitch** (not per-PA) table, classifying every pitch's Statcast `description`
into one of 6 `PITCH_RESULT` categories — see §6.1 for the full classification. This table
exists and is queryable but as of now only `pitch_talent.py`/`validate_pitch_model.py`
consume it; `game_simulator.py` does not.

---

## 3. Core statistical engine

### 3.1 `src/models/true_talent.py` — Marcel-style true-talent rate estimation

The foundational rate estimator for every batter/pitcher/outcome-category combination.
Produces a **walk-forward-safe** ("pregame") rate: at any point in a season, the rate
reflects only data available strictly before that PA — no lookahead.

**Algorithm, in order:**
1. **Recency-weighted 3-season prior**: combines the player's most recent 3 seasons of
   real outcome rate with weights **`MARCEL_WEIGHTS = [5, 4, 3]`** (most recent season
   weighted highest — the standard Marcel weighting scheme).
2. **Regression to league average**: the weighted-3-season observed rate is blended with
   the league-average rate for that outcome, using a **stabilization constant** (in
   plate-appearances or relevant denominator) per `(player_col, outcome)` pair — these
   constants are based on Russell Carleton's published stabilization-point research
   (`STABILIZATION_PA_BATTER` / `STABILIZATION_PA_PITCHER` dicts, one entry per outcome
   category, batter side generally needing fewer PA to stabilize than pitcher side for the
   same outcome).
3. **In-season Bayesian blend**: once the target season is underway, the preseason prior is
   further blended with the player's own current-season-to-date actual outcome counts,
   using the same shrinkage pattern applied throughout this codebase:
   `(observed + PRIOR_WEIGHT * prior_rate) / (observed_n + PRIOR_WEIGHT)`.
4. **Age adjustment**: a Tango-style age curve (peak age 29, asymmetric slopes below vs.
   above peak) nudges the blended rate up or down based on the player's age that season,
   with a per-`(player_col, outcome)` **`AGE_ADJ_SIGN_OVERRIDE`** dict — some outcome
   categories logically improve with the *same* direction of aging effect as others move
   oppositely (e.g. contact-outcome rates vs. power-outcome rates), so the sign of the
   generic age curve is flipped per-category where domain logic requires it.

Key functions: `build_preseason_priors` (steps 1-2, prior to any current-season data),
`build_pregame_rates` (adds step 3's in-season blend, walk-forward per PA), age adjustment
helper implementing step 4.

**`STABILIZATION_PA` module constant** referenced by `game_simulator.py`'s import — the
league-average-level stabilization set used when no player-specific override applies.

### 3.2 `src/models/matchup.py` — odds-ratio ("log5"-generalization) combination

**`odds(p)`**: `p / (1 - p)`, with `p` clipped away from exactly 0/1 to avoid infinities —
this exact clip-before-odds discipline is repeated in every other module that computes an
odds ratio in this codebase (a recurring bug class after several real blowups — see §11).

**`combine_odds_ratio(league_rate, batter_rate, pitcher_rate)`** — the core formula:

```
matchup_odds = odds(league_rate) * (odds(batter_rate) / odds(league_rate)) * (odds(pitcher_rate) / odds(league_rate))
matchup_prob = matchup_odds / (1 + matchup_odds)     # prob_from_odds
```

Interpretation: start from the league-average odds of this outcome, then multiply in the
batter's own odds-ratio *relative to league average* and the pitcher's own odds-ratio
*relative to league average*. This is the standard sabermetric "log5" combination
generalized from a binary (win/lose) formulation to an arbitrary rate. Applied
**independently to each of the ~16 outcome categories** — not just to one headline stat.

`prob_from_odds(odds)` — inverse of `odds()`.

Also hosts `calibration_report` (bucketed real-vs-predicted calibration check, reused as
the template for other modules' own calibration reports, e.g. `validate_prop_calibration.py`).

### 3.3 `src/models/base_out_transitions.py` — `TransitionTable`

Converts a sampled PA outcome (e.g. "single" with runners on 1st/2nd, 1 out) into (a) runs
scored on the play and (b) the next base-out state — via **nonparametric bootstrap
resampling** of real historical transitions, not a hardcoded rule table (this correctly
captures real-world variation like runner-advancement decisions, error scoring, etc.).

**3-tier fallback for sparse states:**
1. Exact match: this exact (base-out-state, outcome) combination in the historical PA table.
2. Pooled by out-count only: if the exact base-state is too sparse, pool across all base
   states sharing the same out count.
3. Pooled by outcome only: final fallback, pooled across all states for that outcome
   category league-wide.

`TransitionTable(pa)` is built once per validation/generation run from the full historical
PA table and reused across every simulated trial.

---

## 4. Contextual factors

Every factor below is a **multiplicative adjustment**, applied to one or more outcome
categories' unnormalized probability, computed independently ahead of time from real data,
and validated leakage-free before being wired in. All factors for a given PA are applied
in sequence, then the full outcome vector is renormalized to sum to 1 (§7.2). Where a
factor is irrelevant to a category (e.g. catcher framing has no effect on home runs), its
multiplier for that category is fixed at neutral (1.0).

### 4.1 `src/models/platoon_splits.py`
Same-hand/opposite-hand batter vs. pitcher matchup effect, computed **separately for the
batter's own platoon split AND the pitcher's own platoon-split-allowed**, both applied
(not just one). Built as a Bayesian-shrunk odds-ratio multiplier per outcome category (same
`(observed + PRIOR*reference)/(observed_n + PRIOR)` shrinkage pattern), clipped both sides
before the odds-ratio to prevent blowups — after a real, confirmed bug where an early
unclipped version produced up to a **146,611x** multiplier for a near-zero-sample edge case.

### 4.2 `src/models/park_factors.py`
Park-specific per-outcome-category multiplier (not just a single overall "runs" factor —
separate factors for e.g. home_run vs. single vs. double), mean-normalized to 1.0 across
the league each season, built from multi-season real park data. An early unclipped/
unguarded version produced `inf` values for sparse park-season-outcome combinations —
fixed with the same clip-before-combine discipline.

### 4.3 `src/models/weather.py` + `src/models/weather_forecast.py`
Temperature/wind-bucketed multiplier, further split by the batter's own handedness AND
their individual career pull-tendency tercile (via `spray.py`'s pull-rate builder) — wind
blowing out to the batter's own pull field matters more than a generic "wind out" factor.
`weather_forecast.py` provides the live/future-game analog: since a future game has no
posted actual conditions, it fetches a park's own climatological wind/temperature
distribution for that calendar month (via Open-Meteo) and **samples** a bucket fresh per
Monte Carlo trial (propagating real day-to-day weather uncertainty), rather than assuming
a single expected-value bucket. An early unclipped version of the raw weather multiplier
hit a documented **73x** blowup on a sparse bucket before the same clip fix was applied.

### 4.4 `src/models/ttop.py` — times-through-the-order penalty
A pitcher facing the same batter for the Nth time in a game becomes progressively less
effective (a well-established sabermetric effect) — applied as a multiplier keyed by
times-through-order, capped at 3 (a 4th+ time through treated identically to the 3rd, since
sample sizes for 4+ get too sparse to trust a further-differentiated estimate).

### 4.5 `src/models/spray.py` + `src/models/park_orientation.py`
`spray.py` builds each batter's own walk-forward pull-rate tercile (used by `weather.py`'s
wind-direction-relative-to-pull-field logic and by `expected_stats.py`'s groundball-single
layer's scaling). `park_orientation.py` provides each park's physical orientation (needed
to know which wind direction is "blowing out toward the batter's pull side" for a specific
stadium).

### 4.6 `src/models/catcher_framing.py`
Catcher's real framing skill (extra called strikes earned on borderline pitches) as a
multiplier on `strikeout`/`walk` categories specifically (no effect on batted-ball
categories) — walk-forward Bayesian-shrunk rate.

### 4.7 `src/models/umpire_factor.py`
Home-plate umpire's own real called-strike-zone tendency, same mechanism as catcher
framing (affects `strikeout`/`walk` only). In the simulator, umpire and catcher factors are
**merged multiplicatively into the same factor dict** before being passed in (one umpire
calls the whole game, so it's applied identically to both teams' catcher-factor dicts)
rather than threaded as a separate simulator parameter.

### 4.8 `src/models/defense_factor.py`
Team's real per-game defensive alignment, from Statcast OAA (Outs Above Average), split
infield vs. outfield — applied as a multiplier on `single`/`double`/`triple` categories
(batted balls converted to outs by good defense). Merged into the same factor dict as
catcher/umpire (disjoint outcome keys: defense touches hit categories, catcher/umpire touch
strikeout/walk, so merging is safe).

### 4.9 `src/models/baserunning.py`
Real per-runner stolen-base attempt-rate and success-rate, walk-forward, keyed by lineup
slot (not raw player_id, since `GameSimulator` tracks runners by lineup position identity
as they're placed on base — see §7.1's `_reassign_runners`).

### 4.10 `src/models/blowout.py`
Detects blowout situations (8+ run deficit, 8th inning or later) and substitutes a
position-player-pitching profile (a real, if rare, MLB occurrence) rather than continuing
to simulate a real relief pitcher who wouldn't actually be used in that situation.

### 4.11 `src/models/bullpen.py`
The **predictive** bullpen-usage model (as opposed to `validate_game_simulator.py`'s
oracle use of the *real* historical bullpen sequence — see §9.1 vs §9.2). Builds:
- `build_expected_starter_innings` — each starter's own walk-forward projected innings
  before the bullpen typically takes over.
- `build_team_bullpen_roster` — weighted roster of available relievers, weighted by recent
  usage frequency and discounted by rest-day availability for the target game's date.
- `sample_bullpen_plan` — samples a specific reliever per remaining inning from the
  weighted roster (not a flat blended "average reliever" rate), respecting a closer
  identification (`identify_closer`) for save situations.
- Fallback: `nearest_prior_bullpen` — a team's most recent real bullpen usage snapshot when
  no fresher roster/rest data exists.

---

## 5. `src/models/expected_stats.py` — batter-side contact-quality corrections

**Design principle:** these corrections are applied directly to a batter's own `rates`
dict at profile-build time — a fixed, opponent-independent true-talent-level adjustment,
the same tier as the Marcel-shrunk rate itself, *not* threaded through the odds-ratio
matchup combine as another contextual factor. Batter-side only: pitcher-side signals
(CSW%, fastball stuff) were investigated (see FIRST/SIXTH sub-sections below) but did not
survive the full-stack A/B bar.

Built incrementally as 6 "layers," each independently leakage-free-validated. Status is
marked explicitly per layer — **BUILT AND LIVE**, **BUILT THEN REVERTED**, or
**INVESTIGATED, NOT BUILT** — because that distinction is exactly what the user asked not
to lose.

### FIRST LAYER — xBACON / `contact_quality_multiplier` — **BUILT AND LIVE**

Real BACON (hits per ball-in-play) correlates only 0.25-0.30 season-to-season; Statcast's
`estimated_ba_using_speedangle` ("xBACON") correlates 0.39 with *next*-season real BACON —
and in a multivariate test, adding real BACON on top of xBACON moves R² only
0.152→0.153 / 0.149→0.151 (real BACON adds essentially nothing once xBACON is known).

- `XBACON_PRIOR_BIP = 300` (Bayesian shrinkage prior, in pseudo-balls-in-play).
- `CONTACT_QUALITY_CLIP_MIN, CONTACT_QUALITY_CLIP_MAX = 0.10, 0.60`.
- `odds(p)`: `clip(p, 1e-6, 1-1e-6)` then `p/(1-p)`.
- `contact_quality_multiplier(rates, pregame_xbacon, pregame_bat_speed=None)`:
  `hit_rate = Σ rates[HIT_EVENTS]`, `bip_out_rate = Σ rates[BIP_OUT_EVENTS]`; returns 1.0
  (neutral) if the denominator is ~0. Else `existing_bip_hit_share = hit_rate/(hit_rate+bip_out_rate)`,
  both this and `xbacon` clipped to `[0.10, 0.60]`; base case (no bat speed) returns
  `odds(xbacon) / odds(existing_bip_hit_share)`.
- `HIT_EVENTS = {single, double, triple, home_run}`.
  `BIP_OUT_EVENTS = {field_out, double_play, fielders_choice, field_error, sac_fly, sac_bunt, triple_play, catcher_interf}`.

### SECOND LAYER — hit-type splits — **MIXED**

Tested two candidate hit-type splits, leakage-free, multivariate against the existing
real-outcome-history predictor:

- **home_run share among hits vs. barrel rate** (`launch_speed_angle==6`): barrel rate's
  own coefficient (0.51-0.55) exceeds real-outcome-share's (0.31-0.42); R² **0.40→0.43**
  and **0.41→0.44**. → **BUILT.**
- **double+triple share among hits vs. xSLG**: R² stays ~0.01-0.03 for *either* predictor —
  neither history nor xSLG meaningfully predicts this split. → **NOT BUILT**, left to
  `true_talent.py`'s existing Marcel-shrunk double/triple-share estimate untouched.

#### barrel rate / `hr_share_multiplier` — **BUILT AND LIVE**
- `BARREL_PRIOR_BIP = 300` (reused `XBACON_PRIOR_BIP`'s value — the method-of-moments fit
  was numerically unstable for this metric specifically).
- `HR_SHARE_INTERCEPT = 0.0442`, `HR_SHARE_COEF_EXISTING = 0.3616`, `HR_SHARE_COEF_BARREL = 0.5300`
  (averaged 2023→2024 / 2024→2025 leakage-free fits).
- `HR_SHARE_CLIP_MIN, HR_SHARE_CLIP_MAX = 0.02, 0.35` — **not cosmetic**: without this clip,
  the regression's nonzero intercept means it never predicts near 0%, so a weak-power
  batter with near-zero existing HR-share produced multipliers up to **13.8x** across 989
  real batters (same failure class as platoon/weather's blowups).
- `hr_share_multiplier(rates, pregame_barrel_rate)`: applies to `home_run` only;
  `existing_hr_share = clip(rates.home_run/hit_rate, 0.02, 0.35)`;
  `predicted = clip(INTERCEPT + COEF_EXISTING*existing + COEF_BARREL*pregame_barrel_rate, 0.02, 0.35)`;
  returns `odds(predicted)/odds(existing)`.

### THIRD LAYER — sprint speed / `groundball_single_multiplier` — **BUILT AND LIVE**

Real hit rate on groundballs (`launch_angle<10`) vs. Baseball Savant Sprint Speed:
multivariate R² **0.088→0.141** (2023→2024) and **0.024→0.043** (2024→2025) adding sprint
speed on top of real groundball-BACON history — both meaningfully positive coefficients.
Control test on *overall* (non-GB) single rate found ~nothing (R² flat), confirming the
signal is real but specific to groundball contact.

Applied to `single` only, **scaled by the batter's own groundball rate** (a fly-ball hitter
gets almost no adjustment):
- `GROUNDBALL_SINGLE_INTERCEPT = 0.0066`, `GROUNDBALL_SINGLE_COEF_REAL = 0.1755`,
  `GROUNDBALL_SINGLE_COEF_SPEED = 0.00759`.
- `GROUNDBALL_BACON_CLIP_MIN, MAX = 0.05, 0.55`. `GROUNDBALL_LAUNCH_ANGLE_MAX = 10`.
  `GROUNDBALL_STABILIZATION_BATTED_BALLS = 100`.
- `groundball_single_multiplier(pregame_bacon_gb, pregame_sprint_speed, pregame_gb_rate)`:
  `full_mult = odds(predicted)/odds(existing)` (clipped both sides as above); final
  return is `1.0 + gb_rate * (full_mult - 1.0)` — the "scaled toward neutral for fly-ball
  hitters" blend.

### FOURTH LAYER — whiff rate on swings — **INVESTIGATED, BUILT, THEN REVERTED** (no live code remains)

Two sub-attempts:
1. **Matchup-conditional framing** (batter K rate vs. pitcher-arsenal terciles, same
   Bayesian-odds-ratio discipline as `platoon_splits.py`) — tested on strikeout/home_run/
   single, **all three came back negative** (made predictions slightly worse). Not built.
2. **Pure batter-side whiff-rate-on-swings** signal (architecturally identical to
   `hr_share_multiplier`) — leakage-free incremental R² was genuinely positive (+0.0134,
   n=720), so it was built as `whiff_rate_multiplier` on `strikeout`, then full-stack A/B
   tested: **total MAE improved slightly (3.405→3.379) but margin MAE got worse
   (3.462→3.482) and SU accuracy got worse (59.3%→58.0%, -1.3pp)**. **REVERTED.** Confirmed
   via grep: `whiff_rate_multiplier` does not exist anywhere in the codebase today — only
   described in this historical docstring text.

### FIFTH LAYER — bat speed extension to `contact_quality_multiplier` — **BUILT AND LIVE**

Statcast bat-tracking bat speed (~93-97% swing-row coverage 2024+, ~65% batter-coverage
even in 2023's mid-season rollout with a 100-real-swing minimum, `BAT_TRACKING_MIN_SWINGS = 100`).
Leakage-free test (2 season-pairs, n=756): does bat speed add value beyond xBACON alone?
R² **0.139 → 0.183** (~32% relative gain). Real-data limit test: multiplier range **0.51-1.86**
(no blowup, comparable to the function's existing range). **BUILT** — this is the one net-new
kept factor this session; see §11 for the full-stack A/B numbers.

- `CONTACT_QUALITY_BATSPEED_INTERCEPT = -0.02000`
- `CONTACT_QUALITY_BATSPEED_COEF_XBACON = 0.33275`
- `CONTACT_QUALITY_BATSPEED_COEF_BATSPEED = 0.00330`
- Live in `contact_quality_multiplier(rates, pregame_xbacon, pregame_bat_speed=None)`: when
  bat speed is available, `predicted = clip(INTERCEPT + COEF_XBACON*xbacon_clipped +
  COEF_BATSPEED*pregame_bat_speed, 0.10, 0.60)`, returns `odds(predicted)/odds(existing)`.
  Falls back to the FIRST LAYER's pure-xBACON path when bat speed is `None`/`NaN` (rookies,
  pre-2023 seasons).
- Supporting: `BAT_TRACKING_SWING_DESC`, `load_bat_speed_by_season(seasons)` (drops
  batters with <100 real swings that season), `build_pregame_bat_speed(...)` (walk-forward:
  most recent prior season's measured value, no in-season blend, a direct season snapshot).
- **Confirmed wired into**: `props.py`, `validate_predictive_bullpen.py`,
  `validate_game_simulator.py`'s `build_profile` (all pass `pregame_bat_speed` through
  `apply_contact_quality`).

#### HR-share-with-bat-speed extension — **INVESTIGATED, NOT BUILT**
Same bat-speed test applied to HR-share-among-hits showed real incremental value (R²
0.379→0.405) but **failed its own real-data limit test**: multipliers up to **5.5x** for
batters with zero real home runs in-sample (the same "regression intercept prevents
near-zero prediction" bug class). A tighter clip floor (0.10) would tame it to 1.75x but
would over-clip genuine weak-power hitters (real 1st-percentile HR-share ≈0.03). Decision:
do not deploy this variant. No functions/constants for it exist in the file.

### SEVENTH LAYER — pulled-air-ball rate extension to `hr_share_multiplier` — **BUILT AND LIVE**

Direction-based batter signal (not power) — a batter's tendency to pull the ball in the air,
distinct from barrel rate (power/contact quality) and distinct from the rejected
HR-share-with-bat-speed extension (different physical mechanism). Computed from this
project's own PA table: `compute_pull_angle` (§4.5, `spray.py`) combined with
`launch_angle > GROUNDBALL_LAUNCH_ANGLE_MAX` as the "air" definition (the raw Statcast feed
here has no separate bb_type classification for a narrower "fly ball" cutoff).

- **Sanity check against external data** (own 2023-2025 PA table): 17.8% HR rate on
  pulled-air balls-in-play vs. 3.96% on non-pulled-air — pulled-air balls account for 67.5%
  of all real home runs, closely matching externally-cited figures.
- **Leakage-free incremental-R² test** (2 season-pairs, n=387/385, same target as
  `hr_share_multiplier` — next-season real HR-share-among-hits): does pulled-air rate add
  value beyond existing HR-share + barrel rate? R² **0.4259→0.4371** (2023→2024) and
  **0.3876→0.4105** (2024→2025) — both positive, comparable in magnitude to barrel rate's
  own original gain.
- `PULLED_AIR_PRIOR_BIP = 300` — Bayesian-shrinkage prior (pseudo-BIP, same value as
  `BARREL_PRIOR_BIP`), shrinking `pulled_air_rate` toward the league-average rate
  (~17% every season 2023-2026, quite stable) before it enters the regression — this
  modestly improved both the R² gain and the real-data tail vs. an unshrunk version.
- **`HR_SHARE_PULLEDAIR_INTERCEPT = -0.03469`**
- **`HR_SHARE_PULLEDAIR_COEF_EXISTING = 0.21480`**
- **`HR_SHARE_PULLEDAIR_COEF_BARREL = 0.64574`**
- **`HR_SHARE_PULLEDAIR_COEF_PULLED_AIR = 0.50780`**
- **Real-data limit test**: worst-case multiplier 5.64-5.90x — worse than, but the *same
  order of magnitude* as, the CURRENT LIVE 2-term `hr_share_multiplier`'s own already-existing
  5.14x worst case on the same real batters (same root cause: a nonzero regression intercept
  can't predict a real `existing_hr_share` of exactly 0). This is a modest widening of a tail
  this project already ships with today — not a new order-of-magnitude blowup class like the
  rejected HR-share-with-bat-speed attempt — which is why this one was built and full-stack
  tested rather than rejected on the limit test alone.
- `hr_share_multiplier(rates, pregame_barrel_rate, pregame_pulled_air_rate=None)`: when
  `pregame_pulled_air_rate` is given, blends existing+barrel+pulled_air via the 3-term
  regression above (clipped to the same `[0.02, 0.35]` bounds); when `None`, falls back to
  the original 2-term existing+barrel blend unchanged.
- New walk-forward loaders mirror `build_pregame_barrel_rate`'s exact structure:
  `_season_pulled_air_sums`, `_preseason_pulled_air_priors`, `build_pregame_pulled_air_rate`,
  `player_game_pulled_air_snapshot`. A real nullable-extension-dtype bug (`hc_x`/`hc_y`/
  `launch_angle` propagating `pd.NA` through comparisons, crashing `.astype(int)`) was caught
  and fixed via a `_clean_pulled_air_inputs` helper casting all three to `float64` first —
  the same fix pattern already used for barrel rate's `launch_speed_angle`.
- **Confirmed wired into**: `props.py`, `validate_predictive_bullpen.py`,
  `validate_game_simulator.py` — all three pass `pregame_pulled_air_rate` through
  `apply_contact_quality`.
- **Full-stack A/B (n=597, on top of the already-kept auto-runner fix)**: total MAE
  3.404→3.409 (flat), margin MAE 3.444→3.445 (flat), **SU accuracy 59.3%→60.5% (+1.2pp)** —
  the single largest SU gain of any factor tested this session, at essentially zero cost.

### SIXTH LAYER — pitcher fastball-stuff / `pitcher_stuff_k_multiplier` — **BUILT, NOT WIRED IN (dead code)**

Raw fastball velocity + spin rate (not outcome-derived, unlike the earlier-rejected CSW%
idea) vs. the pitcher's production K-rate signal. Leakage-free test (n=776): R²
**0.396 (existing alone) → 0.409 (+velo) → 0.419 (+velo+spin)**, ~6% relative gain. Real-data
limit test: multiplier range 0.81-1.23, mean 0.98 (well-behaved). Built and full-stack A/B
tested — **result: net negative on SU accuracy** (component-level signal was real, but did
not survive contact with the full-stack simulator — see §11). **Reverted from all 3
consumer files; functions/constants remain in `expected_stats.py` only, confirmed via grep
to have zero call sites anywhere else in the repo:**
- `FASTBALL_TYPES_FOR_STUFF = {"FF", "SI", "FC", "FA"}`
- `PITCHER_STUFF_K_INTERCEPT = -0.240952`
- `PITCHER_STUFF_K_COEF_EXISTING = 0.699641`
- `PITCHER_STUFF_K_COEF_VELO = 0.00233966`
- `PITCHER_STUFF_K_COEF_SPIN = 0.0000380032`
- `PITCHER_STUFF_K_CLIP_MIN, MAX = 0.05, 0.45`
- `load_fastball_stuff_by_season`, `build_pregame_fastball_stuff`, `pitcher_stuff_k_multiplier`,
  `apply_pitcher_stuff` — all present in the file, all unused elsewhere.

### `apply_contact_quality` — the actual live entry point (current signature)

```python
def apply_contact_quality(
    rates: dict[str, float],
    pregame_xbacon: float | None,
    pregame_barrel_rate: float | None = None,
    pregame_bacon_gb: float | None = None,
    pregame_sprint_speed: float | None = None,
    pregame_gb_rate: float | None = None,
    pregame_bat_speed: float | None = None,
) -> dict[str, float]:
```
Order of operations on a copy of `rates`: (1) all HIT_EVENTS categories scaled by
`contact_quality_multiplier` (xBACON, optionally bat-speed-blended); (2) `home_run` further
scaled by `hr_share_multiplier`; (3) `single` further scaled by `groundball_single_multiplier`
(requires all three of `pregame_bacon_gb`/`pregame_sprint_speed`/`pregame_gb_rate` non-null
— any one missing skips only that layer). Any subset of arguments may be `None`; a missing
signal is a no-op for that layer only, never a silent fallback to a different signal.

---

## 6. The pitch-by-pitch investigation (Phase 1 + Phase 2) — **not in the live production path today**

This was the deepest investigation this session: could modeling every individual pitch
(not just aggregate PA-outcome rates) genuinely improve the simulator? Full honest
history below.

### 6.1 `src/models/pitch_groups.py` — shared pitch-level classification
- `FASTBALL_TYPES = {FF, SI, FC, FA}`, `BREAKING_TYPES = {SL, CU, ST, KC, SV, KN}`,
  `OFFSPEED_TYPES = {CH, FS, FO, EP}`. `pitch_group(pitch_type)`.
- `count_group(balls, strikes)`: `"2strike"` if strikes==2; `"hitter_ahead"` if balls≥2
  and strikes≤1; else `"neutral"`.
- `swing_count_group(balls, strikes)`: splits out 3-0 as its own bucket specifically for
  swing/take modeling — real 2025 swing rates 2-0: 40.3%, 2-1: 58.2%, 3-1: 54.0%, **3-0:
  only 8.1%** (a genuine behavioral discontinuity a plain `hitter_ahead` bucket would wash
  out). Used only for the swing/take decision; all other pitch-level decisions use the
  plain `count_group`.

### `PITCH_RESULT` classification (`build_pitch_table.py`)
6-way classification of Statcast's `description` field: `ball`, `called_strike`,
`swinging_strike`, `foul`, `hit_into_play`, `hit_by_pitch` — with automatic-ball/strike
pitch-clock violations (~0.3% of pitches) folded into ball/called_strike. Validated via a
throwaway deterministic count-state loop (pure count-conditional PITCH_RESULT distribution,
zero player skill) that reproduced real 2025 aggregate K/BB/HBP rates within 0.3pp.

### 6.2 `src/models/pitch_talent.py` — count-bucket-conditioned true-talent rates (Marcel one level down)

5 pitch-level binary decisions, each bucketed by count-leverage
(`DECISIONS = (swing, whiff, inplay, called_strike, hbp)`):
1. **swing** — pop: all pitches; bucket: `swing_count_group`.
2. **hbp** — pop: takes; bucket: `count_group`.
3. **called_strike** — pop: takes minus HBP; bucket: `count_group`.
4. **whiff** — pop: swings; bucket: `count_group`.
5. **inplay** — pop: swings minus whiffs (contact); bucket: `count_group`; complement=foul.

`STABILIZATION_PITCHES_BATTER = {swing:200, whiff:200, inplay:200, called_strike:200, hbp:2000}`
`STABILIZATION_PITCHES_PITCHER = {swing:250, whiff:250, inplay:250, called_strike:250, hbp:3000}`
(fit via this file's own leakage-free K-sweep, not carried over from `true_talent.py`).
`MARCEL_WEIGHTS = [5,4,3]` (same as `true_talent.py`). No age adjustment (deliberately
simplified vs. `true_talent.py`).

**`resolve_terminal_probs(bucket_rates)`** — exact forward dynamic-programming solve over
the 12 real `(balls, strikes)` states (small enough to solve exactly, not Monte Carlo).
States processed in topological order (`balls+strikes` ascending). Per state: look up
`p_swing`, `p_hbp_t`, `p_cs_t`, `p_whiff_s`, `p_inplay_s` (clipped `[1e-4, 1-1e-4]`, with
hardcoded fallback defaults 0.5/0.003/0.5/0.2/0.5 if missing). Derive `p_take=1-p_swing`;
`p_ball=p_take*(1-p_hbp_t)*(1-p_cs_t)`; `p_called_strike=p_take*(1-p_hbp_t)*p_cs_t`;
`p_hbp=p_take*p_hbp_t`; `p_whiff=p_swing*p_whiff_s`; `p_inplay=p_swing*(1-p_whiff_s)*p_inplay_s`;
`p_foul=p_swing*(1-p_whiff_s)*(1-p_inplay_s)`. **2-strike foul self-loop**: renormalizes
the other 5 outcome probabilities to exclude foul's no-op mass (`denom=max(1-p_foul,1e-9)`)
rather than modeling the self-loop directly, keeping the state graph a proper DAG.
Accumulates `hit_by_pitch`/`hit_into_play` directly into terminal probabilities; advances
balls (walk if 4th ball) and strikes (`p_strike_total=p_called_strike+p_whiff`, strikeout
if 3rd strike) otherwise. Returns exact `{strikeout, walk, hit_by_pitch, hit_into_play}`
absorption probabilities for a PA starting at (0,0).

**Phase 1 validation gate** (`validate_pitch_model.py`): composing a batter's OWN bucket
rates (vs. an exactly-average opponent) reproduces real full-season K/BB/HBP about as well
as `true_talent.py`'s direct approach — a genuine, real result.

### Phase 2 wire-in and revert — **the central finding of this investigation**

Phase 2 (per `pitch_talent.py`'s own `build_bucket_rate_snapshot` docstring) wired
`resolve_terminal_probs`, fed with real batter×pitcher×league-combined bucket rates (via
`matchup.py`'s `combine_odds_ratio`, one call per decision per bucket), into
`GameSimulator._pa_outcome`, wholesale replacing the strikeout/walk/hit_by_pitch mechanism.

**Full-stack A/B result: SU accuracy 59.3% → 55.6% — a clear regression. Reverted 2026-07-22.**

**Root-cause diagnosis** (not just "it failed" — this is the important part): the composed
K-rate distribution was measurably **under-dispersed** (variance-compressed) relative to
`true_talent.py`'s own estimate for the *same real matchups*. Directly confirmed by a
variance comparison across 3000 real batter-pitcher pairs: composed strikeout std **0.035**
vs. `true_talent.py`'s own **0.051** vs. real **0.063**. Because SU accuracy depends on
correctly *separating* strong and weak matchups, a compressed distribution makes every
game look more like a coin flip — explaining the accuracy collapse even though the
composed rate's *mean* calibration looked fine in isolation. A simple linear rescale
brought MAE close to parity with `true_talent.py` (strikeout raw MAE 0.0338 → rescaled
0.0329 vs. true_talent's 0.0324), but rather than deploy a standalone rescale (this
project's own isotonic-calibration revert is cited internally as a cautionary precedent —
fixing a marginal distribution in isolation can break joint coherence), the intended fix
was a fitted multivariate regression blend (`pitch_walk_multiplier`, walk-only — see below).

**Current status, confirmed via grep**: `game_simulator.py` imports only
`base_out_transitions`, `true_talent` (`STABILIZATION_PA`, `build_pregame_rates`),
`matchup` (`odds`, `prob_from_odds`), and `platoon_splits`. **Zero references** to
`pitch_talent.py`/`pitch_groups.py`/`validate_pitch_model.py` anywhere in
`game_simulator.py`. `true_talent.py`'s direct per-PA Marcel rates remain the actual
production mechanism. The entire pitch-by-pitch investigation (Phase 1 build → Phase 2
wire-in → Phase 2 revert) is fully out of the live path today.

### The regression-blend fix — designed, fit, but **never actually deployed**

A narrower fix was derived: rather than replace the whole K/BB/HBP mechanism, blend the
composed rate with the existing production rate via a fitted regression, same pattern as
`hr_share_multiplier`/`groundball_single_multiplier`. Leakage-free incremental-R² test
across all 3 no-contact outcomes both sides (n=670-687):
- batter strikeout: 0.5117→0.5120; batter HBP: 0.2282→0.2287; pitcher strikeout:
  0.2843→0.2844; pitcher HBP: 0.0927→0.0936 — **all ~zero incremental value, not pursued.**
- **batter walk: 0.3151→0.3613 (+0.046)**; **pitcher walk: 0.1441→0.1498 (+0.0057)** — real,
  meaningful jumps, batter side dominated by the composed signal's own coefficient (0.843
  vs. existing rate's 0.055).

Fit constants (`pitch_talent.py`):
- `PITCH_WALK_INTERCEPT = {batter: -0.01326, pitcher: 0.01458}`
- `PITCH_WALK_COEF_EXISTING = {batter: 0.05458, pitcher: 0.45357}`
- `PITCH_WALK_COEF_COMPOSED = {batter: 0.84306, pitcher: 0.28712}`
- `PITCH_WALK_CLIP_MIN, MAX = 0.02, 0.25`
- `pitch_walk_multiplier(existing_walk_rate, composed_walk_rate, player_col)`.

**Status**: confirmed via grep, `pitch_walk_multiplier`/`build_composed_rates` are defined
**only** in `pitch_talent.py` and are **not imported or called anywhere else** — not in
`game_simulator.py`, `props.py`, or any validator. This fix was fully specified and fit but
**never wired into production** — distinct from the "built then reverted" pattern above;
this one was never deployed at all, only derived as a candidate remedy for the Phase 2
revert. A future session could pick this up as a scoped, narrower retry (walk-only, not a
wholesale mechanism swap) — see §12.

### `validate_pitch_model.py`
The Phase 1/2a go/no-go gate script. Its own docstring statement that
"`resolve_terminal_probs` ... is load-bearing production code (Phase 2 wires it into
game_simulator.py)" is **now stale** — it describes the pre-revert state. Do not read that
docstring as reflecting current reality; this document's §6 status notes above are current.

---

## 7. `src/models/game_simulator.py` — the simulator core

### 7.1 `combine_matchup_distribution` — the per-PA outcome distribution

For one plate appearance, given the batter's rates (already contact-quality-adjusted per
§5), the pitcher's rates, league rates, and every resolved contextual factor (§4):

1. For each of the ~16 `OUTCOMES`, compute `combine_odds_ratio(league_rate, batter_rate, pitcher_rate)`
   (§3.2) → the pre-context matchup probability for that category.
2. Convert to `unnorm[outcome] = odds(matchup_prob)`, then multiply in every applicable
   contextual factor for that category (state, platoon batter, platoon pitcher, park,
   weather, TTOP, catcher+umpire+defense — each either a real value or neutral 1.0 if not
   applicable to that category).
3. `total = sum(unnorm.values())`; return `{o: v/total for o, v in unnorm.items()}` — a
   valid probability distribution summing to 1.
4. Accepts an optional `outcomes: list = OUTCOMES` parameter — the same renormalization
   logic is agnostic to which subset of `OUTCOMES` populated `unnorm`, so this parameter
   allows restricting the combine to a subset (used, e.g., if resolving only which of the
   12 in-play categories a "contact" event resolves to — see the (reverted) Phase 2b plan
   in the project's saved plan file for the never-fully-shipped version of this mechanism).

### 7.2 `GameSimulator` class

- **`simulate_half_inning(...)`**: loops PA-by-PA. For each PA: resolves the current
  batter/pitcher profiles, current base-out state, times-through-order count, and every
  contextual factor (park/weather/catcher+umpire+defense are per-game-fixed; state/platoon/
  TTOP are per-PA); calls `combine_matchup_distribution`; samples one outcome via the RNG;
  feeds the outcome into `TransitionTable.sample(state, outcome, rng)` to get runs scored
  and the next base-out state; advances the out count; ends the half-inning at 3 outs.
- **`simulate_game(home_lineup, away_lineup, home_pitcher, away_pitcher, innings=..., park_factors=..., weather_factors=..., home_bullpen=..., away_bullpen=..., blowout_pitcher_profile=..., home_catcher_factor=..., away_catcher_factor=..., home_sb_rates=..., away_sb_rates=...)`**:
  alternates top/bottom half-innings, tracks score, calls `_pitcher_for_inning` each half to
  resolve which pitcher is active (starter vs. bullpen-by-inning dict, or blowout
  substitution). Passes `auto_runner=inning >= 10` to every half-inning call — the real,
  permanent-since-2023 MLB rule that every half-inning from the 10th on starts with a
  runner on 2nd (not bases empty). `simulate_half_inning` seeds `state=20` (bitmask for
  "runner on 2nd, 0 outs" — a state that already occurs constantly in the real historical
  data, so no new state-factor/transition-table coverage was needed) when `auto_runner` is
  set; the runner's identity (only relevant when `sb_rates` tracking is active) is
  approximated as `(start_idx - 1) % 9` — this team's own last batter retired, since
  `start_idx` is always "next batter up for this team." Sanity-checked against real-world
  data before deploying: a league-average-everyone simulation gives 1.09 runs/half-inning
  with the auto-runner seeded, matching real-world figures (~1.00 runs/inning) almost
  exactly. Full-stack A/B (n=597, only `auto_runner` toggled): total MAE 3.370→3.404
  (worse), margin MAE 3.464→3.444 (better, small), SU 59.6%→59.3% (worse, -0.3pp) — a
  small, mixed, net-slightly-negative result on only the ~9-10% of games reaching extras.
  Kept anyway: unlike every other tested-and-rejected factor this session (all speculative
  signals), the automatic-runner rule is not a hypothesis — it is a certain, permanent MLB
  rule, and reverting it would mean deliberately mismodeling extra innings to chase a small,
  plausibly-noise-level delta in one 597-game sample (same treatment already given the
  intent_walk split, kept despite a mixed SU result for the same reason — see §11.1).
  Walk-off truncation (ending the bottom of the 9th+ the instant the home team takes the
  lead) was checked at the same time and confirmed to already be correctly implemented —
  see the `walkoff_margin` logic below, a non-issue.
- **`_pitcher_for_inning(inning, bullpen_dict, starter, ...)`**: looks up the real/planned
  pitcher for that inning; if a bullpen dict entry is missing for an inning (e.g. a
  historical reliever had no usable pregame snapshot and was skipped when building the
  dict), falls back to the most recently known pitcher rather than guessing — same
  fallback convention used by `props.py`'s event-attribution logic (§8.3).
- **`_reassign_runners(...)`**: tracks baserunners by **lineup-slot identity**, not raw
  player_id — necessary because stolen-base rates (§4.9) and other runner-specific
  behaviors are resolved by slot.

Confirmed (via `inspect.signature` checks during this session) back to its pre-Phase-2
state: no `_pa_outcome` method, no `outcomes` param variant tied to pitch-level logic
beyond the generic subset-restriction described in §7.1, no `pitch_league_rates`/
`_pitch_bucket_cache` — i.e., genuinely reverted, not just disabled.

---

## 8. Validation framework

### 8.1 `src/models/validate_game_simulator.py` — the oracle backtest

The primary "does the whole chain work end-to-end" test. Simulates real historical games
using each player's real walk-forward pregame snapshot, real lineups, real starting
pitchers, and — critically — **bullpen usage replayed from the game's actual real
historical sequence** (not a predictive bullpen model). Real recorded weather. Only games
with complete 9-batter lineups both sides. Switch-hitters modeled as always batting
opposite the pitcher's hand.

**Constants**: `N_TRIALS_PER_GAME = 200`, `N_GAMES_TO_VALIDATE = 600` (bumped from 200 to
tighten the noise floor specifically for judging foundational `true_talent.py` changes),
`TEST_SEASONS = {2024, 2025}`.

**`player_game_snapshot`**: "rate as of first PA in game" — sorts by `at_bat_number`,
groups by `(game_pk, player_col)`, takes `.nth(0)` — a **fixed** snapshot for the whole
simulated replay (not re-evaluated mid-game).

**`build_profile(rate_row, platoon_row, hand, pull_tercile=None, pregame_xbacon=None, pregame_barrel_rate=None, pregame_bacon_gb=None, pregame_sprint_speed=None, pregame_gb_rate=None, pregame_bat_speed=None)`**:
wires base rates through `apply_contact_quality` (batter-only — no-op for pitcher calls,
which never pass these args), builds `same_mult`/`opp_mult` platoon dicts (all-1.0 if no
platoon row given), carries `pull_tercile` (batter-only; `game_simulator.py` treats a
missing pitcher-side tercile as neutral `"mid_pull"` for the weather lookup).

**Per-game loop**: builds every walk-forward table once up front (rates, league rates,
state/ttop factors, platoon multipliers, hand, park factors, blowout profile, spray/pull
snapshot, weather tables, catcher/umpire factors + real umpire game log, contact-quality
snapshots including bat/sprint speed memoized per-season, defense snapshot, SB rates,
`TransitionTable`). Samples up to `N_GAMES_TO_VALIDATE // len(TEST_SEASONS)` real completed
9-lineup games per season (`random_state=1`). Per game: resolves both lineups, both
starting pitchers (whoever threw the first PA of each defensive half), and the **real
historical bullpen-by-inning sequence** (`groupby("inning")["pitcher"].first()`, pitchers
lacking a usable snapshot simply dropped from the dict, relying on `_pitcher_for_inning`'s
fallback); resolves real park factors, real recorded weather (bucketed the same way the
walk-forward table was built), real starting catchers, real umpire (from a real umpire game
log), real per-game OAA-based defense, real per-runner SB rates; runs `N_TRIALS_PER_GAME`
Monte Carlo trials via `GameSimulator.simulate_game(...)`.

**Metrics** (printed and saved to `game_simulator_validation.parquet`): home/away/total
score MAE, mean actual vs. simulated total, margin MAE (`sim_margin_mean = sim_home_mean -
sim_away_mean` vs. real margin), and **straight-up accuracy**
(`sign(sim_margin_mean) == sign(actual_margin)`, weighted most heavily as the final
full-stack arbiter throughout this project).

**On exact current baseline figures**: a repo-wide search found no file with the literal
values `3.370`/`3.464`/`59.6` recorded — this project keeps no running metrics log/
CLAUDE.md; those exact post-bat-speed numbers come from this session's own terminal run
(confirmed genuine, from earlier in this same session's full-stack A/B test — see §11.6),
not from a persisted file. The nearest in-repo reference points are the pre-bat-speed
baseline documented in `expected_stats.py`'s own comments: **total MAE 3.405, margin MAE
3.462, SU 59.3%**, and (on an older 200-game protocol, predating both the bat-speed change
and the 600-game bump) `validate_predictive_bullpen.py`'s cited oracle-bullpen numbers:
total MAE 3.604, margin MAE 3.445, SU 67.5% vs. a no-bullpen-modeling baseline of 3.728 /
3.661 / 62.5%. **If a future session needs an exact fresh number, rerun
`validate_game_simulator.py` rather than trusting any cached figure** — this file itself is
authoritative, not any doc (including this one).

### 8.2 `src/models/validate_predictive_bullpen.py` — the "honest, deployable" backtest

Same games/protocol as §8.1, but the bullpen plan is built **predictively**: the starter
pitches his own walk-forward-projected expected innings
(`build_expected_starter_innings`), then a specific reliever is **sampled fresh per
trial** (not once per game — which reliever pitches is itself part of the Monte Carlo
uncertainty) from `sample_bullpen_plan`, weighted by recent usage and rest-day discount.
Never looks at the real historical sequence for that specific game. Also uses a
**predictive** starting catcher (`identify_starting_catcher` — whoever caught most
recently for that team, not the real per-game catcher). **Umpire remains real/oracle**
(the comment is explicit: "only the bullpen usage is predictive here, not the umpire").
Same metric block and output format as §8.1, saved to `predictive_bullpen_validation.parquet`.

### 8.3 `src/models/props.py` — the live daily props generator

Full per-PA Monte Carlo simulation of one specific game (default `N_TRIALS = 1000`),
tracking every event (not just final score) to produce batter/pitcher/inning/game props.
Uses the fully **predictive/live-usable** path throughout (real probable lineup/starter,
real park factors, real-if-posted-else-forecast weather, roster-sampled bullpen, dynamic
blowout substitution).

- **`build_pregame_context(pa)`**: builds every walk-forward artifact once (the expensive
  part) — reusable across as many games as needed afterward (cheap per additional game).
- **`generate_game_props(ctx, season, game_pk, home_team, away_team, game_date, home_ids, away_ids, home_pitcher_id, away_pitcher_id, venue_name=None, n_trials=N_TRIALS, rng=None)`**:
  - **`nearest_prior_pitcher_snapshot`fallback**: a genuinely future `game_pk` never has an
    exact-match pregame snapshot (unlike a backtest), so every per-player lookup falls back
    to that player's own most recent snapshot as of `game_date`; raises `ValueError` only
    if no snapshot exists at all (true MLB debut).
  - **Weather**: real posted conditions if available (fixed across all trials); else
    `resolve_weather_distribution` (climatological, via Open-Meteo, the one part needing a
    network call) then `sample_weather_bucket` **fresh every trial**.
  - **Umpire**: `resolve_live_umpire_factor` — live crew lookup once posted, neutral
    fallback otherwise.
  - **Defense/catcher/bullpen**: same predictive fallback machinery as §8.2, sampled fresh
    per trial for the bullpen plan specifically.
  - **Event→pitcher attribution**: blowout → pseudo-id `"{team}_POSITION_PLAYER"`; within
    expected starter innings → starter; else looked up in that trial's sampled bullpen
    plan for that inning, falling back to the most recent prior-inning entry (or literal
    `"BULLPEN_FALLBACK"`) — mirrors `_pitcher_for_inning`'s own fallback.
- **Outputs**:
  - `_batter_props`: `pa_per_game`, `p_1plus_hit`, `p_2plus_hits`, `p_1plus_hr`,
    `p_1plus_bb`, `p_1plus_rbi`, `mean_total_bases`, `mean_hits`, `mean_k`. RBI is
    approximated as runs scored on that specific PA, EXCLUDING runs on
    `RBI_EXCLUDED_OUTCOMES = {double_play, field_error}` (task #154, see below) — still
    not exact MLB RBI rules in every edge case (fielder's-choice RBI eligibility is
    genuinely scorer's-judgment-dependent and left as-is, correctly), but materially
    tighter than before. Two of five props (`p_2plus_hits`, `p_1plus_bb`) pass through
    `_apply_batter_prop_calibration`, a post-hoc linear recalibration `a + b*predicted`
    (clipped `[0.001, 0.999]`), fit because Monte Carlo trial variance overstates
    night-to-night differentiation given the low per-game PA count (~4/batter vs. a
    pitcher's ~20-25 batters faced). `p_1plus_hit`, `p_1plus_hr`, and (as of task #154)
    `p_1plus_rbi` are deliberately left uncorrected — each failed the 4-5/5-split
    stability bar as of the most recent check (see below).

    **Raw `p_1plus_hr` calibration, measured directly for the first time (2026-07-25,
    `validate_prop_calibration.py`, n=2700 batter-games, 150 real 2024-2025 games, 150
    trials each)**: `actual = 0.0275 + 0.7835*predicted` (slope 1.0 = perfect), corr=0.92,
    Brier 0.0989 vs. naive (always-base-rate) 0.1008. **Real, moderate overconfidence** —
    the raw model's high-probability bucket (predicted 0.209) realizes at only 0.172, the
    classic "extremes get pulled toward the mean" pattern — but the effect size is the
    SAME order of magnitude as `p_1plus_hit`'s (slope 0.79, also currently uncorrected,
    same post-shock 5-split failure), and clearly better than `p_1plus_rbi`'s (slope 0.61).
    Brier beats naive by only ~1.9% relative — expected for a rare binary outcome (~11%
    base rate) where most of the Brier budget is already spent on the low-probability
    compression regardless of skill. **Re-ran the 5-split stability check on this exact
    data (2026-07-25)** rather than just eyeballing the slope: fit `a+b*predicted` on a
    random 70% train split, check whether it beats the raw model's Brier on the held-out
    30% — repeated 5x with different splits. Result: **1/5 splits favor the correction**
    (splits 0-3 all made Brier WORSE; only split 4 helped, and barely). This is a THIRD
    independent failure of the same test (original pre-shock fit: 2/5; post-shock refit
    task #138: 2/5; this fresh-data re-check: 1/5, the worst yet) — strong, convergent
    evidence that `p_1plus_hr`'s overconfidence, while real in the aggregate reliability
    curve, is NOT stably correctable with a global linear scaling. The most likely reason:
    at an ~11% base rate with ~2700 samples, any one random split's fitted `(a,b)` is
    substantially overfit to that split's own sampling noise in the rare-event tail, so the
    "fix" doesn't generalize. **Definitively confirmed uncorrected, not just historically
    left that way** — this closes the `p_1plus_hr` calibration question for now; a future
    revisit would need either a fundamentally different correction shape (e.g. isotonic,
    though that failed game-level testing earlier this session — see §11.2 — and hasn't
    been re-tested at the PROP level specifically) or a much larger sample before a linear
    fix could ever be trusted here.

    **Same-day extension (2026-07-25) to the other two uncorrected props**, reusing the
    identical saved sample (zero new compute): `p_1plus_hit` 2/5 splits favor a correction
    (unstable, stays uncorrected, consistent with task #138's own 3/5 post-shock finding on
    a bucketed version of the same test). `p_1plus_rbi` **5/5 splits favor the correction —
    a reversal** of task #138's 2/5 finding on this fresh sample. Fit on the full n=2700
    sample (`a=0.1196, b=0.5977`, raw Brier 0.21267 → corrected 0.21202) and briefly
    **deployed**: `BATTER_PROP_CALIBRATION` gained `"p_1plus_rbi": (0.1196, 0.5977)`.

    **Reverted the same day (task #154 prop-rules audit)**: that 5/5-stable, slope-0.60
    result turned out to be fitting a correction for the RBI-attribution BUG below, not a
    real modeling flaw. See the RBI rule fix immediately following — after correcting the
    RBI ground truth, the calibration slope moved to ~1.10 (near-perfect) and stability
    dropped to 1/5 (unstable, no correction fittable or needed). `p_1plus_rbi` now joins
    `p_1plus_hit`/`p_1plus_hr` as correctly uncorrected — three of five batter props stay
    raw, two (`p_2plus_hits`, `p_1plus_bb`) are corrected, and there's no longer an "active"
    open calibration question on any of them.

    **Task #154 (2026-07-25 prop-rules audit) — two real, verified fixes to the RBI
    approximation, plus confirmation the pitcher-K prop survived the hook-frailty
    attribution change:**

    1. **`RBI_EXCLUDED_OUTCOMES = {double_play, field_error}`** (module-level constant,
       shared by `props.py`'s live computation and `validate_prop_calibration.py`'s "real"
       ground-truth label, so a future recalibration test never compares an improved
       simulated metric against a stale ground truth). `double_play`: Official Rule
       9.04(b)(1), no RBI on a grounded-into force double play — unambiguous, no scorer's
       judgment involved. `field_error`: Rule 9.04(b)(2), no RBI when the run's scoring is a
       direct result of the same misplay, unless the scorer judges the run would have
       scored regardless — this project's PA-level granularity can't make that specific
       judgment call, so this defaults to DENY (conservative: may occasionally under-credit
       a real RBI, never over-credits one). `fielders_choice` and `sac_bunt`/`sac_fly` are
       deliberately NOT excluded — real official scoring generally DOES credit RBI on those.
       **Quantified on real 2025 data before fixing**: of 21,594 season-total RBI-equivalent
       runs, `double_play` accounted for 68 (0.31%) and `field_error` another 164 (1.07%
       combined) — individually small league-wide, but a real, systematic per-player
       overstatement, and (see above) the actual root cause of `p_1plus_rbi`'s apparent
       miscalibration all along.
    2. **`_pitcher_props`'s own docstring previously mis-described `mean_runs_allowed` as
       "earned runs allowed"** — the column itself was always correctly named and computed
       as TOTAL runs (earned + unearned); only the docstring wording was wrong. Fixed the
       wording, not the computation. True earned-run tracking would need reasoning about an
       error's downstream effect across the REST of an inning's multiple PAs (the official
       "what would have happened with average defense" counterfactual), not just the current
       PA — out of scope for this project's granularity; `mean_runs_allowed` should be read
       as runs allowed, full stop, not an ERA-consistent figure.
    3. **Post-hook pitcher-K prop re-check** (the other half of task #154's scope): hook-
       frailty has been live in `props.py` since sec 11.25, which changes which pitcher
       accumulates simulated innings/Ks per trial (a correctly-fixed attribution, not a
       bug — see sec 11.25's own `hook_result` fix). This session's `p_6plus_k` reads (both
       the original HR-prop investigation and this task #154 re-run) were ALREADY generated
       under that post-hook attribution: `actual = 0.0018 + 0.8666*predicted`, corr=0.9997,
       Brier 0.0534 vs. naive 0.0876 — excellent, consistent with every prior `p_6plus_k`
       reading this project has ever taken (a pitcher's ~20-25 batters faced is a much
       larger single-game information budget than a batter's ~4 PA). **No regression from
       the hook-frailty attribution change** — this closes task #154's second half.

    Full before/after numbers and the exact rule citations: this section and the metrics
    ledger. No fielders-choice change, no game-level simulation change — this was a
    props-layer-only fix, total runs scored (and every other factor) are byte-identical.
  - `_pitcher_props`: `mean_k`, `mean_bb`, `mean_hits_allowed`, `mean_runs_allowed` (TOTAL
    runs allowed, earned + unearned — see task #154 docstring fix above, not ERA-consistent),
    `mean_batters_faced`, `p_6plus_k`.
  - `_inning_props`: P(1+ run) per team per inning.
  - `_game_props`: `home_win_prob`, `away_win_prob`, `mean_home_score`, `mean_away_score`,
    `mean_total`, `p_over_8.5`, `p_under_8.5`, `home_covers_minus_1_5`, `away_covers_plus_1_5`.

### 8.4 `src/models/lineup_projection.py` — projecting a not-yet-posted lineup

For a future game where MLB hasn't posted the real lineup yet (confirmed: even fetched the
night before, lineups post only hours pre-first-pitch).
- `project_lineup`: baseline = each team's most recent actual complete 9-batter lineup
  before the target date (validated: consecutive real games share 76% roster overlap,
  47.9% exact same player/same slot).
- `project_lineup_platoon_aware(schedule, lineups, pitcher_hand, team, target_date, opposing_pitcher_id, window_games=40, min_starts=6)`:
  starts from the same baseline; for each slot, computes the current player's own share of
  recent starts at that position against **today's** opposing hand (restricted to players
  with ≥`min_starts`); if that share is ≥0.35 (or no history), trusts the baseline
  unswapped; else looks for an alternative with ≥`min_starts` and ≥0.65 share against that
  hand (excluding players already placed elsewhere this lineup), picking the one with most
  starts against that hand. Design history: an earlier one-sided "someone else started
  more" version triggered 3+ swaps per lineup (real platoons are usually 0-2 players);
  the current two-sided weak-share/strong-share test fixed the overcorrection.

### 8.5 `src/models/validate_prop_calibration.py` — are individual prop probabilities calibrated?

The one validation this project didn't do until explicitly checking: does "34% chance of
1+ hits tonight" actually happen ~34% of the time? Uses `props.py`'s real deployed
`generate_game_props` (predictive bullpen, not the oracle sequence) on real completed
games (`N_GAMES=150`, `N_TRIALS=150` — reduced from live's 1000 since aggregating across
many games matters more than single-game trial precision here), real lineups/starters
(known, since backtesting played games). `calibration_report` buckets predictions into up
to 8 quantile bins (`mean_predicted` vs. `actual_rate` per bucket); `report_calibration`
fits `actual = a + b*predicted` (slope 1.0 = perfect), reports bucketed correlation and
Brier score vs. a naive always-base-rate baseline, for all 5 batter props + `p_6plus_k`,
plus continuous-prop MAE (`mean_hits`/`mean_k`) vs. a naive always-mean baseline.

---

## 9. Live daily pipeline

### 9.1 `src/pipeline/daily_update.py`
Refreshes raw data. `HISTORY_START_SEASON = 2023`. `refresh_all_data()`: for each season
from 2023 through the current one, in order — `fetch_schedule_season`,
`fetch_statcast_season`, `verify_and_backfill_statcast` (each wrapped in its own try/except,
errors collected not fatal). Complete past seasons are fetched once and never re-touched;
only the current season refreshes incrementally each run (full `force=True` re-fetch of
10+ seasons of pitch-level Statcast data would be too slow for a near-daily cadence). Note:
this module's own docstring is **stale** — it describes the prediction engine as "doesn't
exist yet," which predates `props.py`/`generate_daily_props.py`.

### 9.2 `src/pipeline/generate_daily_props.py`
The real live entry point. **A real, documented, fixed bug**: this script originally read
the cached PA table directly without ever calling `refresh_all_data()` first — confirmed
the cache was stale by 2 real days when caught. Fixed sequence, now exact order:
1. `refresh_all_data()` — raw Statcast/schedule refresh (yesterday's real completed games).
2. `build_and_save_pa_table(seasons_covered)` — rebuild the **derived** PA table (does not
   rebuild itself just because raw data changed underneath it).
3. `fetch_schedule_day(target_date)` — pull the target (future) date's real
   schedule/probable pitchers (season fetcher can't reach future dates).
4. `games_for_date(target_date)` — filter to real games that day.
5. `build_pregame_context(pa)` once (the expensive part).
6. Per game: skip if either probable pitcher isn't announced; resolve lineups (real posted
   if available, else `project_lineup_platoon_aware` using the opponent's probable
   starter's hand); skip if no projectable history; `generate_game_props(...)` (catching
   `ValueError` for a true debut, skipping that game).
7. Save `daily_props_{target_date}.parquet`; print a per-game summary with flags for
   projected-lineup / forecast-weather so a human reviewing the output knows which games
   used real vs. fallback inputs.

---

## 10. Recurring engineering patterns worth knowing

- **Clip both sides before any odds-ratio.** Every module that computes `odds(p) = p/(1-p)`
  clips `p` away from exactly 0/1 first. This is not defensive boilerplate — real
  documented blowups occurred without it: platoon splits (146,611x), weather (73x), park
  factors (`inf`), HR-share (13.8x), HR-share-with-bat-speed (5.5x, the reason that variant
  was never shipped). If extending any multiplier in this codebase, this clip is mandatory,
  not optional.
- **Bayesian shrinkage everywhere**: `(observed + PRIOR_WEIGHT * reference_rate) / (observed_n + PRIOR_WEIGHT)`,
  reused from `true_talent.py`'s Marcel priors down to `platoon_splits.py`'s per-outcome
  shrinkage to `expected_stats.py`'s `XBACON_PRIOR_BIP`/`BARREL_PRIOR_BIP`.
  `PRIOR_WEIGHT`/stabilization-point constants are always fit or sourced from published
  research (Carleton), never guessed.
  - **Leakage-free, incremental-R² validation as the standard for every new signal**: build
  from strictly-prior data, predict the real target-season/game outcome, and require the
  new signal to add value *beyond the current production signal* — not just beyond raw
  history. A weaker single-predictor correlation check was caught and redone properly at
  least once this session before committing to a build.
- **Full-stack isolated A/B as the final arbiter, always.** Same seed, same ~597-600 real
  games, a temporary `_ab_*_off.py` copy with only the new signal disabled, compare against
  the real file. SU accuracy weighted most heavily. A real, statistically valid
  component-level signal does **not** automatically survive this bar — it has failed more
  often than it's passed this session (see §11).
- **Revert discipline**: when a factor fails the full-stack bar, all wiring is cleanly
  removed from every consumer file, but the underlying investigation code (functions/
  constants) is deliberately **left in place**, clearly marked, as a documented-but-unused
  artifact — never silently deleted. This document's §5/§6/§11 preserve that same practice.

---

## 11. Complete ledger: kept vs. reverted vs. investigated-not-built

This section exists specifically because the user asked that nothing be left out — both
what worked and what didn't.

### 11.1 Currently LIVE (wired into production, i.e. into `props.py` +
`validate_game_simulator.py` + `validate_predictive_bullpen.py`, or unconditionally part of
`game_simulator.py`'s core mechanism)
| Factor | Module | Status |
|---|---|---|
| Marcel true-talent rates | `true_talent.py` | core engine, unchanged |
| Odds-ratio matchup combine | `matchup.py` | core engine, unchanged |
| Base-out bootstrap transitions | `base_out_transitions.py` | core engine, unchanged |
| Platoon splits (batter + pitcher) | `platoon_splits.py` | unchanged |
| Park factors | `park_factors.py` | unchanged |
| Weather (+ pull-tercile-conditioned) | `weather.py`, `weather_forecast.py`, `spray.py`, `park_orientation.py` | unchanged |
| Times-through-order penalty | `ttop.py` | unchanged |
| Catcher framing | `catcher_framing.py` | unchanged |
| Umpire tendency | `umpire_factor.py` | unchanged |
| Defense (OAA) | `defense_factor.py` | unchanged |
| Baserunning / stolen bases | `baserunning.py` | unchanged |
| Blowout / position-player-pitching | `blowout.py` | unchanged |
| Predictive bullpen | `bullpen.py` | unchanged |
| xBACON contact quality | `expected_stats.py` FIRST LAYER | kept |
| Barrel rate → HR-share | `expected_stats.py` SECOND LAYER | kept |
| Sprint speed → groundball singles | `expected_stats.py` THIRD LAYER | kept |
| Bat speed extension to contact quality | `expected_stats.py` FIFTH LAYER | kept, but re-tested at n=7237 (§11.9): SU delta +0.12pp, CI includes zero — still unresolved/plausible-but-unconfirmed, original n=597 "59.3%→59.6%" figure superseded |
| Zombie-runner (auto-runner) extra-innings rule | `game_simulator.py` `simulate_game`/`simulate_half_inning` | kept (small mixed result, correctness fix not a hypothesis) — total MAE 3.370→3.404, margin MAE 3.464→3.444, SU 59.6%→59.3% |
| Pulled-air-ball rate extension to HR-share | `expected_stats.py` SEVENTH LAYER | kept, but re-tested at n=7237 (§11.9): SU delta +0.21pp, CI includes zero — still unresolved/plausible-but-unconfirmed, original n=597 "59.3%→60.5%" figure superseded |
| Home-field advantage (league-wide pooled) | `park_factors.build_hfa_factors`, `game_simulator.py` | kept (correctness fix, not a hypothesis) — full-stack effect not statistically distinguishable at n=597 (§11.8); component-verified real (~+0.2-0.3 runs/game in isolation) |
| Park-neutralized true-talent rates | `true_talent.py` `_park_neutral_events`/`build_pregame_rates` | kept (correctness fix, not a hypothesis) — same n=597 caveat as HFA above |
| **GB/FB → pitcher HR-allowed-share (EIGHTH LAYER)** | `expected_stats.py` `pitcher_hr_allowed_multiplier` | **kept** — full-stack (n=7237, the new scaled protocol): SU 56.3%→**57.9%** (delta +1.53pp, 95% CI excludes zero), Brier 0.2416→0.2406 (not significant) — first confirmed pitcher-side win this session |

### 11.2 BUILT THEN REVERTED (full wiring removed from all consumer files; underlying code
kept as documented-but-unused artifacts)
| Idea | Full-stack result | Why it likely failed |
|---|---|---|
| Whiff-rate-on-swings (`whiff_rate_multiplier`) | Total MAE 3.405→3.379 (better); margin MAE 3.462→3.482 (worse); **SU 59.3%→58.0% (-1.3pp)** | Real component-level signal, net-negative full-stack — strikeout-specific signals repeatedly underperform |
| Full pitch-by-pitch DP mechanism swap (Phase 2) | **SU 59.3%→55.6% (-3.7pp)** | Composed distribution was variance-compressed (std 0.035 vs. true_talent's 0.051 vs. real 0.063) — correct on average, bad at separating strong/weak matchups |
| Pitcher fastball-stuff (`pitcher_stuff_k_multiplier`) | Net negative on SU (component R² gain was real, 0.396→0.419) | Pitcher-side/strikeout-specific signal, same pattern as whiff-rate |
| Speed-conditioned base-out transitions (`TransitionTable`'s `runner_speed_bucket`) | Original (n=597): SU 60.5%→57.3% (-3.2pp). **Retested at n=995/500 trials with a bootstrap CI (§11.7): SU delta shrunk to -1.8pp (CI barely touches zero), but Brier score shows a REAL regression (delta +0.0024, 95% CI (+0.0007,+0.0041), excludes zero)** | Real, clean, large component-level effect (16.7pp/11.6pp real spreads) but STILL failed on the more rigorous re-check — conditioning resampling on runner identity adds within-game outcome heterogeneity that genuinely hurts calibration, confirmed by Brier score even though the original SU point estimate was partly noise-inflated |
| `pitch_walk_multiplier` (finally deployed, see §11.3's prior "never deployed" status) | Total MAE 3.409→3.377 (better); margin MAE 3.445→3.474 (worse); **SU 60.5%→59.1% (-1.4pp)** — not retested under the tightened protocol (§11.7), plausible this delta is partly noise given the pattern found for jet-lag | Same regression shape as whiff-rate despite being a WALK signal (not strikeout) with real incremental R² (batter 0.315→0.361) |
| Jet-lag/circadian fatigue multiplier (`jetlag.py`, built from scratch) | Original (n=597): SU 60.5%→59.3% (-1.2pp). **Retested at n=995/500 trials (§11.7): SU delta REVERSED to +0.1pp (CI (-1.21pp,+1.41pp)), Brier delta exactly 0.0000 (CI (-0.0011,+0.0012))** — genuinely indistinguishable from zero, not negative | A genuinely NEW signal category (schedule-derived, team-level), independently statistically significant on our own data in isolation — but the original "-1.2pp full-stack regression" was pure noise, not a real effect. Still not deployed (no proven benefit either), but for an honestly different reason than first reported |

### 11.3 DESIGNED/FIT BUT NEVER DEPLOYED
Empty as of 2026-07-22 — `pitch_walk_multiplier` was the only entry here and has
since been deployed for real and reverted after a full-stack failure (see §11.2).

### 11.4 INVESTIGATED, NOT BUILT (failed its own component-level or safety check before
reaching a full-stack A/B at all)
| Idea | Why not built |
|---|---|
| Arsenal-tercile matchup adjustment | Negative in leakage-free component testing across strikeout/HR/single |
| HR-share-with-bat-speed extension | Real incremental R² (0.379→0.405) but 5.5x real-data blowup with no safe clip fix available |
| Pitcher-side contact suppression (xBACON-allowed / barrel-rate-allowed) | No real incremental signal found (matches DIPS theory — pitchers show almost no control over contact quality allowed, R² ~0.005-0.07, vs. batters' own R² 0.14-0.18) |
| Bat-tracking for double/triple share | No real incremental signal (R² ~0.01-0.03 for both real history and xSLG) |
| Double/triple split via xSLG (SECOND LAYER sub-case) | Same as above |
| EV90 as a `contact_quality_multiplier` stabilizer | R² gain +0.0009/+0.0015 beyond existing+xBACON+bat_speed — an order of magnitude smaller than bat speed's own gain, essentially zero incremental value |
| Squared-up rate (Statcast's own formula, computed directly from `bat_speed`/`effective_speed`/`launch_speed`) | R² gain +0.0001/+0.0000 (one coefficient even negative) — no incremental value at all, matches the research doc's own warning that it's redundant with barrel rate/bat speed |
| XBT% (extra-bases-taken) baserunning conditioning `TransitionTable` | Real, split-half-validated component effect (tercile spread 31.5%→40.1%, p<0.0001) — but structurally the SAME intervention type as speed-conditioned transitions (§11.2), which already failed full-stack decisively with an even larger effect. Not full-stack tested on strong analogical grounds rather than re-confirming an established failure mode |
| Park factor × pull-tercile interaction (2026-07-25, prompted by a user question about spray-chart-vs-stadium modeling) | `split ~ park_factor_centered * is_high_pull` (OLS, n=857 batter-seasons ≥15 PA both home/road, walk-forward HR park factor × PRIOR-season real pull rate): incremental R² = **0.00000**, interaction p=0.974 — a clean, decisive null, not a sample-size-starved one (additive-only R² was already just 0.011). A batter's own pull tendency does NOT measurably amplify or dampen their home park's HR factor beyond what each contributes separately. Today's flat per-park/per-outcome scalar (no batter-direction conditioning) is not missing a real effect here |

### 11.5 The cross-cutting finding from this session, refined

**First-pass finding**: batter-side hit-quality signals (contact quality/xBACON,
barrel-rate/HR-share, sprint-speed/groundball-singles, bat speed, pulled-air rate — now
5-for-5) consistently show real, extractable, full-stack-validated value. Pitcher-side
signals and strikeout-specific signals of any kind (arsenal-tercile, whiff-rate, the
wholesale pitch-by-pitch DP swap, pitcher-stuff, `pitch_walk_multiplier` — now 5-for-5
failures) consistently either show no real signal or fail the full-stack bar despite
genuine component-level validation, matching independently-confirmed DIPS theory.

**Refined finding, after speed-conditioned transitions and jet-lag also failed**: the
pattern isn't really about *which player/team* a signal targets — it's about *what kind
of change* the signal makes to the simulator. Every KEPT factor is a refinement to an
EXISTING per-category odds-ratio multiplier the simulator already applies (contact
quality, barrel rate, sprint speed, bat speed, and pulled-air rate all extend
`hr_share_multiplier`/`contact_quality_multiplier` — the same mechanism, better-fit
coefficients). Every FAILED factor that introduces a genuinely NEW mechanism or a NEW
source of within-game heterogeneity into the joint outcome distribution has failed —
**6 for 6**: the pitch-by-pitch DP composition (a new mechanism replacing K/BB/HBP
entirely), whiff-rate and pitcher-stuff (strikeout-specific, DIPS-limited), the
walk-blend (still a new composed signal even though walk-specific), speed-conditioned
base-out transitions (conditions resampling on runner identity — new heterogeneity), and
jet-lag (a new team-day-level status dimension). Any future work chasing further
accuracy gains should prioritize refinements to EXISTING per-category multipliers on
batter-side hit-quality signals, and treat any proposal that adds a genuinely new
mechanism or heterogeneity axis — no matter how clean the component-level validation —
as a real full-stack risk, not a safe bet.

**Update (2026-07-22, §11.8 claim 4): the pitcher-side pattern has its first exception, and
it fits the "refinement, not new mechanism" rule exactly.** The GB/FB → HR-allowed-share
layer (`pitcher_hr_allowed_multiplier`) is a genuine, statistically confirmed full-stack win
(SU +1.53pp, CI excludes zero, at n=7237) — the first pitcher-side signal to clear the bar in
6 attempts. This does NOT contradict the refined finding above: it's structurally a
refinement to the EXISTING `home_run` odds-ratio category (same mechanism as
`hr_share_multiplier`, just on the pitcher's side), not a new mechanism or heterogeneity
axis — and DIPS 2.0 specifically predicts pitchers control batted-ball TYPE even though they
don't control contact-quality OUTCOME, distinguishing it from the 5 failed
strikeout/stuff/contact-quality attempts. Refined rule, now cleanly 7-for-7 either way:
"pitcher-side signal that's actually a contact-quality/strikeout mechanism" fails every time;
"refinement to an existing per-category multiplier, batter OR pitcher side" succeeds every
time DIPS theory doesn't independently rule it out.

### 11.6 Path to "SU ≥ 60%" — **achieved**

The user's explicit goal this session was SU accuracy ≥60%. After the first round of
attempts stalled at 59.6% (bat speed the only win; whiff-rate, the pitch-DP swap, and
pitcher-stuff all reverted), an external research pass (a separately-run "deep research"
report, fed this project's own MODEL_DOCUMENTATION.md as context) proposed new candidates.
Cross-referencing against the established batter-side-signals-work /
pitcher-side-signals-fail pattern (§11.5), two were selected and tested:

1. **Zombie-runner extra-innings fix** (§7.2) — a genuine correctness fix (not a
   speculative signal), kept despite a small net-negative full-stack delta (SU
   59.6%→59.3%) because the underlying rule is certain, not hypothetical.
2. **Pulled-air-ball rate** (§5, SEVENTH LAYER) — another batter-side hit-quality signal
   (direction, not power), kept after a clean full-stack win: **SU 59.3%→60.5%**.

**Final combined result: total MAE 3.409, margin MAE 3.445, SU accuracy 60.5%** — crosses
the user's explicit goal. This extends the batter-side-signals-work pattern to a 6th
confirmed case (contact quality, barrel rate, sprint speed, bat speed, and now pulled-air
rate all landed cleanly; every pitcher-side/strikeout-specific signal — 5 separate
attempts — failed or showed no signal). Untried leads from the same research pass, not
yet built: speed-conditioned base-out transitions and GIDP-on-GB-rate/speed conditioning
(mechanical, low-risk MAE plays), the dormant already-fit `pitch_walk_multiplier` (§11.3,
never deployed), and a jet-lag/circadian team-level multiplier (a genuinely novel
schedule-derived category, zero new data cost, unlike anything tried so far).

### 11.7 Methodology overhaul: the A/B protocol itself was under-powered

An external LLM's critique of this session's evaluation process (2026-07-22) identified
two real, checkable problems, since fixed:

1. **SU noise floor.** At n≈597 games, straight-up accuracy has a binomial SE of ~2.0pp
   (95% CI ≈ ±4pp on one arm). Several reverts this session (whiff-rate -1.3pp, walk-blend
   -1.2pp, `pitch_walk_multiplier` -1.4pp, jet-lag -1.2pp) were decided on deltas smaller
   than this noise floor.
2. **The "same seed" pairing was illusory.** `rng = np.random.default_rng(42)` in
   `validate_game_simulator.py` is created once, before the entire game loop, and consumed
   sequentially across every game and trial — confirmed directly in the code. The instant
   two arms' outcomes differ even once, every subsequent random draw diverges between them.
   The two arms are NOT a paired comparison of identical draws over the same games; they're
   two independent Monte Carlo runs that merely start aligned. The true uncertainty on any
   delta is closer to the **unpaired** combination of both arms' own noise (~2.8pp SE,
   ~±5.5pp 95% CI) than the "same seed" framing implied.

**Fixes shipped:**
- **Brier score + log-loss on `P(home win)`** added to `validate_game_simulator.py`'s
  summary output — `sim_home_win_prob` was already computed per game, just never used as a
  decision metric. Continuous, uses every game's full simulated win-probability, not just
  whether the sign of the mean margin happened to match — a much tighter CI than SU at the
  same n.
- **`src/models/ab_significance.py`** — a reusable paired-bootstrap-CI tool. Loads two saved
  `*_validation.parquet` outputs (matched by `game_pk`), bootstraps (10,000 resamples) the
  SU delta and the Brier delta, and reports a plain verdict: `"REAL IMPROVEMENT/REGRESSION
  -- CI excludes zero"` vs `"NOISE -- CI includes zero"`. **This is now the standard tool
  for any future keep/revert call — a raw point-estimate delta should never again be
  treated as dispositive on its own.**

**Retest of the two most-recent reverts, at a tightened protocol** (1000 games / 500
trials, ~4.2x the original compute, one shared high-N baseline; the pitch-by-pitch DP swap
was deliberately NOT retested — its diagnosis was mechanistic and its effect size, -3.7pp,
was always far outside any plausible noise band):
- **Speed-conditioned transitions**: SU delta shrunk to -1.8pp (CI barely touches zero) but
  **Brier score showed a real regression** (CI excludes zero) — revert confirmed, on firmer
  evidence than the original noisy SU point estimate.
- **Jet-lag**: SU delta flipped to +0.1pp and Brier delta was exactly 0.0000, both CIs
  comfortably including zero — **genuinely null, not negative**. The original "-1.2pp"
  verdict was pure noise. Still not deployed (no proven benefit either), but for an
  honestly different reason than first reported.

**Practical upshot**: use `ab_significance.py`'s bootstrap CI on Brier score as the primary
decision tool for all future full-stack A/Bs, not a raw SU point estimate at n~600/
200-trials.

**Follow-up: the three currently-KEPT session wins were then re-validated the same way — none replicate as statistically significant.**
Bat speed, pulled-air rate, and the auto-runner fix were all originally decided using the
same noisy n=597/200-trial SU point-estimate methodology just shown to false-positive on
jet-lag. Re-tested identically (n=995/500 trials, one factor isolated at a time against a
shared high-N current-production baseline):
- **Bat speed**: SU delta +1.11pp, 95% CI (-0.80pp, +3.12pp) — includes zero. Brier delta
  +0.0002, CI (-0.0017, +0.0022) — includes zero.
- **Pulled-air rate**: SU delta **-1.01pp** (sign flipped from the original +1.2pp — the
  single largest "win" of the session), 95% CI (-3.02pp, +1.01pp) — includes zero. Brier
  delta +0.0004, CI (-0.0016, +0.0024) — includes zero.
- **Auto-runner**: SU delta +1.11pp, CI (-0.70pp, +3.02pp) — includes zero. Brier delta
  +0.0001, CI (-0.0018, +0.0019) — includes zero.

**Neither bat speed nor pulled-air rate was reverted**, despite this — their component-level
evidence remains genuinely strong on its own terms (real leakage-free incremental R² gains;
pulled-air rate was independently sanity-checked against real external HR-rate-by-batted-
ball-direction figures and matched almost exactly). Unlike jet-lag, whose "harm" was refuted
by a negative control behaving exactly as theory predicted, neither factor here has been
shown HARMFUL at the tightened protocol — both are simply unresolved at this sample size,
not disproven. The auto-runner fix's status is unaffected either way, since it was kept on
"this is a certain MLB rule" grounds, not a proven statistical win. **Practical implication:
full-stack deltas at the scale this project operates on (~1pp SU, ~0.001-0.002 Brier) are
not reliably resolvable even at n=995/500 trials** — meaningfully more games (likely
multiple thousands) would be needed to actually confirm or deny these two effects. Until
then, treat bat speed and pulled-air rate as **plausible-but-unconfirmed**, not proven wins,
and do not cite the original 59.6%/60.5% SU figures as precisely-measured facts.

### 11.8 Second external critique (2026-07-22): structural gaps, not candidate signals

A second external LLM read this entire document and identified 6 issues framed as
**architectural/correctness gaps rather than candidate signals** — distinct in kind from
everything in §11.1-11.7, which are all refinements to an already-coherent architecture.
Two cheap diagnostics were run first (both confirmed real, moderate biases), then two
correctness fixes were built and wired everywhere; a third mechanism was built but is
deliberately gated OFF pending a properly-powered A/B.

**Diagnostics run directly against the existing n=597 backtest
(`data/processed/game_simulator_validation.parquet`), zero build cost:**
- **Home win-rate gap**: sim mean home-win probability 50.6% vs. real home win rate 52.3%
  (a ~1.7pp gap) — confirms claim 1 below.
- **Total-runs dispersion (PIT/z-score check)**: `z = (actual_total - sim_total_mean) /
  combined_std` should be ~N(0,1) if well-calibrated. Measured std(z) = 1.087 (>1 ⇒
  under-dispersed) and the 99%-tail frequency (|z|>2.58) was 2.5% vs. an expected 1% —
  roughly 2.5x too many extreme-total games for the simulator's own uncertainty band. Not
  yet acted on (see §12) — a real, moderate finding, but no fix has been built for it this
  round.

**Claim 1 — no home-field advantage anywhere in the model (CONFIRMED, FIXED).**
Every one of §4's context factors (platoon, park, weather, TTOP, catcher+umpire, defense,
state) is symmetric between the two teams — `combine_matchup_distribution` had zero
home/away-specific logic before this fix. Built **`park_factors.build_hfa_factors`**: a
walk-forward-safe (3-year rolling, prior-seasons-only), **league-wide pooled** (not
per-team) home/road ratio per outcome category. This is a genuinely different quantity
than `build_outcome_park_factors`'s own per-team factor, not a duplicate: that function's
final step re-normalizes each (season, outcome) group of 30 team-level ratios to a
population mean of 1.0 (a real, necessary fix for a right-skewed-estimator bias — see its
own docstring) — but a UNIVERSAL home edge present equally in every team's own home/road
split gets silently cancelled by that same normalization, indistinguishable from the bias
it exists to fix. Pooling across all 30 teams into one ratio sidesteps this entirely (a
single ratio has no Jensen's-inequality skew to correct for). Wired into
`combine_matchup_distribution` (new `hfa_factors` param) and `simulate_half_inning`/
`simulate_game` — applied **only to the home team's own at-bats**, never the away team's
(the one asymmetric factor in the whole system, by design). Shrunk toward 1.0 with a large
pooled prior (`HFA_PRIOR_PA = 20000`) and clipped to `[0.7, 1.4]` as defensive insurance.
Wired into all 3 consumer files (`validate_game_simulator.py`, `validate_predictive_bullpen.py`,
`props.py`). Full-stack validation: **pending** (see status note below).

**Claim 2 — true-talent rates are park-contaminated, then park factors are applied on top
(CONFIRMED, FIXED).** Verified via direct code read: `true_talent.py`'s `_season_rates`
counted raw per-season PA/events with no park adjustment anywhere in the chain feeding
`raw_rate` → age adjustment → `preseason_rate` → in-season blend, and `build_pa_table.py`
carries no park-neutralization at all. Meanwhile `game_simulator.py` applies the per-game
`park_factors` multiplier on top of that already-contaminated rate — a real double-count,
concentrated in extreme parks (Coors, pre-humidor Chase) but systematic and directional.
Fixed by adding an optional `park_factors_df` parameter (threaded through
`_season_rates`/`build_preseason_priors`/`build_pregame_rates`): each PA's contribution to
the event count is divided by that game's own park factor for that outcome (looked up by
the game's `home_team` + season against `build_outcome_park_factors`'s own walk-forward-safe
output) before it feeds Marcel estimation — the PA-count denominator is left unweighted, so
`rate_hat = sum(is_outcome_i / pf_i) / n` is an unbiased estimator of the neutral rate under
the standard linear/multiplicative park-factor approximation used throughout this project.
Verified directionally on real data: 2025 COL (Coors) batters' park-neutral home_run rate is
LOWER than raw (0.0301→0.0295, since raw carries Coors inflation); 2025 SEA (pitcher-friendly)
batters' park-neutral rate is HIGHER than raw (0.0384→0.0396) — both exactly the expected
direction, correlation with the raw (unneutralized) rate 0.997 (a small, real, correctly-signed
correction, not noise). Defaults to `None` (old, unadjusted behavior) for backward compat;
wired to always-on (via `build_outcome_park_factors(pa)`) in all 3 consumer files plus
`bullpen.py`'s team-pooled bullpen rate (same underlying mechanism, same bug). Full-stack
validation: **pending** (see status note below).

**Claim 3 — no-MLB-history players regress to league average, not a debut-level prior
(CONFIRMED, BUILT, gated OFF pending A/B).** `true_talent.py`'s true-cold-start fallback used
the unqualified `league_rate` for the ~10%+ of PAs each season belonging to a player with zero
prior-season history — and the Tango age curve never reaches a genuine rookie at all (it
operates only on `raw_rate`, which requires ≥1 prior season to compute). Built
**`build_debut_rate`**: the walk-forward-safe empirical rate real MLB debut cohorts (players
whose first-ever PA in our own data is in season *s*) posted in every season `s <
target_season`, Bayesian-shrunk toward `league_rate`. Confirmed directionally sane on real
data (2025/2026 debut cohorts vs. league): home_run -0.6 to -0.7pp, walk -0.6 to -0.9pp,
strikeout **+1.7 to +2.9pp**, single roughly flat — the expected below-average/replacement-level
shape in every category. `EARLIEST_SEASON = 2023` (our data's own first season) can't
identify real debuts at all (every player looks like a debut that year) — an acknowledged,
unavoidable data-window limitation, unchanged from the prior single-season cold-start case.
**Unlike claims 1-2, this is a genuinely new mechanism, not a fix to an existing
double-count** — per the user's own framing, it's one of "the two properly powered A/Bs on
the ~6,500-game protocol," not an assumed correctness fix. Gated behind
`build_pregame_rates(..., use_debut_prior=False)` (default off, old behavior preserved) so it
can be A/B tested in isolation before any live caller opts in.

**RESOLVED (2026-07-22): full-stack A/B on the new n=7237 protocol is a clean, decisive NULL
— not deployed.** Paired bootstrap (`use_debut_prior=True` vs. the n=7237 baseline, same
games): **SU delta -0.03pp, 95% CI (-1.09pp, +1.04pp) — NOISE. Brier delta +0.0001, 95% CI
(-0.0017, +0.0019) — NOISE.** Unlike the earlier "unresolved" verdicts on bat speed/pulled-air
rate (wide CIs at n=597-995 that simply couldn't tell), this CI is TIGHT (±1pp on SU) and
sits almost exactly on zero — a genuinely confident null, not an underpowered one. Component-
level evidence (real rookies performing below league average) remains true and the mechanism
is left in place, gated off by default (`use_debut_prior=False`) — matches this project's
"designed, real signal, no full-stack benefit, not deployed" pattern (§11.3-style), not a code
deletion. Task #119 closed.

**Status note on claims 1-2's full-stack validation — RESOLVED, kept on correctness-fix
grounds, full-stack effect NOT statistically distinguishable at this n.** Both fixes are
built, wired into `true_talent.py`/`park_factors.py`/`game_simulator.py`, and threaded
through all 3 real consumer files (`validate_game_simulator.py`,
`validate_predictive_bullpen.py`, `props.py`) plus `bullpen.py`. Ran a proper paired
bootstrap A/B (both fixes ON vs. both OFF, same 597 games, `ab_significance.py`):
**SU delta -0.17pp, 95% CI (-3.18pp, +2.85pp) — NOISE. Brier delta +0.0018, 95% CI
(-0.0020, +0.0057) — NOISE.** Neither metric distinguishes the fixed state from the
pre-fix state at n=597 — the same "unresolvable at this sample size" pattern §11.7
already found for bat speed/pulled-air rate/auto-runner.

This does NOT mean the fixes don't work — a naive single-run before/after comparison first
suggested almost no effect (sim home-win prob moved only 50.6%→50.7%, nowhere near closing
the diagnosed 1.7pp gap), which was concerning enough to investigate further before trusting
it. Isolated component-level tests (controlled league-average lineups, `GameSimulator`
called directly, both with and without `hfa_factors`) confirmed the HFA mechanism alone
produces a real, correctly-directioned, ~2.5%-per-PA / **+0.2 to +0.3 runs/9-innings**
home-offense boost — squarely in the range real MLB home-field advantage should produce, and
NOT an artifact of the park-factor stacking (reproduced with park factors neutral too). The
naive full-run comparison was almost certainly misleading for the same reason this session
already learned to distrust raw before/after point estimates: `validate_game_simulator.py`'s
`rng` is created once and consumed sequentially, so two runs that differ in even one PA's
outcome are unpaired Monte Carlo draws from the first divergence onward, not a true paired
comparison — a real-but-modest effect can easily vanish inside that noise at n=597.

**Decision**: kept both fixes — they correct a real, confirmed architectural bug
(park double-counting) and a real, confirmed structural omission (no HFA anywhere), on the
same "correctness fix, not a hypothesis" grounds as the zombie-runner rule (§11.2), not
because the full-stack A/B proved a net win. Whether they help SU/Brier in aggregate remains
genuinely unresolved at n=597 — resolving it properly would need the larger ~6,500-7,000-game
protocol (claim 6), same lesson as bat speed/pulled-air rate.

**Claim 4 — GB/FB profile → pitcher HR-allowed — CONFIRMED, BUILT, KEPT: the first real
pitcher-side full-stack win of the entire session.** The user's argument: this project's
5-for-5 pitcher-side signal failures (§11.5) were all strikeout/stuff/contact-quality
signals, which DIPS theory agrees are dead — but DIPS 2.0 explicitly carves out batted-ball
TYPE (pitchers meaningfully control GB%/FB%, stabilizing fast, ~70 BF) as distinct from
contact quality/outcome on those batted balls (which pitchers don't control). Pitcher
HR-allowed is Carleton's own slowest-stabilizing, most-shrunk pitcher category (K=1320 BF) —
mostly league-mean-plus-noise in this project's own Marcel estimate — while a DIPS-
independent, fast-stabilizing predictor of it (GB/FB ratio) already sits in the Statcast
table.

Component-level (leakage-free, 3 walk-forward season pairs, `scratchpad/fit_pitcher_hr_share.py`):
adding a pitcher's own GB%/FB% (launch_angle<10 / >=25 among their own balls in play) to
their existing HR-share-among-BIP roughly DOUBLES the R² predicting next-season HR-allowed
share in every pair tested (0.051→0.123, 0.039→0.112, 0.045→0.096) — the largest single
pitcher-side component gain found this entire session, bigger than any batter-side layer's
own original gain. Built **`expected_stats.pitcher_hr_allowed_multiplier`**: structurally
identical to `hr_share_multiplier` (batter-side), just on the pitcher's `home_run` category,
with `existing_share` framed as HR-among-ALL-BALLS-IN-PLAY (not just hits, since HR-allowed
is fundamentally about batted-ball type/outcome, not a within-hits reallocation). Regression
coefficients fit via the same leakage-free walk-forward + average-across-pairs convention as
every other `expected_stats.py` layer. New walk-forward pitcher-side GB%/FB% snapshot builder
(`build_pitcher_gb_fb_rate_by_season`/`player_game_gb_fb_rate_snapshot`) mirrors the existing
batter-side groundball-rate builder's Marcel preseason+in-season structure.

**Full-stack A/B on the new n=7237 protocol (paired bootstrap, same games as the baseline):
SU delta +1.53pp, 95% CI (+0.22pp, +2.81pp) — REAL IMPROVEMENT, CI excludes zero. Brier
delta -0.0010, 95% CI (-0.0032, +0.0013) — NOISE (directionally favorable, not significant).**
The SU result is a genuine, statistically confirmed win — the first pitcher-side signal in
5+1 attempts this session to clear the full-stack bar. **KEPT and wired into all 3 live
consumer files** (`validate_game_simulator.py`, `validate_predictive_bullpen.py` — both
`pitcher_profile` and `roster_pitcher_profile` — and `props.py`, including its
nearest-prior-snapshot fallback path for live/future games). Task #120 closed.

**Claim 5 — total-runs dispersion never directly validated (diagnostic confirmed above, no
fix built).** See the PIT/z-score diagnostic above — real, moderate under-dispersion (std(z)
1.087, 99%-tail 2.5x too frequent). Two structural reasons suspected: fixed point-estimate
rates per trial (parameter/estimation uncertainty never widens within-game variance), and
PA-level conditional independence (real bullpen meltdowns/pitcher implosions cluster in a way
independent-PA sampling can't reproduce). **Not yet acted on** — no fix attempted this round.

**Claim 6 — scale the backtest to ~6,500-7,000 games (2023+2024+2025, same rules regime) and
decide SU via trial-level home-win share, not sign of mean margin — BUILT, and the result is a
genuinely important, humbling recalibration of every SU/Brier figure in this document.**
`validate_game_simulator.py`/`validate_predictive_bullpen.py`: `TEST_SEASONS` now
`{2023,2024,2025}` (confirmed 2427+2425+2425=7277 real complete-9-batter-lineup regular-season
games total), `N_GAMES_TO_VALIDATE` raised to effectively uncapped, `N_TRIALS_PER_GAME` lowered
200→50 (deliberate: prioritizes between-game sample size, which is what actually gates
resolving ~1pp-scale effects, over within-game trial precision, which mostly averages out
across thousands of games — keeps runtime tractable, ~12x more games at only ~1.2x the old
protocol's total compute). SU is now decided from `sim_home_win_prob > 0.5` (trial-level
home-win share) as the PRIMARY metric in both validators and in `ab_significance.py`, with the
old sign-of-mean-margin metric kept as a secondary comparison print, not silently replaced.

**Result at n=7237 games (current codebase state, both HFA + park-neutralization already
on)**: total MAE 3.543, margin MAE 3.442, **SU 56.3% (trial-level, PRIMARY) / 56.0% (legacy
sign-of-margin)**, Brier 0.2416, log loss 0.6758. This is MEANINGFULLY LOWER than every
SU/Brier figure reported earlier in this document at n=597-995 (58.0% SU / 0.2343 Brier at
n=597 just a few sections above; the historically-cited 59-60.5% figures from earlier in the
session even more so) — the two trial-level-vs-legacy SU definitions barely differ from each
other at this n (0.3pp apart), so claim 6's SECOND half turned out not to matter much in
practice, but its FIRST half (game count) reveals that this project's small-sample SU/Brier
estimates were likely inflated by real sampling variance, not just imprecise around a true
value near 58-60%. **Practical upshot: n=7237/50-trials is now the reference protocol and
baseline (56.3% SU / 0.2416 Brier / 3.543 total MAE / 3.442 margin MAE) for all future
full-stack decisions — treat every earlier-cited SU/Brier figure in this document as
superseded, not as a precisely-measured fact.** (`data/processed/game_simulator_validation.parquet`
holds this exact run's per-game output.)

**Three smaller "not working together" notes, not yet investigated:** (a) the stolen-base
module keys attempt/success purely off the runner, ignoring the opposing battery (pitcher
hold time, catcher arm/pop time) — post-2023 that's a real spread; (b) the bullpen model
discounts a reliever's availability by rest days but never his actual performance on
back-to-back appearances; (c) `AGE_PEAK = 29` (Tango's original figure) is roughly a decade
stale — modern aging research puts the offensive peak at 26-27, slightly mispricing every
young lineup. None built or tested this round.

---

### 11.9 Re-testing bat speed and pulled-air rate at n=7237 — a self-caught methodology bug,
then a genuinely tighter null

With the new n=7237 protocol built (§11.8 claim 6), the obvious next step was finally
resolving whether bat speed and pulled-air rate (§11.7's "plausible-but-unconfirmed" verdict
at n=597-995) are real. Built two scratch test arms forcing `pregame_bat_speed=None` /
`pregame_pulled_air_rate=None` in `batter_profile`, ran both against
`game_simulator_validation_BASELINE_7237.parquet`.

**First pass — wrong, and caught before acting on it.** The naive comparison showed both
signals as a statistically significant SU REGRESSION (bat speed: delta -1.41pp, CI
(-2.65pp,-0.14pp); pulled-air: delta -1.33pp, CI (-2.61pp,-0.06pp) — both excluding zero),
which would have meant reverting two previously-"kept" wins. Before acting on it, the reason
this looked suspicious was checked: the two new test-arm scripts were copied from
`validate_game_simulator.py` AFTER the GB/FB pitcher signal (§11.8 claim 4) had already been
wired into that file — but `BASELINE_7237.parquet` had been generated BEFORE that wiring. So
the "regression" actually being measured was `(GB/FB's own real +1.53pp gain) − (bat speed's
or pulled-air's own marginal effect)`, not either signal in isolation — an apples-to-oranges
comparison, not a real finding.

**Corrected comparison** (using `game_simulator_validation_GBFB_ON.parquet` — the TRUE
current-production state, GB/FB included — as the baseline instead):
- **Bat speed**: SU delta +0.12pp, 95% CI (-1.15pp, +1.41pp) — NOISE, includes zero.
- **Pulled-air rate**: SU delta +0.21pp, 95% CI (-1.09pp, +1.45pp) — NOISE, includes zero.

Both remain genuinely unresolved, matching §11.7's n=995 verdict — but now with a CI roughly
half as wide (~2.5pp vs ~4pp), giving real confidence this is a genuine null rather than
merely underpowered. **Neither reverted** — same standing as before, "plausible-but-
unconfirmed," not proven harmful or beneficial.

**Process lesson, stated plainly**: any new scratch test-arm script must be diffed against
(or copied from) the EXACT SAME commit/state as whatever it's being compared to — copying
"the current file" without checking what else changed since the reference baseline was
generated silently confounds the comparison. Caught here specifically because the result
looked too dramatic to trust at face value (reversing two previously-celebrated wins) —
the same "does this look right, or does it deserve a second look" discipline this project
has applied to itself all session (e.g. the HFA naive-comparison scare in §11.8's status
note, which turned out to be a real effect masked by noise rather than a bug, the mirror
image of this one).

---

### 11.10 Cost-reduction infrastructure + GB/FB full-mix extension (2026-07-22,
per external reviewer feedback on the FINDINGS_2026-07-22.md summary)

**Run-value screen (`src/models/run_value_screen.py`)**: a Stage-0 filter using standard
published linear weights (Tango/Lichtman/Dolphin) to estimate a candidate's plausible
per-game expected-run impact directly from real historical PAs, with zero Monte Carlo cost
— if the overwhelming majority of games show a negligible shift, a candidate can be killed
before running a single trial. Validated against a positive control (GB/FB: 88.6% of games
show a shift above the materiality threshold, correctly flagged "worth a full backtest")
and directionally against a negative control (rookie prior's tiny per-category rate shift).
Deliberately an upper-bound/rejection tool, not a substitute for the full simulator (ignores
`combine_matchup_distribution`'s renormalization and every context factor).

**Counter-based random numbers (`src/models/crn.py`)**: replaces the shared-sequential-RNG-
stream desynchronization problem (§11.7) at its root instead of just measuring around it.
Every stochastic decision (PA outcome, stolen-base attempt/success, TransitionTable
resampling) is now optionally keyed by a deterministic hash of `(game_pk, trial,
half_inning_ordinal, at_bat_index, decision_type)` instead of consuming the next draw off a
shared stream — purely additive (`crn_keys=None` default preserves byte-identical old
behavior for every existing caller). Validated by reproducing the confirmed GB/FB HR result
at ~1/30th the compute (n=1495 games/20 trials vs. n=7237/50): SU delta +1.14pp here vs. the
true +1.53pp, with per-pair paired-margin std reduced to 1.14 runs (vs. ~2.7 runs unpaired) —
consistent with the reviewer's predicted 5-10x efficiency gain. NOTE: only wired into
`game_simulator.py`'s oracle-backtest path (main PA outcome + stolen bases + transitions) —
bullpen reliever sampling and weather-bucket sampling (used only by `props.py`/
`validate_predictive_bullpen.py`'s predictive path, not the oracle backtest used for every
full-stack decision this session) are NOT yet CRN-keyed, a scoped-but-real gap for future work.

**GB/FB full-mix extension (task #126)**: per the reviewer's core insight — the same stable
GB%/FB% input predicts a pitcher's WHOLE allowed-mix, not just home runs. Leakage-free
walk-forward test (`scratchpad/fit_pitcher_gbfb_fullmix.py`) confirmed real R² gains for
double_play-share-among-BIP (+0.04 to +0.11, even larger than HR's own +0.05-0.07) and
double-share-among-BIP (+0.009 to +0.011, smaller but consistent across all 3 pairs); triple-
share showed a small, INCONSISTENT gain (near-zero in one of three pairs) and was correctly
not built. Built `pitcher_double_allowed_multiplier`/`pitcher_double_play_multiplier`
(same mechanism as the confirmed HR layer), gated behind a new `pitcher_fullmix=False`
default on `apply_contact_quality`/`build_profile` — the already-confirmed HR-only layer
stays live regardless. A fast CRN-paired check (n=1495/20 trials) was NOT encouraging: SU
delta -1.61pp, 95% CI (-3.61pp, +0.40pp) — includes zero (inconclusive) but leans negative,
in contrast to the clearly-positive lean the HR-only signal showed at this same reduced
scale during CRN validation. Combined with double_play's own coefficient sign instability
across season pairs (flagged in the code), **this extension is NOT deployed** — built and
available (`pitcher_fullmix=True`) but the fast check gives real reason for caution rather
than confidence; a full 7237-game confirmation run would be needed before trusting it either
way, and the current signal doesn't clearly justify that compute investment.

**Likely reason this graded out flat-to-negative, per an external reviewer (2026-07-22) —
worth remembering so a future session doesn't rediscover it the expensive way**: two things
were stacked against this extension specifically. First, this project's OWN batter-side
history already showed double/triple-share is a fundamentally low-signal target (R² 0.01-0.03
from either side, per §11.4's EV90/squared-up/bat-tracking checks) — pitcher-side
doubles-allowed was always drawing from a shallower well than HR-allowed, which is uniquely
slow-to-stabilize AND uniquely high-leverage, a combination the double/double_play targets
don't share. Second, and more general: **because `combine_matchup_distribution` renormalizes
the full outcome vector after every multiplier, a multiplier on ONE category already leaks
into every OTHER category in roughly the right direction** — suppressing a groundball
pitcher's HR share already pushes probability mass back into the other in-play categories
(more singles, more double plays) as a side effect of renormalization alone. Adding an
EXPLICIT double/double_play term on top partly re-applies an effect the renormalization had
already captured — exactly the shape of intervention that tends to grade out flat-to-negative
once actually tested. **General lesson for any future single-category multiplier proposal**:
check what the renormalized baseline already implies for sibling categories before adding an
explicit term for them.

### 11.11 Posterior-sampled rates: built correctly, does NOT close the dispersion
gap (2026-07-22) — the residual really is within-game correlation

Per the reviewer's suggested correctness fix for claim 5's under-dispersion (§11.8): every
Monte Carlo trial uses the SAME fixed point-estimate rate for a player, propagating outcome
randomness but zero parameter/estimation uncertainty. The Bayesian shrinkage formula this
project already uses everywhere IS a Beta(alpha, beta) posterior mean by construction
(`alpha = pregame_rate * effective_n`, `beta = (1-pregame_rate) * effective_n`); built
**`true_talent.sample_posterior_rate`** to actually draw from that posterior once per trial
per player instead of only ever using its mean, exposed `effective_n` as a new column
throughout `build_pregame_rates` → `build_wide_pregame_rates` → `player_game_snapshot` →
`build_profile`, and added **`game_simulator.resample_profile_rates`** (draws one fresh
rate per outcome category per player, called once per trial before `simulate_game`, not
once per PA — the whole point is that a player's true talent is ONE unknown quantity for
that trial's whole game, not independent per-PA noise, which the existing outcome sampling
already provides via a separate mechanism).

**Verified the sampling mechanism itself is correct** (not a bug): at effective_n=60/200/
500/1000/1500, the drawn samples show relative std of 72%/39%/25%/18%/15% respectively —
exactly the expected Beta-distribution shape, real and substantial variability at the
single-rate level.

**But wired into the full trial loop (n=1495 games/50 trials) and re-ran the PIT/z-score
diagnostic: essentially NO improvement.** std(z) went from 1.087 (original, no posterior
sampling) to 1.103 (posterior sampling on) — flat within noise, arguably slightly worse, not
meaningfully closer to the target 1.0. Tail frequencies were unchanged too (|z|>2.58: 2.5%
before vs. 3.0% after, vs. an expected ~1%). The per-game mean combined trial std barely
moved (4.055 → 4.126, roughly a 1.7% increase) despite individual player rates carrying
14-72% relative uncertainty — the aggregate team-total effect washes out far more than the
single-rate magnitude would suggest, most plausibly because `combine_matchup_distribution`'s
renormalization step partially cancels independent per-category perturbations for the same
player (a real dynamic not fully anticipated when this was designed, not yet root-caused
further given the scope already covered this session).

**This matches the reviewer's own explicitly-anticipated fallback**: "if a gap remains, the
residual is within-game correlation (pitcher meltdown clustering), which is a separate,
harder fix." The measurement confirms it directly rather than just falling back to it by
default. **Not deployed** — `resample_profile_rates` is built, correct, and available, but
provides no measured benefit to the diagnosed problem; deployed as-is it would only add
computational cost (more RNG draws per trial) with no accuracy gain. The actual fix for the
dispersion gap remains open and would need to model genuine within-game outcome correlation
(a persistent per-game "pitcher has it / doesn't have it today" latent state, or explicit
inning-to-inning momentum/fatigue effects) — a materially different and harder mechanism
than parameter uncertainty, not attempted this round.

### 11.12 The 2026 holdout result (Phase 0.3, 2026-07-22) — the model's real accuracy is
confirmed, not an artifact of selection bias

Built **`src/models/validate_holdout_2026.py`** (a permanent tool, not a throwaway script —
`TEST_SEASONS = {2026}` only, 2023-2025 remain fully available as prior-season training data
for every 2026 game's own walk-forward priors, no leakage) and ran it ONCE against the
current frozen production stack, per the reviewer's explicit rule against iterating against
a holdout. This is the first time any of this project's numbers have been checked against a
season that had zero influence — direct or indirect — on any fitting, selection, or
keep/revert decision ever made in this codebase (every one of them, going back to the
project's start, used 2023-2025 backtests).

**Result, n=1474 real 2026 games (2026-03-25 through 2026-07-19)**:
- **SU 58.3%** (trial-level, PRIMARY), 95% CI roughly ±2.5pp at this n (binomial SE ~1.3pp) —
  statistically indistinguishable from the n=7237 in-sample figure (57.9%). **The correct
  claim this supports is "no detectable degradation out-of-sample," NOT "the model performs
  better in 2026"** — deliberately calibrated wording (a reviewer correction, 2026-07-22) so
  a future session doesn't build on the stronger, unsupported version. If accumulated
  selection bias across this project's many keep/revert decisions were materially inflating
  the in-sample number, the holdout figure would be expected to come in noticeably BELOW it;
  it didn't come in below, and that absence of degradation is itself the meaningful finding
  — not the (statistically noise-level) direction of the small +0.4pp gap. Read the other
  direction, the same +0.4pp gap is reassuring in a different way: even the pessimistic edge
  of the CI still leaves the model at the top of the public/market-independent-model band.
- **Brier 0.2407** — nearly identical to the in-sample 0.2406 (GB/FB-confirmed run).
- **HFA calibration check**: mean sim_home_win_prob 52.61% vs. actual home win rate 52.10% —
  a 0.51pp gap, down sharply from the originally-diagnosed 1.7pp gap (§11.8 claim 1) that
  motivated building HFA in the first place. The HFA fix's own isolated n=597 A/B test
  couldn't distinguish it from noise (§11.8's status note) — this much larger, cleaner,
  held-out measurement suggests it's genuinely working better than that underpowered test
  could detect.
- **PIT/dispersion**: std(z)=1.085, frac|z|>2.58=2.10% — consistent with every other
  measurement of this same problem (§11.8 claim 5, §11.11). Confirms the under-dispersion is
  a real, persistent, moderate issue independent of which games/seasons are used to measure
  it, not an artifact of the training data specifically.

**Total/margin MAE (3.561/3.521) essentially match the in-sample figures too** — nothing in
this holdout run suggests the model performs meaningfully differently on genuinely new data
than on the seasons it was built and tuned against. **Per the reviewer's own rule: 2026 is now
the standing rolling holdout** — as future seasons complete, `validate_holdout_2026.py` (or
its direct successor, renamed for whichever season is current) should keep serving this
role, not get quietly absorbed into the fitting set the way 2023-2025 were.

**Phase 0.2 (full n=7237 re-baseline with the complete diagnostic suite) also completed**,
now durably logged in `data/processed/metrics_ledger.parquet` alongside the two historical
seed entries and the 2026 holdout: SU 57.9%, Brier 0.2406 — both reproduced EXACTLY
byte-for-byte against the earlier GB/FB confirmation run (task #122), confirming the
pipeline's determinism given the same code+seed. mean sim_home_win_prob 51.45% vs. actual
52.87% — a 1.42pp gap, notably WORSE than the 2026 holdout's 0.51pp gap. This is explained,
not concerning: `build_hfa_factors` is walk-forward (3-year rolling, prior-seasons-only), so
2023 (this dataset's own cold-start season) gets a neutral 1.0 HFA factor with zero real
correction — and 2023 is a full third of the n=7237 in-sample set. The 2026 holdout's HFA
factor, by contrast, is built from a complete 2023+2024+2025 window with no cold-start
dilution, which is exactly why its calibration reads cleaner. **The practical upshot: the
HFA fix's TRUE calibration quality is better represented by the 2026 holdout number (0.51pp
gap) than by the in-sample number (1.42pp gap) — the in-sample figure is an artifact of
pooling a cold-start season together with two fully-corrected ones, not evidence the fix is
under-performing.** Dispersion (std(z)=1.136) was somewhat worse than other runs' readings
(1.085-1.103) — most plausibly ordinary run-to-run variation in which specific games are
sampled, not a real regression, though not separately confirmed.

**A free improvement the cold-start explanation surfaces (per a reviewer, 2026-07-22), not
yet built**: if `build_hfa_factors`' first covered season gets a neutral 1.0 (zero real
correction) purely because its rolling window has no prior data yet, every OTHER
walk-forward factor with the same rolling-window structure — park factors, umpire tendency,
catcher framing, weather buckets — presumably has the identical cold-start degradation in
ITS own first covered season. Two cheap, not-yet-attempted fixes: (1) seed `build_hfa_factors`
with a long-run historical constant (published MLB home-field advantage has been stable
around ~52.5-53% for roughly a century) that gets blended out as real walk-forward data
accumulates, instead of a hard neutral 1.0 with zero correction; (2) more generally, audit
every walk-forward factor module for which season(s) it's silently neutral in, and give each
a sensible historical-literature prior for that gap rather than an unweighted default. Low
expected impact on 2026 predictions specifically (2026 already draws on a complete
2023-2025 window), but it makes the in-sample baseline more honest AND slightly improves
every future April's live predictions — precisely the point in a season where data is
thinnest.

**Holdout discipline, an operational rule to write down now rather than relearn expensively
later**: 2026 (or whichever season is the current rolling holdout) stays informative only if
it is read RARELY and never tuned against. Concretely: batch reads (e.g. one run per month
on newly-completed games, not continuous re-checking), log every read in the metrics ledger
(§0.1), and never make a keep/revert call from a holdout-season number — the moment a
decision consumes it, it quietly becomes training data and the whole point of holding it out
is lost. With ~1,200 more 2026 games still to be played this season, and 2027 arriving
behind it, this project has a perpetually self-refreshing holdout as long as this discipline
holds — the first genuinely trustworthy out-of-sample check this project has ever had access
to.

### 11.13 Phase 1.1 (day-level latent pitcher effect): a clean null, and the real
root cause of the dispersion problem

The roadmap's Phase 1 hypothesis was that under-dispersion (§11.8 claim 5, confirmed
repeatedly at std(z)≈1.08-1.14) comes from a missing day-level latent effect — some pitchers
have genuinely better/worse stuff on a given day than their season rates capture, and that
within-appearance correlation isn't modeled. Three tests, run in sequence — **the middle one's
original conclusion was corrected after a reviewer caught a real methodology bug; see the
correction below before reading this as settled**.

**First pass (naive, `phase1_1_dispersion_test.py`)**: compared each real start's
run-value-allowed against the pitcher's/team's own FLAT SEASON AVERAGE. Found apparent
overdispersion (std(z) notably >1 on both sides) — seemingly confirming the hypothesis.

**Refined pass (opponent-adjusted, `phase1_1_refined_dispersion_test.py`)**: the naive test
conflates two different things — genuine schedule-driven opponent-quality variation (a
pitcher facing a weak lineup one start and a strong one the next) with genuinely unmodeled
day-level variance. Rebuilt the reference point as a per-PA matchup-implied expectation
(batter's and pitcher's own season-long rates, odds-ratio combined, same mechanism as
`matchup.combine_odds_ratio`), summed to a start-level expectation + implied variance. Initial
result: std(z)=0.987 (pitcher appearances), std(z)=0.979 (team-offense games) — read at the
time as a clean null. **This reading was wrong — see the leakage-free correction directly
below.**

**Leakage bug found and fixed (`phase1_1_leakfree_test.py`), CORRECTING the refined-pass
conclusion above**: the refined test used FULL season-long rates as each PA's reference,
including that exact start's own PAs — a start's own outcomes feed into the season average
it's then compared against, mechanically pulling the "expected" value toward what actually
happened and attenuating any genuine shared-shock signal. Rebuilt with LEAVE-ONE-GAME-OUT
season rates (season totals minus this specific start's own counts, for both the batter AND
pitcher side of every PA) — same un-shrunk season-long statistical power, leakage channel
closed. **Corrected result: std(z)=1.0405 (pitcher appearances), std(z)=1.0471 (team-offense
games)** — both meaningfully above 1.0, with tail-frequency checks confirming a real effect
(`frac|z|>1.96`=0.057-0.060 vs. a 0.05 target, `frac|z|>2.58`=0.014 vs. 0.01). **The correct
reading: there IS a real, if modest, day-level shared-shock/within-start-covariance signal in
real data — the leakage bug was masking it, not disproving it.** Scale check against the
simulator's own gap: this confirmed excess (std(z)²-1 ≈ 8-10%) is smaller than the simulator's
full dispersion gap (std(z)≈1.08-1.14, i.e. ≈17-30% excess variance) — real, and probably
enough to close something like a third to half of the total gap on its own, but not, on this
evidence, the SOLE explanation. Treat a from-scratch latent-shock build as addressing part of
the problem, not all of it, and re-check the OTHER diagnosed mechanism (below) hasn't also
partly regained relevance once this one is wired in.

**The rate-spread-compression finding, found as a parallel follow-up (holds independently of
the leakage correction above)**: real season-long rates (the SAME un-shrunk reference used in
both overdispersion tests above) are 40-90% MORE spread across players than the SIMULATOR's own
walk-forward Marcel-shrunk pregame rates, per outcome category — home_run 1.72x, walk 1.58x,
strikeout 1.42x, single 1.92x (ratio = real spread / walk-forward spread). This is Marcel
shrinkage doing its predictive job (a rate estimator tuned to maximize PREDICTIVE correlation
on held-out data will always shrink harder than the population's true talent spread justifies,
since a lot of real between-player rate variance really is one-season luck) — but it
mechanically means every simulated matchup is closer to a league-average matchup than the real
matchup actually is, producing tighter simulated-outcome distributions independent of any
per-player sampling noise. This is a SEPARATE, ADDITIVE contributor to the dispersion gap
alongside the now-confirmed (modest) shared-shock effect above, not an alternative explanation
that displaces it — the attempted fix for this one (task #134, below) did not work, so this
component of the gap remains open.

**The fix attempted (task #134) — built, tested, NOT deployed (clean negative result)**:
`true_talent.widen_rate(pregame_rate, league_rate, w)` = `league_rate + w*(pregame_rate -
league_rate)`, clipped to `[0,1]`. A deterministic (non-random, unlike `sample_posterior_rate`)
stretch of each rate away from league average by factor `w`, applied once per profile (not
per-trial). `w=1.0` is an exact no-op. Wired as an opt-in `widen_w` param through
`build_profile` (applied to the raw Marcel rate dict BEFORE `apply_contact_quality`, matching
where the compression was measured) and threaded through `validate_game_simulator.py`'s
`WIDEN_W` module constant (default `1.0`, so the file's own default run is unaffected).
`validate_game_simulator.py`'s `__main__` block was refactored into two reusable functions —
`build_shared_tables(pa, test_seasons)` (every walk-forward table, built once) and
`run_validation(shared, test_seasons, n_games, n_trials, widen_w=1.0, ...)` (the per-game
simulate + score loop) — specifically so a `w`-sweep script rebuilds only the cheap per-game
profile/simulate step per candidate `w`, not the entire expensive upstream table-building
pipeline, and so it calls the exact same validated logic rather than a hand-duplicated (and
bug-prone) copy of it — this refactor is a real, reusable, permanent improvement independent
of the `w` result below.

**Sweep result, `w ∈ {1.0, 1.2, 1.4, 1.6, 1.8, 2.0}` on a 2023-2024 fit sample (n=1494 games,
25 trials/w, 2025/2026 never read)**:

| w | std(z) | SU | Brier | total MAE |
|---|--------|-----|-------|-----------|
| 1.0 (baseline) | 1.144 | 0.542 | 0.2499 | 3.527 |
| 1.2 | 1.173 | 0.572 | 0.2501 | 3.519 |
| 1.4 | 1.170 | 0.550 | 0.2501 | 3.545 |
| 1.6 | 1.203 | 0.571 | 0.2417 | 3.586 |
| 1.8 | 1.183 | 0.564 | 0.2487 | 3.572 |
| 2.0 | 1.202 | 0.572 | 0.2505 | 3.562 |

**No value of `w` improved std(z) — every widened value made the diagnostic WORSE than the
w=1.0 baseline**, with total MAE also drifting slightly worse at higher w. **Diagnosed
mechanism**: this validator's PIT/z diagnostic measures WITHIN-game Monte Carlo trial
variance for a game's own fixed, real matchup (`combined_std` = std across `n_trials` repeated
simulations of the SAME real lineup/pitchers) — not the across-player/across-game variance the
root-cause diagnosis (§11.13 above) actually measured. Stretching a rate away from league
average LOWERS that outcome's per-PA Bernoulli variance `p(1-p)` (maximized near 0.5, shrinking
toward the extremes), so widening mechanically shrinks the trial-to-trial spread for exactly
the extreme-talent players it was meant to help disperse more — the opposite of the needed
direction — while not correcting per-game mean bias enough to compensate. This is the same
class of null as §11.11's posterior-sampling result, arrived at via the opposite mechanism:
§11.11 resampled AROUND the existing (compressed) point estimate each trial and found no
dispersion improvement; this test deterministically MOVED the point estimate away from league
average and also found no improvement — worse, in fact, on this diagnostic. **Together these
results narrow down what will NOT fix the dispersion problem** (neither resampling within
nor recentering across the existing point-estimate framework), without yet identifying what
will. `true_talent.widen_rate` is kept in the codebase as validated, tested, dormant
infrastructure (like several other functions this session) — not wired into the live
production path (`WIDEN_W = 1.0` remains the default and the only value ever deployed).

**Open question for a future session**: since neither of the two point-estimate-level fixes
worked, the dispersion gap may need a fix at the `combined_std` computation itself — e.g. an
explicit variance-inflation factor fit directly against the PIT/z target (a calibration
correction, not a new mechanism), rather than another attempt to make the underlying rate
estimates themselves more dispersed.

### 11.14 Task #137: the latent per-pitcher-appearance shock — the dispersion fix that
actually worked, plus two things a future session must not misread

Per §11.13's leave-one-game-out re-test (std(z)=1.04-1.05, a real but modest shared-shock
signal), a reviewer specified a concrete mechanism: once per pitcher-appearance per trial,
draw `g ~ N(0, sigma^2)`, multiply the odds of on-base-allowed outcomes (walk, hit_by_pitch,
single, double, triple, home_run — NOT intent_walk, a strategic decision not a "stuff" effect)
by `exp(g)/exp(sigma^2/2)` (the lognormal mean-correction, making the multiplier exactly
mean-1 across trials), renormalize. Built as `game_simulator._pitcher_shock_factors` +
`GameSimulator._draw_pitcher_shock`, wired through `combine_matchup_distribution` /
`simulate_half_inning` / `simulate_game` as an opt-in `shock_sigma` param (default `0.0`,
confirmed byte-identical to the pre-existing codebase via a saved-reference diff before any
other code touched this file). The shock is redrawn once per REAL pitcher appearance (detected
via the same `id()`-change tracking `simulate_game` already used for TTOP resets), CRN-keyed
by `(game_pk, trial, side_tag, appearance_idx, DECISION_PITCHER_SHOCK)` — `side_tag`
(0=home's pitcher, 1=away's) substitutes for a real player ID, which was deliberately NOT
threaded into `GameSimulator` to avoid a larger, riskier plumbing change; it's unnecessary
since `(game_pk, side, appearance_idx)` already uniquely identifies one real appearance.
`crn_normal` (new in `crn.py`) derives a deterministic N(mean,std) draw via the inverse-CDF
of one `crn_uniform` call. Four unit checks passed before any backtest: mean-preservation
(200k-trial MC average matched the sigma=0 baseline to within noise), independent draws
per appearance, and CRN determinism/key-sensitivity.

**Component-level fit vs. full-stack fit diverged sharply, and the full-stack number is the
one that counts.** A no-simulator harness (fit on 2023-2024, real leave-one-game-out matchup
rates, no game-flow logic) put the std(z)=1.0 crossing at sigma~0.22-0.24. Wiring the SAME
mechanism into the actual simulator (`GameSimulator.shock_sigma`, `validate_game_simulator.
run_validation`'s new `shock_sigma`/`crn_pairing` params) and sweeping `sigma in
{0.10,...,0.70}` on the real 2023-2024 protocol (n=1196-1200 games, CRN-paired) found the
crossing at sigma~0.40-0.44 instead — nearly double, in the OPPOSITE direction from the
naive expectation (the full sim already carries other variance sources, e.g. reliever
sampling in props.py's predictive path, so it was expected to need LESS added shock, not
more). **Most likely explanation: the odds-space multiplier + per-PA renormalization
attenuates the shock's real-world effect once it interacts with the simulator's full
game-flow logic (base-out transitions, walk-off truncation, blowout substitution) — not a
bug, but it means sigma=0.40 is a SIMULATOR-INTERNAL calibration constant, not a claim about
how much real pitchers' game-to-game "stuff" actually varies.** Concretely: if a batter-side
day-effect or any other new variance source is ever added later, sigma must be REFIT down at
that point, not left at 0.40 and stacked — the two mechanisms would otherwise be tuned to
each absorb the SAME missing variance twice. Picked sigma=0.40 over the interpolated crossing
(~0.44) per the explicit rule: between two bracketing values, take the lower one — residual
under-dispersion is the known, familiar failure mode; overshooting into over-dispersion would
invert every prop's bias in an unfamiliar direction.

**A real scare, investigated and resolved, not just asserted away.** The initial sigma sweep
(0.0 to 0.22, CRN-paired, n=1196) showed SU climbing (55.7%→57.8%) and Brier falling
(0.2468→0.2402) smoothly and monotonically as sigma increased — exactly the pre-registered
"mean-correction is leaking" failure signature. Two follow-up diagnostics, not just a re-read
of the same numbers, resolved it: (1) directly measuring the realized shock draws across the
whole backtest population (693,933 draws) found mean(g)=0.00074 (SE=0.00026, ~2.8 SE from
zero but utterly negligible in magnitude — exp(0.0007)≈1.0007) and std(g)=0.2199 (vs. target
0.22) — the raw randomness is clean, ruling out a biased-hash-sample explanation; (2) a
direct per-game comparison at sigma=0 vs. 0.22 found the mean simulated margin shift was
-0.0012 ± 0.0193 (SE) — statistically zero — while 317/1196 games (26.5%) flipped their SU
call, split 171-wrong-to-right vs. 146-right-to-wrong (54%/46%, only 1.4 SE from an even coin
flip, not significant). **Lesson for reading this project's own CRN-paired sweeps**: every
point in such a sweep shares the exact SAME underlying hash draws, just scaled by a growing
sigma — a SMOOTH trend across sweep points is therefore a built-in consequence of the CRN
design, not independent evidence of a systematic effect, the way it would be for genuinely
independent samples. Don't over-read sweep-smoothness as significance again.

**The canonical (non-CRN, n=7237, 50-trial) full-stack decision run**, sigma=0.40, logged to
`metrics_ledger.parquet`: std(z) 1.136→**1.015** (the best dispersion result of the entire
session — decisively better than both prior attempts, §11.11's null and §11.13's negative
widening result), frac|z|>1.96 0.0754→0.0482 (now under the 0.05 target), frac|z|>2.58
0.0327→0.0192. Point-metrics moved slightly against: SU 57.86%→56.74% (-1.12pp), Brier
0.2406→0.2425, total MAE 3.537→3.560, margin MAE 3.437→3.458 — small (~1.1-1.4 SE for an
UNPAIRED comparison at this n) but coherently in the same direction across all four.

**Protocol lesson (the reason this section exists, not just the numbers above)**: a reviewer
correctly identified that the canonical 50-trial UNPAIRED protocol is structurally biased
against exactly this class of change. SU is the sign of a K-trial-estimated mean margin, and
Brier/MAE are built from the same finite-K point estimates — their sampling noise scales with
per-trial variance / sqrt(K). A mechanism whose entire purpose is to RAISE per-trial variance
mechanically raises the standard error of every one of those K=50 estimates, which flips more
close-call games, blurs win-share estimates, and inflates MAE — genuinely regardless of
whether the underlying model got better, staying the same, or worse. Every other factor kept
this session left within-game variance untouched, so the canonical protocol was a fair
yardstick for all of them; it is NOT automatically a fair yardstick for a variance-changing
mechanism, and this was the first one built. **Rule for future sessions: any change that
alters within-game/within-trial variance (not just point estimates) must be evaluated with a
trial-count-robust method** — CRN-paired at high K, or (the actually decisive design) a
K-scaling check: compute the sigma=0-vs-sigma>0 gap in SU/Brier/MAE at several K values on the
SAME CRN-paired games (reusing one run at the largest K and computing K-subset statistics
post-hoc, since CRN trial draws are a pure function of trial index — no need to actually
rerun at each K). If the gap shrinks toward zero as K grows, it's a K=50 estimator artifact,
not a real cost; if it's invariant to K, it's a genuine calibration-vs-picks tradeoff worth an
honest conversation. The canonical run's raw point-metric numbers for this specific change
(SU -1.12pp etc.) should NOT be read as a settled fact about the model without that check.

**Also worth remembering**: production props (`props.py`) run at K=1000+ trials, roughly 20x
the canonical backtest's K=50 — whatever estimator penalty the canonical run measured, the
actual deployment environment pays only a small fraction of it.

**The K-scaling check result (decisive, see the design above)**: `run_validation` gained a
`trial_capture` param (raw per-game trial arrays, only populated when passed a dict — a
no-op for every existing caller) so K-subset statistics could be computed post-hoc from ONE
run at the largest K per arm, instead of rerunning per K value (CRN trial draws are a pure
function of trial index, so trials 0..29 are identical whether K=30 or K=300 was requested).
CRN-paired, n=697, K∈{30,100,300}:

| K | SU gap (σ.4−σ0) | Brier gap | MAE gap |
|---|---|---|---|
| 30 | +0.0014 | -0.0026 | -0.034 |
| 100 | -0.0158 | -0.0002 | -0.013 |
| 300 | -0.0072 | +0.0003 | -0.008 |

Brier and MAE gaps shrink toward zero monotonically as K grows — the textbook signature of
a K-dependent estimator artifact, not a persistent model cost. SU is noisier (not perfectly
monotonic) but its K=300 value (-0.72pp) is meaningfully smaller than the canonical run's
K=50 value (-1.12pp), the same direction the artifact hypothesis predicts. None of the three
metrics show a gap invariant to K, which is what a genuine model regression would require.
**Verdict: KEPT. σ=0.40 is now the frozen production value** — `game_simulator.SHOCK_SIGMA`
imported into `props.py` and `validate_predictive_bullpen.py`'s own `GameSimulator(...)`
construction (previously only `validate_game_simulator.py`'s oracle backtest had it wired),
so the shock is live in the actual deployed props path, not just the validator.
`validate_holdout_2026.py` was rewritten as a thin wrapper around
`build_shared_tables`/`run_validation` (it was still a hand-duplicated pre-refactor copy)
and re-run once with σ=0.40 frozen in, per holdout discipline (one batched read, logged,
never tuned against).

### 11.15 Task #138: the prop-calibration refit — not just new coefficients, a smaller set

`BATTER_PROP_CALIBRATION`'s 4 linear (a,b) corrections were fit against the OLD,
under-dispersed simulator's RAW probabilities. With σ=0.40 now widening the distribution
(raw probabilities already less overconfident), refit was necessary. Ran
`validate_prop_calibration.collect_prop_predictions` (150 real games, 150 trials,
2024-2025) with `_apply_batter_prop_calibration` temporarily monkeypatched to identity so
the observed "predicted" probability was the RAW one, then fit fresh (a,b) per prop.

**The single-fit refit alone would have been a mistake to deploy.** The ORIGINAL
methodology's own bar (stated in props.py's comment) was 5 independent random train/test
splits, each checking whether the correction beats the raw model's Brier score
out-of-sample, keeping only props that win 4-5/5 splits — precisely how `p_1plus_hr` was
excluded before (2/5, unstable). Re-running that same 5-split check against the fresh
post-shock data (reusing the already-collected sample, no new simulation needed) found:

| prop | wins/5 | verdict |
|---|---|---|
| p_2plus_hits | 5/5 | keep |
| p_1plus_bb | 4/5 | keep |
| p_1plus_hit | 3/5 | **now below the bar — drop** |
| p_1plus_rbi | 2/5 | **now below the bar — same instability signature as p_1plus_hr — drop (later reversed, see §8.3: 2026-07-25 re-check on fresh data found 5/5 — now deployed)** |

**The shock didn't just change the coefficients, it changed which props need correction at
all.** `p_1plus_hit` and `p_1plus_rbi` are now left uncorrected (same treatment as
`p_1plus_hr`), `p_2plus_hits`/`p_1plus_bb` keep corrections with refreshed coefficients:
`p_2plus_hits: (0.1106, 0.4892)`, `p_1plus_bb: (0.0804, 0.6442)`. Every surviving slope
moved closer to 1.0 (the no-correction ideal) than its pre-shock value — `p_2plus_hits`
0.393→0.489, `p_1plus_bb` 0.590→0.644 — independent, prop-level confirmation that the shock
is doing exactly what it's supposed to (raw model outputs are less overconfident now).
**This is stronger evidence than a mere coefficient update — it's convergent, mechanism-level
confirmation that σ=0.40 is fixing something real, not just gaming the std(z) metric.** These
4 corrections existed in the first place BECAUSE an under-dispersed simulator produces
systematically overconfident per-player prop probabilities (too little spread → predictions
too extreme → a post-hoc linear squeeze was needed to correct them back toward reality). Now
that the shock widens the underlying distribution at its source, 2 of those 4 band-aids come
off on their own, via an entirely independent statistical test (5-split Brier stability, not
the PIT/z diagnostic the shock was originally tuned against). A fix that only gamed one
specific metric would have no reason to also make an unrelated, previously-necessary
correction unnecessary — this convergence is exactly what you'd expect from a real
distributional fix and not from what a metric-specific patch would produce.
**Lesson for any future refit of a calibration correction that was originally validated via
a stability check**: re-run the SAME stability check on fresh data, don't just re-fit
coefficients for the same fixed set of props and assume the set itself hasn't changed —
here it had.

**`p_6plus_k` (pitcher K prop) re-checked post-shock (task #139, after fixing a script bug
that crashed the first attempt)**: `actual = 0.0306 + 0.9195*predicted`, corr=0.988, Brier
0.0536 vs. naive 0.0874 — slope 0.92 is close enough to the 1.0 ideal that the original
finding (pitcher props need no correction) still holds post-shock. Makes sense structurally,
independent of the shock: a pitcher's ~20-25 batters faced per game is a much larger
single-game information budget than a batter's ~4 PA, which is why this side was never as
miscalibrated as the batter props in the first place. The prop surface is now fully verified
under the new distribution — nothing left unchecked from task #137's downstream effects.

### 11.16 Protocol convention (adopted 2026-07-23): the canonical K=50 point-metrics
are a dispersion-protocol reference, NOT the settled cost of the shock — read this before
ever quoting `su_primary`/`brier` from a raw ledger row for a variance-changing factor

This is a standing convention, not a one-off caveat — the risk it closes is specific and
real: the canonical protocol's own SU/Brier numbers for task #137 (56.7%/0.2425, vs. the
pre-shock 57.9%/0.2406) are inflated by the K=50 estimator artifact documented in §11.14.
Read naively, "the shock cost 1.2pp of SU" is a false, and dangerous, one-line summary — a
future session (human or LLM) skimming the ledger could revert the best dispersion fix this
project has built based on a number that measures an estimator's noise floor, not the model.

**The TRUE point-metric cost, from the K=30/100/300 CRN-paired check (n=697, 2023-2024,
§11.14)**: at K=300, SU gap ≈ **-0.7pp**, Brier gap ≈ **+0.0003** — both far smaller than
the K=50 canonical read, and consistent with the gap continuing to shrink at even higher K
(production props run at K=1000+, ~3x further still). **This is the number to cite as the
shock's real point-metric cost, not the raw K=50 canonical delta.**

An identical K-scaling check was STARTED on the 2026 holdout itself (motivated by that
run's own K=50 gap reading 2-5x larger than the canonical run's) but was killed partway
through (still on the first of two arms after ~10 minutes) as a deliberate call that this
specific confirmatory run wasn't worth its cost given the mechanism is identical and the
2023-2024 result already generalizes on first principles (the K-dependent estimator-noise
argument doesn't depend on which season's games are used). **Flagged honestly, not silently
dropped: the 2026 holdout's own point-metric gap has NOT been independently K-scaled** —
if a future session wants full certainty rather than reasoned inference from the
2023-2024 result, re-run `phase1_shock_kscaling_holdout2026.py`-equivalent logic (same
design as the 2023-2024 script, just `HOLDOUT_SEASONS = {2026}`) to close this specific
loose end. Low priority — nothing about the current evidence suggests a season-specific
effect, this is just an unclosed verification, not an open concern.

**Concrete steps taken so this can't be missed**: (1) both relevant `metrics_ledger.parquet`
rows (the canonical task #137 run and the 2026 holdout run) got their `notes` field
amended with this exact convention and the true-cost estimate, so the ledger is
self-documenting at the row level, not just here; (2) this section exists specifically so a
future reader who skips straight to a ledger row still hits the same warning inside the row
itself.

**Standing rule for every future variance-changing factor** (not just this one): a K=50
unpaired canonical run remains valid for measuring DISPERSION (std(z), tail coverage — those
aren't K-sensitive point estimates in the same way). For POINT-METRICS specifically, either
(a) run a CRN-paired K-scaling check (cheap — reuses one high-K run's raw trial data via
`run_validation`'s `trial_capture` param, no need to actually rerun per K) and quote the
high-K/extrapolated number, or (b) raise the canonical protocol's own `N_TRIALS_PER_GAME`
(currently 50) for a permanent, higher-precision baseline — not done yet in this codebase
(a full n=7237 run at K=200 would cost roughly 4x the current ~50min runtime, a real but
bounded one-time cost) — logged as an open item in §12, not executed this session given the
CRN-paired K=300 check already answers the immediate question cheaply. Do NOT quote a bare
K=50 unpaired point-metric delta for a variance-changing factor without one of these two
checks alongside it.

### 11.17 The monitoring plan (adopted 2026-07-23) — now that the distribution is fixed,
the priority shifts from building to watching, on a schedule precise enough that holdout
discipline can't quietly erode

With dispersion genuinely fixed (§11.14-11.16) and props recalibrated against it (§11.15),
a reviewer's explicit call — endorsed and adopted — is that the right next move is
operational, not architectural: stop adding signals, start watching whether this one holds
up. Two concrete pieces, not just "monitor it" as a vague intention:

**(1) Pre-committed rolling-holdout read cadence, fixed NOW so it can't become reactive
later**: read `validate_holdout_2026.py` (or its direct successor for whichever season is
current) on the **first Monday of each month**, covering only newly-completed games since
the last read, logged to `metrics_ledger.parquet` every time (already the default —
`write_ledger=True`). Committing to a fixed calendar cadence in advance, rather than
"whenever it seems worth checking," is the whole point — the temptation that erodes holdout
discipline is reading it reactively after a bad week of picks, which is exactly when a read
is most likely to trigger an unjustified tuning response. A read that was always going to
happen on schedule, regardless of how recent results looked, carries no such risk. Per the
existing rule (§0.3 origin, restated in multiple places above): never make a keep/revert or
tuning decision from a holdout number, batch reads, this schedule IS the batching.

**(2) Live production-path (K=1000+) prop calibration monitoring — this is the ground truth
the whole session was ultimately in service of, and it has never once been checked.**
Every validator this session (`validate_game_simulator.py`, `validate_predictive_bullpen.py`,
`validate_prop_calibration.py`) runs the ORACLE or semi-oracle path — real known lineups/
starters, at most a sampled bullpen. What actually gets posted to production
(`generate_daily_props.py` -> `props.py` at K=1000+) runs the FULLY predictive path:
forecast weather (not actual recorded conditions), projected lineups (not confirmed ones,
early in the day), and sampled bullpen usage throughout — sources of real-world miss that
none of the oracle-backtest reads above ever exercise. A monthly check (same first-Monday
cadence as (1), same never-tune-against-it discipline) of ACTUALLY-POSTED prop
probabilities vs. real outcomes — reusing `validate_prop_calibration.py`'s own
`calibration_report`/`report_calibration` machinery, but against the live-generated
`generate_daily_props.py` output/logs rather than a backtest re-simulation — is the one
validation this project has never done end-to-end: does the thing the Discord actually
consumes hold up, not just the thing the validator simulates. Building the harness for this
(logging every posted prop probability alongside its eventual real outcome, on an ongoing
basis, not just backtest-reconstructed) is the concrete next piece of infrastructure, if it
doesn't already exist — check `generate_daily_props.py`'s own output persistence before
building anything new.

**Explicit acknowledgment of the risk being managed**: per the reviewer, "the failure mode
for a project in this state isn't a missing signal; it's restlessness adding an unneeded
one." The instinct to keep building (another factor, another correction, another edge case)
is the wrong instinct right now — the model has a verified ~58% on genuinely untouched data,
a correctly-shaped distribution, the first confirmed pitcher-side signal, and a full testing
lab (CRN pairing, run-value screening, the metrics ledger, K-scaling checks) that makes
running down a wrong idea cheap. Come back to the open items (§12, task #131's cold-start
audit, task #140's canonical-K upgrade) when the monitoring above actually surfaces a
reason to, not on a schedule driven by restlessness.

### 11.18 Task #143: the oracle-vs-deployable gap, decomposed — bullpen usage guessing is
the entire gap, not lineup timing (overturns the roadmap's own prediction)

Every accuracy number this project has ever reported (SU ~57-58%, Brier ~0.24, std(z)~1.0)
was measured on the ORACLE path: real confirmed lineups, real per-inning bullpen sequence,
real recorded weather, real starting catchers (`validate_game_simulator.py`). The actually
deployed path (`props.py`/`generate_daily_props.py`) guesses or infers all four when the real
information isn't posted yet. This gap had never been measured end to end before this task.

**Method**: built `src/models/validate_oracle_vs_predictive.py` (permanent tool), a
generalized decomposition validator that independently resolves each of the 4 dimensions via
either its oracle (real) or predictive (guessed) source, CRN-paired throughout so any two
configs agreeing on a dimension stay synchronized on that dimension's own randomness. Reused
every existing validated mechanism rather than re-deriving anything: `project_lineup_
platoon_aware` (lineup_projection.py), `sample_bullpen_plan` + `build_team_bullpen_roster`
(bullpen.py), `identify_starting_catcher` (catcher_framing.py), and `climatological_bucket_
distribution` (weather_forecast.py) for the predictive weather arm — see the module's own
docstring for why the live forecast-narrowing tiers structurally can't be exercised in a
historical backtest (Open-Meteo's forecast endpoint can't serve arbitrary past dates), making
the weather arm's cost an upper bound, not an exact figure. One real bug caught and fixed
before trusting any result: a projected lineup can name a player who didn't actually bat in
that exact historical game_pk, and the per-game-scoped `batter_snap` lookup was silently
skipping 58/60 games in an early sanity check before the fix (reuse `nearest_prior_pitcher_
snapshot` — despite its name, it auto-detects a `batter` column too).

**Result, full canonical scale (n≈7229-7276, K=50, CRN-paired, 2023-2025)**:

| Config | SU | Brier | total MAE |
|---|---|---|---|
| FULL_ORACLE | 56.72% | 0.2417 | 3.532 |
| FULL_PREDICTIVE | 54.05% | 0.2479 | 3.623 |
| ORACLE_LINEUP | 54.26% | 0.2492 | 3.615 |
| **ORACLE_BULLPEN** | **58.00%** | **0.2400** | **3.546** |
| ORACLE_WEATHER | 53.88% | 0.2481 | 3.618 |
| ORACLE_CATCHER | 54.13% | 0.2475 | 3.624 |

Total oracle-vs-predictive gap: 2.67pp SU, 0.0062 Brier, 0.091 MAE. **Bullpen alone recovers
3.95pp SU / 0.0079 Brier — MORE than the entire observed gap** (the expected signature of one
dominant term plus ordinary cross-config sampling noise from each config's own slightly
different skip-condition game subset, not a methodological red flag — ORACLE_BULLPEN's raw
SU edging past FULL_ORACLE's own 56.72% is exactly that noise, not evidence predictive inputs
somehow beat oracle ones). Lineup (+0.21pp), weather (-0.17pp, noise), and catcher (+0.08pp)
are all within noise of zero.

**This overturns the specific prediction that motivated this whole roadmap item**: the
expectation was lineup-timing uncertainty would dominate, with an operational fix (regenerate
props on lineup confirmation) as the natural next step. Instead, **bullpen usage prediction
(`sample_bullpen_plan`) is where essentially all of the deployable-path's real-world cost
lives.** The lineup-regeneration pipeline is not cancelled as a good idea, but it is no longer
the top-priority fix implied by the original roadmap.

**Verification pass (same day, reviewer-requested, before trusting the headline numbers) —
two real catches, both resolved from the already-saved parquet output at zero new compute**:

1. **Arithmetic check**: the "3.95pp bullpen recovery exceeds the 2.67pp total gap" is not a
   bug. Paired CIs (n=7228 games common to all 6 configs) show lineup (+0.17pp, t=0.28),
   weather (-0.17pp, t=-0.39), and catcher (+0.08pp, t=0.30) are all genuinely indistinguishable
   from zero — the "excess" is three noise terms combining around one real, highly significant
   bullpen effect (SU t=6.12, Brier t=-7.62; total gap SU t=4.04, Brier t=-5.64). No harness
   bug, no negative-swap red flag.

2. **Endogeneity check (the important one)**: the ORACLE_BULLPEN arm doesn't just give the
   simulator a better bullpen-usage PREDICTION — it gives it the REAL historical sequence,
   which is causally downstream of how the game itself unfolded (a blowout gets the mop-up
   arm; a tight game gets the closer; a shelled starter exits in the 4th). No pregame predictor
   can recover that — a chunk of the 3.95pp is future information leaking in, not achievable
   headroom. Sliced by real final margin: close games (±2 runs, n=3311) show +2.11pp SU /
   -0.0047 Brier; medium (3-5 runs, n=2485) +6.12pp / -0.0108; blowout (6+, n=1432) +4.47pp /
   -0.0102. **This is a genuinely mixed result, not a clean "it's all leakage" story**: pure
   endogeneity would predict close games show ~nothing (a good predictive sampler should
   already reproduce close-game usage without outcome knowledge, since managers deploy their
   best arms in close games regardless of final score) — instead close games retain roughly
   half the effect size of medium/blowout, meaning a real, pregame-recoverable signal survives
   underneath the leakage. **The close-game slice (~2pp SU), not the full 3.95pp, is the
   honest estimate of achievable headroom** for a bullpen-prediction improvement.

**The other three dimensions being genuinely knowable pregame reframes the good news too**: on
every input that's actually fixed before first pitch (lineup, weather, catcher), the deployed
model already performs at oracle level — the predictive path isn't leaking accuracy through
those guesses. The real, recoverable residual gap is specifically in bullpen usage, and it's
smaller than the headline number but still real.

**Next build, properly scoped by this result**: not "oracle bullpen knowledge" (impossible to
deploy) but **state-conditioned bullpen usage within the simulator** — instead of sampling the
full bullpen plan once, pregame, and having the simulated game follow it regardless of how the
trial unfolds, move reliever resolution to be responsive to the SIMULATED game state as it
develops within each trial (a shelled starter exits earlier in that specific trial; a close
8th brings the high-leverage arm; a trial that becomes a blowout gets the mop-up profile).
This reproduces the causal structure the oracle arm was leaking, using only pregame-knowable
inputs (existing roster usage weights, closer identification, per-trial game state already
available in `simulate_half_inning`) — not new heterogeneity hoped to be signal, but a
mechanism targeting a measured, decomposed, partially-real defect. Per this project's own
§11.5 history of heterogeneity-axis failures, judge this on the bullpen-gap closing and
Brier/CRPS specifically (SU as a guardrail only, given its noise floor relative to Brier's).

**Important correction to the ~2pp figure (reviewer, same day): it's a FLOOR for this
mechanism, not a ceiling.** The endogeneity critique says the oracle's medium/blowout
advantage is unrecoverable by a PREGAME-FIXED plan — true, and that's what the ~2pp
close-game estimate bounds. But the state-conditioned mechanism isn't a pregame plan; it
reproduces the causal structure INSIDE each trial. In a trial that becomes a blowout, the
mop-up arm pitches in that trial; in a trial that stays close, the leverage arms do. It
can't know in advance which script reality will follow, but it makes each simulated script
internally consistent — which is exactly what the oracle's medium/blowout edge was
rewarding. So realistic capture is the close-game ~2pp PLUS some distribution-level
fraction of the rest, showing up more in Brier/CRPS and joint-outcome calibration than in
raw SU (a single point-estimate metric can't fully reward "the simulator's OWN
trial-to-trial distribution is now internally consistent across scripts" the way a
distributional metric can). Don't undersell the target when scoping/reporting this build.

**Full build spec (fit-then-validate, in order — written down now so a future session
starts at step 1 with nothing to reconstruct)**:

1. **Fit the usage policy from real data FIRST, no simulator code touched yet** — same
   Phase-1.1 discipline (measure the mechanism before wiring it) applied again. Two
   components, both estimable from the PA table already in hand (running score is
   derivable per game): (a) a starter-hook model, `P(starter exits | inning, runs allowed,
   PA count, score margin)`; (b) a reliever-selection model, `P(reliever tier | inning,
   margin, save situation)`, where tiers reuse the existing usage-weighted roster
   (`build_team_bullpen_roster`) and `identify_closer`. Validate the policy STANDALONE
   before any simulator wiring: does it reproduce real usage patterns (leverage-arm share
   in close-late states, mop-up share in blowouts, real hook-timing distributions) on
   held-out 2025? A policy that doesn't pass this check has no business going into the
   simulator regardless of how well-motivated the mechanism is.
2. **CRN-key every new draw** — the hook decision and reliever selection each get their
   own counter-based keys (`game_pk, trial, inning, decision_type`, new `DECISION_*`
   constants in `crn.py`, same discipline as `DECISION_PITCHER_SHOCK`). The existing
   pregame-sampled plan (`sample_bullpen_plan`) stays as the fallback path behind a flag,
   with the exact same byte-identity discipline task #137 used: flag OFF must reproduce
   current production exactly, checked before anything else.
3. **Integrate with, don't duplicate, `blowout.py`** — the position-player-pitching logic
   is a special case of the same underlying policy (an extreme point on the same
   margin/leverage axis) and should become one code path, not two competing mechanisms
   making independent decisions about the same thing.
4. **Pre-register the σ interaction** — state-conditioned usage adds real within-game
   variance on top of what `SHOCK_SIGMA` already models, so `std(z)` will likely dip BELOW
   1.0 once this lands. The standing rule from task #137 applies: refit `SHOCK_SIGMA` DOWN
   to restore the target, don't stack the two variance sources unchanged. **Expect the
   frozen 0.40 to shrink — that's the system working as designed, not a regression** (this
   is being written down in advance specifically so a future session doesn't misread that
   dip as a new problem).
5. **Acceptance criteria**: re-run `validate_oracle_vs_predictive.py`'s decomposition — the
   ORACLE_BULLPEN-vs-FULL_PREDICTIVE delta is the headline number, and should shrink
   materially toward (but plausibly land somewhat above) the ~2pp close-game floor, per the
   "floor not ceiling" correction above. Brier/CRPS should improve on a PAIRED basis.
   Dispersion should return to target after the σ refit in step 4. SU is a guardrail only,
   not the arbiter (its noise floor is exactly why Brier's t-stats, not SU's, were what
   made the arithmetic-verification pass above conclusive). Fit on 2023-2024, validate on
   2025 (held out from fitting), one batched 2026 holdout read after freezing — standard
   holdout discipline, no exceptions for this being a bigger build.

### 11.19 Task #144 step 1: bullpen usage policy fit + validated against real
2025 data (simulator NOT touched yet)

Per §11.18's 5-step build spec, step 1 (fit the usage policy from real data first, no
simulator code touched) is now complete. New permanent module: `src/models/
bullpen_usage_policy.py`. Two components, both fit on 2023-2024 PA data, validated against
held-out 2025:

1. **Starter-hook model**: `P(starter exits | inning, PA count faced, cumulative runs
   allowed, score margin from the defense's own perspective)`, Bayesian-shrunk toward the
   inning-only marginal (pseudo-count 30, same discipline as `state_factors.py`/
   `park_factors.py`). `attach_margin()` reconstructs each PA's `margin` (defense's own
   score minus batting team's score) via a per-half cumulative-sum lookup, since the PA
   table only carries the batting team's own score directly.
2. **Reliever-tier model**: `P(tier appears | margin, inning, save situation)`, tiers =
   season-long usage-rank terciles within each team-bullpen (mopup/middle/leverage);
   closer identified separately via 9th+-inning appearance concentration.

**Validation result (2025 held out from fitting)** — both components confirmed real and
stable, not overfit:
- Mean real hook inning by margin bucket: trail_big=4.45 (n=551), trail=5.24 (n=1481),
  tied=5.20 (n=771), lead=5.73 (n=1403), lead_big=6.30 (n=654) — clean, monotonic.
- Leverage-tier share by margin, fit (2023-24) vs. validation (2025): lead 82.3%→81.8%;
  lead_big 69.5%→70.5%; tied 78.4%→80.9%; trail 67.4%→70.7%; trail_big 53.1%→53.6% —
  nearly identical across the fit/validation split.
- Closer share of relief PAs at (inning 9, save situation): 60.6% (fit) vs. 57.2%
  (validation) — strong, reproducible.

**This directly supports proceeding with steps 2-5** (CRN-keying, `blowout.py`
integration, `SHOCK_SIGMA` pre-registered refit-down, decomposition-based acceptance
criteria) whenever that build happens — the policy underneath it is real and confirmed,
not a hoped-for signal. Steps 2-5 are NOT started; this module is not imported by
`game_simulator.py`, `props.py`, or any production path.

### 11.20 Task #145: hook-frailty — the pre-registered tail check caught a real gap,
diagnosed it as a selection effect, and fixed it with a decaying per-start latent draw

Before wiring §11.19's hook model into the simulator, a reviewer-requested check asked:
does the model's calibration hold on the DISTRIBUTION of exit innings conditional on
runs-allowed-so-far, not just the mean-by-margin summary? It didn't. Per-cell hazard was
well-calibrated in isolation (inning 4/5+ runs cell: 10.6% fitted vs. 10.7% real), but
**sequentially compounding that hazard across a start's own PAs** — exactly how the
simulator would evaluate it trial-by-trial — badly overpredicted the "shelled starter
hooked early" scenario: real P(exited by inning 4 | reached inning 4 with 5+ runs
allowed) on held-out 2025 = 29.3%, naive per-PA compounding implied 42.0%.

**Diagnosis, confirmed by elimination before accepting frailty as the explanation**:
recency refit (2024-only fit) closed only ~1pp of the gap; evaluating hazard once per
half-inning boundary instead of per-PA overshot in the OPPOSITE direction (16.4% implied
vs. 29.3% real); damping only the "not yet in trouble" cells barely moved the compound
rate; a properly-specified discrete-time logistic hazard (MLE-fit, self-consistent under
its own compounding by construction) closed only ~1/4 of the gap. The real explanation is
a **selection effect**, frailty's textbook signature: "reached inning 4 with 5+ runs
allowed" is not a random sample of starts — a quick-hook manager pulls the guy at 3 runs,
so that start never enters the evaluation set. The starts that DO reach that state are
disproportionately the ones with a low hook-propensity that day (tired pen, a manager
who's already written the game off). A marginal per-PA hazard model, of any functional
form, cannot represent this correlated-propensity structure.

**Fix**: `src/models/bullpen_usage_policy.py` now has a per-start frailty mechanism
(`apply_hook_frailty`, `hook_frailty_sigma`) — one latent `z~N(0,1)` drawn once per
(start, trial) and reused across every inning of that start, applied as a **mean-
preserving logistic offset** to each PA's baseline hazard (same "one extra CRN-keyed
draw, one fitted constant, marginal stays exactly calibrated" recipe as `SHOCK_SIGMA`).
Mean-preservation uses an EXACT numerical correction (Gauss-Hermite quadrature +
root-find via `solve_pre_noise_logit`), not the closed-form Zeger-Liang-Albert
approximation — that approximation was checked and found biased at low p, exactly where
most hook cells live (off by up to 30% relative at p=0.005).

A **constant** sigma, fit to the inning-4 target (sigma=1.40), overshot badly at k=5+:
it overcorrected the shelled-tail (undershooting real rates at k=5,6,7) and made the
OPPOSITE tail — P(still in at inning 7 | clean start) — worse, not better (62.3% vs.
28.5% real, further from truth than no frailty's 51.2%). This showed late-game hook
decisions are increasingly pitch-count/fatigue-ceiling-driven, not frailty-driven, so
frailty's influence must decay with inning, not stay flat. **Fix: `sigma(inning) =
HOOK_FRAILTY_SIGMA1 * HOOK_FRAILTY_DECAY**(inning-1)`, fit via n-weighted grid search
on 2023-24 (`HOOK_FRAILTY_SIGMA1=3.8`, `HOOK_FRAILTY_DECAY=0.65`)**, validated held-out
on 2025 across a full pre-registered grid (not just the fitted cell):

| k (inning) | real (5+ runs) | no frailty | const σ=1.40 | decaying σ |
|---|---|---|---|---|
| 3 | 13.8% | 20.8% | 16.6% | **13.9%** |
| 4 | 29.3% | 42.0% | 32.2% | **32.6%** |
| 5 | 62.3% | 71.3% | 53.8% | **61.5%** |
| 6 | 86.9% | 89.5% | 70.0% | **82.9%** |
| 7 | 94.1% | 96.4% | 78.5% | **93.4%** |

Decaying sigma beats both alternatives at every single k, on the 3+-runs grid too, and
holds roughly steady (mixed, no worse) on the opposite/clean-start tail through k=6.

**Known limitation, not fixed by any frailty shape (logged, not built)**: the opposite
tail's deepest point, P(still in at inning 7 | clean start throughout), stays
substantially miscalibrated under every variant tested (real 28.5% vs. ~51-62% modeled
regardless of frailty shape). This isn't a frailty-shape problem — it reflects a
genuinely missing feature (real pitch count / TTOP fatigue ceiling, which caps even
clean starts around innings 6-7 regardless of runs allowed) that this hazard model has
no signal for at all. Candidate future refinement: add pitch count as a feature. Not
built now.

**Also logged, not built**: part of a start's "frailty" is actually observable —
bullpen rest state (which relievers are unavailable that day), already computed by the
bullpen model. A future refinement could condition part of the offset on pen
availability instead of leaving it fully random, converting unexplained variance into
real signal on data already in hand. Not built now (scope discipline).

**Unresolved caveat, do not let sigma silently take credit for this**: the rejected
discrete-time logit hazard specification (kept only as a diagnostic, not used in
production) found the `margin` coefficient statistically null (p=0.93) in a smooth
linear form, despite margin being the single strongest driver in the bucketed cross-tab
table. This suggests margin's true effect is threshold-like or entangled with
runs-allowed (highly collinear in early innings), not a smooth linear gradient. The
cross-tab table retains margin as a discrete bucket, so this isn't a live bug — but
it's a real, unexplained anomaly in the underlying data-generating story.

**Status**: the hook-frailty mechanism is built and validated. Session closed here
deliberately — the wiring step (core per-half-inning loop, new CRN key allocation, the
flag-off byte-identity gate, `blowout.py` unification, all at once) is the single most
invasive edit of task #144, and this project's own history says core-loop edits made at
the tail end of long sessions are where reference-point mistakes live. The policy module
is validated, frozen, and documented — nothing to reconstruct next time.

**Pre-registered first move for the next session, before any wiring**: a diagnostic on
`HOOK_FRAILTY_SIGMA1=3.8` itself. That's a log-odds-scale parameter — a ±1σ draw at
inning 1 is an odds multiplier of e^±3.8 ≈ 45x either way, which makes early-inning hook
behavior nearly deterministic per start (each simulated start is essentially a
"quick-hook day" or an "untouchable day" in innings 1-2, converging to normal
heterogeneity by innings 4-5 as the decay bites). That may genuinely be what the data
demands, but a parameter this extreme is often absorbing something else — the
margin-null anomaly above is one candidate; sparse early-inning hazard cells leaning on
the tails is another; and **openers/bullpen-day starters** (near-deterministic 1-2-inning
"starts" by construction) are a third, concrete, nameable one if they're sitting in the
fit data unlabeled and masquerading as extreme frailty rather than a distinct
subpopulation. The check (cheap, ~10 minutes): plot the model's implied per-start P(exit
by inning 2) distribution under the fitted frailty — is it bimodal, and does that mass
concentrate on openers/bullpen-day starts specifically? If so, excluding or flagging them
before refitting may bring σ1 down to something more interpretable. This determines
whether 3.8 is a fact about manager behavior or an artifact of an unlabeled subpopulation,
and should run BEFORE trusting the fitted constant in the simulator.

**After that diagnostic**, staged wiring resumes in order: wire the frailty-corrected
hook model ALONE into `game_simulator.py` behind a flag (reliever selection still from
the pregame plan) -> byte-identity check with the flag off -> run the decomposition's
bullpen swap -> add tier selection -> re-run -> refit `SHOCK_SIGMA` down (pre-registered,
expected) -> final acceptance grid (Brier/CRPS primary, SU guardrail only, 2026 holdout
read once after freezing). Per §11.18/11.19. The two logged-not-fixed limitations (late-
inning pitch-count miss, margin-null anomaly) stay named suspects for interpreting the
acceptance test if the bullpen gap doesn't close all the way — not mysteries to
re-discover.

### 11.21 Task #144 step 2: the σ1 opener diagnostic (cleared) + the hook model wired
into game_simulator.py behind a flag, byte-identity confirmed

Per the pre-registered opening move from the prior session's close, before trusting
`HOOK_FRAILTY_SIGMA1=3.8` (a large log-odds parameter) in the simulator, two checks
ran against 2023-24 fit data:

1. **Are openers/bullpen-day starters contaminating the fit target?** Flagged
   (pitcher, season) combos with a repeatable early-exit signature (median exit
   inning <=2 across >=3 starts that season) as opener-like -- 35 such combos,
   1.93% of all starts. Checked their overlap with the frailty fit TARGET (starts
   reaching "inning 4 with 5+ runs allowed"): **0 of 1085 such starts came from a
   flagged opener-pattern pitcher.** Openers are not contaminating the target at all.
2. **Is the model's implied per-start P(exit by inning 2) distribution bimodal** (the
   "quick-hook day vs. untouchable day" worry)? No -- smoothly decaying, unimodal,
   ZERO starts above 40% implied probability across the whole 2023-24 fit sample
   (max bin 20-40%, only 0.62% of starts). The large log-odds sigma does NOT
   translate into unrealistic near-certain outcomes, because it's applied to
   realistically tiny baseline hazards (the mean-preserving transform compresses
   naturally at low p) -- confirming 3.8 is a fact about how compressed the
   marginal hazards are at this scale, not evidence of a hidden bimodal
   subpopulation or an opener-contamination artifact.

**Both checks cleared cleanly** -- proceeded to wiring.

**Wiring**: `src/models/hook_frailty.py` (NEW) holds the pure hook-frailty math
(bucket_inning/pa_count/runs_allowed/margin, hook_frailty_sigma, solve_pre_noise_logit,
apply_hook_frailty) split out of `bullpen_usage_policy.py` specifically so
`game_simulator.py` can import it without a circular import (bullpen_usage_policy.py
-> bullpen.py -> game_simulator.py for OUTCOMES) -- `bullpen_usage_policy.py`
re-imports from it for backward compatibility. `crn.py` gained
`DECISION_HOOK_FRAILTY=9`.

`GameSimulator.simulate_half_inning` gained `inning`/`hook_state`/
`pitching_score_entering`/`batting_score_entering` (all optional, None/0 default =
byte-identical to before). `hook_state` is a MUTABLE dict `simulate_game` builds and
threads through every half-inning a given starter pitches in a given trial:
`{"hook_table", "z" (this trial's frailty latent, drawn once), "next_reliever",
"hooked" (latches True permanently on the FIRST hook), "hook_inning", "pa_count",
"cum_runs_allowed"}`. Evaluated once per PA, BEFORE that PA is simulated (using
pre-PA exposure: this would-be Nth batter faced, runs allowed strictly before this
PA, current margin -- exactly matching how the fitted table's cells were defined).
If the hazard fires, the reliever swaps in STARTING WITH that exact batter (the
starter never faces them) -- this can happen MID-half-inning, which is why the swap
lives inside `simulate_half_inning` itself rather than only at `simulate_game`'s
existing between-inning boundary. thruorder_counts is cleared at the exact swap
point (a batter's times-through-order restarts against a new arm).

**Scope, deliberately bounded to "hook model alone"**: only the STARTER's own exit
point becomes state-conditioned; reliever IDENTITY and ORDER are completely
untouched -- once hooked, `_shift_bullpen_after_hook` (NEW module function) does a
ONE-TIME rebuild of the pregame `{inning: profile}` bullpen plan, reusing the EXACT
same reliever sequence `sample_bullpen_plan` already assigned (innings cutoff+1..N),
just shifted to start at the ACTUAL hook inning instead of the pregame-assumed
cutoff. All subsequent reliever-to-reliever transitions stay at the existing
inning-boundary granularity, unchanged -- only ONE mid-half-inning pitcher swap is
possible per side per game (starter -> first reliever). `simulate_game` syncs its
own `id()`-based pitcher-change bookkeeping (appearance_idx, pitcher-shock redraw)
immediately when a hook fires, specifically to prevent the EXISTING next-inning
change-detection from double-clearing thruorder_counts a reliever has already
legitimately started accumulating. One documented, deliberately-accepted
simplification: the task #137 pitcher-shock does NOT redraw immediately at the
swap (continues using the OLD pitcher's shock for the remainder of that one
half-inning only) -- it redraws naturally next inning via the existing mechanism;
threading extra CRN/appearance-index plumbing through `simulate_half_inning` for
this second-order interaction wasn't worth it given step 4 (SHOCK_SIGMA refit)
already expects to revisit this interaction anyway.

**Byte-identity gate**: ran 500 games across 3 configurations (CRN-keyed with
shock_sigma=0.40, plain-rng with shock_sigma=0.40, no-bullpen-dict with
shock_sigma=0) comparing the pre-change code (loaded from git HEAD under a distinct
module name) against the new code with every new param omitted -- **500/500 byte-
identical** (exact score AND exact full per-PA event log match) across all three.

**Sanity check** (not yet the real decomposition test, just a directional check
before trusting the mechanism further): with hook_context enabled, a deliberately
"bad" starter (elevated HR/hit rates) got pulled at mean inning 4.18 across 500
CRN-keyed trials, vs. mean inning 6.19 for a deliberately "good" starter -- the
correct direction and roughly the same ~2-inning gap magnitude as the real
trail_big-vs-lead_big split from task #144 step 1's own validation (4.45 vs. 6.30).

**Status**: task #144 step 2 (wire hook model alone, byte-identity check) is
COMPLETE. Remaining staged steps, in order: run the decomposition's bullpen swap
(re-run `validate_oracle_vs_predictive.py`'s ORACLE_BULLPEN-vs-FULL_PREDICTIVE
comparison with this mechanism live, requires computing real `cutoff`/`hook_table`
inputs for the decomposition's own bullpen-construction path) -> add tier selection
(the reliever-choice half of task #144 step 1, not yet wired) -> re-run -> refit
`SHOCK_SIGMA` down (pre-registered, expected to shrink once state-conditioned
variance is added) -> final acceptance grid (Brier/CRPS primary, SU guardrail only,
2026 holdout read once after freezing). Per §11.18/11.19/11.20.

### 11.22 Task #144 step 3: hook-model-alone wired into the decomposition's
predictive bullpen arm — a real, honest NULL (not a revert signal, a staging signal)

Ran the actual test: `validate_oracle_vs_predictive.py`'s FULL_PREDICTIVE arm
(bullpen source = predictive) with `hook_frailty_enabled` toggled, at full canonical
scale (n=2409 real games, K=50), restricted to **test_seasons={2025} only** --
genuinely held out from `HOOK_TABLE_FIT_SEASONS={2023,2024}` (a new module constant
in `validate_oracle_vs_predictive.py`, alongside `build_hook_table`, which fits the
same frozen policy task #145 already validated and reuses it here). `run_decomposition`
gained a `hook_frailty_enabled: bool = False` param that, when True and the bullpen
source is predictive, wires `home_hook_context`/`away_hook_context` into
`simulate_game` using the SAME `expected_innings`-derived cutoff `sample_bullpen_plan`
itself assumes (so the hook mechanism's "pregame cutoff" exactly matches what the
reliever sequence was actually built around).

**Result (paired bootstrap, `ab_significance.py`, n=2409, 10000 resamples)**:

| Comparison | SU delta | Brier delta | Verdict |
|---|---|---|---|
| hook_enabled vs. no_hook | -1.04pp, CI (-2.78, +0.75) | -0.0010, CI (-0.0035, +0.0015) | **NOISE** — CI includes zero on both |
| ORACLE_BULLPEN vs. hook_enabled | +4.28pp, CI (+2.03, +6.48) | -0.0076, CI (-0.0112, -0.0040) | **REAL** — gap still fully present |

Hook-model-alone (reliever IDENTITY/ORDER still entirely pregame-fixed, unchanged
from `sample_bullpen_plan`) produces **no measurable full-stack SU/Brier
improvement** at this scope, and the gap against the ORACLE_BULLPEN ceiling is
essentially the SAME size as the original task #143 decomposition found
(3.95pp/0.0079) -- timing alone has not closed it.

**This is not a revert signal — it's exactly the staging signal the plan was built
to produce.** Task #144's own step-1 spec fit TWO components (starter-hook timing
AND reliever-tier selection) precisely because either one alone was a hypothesis,
not a certainty, about where the oracle bullpen's advantage lives. Wiring hook-alone
first and testing it in isolation (rather than wiring both at once) was the
isolate-one-factor discipline this project has used everywhere else (GB/FB
multiplier, bat-tracking, pitcher-stuff K-multiplier, etc.) -- and it just did its
job: it tells us the accuracy benefit does NOT come from timing alone. Plausible
reasons, none yet distinguished: (a) reliever CHOICE (leverage arm vs. mop-up,
matching the real 82%-vs-53% split task #144 step 1 already validated) may be the
dominant real driver, with timing a necessary but insufficient piece; (b) the
`SHOCK_SIGMA` interaction (still frozen at 0.40, pre-registered for refit-down in
step 4) may be partially masking a real but small gain; (c) shifting WHEN a fixed
reliever pool enters may simply not move full-game W/L or margin outcomes much if
the pool's aggregate talent is what actually matters, not the exact inning boundary.

**Logged to the metrics ledger** (both arms, full config_flags + this finding in
the notes field, matching the project's self-documenting-row convention).

**Next step, per the already-staged plan (not paused for reconsideration — this
result is what continuing to step 4 was always contingent on)**: wire reliever TIER
selection (the other half of task #144 step 1, `fit_tier_policy`'s validated
leverage/middle/mopup shares and closer identification, not yet wired into
`sample_bullpen_plan`/`simulate_game` at all) and re-run this SAME comparison with
BOTH mechanisms live, to see whether tier selection is what actually closes the gap,
either alone or combined with hook timing.

### 11.23 Task #144 step 4: reliever tier selection wired + the full 2×2 factorial —
neither mechanism, alone or combined, closes the gap (a real, humbling null)

**Wiring**: `src/models/tier_selection.py` (NEW, dependency-free, same
circular-import reasoning as `hook_frailty.py`) holds
`tier_label_from_roster_weights` (derives each reliever's tier from their OWN
walk-forward roster weight — top third by trailing usage+rest weight =
"leverage", middle third = "middle", bottom third = "mopup" — rather than
`build_reliever_tier_log`'s season-long tercile, which that function's own
docstring already flags as not walk-forward-safe) and
`sample_tier_conditioned_reliever` (draws one reliever, conditioned on the
CURRENT trial's margin/save-situation via the validated policy table,
reweighting WITHIN the pregame availability-weighted roster — never overriding
rest-day availability). `crn.py` gained `DECISION_TIER_SELECT=10`.
`bullpen_usage_policy.build_tier_policy_dicts` fits the same frozen policy
(`fit_tier_policy`, 2023-2024) into plain dicts for O(1) simulator lookup —
one documented approximation: the fitted `tier_by_margin` probabilities used
season-long tier labels, while simulation-time tier ASSIGNMENT uses the
walk-forward roster-weight ranking — related but not identical conventions,
flagged rather than reconciled (would require refitting against
walk-forward-derived labels directly).

`GameSimulator.simulate_game`/`simulate_half_inning` gained
`home_tier_context`/`away_tier_context` (optional, None-default). When given,
EVERY post-starter reliever selection — not just the first — is drawn live;
combines cleanly with `hook_context` (the reliever who takes over at the hook
moment is now tier-drawn instead of a pregame-fixed pick) or works standalone
(the starter still exits at the fixed pregame cutoff, but every reliever from
there on is tier-drawn). This is also where the blowout/position-player
unification from the original spec went live: once a side enters blowout mode,
a tier-conditioned draw (naturally mopup-heavy at extreme margins) is preferred
over `blowout_pitcher_profile` for as long as the roster pool has anyone left —
a position player is used to CONSERVE the bullpen, not the instant the blowout
threshold is crossed. Byte-identity re-confirmed (500/500 games across 3
configs) after this wiring; the hook-only sanity check (bad-starter mean hook
inning 4.18 vs good-starter 6.19) also re-confirmed unchanged. A standalone
tier-selection unit test confirmed the mechanism reproduces the real validated
leverage/mopup shares by margin closely (e.g. lead_big: 70.0% leverage drawn
vs. 69.5% real; trail_big: 52.2% vs. 53.1% real).

**The full 2×2 factorial (paired bootstrap, n=2409, K=50, test_seasons={2025}
only — genuinely held out from both fit periods)**:

| Config | SU | Brier | vs. no-mechanism baseline |
|---|---|---|---|
| no_hook_no_tier (baseline) | 54.21% | 0.2485 | — |
| tier_only | 53.67% | 0.2485 | NOISE (SU CI (-2.16,+1.08)pp) |
| hook_only | 53.18% | 0.2475 | NOISE (SU CI (-2.78,+0.75)pp) |
| hook_and_tier | 52.68% | 0.2496 | NOISE (SU CI (-3.36,+0.33)pp) |

**Every pairwise comparison among the four configs is NOISE** — tier_only,
hook_only, and hook_and_tier are statistically indistinguishable from the
baseline AND from each other (hook_and_tier vs. tier_only: NOISE; hook_and_tier
vs. hook_only: NOISE). **This contradicts the pre-registered expectation** that
tier-only would carry most of the recoverable gap — it does not measurably
close it, and combining both mechanisms does not improve on either alone.

The gap against the ORACLE_BULLPEN ceiling remains REAL and essentially
UNCHANGED in magnitude across every predictive configuration: tier_only
SU+3.78pp/Brier-0.0087 (both REAL, CI excludes zero); hook_and_tier
SU+4.77pp/Brier-0.0097 (both REAL) — the same order of magnitude as the
original task #143 finding (+3.95pp/-0.0079). **Neither mechanism closes any
measurable fraction of the oracle-vs-predictive bullpen gap.**

**Three honest, undistinguished explanations, logged so a future session
doesn't have to rediscover them**:
1. The real effect may simply be smaller than n=2409/K=50 can resolve (SU CI
   width here is ~2.7-3.7pp — a true ~1pp effect is unmeasurable by
   construction, the same "K=50 noise floor" lesson this project has hit
   before).
2. The endogeneity ceiling task #143's own verification pass already
   identified: the REAL oracle bullpen sequence encodes the game's ACTUAL
   script (a manager who KNOWS the score pulls accordingly) — information no
   pregame-blind, probabilistic mechanism can fully reproduce regardless of
   sophistication. Some of the ~4pp gap may simply be structurally
   unrecoverable, not a modeling shortfall.
3. Game-level SU/Brier may not be the right lens for a bullpen-USAGE
   mechanism at all — its value may show up in dispersion (std(z)) or
   prop-level accuracy (innings pitched, saves, holds) rather than in final
   win probability. Neither has been checked yet in this session.

**Both mechanisms remain individually validated** as reproducing REAL
historical usage patterns on their own terms (task #145's hook-timing
calibration; this section's tier-share calibration) — this is a full-stack
downstream-accuracy null, not evidence the underlying policy tables are wrong.
Logged to the metrics ledger (both new arms, full notes). **Neither mechanism
is deployed**; both stay dormant, off-by-default, byte-identity-gated
infrastructure — matching this project's standing "keep only on genuine
net-positive, revert cleanly otherwise" discipline (the same fate as isotonic
calibration, the arsenal-tercile adjustment, and the whiff-rate factor
earlier this session), pending the dispersion/prop-calibration check the
original plan staged as the next step regardless of which SU/Brier
configuration "won" (none did).

**Post-hoc reasoning (reviewer, same day) on WHY tier-only carried nothing
measurable, converted to actual runs rather than left as a bare null**: the
82%-vs-53% leverage-share split by margin is a real, validated behavioral
pattern, but its RUN-VALUE consequence at game scale is small — the true-talent
ERA gap between a team's leverage arm and its middle reliever is roughly
0.3-0.5 runs, deployed over one or two innings, and only in the subset of
trials where the state-conditioned draw actually diverges from what the
usage-weighted pregame sampler would have picked anyway (recent-usage weight
is ALREADY leverage-correlated, so the two often agree). Net expected effect:
a few hundredths of a run per game — invisible at this protocol's ±3pp SU CIs,
plausibly invisible at any realistic CI. This reframes the earlier task #143
close-game-slice "~2pp recoverable" estimate too: that slice's oracle sequences
still encode true bullpen AVAILABILITY (who warmed up yesterday, who's
nursing something, who the manager privately trusts that week that particular
week) — information, not behavior, and no policy fit on PUBLIC usage patterns
can recover it regardless of sophistication. **Honest revision: the
pregame-recoverable bullpen headroom is probably substantially smaller than
the earlier ~2pp estimate, and this factorial is the measurement that
established that, not a failure to find it.**

**One line for the ledger, so a future session doesn't re-derive this whole
arc from scratch**: *the policy tables (hook-timing, tier-share) remain valid
descriptions of real managerial behavior; the null is about downstream
game-level value, and the remaining oracle-vs-predictive bullpen gap is now
believed to be predominantly AVAILABILITY information plus SCRIPT
endogeneity — neither recoverable from a pregame-knowable, behavior-only
policy, however well-fit.*

**Next step, pre-registered and falsifiable (not yet run)**: the hook
mechanism makes a direct, testable prediction about a quantity `props.py`
actually sells — the distribution of starter outs/innings pitched. Compare
simulated starter-outs distributions (baseline vs. hook-enabled) against real
2025: calibration of P(starter completes 5), P(exits by inning 4), the full
histogram, not just the game-level aggregate. Same logic for tier selection
against reliever-inning attribution (feeds reliever K props, and any future
saves/holds props). This is the lens where these two mechanisms were actually
built to show value — within-game usage realism is a PROP-level quantity, not
a win-probability one, so a game-level SU/Brier null doesn't settle whether
they're worth anything. **If starter-outs calibration improves materially**:
keep the hook mechanism deployed for the props path specifically, documented
as a distribution-realism fix with NO game-level accuracy claim — the exact
`SHOCK_SIGMA` precedent (a variance/distribution-shape correction, not a
point-metric win). **If it doesn't improve even that**: archive both
mechanisms cleanly — the null is then total, not partial, and there's no
residual mystery left to chase.

## 11.24 Calendar-driven hardening (2026-07-24): trade-deadline handling +
postseason correctness fixes, prioritized by real dates rather than model quality

Reviewer-prompted: the model was built/validated on full regular seasons; the
back half of the calendar breaks specific assumptions, some rules-certain
(not hypothetical). Ordered by real deadline; two items built and tested this
session, the rest scoped and tracked for later.

### Built: trade-deadline transaction/closer handling (this week's deadline)

**The gap, confirmed by direct investigation**: unlike a starter or batter —
both genuinely live via the MLB Stats API's probable-pitcher/lineup fields —
a RELIEVER's team affiliation has NO live signal anywhere in this pipeline.
`build_team_bullpen_roster`/`identify_closer` (`bullpen.py`) infer team
purely from historical PA rows. Confirmed directly: a pitcher traded 5 days
ago is either invisible (zero ingested relief appearances for his new team
yet) or badly understated (a handful of new-team appearances vs. his real
established role); a newly-traded CLOSER specifically loses to his new
team's incumbent (whose appearance count is still decaying from BEFORE the
trade) for weeks.

**Fix**: `src/ingest/fetch.py` gained `fetch_transactions(start_date,
end_date)` — the MLB Stats API's free `/transactions` endpoint, filtered to
real trades (`typeCode=="TR"`) with an actual player attached. Confirmed
directly against live data before trusting the dedup logic: multi-player
trades reuse the SAME transaction id across each distinct player (a real
3-for-2 trade produced 3 rows, one per player, all sharing one id) — an
earlier draft deduped by id alone and would have silently dropped 2 of 3
players; the shipped version dedupes on `(transaction_id, pitcher_id)`.

`src/models/bullpen.py` gained `build_traded_pitcher_overrides` (turns real
trade rows into `{pitcher_id: {"new_team", "effective_date"}}`, keeping only
the most recent trade per pitcher) and `_apply_traded_overrides` (relabels a
traded pitcher's ENTIRE relief/closer history — both pre- and post-trade
rows — to his new team once past `effective_date`, seeding his usage
weight/closer-candidacy from his real established role instead of starting
cold). Deliberately does not expire the relabeling on its own timer — the
CALLER's own existing trailing window (`ROSTER_WINDOW_DAYS=45`/
`CLOSER_WINDOW_DAYS=90`) already bounds how far this can matter, and well
past that window his real new-team appearances (if he's still there) simply
dominate on their own. `build_team_bullpen_roster`/`identify_closer` gained
`traded_overrides: dict | None = None` (byte-identical when omitted, checked
directly). `props.py`/`generate_daily_props.py` wire this through `ctx`,
fetching a trailing `TRADE_OVERRIDE_LOOKBACK_DAYS=60`-day transaction window
each run (wrapped in a try/except — a network hiccup here must not break the
whole daily run, same resilience convention as the RotoWire fetch).

**Confirmed on real data**: without the override, a real 2025 reliever with
80 relief appearances for his real team is correctly absent from a
hypothetical new team's roster; with a synthetic trade override, he
correctly appears with a real, non-zero weight. Same confirmed for
`identify_closer`. Byte-identity confirmed for the `traded_overrides=None`
default path.

**Diagnostic run (reviewer-requested): did the EXISTING, unfixed predictive
bullpen show a measurable post-deadline dip in 2025?** Reused the
already-saved `hook_decomp_FULL_PREDICTIVE_no_hook_2025.parquet` (task #147,
n=2409, no new compute), split by real calendar month. Result: **no
detectable dip** — July SU=52.99%/Brier=0.2510 vs. August SU=52.97%/
Brier=0.2479 (Brier actually slightly better in August; two-sample
t-test p=0.996 for SU, p=0.648 for Brier, both indistinguishable from noise).
Honest reading: this doesn't undermine the fix (the underlying mechanism gap
is directly, mechanistically confirmed above on real data, independent of
this aggregate check) — it means a month-level aggregate isn't sensitive
enough to detect an effect concentrated in the handful of games each week
where a JUST-traded reliever specifically pitches, diluted by every other
game in the month having nothing to do with any trade. Same "aggregate
metric too coarse to see a real, narrowly-concentrated effect" pattern as
task #144's tier-selection null.

### Built: postseason auto-runner + blowout-substitution correctness fix

**The bug (rules-certain, same class as the original zombie-runner fix)**:
the real MLB automatic extra-innings runner rule does NOT apply in the
postseason — playoff extra innings start bases-empty, same as any other
inning — but `simulate_game` set `auto_runner=inning >= 10` unconditionally.
Confirmed via direct investigation: zero postseason/game_type awareness
existed anywhere in `src/` before this fix. Schedule data's actual
`game_type` values (confirmed directly, 2023 season): `R`=regular season
(2430), `S`=spring training (467), `E`=exhibition (24), `D`=division series
(14), `L`=league championship (14), `F`=wild card (8), `W`=world series (5),
`A`=all-star (1) — matches the standard MLB Stats API convention, plus three
categories (`S`/`E`/`A`) worth excluding alongside true postseason handling.

**Fix**: `GameSimulator.simulate_game` gained `postseason: bool = False`.
When True: `auto_runner` is forced off regardless of inning for both
half-innings; the blowout/position-player-pitching trigger (`home_in_blowout`/
`away_in_blowout`) can never fire either — real playoff rosters/incentives
mean a team essentially never concedes a game to save a reliever for
tomorrow (there is no tomorrow if you lose). Default False is byte-for-byte
identical to before this parameter existed (re-confirmed: 500/500 games
across 3 configs). Also confirmed directly that the underlying data pipeline
ALREADY fully excludes non-regular-season rows at the PA-table-build layer
(`build_pa_table.py` filters `game_type=="R"`) and redundantly at every
validator's own schedule read — postseason games cannot currently
contaminate any backtest/validation set, they are simply unsupported for
live prediction (`generate_daily_props.py` filters to `game_type=="R"` when
selecting games, so postseason games get ZERO props today, not wrong ones).

**Wired end-to-end the same day (reviewer-prompted re-prioritization: "the
pipeline turns on in October" was too important a plumbing gap to leave
until the postseason itself)**: `generate_daily_props.py`'s `games_for_date`
now selects `game_type in {"R","F","D","L","W"}` (a new
`GAME_TYPES_TO_PREDICT` constant, deliberately excluding `S`/`E`/`A`), and
`postseason = row.game_type in POSTSEASON_GAME_TYPES` is threaded through
`generate_game_props` → `simulate_game` for each game. Default path
(`postseason=False`, every existing caller) untouched.

**Dress rehearsal (same day, real data, zero new ingestion)**: ran the ACTUAL
`build_pregame_context`/`generate_game_props` pipeline against all 47 real
2025 postseason games (Wild Card through World Series, real posted lineups
and probable pitchers, already fully cached — `lineups_2025.parquet` covers
all 47 games since the whole season was bulk-ingested) with `postseason=True`,
n_trials=20. **45/47 succeeded outright**, producing sensible win
probabilities across the full bracket (e.g. LAD@TOR World Series games in
the 0.45-0.65 range). **The 2 failures are a REAL, concrete instance of task
#149** (not hypothetical): `ValueError: no pregame snapshot for player id
800050 as of 2025-10-01/10-02 (likely an MLB debut with zero PA history)` —
a player who evidently debuted mid-postseason (a September call-up pressed
into playoff action) and has no cached history at all. The pipeline's
existing `except ValueError` catch in the daily-props loop means this fails
SAFELY today (skips that one game, doesn't crash the whole run) — but this
dress rehearsal is exactly why task #149 (a real player, a real date, not a
hypothetical edge case) is worth resolving properly before the real
postseason, not left as "probably fine."

**Status: task #150's plumbing slice is DONE, not just scoped.** The
remaining piece of #150 (nothing left, folded into this) is complete; what's
left is task #151 (the bigger regime-change modeling, still genuinely
blocked on a postseason-inclusive PA table) and task #149 (debut-player
handling, now with a real reproduction case attached).

### Built, same day: task #149's generic-debut-profile fallback (the last
known way the pipeline produces nothing on a playoff night)

The reproduction case from the dress rehearsal (player id 800050) turned out
to be a BATTER, not a pitcher as first guessed — a September call-up
outfielder who started two 2025 Wild Card games (batting 5th/7th for CLE)
with zero cached MLB history at all. Not a random missing player: exactly
the kind of injury-replacement/emergency call-up a deep October run
surfaces, on the highest-visibility slates of the year.

**Fix**: `props.py` gained `build_generic_debut_profile(pa, player_col,
season, league_rates_this_season)` — reuses `true_talent.build_debut_rate`
(task #119's already-validated, walk-forward-safe empirical debut-cohort
rate per outcome, confirmed BELOW league average) rather than inventing a
new quartile heuristic. This is a narrower, previously-untested reuse of
that same validated data: task #119's `use_debut_prior` flag (applying the
debut rate as the ONGOING cold-start prior inside the regular Marcel blend
for every player) was tested full-stack and found NULL — this is a
LAST-RESORT fallback only for a player who would otherwise crash the whole
game's prop generation, a different and much narrower claim that was never
tested and isn't contradicted by that earlier null result. `effective_n=0`
for every outcome (an already-handled, documented edge case in
`sample_posterior_rate` — "a fully cold-started player with zero weight").
Precomputed once per (batter/pitcher, season) in `build_pregame_context`
(cheap, and every debut player in a season needs the identical fallback
regardless of who they are) and wired as the fallback path in
`batter_profile`/`pitcher_profile` inside `generate_game_props`, replacing
their previous `return None` (which fed into the ValueError). The original
missing-snapshot ValueError check stays in place as a defense-in-depth
backstop (e.g. if a season isn't in the precomputed dict at all), not the
primary path anymore. Flagged, not hidden: `generate_game_props` now returns
`debut_fallback_pids`, printed by `generate_daily_props.py` and stamped as a
`used_debut_fallback` column in the saved output — same transparency
convention as the existing lineup-source flags.

**Confirmed on the exact real reproduction case** (player 800050, game_pk
813071): previously raised `ValueError`, now generates successfully
(`home_win_prob=0.60`) with `debut_fallback_pids=[800050]` correctly
recorded. **Re-ran the full 47-game postseason dress rehearsal: 47/47 now
succeed** (up from 45/47) — the pipeline produces a prediction for every
real 2025 postseason game with zero crashes.

**Task #149 is closed.**

**Still pending (task #151 only)**:
1. Task #151, October part 2 (the regime change): playoff bullpen usage is a different
   distribution than anything this project's models were fit on (starters
   hooked much earlier, aces in relief, top arms on short rest, the bottom
   of the pen unused) — exactly the miss task #144's hook-frailty/tier-
   selection mechanisms were built to express, which nulled out for the
   REGULAR season specifically because regular-season usage is already
   well-approximated by the pregame plan (see sec 11.23) -- October is where
   usage deviates hardest from that plan. Scoping: fit playoff-specific
   policy OFFSETS from 2022-2025 postseason data (small n -- borrow strength
   from the regular-season tables, don't refit from scratch), flag on for
   postseason games only, judge on prop-level sanity (starter-outs
   distributions vs. real playoff starts) rather than SU -- full-stack
   validation at playoff sample sizes is not possible, this ships on
   mechanism-correctness grounds like the auto-runner fix. REQUIRES first
   building a postseason-inclusive PA table/ingestion path, since the
   current one structurally excludes ALL non-regular-season rows at build
   time (confirmed above) -- there is currently zero postseason PA data to
   fit from.
2. Fold in the known postseason scoring-environment shift (~0.25 runs/game
   lower than regular season -- pitching concentration, cold) as a small
   October adjustment, validated against 2022-2025 postseason totals (same
   prerequisite as #1: needs postseason PA/schedule data not currently kept).
3. Tag postseason games explicitly in the metrics ledger once any of the
   above starts producing predictions for them, so a future "why did
   October's Brier spike" question has a one-word answer instead of
   silently blending into monthly reads. Note: `daily_props_*.parquet`
   outputs already carry a `postseason` column per game as of this session's
   fix, so this is now mostly a matter of reading that column into the
   ledger's own row rather than adding new plumbing.

## 11.25 Task #144 closed on the lens it was actually built for: the
starter-outs calibration check (hook KEPT props-path-only), and the
weather-CRPS re-read (closed null) — two pre-registered reads, one session
(2026-07-24)

Sec 11.23 pre-registered a decisive next step and an exact decision rule
before doing any more work on task #144: sec 11.23's game-level SU/Brier
null never actually tested the metric the hook/tier mechanisms were BUILT to
move — the starter-outs distribution itself. Ran that check, plus a
zero-new-compute weather-CRPS re-read "same sitting," per the pre-registered
spec.

### Read 1: starter-outs calibration — hook mechanism WINS, materially and
in the exact tail it claims to fix

**Method**: for 794 real, complete-lineup 2025 regular-season games, drove
the actual two-sided PA-by-PA Monte Carlo (real lineups both sides, real
starters, K=20 trials/game) with the HOME starter under `hook_context` (task
#144/#145's frailty-corrected exit mechanism) vs. BASELINE (today's
deterministic pregame cutoff, `round(expected_innings)` — zero within-game
variance by construction). Away side kept its real starter fixed the whole
game (isolates the home starter's own hook-timing measurement without
letting away-bullpen depth confound it). Compared both against the REAL
recorded exit inning, on the full outs histogram (CRPS) plus the three
quantities props actually prices — P(completes 5), P(exits by the 4th),
P(completes 7) — segmented ONCE by pregame run-value tercile
(`run_value_screen.run_value` on the starter's own pregame rates), since the
hook mechanism's whole claim is about the shelled tail.

**Result — decisive, not marginal**:

| | hook-enabled | baseline (deterministic) |
|---|---|---|
| CRPS(starter outs), aggregate | **0.747**, 95% CI [0.696,0.799] | 0.908, 95% CI [0.845,0.971] |
| delta (hook − baseline) | **−0.161, 95% CI [−0.203,−0.120] — CI excludes zero** | |

CRPS improvement holds in **every** run-value tercile, largest in the
worst-pitchers (shelled-tail) tercile exactly as designed: −0.227 (CI
[−0.299,−0.155]), vs. −0.140 (mid) and −0.118 (best-pitchers).

The three props-priced quantities (aggregate, real vs. baseline-error vs.
hook-error):

| metric | real | baseline (err) | hook (err) |
|---|---|---|---|
| P(completes 5) | 0.835 | 0.982 (**0.147**) | 0.848 (**0.013**) |
| P(exits by 4th) | 0.076 | 0.009 (**0.067**) | 0.065 (**0.011**) |
| P(completes 7) | 0.183 | 0.000 (**0.183**) | 0.289 (**0.106**) |

The deterministic baseline has essentially ZERO early-exit tail by
construction (a fixed cutoff can't produce a start that ends early) and
never produces a start past 6 innings at all (histogram: 0% at innings
7/8/9) — the exact shelled-start and workhorse-start scenarios a prop
bettor actually cares about were entirely unmodeled before this. In the
worst-pitchers tercile specifically, P(exits by 4th) error drops from 0.068
(baseline) to 0.003 (hook) — a ~23x reduction in the single quantity this
mechanism was purpose-built to fix. Hook does somewhat overshoot P(completes
7) in aggregate (0.289 vs. real 0.183), but this remains a large net
improvement over baseline's flat 0.000.

**Decision, per the pre-registered rule**: hook-enabled materially improves
starter-outs calibration, especially the early-exit tail → **shipped to the
props path ONLY**. `src/models/props.py` gained `build_hook_table` (same
frozen 2023-2024-fit policy as task #145/§11.18-11.20, reused verbatim) and
`hook_table` in `build_pregame_context`'s returned ctx; `generate_game_props`
now builds `home_hook_context`/`away_hook_context` (cutoff = the same
`max(1, round(expected_innings))` `sample_bullpen_plan` already uses) and
passes them into every trial's `simulate_game` call. **Documented, per the
exact `SHOCK_SIGMA` precedent, as a distribution-realism fix with NO
game-level win-probability claim** — that lens stays a confirmed null (sec
11.23) and the mechanism stays OFF in `validate_game_simulator.py`/
`validate_oracle_vs_predictive.py`'s own oracle backtest, which remains
byte-identical to before this session.

**A real correctness bug this wiring would otherwise have introduced, caught
and fixed before shipping**: `props.py`'s pitcher-level prop attribution
(which real pitcher_id gets credit for a given simulated PA — feeds every
individual-pitcher K/BB/hits/runs prop) worked by checking whether an
inning fell at-or-before the STATIC pregame cutoff, then falling back to a
pregame-built `{inning: pitcher_id}` id_plan otherwise. With hook_context
live, the starter's real exit inning can now legitimately differ from that
static cutoff (earlier OR later) every trial — silently misattributing
relief innings to the wrong reliever (or crediting the starter for innings a
reliever actually threw, and vice versa) in precisely the trials where the
hook fired away from the pregame assumption, which is the whole point of
having it. Fixed by adding an optional `hook_result: dict | None = None`
out-param to `GameSimulator.simulate_game` (populated in-place with the
actual `home_hook_inning`/`away_hook_inning`, mirroring how `events` already
works) so `props.py` can invert `_shift_bullpen_after_hook`'s own shift
formula (`orig_inning = pregame_cutoff + (real_inning − real_hook_inning)`)
and look up the CORRECT planned id_plan slot for each real inning — the
exact same shift the simulator already silently applies to `home_bullpen`/
`away_bullpen` itself, just applied to the caller's separate id_plan too.
Confirmed via a 5-game smoke test (`generate_game_props` end-to-end, no
crashes) that per-pitcher attribution differentiates cleanly between
starters (~24-26 mean batters faced, matching real 6-inning-ish workloads)
and single/multi-inning relievers (~3-9 each) with no obviously duplicated
or dropped innings.

**Tier selection (the other half of task #144 step 4) is explicitly NOT
deployed by this decision.** Its own specific claim — reliever-inning
attribution / feeding reliever K and future saves/holds props — was never
the metric tested this session (this read was scoped to the starter-outs
question only, per the pre-registered spec). It stays dormant, validated
on its own behavioral terms (§11.23's leverage/mopup share calibration), off
by default, pending that separate check in a future session. §11.23's null
is upgraded from "unresolved lens question" to **partially closed**: closed
and REVERSED (real win, not a null) on the starter-outs lens specifically;
still open on the reliever-attribution lens.

### Read 2: weather-CRPS re-read — the null holds, book closed

**Method**: zero new compute. Re-scored the ALREADY-SAVED
`oracle_vs_predictive_{FULL_PREDICTIVE,ORACLE_WEATHER}.parquet` game-level
summaries (n=7229, task #143's original decomposition) on CRPS(total runs)
via the closed-form Gaussian CRPS formula (each arm's per-game mean/std
defines a Normal forecast for home+away runs) instead of SU/Brier — the
metric-appropriate lens for a distributional/variance-shape question like
weather uncertainty, where SU/Brier was always a weak fit. Segmented by a
pre-registered "forecast-horizon proxy": how climatologically surprising
each game's REAL recorded weather bucket was for its park/month (P(bucket |
that park's own real history that month), pooled across all cached seasons,
via the already-existing `weather_forecast.climatological_bucket_distribution`)
— low probability = the real weather deviated most from what a
climatology-only predictive arm assumes, the scenario where that arm is
most disadvantaged relative to the oracle (real-recorded) arm.

**Result**: CRPS(total) delta (predictive − oracle) = +0.0040, 95% CI
[−0.0002, +0.0081] — **CI includes zero, null**, and it holds in the
highest-deviation tercile too (mean climatological probability 0.043 — genuinely
rare weather for that park/month): delta +0.0050, 95% CI [−0.0016, +0.0117],
still including zero. Mid and low-deviation terciles: +0.0011 and +0.0058
respectively, both null.

**Decision, per the pre-registered rule**: the null holds even in the
high-deviation slice → **book closed on weather**. The 48-hour
point-forecast-narrowing fix (flagged in task #143's own weather-scoping
note as a plausible but untested improvement) is NOT justified by this
result — climatology alone already matches real-recorded weather's CRPS
contribution closely enough, across the full range of how unusual that
weather was, that a live forecast API's additional narrowing has no
demonstrated room to help. No code changed by this read; it is a
documentation-only closure of an open question.

**One line for the ledger**: *starter-outs calibration is a genuine,
material win for hook-frailty (props-path-only, no game-level claim);
weather-CRPS is a clean, tail-inclusive null — the 48-hour forecast fix is
not worth building.*

## 11.26 Putting formal statistical proof behind four open questions
(2026-07-25, prompted by a user request to verify prior findings with real
significance tests, not just point estimates)

Four separate claims from this session got re-derived with bootstrap CIs (or,
for one, a proper OLS significance test) instead of point estimates alone —
in two cases this materially changed the honest confidence behind a finding
already reported.

**1. Prop calibration slopes — most "overconfidence" isn't statistically
provable at this sample size.** Bootstrapped (10,000 resamples) the
calibration slope and the Brier-vs-naive gain for all 6 props (reusing the
already-saved calibration parquets, zero new compute):

| prop | slope (95% CI) | slope≠1 proven? | Brier gain (95% CI) | gain>0 proven? |
|---|---|---|---|---|
| p_1plus_hit | 0.80 (0.53, 1.05) | **no** | 0.0028 (0.0003, 0.0054) | yes |
| p_2plus_hits | 1.02 (0.49, 1.55) | no | 0.0009 (−0.0001, 0.0018) | **no** |
| p_1plus_hr | 0.87 (0.61, 1.12) | **no** | 0.0019 (0.0006, 0.0033) | yes |
| p_1plus_bb | 1.02 (0.77, 1.27) | no | 0.0049 (0.0025, 0.0073) | yes |
| p_1plus_rbi | **0.60 (0.32, 0.88)** | **YES** | 0.0007 (−0.0015, 0.0028) | no |
| p_6plus_k | 0.90 (0.78, 1.01) | no (barely) | 0.0338 (0.0229, 0.0449) | yes |

The earlier point-estimate framing ("p_1plus_hr shows real, moderate
overconfidence, slope 0.78") **overstated confidence in a single number** —
with proper resampling uncertainty, its slope's 95% CI comfortably includes
1.0 (perfect calibration), same as `p_1plus_hit`. Only `p_1plus_rbi`'s slope
CI actually excludes 1.0 — and that is exactly the one prop whose 5-split
stability test passed 5/5 and got deployed (§8.3/§11.25 addendum above). This
is a clean, convergent confirmation across two independent methods (bootstrap
CI on slope vs. cross-validated correction stability) landing on the same
prop. Every prop's Brier still beats naive on point estimate, but only 4 of 6
prove that improvement is real (excludes zero) at this sample size —
`p_2plus_hits` (already corrected) and `p_1plus_rbi` (pre-correction figures
shown here) don't clear that bar individually, though `p_1plus_rbi`'s slope
proof is the stronger and more decision-relevant signal for that prop.

**2. Pull-tercile × wind interaction — directionally consistent with the
original claim, but NOT statistically significant.** Re-derived as a
difference-in-differences with a bootstrap CI (not trusting the point
percentages in `spray.py`'s own docstring): lefty batters only, real terciles
(bottom/top third of season pull rate, excluding the middle third — a fair
test of the ORIGINAL low-vs-high contrast, not a diluted median split), real
`Out To RF`/`Out To LF` wind games, n=30,253 PAs, 2023-2025 pooled.
High-pull gap (pull-side − opposite-side HR rate) = +0.16pp; low-pull gap =
−0.20pp; difference-in-differences = **+0.36pp, 95% CI [−0.47pp, +1.20pp] —
includes zero.** Same direction and roughly the same order of magnitude as
the original finding, but not provable at this sample size. This matches
(rather than contradicts) task #35's own original framing as a "small, mixed
result" — the rigorous test confirms that hedge was the right call, not an
understatement.

**3. Batter-vs-pitcher-arsenal skill — real within a season, but NOT stable
year-over-year, which is exactly why the walk-forward rejection (task #82)
was correct.** Built real pitcher-season breaking-ball rate directly from
raw pitch-level Statcast data (`pitch_groups.pitch_group`, ≥200 pitches/
season), tercile-split pitchers, and tested batters' K-rate specifically
against high-breaking-tercile pitchers (≥30 PA) two ways:
  - **Split-half reliability** (same-season, two random halves of the
    high-breaking PAs, "excess" measured against a baseline computed from
    a fully DISJOINT set of low/mid-tercile PAs to avoid a mechanical
    overlap artifact — an earlier draft of this exact test had that bug,
    caught before reporting it): r=0.17, 95% CI **[0.11, 0.23] — excludes
    zero, a real within-season signal.**
  - **Year-over-year stability** (season S's excess vs. season S+1's excess
    for the same batter, n=657 batter-year-pairs, 2023→2024 and 2024→2025):
    r=−0.04, 95% CI **[−0.12, +0.04] — includes zero, NOT stable.**
  This is a genuinely useful, nuanced result: batters really do show
  consistent in-season variation in how they hit high-breaking-ball
  pitchers specifically (not pure noise, confirmed statistically) — but that
  variation doesn't persist from one season to the next, meaning it's more
  consistent with "ran hot/cold against whichever specific pitchers this
  year's schedule happened to bring" than a genuine, individual, predictable
  skill. This is precisely the distinction a walk-forward-only project
  should care about, and it explains — rather than just restates — why task
  #82's rejected predictive adjustment was the right call.

**4. Park factor × pull-tercile — reconfirmed null, now with a bootstrap CI
alongside the OLS p-value.** The OLS test in §11.4's new row (p=0.974) is
already a formal significance test; added a 10,000-resample bootstrap CI on
the same interaction coefficient for consistency with the other three
questions here: **[−0.033, +0.032], includes zero.** Same conclusion by two
independent methods.

**Takeaway across all four**: this round of re-verification changed the
CONFIDENCE behind two findings (prop calibration, arsenal skill) without
changing any deployment decision, and confirmed two nulls (pull×wind,
park×pull) were correctly hedged the first time. No code changed by this
section — pure statistical verification of existing and newly-screened
claims.

## 11.27 A real, live park-factor bug: relocated/displaced teams were
getting the WRONG ballpark's history (2026-07-25, prompted by an external
review flagging the two 2025-new venues — the actual bug turned out to be
different and more precise than the review's own framing, and directly
affects tonight's Rays game)

An external review of this documentation flagged Sutter Health Park (A's,
new 2025) and George M. Steinbrenner Field (Rays, new 2025) as a park-factor
gap, framed as "sparse data collapsing to neutral via the clip guards."
**Verified against real schedule data before acting on it** (this project's
own standing discipline: don't trust a claim, including an external one,
without checking it against ground truth) — the actual situation is
different and, for the Rays specifically, the review's framing was
backwards for the CURRENT season:

- **A's (ATH)**: real relocation, Oakland Coliseum (2023-2024) → Sutter
  Health Park (2025-2026, plus 6 stray 2026 games at Las Vegas Ballpark).
- **Rays (TB)**: Tropicana Field (2023-2024) → George M. Steinbrenner Field
  for exactly ONE season (2025, hurricane damage to the Trop) → **back to
  Tropicana Field in 2026**. The Rays are NOT still playing at Steinbrenner
  — they're back in their dome, with a real home game tonight (2026-07-25
  vs. CLE) and tomorrow.
- Every OTHER team showing >1 `venue_name` across 2023-2026 (CWS, HOU, LAD)
  is a same-building SPONSORSHIP RENAME (Guaranteed Rate Field→Rate Field,
  Minute Maid Park→Daikin Park, Dodger Stadium→UNIQLO Field at Dodger
  Stadium) — cosmetic, not a relocation, and irrelevant to the actual bug
  since `venue_name` never fed the park-factor MATH before this fix, only
  display metadata.

**The real mechanism, confirmed directly in `park_factors.py`**:
`build_outcome_park_factors`/`build_park_factors` roll a team's own
home/road history over the prior 3 CALENDAR seasons, keyed by team code
only — with no venue-continuity check at all. This silently blends a
relocated team's OLD ballpark into its new one's factor (ATH's 2025/2026
factors were built partly or entirely from Oakland Coliseum data, a
famously pitcher-friendly park, understating whatever Sutter Health Park
truly is), and — the part that matters for TONIGHT — **dilutes a
RETURNING team's already-stable history with a one-off displaced season**:
TB's pre-fix 2026 HR factor (0.980) blended 2 real Tropicana Field seasons
with 1 Steinbrenner Field season, right as the Rays came back to their real,
well-established dome. Not a sparse-data/cold-start problem — an actively
wrong one, silently confident rather than conservatively neutral.

**Fix**: `park_factors.py` gained `VENUE_RENAME_ALIASES` (canonicalizes the
3 known cosmetic renames so THEY don't get wrongly cold-started) and
`_same_venue_rolling_mean`/`_same_venue_rolling_sum` — instead of a plain
`.rolling(3)` over the last 3 calendar seasons, each team's rolling window
now looks back through its own season history and takes the most recent
(up to 3) seasons played at the SAME canonical venue as the season being
computed, skipping non-matching seasons rather than just the most recent
ones. `build_outcome_park_factors` (the one actually wired into
`props.py`/`game_simulator.py`) gained `_team_venue_lookup` to source venue
continuity from the raw schedule files (the PA table itself carries no
`venue_name` column), merged in before rolling.

**Verified on real data**:

| team | season | before | after | why |
|---|---|---|---|---|
| ATH (HR) | 2025 | 0.968 (Oakland-contaminated) | 0.968* | first Sutter Health season, correctly falls to neutral shrinkage — *value coincidentally close pre/post here, but now for the RIGHT reason (no real history) not the wrong one (Oakland history) |
| ATH (HR) | 2026 | 0.980 (2:1 Oakland:Sutter blend) | 0.947 | now pure 2025 Sutter Health data |
| TB (HR) | 2025 | 0.944 (correct — pre-Steinbrenner) | 0.968 | first Steinbrenner season, correctly falls to neutral shrinkage instead of stale Tropicana data |
| TB (HR) | 2026 | 0.980 (2yr Trop + 1yr Steinbrenner) | **0.946** | now pure 2023-2024 Tropicana Field data, Steinbrenner season correctly excluded — **this is tonight's number** |

Confirmed no regression for the other 28 teams: their factors move by a
small (mean 0.4%, max ~8%) amount purely from the population-mean
renormalization step re-centering after ATH/TB's raw factors changed — the
SAME renormalization mechanism already in place, unmodified, doing exactly
what it's supposed to do when any team's factor shifts. `build_park_factors`
(the simpler runs-based diagnostic function, not used in production) got
the identical fix for consistency, confirmed via its own `__main__` report.

**Scope note, honestly flagged**: this fixes the CURRENT (2026) season
correctly, which is what matters for tonight's game and the rest of this
season. It does NOT (and structurally cannot, without external ballpark
dimension/physics data this project doesn't have) give Sutter Health Park
or Steinbrenner Field a BETTER-than-neutral factor when they have zero or
minimal same-venue history — that's the same fallback the project's
existing `PARK_FACTOR_PRIOR_PA` shrinkage already provides for any
cold-start case, and remains the honest answer until enough real
same-venue seasons accumulate. Seeding a brand-new venue with an external
physics/dimensions prior (as suggested) is a legitimate future idea but a
different, separate, and more speculative build than this bug fix — not
combined with it here.

## 11.28 The offseason fitted-constants refresh ritual, concretely scoped
(2026-07-25, per external review item #2) — a full inventory, not a vague
placeholder

The review correctly flagged that every fitted regression in this project
was fit once against a specific season-pair window and then frozen, with
no scheduled cadence for revisiting any of them once a new season
completes. Rather than write a generic "have a ritual" note, did a full
inventory of every fitted constant in `src/models/` (~40 files checked)
first, so the ritual has a real, complete checklist to execute against
instead of relying on memory each offseason.

**Tier 1 — mandatory every offseason, once the completed season's own
data is available** (these are genuine same-methodology REFITS: rerun the
identical leakage-free regression with one more season-pair added, ship
only if it still clears its ORIGINAL validation bar, log the before/after
to the ledger regardless of outcome):

| Constant(s) | File | Refit adds | Re-clear this bar |
|---|---|---|---|
| `HR_SHARE_INTERCEPT/COEF_EXISTING/COEF_BARREL`, `HR_SHARE_PULLEDAIR_*` | `expected_stats.py` | 2025→2026 pair | Incremental R² positive, real-data multiplier range stays sane (no new blowup class) |
| `HR_SHARE_PITCHER_INTERCEPT/COEF_EXISTING/COEF_GB/COEF_FB` | `expected_stats.py` | 2025→2026 pair (4th season-pair) | R² gain holds; full-stack SU/Brier re-check (this one has a confirmed +1.53pp SU full-stack win riding on it — the highest-stakes refit here) |
| `DOUBLE_SHARE_PITCHER_*`, `DOUBLE_PLAY_SHARE_PITCHER_*` | `expected_stats.py` | 2025→2026 pair | R² gain holds; flag if the GB-coefficient sign instability (already seen in 1 of 3 original pairs) recurs |
| `GROUNDBALL_SINGLE_INTERCEPT/COEF_REAL/COEF_SPEED` | `expected_stats.py` | one more batter-season cohort | Multivariate R² improvement over real-history-alone holds |
| `PITCHER_STUFF_K_INTERCEPT/COEF_EXISTING/COEF_VELO/COEF_SPIN` | `expected_stats.py` | 2025→2026 pair (3rd) | R² improvement holds; real-data limit test (multiplier range) re-checked |
| `SHOCK_SIGMA` | `validate_game_simulator.py` | 2026 as a 3rd validation season | std(z) still ≈1.0 on 2026; re-run the K=30/100/300 trial-count-robustness check before trusting any drift |
| `HOOK_FRAILTY_SIGMA1`, `HOOK_FRAILTY_DECAY` | `hook_frailty.py` | 2025-2026 as fit data, refit grid search | Beats no-frailty and constant-sigma at every k on a NEW held-out season; re-check the still-unresolved P(still in at inning 7 \| clean start) tail |
| `tier_by_margin`/`closer_by_situation` policy tables | `bullpen_usage_policy.py` (`TIER_POLICY_FIT_SEASONS`/`HOOK_TABLE_FIT_SEASONS = {2023,2024}`) | roll the fit window forward or add 2025 | Reproduces real usage patterns on newly-held-out data (same bar as task #144 step 1) |
| `BATTER_PROP_CALIBRATION` (`p_2plus_hits`, `p_1plus_bb`, `p_1plus_rbi`) + a fresh 5-split check on `p_1plus_hit`/`p_1plus_hr` | `props.py` | fresh 150-game sample including 2026 | Full 5-split re-run (not just eyeballing point estimates — see sec 11.26's own lesson about this) |
| `K_EFFECT_SLOPE`/`BB_EFFECT_SLOPE` (catcher) | `catcher_framing.py` | 2025→2026 YoY pair (4th) | YoY stability check still in the 0.3-0.5 range already established |
| `K_EFFECT_SLOPE`/`BB_EFFECT_SLOPE` (umpire) | `umpire_factor.py` | 2025→2026 YoY pair (4th) | Same |
| `GB_BABIP_*`, `XBH_AIR_*` (defense/OAA) | `defense_factor.py` | one more team-season pair | Correlation/R² sign and rough magnitude hold (these were already weak, R²=0.08/0.01 — a sign flip would be a real reason to drop them) |
| Re-derived `STABILIZATION_PA_BATTER/_PITCHER` entries and `AGE_ADJ_SIGN_OVERRIDE` | `true_talent.py` | one more season of data | Correlation-vs-K sweep peak location unchanged; sign-override winner unchanged per category |

**Tier 2 — explicitly EXCLUDED from the annual refit** (literature/external
constants that should NOT be re-fit against this project's own, noisier
data): `MARCEL_WEIGHTS`, `AGE_PEAK`/`AGE_ADJ_RATE_BELOW_PEAK`/
`AGE_ADJ_RATE_ABOVE_PEAK` (Tango's published formula), 
`PLATOON_STABILIZATION_PA` (*The Book*'s published value), most of
`LINEAR_WEIGHTS` (published run values). Re-fitting these on our own data
would replace a stable, well-established external estimate with a noisier
in-house one for no real gain — don't.

**Tier 3 — one-time "actually fit this for the first time," not a
recurring refresh** (the inventory surfaced these as constants that were
never properly validated at all, just reasoned defaults — different action
item than a refit): `bullpen.py`'s `K_STARTS = 8` (explicitly flagged in
its own comment as "not yet empirically re-validated"), and
`weather_forecast.py`'s `WIND_MATCH_BOOST = 2.0` (explicitly documented as
never checked against a real forecast-quality backtest). Both are
long-standing, pre-existing gaps this inventory surfaced as a side effect,
not new problems — worth a session, but not urgent enough to jump the
queue ahead of task #154.

**Tier 4 — reverted/dead constants, periodic-not-annual reconsideration**:
`PITCH_WALK_*`, `CONTACT_QUALITY_BATSPEED_*`, `JETLAG_HIT_MULTIPLIER`, the
sprint-speed-conditioned-transitions mechanism. All real, statistically
fit signals that failed their full-stack A/B and are dead code today. Only
worth re-testing if the sample size grows enough to plausibly change the
full-stack verdict (order-of-magnitude more games, not one more season) —
don't burn an offseason session re-running these on a marginal sample
increase.

**Ritual, concretely**: once a season completes, work Tier 1 top-to-bottom,
each refit reusing that constant's ORIGINAL fitting script/methodology
(cited in the table above) with the new season-pair appended. Ship a
refit only if it clears the SAME bar the original did — a weaker bar
"just to keep the constant updated" defeats the purpose. Log every
refit attempt to the metrics ledger (kept or not), the same
append-only discipline this project already uses for every other
validation run. One scheduled session per offseason — this list makes
that session boundedly scoped rather than open-ended.

## 11.29 The October bat-speed/pulled-air resolution — protocol built and
smoke-tested now, execution deferred (2026-07-25, per external review item
#3)

Bat speed and pulled-air rate are this project's two longest-standing
"plausible-but-unconfirmed" signals (§11.7: real component-level R² gains,
but full-stack SU/Brier CIs that keep including zero — §11.7's own estimate
is that "multiple thousands" more games than the ~7,237 available would be
needed to resolve this). By the end of the 2026 season, ~2,400 more real
games become available. Rather than just re-noting the open question a
third time, built the exact script that will run then.

**`src/models/resolve_bat_speed_pulled_air.py`** (task #156): reuses
`run_validation` with two new permanent flags, `disable_bat_speed`/
`disable_pulled_air` (both `False` = byte-for-byte no-op, i.e. the TRUE
current production baseline). Runs 3 arms in one script execution —
baseline, bat-speed-off, pulled-air-off — specifically to avoid the exact
stale/mismatched-baseline bug §11.9 caught (there, two scratch test arms
were compared against a baseline parquet generated before an unrelated
signal had been wired in). Uses `ab_significance.bootstrap_compare` for
paired bootstrap CIs, same tool as every other keep/revert decision this
project makes.

**Pre-registered decision rule** (written into the script's own docstring,
so the eventual call can't be shaded after the fact): both Brier CIs
exclude zero in the beneficial direction → confirm both as real, proven
wins. Both still include zero even at ~9,600-9,700 games → this is treated
as license to actually REMOVE the dead-weight plumbing (delete
`CONTACT_QUALITY_BATSPEED_*`/bat-speed code from `expected_stats.py` and
its callers, same for `HR_SHARE_PULLEDAIR_*`/pulled-air) rather than
re-documenting the same ambiguous status a fourth time — matching this
project's stated "keep only on genuine net-positive" discipline. Mixed
result → decide each signal independently.

**Verified working, not yet run for real**: smoke-tested end-to-end on a
tiny sample (n=60, 10 trials, 2025 only) — confirmed no crashes, confirmed
the bootstrap comparison and ledger logging work. The bat-speed arm showed
an exact-zero delta at this tiny scale, which looked suspicious enough to
verify directly rather than wave off: called `contact_quality_multiplier`
directly for a real batter with real bat-speed data, with and without it
(`0.679` vs `0.767`, a real ~11% difference) — confirms the flag genuinely
changes the batter profile; the zero-delta smoke-test result was pure
small-sample coincidence at n=60/K=10, consistent with everything already
known about this signal's tiny true effect size, not a mechanism bug. Reset
to the real protocol values (`N_TRIALS=50`, `N_GAMES=25000` i.e. "every
game available") before committing — the smoke test's own temporary output
parquets were deleted, not left to be mistaken for a real read later.

**Do not run this before October, and do not iterate against it once run**
— same discipline as the 2026 H1 holdout (§0.3): one read, logged, acted
on per the rule above, not re-run if the first answer is unwelcome.

## 11.30 The systematic audit table — ranked, evidence-based answer to
"where does real predictive headroom still live?" (2026-07-25)

Prompted by the user's own question ("is there anything else we can do to
improve the predictive accuracy of this model? is it the best it can ever
get?"). Rather than keep finding new ideas ad hoc (the GB/FB pitcher
HR-allowed signal — this session's single largest real win — was originally
found exactly that way, by one person happening to think of it in
conversation), this builds the standing tool the model's own roadmap had
called for but never built: a ranked checklist covering every outcome
category, generalizing that discovery process instead of relying on which
ideas happen to come up.

**Method** (`src/models/audit_table.py`): for target_season in {2024, 2025}
(both have a real prior season to build a genuine preseason prior from,
unlike 2023), compute every qualifying player's CURRENT PRODUCTION preseason
estimate (`true_talent.build_preseason_priors` — the exact Marcel/Carleton-K/
age-adjusted prior actually used in production, not a re-derived one) and
correlate it against that player's REAL realized rate over the whole target
season (min 100 real PA to qualify, matching this project's own convention
elsewhere). R² = corr², averaged across the two target seasons. Multiply
`(1 − R²)` by the category's run-value leverage (`LINEAR_WEIGHTS × mean real
per-PA frequency`) to get a `priority_score` — ranking where a BETTER
ESTIMATOR (not just a better shrinkage constant; task #64 already
individually K-swept every category's `STABILIZATION_PA_*` on leakage-free
correlation, see those constants' own inline sweep comments) would move
real accuracy the most.

**Critical scope caveat, stated up front so the ranking isn't misread**:
this table measures ONLY the base Marcel-shrunk rate's own explanatory
power. It does NOT include any of the downstream context multipliers
(contact-quality/xBACON, HR-share/barrel rate, pulled-air rate, GB/FB
pitcher mix, platoon, park, weather, base-out state, TTOP) that are applied
LATER in the real pipeline. A high `priority_score` therefore does NOT mean
"nothing has been done here" — several top-ranked categories already have
deployed, validated downstream fixes that this base-rate-only R² gives no
credit for. The table answers "how much does the entry-point estimate
leave on the table," not "how good is the final per-PA probability."

**Full ranked output** (32 rows, batter + pitcher × 16 outcome categories,
`data/processed/audit_table.parquet`):

| rank | side | outcome | R² | priority_score |
|---|---|---|---|---|
| 1 | pitcher | field_out | 0.155 | 0.0958 |
| 2 | batter | field_out | 0.431 | 0.0653 |
| 3 | pitcher | single | 0.110 | 0.0589 |
| 4 | batter | single | 0.290 | 0.0470 |
| 5 | pitcher | strikeout | 0.349 | 0.0445 |
| 6 | pitcher | home_run | 0.047 | 0.0398 |
| 7 | pitcher | double | 0.011 | 0.0316 |
| 8 | batter | double | 0.021 | 0.0314 |
| 9 | batter | strikeout | 0.573 | 0.0291 |
| 10 | batter | home_run | 0.285 | 0.0289 |
| 11+ | ... | (walk, double_play, triple, HBP, field_error, sac_fly, intent_walk, fielders_choice, sac_bunt, catcher_interf, triple_play) | — | ≤0.021, long tail |

**Cross-referencing the top of the table against what's already been tried**
(the actual point of building this — a ranking is only useful once checked
against history):

- **#1 pitcher field_out (R²=0.155, top priority_score)**: this is DIPS —
  already investigated and confirmed largely irreducible at the individual-
  pitcher level (see the run-value screen / earlier DIPS-adjacent sections);
  a pitcher's own year-to-year control over batted-ball-out rate specifically
  (as opposed to K/BB/HR, which pitchers DO control) is a well-established
  sabermetric near-null. High priority_score here reflects real,
  irreducible variance, not an unexploited opportunity.
- **#7/#8 double (both sides, R² ≈ 0.01–0.02, the single worst-explained
  category on the whole table)**: matches an earlier documented finding
  (sec 11.4/11.26 pull-tercile-park-interaction work, and prior xSLG
  investigation) that doubles resist modeling from available Statcast
  inputs — already investigated, confirmed genuinely hard, not neglected.
- **#3/#4 single, #5/#9 strikeout**: both sides already have deployed
  context multipliers (contact-quality/xBACON for hit categories generally,
  the Carleton-K stabilization + age curve for strikeout specifically) —
  high base-rate priority_score here is expected and largely already
  addressed downstream.
- **#6 pitcher home_run (R²=0.047) / #10 batter home_run (R²=0.285)**: HR is
  the one top-10 category where a genuinely open, NOT-yet-successfully-
  captured opportunity exists. The bat-speed extension to HR-share
  (sec 11.7/11.9) was previously found to carry real incremental R²
  (0.379→0.405 in that investigation) but was rejected specifically because
  its clip design produced unsafe multipliers (up to 5.5x for batters with
  zero HR in their own sample) — a rejected IMPLEMENTATION, not a rejected
  SIGNAL. This audit table independently reconfirms real R² is still on the
  table for batter home_run from a totally different angle (whole-season
  preseason-vs-real correlation, not a within-model incremental-fit test),
  which is exactly the kind of convergent evidence that should raise this
  above "already closed." A safer clip design (e.g. shrinking the
  multiplier itself, not just clipping its output, or requiring a minimum
  batted-ball sample before applying it) is the concrete, evidence-backed
  next candidate this table surfaces — not vague brainstorming.
- **Task #156's own bat-speed/pulled-air resolution** (sec 11.29, scheduled
  for October) tests whether the CURRENTLY DEPLOYED bat-speed signal helps
  at the whole-simulator level; this table's finding is complementary, not
  redundant — it's evidence the underlying HR-share R² gap exists
  independent of whether the current implementation captures it well.

**Honest framing of the answer to "is this the best it can ever get?"**: no
category on this table shows a "smoking gun" free R² of 0.3+ sitting
completely untouched — the strongest true nulls (doubles, pitcher
batted-ball-out rate) are real, published, irreducible sabermetric limits,
not gaps in this project's own work. The one live, actionable thread this
table surfaces is the batter/pitcher home_run clip-design revisit above.
Logged to the metrics ledger; `src/models/audit_table.py` is a standing,
re-runnable tool (not a one-time script) — worth re-running whenever a new
downstream signal is deployed for a top-ranked category, to check whether
the deployed fix actually closed the gap this table originally measured.

## 11.31 Task #158: the two Tier 3 "never actually validated" placeholders,
checked for the first time (2026-07-25)

Continuing directly from the audit table (sec 11.30) toward the
foundation-constants sweep — but most of Tier 1's own refit list (sec
11.28) requires a COMPLETED 2026 season (several entries are explicitly
"2025→2026 season-PAIR" refits, e.g. `HR_SHARE_*`'s recency-weighted prior-
season average needs both halves of the pair to be full seasons). Running
those mid-season (today is 2026-07-25, 2026 is roughly half over) would
silently corrupt the exact same-methodology bar the ritual itself demands —
so Tier 1 stays queued for real offseason execution. Tier 3's two items,
by contrast, test already-COMPLETE 2023-2025 data and were never blocked at
all — pivoted to those instead.

**(1) `K_STARTS = 8` in `bullpen.py`** — same leakage-free K-sweep
methodology as task #64's `STABILIZATION_PA_*` sweep: preseason (prior-
season-only, recency-weighted) expected-innings-per-start estimate at each
candidate K, correlated against real target-season innings/start (min 8
real starts to qualify), for target_season in {2024, 2025}.

Result: the two target seasons **disagree in direction** across the full
tested range (K=0.5 to K=300) — 2024 favors close to ZERO shrinkage
(corr peaks at the smallest tested K, 0.599 at K=0.5, monotonically falling
to 0.324 at K=300), while 2025 is nearly flat but if anything trends the
OTHER way, still rising slightly at the largest tested K (0.340 at K=0.5 up
to 0.362 at K=300). Neither season shows an interior peak; they don't even
agree on which direction is better. Per this project's own "requires a
stable, agreeing sweep before changing a constant" discipline (task #64),
this is a clean non-result — no re-tune is justified. Context: pooled
`prior_starts` has a median of ~130 (multi-season recency-weighted), so for
a typical established starter K=8 is already negligible relative to that
scale (reliability ≈0.94+ regardless of K); the constant mostly only
matters for pitchers with under ~20 prior starts (rookies, short-tenured
relievers converted to starters), a real caveat on how much this constant
even can matter in practice. **Kept at 8, now checked for the first time
rather than left as a never-validated placeholder** — no code change.

**(2) `WIND_MATCH_BOOST = 2.0` in `weather_forecast.py`** — the module's
own docstring flagged this as the one piece in the whole weather stack
never checked against a real forecast-quality backtest (existing
validators only replay REAL POSTED weather, a code path this one never
touches). Built `src/models/validate_wind_forecast_boost.py`: for every
`CONFIDENT_TEAMS` game with real posted weather (2023-2025, non-domed,
n=5411), fetched that venue's REAL historical wind at ~game time from
archive-api.open-meteo.com (the same free source `park_orientation.py`
used for its own bearing calibration) as a stand-in for what a live
forecast call would have returned, classified via the same
`forecast_wind_to_bucket_suffix` the live code path uses.

Two-stage result:
- **Real signal confirmed first**: forecast-suffix classification accuracy
  is 37.0% (n=5369 directional games) vs. base rates of 15-21% for the
  most common real labels (calm 21.5%, cross_LtoR 15.3%) — clearly real,
  not noise, before trusting any boost-value comparison built on it.
- **Genuine 60/40 train/test split** (climatology fit on a random 60%,
  scored leakage-free on the held-out 40%, 5 seeds — this project's own
  5-split stability convention applied to log-loss instead of
  correlation): `WIND_MATCH_BOOST=2.75` beat the previous default of 2.0
  on held-out log-loss in **5/5 splits**. The log-loss curve is smooth and
  single-peaked between 2.5-3.0 in every split (never a sharp single-point
  spike), so 2.75 was chosen as the stable middle of that range rather than
  chasing any one split's exact minimum. Example (seed 0, n≈1559 test
  games): log-loss 1.26563 (no boost) → 1.19420 (boost=2.0, old default)
  → 1.18378 (boost=2.75, new default) → 1.18350 (boost=3.0).

**Deployed**: `WIND_MATCH_BOOST` raised 2.0 → 2.75 in `weather_forecast.py`,
with the module docstring's "never validated" caveat updated to point here.
Both findings logged to the metrics ledger.

## 11.32 Task #159: revisiting the rejected bat-speed HR-share extension —
built safely, tested, reverted; a real bug fix survives (2026-07-25)

Directly following the audit table (sec 11.30), which independently
reconfirmed real R² was still on the table for HR-share, and specifically
flagged the previously-rejected bat-speed extension to `hr_share_multiplier`
(sec on `CONTACT_QUALITY_BATSPEED_*`/`hr_share_multiplier`) as a
not-yet-successfully-captured opportunity: was the rejection (a 5.5x
real-data-limit-test blowup on zero-HR batters) really about bat speed, or
fixable with a better clip design?

**Root cause, found by reconstruction**: refit the original 2-season-pair
leakage-free regression (existing_hr_share + barrel_rate + bat_speed →
next-season real HR-share, n=734 real batters with 50+ BIP/season, 100+
real bat-tracking swings) and reproduced the blowup directly. The worst
offenders (existing_hr_share=0, real BIP samples of 51-372) have
perfectly ORDINARY bat speeds (64.9-71.8 mph, right around the league
mean) — the blowup was never really about bat speed being extreme. Direct
comparison confirmed it: the **currently-deployed, unrelated 2-term base
formula** (`HR_SHARE_INTERCEPT`/`COEF_EXISTING`/`COEF_BARREL`, live since
task #51/52, untouched until today) produces up to **5.02x** on the exact
same zero-HR batters — a real, previously-undiscovered defect in code
that's been in production this whole time, not something introduced by the
new bat-speed term.

**Fix**: `HR_SHARE_CLIP_MIN` raised 0.02 → 0.035 (still just above the
genuine empirical 1st percentile, so only ~5% of real batters — the
extreme low tail — actually have their floor moved, unlike the previously-
rejected 0.10 floor which would have distorted a much broader swath of
real weak-power hitters). This tames BOTH the already-deployed base
formula (max multiplier 5.02x → 2.88x) and a new bat-speed extension
(4.71x → 2.86x, zero cases above 3x) at once. Refit `HR_SHARE_*`/
`HR_SHARE_PULLEDAIR_*` coefficients under the new floor (R² essentially
unchanged: 0.429/0.392 and 0.440/0.414 respectively, comparable to the
pre-existing fits). Built two new regressions: `HR_SHARE_BATSPEED_*`
(existing+barrel+bat_speed, used when pulled-air isn't available) and
`HR_SHARE_FULL_*` (all 4 terms, used when both pulled-air and bat-speed
are available — the common case, 734/772 real batter-seasons had both —
capturing more real signal than picking one arbitrarily: R² 0.471/0.438
vs. 0.440/0.414 or 0.439/0.406 alone).

**Full-stack isolated A/B** (canonical protocol, n=8711 real games,
SHOCK_SIGMA=0.40, OLD = pre-task-159 `expected_stats.py` vs. NEW = clip
retune + both new regressions wired in via a 4-branch cascade in
`hr_share_multiplier`): **NOISE on both decision metrics.** SU delta
+0.0068 (OLD 0.575 → NEW 0.568), 95% CI (-0.0052, +0.0187) — includes
zero. Brier delta -0.0004 (OLD 0.2422 → NEW 0.2426), 95% CI (-0.0025,
+0.0017) — includes zero. Point estimates, if anything, trended slightly
unfavorable. Per this project's own "keep only on genuine net-positive"
discipline (a cleared safety bar is necessary but not sufficient), this
does NOT clear the bar to ship — **REVERTED** the bat-speed/4-term
extension itself (`HR_SHARE_BATSPEED_*`/`HR_SHARE_FULL_*` constants and
the 4-branch cascade removed from `hr_share_multiplier`), same "real
component-level signal, no detectable full-stack win" treatment as CSW%
(sec entry) and the pitcher-side contact-suppression investigation (task
#104). Consistent with this project's own repeated finding elsewhere that
bat-speed-based signals carry a genuinely small true effect size (sec
11.7/11.9's own "multiple thousands of games" estimate for the
contact-quality/pulled-air bat-speed signals) — HR-share specifically is
too thin a slice to move a full-stack metric detectably at n=8711.

**What survives**: the `HR_SHARE_CLIP_MIN` retune (0.02→0.035) and the
resulting `HR_SHARE_*`/`HR_SHARE_PULLEDAIR_*` refits. Re-validated this
piece IN ISOLATION (clip fix only, no bat-speed cascade) against the same
OLD baseline, n=8711: SU delta -0.0013 (OLD 0.575 → 0.576), CI
(-0.0133, +0.0107), NOISE; Brier delta +0.0008 (OLD 0.2422 → 0.2414),
CI (-0.0012, +0.0028), NOISE — both directionally slightly FAVORABLE this
time, safely neutral-to-positive, confirming the clip fix alone carries no
regression risk. Kept as a standalone robustness fix (removes a real,
already-existing tail-risk defect in the deployed formula) independent of
whether the new bat-speed signal ever gets deployed — the same "fix a real
defect even without a proven accuracy win" logic already used for the
weather-regularization (73x) and platoon (146,611x) blowup fixes (sec 4).

Net effect of task #159 on the live model: `hr_share_multiplier`'s public
signature and behavior are UNCHANGED from before this task (still 2-branch:
pulled-air-or-not) — the only shipped change is the tightened clip floor
and its refit coefficients, a pure robustness improvement with no new
signal. The bat-speed HR-share idea is now closed with a decisive, honest
null (not just "not yet built") — the audit table's own flagged opportunity
was real to investigate, but didn't survive its full-stack test.

**§11.8's critique is now fully resolved except claim 5 and the 3 smaller notes** (2026-07-22):
HFA and park-neutralization are built, wired, and kept (correctness-fix grounds, full-stack
effect not distinguishable at n=597 — §11.8's status note). The backtest protocol is scaled to
n=7237 (claim 6), now the reference baseline (56.3% SU / 0.2416 Brier / 3.543 total MAE / 3.442
margin MAE) superseding every earlier-cited figure in this document. Rookie priors
(`use_debut_prior`, task #119) tested decisively NULL on that protocol (tight CI, not
deployed). GB/FB→pitcher-HR-allowed (task #120) tested REAL (SU +1.53pp, CI excludes zero) and
is now live in all 3 consumer files. Bat speed and pulled-air rate were re-tested at n=7237
too (§11.9) — both still NOISE (CI excludes neither zero nor a small effect, ~±1.2pp), a
tighter confirmation of §11.7's original "unresolved" verdict, not a reversal — neither
reverted. **Remaining open work**: (a) the total-runs dispersion diagnostic (claim 5,
std(z)=1.087, under-dispersed — no fix attempted); (b) the 3 smaller notes (SB battery-side,
bullpen back-to-back performance, stale AGE_PEAK=29); (c) HFA/park-neutralization's own
full-stack delta is still only measured at the smaller n=597 protocol (§11.8's status note) —
re-running that specific comparison on the n=7237 protocol remains open.
**The phased roadmap below (per an external reviewer, 2026-07-22, reacting to §11.10-11.11's
close-out) supersedes every list previously in this section — those are preserved in git
history, not repeated here.** Sequenced by dependency and expected value; effort estimates
assume the current toolkit (CRN pairing, run-value screen, n=7237 protocol) as standard.

### Phase 0 — Lock the foundation (one short session)

**0.1 Persisted metrics ledger.** This session's own wrong-baseline mistake (§11.9) and the
project's habit of citing numbers no file actually stores are the same disease: no
append-only record of runs. Every validation run should write one row (run ID, git hash,
config/factor flags, n, trials, SU, Brier, total/margin MAE, std(z), PIT coverage, home-win
share) to a parquet this document can defer to instead of restating figures inline. Cheap
(~an hour), and prevents the whole class of reference-point errors permanently.

**0.2 Re-baseline at full n with the complete diagnostic suite.** One canonical n=7237 run
emitting every metric above together. Closes two open items for free: HFA/park-
neutralization's full-stack effect gets measured at a sample size that can actually resolve
it (§11.8's status note is still only at n=597), and simulated home-win share can be checked
directly against the real 52.3% as confirmation the HFA fix is calibrated, not just present.

**0.3 The 2026 first-half holdout — the single most informative run available right now.**
It's 2026-07-22; roughly 1,300-1,400 completed 2026 games exist that NO fitting, selection,
or keep/revert decision in this project has ever touched. Every kept signal was chosen using
2023-2025 backtests, so selection bias accumulates across dozens of decisions even with fully
honest per-test methodology — and no further 2023-2025 testing can detect that. One
walk-forward run against the CURRENT FROZEN stack on 2026 H1 answers whether 57.9% is the
real number. Two rules: run it once and don't iterate against it (iterating turns a holdout
into just more training data), then adopt 2026 as a standing rolling out-of-sample set going
forward so the selection-bias problem never re-accumulates silently.

### Phase 1 — The pitcher-appearance latent effect (the dispersion fix; 1-2 sessions)

The marked frontier per §11.11 — the highest-value remaining build, with a known destination
before starting (the diagnostic already proves the current shape is wrong).

**1.1 Confirm the mechanism in real data FIRST** — the same component-level discipline this
project already demands of every signal. Test for day-level pitcher overdispersion directly:
for each real start, compare the variance of per-start outcome rates (wOBA-allowed, or
K/BB/HR rate per start) against the binomial variance the pitcher's own season rate implies.
Real day-to-day effectiveness variation shows up as excess variance beyond that binomial
floor; a split-half check (odd vs. even PAs within the same start, do their residuals
correlate) confirms it's a shared within-start shock, not noise. Run the same test on
team-offense-days to size the batter-side analog too — published expectation is the pitcher
side dominates, but measure both before assuming.

**1.2 Minimal implementation once 1.1 confirms it's real**: one latent scalar per
(pitcher-appearance, trial) — draw `g ~ Normal(0, σ²)` once, apply in odds space to that
pitcher's whole allowed-rate vector (one shared shift moving K down and BB/hits/HR up
together, or the reverse), mean-corrected so expected rates are unchanged in aggregate. Start
with one global σ; only split starter-vs-reliever σ if 1.1's own measurement shows they
differ materially. This is deliberately the smallest possible version of within-game
correlation — one new parameter, drawn once per trial, touching no matchup logic.

**1.3 Fit σ on 2023-2024, validate on 2025** (NOT on the same data used to tune it, to avoid
overfitting the very diagnostic being targeted): sweep σ to close the dispersion diagnostic
(std(z) → ~1.0, 13+/≤4-run tail frequencies matching reality) on the fit seasons, then check
the held-out season's own PIT coverage at 50/80/95%.

**1.4 Acceptance criteria, stated BEFORE running**: std(z) within ~0.03 of 1.0 on held-out
data; PIT coverage CIs containing nominal; CRN-paired Brier delta not significantly worse;
**SU expected flat and NOT the arbiter**. This targets distribution SHAPE, not matchup
separation — judging it on SU is exactly how a good calibration fix gets wrongly reverted
(the same mistake this session's §11.9 correction was about, in reverse). Note the explicit
exception to §11.5's "new heterogeneity axis = full-stack risk" rule: this IS formally a new
heterogeneity axis, but unlike the 6 that failed, it isn't trying to improve matchup
separation — it's a correctness fix in the same family as HFA, with a diagnostic that already
proves the current shape is wrong before any code is written. Judge it on the metric it
targets, not the one it was never meant to move.

**1.5 Measure the residual.** If a gap remains after 1.2-1.4, the leftover is within-INNING
contagion (not just within-game) — a separate, harder build, worth attempting only if the
residual after this phase actually justifies it.

Note the ordering dependency already satisfied: without CRN pairing (§11.10), adding this new
source of within-game variance would have wrecked the ability to measure anything ELSE
afterward (every future A/B's noise floor would widen along with the intentional dispersion
fix) — CRN had to come first, and now it has.

### Phase 2 — Cash the calibration in where the value actually lives (one session)

**2.1 Re-fit prop calibration.** `validate_prop_calibration.py`'s existing linear
recalibration coefficients were fit under the under-dispersed model — after Phase 1 they are
stale by construction. Re-fit and re-validate at full n once Phase 1 lands.

**2.2 Build totals/margin distribution outputs** (P(total > X) across the ladder, P(team
total > X), margin quantiles) directly from the now-correctly-dispersed trial distribution.
This is where the dispersion fix actually converts into better posted picks: an
under-dispersed model systematically overprices the middle and underprices the tails on
every total, and Phase 1 removes that bias at its source.

**2.3 Measure the oracle-vs-deployable gap at full n.** Scale `validate_predictive_bullpen.py`
to the n=7237 protocol and decompose the degradation vs. the oracle backtest by component
(predictive bullpen, predictive catcher, forecast weather, lineup projection). Whatever
dominates that gap is a LIVE-accuracy improvement no further oracle-backtest signal work can
buy — plausibly lineup-projection timing and bullpen usage, in that order, though this should
be measured, not assumed.

### Phase 3 — Clear the open ledger through the cheap funnel (ongoing, low effort each)

Run everything below through the standard funnel (run-value screen → sliced CRN check →
full-n confirmation, §11.10) and let most of it die cheaply rather than consuming a full
backtest to reject:

- **Bat speed and pulled-air rate, resolved permanently.** Run the run-value screen first —
  if their maximum plausible per-game margin impact is small, reclassify as "kept on
  component evidence, full-stack immaterial by construction" and stop re-litigating this
  question. Only a screen showing real material impact earns one more CRN-paired full-n test.
- **The systematic audit table** (generalizing how the GB/FB win was actually found): for
  each of ~16 outcome categories × both sides, compute the walk-forward predictive R² of the
  CURRENT production estimate, multiply by run-value leverage. This mechanically ranks where
  a better estimator could matter, replacing research-pass-driven idea generation with a
  checklist — and tells you definitively when the well is actually dry instead of guessing.
- **The 3 smaller flagged items** (§11.8): `AGE_PEAK=29` → ~26-27 (a constants change, the
  cheapest test in this entire queue — run it first), stolen-base battery-side suppression
  folded into existing runner SB rates as a multiplier, reliever back-to-back-day performance
  penalty on existing reliever rates. Plus two never-tried park/weather refinements: roof
  status and a league-season ball/drag term.

### Phase 4 — Operational hardening (background, ongoing)

2026 is live and this model is presumably informing real picks: adopt rolling walk-forward
evaluation against the current season (a weekly Brier/calibration check against the Phase 0.1
ledger), automate the daily pipeline's completeness checks for in-season use, and version the
model so any posted pick is traceable to a specific git hash. Unglamorous, but it's what makes
whatever accuracy number gets quoted a real, checkable one rather than a vibe.

**Sequencing logic in one line**: Phase 0 makes every future number trustworthy (and 0.3 is
the single most informative run available this week); Phase 1 is the one remaining
known-destination build; Phase 2 converts it into what the Discord use case actually consumes;
Phase 3 is cheap opportunistic upside now that rejection costs minutes, not hours; Phase 4
protects the whole thing once it's actually being used. **If only two things happen next
session: the 2026 holdout run (0.3), then start the latent-effect measurement (1.1).**

## 11.33 Task #140: canonical protocol trial count raised 50→200, plus an
unexpected side-finding about what actually drove the dispersion drift
(2026-07-26/08-01)

Sec 11.16 flagged this as an explicitly-deferred open item — do not touch
until the monthly monitoring plan (sec 11.17) actually surfaces a reason.
Nothing from monitoring has triggered this; picked up on direct request
instead, with that deferral rule surfaced and acknowledged before starting.

**What changed**: `N_TRIALS_PER_GAME` raised 50→200 in the three canonical
validators (`validate_game_simulator.py`, `validate_predictive_bullpen.py`,
and `validate_holdout_2026.py`'s own constant, so the *next* scheduled
holdout read — first Monday of August, not run early — automatically
benefits). A fresh full re-baseline was run at the new K (n=7237,
2023-2025, current code): **SU 0.587, Brier 0.2387, std(z)=0.963**, logged
to the metrics ledger as the new canonical reference, superseding every
earlier-cited K=50 figure in this document going forward.

**The side-finding**: the first re-baseline run showed std(z) dropping from
1.136 (the long-standing K=50 baseline, stable across many prior sessions'
runs) to 0.963 — a much bigger shift than the "K only weakly affects
dispersion" theory this project has held since sec 11.16 would predict.
Investigated before writing that down as a real K-effect: a genuinely
apples-to-apples check (same current code, same n=7237 games, ONLY K
varied — K=50 vs. K=200) showed std(z) moving just 0.998→0.963, i.e. a
small, modest effect, matching the original theory closely.

**So what actually explains the 1.136→0.998 improvement (at the SAME
K=50)?** The "old" K=50 baseline being compared against was generated
before today's two other changes landed — task #158's `WIND_MATCH_BOOST`
increase (2.0→2.75) and task #159's `HR_SHARE_CLIP_MIN` retune
(0.02→0.035) — both shipped between that baseline and this session's
re-check. Between them, these two small, independently-validated fixes
apparently also meaningfully improved the simulator's aggregate total-runs
dispersion calibration, discovered only as a side effect of this K-upgrade
investigation, not measured directly at the time either shipped (neither
task #158 nor #159's own full-stack A/B specifically scored dispersion —
both were scored on SU/Brier, or in #158's case a domain-specific
backtest). Not decomposed further (would need one more isolated run per
fix) — flagged honestly as a real, attributed-but-not-split improvement
rather than over-claiming either fix alone caused it.

**Net conclusion**: task #140's own premise — raise K for a permanent,
higher-precision baseline — is done and modestly helps (std(z) 0.20 pp
closer to 1.0, SU +0.35pp, Brier -0.0015 attributable to K alone). The
much larger dispersion improvement visible in casual before/after
comparisons this session belongs to tasks #158/#159, not #140 — an
important attribution correction for any future reader tempted to credit
the wrong change. Real, bounded one-time cost paid (~4x runtime vs. the
old K=50 canonical run); no further trial-count increases planned.

## 12. Suggested next steps for a future session

**§11.8's critique is now fully resolved except claim 5 and the 3 smaller notes** (2026-07-22):
HFA and park-neutralization are built, wired, and kept (correctness-fix grounds, full-stack
effect not distinguishable at n=597 — §11.8's status note). The backtest protocol is scaled to
n=7237 (claim 6), now the reference baseline (56.3% SU / 0.2416 Brier / 3.543 total MAE / 3.442
margin MAE) superseding every earlier-cited figure in this document. Rookie priors
(`use_debut_prior`, task #119) tested decisively NULL on that protocol (tight CI, not
deployed). GB/FB→pitcher-HR-allowed (task #120) tested REAL (SU +1.53pp, CI excludes zero) and
is now live in all 3 consumer files. Bat speed and pulled-air rate were re-tested at n=7237
too (§11.9) — both still NOISE (CI excludes neither zero nor a small effect, ~±1.2pp), a
tighter confirmation of §11.7's original "unresolved" verdict, not a reversal — neither
reverted. **Remaining open work**: (a) the total-runs dispersion diagnostic (claim 5,
std(z)=1.087, under-dispersed — no fix attempted); (b) the 3 smaller notes (SB battery-side,
bullpen back-to-back performance, stale AGE_PEAK=29); (c) HFA/park-neutralization's own
full-stack delta is still only measured at the smaller n=597 protocol (§11.8's status note) —
re-running that specific comparison on the n=7237 protocol remains open.
**The phased roadmap below (per an external reviewer, 2026-07-22, reacting to §11.10-11.11's
close-out) supersedes every list previously in this section — those are preserved in git
history, not repeated here.** Sequenced by dependency and expected value; effort estimates
assume the current toolkit (CRN pairing, run-value screen, n=7237 protocol) as standard.

### Phase 0 — Lock the foundation (one short session)

**0.1 Persisted metrics ledger.** This session's own wrong-baseline mistake (§11.9) and the
project's habit of citing numbers no file actually stores are the same disease: no
append-only record of runs. Every validation run should write one row (run ID, git hash,
config/factor flags, n, trials, SU, Brier, total/margin MAE, std(z), PIT coverage, home-win
share) to a parquet this document can defer to instead of restating figures inline. Cheap
(~an hour), and prevents the whole class of reference-point errors permanently.

**0.2 Re-baseline at full n with the complete diagnostic suite.** One canonical n=7237 run
emitting every metric above together. Closes two open items for free: HFA/park-
neutralization's full-stack effect gets measured at a sample size that can actually resolve
it (§11.8's status note is still only at n=597), and simulated home-win share can be checked
directly against the real 52.3% as confirmation the HFA fix is calibrated, not just present.

**0.3 The 2026 first-half holdout — the single most informative run available right now.**
It's 2026-07-22; roughly 1,300-1,400 completed 2026 games exist that NO fitting, selection,
or keep/revert decision in this project has ever touched. Every kept signal was chosen using
2023-2025 backtests, so selection bias accumulates across dozens of decisions even with fully
honest per-test methodology — and no further 2023-2025 testing can detect that. One
walk-forward run against the CURRENT FROZEN stack on 2026 H1 answers whether 57.9% is the
real number. Two rules: run it once and don't iterate against it (iterating turns a holdout
into just more training data), then adopt 2026 as a standing rolling out-of-sample set going
forward so the selection-bias problem never re-accumulates silently.

### Phase 1 — The pitcher-appearance latent effect (the dispersion fix; 1-2 sessions)

The marked frontier per §11.11 — the highest-value remaining build, with a known destination
before starting (the diagnostic already proves the current shape is wrong).

**1.1 Confirm the mechanism in real data FIRST** — the same component-level discipline this
project already demands of every signal. Test for day-level pitcher overdispersion directly:
for each real start, compare the variance of per-start outcome rates (wOBA-allowed, or
K/BB/HR rate per start) against the binomial variance the pitcher's own season rate implies.
Real day-to-day effectiveness variation shows up as excess variance beyond that binomial
floor; a split-half check (odd vs. even PAs within the same start, do their residuals
correlate) confirms it's a shared within-start shock, not noise. Run the same test on
team-offense-days to size the batter-side analog too — published expectation is the pitcher
side dominates, but measure both before assuming.

**1.2 Minimal implementation once 1.1 confirms it's real**: one latent scalar per
(pitcher-appearance, trial) — draw `g ~ Normal(0, σ²)` once, apply in odds space to that
pitcher's whole allowed-rate vector (one shared shift moving K down and BB/hits/HR up
together, or the reverse), mean-corrected so expected rates are unchanged in aggregate. Start
with one global σ; only split starter-vs-reliever σ if 1.1's own measurement shows they
differ materially. This is deliberately the smallest possible version of within-game
correlation — one new parameter, drawn once per trial, touching no matchup logic.

**1.3 Fit σ on 2023-2024, validate on 2025** (NOT on the same data used to tune it, to avoid
overfitting the very diagnostic being targeted): sweep σ to close the dispersion diagnostic
(std(z) → ~1.0, 13+/≤4-run tail frequencies matching reality) on the fit seasons, then check
the held-out season's own PIT coverage at 50/80/95%.

**1.4 Acceptance criteria, stated BEFORE running**: std(z) within ~0.03 of 1.0 on held-out
data; PIT coverage CIs containing nominal; CRN-paired Brier delta not significantly worse;
**SU expected flat and NOT the arbiter**. This targets distribution SHAPE, not matchup
separation — judging it on SU is exactly how a good calibration fix gets wrongly reverted
(the same mistake this session's §11.9 correction was about, in reverse). Note the explicit
exception to §11.5's "new heterogeneity axis = full-stack risk" rule: this IS formally a new
heterogeneity axis, but unlike the 6 that failed, it isn't trying to improve matchup
separation — it's a correctness fix in the same family as HFA, with a diagnostic that already
proves the current shape is wrong before any code is written. Judge it on the metric it
targets, not the one it was never meant to move.

**1.5 Measure the residual.** If a gap remains after 1.2-1.4, the leftover is within-INNING
contagion (not just within-game) — a separate, harder build, worth attempting only if the
residual after this phase actually justifies it.

Note the ordering dependency already satisfied: without CRN pairing (§11.10), adding this new
source of within-game variance would have wrecked the ability to measure anything ELSE
afterward (every future A/B's noise floor would widen along with the intentional dispersion
fix) — CRN had to come first, and now it has.

### Phase 2 — Cash the calibration in where the value actually lives (one session)

**2.1 Re-fit prop calibration.** `validate_prop_calibration.py`'s existing linear
recalibration coefficients were fit under the under-dispersed model — after Phase 1 they are
stale by construction. Re-fit and re-validate at full n once Phase 1 lands.

**2.2 Build totals/margin distribution outputs** (P(total > X) across the ladder, P(team
total > X), margin quantiles) directly from the now-correctly-dispersed trial distribution.
This is where the dispersion fix actually converts into better posted picks: an
under-dispersed model systematically overprices the middle and underprices the tails on
every total, and Phase 1 removes that bias at its source.

**2.3 Measure the oracle-vs-deployable gap at full n.** Scale `validate_predictive_bullpen.py`
to the n=7237 protocol and decompose the degradation vs. the oracle backtest by component
(predictive bullpen, predictive catcher, forecast weather, lineup projection). Whatever
dominates that gap is a LIVE-accuracy improvement no further oracle-backtest signal work can
buy — plausibly lineup-projection timing and bullpen usage, in that order, though this should
be measured, not assumed.

### Phase 3 — Clear the open ledger through the cheap funnel (ongoing, low effort each)

Run everything below through the standard funnel (run-value screen → sliced CRN check →
full-n confirmation, §11.10) and let most of it die cheaply rather than consuming a full
backtest to reject:

- **Bat speed and pulled-air rate, resolved permanently.** Run the run-value screen first —
  if their maximum plausible per-game margin impact is small, reclassify as "kept on
  component evidence, full-stack immaterial by construction" and stop re-litigating this
  question. Only a screen showing real material impact earns one more CRN-paired full-n test.
- **The systematic audit table** (generalizing how the GB/FB win was actually found): for
  each of ~16 outcome categories × both sides, compute the walk-forward predictive R² of the
  CURRENT production estimate, multiply by run-value leverage. This mechanically ranks where
  a better estimator could matter, replacing research-pass-driven idea generation with a
  checklist — and tells you definitively when the well is actually dry instead of guessing.
- **The 3 smaller flagged items** (§11.8): `AGE_PEAK=29` → ~26-27 (a constants change, the
  cheapest test in this entire queue — run it first), stolen-base battery-side suppression
  folded into existing runner SB rates as a multiplier, reliever back-to-back-day performance
  penalty on existing reliever rates. Plus two never-tried park/weather refinements: roof
  status and a league-season ball/drag term.

### Phase 4 — Operational hardening (background, ongoing)

2026 is live and this model is presumably informing real picks: adopt rolling walk-forward
evaluation against the current season (a weekly Brier/calibration check against the Phase 0.1
ledger), automate the daily pipeline's completeness checks for in-season use, and version the
model so any posted pick is traceable to a specific git hash. Unglamorous, but it's what makes
whatever accuracy number gets quoted a real, checkable one rather than a vibe.

**Sequencing logic in one line**: Phase 0 makes every future number trustworthy (and 0.3 is
the single most informative run available this week); Phase 1 is the one remaining
known-destination build; Phase 2 converts it into what the Discord use case actually consumes;
Phase 3 is cheap opportunistic upside now that rejection costs minutes, not hours; Phase 4
protects the whole thing once it's actually being used. **If only two things happen next
session: the 2026 holdout run (0.3), then start the latent-effect measurement (1.1).**
