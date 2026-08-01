# NFL Prediction Model — Complete Technical Documentation

Audience: an LLM (or engineer) with no prior context on this codebase, needing a complete
mental model of how every piece works, why it exists, and how it rolls up into a final
score/prop prediction. Nothing is intentionally omitted — this includes dead ends,
rejected hypotheses, and investigated-but-never-built ideas, because this project's own
practice is to keep that history visible rather than erase it. If a section here looks
like it contradicts another, read both fully before assuming it's an error — several
findings are deliberately narrow (e.g. wind is rejected for total points but validated for
QB/WR yardage rates — different target variables, not a contradiction).

All paths are relative to the repo root: `/Users/brettscully/Desktop/sports-models/nfl`.
This project lives inside a multi-sport `sports-models/` workspace (siblings: `mlb/`,
`nhl/`, `nba/` scaffolded-only, `site/` planned) but is fully self-contained — own `.venv`,
own `data/` cache, no shared code with any sibling project.

---

## 0. One-paragraph summary

For a given matchup, the model estimates each team's offensive and defensive strength via
a recursive, opponent-adjusted EPA/play rating (Layer 1), converts the rating difference to
a point margin and total via a linear calibration fit on real historical margins. **As of
2026-07 (two review rounds), NEITHER margin NOR total blends with the market anymore** — a
full ATS%/O-U%/CRPS/signed-bias panel found both blends statistically indistinguishable from
a coin flip against the closing line (margin also carried a real negative bias market-alone
didn't have; total's own-model weight was additionally found to be declining over time in a
rolling-origin check — §4.3). Production now uses the market `spread_line`/`total_line`
directly (falling back to Layer 1's own calibration only when no line is published yet for a
game), with a validated QB-swap adjustment and a symmetric cornerback-injury adjustment
layered on top of margin (§5, §6 — both refit against this exact residual, since applying
coefficients validated against the OLD blended residual on top of the market number was
found to double-count real information the market already prices in, §6.1.1).
`market_blend.py`'s Ridge-blend machinery is kept as a generic utility (still used by
`validate_game_simulator.py`'s own historical comparison baseline) but is no longer live in
production for either quantity. A full player-props layer (Layer 2/3) projects
target/carry share, TD probability, yards, receptions, and full QB passing statlines for
every rostered skill player, with validated position-specific injury reallocation (a
ruled-out RB's carries redistribute to the backfield; a ruled-out WR/TE's targets do not),
a draft-capital rookie prior, and a Clay-projection fallback for players not yet covered by
draft data. A validated wind adjustment activates once real forecasts exist. An automated
weekly + gameday-refresh pipeline regenerates everything and publishes a shareable UI.

**Current validated performance** (margin, held-out TEST=2022-2025, walk-forward, no
lookahead): our own model alone MAE ≈10.0 / 63.8% straight-up; the real market alone MAE
≈9.49 / 67.5% SU — production margin is now the market number plus the QB-swap/CB
adjustments, not a blend of the two (§4.3). The entire live betting edge of the game-side
margin model lives in that adjustment layer. **Honest state as of review round 4**: the
CB-flagged subset's ATS% (§6.1.1) — first reported as 62.4% and treated as decided — was
caught being resubstitution (the coefficient was fit on data overlapping its own test
window); the walk-forward, honest out-of-sample version came back close (61.8%), but round 4
correctly showed that closeness doesn't corroborate the fit (ATS is magnitude-insensitive, so
walk-forward and resubstitution ATS are nearly the same statistic). A permutation test
extended to the full specification search (not just the winning configuration) put the real
number at only the ~95th percentile of the corrected null — right at the edge of conventional
significance, not the ~97.5th percentile a narrower test had suggested. Held provisionally:
real-looking, not yet proven as strongly as previously claimed, with the production
coefficient conservatively set (shrunk to the rolling-origin fold median) regardless of how
this resolves. Player props: target share corr 0.74, carry share corr 0.88 (real, strong
signal); TD-probability
calibration is an isotonic regression refit every pipeline run (§9.4), replacing an earlier
hand-fit linear map that was a recurring stale-coupling risk; yards/receptions carry
weak-but-real signal (0.5-3.7% MAE improvement over naive); QB passing yards similarly
weak-but-real; interceptions are close to pure noise. See §13 for the full ledger of
everything tested, kept, and rejected — it is long, because this project's discipline is to
test rigorously and report honestly even when the answer is "no effect," which is the
outcome roughly 80% of the time.

---

## 1. Architecture overview

```
raw NFL data (nfl_data_py)                                 (src/ingest/fetch.py)
  schedules · play-by-play · weekly rosters · injuries · snap counts
        │
        ├── silent partial-fetch guard: verifies every requested season is present in the
        │   result, retries on a genuine gap, raises immediately (no wasted retries) if only
        │   the newest not-yet-started season is missing — §12.1, a real bug once found here
        │
        ▼
our own weekly player-stat aggregation from raw PBP        (build_weekly_stats_from_pbp.py)
  (nfl_data_py's own weekly aggregate lags the season; PBP is the trusted source)
        │
        ▼
Layer 1: recursive opponent-adjusted EPA power ratings      (src/models/ratings.py)
        │
        ├── margin/total calibration (OLS, CALIBRATION_SEASONS)
        │
        ▼
MARGIN: real Vegas spread_line directly (Layer 1 calibration only as a fallback
        when no line is published yet) -- market blend REMOVED 2026-07, §4.3
TOTAL:  real Vegas total_line directly, same fallback rule -- market blend ALSO
        REMOVED 2026-07 (round 2, identical panel/kill-criterion as margin), §4.3
        │
        ▼
QB-swap adjustment (in-season + offseason bootstrap), refit against the MARGIN
above -- SWAP_B_MARKET in weekly_update.py     (qb_adjustment.py, predict_2026.py)
        │
        ▼
injury adjustment (symmetric CB Out flag; skill/OL dropped, not distinguishable
from 0 once market pricing is netted out), refit against the same residual
                                                              (injury_adjustment.py)
        │
        ▼
final margin/total prediction  →  predictions_{season}_wk{week}.parquet
        │
        ▼
Layer 2/3: player props
  ├── target/carry share (recency-weighted)                 (player_usage.py: ShareEngine)
  ├── TD rate, yards/touch, catch rate (Bayesian shrinkage)  (player_usage.py: TdRateEngine)
  ├── QB passing statline (completions/yards/TDs/INTs)       (qb_passing_stats.py)
  ├── injury/PUP/IR detection + position-specific reallocation (injury_reallocation.py)
  ├── rookie fallback (Clay projections, no NFL history)     (injury_reallocation.py)
  └── wind adjustment (QB YPA, WR/TE YPT)                    (weather_adjustment.py)
        │
        ▼
props_{season}_wk{week}.parquet  +  full per-game statlines printed
        │
        ▼
automated orchestration                                      (src/pipeline/weekly_update.py)
  ├── Tuesday early-week run
  └── Sunday gameday-refresh run (catches Friday final injury report + settled weather)
        │
        ▼
UI generation + Claude Artifact publish                      (build_predictions_page.py)
```

Every adjustment layer in this system was validated the same way: build the signal
walk-forward (only data strictly before the target game), predict the real outcome on a
held-out test window never touched during tuning, and check for a genuine MAE/significance
improvement over the current production baseline — not a standalone correlation. This is
why §13 lists roughly 4x as many rejected ideas as kept ones; that ratio is a sign the
discipline is being followed, not that the project is failing.

---

## 2. Data ingestion

### 2.1 `src/utils/paths.py`
`PROJECT_ROOT`, `DATA_RAW`, `DATA_PROCESSED` path constants. No business logic.

### 2.2 `src/ingest/fetch.py`
Thin wrapper around `nfl_data_py`, one `fetch_*` function per endpoint (`fetch_schedules`,
`fetch_pbp`, `fetch_weekly_rosters`, `fetch_weekly_data` [unused in production — see below],
`fetch_snap_counts`, `fetch_injuries`, `fetch_ngs`). Each is idempotent via `_cached()`
(skips the network call if the parquet already exists, unless `force=True`).

**`_cached()`'s completeness guard** (added after a real incident — §12.1): `nfl_data_py`'s
multi-season loaders can silently drop a season from the middle of a requested range with
**no exception at all**. Confirmed directly: an 11-season pbp pull once came back missing
2017 and 2023 entirely, no error, no warning distinguishable from a normal run. `_cached()`
now checks every requested season is actually present in the result; if a *non-trailing*
season is missing, it retries (up to `MAX_FETCH_RETRIES`, transient issues resolve); if
*only* the newest requested season is missing, it raises immediately without wasting
retries (that's the normal "season hasn't started yet" case — the caller's own fallback,
`_fetch_with_fallback` in `weekly_update.py`, handles it by trimming the range).

`import_weekly_data()` (nfl_data_py's own weekly player-stat aggregate) is fetched by
`fetch_weekly_data()` but **deliberately not used anywhere in production** — it lags real-
time by design and has not published a 2025+ release. All weekly player stats are derived
from our own trusted PBP data instead (§2.3). Confirmed via a 2026-07 housekeeping audit that
nothing calls `fetch_weekly_data()`/`fetch_ngs()` outside `fetch.py` itself; `fetch_all()`
(only invoked from this module's own `__main__`, a manual full-cache-rebuild utility) no
longer includes `fetch_weekly_data()` in its default set for this reason — the function
itself is kept (harmless, occasionally useful for an ad-hoc comparison), just not spent on
by default.

### 2.3 `src/ingest/build_weekly_stats_from_pbp.py`
`build_weekly_stats(pbp, rosters)` aggregates PBP into one row per (season, week, team,
player): `targets, receptions, receiving_yards, receiving_tds` (receiver side),
`carries, rushing_yards, rushing_tds` (rusher side), and `attempts, passing_epa,
completions, passing_yards, passing_tds, interceptions` (passer side — the passing-stat
columns were added later, specifically to support the QB passing-statline feature, §5.2;
the original version only had `attempts`/`passing_epa`). Position and display name are
joined in from `weekly_rosters`. Drop-in-compatible schema with the legacy
`weekly_player_stats_2016_2024.parquet` this replaced.

### 2.4 `src/ingest/parse_clay_pdf.py`
One-time PDF extraction (pdfplumber + regex) of Mike Clay's 2026 ESPN Projection Guide —
the only source in this project with real 2026-offseason information (incoming QB starters,
rookie season-total projections) that `nfl_data_py` structurally cannot have for an
unplayed season. Produces `clay_2026_schedule.parquet` (superseded once
`nfl_data_py.import_schedules([2026])` was confirmed to have the real schedule natively —
verified to match Clay's Week 1 data exactly) and `clay_2026_player_projections.parquet`
(still load-bearing — see §5.5, §6.5). `TEAM_NAME_TO_CODE` / `OPP_CODE_FIXES` handle 5 team-
code mismatches (CLV/BLT/ARZ/LAR/HST → CLE/BAL/ARI/LA/HOU) between Clay's naming and
nflverse's.

---

## 3. Layer 1 — team power ratings (`src/models/ratings.py`)

### 3.1 `PowerRatingEngine`
Each team carries an **offensive rating** (EPA/play generated, opponent-adjusted) and
**defensive rating** (EPA/play allowed). Additive prediction model:

```
home_net = off_rating[home] + def_rating[away]
away_net = off_rating[away] + def_rating[home]
rating_diff = home_net - away_net
```

`update()` is a plain EMA per side: `new = (1-alpha)*old + alpha*target`, where `target`
subtracts out the opponent's current rating from the observed EPA/play — this is what makes
it opponent-*adjusted*, not just a rolling average. Tuned defaults: `alpha=0.06`,
`off_shrink=0.20`, `def_shrink=0.50` (fraction of rating **kept**, i.e. `1-shrink`, applied
at each season boundary — offense is more persistent than defense, hence the smaller
shrink). These were grid-searched in `tune.py` (§10.1) and re-validated later (§13) as
still near-optimal even when re-tested specifically for Week 1/early-season performance.

A subtle, once-real bug: `nets()` must **add** the opponent's defensive rating, not
subtract it (`home_net = off[home] + def[away]`) — an earlier version used subtraction and
was caught via an A/B correlation test (0.19 → 0.40 once fixed) before any user-facing
report was generated.

### 3.2 `build_dataset(pbp_seasons, schedule_seasons=None)`
Builds the full historical matchup table from PBP + schedule, aligning `team_a`/`team_b` to
`home`/`away`. Accepts *independent* pbp/schedule season ranges (added because schedules
can be published for a season before any real plays exist for it yet — passing the same
range for both would try to read a pbp parquet file that was never written).

### 3.3 Calibration windows — two different ones, deliberately
- **Historical backtesting** (`backtest.py`, `tune.py`, `predict.py`): `TRAIN={2018-2021}`,
  `TEST={2022-2025}`, burn-in `{2016,2017}`. This split exists so a disjoint holdout can
  honestly validate anything tuned on TRAIN.
- **Forward-looking 2026 production** (`predict_2026.py`, `weekly_update.py`):
  `CALIBRATION_SEASONS={2022,2023,2024,2025}` — deliberately *not* the historical TRAIN
  window, because 2019-2020 show badly anomalous near-zero home-field advantage (empty/
  reduced-crowd COVID seasons) and 2018/2020 run 3-4 points hot on total scoring. A
  calibration fit on the historical TRAIN window systematically underpredicts margin and
  overpredicts totals when applied to a normal season — confirmed via a per-season bias
  check showing -0.9 to -1.8 pt margin bias and +0.7 to +3.0 pt total bias in every one of
  2022-2025. There's no leakage concern in using the most recent seasons for an actual
  forward forecast; 2026 will resemble 2022-2025 far more than 2018-2021.

### 3.4 EPA signal-quality investigation (review #2.4, 2026-07) — TESTED, MOSTLY NULL, ONE REAL REJECTION

Four candidate refinements to the raw all-play EPA/play input, each tested independently via
the same `PowerRatingEngine` (only the EPA definition changed), TRAIN=2018-2021/TEST=2022-2025:

- **Early-down EPA (1st/2nd down only)**: MAE 10.055 vs. baseline 10.000, straight-up
  unchanged (63.75%), p=0.366 — clean null, no improvement.
- **Winsorized EPA (clip at TRAIN-fit 1st/99th percentile, [-4.663, 3.740])**: MAE 10.041 vs.
  baseline 10.000, p=0.0014 — **statistically significant, but HARMFUL**, the opposite
  direction from the review's "pure variance reduction" expectation. Likely explanation: the
  clipped outlier plays (long touchdowns, turnovers) carry real signal about a team's
  explosiveness/big-play-proneness that clipping discards rather than noise it removes.
- **Turnover-neutralized EPA** (fumbles credited at the pooled 50/50 lost-vs-recovered
  average EPA, `-4.963` and `-0.743` respectively → neutral `-2.853`, n=5,321 fumbles): MAE
  9.998 vs. baseline 10.000 (p=0.927, not significant — a clean null on the primary metric),
  though straight-up ticked up (64.77% vs. 63.75%) — not chased further given the MAE null.
- **Success rate as an additional co-input**: a real, self-caught bug during this
  investigation — the first pass used the raw SAME-GAME realized success rate as a
  regressor for that same game's margin (lookahead leakage), which produced an implausible
  +80.9 coefficient and an apparent ~18% MAE "improvement." Once fixed to use a genuine
  walk-forward, pregame success-rate rating (run through the identical `PowerRatingEngine`
  mechanism as EPA itself, not the raw realized value), the result reverted to a clean null
  on MAE (10.006 vs. 10.000). **Lesson**: exactly the same class of error as the `np.polyfit`
  pattern in §12.2 — a plausible-looking, large effect size is itself a reason to check for
  leakage before trusting it, not after.

None of the four candidates are wired into the live pipeline; the current all-play,
unwinsorized, observed-turnover EPA/play remains the shipped signal.

### 3.5 Rating-update mechanism investigation (review #2.5, 2026-07) — TESTED, REJECTED

Two candidate refinements to `PowerRatingEngine`'s update rule, each tested via a subclassed
engine, TRAIN=2018-2021/TEST=2022-2025:

- **Declining-gain EMA** (`alpha_t = k/(k + games_played_this_season)`, replacing the fixed
  `alpha=0.06`): **decisively rejected**. A hard reset of the games-played counter at each
  season boundary makes `alpha=1.0` for a team's very first game of a new season — its entire
  rating gets overwritten by one noisy game's EPA, discarding the season-shrunk prior
  entirely. Every `k` tested (2, 4, 8, 16) was significantly worse than baseline on both
  overall TEST MAE and the specific weeks-1-6 window this was meant to help (p=0.0013 to
  p<0.0001). A softer season-boundary reset (carrying over a baseline "confidence" instead of
  a hard reset to 0) closed most of the gap as the carryover grew, but never beat baseline —
  best case (`k=8`, `season_reset_games=16`) was statistically indistinguishable, not better
  (p=0.156). The existing fixed-alpha + season-shrink approach already strikes a reasonable
  balance; this project's own data doesn't support a declining-gain refinement of it.
- **Market-anchored preseason prior** (ridge regression of historical `spread_line` on
  home/away team dummies, walk-forward — using only strictly-prior seasons — as the season-
  opening rating instead of shrinking toward zero): also null. Full replacement
  (`blend_weight=1.0`, the literal reading of the review's suggestion) was numerically worse
  across the board, including in the specific weeks-1-6 window it targeted (MAE 10.39 vs.
  baseline's 10.07), though not significant (p=0.43). Partial blends (0.5, 0.25) were
  statistically indistinguishable from baseline in either direction (p=0.52, p=0.20) — closer
  to a no-op than an improvement as the market weight shrinks toward the current behavior.

Neither candidate is wired into the live pipeline; `PowerRatingEngine`'s fixed
`alpha=0.06`/`off_shrink=0.20`/`def_shrink=0.50` remain unchanged.

---

## 4. Market blend (`src/models/market_blend.py`)

**The single highest-value discovery this project made.** `nfl_data_py`'s schedule export
already contains real historical Vegas closing lines — `spread_line`, `total_line`,
`home_moneyline`, `away_moneyline`, `home_spread_odds`, `away_spread_odds`, `over_odds`,
`under_odds` — 100% coverage 2016-2025, and lines are published for upcoming games well
before kickoff (confirmed available for 2026 Week 1 during the *offseason*, before any 2026
games were played). This sat completely unused for most of the project's history; the
assumption had been that using real market data would require a new paid data source.

**Sign convention** (verified empirically, not assumed): `spread_line` is signed to match
`home_score - away_score` directly — positive means home favored by that many points, no
negation needed. Regressing actual margin on `spread_line` gives slope ≈1.04, intercept
≈0, corr 0.446; using it directly as a margin predictor gives MAE 9.79 / 66.3% straight-up
across 2016-2025 — matches known real-world Vegas performance almost exactly, confirming
the data is genuine and correctly interpreted.

### 4.1 Baselines (TEST=2022-2025, margin)
| Approach | MAE | Straight-up |
|---|---|---|
| Our model alone | 9.9997 | 63.75% |
| Market alone | 9.4945 | 67.53% |
| Fixed 50/50 blend | 9.6477 | 67.07% |
| OLS-optimal fixed blend | 9.5497 | 67.16% |

### 4.2 The multicollinearity trap, and why RidgeCV is used instead of plain OLS
Our margin and the market spread correlate at ~0.83-0.84 — strong enough that plain OLS
blend weights are **numerically unstable**: fitting on different historical windows swings
the "our model" coefficient from -0.06 to +0.17 depending on exactly which years are
included (2022-2025 alone even gives a *negative* weight on our own model, which doesn't
generalize and reflects overfitting to a small, collinear sample). `RidgeCV` (5-fold,
`alphas=np.logspace(-1,4,40)`) fit on a **dynamically-growing window** (2018 through the
season *before* the one being predicted — grows every year, excludes the target season)
gives consistent, sensible, always-positive weights: **margin ≈0.02-0.17 weight on our
model, ≈0.98-0.88 on market; total ≈0.11-0.13 on our model, ≈0.91-1.01 on market** (exact
values shift slightly each pipeline run as the fitting window grows — this is intentional,
not noise). `fit_market_blend()` / `apply_blend()` are the two functions; `apply_blend`
degrades gracefully (returns the un-blended prediction) if no market line exists yet for a
specific game.

An nfelo-inspired **dynamic per-team-error-weighted blend** (blend weight toward market
scales with how poorly our model has tracked that *specific team* recently, via an EWMA of
tracked absolute error) was also built and tested — it **underperformed** the simple fixed
Ridge blend (MAE 9.65 vs 9.55). Not shipped; the extra sophistication didn't earn its keep
here, though the general idea has real external support (nfelo's own writeups).

### 4.3 Wired into `weekly_update.py` — BOTH MARGIN AND TOTAL BLENDS REMOVED (2026-07)

**Margin (round 1).** A third-party review flagged that §4.1's baseline table already
implies the blend adds little-to-nothing for margin, and that grading everything on MAE
(not ATS%, which is what a betting model actually needs) could be hiding real harm. Built
`src/models/scoring.py` (`ats_win_rate`, `ou_win_rate`, `signed_bias`, `crps_gaussian`,
`brier_score`, `log_loss_gaussian`, all with bootstrap CIs) and re-ran the comparison with
the **exact growing-window methodology `weekly_update.py` actually uses** (not a single
fixed-window approximation). Full panel, TEST=2022-2025, n=1058:

| | MAE | ATS% vs. closing line | Signed bias (95% CI) | CRPS |
|---|---|---|---|---|
| Our model alone | 10.000 | 49.05% [46.1, 52.1] | −1.240 [−1.97, −0.47] | 7.234 |
| Market alone | 9.494 | — (trivial) | −0.574 [−1.27, +0.20] | 6.918 |
| Blend (old prod) | 9.533 | **49.62% [46.6, 52.6]** | **−0.986 [−1.70, −0.22]** | 6.942 |

Two things MAE alone hid: (1) the blend's ATS win rate is statistically indistinguishable
from a coin flip and **below the ~52.4% breakeven** against standard −110 vig — betting its
disagreement with the closing line would not have been profitable; (2) the blend carries a
**real, statistically distinguishable negative bias** (CI excludes 0) that market-alone does
not — a direct, mechanical consequence of blending in our own model's own real bias.

**Decision (confirmed with the user before implementing): for MARGIN, the blend is
removed.** `weekly_update.py` now uses the market `spread_line` directly as the margin base,
falling back to Layer 1's own calibration only when no line is published yet for a game.
The validated QB-swap/injury adjustments (§5, §6 — both refit against this exact residual,
see §6.1.1) are layered on top, same as before.

**Total (round 2).** Round 1 kept the total blend on MAE + a single p-value alone (MAE 10.20
vs. 10.19, p=0.43, "no evidence of harm") — the exact same thin evidence state that hid the
margin blend's harm until the full panel above was run, and totals carry a *larger*
own-model weight (0.11–0.13) than margin's pre-removal 0.02–0.17, so the asymmetry in rigor
ran exactly the wrong way. Ran the identical panel on totals
(`src/models/validate_market_blend_totals.py`), same growing-window methodology,
TEST=2022-2025, n=1087:

| | MAE | O/U% vs. closing line | Signed bias (95% CI) | CRPS |
|---|---|---|---|---|
| Our model alone | 10.652 | 49.95% [47.0, 52.9] | +1.681 [+0.87, +2.49] | 7.563 |
| Market alone | 10.189 | — (trivial) | −0.720 [−1.50, +0.09] | 7.284 |
| Blend (old prod) | 10.201 | **50.14% [47.2, 53.0]** | −0.326 [−1.11, +0.48] | 7.289 |

**Correction (review round 3, #4): the O/U% test above was not decisive, and calling it "the
same decisive failure pattern that killed the margin blend" — as this section originally
said — overstated what the number shows.** Before treating an ATS/O-U test as a kill
criterion, check whether it had the *statistical power* to distinguish real signal from
noise in the first place: compute the component's disagreement-with-the-line SD and the
resulting best-case win rate at a *perfectly* informative signal of that size. Here:
SD(blended_pred − total_line) ≈ 0.31 points (blend weight ~0.06–0.17, real disagreement SD
~3.2–3.3 points — 0.1 × 3.25 ≈ 0.31), and O/U% moves roughly 2%/point near the line, so even
a perfectly informative 0.31-point signal would only move O/U% to **~50.6%**. The observed CI
[47.2%, 53.0%] comfortably contains that best-case value — **the test could not have told a
real small signal apart from pure noise**, which is a different (and much weaker) finding
than "decisively failed."

The decision to remove the total blend still stands, but on different, real grounds: (1) no
measurable contribution to MAE/CRPS either — the blend (10.201) and market-alone (10.189) are
statistically indistinguishable there too, a genuine null; (2) a rolling-origin fold check
found the blend beats market-alone on MAE in only **2 of 4 folds** (a coin-flip rate, added
2026-07 for parity with how margin's own removal was actually decided — §10.5's demo used
this exact statistic, which round 2's total check omitted, checking only sign consistency of
the weight, 1.000); (3) the weight itself is declining across folds (0.158→0.168→0.062→0.067)
rather than stable. **Recorded honestly: removed for parsimony — no measurable benefit on any
metric that had real power to detect one — not because the O/U% panel decisively failed it**,
which is what happened to margin (the margin removal rests on real, higher-power evidence:
signed bias with a CI excluding zero, and beating market-alone in only 2 of 8 rolling-origin
folds — both genuinely powered tests, unlike total's O/U% check).

**General rule for future use of this kill criterion** (review round 3, #4): before treating
an ATS/O-U test as decisive for any low-weight component, compute its disagreement-with-the-
line SD and the resulting best-case win rate. If that best-case sits inside the observed CI,
the test had no power, and the decision must rest on bias/CRPS/fold-consistency instead — not
the ATS/O-U number, which will be uninformative by construction at that weight. When ATS *is*
the right test (enough weight/disagreement magnitude to have real power, as it did for the
margin blend and the CB adjustment, §6.1.1), condition it on disagreement magnitude where
possible — pooling over games with a tiny disagreement guarantees something close to a coin
flip regardless of whether real signal exists.

**Decision: for TOTAL, the blend is also removed**, per the corrected reasoning above.
`weekly_update.py` now uses the market `total_line` directly, falling back to Layer 1's own
total calibration only when no line is published yet. `market_blend.py`/`fit_market_blend`/
`apply_blend` are kept as generic, still-correct utilities (`validate_game_simulator.py` uses
them for its own historical comparison baseline) but are no longer used by any part of the
live production pipeline for either margin or total. This is a real, if uncomfortable,
example of §12.5's principle in the opposite direction: sometimes the walk-forward-honest
answer is "stop doing the thing you built," not "refit its coefficient" — and it turned out
to apply twice, on two different kinds of evidence.

---

## 5. QB layer

### 5.1 `src/models/qb_adjustment.py` — in-season swap adjustment
Diagnosis: Layer 1's team rating updates slowly (`alpha=0.06`, deliberately sticky), so
when a team's starting QB changes (injury, benching, bye-week return), the very next game
is predicted using a rating still reflecting the *old* starter. Measured: MAE on QB-change
games 10.70 vs 9.79 for no-change games, with a systematic +4.2pt bias when the *away*
team's QB changes.

**Fix**: `QbRatingEngine` — a per-QB EPA/dropback rating (`alpha=0.15`, `season_shrink=0.15`
— gentler than team-level shrink, since QB skill is more persistent than team context),
gated on `MIN_ATTEMPTS=5` so mop-up snaps don't move it. Ground-truth starters come from
`schedules.home_qb_id`/`away_qb_id` (not snap-count heuristics). A one-game **swap delta**
= `new_starter_rating - previous_starter_rating` is applied only on the first game after a
detected change (zero otherwise — Layer 1's own recursive update catches up on its own
after that). `SWAP_B_LAYER1=6.616` (points per unit of rating gap) was the coefficient as
originally validated — fit against the **pure Layer-1 residual**. `weekly_update.py` applied
it on top of the market-blended margin at the time, though (§4.3's original finding), so it
was refit against that blend residual instead — **`SWAP_B_MARKET`=2.970**, roughly half the
original value (see §6.1.1 for the full refit methodology and honest caveats). The margin
blend itself was subsequently removed entirely (§4.3, §0/§1) — `SWAP_B_MARKET` is now applied
on top of the market `spread_line` directly rather than a blended prediction, but it was
refit against the blend residual specifically because at the time of that refit the two were
numerically close (the blend was 88-98% the closing line), so the value carries over as the
correct one for the current market-spread-line-based production path too; it has not needed
a further refit since. `predict_2026.py`'s own standalone `SWAP_B_LAYER1=6.616` is
intentionally left unchanged — that script never blends with or uses the market at all, so
the original Layer-1-residual value is still the correct one *there*; do not "sync" the two
(named for their residual basis specifically to prevent that, since a bare shared `SWAP_B`
name across two files is the same trap class as the `np.polyfit` arg-order bug — §12.2).

`build_starter_sequence(schedules)` computes `qb_changed` via `groupby("team")["qb_id"
].shift(1)` across the *full* sorted season/week sequence — no season-boundary reset —
meaning a team's Week 1 game's "previous starter" is literally whoever started their last
game of the prior season. This means the swap-delta mechanism **already covers offseason
transitions**, not just in-season ones, without any special-casing.

### 5.2 Offseason QB-swap bootstrap (`predict_2026.py`, ported into `weekly_update.py`)
`nfl_data_py`'s schedule has **no starter QB for any unplayed game** (confirmed directly:
`home_qb_id`/`away_qb_id` are NaN for every 2026 Week 1 row) — so a team whose real starter
changed via trade/free-agency/retirement is invisible to every engine here until that team
actually plays a game. Clay's PDF (§2.4) is the only source with that information.
`find_qb_changes()` compares each team's 2025 primary starter (by **games started**, the
correct convention — sorting by the `games` column alone is ambiguous, since a benched
backup stays on the active roster all season too) against Clay's 2026-projected starter
(by **pass attempts**, which correctly discriminates a true starter (~450-550+ projected
attempts) from a handcuff (<100)). `SWAP_B` is reused for the point-value adjustment.

This whole block is scoped tightly: only fires when `pred_season==2026 and pred_week==1
and clay_path.exists()` — it naturally stops mattering the moment Week 1 is actually played
(real starter data takes over), and in a future season with no new Clay extraction, it
silently no-ops rather than guessing. **Validated specifically for the offseason case**
(not just assumed to transfer from the in-season fit): 47 historical Week-1 offseason-swap
games, MAE 7.55 → 7.39 with the adjustment applied — smaller effect than in-season swaps
(11.41 → 11.29) but real and correctly signed.

### 5.3 `src/models/qb_passing_stats.py` — full passing statline (added 2026-07)
Before this, the props pipeline had **no QB passing projection at all** — a QB's only
"prop" was his own rushing production. `build_qb_passing_engines(qb_weeks)` builds four
`TdRateEngine` instances (Bayesian/Laplace shrinkage, reused generically from
`player_usage.py`) over `completions`, `passing_yards`, `passing_tds`, `interceptions`, all
keyed per `attempts`. Prior weights, swept out-of-sample (TRAIN=2018-2021/TEST=2022-2025,
QB-weeks with ≥5 attempts):

| Stat | prior_weight | MAE (adjusted vs naive) |
|---|---|---|
| completions | 300 | 2.371 vs 2.424 (~2.2% better) |
| passing_yards | 150 | 43.838 vs 44.710 (~2.0% better) |
| passing_tds | 150 | 0.860 vs 0.868 (~0.9% better) |
| interceptions | 300 | 0.663 vs 0.664 (~0.15% better — essentially noise) |

Honest framing (matches the yards/receptions pattern in §6.2): completions/yards carry the
same weak-but-real signal documented for receiving/rushing yards. TD rate weaker still.
Interceptions are close to pure noise per-QB — consistent with turnover luck being non-
predictive everywhere else it was tested in this project (§13) — included for statline
completeness, not because it's a differentiated skill estimate.

`presumed_starter_qb_id(team, current_roster, qb_weeks, override_name=None,
override_name_to_id=None)` — best-effort starter identification: whoever on the current
roster had the most real attempts for that team in the most recent season on record.
`override_name`/`override_name_to_id` let the §5.2 bootstrap force a known offseason swap
to win over pure history (a brand-new starter has no attempts for the new team yet to rank
by). `project_passing_stats(qb_id, proj_attempts, engines)` returns the full rounded
statline dict.

---

## 6. Injury layer — two distinct systems for two distinct purposes

### 6.1 `src/models/injury_adjustment.py` — game-level margin adjustment
Four validated flags: **skill (WR/TE/RB), offensive line (T/G/C), cornerback**. Two ID
strategies depending on data availability:
- WR/TE/RB: identified by pregame target/carry share (Layer 2 output), joined to the real
  injury report by `gsis_id` (reliable direct match).
- O-line and CB: box scores don't cover linemen/corners and fantasy ID crosswalks (even
  `weekly_rosters`' own `pfr_id`) cover O-line at under 1% coverage — so these are joined
  by **normalized name** within the same (season, week, team), a small, disambiguated
  search space rather than a risky global fuzzy match.

In every case, share/snap-pct is **forward-filled** across weeks a player didn't play,
before picking the "presumed starter" — necessary because a genuinely Out player has zero
box-score stats that week, so naively picking "top performer among players who played" would
circularly exclude exactly the injured players being detected.

**CB uses the top THREE** by snap share, not two — modern nickel packages make the 3rd
corner a near-full-time starter (tested top-2 through top-5: top-3 was clear best, MAE
9.948→9.919 over top-2; top-4/5 dilute the signal enough the home-side coefficient's sign
becomes noise). Only the **"Out"** designation is used — "Doubtful"/"Questionable" were both
tested and made things worse (Doubtful broke the home-CB coefficient's sign; Questionable
pushed test MAE above the unadjusted baseline). Positions tested and **rejected**:
DE/EDGE (sign flips train-to-test), safety, linebacker (no improvement / backwards sign).

Final joint model (`JOINT_COEFS`, fit 2018-2021, validated 2022-2025):
```
margin_adjustment = -0.382 + 1.434*away_skill_out + 1.447*away_ol_out
                          - 1.217*home_cb_out    + 3.184*away_cb_out
```
Test MAE 10.000 → 9.906. CB is by far the largest effect and the only one working in *both*
directions — losing a top corner is the single most exploitable injury signal found in this
project. A separate **`JOINT_COEFS_FORWARD`** exists specifically for `predict_2026.py` and
`weekly_update.py`. `JOINT_COEFS` stays frozen at its original historical fit for backtesting
honesty (`predict.py`); `JOINT_COEFS_FORWARD` is what the forward-looking paths use, and it
has gone through 3 revisions — see §6.1.1.

### 6.1.1 `JOINT_COEFS_FORWARD` refit history (v1 → v3, 2026-07)

**v1** (pooled 8-season era-consistent-residual fit, against the pure **Layer-1** residual):
since refitting directly against the newer CALIBRATION_SEASONS-based base prediction moved
every coefficient by more than the ~150-300 flagged-game sample could justify as pure
signal, each game's residual was scored against its own era's well-fit calibration, then
pooled, giving a lower-variance middle-ground estimate:
```
margin_adjustment = -0.339 + 2.562*away_skill_out + 1.195*away_ol_out
                          - 3.430*home_cb_out     + 3.792*away_cb_out
```

**v2** (fit against the market-**blend** residual instead): a third-party review (2026-07)
flagged that `weekly_update.py` applies this adjustment (and the QB swap_delta, §5.1) ON TOP
OF the already market-blended prediction, while both were validated against the pure Layer-1
residual — likely double-counting information the market already prices in. Refitting
against `actual_margin - blended_pred` (same era-consistent pooling, n=2127) confirmed it:
every coefficient shrank, and `away_skill_flag`/`away_ol_flag` were no longer individually
distinguishable from zero (t=+1.64, t=+1.14) — only the CB terms survived (home_cb t=-2.43,
away_cb t=+3.27):
```
margin_adjustment = -0.077 + 0.000*away_skill_out + 0.000*away_ol_out
                          - 2.576*home_cb_out     + 3.351*away_cb_out
```
Honest empirical caveat: a strict walk-forward check (fit 2018-2021 only, score 2022-2025
blind) showed v1 and v2 statistically indistinguishable on point-MAE (9.459 vs 9.493) — this
fix is not primarily an accuracy win. It matters because v1's larger coefficients overstate
the true edge size, which is what actually gets used for bet sizing (an overstated edge
risks real overconfidence in Kelly-style staking, even when MAE can't see the difference —
exactly the "wrong objective function" critique the same review raised, §12.5-adjacent).

**v3** (symmetric CB constraint, review §1.6): v2 left `home_cb_flag` (-2.576) and
`away_cb_flag` (+3.351) as independent, differently-sized coefficients. Reparameterizing as
ONE coefficient on `(away_cb_flag - home_cb_flag)` — forcing home/away effects equal and
opposite — roughly doubles the effective sample for that coefficient; the pooled t-stat
jumps from -2.43/+3.27 (independent) to +4.01 (symmetric). A strict walk-forward check
confirms it generalizes better, not just fits better in-sample: on the CB-flagged holdout
subset (n=183), MAE 9.968→9.836 and signed bias +1.596→+0.542 (a real improvement in
calibration, not just noise). **Current production value:**
```
margin_adjustment = -0.018 + 0.000*away_skill_out + 0.000*away_ol_out
                          - 2.977*home_cb_out     + 2.977*away_cb_out
```
The QB swap_delta coefficient was refit jointly alongside this (§5.1): **`SWAP_B_MARKET=2.970`**
(from the original 6.616), t=+1.52 — kept despite sitting just under conventional
significance, since a stale-rating-catches-up mechanism is well-established and not an
ad-hoc multi-way fit like the injury flags were.

**v3 closed the gap the round-2 review flagged, but round 3 found the closure itself was
methodologically flawed** (§2.1/§2.2 there, then §1 of round 3): since the margin market
blend was removed, this CB/QB-swap layer is the *entire* live betting edge of the game-side
margin model, and it had only ever been graded on MAE/signed bias, never the full
ATS%-with-bootstrap-CI panel that decided the blend's own fate — nor run through the
rolling-origin CV harness (§10.5) the way the market blend was, despite being the survivor
of a wide search (top-2/3/4/5 corners, 3 injury-report designations, several position
candidates, three residual bases) and therefore exactly the kind of coefficient §13.0's
multiple-comparisons caveat should worry about most. `src/models/validate_adjustment_layer.py`
(added 2026-07, extended 2026-07) closed the measurement gap, but the first pass mis-scored
what it measured:

- **Full ATS panel, CB-flagged subset, RESUBSTITUTION** (TEST=2022-2025, n=186): ATS%=62.4%,
  95% CI [55.4%, 68.8%]. **This number is in-sample and was originally reported as if it
  weren't.** It uses `JOINT_COEFS_FORWARD`'s pooled 2018-2025 coefficient (2.977) — fit on
  data that includes the very 2022-2025 games being scored. That's resubstitution, not an
  honest out-of-sample test, and the CI was computed as though it were one.
- **Full ATS panel, CB-flagged subset, WALK-FORWARD**: for each TEST season, score using ONLY
  the coefficient fit on strictly-prior seasons (the rolling-origin fold coefficients below),
  pooled across all 4 folds. **ATS%=61.8%, 95% CI [54.8%, 68.3%]** — nearly identical to the
  resubstitution number. **Round 3 read this as "a genuine, if narrow, vindication of the
  underlying edge's direction and rough size" — round 4 correctly identified that this
  overclaims.** ATS is a sign-of-disagreement metric: for a single-signed adjustment on top of
  the market line, it is *entirely* magnitude-independent, and every rolling-origin fold
  coefficient is positive — so the bet direction is identical in every fold regardless of
  magnitude. Walk-forward and resubstitution ATS are very nearly the *same statistic* (the raw
  win rate of "bet against the team missing a top-3 corner"). Their closeness is evidence the
  metric can't see the contamination that was actually wrong with the resubstitution number,
  not evidence the in-sample fit was sound. Only the permutation test below can actually speak
  to that.
- **Permutation test, single configuration** (shuffle each team's real CB-out flags across its
  own played weeks *within season*, preserving its real annual flag count, re-run the identical
  pooled-fit-then-score procedure 1,000 times): null mean 54.3%, std 3.9%. Resubstitution ATS
  (62.4%) sits at the 97.7th percentile; walk-forward ATS (61.8%) at the 97.5th. **Round 3
  reported this as settling the forking-paths question — it does not.** This null only prices
  in "fit and scored on the same games." It does not price in the *specification search* that
  chose this exact configuration (top-3 corners, Out-only, symmetric) out of the family actually
  explored (top-2/3/4/5 corners × 3 designations × several position candidates × 3 residual
  bases × the symmetry constraint). Back-of-envelope: the expected *maximum* of n draws from
  this null is ~60.3% at n=10, ~61.1% at n=15 — uncomfortably close to the observed 61.8-62.4%.
- **Permutation test, extended to the full specification search** (review round 4, #1, the
  decisive correction): for each of 300 shuffles, re-ran a 12-config grid search (corner-count
  ∈ {2,3,4,5} × designation ∈ {Out, Out+Doubtful, Out+Doubtful+Questionable}, symmetric
  constraint held fixed since that was a one-time architectural decision, not an ongoing
  sweep), selected the winner by |t-stat| — the same kind of criterion that historically chose
  top-3-over-top-2/4/5 and symmetric-over-independent — then scored *that shuffle's own
  winning configuration's* ATS. This is the null of "the best result the whole procedure finds
  on noise," the correct reference for a number that survived the whole procedure. **Result:
  null mean 55.3%, std 3.8%, 95th percentile 61.8%. The real walk-forward ATS (61.8%) sits at
  almost exactly the 95th percentile — right at the edge of conventional significance, not
  comfortably in the tail.** This is a real, substantial downgrade from the single-configuration
  read, and it lands almost exactly where the review's back-of-envelope arithmetic predicted.
- **Null-center diagnostic** (review round 4, #2): re-ran the single-configuration permutation
  with walk-forward (not resubstitution) fitting — null mean moved from 54.3% to **52.4%**,
  closer to 50% but not fully there. The resubstitution-advantage explanation is *confirmed as
  a real, partial contributor* (the null moved the expected direction) but does not fully
  explain the off-center null on its own. Checked directly for a residual cause: real,
  substantial variation in CB-flag incidence exists both by team (SF flagged 25 team-weeks
  across the dataset vs. many teams far fewer) and by season (3.1%-8.9%) — consistent with,
  though not proof of, a residual non-random-selection effect beyond pure resubstitution. Not
  fully pinned down this round; flagged as open.
- **Decision (v4, current production, unchanged by the above): `JOINT_COEFS_FORWARD`'s CB
  coefficient stays shrunk from ±2.977 to ±2.446** (the rolling-origin fold median). This
  precaution was about the *point-estimate coefficient's magnitude* specifically (every honest
  fold estimate sits below the pooled value) and is unaffected by the ATS/permutation
  re-analysis above — see §6.1.1's staking note below for why ATS and the coefficient are not
  interchangeable evidence for this purpose.
- **Snap-share-rank discriminator** (CB-flagged TEST games, split by the flagged corner's
  rank): rank 0/CB1 n=37 (MAE 11.19, bias −0.79), rank 1/CB2 n=52 (MAE 10.43, bias +0.64),
  rank 2/CB3 n=103 (MAE 8.99, bias −0.57) — no clean "concentrated at CB1, tapering at
  CB2/CB3" pattern the review's hypothesized mechanism would predict; two-thirds of flags
  fire on the *third* corner, and the effect doesn't concentrate on the first. Genuinely weak
  evidence against the stated "cornerback quality" mechanism, not merely inconclusive.
- **Lookahead audit, confirmed clean**: checked directly (not just re-read the code) whether
  the top-3-CB-by-snap-share ranking (`build_presumed_starters_by_name`) could be using
  full-season data rather than season-to-date — a ranking built from full-season snaps would
  manufacture exactly this kind of implausibly strong, mechanism-free result (a corner hurt
  early ranks low and never gets flagged; one who played all year ranks high). Truncated
  `snap_counts` to weeks ≤10 of a real season and compared the resulting week-10 ranks/shares
  against the same computation run on the full season: **zero mismatches across 96
  team-week-player rows.** The ranking is genuinely season-to-date; this specific lookahead
  concern is ruled out, not just asserted clean.

**Staking note (review round 4, #3): use the coefficient's implied edge, not the observed
ATS, for sizing.** The shrink from ±2.977 to ±2.446 was explicitly done to avoid overstating
edge for Kelly-style sizing — but ATS is magnitude-insensitive, so the shrink couldn't and
didn't move the ATS a bettor would actually size from (confirmed: walk-forward ATS is
unchanged by the shrink). The coefficient (±2.446 × ~3% ATS/point ≈ 57.3% implied edge) is fit
on the *magnitude* of ~2,127 pooled residuals; the ATS uses only the *sign* of 186 outcomes — a
much smaller, less informative basis. 57.3% sits comfortably inside the walk-forward ATS's own
CI [54.8%, 68.3%], so there's no contradiction, but they are not equally good estimators: size
off the coefficient-implied edge and its own CI, not the observed walk-forward ATS.

**Bias trade-off, checked directly — result differs from the back-of-envelope prediction.**
The review predicted the shrink would move signed bias on the CB-flagged holdout back up from
v3's +0.542 toward "+0.8 or +0.9," extrapolating from the historical v2→v3 bias improvement
(+1.596→+0.542). Computed directly on the *current* production formula (market line + QB-swap
+ CB term — the historical +1.596/+0.542 figures were computed on the pre-removal Layer-1/
blend-residual base, which no longer exists in production): signed bias is **−0.282 at ±2.977
vs. −0.290 at ±2.446** — essentially unchanged, and negative rather than the predicted
positive range. The specific numerical prediction doesn't hold against today's actual
production formula, though checking it directly (rather than assuming either direction) was
the right instinct regardless, and the underlying point stands: shrinking the coefficient does
not meaningfully worsen bias on the current formula, so the shrink is close to a free
precaution here, not a documented one-sided trade as originally worried.

**Honest bottom line, revised**: the CB adjustment's direction is supported and its lookahead
risk is ruled out by direct experiment, not just code-reading. But the evidence that
corroborates it *independent of the resubstitution error* is weaker than round 3 reported —
once the actual specification search is priced into the null, the walk-forward ATS sits right
at the edge of conventional significance (~95th percentile), not comfortably in the tail
(~97.5th, the single-configuration read). The honest position is to hold this claim
provisionally: real-looking, not yet proven at the strength previously claimed, with the
production coefficient conservatively set regardless of how this resolves.

### 6.2 `src/models/injury_reallocation.py` — props-level detection + reallocation (added 2026-07)
A **completely separate mechanism** from §6.1, solving a different problem: §6.1 answers
"how much does the game margin shift when a starter is out," this answers "who actually
gets the touches instead." Two detection sources:
1. **Real roster status** from `weekly_rosters`' own `status` field (`ACT/CUT/RES/SUS/
   PUP/NWT/RSN/EXE/DEV/UDF/INA/UFA/RSR/TRC/TRD/RFA/TRT/RET/E01/E14`) — sitting completely
   unused before this feature, despite being fetched every single pipeline run.
   `INACTIVE_STATUSES` = everything except `ACT`.
2. **Manual override file** (`data/manual_overrides/known_outs_2026.json`) for verified
   breaking news the automated data hasn't caught up to yet — same stopgap pattern as the
   §5.2 Clay bootstrap. Entries should be web-verified before adding (a real example: Zach
   Charbonnet's July 2026 PUP placement, confirmed via live web search including the
   specific detail that PUP carries a mandatory 4-game absence if still on PUP after roster
   cutdown, before being added).

**Reallocation is validated POSITION-SPECIFIC — this is the load-bearing finding, not a
blanket rule.** Tested against real historical "season-long lead player ruled Out" instances
(injury reports, 2016-2025):
- **RB carries**: proportional reallocation among remaining active backs is a real,
  significant improvement (`reallocate_shares()`) — MAE 4.49 → 3.71 (~17% better), p=0.0003,
  n=443 backup-week observations. Bell-cow succession is real: a clear next-man-up
  typically absorbs the vacated workload.
- **WR/TE targets**: the *same* reallocation makes things **worse** — MAE 1.93 → 2.32
  (~20% worse), p=0.0001, n=513. Vacated targets don't redistribute cleanly across a
  receiving corps the way carries do. For this group, simply excluding the injured player
  (the pre-existing default behavior) is already the validated-better choice — deliberately
  **not** reallocated.

`reallocate_shares(raw_shares, out_keys)` is generic (works on any combined pool of real-
player + rookie-fallback shares, keyed by either a real `player_id` or a synthetic
`"clay:<name>"` string — see §6.3); it renormalizes the active remainder to sum to the full
original total, i.e. the vacated share gets redistributed proportionally, not discarded.

### 6.3 Rookie fallback, folded into the same reallocation pool
`rookie_fallback_rb_rates()` — Clay-projected RBs for a team with **zero real engine
history** (`carry_engine.predict(pid) == 0`). Real mid-project discovery: the trigger
condition must be "zero real history," **not** "absent from the roster snapshot" — a true
rookie can go from completely absent to present-with-zero-history in `weekly_rosters`
*mid-session*, as nflverse's own roster data updates with real transactions (confirmed
directly: this happened once while building this exact feature, between two consecutive
pipeline runs). Uses the same 3-tier name-matching cascade (now `src.ingest.name_matching`,
see §12.2). Returns per-touch rates too (`ypc_rate`, `rush_td_rate`, from Clay's own season
projection ÷ games) so the caller can build a complete prop row directly, not just a share
number.

### 6.3.1 Draft-capital rookie prior (`src/models/rookie_prior.py`, added 2026-07, review #2.6)

Replaces the Clay-PDF dependency above for rookie usage specifically. **A real, previously-
existing gap**: the Clay-based fallback only ever covered RB — a WR/TE rookie with zero real
NFL history got literally zero projected volume in the live pipeline. Draft-capital fills
this and works automatically every season (no manual annual PDF re-extraction).

Fits expected first-season `target_share`/`carry_share` on `(position, round)` using real
draft data (`nfl_data_py`'s own `import_draft_picks` — direct `gsis_id` linkage, no fuzzy
name matching needed at all, unlike the Clay pattern) plus real rookie-season outcomes,
2016-2025. Bayesian-shrunk per `(position, round)` bucket toward the position-wide mean by
bucket sample size — same shrinkage principle as `TdRateEngine`, applied to draft-round
buckets. Validated via leave-one-season-out (fit on all other seasons' rookies, predict the
held-out season): beats a naive position-wide-mean baseline by **~18% on WR/TE target share**
(MAE 0.041 vs. 0.050, p<0.0001, n=377) and **~16% on RB carry share** (MAE 0.124 vs. 0.148,
p<0.0001, n=180).

Downstream per-touch rates (yards/touch, TD rate, catch rate) need **no rookie-specific
handling at all** — `TdRateEngine.predict()` for a player with zero real touches already
simplifies exactly to the league-average rate (`(0 + pw*league_rate)/(0+pw) = league_rate`),
so only the *share* needed an explicit override.

**Wired into `weekly_update.py`**: for RB, draft-capital is merged into the raw-shares pool
first, then Clay's real (if available) projection overwrites it where present — Clay's
current-year scheme-fit/landing-spot information is likely more precise than a historical
prior when it actually exists, so it takes precedence, with draft-capital as the always-on
fallback underneath. For WR/TE (no Clay equivalent exists), draft-capital is the only
source, applied as a direct share override — **not** run through `reallocate_shares` (§6.2),
since that mechanism is specifically validated-harmful for WR/TE and is unrelated to this
fallback. Confirmed working on a real 2026 pipeline run: a rookie WR previously invisible to
the props pipeline (zero real history, no Clay coverage) now gets a real, non-zero
projected statline.

### 6.3.2 Real bug found + fixed: name-collision in the Clay RB fallback's ID resolution
(caught 2026-07 in a full-codebase review, live in production until fixed)

A real 2026 case (Quinshon Judkins) rendered as **two separate prop rows** in
`props_2026_wk1.parquet` for the same physical player — one normal (real, engine-based
carry share) and one via `rookie_fallback_rb_rates` ("rookie fallback (Clay projection, no
NFL history)"), overstating his combined projected volume. First hypothesis was the obvious
one — draft-capital's fallback (§6.3.1) and Clay's fallback (§6.3) both firing for the same
rookie and double-counting in `reallocate_shares` — and `weekly_update.py` already had a
name-based dedup guard against exactly that. It didn't fix it, because that wasn't the
actual bug.

**Real root cause**: two *different* real NFL players share the exact same full name —
Cleveland's RB Quinshon Judkins (real carry history, 0.568 in `carry_engine`) and a Green
Bay DL also named Quinshon Judkins (zero rushing history, obviously). `name_matching.py`'s
`build_name_to_id()` is a single global `name_norm -> player_id` map with last-duplicate-
wins semantics — it has no way to disambiguate two same-named players, and it happened to
resolve "quinshon judkins" to the *wrong* one (the DL). `rookie_fallback_rb_rates` then
checked `carry_engine.predict(<wrong pid>) > 0`, saw 0, and incorrectly treated the real,
established RB as a zero-history rookie needing a Clay-projection fallback.

**Fix**: reordered the 3-tier name-matching cascade *inside `rookie_fallback_rb_rates`
specifically* — try the team+position-scoped tier (`lastname_team_pos_to_id`, tier 2) before
the global full-name tier (`name_to_id`, tier 1), rather than after. Team+position scoping is
strictly tighter than a bare full-name match, so trying it first can't make the common case
(no collision) any worse, and it directly disambiguates same-named players on different
teams/positions without needing to touch `name_matching.py`'s shared cascade (used elsewhere
for QB-swap detection, where this collision class hasn't been observed). Confirmed via a
full pipeline re-run: `props_2026_wk1.parquet` has zero duplicate `(game_id, team, player)`
rows, and Judkins now renders exactly once, with his real, engine-based share.

The `weekly_update.py` name-based dedup guard between draft-capital and Clay's fallback is
kept regardless — it's a real, independent safety net for the genuine case of a true
first-year rookie legitimately covered by both sources at once, which is a different
scenario from this bug.

### 6.4 TD-rate prior_weight fix (`player_usage.py`, raised 15→30, then empirical-Bayes 2026-07)
Real bug caught via a concrete case: Tory Horton (SEA WR, 7 career games before a 2025
injury, 22 targets, 5 TDs — a 22.7% raw rate, ~5x league average) still projected at 15.4%
(3.3x league average) *after* Bayesian shrinkage at the old `prior_weight=15`, producing an
overconfident 42% single-game TD probability. Investigation found `prior_weight=15` was
**never actually swept** — only validated via aggregate decile calibration (which stays
essentially flat, corr 0.998-0.999, MAE within 1%, across the *entire* `prior_weight=10` to
`130` range tested). Population-level calibration can look perfect while still leaving
genuine individual-outlier overconfidence, since deciles average over many players and a
few extreme small-sample cases get diluted within a bucket of better-behaved veterans.
Raising to 30 costs nothing in validated aggregate accuracy (same flat-MAE region) while
meaningfully taming the outlier case (Horton: 42% → 36% after the fix). Applied to both
`rec_td_engine` and `rush_td_engine`; **`catch_rate_engine` was deliberately left at 15** —
that's a different, better-behaved statistic (bounded completion-rate-like quantity, not a
rare-event count) and wasn't part of this specific investigation, so changing it without
testing would violate the project's own discipline.

**Superseded 2026-07 (review #2.3) by `fit_empirical_bayes_prior_weight()`** — a principled
derivation instead of a manual sweep: method-of-moments variance decomposition (a
DerSimonian-Laird-style random-effects estimator) separates real between-player skill
variance from expected within-player sampling noise across career touches (min 20), and
`prior_weight = per_touch_variance / between_player_variance`. Root-cause fix for the exact
bug class the Tory Horton case exposed, this time addressing catch_rate too (never actually
validated at 15). Fit values came back much higher than the swept constants across the
board: rec_td 30→~210-269, rush_td 30→~148-194, catch_rate 15→~41-45 (exact value shifts
slightly depending on the season window fit against — TRAIN=2018-2021 vs. the full available
history the live pipeline uses). Validated (receiving TD rate, TRAIN=2018-2021/TEST=2022-2025):
the EB-fit value tames a 5-TD/22-target outlier to 1.35x league average (vs. 2.58x for the
swept 30), with BETTER top-decile calibration (10% overshoot vs. 38%), at a small aggregate
MAE cost consistent with the already-established flat-MAE region. `ypt_engine`/`ypc_engine`
(continuous, not Bernoulli-shaped) were **not** re-derived this way — the same variance-
decomposition applies in principle with a play-level yardage variance in place of
`mu*(1-mu)`, but that extension wasn't built this session; still swept at 80/300.

### 6.5 What was tested and found NOT worth building
- **Opponent defense quality adjustment for player-level props** (distinct from §6.1's
  game-margin injury flags): built a proper walk-forward opponent-allowed-rate signal
  (both TD-rate-allowed and yards-per-target-allowed, Bayesian-shrunk per team) and tested
  scaling a player's own rate by it. **Null both times** (TD-rate matchup: p=0.21; yards-
  allowed matchup: p=0.76) — team-level defensive quality does not meaningfully improve
  individual player-prop predictions once tested properly.
- **Pass-rush vs. coverage quality, split apart** (motivated by "elite D-line + bad
  secondary should favor the passing game"): built real pressure data (`was_pressure`
  field, 93-100% coverage 2016-2025) into a separate pass-rush-rate-allowed and clean-
  pocket-completion-rate-allowed signal, tested the specific interaction. **Statistically
  significant but harmful** — MAE gets monotonically *worse* as the adjustment is weighted
  more (13.67 → 14.05 at full weight), p<0.0001, n=17,323. Third independent test of
  defensive-matchup quality in this project (alongside the aggregate test above and NFL Big
  Data Bowl player-tracking data, which showed real season-level coverage-quality
  persistence but zero single-game predictive power) — all point the same direction.
  External research independently confirms DVP-style matchup grids are a documented trap in
  the quant/DFS community, not a hidden edge; one narrower variant (shadow-coverage
  matchups) is cited as a big DFS edge but the actual numbers (Julio Jones: 2.81 YPRR in
  shadow vs 2.93 overall) don't back it up either.
- **Team-change harm** (the intuitive worry that a player's whole usage history, built
  under one team/QB/scheme, might not transfer to a new team): tested directly against real
  historical team-changers. **Opposite of the worry** — team-changing WR/TEs are predicted
  *better* than stayers (MAE 0.0504 vs 0.0560 full season, p<0.0001; even weeks 1-4 with the
  new team, p=0.0023). RBs show no significant difference (p=0.15-0.98). A refined "crowded
  receiving room" sub-test (moving in on an established incumbent vs. a vacancy) was
  underpowered (93% of cases were "crowded" by a naive threshold) but showed no signal
  either way, and the bias runs the *reassuring* direction (we slightly under-, not over-,
  project crowded-room changers). Likely explanation: target share / YPRR are highly
  persistent, player-intrinsic traits (~0.70 / >0.60 year-over-year correlation,
  independent research), and team moves aren't random — a team signing/trading for a proven
  WR1 usually does so specifically to keep using him that way.
- **Snap-count trend as a leading indicator**: hypothesis that recent snap% might shift
  before target share catches up. Clean null (corr with target-share residual: 0.002; fitted
  adjustment makes MAE very slightly worse, p<0.0001 but tiny effect size) — snap% and
  target share are already so tightly coupled that snap% adds no new information.
- **Multivariate Ridge ensemble of the above four props-specific signals** (team-changed,
  crowded-room, trailing snap%, opponent yards-allowed), combined jointly rather than
  tested alone: **test R² = -0.00007** (worse than predicting the mean), MAE gets slightly
  worse when combined (p=0.0002). Confirms these don't hold hidden joint signal either — the
  same conclusion the analogous margin-level ensemble test reached earlier in this project.
- **Bottom-up, props-derived team totals** (summing player yard/TD projections into a team
  total, as an alternative/ensemble input to the top-down regression total): bottom-up alone
  MAE 10.97 vs top-down 10.58; blending barely moves it (10.57, p=0.167, tiny fitted weight
  on the bottom-up signal). Also directly answers "could props help model scores" — no.
- **Volume-based opponent effects** (review #3.3, 2026-07): a genuinely distinct angle from
  the RATE-based opponent tests above — does an opponent's defensive quality affect a team's
  offensive VOLUME (total plays run, red-zone trips) rather than per-touch efficiency? Tested
  against real walk-forward pregame `def_rating` (TRAIN=2018-2021/TEST=2022-2025): **clean
  null on both fronts**. Total plays: TRAIN corr=0.054, TEST MAE unchanged (6.746 naive vs.
  6.747 with signal), p=0.877. Red-zone trips: TRAIN corr=0.129, negligible MAE improvement
  (1.283 vs. 1.280), p=0.369. Extends the already-established opponent-rate null to volume as
  well — consistent with §7.1's separate finding that pace/plays-per-game is close to
  unpredictable game-to-game regardless of opponent.

---

## 7. Game environment (`src/models/game_environment.py`)

### 7.1 Pace — flat baseline, deliberately
Team-level plays/game is close to unpredictable game-to-game (correlation ~0.05 with a
recency-weighted own-team average) — dominated by game-flow noise (overtime, weather,
injuries, muffed snaps), not stable team identity. `LEAGUE_AVG_PLAYS` is used flat, not a
per-team model (tested and didn't earn its keep) — **but it is no longer a frozen literal**:
review round 2's stale-coupling sweep (§12.2.1) found it hardcoded at `62.859` and never
refreshed as new seasons accumulate, with real drift confirmed (2024-2025-only mean 61.04
plays/team-game). `weekly_update.py` now computes it fresh from `CALIBRATION_SEASONS` every
run (61.70 on a recent live run); only `predict_props_2026.py`'s frozen one-time reference
copy stays a literal.

### 7.2 Neutral-script pass rate — real, simple EWMA
Real, useful persistence (~0.25 game-to-game, ~0.33 year-over-year) reflecting genuine play-
calling identity. `PassRateEngine`: simple per-team EWMA (`alpha=0.15`,
`season_shrink=0.5`), no opponent adjustment (tested, didn't help). `build_pass_rate_table`
restricts to "neutral script" plays (`|score_differential| <= 8`, `MIN_NEUTRAL_PLAYS=5`) to
isolate play-calling identity from game-script reaction.

### 7.3 Game-script adjustment
Full-game pass rate shifts from the neutral-script baseline based on Layer 1's predicted
margin (`build_full_game_pass_rate` + a linear fit of `pass_rate_residual ~ team_margin`,
refit on `CALIBRATION_SEASONS` each pipeline run). This is what actually drives the props
pipeline's pass/rush attempt split per game (`pass_rate = pr_engine.predict(team) +
script_intercept + script_slope*team_margin`, clipped to [0.30, 0.80]).

**Bug fixed 2026-07** (found during the review's housekeeping audit, §12.2):
`weekly_update.py` and `validate_props_pipeline.py` both computed this fit as
`script_slope, script_intercept = np.polyfit(...)[::-1]` — which actually assigns
`script_slope=intercept` and `script_intercept=slope` (this module's own correctly-written
`fit_calibration()`, the original source of this calculation, uses the matching
`script_intercept, script_slope = fit_calibration(...)` order). Confirmed against real
data: this flipped the sign and roughly doubled the magnitude of the "a team that's winning
passes less" effect (correct: intercept=0.0066, slope=-0.0034; as shipped:
`script_slope`=0.0066, `script_intercept`=-0.0034) for every prop prediction's pass/rush
split, for as long as `weekly_update.py` has been the live pipeline. Both now use
`src/utils/stats.py`'s `fit_linear()`, which returns `(intercept, slope)` unambiguously and
asserts the fitted sign matches the raw correlation.

**v2 (2026-07, review #1.5)**: added the market-implied team total as a second regressor
(`fit_game_script`/`apply_game_script`, replacing the single-variable `fit_calibration` call).
`implied_team_total = total_line/2 +/- spread_line/2` per team — distinct from margin (who's
favored) since it also captures the overall scoring environment (shootout-type games see more
passing at a given margin than a low-total grind-it-out game). Validated walk-forward
(TRAIN=2018-2021/TEST=2022-2025): MAE 0.07141→0.07080, p=0.0406, **same-direction improvement
in all 4 individual TEST seasons** (not one anomalous year). Note: the raw univariate
correlation of implied_team_total with the residual is *negative* while its coefficient here
is *positive* — not a fragile-signal red flag (contrast the rejected wind-on-totals test
below): implied_team_total conflates "how favored this team is" (corr 0.38 with team_margin)
with "the scoring environment," and conditioning on team_margin isolates the latter, which has
a sensible, interpretable sign. Separately: as of the Phase 0 margin-blend removal (§4.3),
`team_margin` itself is already market-derived (it flows from `our_margin`, which is now built
from `spread_line` directly) — so "replace Layer 1's margin with the market's" was effectively
already achieved as a side effect of that architecture change, before this v2 fit was added.

### 7.4 Total-points calibration
`total = c + d*pregame_total_signal + e*is_indoor`. Dome/closed-roof games score ~4-6 points
more (total) than pure outdoor — real, consistent, kept. **Wind was tested for total points
specifically and rejected** — its estimated effect flipped sign between an isolated
outdoor-only test and a joint fit with the other terms, the signature of a fragile, non-
robust signal at the *total-points* level. This does **not** contradict §8's later, separate
finding that wind affects QB/WR **yards-per-touch rates** specifically — different target
variable, different (much larger, more granular) sample, properly re-validated.

**Kicker quality, tested and rejected (2026-07, review #2.7)**: `field_goal_result`,
`kick_distance`, `kicker_player_id` are already in PBP, unused. Built a distance-adjusted,
Bayesian-shrunk per-kicker rating (expected FG% by 5-yard distance bucket, fit on TRAIN;
kicker's shrunk delta = actual-vs-expected makes, shrunk toward 0 by attempt volume — same
shrinkage principle as `TdRateEngine`, swept `prior_weight` 5-80). Raw correlation between the
resulting delta and the total-points base-model residual is ~0.02-0.03 on both TRAIN and
TEST — essentially no relationship. A joint 4-variable regression showed a "significant"
positive coefficient at every prior_weight tested, but TEST MAE was *consistently worse*
(10.544→10.58x) at every one — the coefficient was fitting noise in a small added-variable
regression, not real signal. Clean null; not shipped.

---

## 8. Weather adjustment (`src/models/weather_adjustment.py`, added 2026-07)

Distinct from §7.4's rejected total-points wind test. Tested wind's effect on **per-touch
efficiency rates** specifically (QB yards/attempt, WR/TE yards/target, QB completion rate),
walk-forward, TRAIN=2018-2021 fit / TEST=2022-2025:

| Target | Result |
|---|---|
| QB yards/attempt | MAE 44.72 → 44.09, **p=0.0022** — validated, shipped |
| WR/TE yards/target | MAE 13.55 → 13.31, **p<0.0001**, n=9534 — validated, shipped (stronger of the two) |
| QB completion rate | Same directionally-correct (negative) raw correlation, but p=0.34 — **not shipped**, correlation alone isn't sufficient |

`fit_wind_adjustment()` fits `rate_residual = intercept + slope*wind` on outdoor games only
(`roof=="outdoors"`), refit fresh each pipeline run on the full available history (2018-
current). `apply_wind_adjustment()` is a deliberate no-op indoors or when wind isn't known
yet (NaN — real forecasts aren't reliable/present in the schedule data until close to
kickoff). **This is why the wind adjustment currently does nothing for predictions made
weeks in advance** — it only activates once a game is close enough that `nfl_data_py`'s
schedule export has a real forecast, which is exactly the reason the §11.2 gameday-refresh
scheduled task exists.

A real, self-caught bug during this investigation: `np.polyfit(x, y, 1)[::-1]` — the same
slope/intercept-swap mistake documented in §12.2, made a **second time** in this exact
investigation, which initially made the fitted adjustment look actively harmful (MAE nearly
doubled, 2.14→5.78 for high-wind games) before the sign was caught and fixed against the
raw correlation. Lesson: always sanity-check a fitted sign against the raw correlation
before trusting any `np.polyfit`-based result in this codebase.

---

## 9. Player usage / props core (`src/models/player_usage.py`)

### 9.1 `ShareEngine` — target share, carry share
Recency-weighted (`USAGE_ALPHA=0.30`, roughly a 3-4 game effective window — deliberately
much shorter memory than Layer 1's team ratings, since usage roles change fast) per-player
rating, gated so a week with zero statistical involvement (bye, injury, healthy scratch)
carries the rating forward unchanged rather than wrongly crushing it toward zero.
`SEASON_SHRINK=0.35` at year boundaries. Validated: target share corr 0.743, carry share
corr 0.884-0.887 on 2022-2024 holdout — real, strong signal, well above a flat baseline. A
standalone red-zone-share sub-model was tried and dropped — too small a sample for any
smoothing to beat the naive mean across a full grid search.

### 9.2 `TdRateEngine` — generic Bayesian/Laplace-shrunk per-touch rate
`predict = (hits + prior_weight*league_rate) / (touches + prior_weight)`. Reused generically
for TD rate, yards/touch, catch rate, and (§5.3) all four QB passing stats — one class,
many `(league_rate, prior_weight)` configurations. TD-rate and catch-rate `prior_weight` are
now empirical-Bayes-fitted in the live pipeline (rec_td ~210-269, rush_td ~148-194, catch_rate
~41-45 — see §6.4), superseding the earlier swept constants (TD-rate 15→30, catch_rate never
actually validated at 15). Yards/target (80) and yards/carry (300) prior weights are still
swept constants, validated via a proper sweep in `predict_props_2026.py` — real but genuinely
weaker signal than share or TD-rate (0.5-3.7% MAE improvement over naive, documented there as
"included because it's genuine signal, not because it's strong"); not yet re-derived via
empirical-Bayes (continuous stat, would need a play-level variance estimate rather than
`mu*(1-mu)` — see §6.4).

### 9.2.1 Air yards / aDOT / WOPR (added 2026-07, review #2.1) — TESTED AND REJECTED, self-caught methodology bug

`air_yards` and related PBP columns had been sitting unused. aDOT (average depth of target)
reuses `TdRateEngine` generically — it's just another Bayesian-shrunk per-touch rate,
`air_yards/targets` instead of `receiving_yards/targets`, no new class needed. `air_yards_share`
reuses `ShareEngine` the same way `target_share`/`carry_share` do. `compute_wopr()` = standard
fantasy-analytics composite `0.7*target_share + 0.3*air_yards_share`.

**A real, worth-recording mistake**: the first validation pass approximated a player's
projected targets using a flat pass-attempts proxy (`target_share * ~35`) instead of the real,
game-specific pregame pass-rate prediction the live pipeline actually computes. That test
showed strong, "significant" improvements for both candidates — WOPR beating target_share for
receiving-yards MAE (18.821 vs 18.913, p=0.0344) and aDOT improving the yards formula
(p<0.0001) — and both were briefly wired into `weekly_update.py`. A live pipeline re-run
immediately surfaced the problem: real receiving-yards projections had been cut roughly in
half (e.g. a WR1 with a 20%+ target share projected at ~46 yards instead of a sane ~95) because
the fitted calibration's `proj_coef` (0.43) was silently compensating for the ~1.7x scale
mismatch between the flat proxy used at fit time and the real, smaller pass-rate-scaled
volume used at inference time.

**Once corrected** — reconstructing historical proj_targets using the actual pregame
pass-rate prediction (+ game-script adjustment) matching `weekly_update.py`'s real formula —
**both results reversed to null**: WOPR-based receiving-yards MAE was *worse* than
target_share-based (19.20 vs 19.14, p=0.14, wrong direction), and aDOT added no significant
improvement beyond target_share+ypt_rate (MAE 19.206→19.198, p=0.51, n=17,323,
TEST=2022-2025). **Neither is wired into the live pipeline.** `compute_wopr()`/
`fit_adot_calibration()`/`apply_adot_calibration()` are kept as correctly-implemented
utilities — the computations themselves are fine, it was the validation harness around them
that was flawed — available for a future variance/distribution layer, since a decile-
conditional check (unaffected by the scale bug: within every target-share quintile,
high-aDOT players have a consistently higher P(receiving ≥20 yds) than low-aDOT players in
the same quintile, e.g. quintile 3: 0.670 vs 0.594) suggests aDOT carries real information
about outcome variance/boom-potential that a point-estimate correction can't exploit but a
real predictive distribution (§14's suggested Phase 2/3 work) could.

**Lesson, matching §12.2's `np.polyfit` pattern**: a validation shortcut that doesn't match
the real production formula can manufacture a false-positive result the real formula won't
reproduce — always test against the actual mechanism being changed, not a simplified stand-in
for it, and re-verify against a live pipeline run (not just a standalone script) before
trusting a "shipped" result.

### 9.3 `src/models/predict_props_2026.py` — the reference/one-time script
Full Week-1-2026 props pipeline, including the 3-tier name-matching cascade (§6.3) and the
Clay rookie-fallback pattern that was later ported into the live pipeline. Kept as a
reference and for the one-time preseason forecast; `weekly_update.py` (§11) is what actually
runs every week now.

### 9.4 `src/models/validate_props_pipeline.py`
End-to-end validation (not just component validation) — "when you run the WHOLE pipeline on
real held-out data, how close are the numbers to what actually happened?" **Regenerated
2026-07 (review round 3, #6) — this canonical section had gone stale itself** (previously
1.92/3.18, now 1.90/2.94, after the game-script sign fix, `LEAGUE_AVG_PLAYS` refresh, and the
TD-calibration work below all moved it): targets MAE 1.90 (naive 2.48), carries MAE 2.94
(naive 5.25).

**TD-probability calibration is `TD_PROB_CALIB_A/B`, AUTO-REFIT LINEAR every pipeline run**
(review round 4, #7 — reverted from isotonic, round 3's choice). Round 3 replaced the
previously-frozen linear map with `sklearn.isotonic.IsotonicRegression`, tested it two ways,
and found a genuinely two-sided result:
- **Strict walk-forward** (both fit on TRAIN=2018-2021 only, scored on TEST=2022-2025):
  isotonic **underperformed** linear at the top decile (overshoot +0.032 vs. +0.015) — a real,
  surprising negative result, plausibly isotonic overfitting TRAIN's specific tail noise with
  only 4 seasons to fit from.
- **Both fit on all available history** (TEST's own player-weeks inside the fit set): isotonic
  modestly **outperformed** linear (+0.006 vs. +0.008).

Round 3 shipped isotonic on the second result. **Round 4 correctly identified this as the same
resubstitution-flavored evidence structure this project rejected for the CB coefficient ten
sections earlier in this same document** (§6.1.1) — "the in-sample comparison favors it" isn't
a safe basis when the out-of-sample comparison points the other way by 2x, especially given
isotonic's known failure mode (tail overfitting on rare events) is exactly what's at stake
here. The two motivations were also being treated as one when they're separable: the genuinely
strong argument — retiring the R0.1 stale-coupling bug class (§12.2.1: the linear map silently
went stale once already when the upstream `prior_weight` moved to empirical-Bayes values) —
comes from *refitting every run*, not from isotonic specifically.

**Fix**: `weekly_update.py` now auto-refits the two-parameter linear map every run (decile-
bucketed `fit_linear`, same construction as the original R0.1 refit) on the same real
historical (raw_td_prob, actual_td) pairs — captures the identical stale-coupling benefit with
the better-demonstrated out-of-sample behavior. Also tested **Beta calibration** (Kull et al.
— `logit(calibrated) = a·log(p) + b·log(1−p) + c`, a 3-parameter monotone map fit via logistic
regression on the two log-transformed features, the honest small-sample middle ground between
linear and isotonic) in the same strict walk-forward comparison: top-decile overshoot **+0.019**
— between linear and isotonic, but not better than linear either. **Decision rule: auto-refit
linear ships as the default; isotonic or Beta are promoted only if one wins the strict
walk-forward comparison outright**, which neither currently does. `predict_props_2026.py`
keeps its own frozen linear `TD_PROB_CALIB_A/B=(0.046,0.769)`, paired with that script's own
frozen swept `prior_weight=30` — internally self-consistent as a historical reference,
unaffected by this change.

### 9.5 Monte Carlo drive-state game simulator (Phase 2, added 2026-07, review #1.4)

**Margin/win-probability wired into `weekly_update.py` as of review round 2 (#3.1) — total
still not wired, see below.** Built and validated as real, working infrastructure, honestly
reported including two negative results along the way, per this project's discipline (§12).
The review's biggest-lift ask: a real predictive *distribution* over (home_score, away_score),
not just a point estimate — unlocking moneyline pricing, alt-line/key-number-aware edge, and
calibrated win probability that the point-estimate model has no way to produce at all.

**Architecture** — reuses this workspace's MLB `game_simulator.py`'s PATTERN (combine context
→ sample → bootstrap-resample a real historical state transition → loop to a terminal
condition → wrap in N Monte Carlo trials → aggregate), not its mechanics:
- `src/models/drive_transitions.py` — the NFL analog of MLB's `TransitionTable`. Builds a real
  historical drive-outcome table directly from nflverse's already-computed drive-level PBP
  fields (`fixed_drive`, `fixed_drive_result`, `drive_time_of_possession`, etc.) — no
  down-by-down mechanics modeled by hand. Two resampling pools: `DriveOutcomeSampler` (drive
  outcomes conditioned on start field-position bucket + market-implied-total quantile) and
  `NextDrivePositionSampler` (next drive's real starting field position, conditioned on how
  the prior drive ended).
- `src/models/game_simulator.py` — `GameSimulator`: alternates possessions (every real
  drive-ending event changes possession, so who has the ball next never needs modeling,
  only starting field position does), consumes real, bootstrap-resampled `time_elapsed` per
  drive against a 3600-second game clock (two 1800-second halves, each opening with a real
  resampled kickoff-return field position), simplified sudden-death overtime (capped at
  `MAX_OT_DRIVES`, an explicit approximation — modern NFL OT rules are more complex and vary
  by era across this dataset's span). Wraps in N Monte Carlo trials, aggregates to
  margin/total means, stds, and win probability.
- `src/models/validate_game_simulator.py` — walk-forward validation: drive-resampling pools
  built ONLY from TRAIN=2018-2021 (no TEST lookahead), scored against TEST=2022-2025 (1,087
  games, 300 trials/game), compared against the current model's Gaussian point-estimate
  approximation (`scoring.py`) on empirical CRPS, win-probability Brier/reliability, and
  straight-up accuracy.

**Two real bugs caught before they corrupted the pools**:
1. A drive's first PBP row is often the kickoff itself, whose `yardline_100` reflects the
   *kicking* team's line, not the receiving team's real post-return starting position
   (confirmed: "kicks from ARI 35" showed `yardline_100=35`, while the actual result —
   "Touchback to the BUF 30" — was on the very next row, `yardline_100=70`). Using the literal
   first row would have fed the simulator systematically wrong field position for every
   kickoff-opened drive. Fixed by using the first non-special-teams play instead.
2. ~0.68% of drives showed implausible point totals (>9, up to 56) — a `fixed_drive`
   boundary data-quality tail (mostly tagged "Opp touchdown"), not a real single-drive
   outcome. Clipped at 9 (a real drive cannot legitimately exceed a TD+2pt) before it could
   corrupt the resampling pool.

**Three iterations on team-strength conditioning, honestly reported**:
- **v1**: conditioned drive resampling on Layer-1 EPA off/def terciles (3 discrete buckets
  each). Validated end-to-end: underperformed the current market-blend approach on *every*
  metric (margin MAE 10.08 vs 9.55, CRPS 7.40 vs 6.97, straight-up 63.5% vs 67.2%, total MAE
  12.12 vs 10.20, Brier 0.228 vs 0.212) — the mechanics checked out (realistic margin std,
  correct directional response to strength, sensible field-position dynamics), the coarse
  3-bucket signal was just far less precise than the market line the current approach is
  already anchored on.
- **v2** (current, shipped state): replaced Layer-1 terciles with the market-implied team
  total (`total_line/2 ± spread_line/2`, quantile-binned — the same signal validated in
  game_environment.py's game-script v2), already opponent-adjusted and continuous rather than
  a coarse split. Large improvement: margin MAE 9.65 vs 9.55 (nearly tied), CRPS 7.05 vs 6.97
  (nearly tied), straight-up 67.3% vs 67.2% (essentially tied, sometimes ahead across re-runs),
  Brier 0.212 vs 0.212 (parity). Total still lagged clearly (MAE 11.62 vs 10.20, CRPS 8.06 vs
  7.30) — each side's score was simulated fully independently, missing the real correlation
  between the two teams' scoring (weather, pace, game script affect both offenses together).
- **v3 experiment, tested and reverted**: drew one shared "game-environment percentile" per
  Monte Carlo trial, nudging both teams' drive draws toward it together
  (`DriveOutcomeSampler.sample`'s `shared_percentile`, `CORRELATION_WEIGHT=0.45`). Made things
  *worse* overall: margin/Brier improved marginally, but total MAE jumped to 14.48 and CRPS to
  11.63 — correlating both sides toward the same percentile shrinks the variance of their
  DIFFERENCE (margin) while inflating the variance of their SUM (total), the opposite of the
  intended fix. Reverted; the mechanism is kept in the code, disabled by default
  (`shared_pct=None` in `simulate_one`), for a future more careful attempt (a much smaller
  correlation weight, or correlating only a subset of drives).

**Wired in (review round 2, #3.1)**: `weekly_update.py` now builds the simulator once per
run (`build_simulator_for_season_range(pbp, schedules)`, fresh bin edges each run — this is
forward production, not a TEST-scoring exercise) and, per game, converts each side's
market-implied team total into a quantile bin (`implied_total_quantile_bin`) and runs
`simulate_game(home_q, away_q, n_trials=300)`. **Display-only addition** — does not replace
`our_margin`/`our_total` above. Adds `sim_margin_std`, `sim_home_win_prob`, and a derived
American-odds moneyline for both sides (`american_odds_from_prob`) to
`predictions_{season}_wk{week}.parquet` and the printed output; falls back to `None` for the
rare game with no market line published yet (same fallback discipline as margin/total).
**Fixed a real, live bug: the displayed moneyline contradicted the displayed spread (review
round 4, #4).** Confirmed directly before believing it: computed the simulator's own implied
margin (`margin_std * Φ⁻¹(win_prob)`) for every game in a live Week 1 2026 slate and compared
to the displayed `our_margin` — **mean absolute gap = 1.58 points**, well above a "this is
cosmetic" threshold. Root cause: the raw simulated margin distribution centers wherever the
drive dynamics land it (margin MAE 9.69, worse at *location* than the market-based
`our_margin`'s 9.49), while `our_margin` is the better location estimate — same "market for
location, model for shape" logic already used to remove the margin market blend (Phase 0).
**Fix**: shift the simulated margin samples by `(our_margin − raw_margin_mean)` before
deriving win probability/moneyline — preserves the simulator's real variance/skew/key-number
mass (the entire reason it exists) while inheriting the market-based margin's location.
Verified exact by construction (`recentered_margins.mean() == our_margin` to floating-point
precision, checked directly on a real game) — a residual ~0.66-point average gap remains in
the *Gaussian-inverse-CDF diagnostic* specifically, which is expected and not a bug: that
diagnostic assumes normality, and the simulator's whole value is capturing real skew a
Gaussian can't, so it won't perfectly invert even after exact recentering.

**Total is now shipped too (review round 4, #5), unblocked by the same fix.** Total was held
back because its MAE (12.28) trails the top-down model's (10.20) — but that gap is location,
and `our_total` already provides the better location estimate. Recentered the same way and
added `sim_total_std` to `predictions_{season}_wk{week}.parquet` (no separate
`sim_total_mean` — the location is `our_total`'s, not an independent estimate). This converts
the total half of the simulator from a blocked project into a shipped capability: real
distributional total pricing (alt-totals, team totals, O/U at any number) the top-down point
estimate structurally cannot produce.

Sanity-checked on a real live run: win probability tracks margin's sign correctly across all
16 Week 1 2026 games, moneylines are now internally consistent with the displayed spread by
construction, simulated margin/total std land in plausible ranges (~13-16 / ~13-15 points),
and the added Monte Carlo cost is negligible (16 games × 300 trials adds well under a second
to a ~1-minute full pipeline run).

**Pace/volume-factor correlation fix (review round 2, #3.2): premise tested, NOT supported,
not built.** The proposed mechanism-based fix for v3's failure was to decompose the shared
per-trial factor into a pace/volume multiplier applied jointly to `time_elapsed` (moving
total via a correlated shift in how many drives BOTH teams get, near-zero effect on margin
since it scales both sides symmetrically) plus independent per-team-strength factors on
drive-outcome quality (unchanged). Before implementing it, tested its premise directly —
exactly this project's own established practice (§9.6's Dirichlet-Multinomial premise test is
the precedent): is the simulator's total-drives-per-game *under*-dispersed relative to real
games, i.e. is there real variance for an added pace factor to fill?

**It is not.** Real total drives/game (both teams combined, `build_drive_table`, n=2,639
games): mean=22.11, std=3.18, var=10.12 (5th-95th pct: 17-28). The CURRENT simulator (v2, no
pace factor), measured directly via the real `GameSimulator.simulate_one` across a varied mix
of quantile matchups (n=3,000 simulated games, not a hand-reimplementation): mean=24.22,
std=3.58, **var=12.85** — already *exceeding* real dispersion, not falling short of it. Adding
an extra shared multiplicative factor on top would inflate total variance further past
reality, the opposite of the intended fix — so it was not built. This is a real, useful
negative result: the "missing correlation" diagnosis in v3's docstring (correlating something
shared across both teams would help total) turned out to be right about the *problem* but
this round's specific proposed *mechanism* (pace/volume via drive count) doesn't have a gap
to fill; whatever the real missing correlation is, it isn't in how many total possessions
both teams get.

**A different, real clue surfaced instead**: the simulated mean itself runs hot — 24.22
simulated vs. 22.11 real drives/game, a ~9.5% overshoot, present even in variance-matched
baseline runs. This is a plausible, more direct lead on v2's total MAE gap (11.62 vs. 10.20)
than a correlation mechanism would have been — worth a future session's attention (candidates:
OT-drive frequency in the simulator vs. real NFL OT rates, or a subtle composition bias in the
bootstrap pools' average `time_elapsed`) — but chasing down that root cause is new work beyond
this item's scope, not done here.

Kicker quality re-tested as a total-scoring input to the simulator specifically (rejected for
the point-estimate model in §7.4, but drive-level modeling might use it differently —
converting a specific field-position-conditional drive outcome into 3 points or 0 is a
different functional role than an additive total-points term) remains a real, untried idea for
a future session.

**The drive-count overshoot's cause, found and fixed — but it didn't fix total (review
round 3, #3).** A specific, falsifiable hypothesis for the 24.22-vs-22.11 overshoot above:
drives ending a half don't end by score/turnover, they end because the clock hit 0:00, so
their recorded `time_elapsed` is whatever fraction of a drive fit before time expired, not a
natural drive length — bootstrap-resampling that into a mid-simulated-half drive slot (the
sampler has no way to know "this is the last drive of the half" when drawing) biases the
pool's mean drive length down and fits more drives into the simulator's fixed clock. Confirmed
directly: "End of half" drives are 7.0% of the real drive pool (n=4,077/58,355), mean
`time_elapsed`=42.6s vs. 165.8s for every other result category — a real, dramatic truncation.
Excluding these rows from both `DriveOutcomeSampler`'s and `NextDrivePositionSampler`'s pools
(`game_simulator.py`'s `build_simulator_for_season_range`; the latter also gets a second,
related fix — a half-ending drive's `next_start_yardline_100` is the *other* team's
second-half-opening kickoff position, not a legitimate same-half transition, since
`build_drive_table`'s `shift(-1)` is grouped by game, not by half):

- **Drive count, substantially improved, not fully closed**: mean 24.22→23.18 (real 22.11) —
  closed ~46% of the mean gap. Variance 12.85→10.97 (real 10.12) — closed ~69% of the variance
  gap, now much closer to (if still slightly above) real dispersion.
- **Margin: essentially unchanged** (`validate_game_simulator.py`, TRAIN=2018-2021 pools,
  TEST=2022-2025, n=1,087): MAE 9.69 vs. 9.65 before, CRPS 7.08 vs. 7.05, straight-up 68.2% vs.
  67.3% — all within noise of the pre-fix numbers. No regression.
- **Total: NOT improved — slightly worse.** MAE 12.28 vs. 11.62 before (vs. 10.20 top-down),
  CRPS 8.46 vs. 8.06 before. This is a genuine, honest surprise: fixing the diagnosed,
  confirmed drive-count/duration bias did not translate into a total-accuracy improvement, and
  if anything moved it slightly the wrong way — plausibly because removing the (low-scoring,
  artificially short) half-ending drives from the pool raises the *average points-per-drive*
  of what remains, partially or fully offsetting the benefit of drawing slightly fewer drives
  per game.

**Kept anyway**: this is a real data-quality correction on its own terms (the pool should not
contain drives whose duration is an artifact of when the half ended, independent of whether it
helps the specific total-MAE metric), margin is not regressed, and the drive-count/variance
match to real games is now substantially closer to accurate. Re-checked whether this reopens
§9.5's pace/volume-factor rejection: it does not — the corrected variance (10.97) is now even
closer to real (10.12) than before, reinforcing rather than reversing the "no meaningful
deficit for a pace factor to fill" conclusion.

**Root cause found and fixed properly (review round 4, #6): model the half boundary
explicitly instead of excluding half-ending drives entirely.** Round 3's blanket exclusion
correctly fixed the duration artifact but also discarded the outcome — nearly always zero
points, a real, common NFL event (a drive cut off by the clock), not an artifact — explaining
why total got slightly worse: the remaining pool's average points-per-drive rose along with
the now-accurate duration.

Principled fix: `DriveOutcomeSampler` keeps sampling duration+outcome together from the
clean, duration-uncontaminated pool as before, but now also holds a separate end-of-half
outcome pool (`sample_end_of_half`, bucketed by start field position, built from the excluded
rows' real point outcomes — kept, not thrown away). `GameSimulator._run_half` adds an
explicit clock check: if a normally-sampled candidate drive's own duration would exceed the
half's remaining time, the real generating process is "this drive got cut off," not "it
completed normally right as the half happened to end" — so its outcome is resolved from the
real end-of-half point distribution instead of its own (would-be, uninterrupted) outcome, and
the half ends there.

**Result — fixes drive count and total together, as hoped, without trading one against the
other**:

| | Original v2 | Round 3 (half-excluded) | Round 4 (explicit boundary) | Real / top-down |
|---|---|---|---|---|
| Drive count mean | 24.22 | 23.18 | 23.27 | 22.11 |
| Drive count var | 12.85 | 10.97 | — | 10.12 |
| Margin MAE | 9.65 | 9.69 | **9.67** | 9.49 (market) |
| Margin CRPS | 6.97 | 7.05 | **7.06** | — |
| **Total MAE** | 11.62 | **12.28** (worse) | **10.49** | 10.20 (top-down) |
| **Total CRPS** | 8.06 | **8.46** (worse) | **7.40** | 7.30 (top-down) |

Total MAE/CRPS are now close to the top-down model's own numbers — genuinely better than
*both* the original v2 and round 3's half-excluded version, with margin essentially unchanged
throughout. This directly confirms the review's diagnosis (half-ending drives have two
properties, only one of which is an artifact) and its proposed fix. Combined with the
recentering fix above (§9.5), the simulated total distribution is now solid enough to ship
with real confidence, not just "unblocked by recentering location while its own shape was
still an open question."

### 9.6 Props touch-count simulator (Phase 3, added 2026-07, review #2.2)

**Not wired into the live pipeline as of this writing** — real, validated infrastructure for a
future props-distribution layer, built after redirecting away from the review's original spec.

The review's spec called for a parametric generative chain: target vector ~ Dirichlet-
Multinomial(attempts, concentration × share_vector), yards|reception ~ Gamma/lognormal, TDs ~
Poisson-binomial. Before building that, tested its core premise directly: is a real player's
week-to-week target count uniformly overdispersed relative to pure multinomial sampling
(exactly what a single Dirichlet-Multinomial concentration parameter models)? **It is not.**
Standardized residuals `(actual - predicted) / multinomial_sd` show mean=0.38, std=1.71 (vs.
the ~0/1 a well-calibrated multinomial would show), with a 95th percentile of 2.96 and a 99th
of 6.0 — a real, heavy right tail. This holds *within* every usage tier (fringe <5% share,
role-player 5-15%, primary >15%), not just as an artifact of mixing them together — the
*median* week in every tier is actually tighter than pure multinomial predicts (our
target-share prediction has real signal), but a persistent minority of large-surprise weeks
(almost certainly real injury-driven role changes, blowout game scripts, coaching decisions
the share engine's EWMA can't anticipate) pulls the mean well above 1. The data-generating
process looks like a mixture, not a single-parameter overdispersion.

**Decision**: rather than assume a parametric mixture shape (Beta-Binomial mixture,
contaminated-normal — more unverified assumptions layered on an already-uncertain premise),
`src/models/props_simulator.py`'s `TouchCountSimulator` reuses the exact successful pattern
from Phase 2's `game_simulator.py`: **bootstrap-resample the real empirical distribution of
standardized residuals** (stratified by usage tier), which reproduces whatever the true shape
is — skew, heavy tails, everything — without assuming a specific family.

**Validated** (walk-forward: residual pools fit on TRAIN=2018-2021, scored on TEST=2022-2025,
1000 Monte Carlo trials/player-week, n=20,915 test player-weeks): P(targets ≥ threshold)
calibration beats the naive Binomial(n,p) assumption implicit in the current point-estimate
pipeline (which has no distributional information at all) at every threshold tested, with the
gap widening at higher, more decision-relevant thresholds:

| Threshold | Bootstrap simulator (corr / Brier) | Naive Binomial (corr / Brier) |
|---|---|---|
| ≥3 | 0.9991 / 0.14616 | 0.9983 / 0.14824 |
| ≥5 | 0.9991 / 0.11631 | 0.9969 / 0.11787 |
| ≥7 | 0.9982 / 0.07940 | 0.9914 / 0.08062 |

Built and validated for target count only (the most decision-relevant volume signal); not yet
extended to receptions/yards/TDs or wired into `weekly_update.py`'s props loop, which still
outputs point estimates only. Same architecture (`build_standardized_residuals`,
`TouchCountSimulator`) directly reapplies to carries (using `carry_share_calc`/`team_carries`)
and, further downstream, to receptions/yards/TDs conditioned on a simulated touch count,
whenever a future session extends this to the full compound chain.

**Re-evaluated with metrics that can discriminate (review round 2, #3.3,
`src/models/validate_props_simulator.py`).** The 0.999x correlations above are decile/bucket-
level aggregates (same construction as `validate_props_pipeline.py`'s calibration table) — the
review correctly identified this as the same saturation pattern §6.4 already documented
("population-level calibration can look perfect while individual outliers are badly
miscalibrated"). Re-ran with row-level metrics (n=16,368 meaningfully-involved TEST
player-weeks) at thresholds ≥3/≥5/≥7:

| Threshold | Bootstrap (Brier / log-loss) | Naive (Brier / log-loss) | mean \|Δprob\| |
|---|---|---|---|
| ≥3 | 0.16753 / 0.50388 | 0.16895 / 0.51199 | 0.049 |
| ≥5 | 0.14325 / 0.44453 | 0.14526 / 0.46217 | 0.060 |
| ≥7 | 0.09948 / 0.32125 | 0.10118 / 0.33876 | 0.045 |

Real, consistent, but individually modest per-metric improvements (log-loss ~1.6-5% relative)
— row-level Brier/log-loss agree directionally with the old bucket-level correlations but,
correctly, don't look nearly as dramatic once not aggregated away.

**Tail calibration at ≥7 targets, top 5% of predicted probability** (the bucket that actually
gets bet): naive Binomial's predicted mean is 0.907 against a realized rate of 0.829 — a real
7.8-point overshoot, i.e. naive is overconfident exactly where it matters most. The bootstrap
simulator's predicted mean is 0.864 against the same 0.824 realized rate — only a 4.0-point
overshoot, essentially halving the tail overconfidence. **This is the single most decision-
relevant number in this section** — it's real evidence the bootstrap approach is doing
something naive Binomial structurally can't (respecting the heavy right tail documented
above), concentrated exactly where a bettor would actually transact.

**Vs.-the-vig framing, and an honest correction of this round's own prior expectation.** A
standard two-way prop line holds ~4.55% (−110/−110). The mean |probability gap| between the
two methods at ≥7 targets is **4.5 points** (median 3.9, 95th percentile 9.6) — comparable in
size to the vig itself, not clearly smaller than it. This round's plan, following the
review's own stated expectation, predicted the improvement "almost certainly" wouldn't clear
a realistic vig — **the actual numbers don't support that specific prediction as confidently
as expected**. The per-bet disagreement between the two methods is large enough that the
choice between them could plausibly flip which side of a real line looks profitable for a
meaningful share of bets, particularly in the tail where the calibration gap is largest. This
doesn't prove the simulator is profitable — that requires real market odds data, which this
project still doesn't have (§13.3, the still-ungated prop-odds experiment, §14/plan R6.13) —
but the honest, walk-forward-consistent conclusion is **"plausibly large enough to be worth
testing against real market odds," not "too small to matter,"** a genuine update from what was
expected going in, not a confirmation of it.

---

## 10. Historical validation / tuning scripts

### 10.1 `src/models/tune.py`
Nested grid search over `(alpha, off_shrink, def_shrink)`: tune-train `{2018-2020}` fits
calibration for each candidate, tune-val `{2021}` picks the winner, final test `{2022-2025}`
is touched exactly once. Grid: `alpha∈{0.06,0.08,0.10,0.12,0.15,0.20}`,
`off_shrink∈{0.20,0.35,0.50}`, `def_shrink∈{0.35,0.50,0.65}`. Chose the current production
defaults (§3.1). A later continuous re-optimization attempt (scipy, not this grid) was
tested and **rejected** — see §13; the coarse grid generalizes better than a fine
continuous search here, a real overfitting-to-a-small-validation-season finding.

### 10.2 `src/models/backtest.py`
Simpler, quicker sanity check on the current tuned defaults (no hyperparameter search, just
calibration fit + evaluation on the same TRAIN/TEST split). Superseded by `tune.py` for
anything requiring hyperparameter search, kept for fast iteration.

### 10.3 `src/models/predict.py`
Full Layer 1 + QB-swap + injury pipeline, run as a module for a concrete demonstration
against a real already-played week (`DEMO_SEASON=2025, DEMO_WEEK=18`), compared against
both the actual result and the Vegas closing line. Also dumps `current_team_ratings.parquet`
— "this is what would seed the next season's Week 1."

### 10.4 `src/models/predict_2026.py`
The original hardcoded Week 1, 2026 forecast script (superseded for ongoing use by
`weekly_update.py`, kept as a reference and for the one-time preseason forecast). Contains
`find_qb_changes()` and `build_qb_name_to_id()` (§5.2), imported directly into
`weekly_update.py` rather than duplicated.

### 10.5 `src/utils/rolling_origin_cv.py` — rolling-origin CV harness (added 2026-07, review #3.1)

Every validation in this project up to this point used a single fixed TRAIN/TEST split
(2018-2021 / 2022-2025) — honest and walk-forward-safe, but a single point estimate can't
distinguish "real, stable signal" from "happened to work on this one split." This harness
(`rolling_origin_folds`, `evaluate_rolling_origin`, `sign_consistency`) instead walks the
origin forward one season at a time, refitting on everything strictly before it and scoring
the next season — producing a per-season fold that reveals whether an effect is consistent
fold-to-fold or fragile, reusable by any future validation script (`fit_fn`/`score_fn`
callbacks, no other assumptions about what's being validated).

**Demonstrated immediately on the exact case it was built for**: `market_blend.py`'s own
docstring already documents its margin-blend weight as multicollinearity-unstable on small
windows (§4.2). Running it through this harness (8 folds, 2018-2025, each trained on all
prior seasons back to 2016) makes that concrete: `our_weight`'s **sign consistency is 0.75**
(flips negative in the 2018 and 2019 test folds, positive from 2020 onward), ranging from
-0.169 to +0.134 across folds. More strikingly, **the blend beats market-alone in only 2 of
8 folds** — a second, fully independent line of evidence (rolling-origin CV, vs. the
proper-scoring-panel comparison that drove §4.3's decision) confirming the margin blend
never reliably added value, not just on the one TEST split already examined.

### 10.6 Pristine 2026 holdout convention (review #3.1)

Every historical validation in this project deliberately touches `TEST=2022-2025` only once
per question asked (this project's own long-standing discipline). **2026, once real games
start, is reserved as a genuinely prospective holdout on top of that** — it must never be
used to fit, tune, sweep, or select any coefficient, hyperparameter, or model choice, for any
component, at any point. It exists specifically so this project's own real forward
performance can be checked honestly against real outcomes nobody could have curve-fit to,
which no amount of held-out-but-already-fully-explored historical data (2022-2025 has now
been directly analyzed in dozens of experiments — see §13's counter below) can substitute
for. Once a CLV (closing-line value) tracking framework exists — proposed, not yet built,
requires acquiring historical opening-line data; see §13.1.1's review-round-2 tracking for
this item — 2026 is also the natural tracking window for it. (This doc's own §5.2 is the QB
offseason bootstrap, unrelated — fixed a stale cross-reference here 2026-07.)

---

## 11. Live automated pipeline

### 11.1 `src/pipeline/weekly_update.py`
The orchestrator. Refreshes all data with `force=True` every run (a few minutes of runtime
buys a lot of robustness — no partial-file merge logic to get wrong), rebuilds every engine
from scratch via walk-forward (Layer 1, QB ratings, injury flags, player usage/TD-rate/
yards/receptions, QB passing engines, market blend, wind adjustment, pass rate), auto-
detects the next unplayed week (`find_next_week()`), and predicts margin/total + full player
props for it. `current_nfl_season()` handles the season-year convention (a "2026 season"
runs Sep 2026-Feb 2027; games Jan-Jun belong to the *prior* season year).

Output is printed **organized by game** (not a global cross-game leaderboard) — every game
score comes with both teams' complete expected statlines, directly satisfying "for every
game score there should also be an expected statline." OUT players print with an explicit
note ("touches reallocated to teammates below") rather than silently vanishing.

### 11.2 Two scheduled runs, not one
- **`nfl-weekly-predictions`** (Tuesdays): the early-week refresh.
- **`nfl-gameday-refresh`** (Sundays, 9:21am local): added specifically because the wind
  adjustment (§8) and injury/roster-status detection (§6.2) are both structurally no-ops
  until information resolves close to kickoff — Friday's "Final" injury report is the last
  official designation before Sunday games, and weather forecasts aren't reliable before
  Fri/Sat. **Honest, explicitly-documented limitation**: NFL games are spread across
  Thursday night, Sunday (1pm/4pm/8pm ET), and Monday night — this single Sunday-morning
  run cannot catch a true last-minute inactive announced 90 minutes before one *specific*
  kickoff (that would require separate runs timed to each individual game window, which
  isn't built). It catches the bulk of real information, which research shows resolves by
  Friday anyway, not true T-90 precision for every game.

Both runs execute the identical `weekly_update.py` + `build_predictions_page.py` sequence —
the only difference is *when* they run, which changes what real-world information is
available to fetch. No separate "lightweight refresh" code path was built; reusing the full,
already-tested pipeline was judged simpler and more robust than a new incremental-state
mechanism.

### 11.2.1 Self-generated CLV line-snapshot log (review round 3, #2)

A real, zero-cost gap, caught before the season made it expensive: `predictions_{season}_
wk{week}.parquet` is **overwritten** every run, so every prior Tuesday-vs-Sunday line
snapshot this project has ever produced was already being discarded — despite the two
scheduled runs above (§11.2) straddling exactly the window CLV (closing-line value) measures,
at zero additional cost, no external data, no purchase.

Fixed: every run now **appends** one row per game to `data/processed/line_snapshots.parquet`
(`game_id, run_timestamp, season, week, market_spread, market_total, our_margin, our_total,
home_cb_flag, away_cb_flag, qb_swap_flag, sim_home_win_prob`) instead of only writing the
current-run predictions file. Confirmed working on two consecutive live runs (16 rows each,
32 total, 2 distinct `run_timestamp` values, all fields populated correctly).

This unlocks, once the season starts and Tuesday/Sunday runs actually diverge: whether
CB-flagged games move toward the model's adjustment between the two snapshots (a direct read
on whether the CB signal is information the market eventually incorporates, or noise),
realized CLV per flagged bet at the ~46 CB-flagged games/season this project already
identifies, and a real prospective data trail for the pristine-2026-holdout convention
(§10.6) — previously a stated convention with no data collection actually behind it. Round
2's own statistical-efficiency argument still applies: distinguishing a 2% ATS edge from
breakeven at 46 bets/season needs on the order of 2,400 bets, while CLV's much lower
per-observation variance gives a usable read (t≈1.7) in a single season. Historical opening
lines (a paid/external data source, still gated per §14) would be strictly better, but this
self-generated version needed no approval and no budget — just not throwing away a file the
pipeline already produces.

**Post-game reconciliation added (review round 4, #8) — done before Week 1 kicks off, per
the review's own deadline framing.** The log above wasn't actually scoreable yet: it recorded
`market_spread`/`market_total` at *run time*, not the *closing* line CLV requires, and had no
final score to grade realized ATS against. Fixed with two additions:
- `bet_side` (`"home"`/`"away"`/`"push"`, derived from the sign of `our_margin − market_spread`
  at snapshot time) — recorded immediately rather than reconstructed later, removing any doubt
  about sign convention when scoring afterward.
- Every pipeline run now also re-merges the *entire* accumulated `line_snapshots.parquet`
  against the current schedule's `home_score`/`away_score` (and that same completed game's own
  `spread_line`/`total_line`, which — once a game is final — **is** the closing line, already
  fetched every run, no new data source). Adds `closing_spread`, `closing_total`, `home_score`,
  `away_score` wherever a game has since finished. This is a cheap merge, not a refetch, so a
  game reconciles automatically the very next pipeline run after it completes — confirmed
  working (0/112 rows reconciled on a live run before Week 1 2026 games exist, as expected;
  the merge logic itself verified directly against a real completed historical game).

Realized CLV per flagged bet is now `(closing − snapshot)` signed by `bet_side`; realized ATS
follows directly from `bet_side` vs. the actual final score. The pristine-2026-holdout
convention (§10.6) is now operational, not just stated.

### 11.3 `src/pipeline/build_predictions_page.py`
Generates a self-contained, dark/light-theme-aware HTML page (`TEAMS` dict: full name +
primary/secondary hex per team, used for card accent colors). One card per game: score,
winner highlighted, plus a collapsible `<details>` "Expected statlines" section per game
(compact QB line + top skill players by TD probability) — native HTML disclosure, no JS
framework needed. `_statline_rows()` explicitly excludes `status_note=="OUT"` rows from the
ranked display. Published as a Claude Artifact, updated in place across weekly runs (same
URL every time — the scheduled-task prompt explicitly instructs WebFetching the current
version before republishing, since a fresh session hasn't "seen" it and the publish would
otherwise be rejected).

**Real gap found and fixed (2026-07, no review prompted this — a direct audit of what's
actually shipped vs. what's just sitting in a parquet file):** the game simulator's
moneyline, win probability, and margin/total distribution (§9.5 — built round 2, recentered
and total shipped round 4) were computed and validated for two full review rounds but never
actually rendered on this page. Added: a moneyline badge next to each team (`sim_home_
moneyline`/`sim_away_moneyline`), and a confidence line under the score
("SEA 62% to win · ±14 margin / ±13 total", from `sim_home_win_prob`/`sim_margin_std`/
`sim_total_std`) — all gracefully blank when a market line isn't published yet for a game,
same fallback convention as margin/total themselves. Verified visually in-browser against the
real live Week 1 2026 page: badges and confidence text render cleanly at both 2- and 3-digit
odds, no layout breakage.

### 11.4 `data/manual_overrides/known_outs_2026.json`
The stopgap for verified breaking news the automated roster-status data hasn't caught up to
yet (§6.2). Format: `{team, player_name, reason, expected_return, verified_date, source}`.
The scheduled-task prompts explicitly instruct checking each run whether an entry's
`expected_return` window has passed or whether real roster status has independently caught
up (in which case flag it as possibly-redundant rather than leaving it stale silently), and
allow adding new *verified* (web-searched or user-confirmed) entries — never unverified
claims.

---

## 12. Recurring engineering patterns and gotchas worth knowing

### 12.1 The silent partial-fetch bug (§2.2)
`nfl_data_py`'s multi-season loaders can drop a season with zero exception. Always verify
season-completeness after any bulk fetch before trusting the result; `_cached()` now does
this automatically, but if a *new* loader function is ever added outside `fetch.py`, it
needs the same guard.

### 12.1.1 End-to-end hand audit of one game's props (review round 2, #4.2)

Three of the four most consequential bugs this project has found (the game-script sign flip,
the aDOT scale mismatch, the Judkins name-collision duplication) were scale/sign/identity
errors that a single careful trace would have caught, some live for extended periods. Rather
than wait for the next one, traced one real game's full props chain **by hand**, recomputing
every intermediate value independently (not just reading the code) and comparing against the
actual live pipeline's printed output: Rashee Rice (KC WR), 2026 Week 1 vs. DEN
(`game_id=2026_01_DEN_KC`, KC home, market_spread=3.0, market_total=42.5 — both used directly
as `our_margin`/`our_total` here, no QB-swap or CB flag active for this specific game, so a
clean pass-through case).

Recomputed independently from raw engine `.predict()` calls (target_share=0.3077,
catch_rate=0.7296, yards/target=8.225, rec_td_rate=0.0551, carry_share=0.0142, KC base
pass_rate=0.6026) through the game-script adjustment (implied team total 22.75 from the
market line, `script_adj=-0.0025`) to `pass_attempts=37.72`/`rush_attempts=25.14`, and
independently confirmed against the live pipeline's `props_2026_wk1.parquet`:

| quantity | hand-computed | pipeline | match |
|---|---|---|---|
| proj_targets | 11.6 | 11.6 | ✓ exact |
| proj_receptions | 8.5 | 8.5 | ✓ exact |
| proj_rec_yards | 95.5 | 95.5 | ✓ exact |
| proj_carries | 0.4 | 0.4 | ✓ exact |
| td_probability (rec-only approx) | 0.431 | 0.438 (rec+rush) | ✓ gap fully explained by the small rush-TD term omitted from the by-hand approximation |

Clean result — no scale/sign/identity error found in this trace. This doesn't prove every
game/player is bug-free, but it's real, positive evidence for the specific chain audited
(share engines → pass-rate/game-script adjustment → volume → per-touch rate engines →
yards/receptions/TD probability), not just an assertion that the code "looks right."

**Re-audited 2026-07** after rounds 3-4 changed the calibration mechanism (auto-refit linear,
§9.4) and the game simulator (recentering, half-boundary modeling, §9.5) — same discipline,
not just re-reading the code. Traced KC (home) vs. DEN, `our_margin`/`our_total` =
market-implied (3.0/42.5, no active CB/QB-swap flags): Rashee Rice's `td_probability=0.430`
back-solved cleanly through the live `TD_PROB_CALIB_A/B=(0.042, 0.820)` auto-refit values to
`raw_td_prob≈0.473` → `proj_rec_tds≈0.641` at a ≈5.6%-per-target rate, plausible for a
high-usage WR1 given `rec_td` EB `prior_weight≈269`. Game-level: `sim_margin_std=14.46`
(real historical range confirmed ~13-14, §9.5), `sim_total_std=12.17` (plausible for real
NFL total-points dispersion), `our_home_pts + our_away_pts` exactly equals `our_total`.
Clean again — the calibration and simulator changes since the last audit didn't introduce a
new scale/sign issue in this chain.

Two minor, non-bug findings from this pass, worth recording rather than fixing:
- `our_home_pts`/`our_away_pts` are each independently rounded to 1 decimal from the
  underlying `our_margin`/`our_total`, so `our_home_pts − our_away_pts` can disagree with the
  displayed `our_margin` by up to ~0.1 point after double-rounding (confirmed: 2.90 vs. 3.00
  on this exact game). Pre-existing since Phase 0, purely cosmetic, far too small to affect
  any real decision — not worth the code churn to fix.
- `DriveOutcomeSampler.sample_end_of_half()` would raise (empty-pool `rng.integers(0, 0)`) if
  ever constructed without real `end_of_half_drives` data — currently unreachable (the sole
  call site, `build_simulator_for_season_range`, always supplies real data, and the dataset
  has 4,077 real end-of-half drives), but a latent fragility if a future caller ever
  constructs `DriveOutcomeSampler` directly against a small/filtered dataset.

### 12.1.2 Injury-flow rehearsal (2026-07, ahead of the season — no real 2026 injury data exists yet)

Two checks, one real, one synthetic (real 2026 injury-report data still 404s from
`nfl_data_py` as of this writing, so a live end-to-end rehearsal isn't possible yet):

- **Real**: the manual-override mechanism (§6.2, `known_outs_2026.json`) already has a real,
  verified entry (Zach Charbonnet, SEA RB, PUP list) sitting in production right now.
  Confirmed directly against the live `props_2026_wk1.parquet`: Charbonnet renders with
  `status_note="OUT"`, `proj_carries=0`, and his workload is correctly reallocated among
  SEA's remaining backs (Emanuel Wilson 12.2 carries, Jadarian Price 17.5 via the
  draft-capital/Clay rookie fallback, etc.) — the whole `out_player_ids_for_team` →
  `reallocate_shares` chain is confirmed working on real, live data, not a hypothetical.
- **Synthetic** (no real CB-out flag exists yet): injected an away-team CB-out flag for a
  real game (KC/DEN, market_spread=3.0) and traced the full downstream chain by hand.
  `apply_joint_adjustment` correctly produced `+2.428` (`-0.018 + 2.446×(1-0)`, exactly
  matching the current production coefficient), `our_margin` correctly moved 3.00→5.43, and
  the recentered game-simulator win-probability/moneyline correctly followed
  (58.3%/−140 → 67.3%/−206) — confirming the CB-flag → margin-adjustment → recentered-
  simulator chain is wired correctly end-to-end, not just each piece in isolation.

Recommend repeating the synthetic check (or, better, confirming directly) once real 2026
injury-report data actually starts publishing — this rehearsal is real verification of the
mechanism, not a substitute for watching it fire on a genuine first case.

### 12.2 The `np.polyfit(...)[::-1]` slope/intercept swap — now fixed via `src/utils/stats.py`
`np.polyfit(x, y, 1)` returns `[slope, intercept]` (highest degree first). A `[::-1]`
reversal, if the receiving variable names aren't *also* swapped to match, silently produces
an intercept and slope in each other's place — plausible-looking but completely wrong
calibration. This happened **three times** in this project: once during a continuous-
hyperparameter re-optimization test, once during the wind-adjustment investigation (§8,
caught both times by noticing the result didn't match a raw-correlation sanity check) — and
a **third, previously-undetected instance** found by a 2026-07 housekeeping audit (review
recommendation #3.4): `weekly_update.py`'s and `validate_props_pipeline.py`'s game-script
pass-rate calculation (§7.3) had shipped with this exact bug, live, for as long as
`weekly_update.py` has been the pipeline — flipping the sign and roughly doubling the
magnitude of the game-script pass-rate adjustment used for every prop prediction.

**Fix**: every raw `np.polyfit` call in the codebase (11 call sites) has been replaced with
`src/utils/stats.py`'s `fit_linear(x, y) -> (intercept, slope)`, which fixes the return order
unambiguously (no more per-call-site reversal decision to get wrong) and **asserts the fitted
slope's sign matches the raw correlation sign**, raising immediately if a future edit
reintroduces this bug rather than letting it ship silently. Every `fit_calibration()`-style
helper across the codebase (`market_blend.py`-adjacent, `qb_adjustment.py`, `game_environment.py`,
`predict.py`, `predict_2026.py`, `tune.py`, `backtest.py`, `weekly_update.py`) now delegates
to it. **Always sanity-check a fitted sign/magnitude against a raw correlation before
trusting any linear-fit result** — this is now enforced automatically, not just a reminder.

### 12.2.1 Stale-coupling sweep (review round 2, #4.2)

Generalizing R0.1's `TD_PROB_CALIB` bug (a downstream calibration correction silently
invalidated by an upstream shrinkage change) into a systematic pass: enumerated every
hardcoded calibration constant in the codebase alongside the upstream quantity it depends on,
checking for drift.

**Found and fixed**: `LEAGUE_AVG_PLAYS = 62.859` — hardcoded, identically, in THREE files
(`weekly_update.py`, `predict_props_2026.py`, `validate_props_pipeline.py`), and unlike every
other rate/coefficient in this pipeline (rating engines, pass rate, wind coefficients, share
engines — all rebuilt fresh every run), never recomputed as new seasons accumulate. Confirmed
real drift: 2016-2025 full-history mean is 62.40 plays/team-game, but 2024-2025-only is
**61.04** — a real, plausibly genuine pace-of-play trend, not noise (and confirms where 62.859
came from: TRAIN=2018-2021's own mean is 62.86, matching almost exactly — this constant was
never wrong when derived, it just never got refreshed). Fixed in the two places that should
track current conditions: `weekly_update.py` now computes it fresh from CALIBRATION_SEASONS
each run (2022-2025 → 61.70 on the live run that produced this), and
`validate_props_pipeline.py` now computes it from TRAIN=2018-2021 only, matching its own
walk-forward discipline (confirms 62.86, consistent with the original literal).
`predict_props_2026.py`'s own copy is left frozen — a one-time historical reference for that
specific preseason forecast, same pattern as `SWAP_B_LAYER1`/`TD_PROB_CALIB` there.

**Checked and confirmed NOT stale**: `JOINT_COEFS_FORWARD`/`SWAP_B_MARKET` were fit against
"actual_margin − blended_pred" back when the margin blend still existed (blend weight on
market was already 88-98%, so blended_pred was already very close to the pure market line) —
now that the blend is fully removed and production margin is exactly the market line, this
was worth checking directly rather than assuming it still holds. It does: R1.3's ATS panel
(§6.1.1) tested the CURRENT production formula directly (market line + these exact
coefficients) and found a real, strong result (62.4% ATS) — an independent empirical
re-confirmation, not just an argument that the residual bases are "close enough." QB passing
stat prior weights (`qb_passing_stats.py`: `YARDS_PRIOR_WEIGHT`/`TD_PRIOR_WEIGHT`/
`INT_PRIOR_WEIGHT`, all still swept constants) have no downstream calibration correction
depending on them, so there's no analogous coupling risk there — unlike `TD_PROB_CALIB`,
nothing would silently go stale if these were later re-fit via empirical-Bayes.

**Re-swept 2026-07** across everything rounds 3-4 touched (margin/total recentering, the
explicit half-boundary drive model, the auto-refit-linear TD calibrator, CLV
snapshot/reconciliation): all clean, no analogous bug found. `JOINT_COEFS_FORWARD`/
`SWAP_B_MARKET` are *deliberately* frozen rather than auto-refit (see the constant's own
comment in `injury_adjustment.py`/`weekly_update.py`) — a considered exception to this
project's "recompute fresh every run" convention, not an oversight, given how much scrutiny
that specific coefficient required to arrive at. One dormant robustness gap found (not a live
bug, see §12.1.1's re-audit): `DriveOutcomeSampler.sample_end_of_half()` would crash on an
empty end-of-half pool, unreachable via the current single call site.

### 12.3 Walk-forward, no-lookahead discipline, everywhere
Every engine's `predict()` is called *before* `update()` for a given row, and every
`run_walk_forward()` sorts by `(season, week)` first. Historical validation always uses a
TRAIN/(VAL)/TEST split, with no lookahead within a single fit — but "touched at most once"
describes the *discipline of any one fit*, not a claim that TEST has only been looked at
once in this project's history. It has been re-examined many times across many distinct
hypotheses (~60-70 and counting — §13.0's running count is the honest figure here, not this
section). Forward-looking production calibration deliberately uses a different, more recent
window (§3.3) for documented substantive reasons, not by accident.

### 12.4 Bayesian/Laplace shrinkage as the one general-purpose tool
Every per-touch rate in this project (TD rate, yards/touch, catch rate, all 4 QB passing
stats, defensive rates in rejected experiments) uses the identical `(hits + pw*league_rate)
/ (touches + pw)` formula via `TdRateEngine`, just with a different `prior_weight` swept per
statistic. This is deliberate — one well-understood, well-tested mechanism reused
everywhere, rather than a different bespoke smoothing method per stat.

### 12.5 Prefer walk-forward re-fit over hardcoded coefficients
Market blend weights, wind coefficients, margin/total calibration, and game-script slope
are all recomputed fresh from real data on *every* pipeline run (not hardcoded from a one-
time fit) — this is what lets the dynamically-growing fitting windows (§4.2) and the
season-by-season recalibration actually improve over time without code changes.

---

## 13. Complete ledger: kept vs. rejected vs. investigated-not-built

### 13.0 Running experiment counter (review #3.1, added 2026-07)

This ledger is already a de facto pre-registration record — every hypothesis tested against
`TEST` (historical 2022-2025, or `CALIBRATION_SEASONS` for forward-looking fits) gets recorded
here whether it worked or not. Making the count explicit matters because of real multiple-
comparisons exposure: at a conventional p<0.05 threshold, a handful of false positives are
expected purely by chance across enough independent tests.

**Approximate running count: ~75-85 distinct TEST-window experiments** as of 2026-07 (the
original review estimated ~30-40 pre-review; round-1 implementation added ~25-30 more; round-2
added roughly 10-15 (TD-calibration refit, EB ≥100-touch diagnostic, the CB ATS/rolling-origin/
snap-rank panel, the pending totals panel) — some genuinely single hypotheses, some small
multi-part investigations counted as one entry each). Given this base rate, expect on the order
of 3-4 "significant" results in this ledger to be false positives, not real effects. **Items
worth the most skeptical re-look under this lens** — not because anything specific is wrong
with them, but because their statistical margin is thin relative to how confidently they're
currently treated:
- Interceptions rate (§5.3) — kept for statline completeness, MAE improvement ~0.15%,
  explicitly documented as "essentially noise" already.
- Passing-TD rate (§5.3) — kept, MAE improvement ~0.9%, the weakest of the four QB passing
  stats.
- The QB swap_delta coefficient post-refit (§5.1, §6.1.1) — t=+1.52 to +1.53 across two
  pooling methodologies, consistently just under conventional significance; kept on mechanism
  plausibility (a stale-rating-catches-up effect is well-established) rather than a clean
  p<0.05, which is a real, stated judgment call, not a strength. Round-2's direct ATS test on
  the QB-swap-flagged subset confirms this thinness rather than resolving it (55.0%, CI
  crosses 50% — §6.1.1).
- The CB-flagged snap-rank discriminator (§6.1.1, round-2) — inconclusive at n=37-103 per rank;
  doesn't confirm the hypothesized single-corner mechanism, but the sample is too small to rule
  it out either.

By contrast, the strong, high-confidence results in this ledger (wind's effect on QB/WR
yards-per-touch at p=0.0022/p<0.0001, RB reallocation at p=0.0003, the draft-capital rookie
prior at p<0.0001 on both target and carry share, the game-script market-implied-total
co-input replicated across a fixed split AND independently via rolling-origin CV in §10.5, and
now the CB adjustment's own ATS%=62.4% with a CI clearing breakeven even at its lower bound,
§6.1.1 round-2) are not the ones this multiple-comparisons caveat should make anyone doubt.

### 13.1 KEPT / LIVE in production
- Layer 1 recursive opponent-adjusted EPA power ratings, tuned `alpha=0.06/off_shrink=0.20/
  def_shrink=0.50` (§3, §10.1)
- Two-window calibration convention: historical TRAIN/TEST vs. forward CALIBRATION_SEASONS
  (§3.3)
- Real Vegas closing line (`spread_line`/`total_line`) used directly for both margin and
  total, Layer 1's own calibration kept only as a fallback when no line is published yet —
  **the Ridge-blend machinery (`market_blend.py`) is no longer live for either quantity as of
  2026-07: margin blend removed in round 1, total blend removed in round 2 on the identical
  panel/kill-criterion, see §13.1.2** (§4.3)
- QB in-season swap adjustment, `SWAP_B_MARKET=2.970` as used by `weekly_update.py` (refit
  against the blend residual 2026-07; `predict_2026.py`'s own `SWAP_B_LAYER1=6.616` stays
  frozen against the pure Layer-1 residual — renamed 2026-07 review round 2 to make the
  residual basis explicit rather than share a bare `SWAP_B` name; see §5.1, §13.1.1)
- QB offseason-swap Clay bootstrap, Week-1-2026-only scoped (§5.2)
- Full QB passing statline (§5.3)
- 4-signal injury adjustment for game margin: skill/OL/CB Out flags, top-3 CB — **as
  originally validated (`JOINT_COEFS`, historical backtesting only); forward production uses
  the reduced, refit `JOINT_COEFS_FORWARD` v3, see §6.1.1** (§6.1)
- Injury/PUP/IR detection + position-specific reallocation: RB yes, WR/TE no (§6.2)
- Rookie fallback via Clay projections, triggered on zero-history not roster-absence (§6.3)
- TD-rate/catch-rate `prior_weight`, empirical-Bayes-fitted (superseding the earlier swept
  15→30) — rec_td ~210-269, rush_td ~148-194, catch_rate ~41-45 (§6.4); the downstream
  TD-probability calibration (`TD_PROB_CALIB_A/B`) is **auto-refit linear every pipeline run**
  (tried isotonic in round 3, reverted in round 4 after it underperformed linear on a strict
  walk-forward test; Beta calibration also tested and didn't beat linear either — see §9.4)
- Flat league-average pace, no team-specific model (§7.1)
- Neutral-script pass rate EWMA + game-script margin-based adjustment, **+ market-implied
  team total as a second regressor (2026-07)** (§7.2-7.3)
- Dome/indoor total-points bump (§7.4)
- Wind adjustment for QB yards/attempt and WR/TE yards/target specifically (§8)
- Target/carry share (ShareEngine), TD rate/yards/catch-rate (TdRateEngine) (§9)
- TD-probability calibration: isotonic regression, refit every pipeline run (§9.4)
- Garbage-time filtering for Layer 1 EPA calculation (validated earlier in project history)
- Doubtful-status and nickel-CB refinements to the injury signal (top-3, not top-2) (§6.1)

### 13.1.1 Third-party review implementation, Phase 0 (2026-07)

A third-party technical review of this model (`MODEL_REVIEW_2026-07.md`, reviewed against
this documentation) produced a prioritized, phased implementation plan. Phase 0 ("foundation
fixes — changes what 'better' means for everything after") is complete:

- **Margin blend removed, total blend kept** (§4.3) — proper scoring-panel evidence (ATS%,
  CRPS, bootstrap-CI signed bias, not just MAE) showed the margin blend's ATS win rate
  (49.6%) is statistically indistinguishable from a coin flip and carries a real negative
  bias market-alone doesn't have. Confirmed with the user before implementing (an
  architecture change, not a coefficient tweak). Total's blend showed no such harm and was
  left as-is.
- **QB-swap/injury coefficients refit against the blend residual** (§5.1, §6.1.1) — both were
  originally validated against the pure Layer-1 residual, but the live pipeline applies them
  on top of the (now partially removed) blended prediction; refitting confirmed real
  double-counting (`SWAP_B` roughly halved; 2 of 4 injury flags no longer significant).
- **Symmetric CB injury coefficient** (§6.1.1 v3) — reparameterizing home/away CB effects as
  one coefficient roughly doubled statistical power and improved held-out calibration.
- **`src/models/scoring.py` added** — ATS%/O-U%, CRPS, signed bias with bootstrap CI,
  Brier/log-loss (Gaussian approximation pending a real distribution layer). This is the
  metric panel that surfaced the margin-blend finding above; use it, not bare MAE, for any
  future margin/total evaluation.
- **`np.polyfit` slope/intercept bug, 3rd instance found and fixed** (§12.2) — a real, live
  bug in the game-script pass-rate calculation (§7.3), plus a new `fit_linear()` helper with
  a built-in sign assertion, now used at every linear-fit call site in the codebase.
- **QB/player name-matching cascade extracted** to `src/ingest/name_matching.py` — previously
  defined in, and imported backwards from, the one-time reference scripts (`predict_2026.py`,
  `predict_props_2026.py`) into the live pipeline (`weekly_update.py`).
- **Refresh-window "material miss" question reframed and measured** (§13.3) — the originally-
  proposed measurement (compare a Sunday snapshot against final inactives) isn't answerable
  with available data (no point-in-time injury snapshots exist, only one row per player-week);
  measured a proxy instead.

**Phase 1** ("cheap wins using data already in hand") is also complete:
- Market-implied team total added as a game-script co-regressor (§7.3 v2) — real, modest,
  consistent-across-seasons improvement (p=0.04).
- Kicker quality rating for totals — investigated, clean null (§7.4).
- Air yards/aDOT/WOPR for props — investigated; the first validation pass had a real,
  self-caught methodology bug (a flawed pass-attempts proxy) that manufactured a false-positive
  result briefly shipped and then reverted; properly re-validated, both came back null (§9.2.1).

**Phase 2** ("score-distribution Monte Carlo layer," the review's biggest-lift item) has a
real, working, honestly-validated first build — margin/win-probability distributions
competitive with the current point estimate, total scoring still behind — see §9.5 for the
full three-iteration writeup. Not yet wired into the live pipeline.

**Phase 3** ("props simulator + prior-fitting refinement") is complete:
- Empirical-Bayes fitted `prior_weight` (§6.4) — real, shipped win; TD-rate/catch-rate priors
  now derived from the actual population variance structure instead of a manual sweep, tames
  small-sample outliers meaningfully better than the previous swept constants.
- Compound props simulator (§9.6) — redirected away from the review's parametric Dirichlet-
  Multinomial spec after testing its core premise and finding real target-count variance is
  mixture-like, not uniformly overdispersed; built and validated a bootstrap-resampled
  alternative instead (beats naive Binomial calibration at every threshold tested). Built for
  target/carry count only; not yet extended to receptions/yards/TDs or wired into the live
  pipeline.
- Volume-based opponent effects (§6.5) — clean null on both total plays and red-zone trips,
  extending the already-established opponent-rate null to volume as well.

**Phase 4** ("Layer 1 signal quality") is complete — entirely null/rejected, but real and
thoroughly validated:
- EPA signal-quality (§3.4): early-down EPA null; winsorized EPA **significantly harmful**
  (p=0.0014, the opposite of the review's expectation — outlier plays carry real signal, not
  noise); turnover-neutralized EPA null; success-rate co-input involved a real self-caught
  lookahead-leakage bug (an apparent ~18% MAE "win" that evaporated once fixed to be
  walk-forward honest) and came back null.
- Rating-update mechanism (§3.5): declining-gain EMA decisively rejected across every
  configuration tested; market-anchored preseason prior null, with full replacement actually
  worse (including in the specific early-season window it targeted).

None of Phase 4's candidates are wired into the live pipeline — Layer 1's existing signal
construction is unchanged.

**Phase 5.1** (draft-capital rookie prior, §6.3.1) is complete and shipped — a real, validated
win (~18%/~16% MAE improvement over naive baseline) that also closes a genuine, previously-
existing gap (WR/TE rookies getting zero projected volume).

Remaining phases (opening-line/CLV infrastructure — requires acquiring a new external data
source, needs explicit user go-ahead before any download; methodology hardening — rolling-
origin CV, pristine holdout, running experiment counter) are tracked in the review-response
plan but not yet built as of this writing — see the plan file for the full phased breakdown
if resuming this work.

**Phase 5.3** (§13.0, this section's own running counter, rolling-origin CV harness,
pristine-holdout convention) is also complete — see §10.5, §10.6.

### 13.1.2 Third-party review, Round 2 (2026-07)

A second, narrower review (`MODEL_REVIEW_ROUND2_2026-07.md`) was done against the updated
documentation above. Its own scorecard on round 1: 8 hits / 6 misses, matching this project's
~80% null rate on new features — every round-1 hit was measurement/architecture/plumbing,
every miss was a new feature, which directly informed how this round is prioritized (live bug
and evaluation-gap items first, new-feature-shaped items last and gated behind explicit
external-data approval).

**Live-correctness items, complete:**
- **`TD_PROB_CALIB_A/B` refit against the current empirical-Bayes engines** — the constant
  `(0.046, 0.769)` was fit when TD-rate `prior_weight` was a swept 30; §6.4's EB fit (~150-270)
  is a ~7x-larger shrinkage the old constant wasn't calibrated against, a real double-correction
  bug, worst in the top decile. Re-ran `validate_props_pipeline.py`'s decile calibration with
  the EB-fitted engines (TRAIN-only fit, matching walk-forward discipline): refit came back
  `(0.022, 0.867)` — much closer to identity than before (deciles 0-8 now calibrate within
  ~1-2%, consistent with noise) but not fully at `(0,1)`: the top decile still overshoots
  (predicted 0.494 vs. actual 0.439, TEST=2022-2025) — a real, smaller residual, kept as a
  refit correction in `weekly_update.py` rather than deleted. `predict_props_2026.py` keeps
  its own frozen `(0.046, 0.769)`, correctly paired with that script's own frozen swept
  `prior_weight=30` — internally self-consistent as a historical reference.
- **`SWAP_B` naming disambiguated** — `weekly_update.py`'s `SWAP_B` (2.970, blend-residual
  basis) and `predict_2026.py`'s `SWAP_B` (6.616, Layer-1-residual basis) were the same name
  for two different-basis coefficients, the same trap class as the `np.polyfit` bug that's
  bitten this project three times. Renamed to `SWAP_B_MARKET`/`SWAP_B_LAYER1` respectively;
  `JOINT_COEFS`/`JOINT_COEFS_FORWARD` (`injury_adjustment.py`) got an explicit residual-basis
  comment at each definition instead of a rename, to avoid rippling into `predict.py`/
  `backtest.py`'s honest historical-backtest usage of `JOINT_COEFS` for little added safety.
- **EB `prior_weight` ≥100-touch diagnostic** — checked whether the population-wide EB fit
  (rec_td ~269 at the live `min_touches=20`) over-shrinks high-volume players by refitting
  restricted to players with ≥100 career touches: **prior_weight went UP, not down** (269→311
  rec_td, 194→240 rush_td) — the opposite of the over-shrinkage failure mode the review
  hypothesized. An independent cross-check (observed year-N→year-N+1 TD-rate correlation among
  receivers with ≥100 targets in year N: r=0.230, n=346) implies a persistence-based
  prior_weight of roughly 427 — same direction again. Honest conclusion: no over-shrinkage
  concern; if anything the current population-wide EB fit is mildly on the *conservative*
  side (less shrinkage than a high-volume-only view would suggest) for the high-volume tail,
  a small, safe-direction effect not worth a touch-volume-dependent `prior_weight` this round.

**Evaluation-gap items, complete:**
- **Full ATS panel on the CB-flagged / QB-swap-flagged subsets, plus rolling-origin CV on the
  CB coefficient and a snap-rank discriminator** (§6.1.1, `src/models/validate_adjustment_layer.py`)
  — the entire live betting edge of the game-side margin model now lives in this adjustment
  layer, and it had only ever been graded on MAE/signed bias. Result: the CB adjustment is a
  **real, statistically meaningful edge** (ATS%=62.4%, 95% CI [55.4%, 68.8%], clears breakeven
  even at the CI floor) with a stable, sign-consistent rolling-origin refit (1.000 sign
  consistency, magnitude range [1.59, 2.63]) — this is a genuine, newly-quantified strength,
  not just a closed measurement gap. The QB-swap adjustment's direct ATS test, by contrast,
  confirms rather than resolves its known thinness (55.0%, CI crosses 50%). The snap-rank
  discriminator came back inconclusive (small per-rank samples, 37-103).
- **Full panel on the total blend, matching margin's rigor** (§4.3,
  `src/models/validate_market_blend_totals.py`) — round 1 kept the total blend on MAE + a
  single p-value alone, the same thin evidence state that hid margin's harm. The identical
  panel found the same decisive failure: O/U%=50.1%, CI [47.2%, 53.0%], indistinguishable
  from a coin flip and spanning well below breakeven, plus a declining (though sign-stable)
  own-model weight in a rolling-origin check. **The total blend was removed**, same as
  margin, on the identical kill criterion the approved plan specified — production now uses
  `total_line` directly. This is a second real architecture change this round, not just a
  measurement exercise.

**Stranded assets, capability items:**
- **Game simulator's margin/win-probability + moneyline wired into `weekly_update.py`** (§9.5)
  — display-only addition alongside the existing point estimates; total deliberately not
  surfaced (still known-weaker than the top-down model). Sanity-checked on a real live run.
- **Props simulator re-evaluated with log-loss/tail-calibration/vs-vig framing** (§9.6) —
  produced a genuine correction to this round's own prior expectation: the bootstrap-vs-naive
  probability gap at the most decision-relevant threshold is comparable in size to a realistic
  vig hold, not clearly smaller than it, so this is plausibly worth testing against real market
  odds rather than "too small to matter" as originally expected going in.
- **Pace/volume-factor correlation fix for simulator totals: premise tested and NOT
  supported, not built** (§9.5) — checked whether the simulator's total-drives-per-game is
  under-dispersed relative to real games (the gap an added pace factor would need to fill).
  It is not — the current mechanism already produces MORE variance (12.85) than real games
  show (10.12). Surfaced a different, more direct clue instead: the simulated mean itself runs
  hot (24.22 vs. 22.11 real drives/game), a plausible lead on the total MAE gap for a future
  session. This is the same "test the premise before building" discipline as §9.6's
  Dirichlet-Multinomial check — a real negative result, not a skipped task.

**Audits, complete:**
- **End-to-end hand audit of one game's props** (§12.1.1) — traced Rashee Rice's full chain by
  hand for a real 2026 Week 1 game; every intermediate value matched the live pipeline exactly.
  Clean result, no bug found in this specific trace.
- **Stale-coupling sweep** (§12.2.1) — found and fixed one real, if modest, instance beyond
  `TD_PROB_CALIB`: `LEAGUE_AVG_PLAYS` was a hardcoded, triplicated, never-refreshed constant;
  confirmed real drift (62.40 historical mean vs. 61.04 in 2024-2025 only) and fixed the live
  pipeline to recompute it fresh each run. Checked `JOINT_COEFS_FORWARD`/`SWAP_B_MARKET` for
  the same risk given the margin blend's removal — confirmed still valid via R1.3's own direct
  empirical test, not just an argument that the residual bases are close.

**Remaining items** (the gated external-data items — opening lines/CLV, the narrow prop-odds
experiment) are tracked in the review-response plan, deferred pending explicit user go-ahead
per this project's standing rule on acquiring new external data.

### 13.2 TESTED AND REJECTED (do not re-suggest without a genuinely new angle)
- Kicker quality rating (distance-adjusted, Bayesian-shrunk) as a total-points regressor —
  clean null, raw corr ~0.02-0.03 with the base-model residual (§7.4)
- WOPR as a target_share replacement, and aDOT as a receiving-yards point-estimate
  correction — both initially showed false-positive improvements from a flawed validation
  proxy, reversed to null once corrected (§9.2.1) — a real, worth-reading methodology lesson
- Team-specific home-field advantage, including Denver altitude specifically — overfits,
  hurts held-out MAE (10.00→10.22, p<0.0001); even Denver's own effect was noise-level
- Game-script/pace effect on total points — near-zero correlation, hurts slightly if applied
- Continuous re-estimation of `alpha`/`off_shrink`/`def_shrink` via scipy optimization —
  overfits the small tune-val season; the coarse grid generalizes better (§10.1)
- 538-style unified single-team-rating structure (no offense/defense split) — underperforms
  the current split approach
- Opponent defense-allowed rate for player props, both aggregate and pass-rush/coverage-
  split versions — null or actively harmful (§6.5)
- DVP-style position-specific matchup grids — external research confirms this is a known
  trap, not a hidden edge (§6.5)
- Team-change / new-scheme "harm" hypothesis for player usage — tested opposite of expected
  (§6.5)
- Snap-count trend as a leading indicator of usage change — clean null (§6.5)
- Multivariate Ridge ensemble of props-specific weak signals — null, slightly harmful (§6.5)
- Bottom-up props-derived team totals as an input to game totals — null; also answers "can
  props help model scores" (no) (§6.5)
- Wind's effect on total points specifically — rejected (unstable sign); **note this is
  distinct from the validated per-touch-rate wind effect in §8**
- NFL Big Data Bowl player-tracking data (route separation, coverage tightness) — real
  season-long persistence, zero single-game predictive power once properly powered
  (272-game full-season test)
- Tree-based models (GradientBoostingRegressor, RandomForestRegressor) vs. linear/Bayesian-
  shrinkage — trees consistently lose, across every problem tested (margin, target share,
  TD rate); xgboost specifically never even loaded (missing libomp on macOS), never needed
  since sklearn's tree models already answered the research question
- Turnover luck (fumble recovery rate) as a regression signal
- Defenders-in-box numbers for rushing advantage
- Coach-specific 4th-down aggressiveness as a portable trait
- Referee crew tendencies
- Coaching-change effect (both for game margin alone and combined with roster turnover in
  a regularized ensemble — the ensemble's apparent "win" was entirely attributable to the
  QB-swap/injury signals already in production, with the new candidates diluting it)
- Roster/personnel turnover (season-to-season snap-weighted continuity) as a game-margin
  signal
- Surface type (grass vs. turf)
- Travel distance / time zone effects
- Divisional-game effect
- Clinched/eliminated rest effect (weeks 17-18)

### 13.3 INVESTIGATED, NOT BUILT (real limitation, not a rejected hypothesis)
- **Real player-prop market data**: unlike game-level spread/total (where real data was
  found hiding in `nfl_data_py`'s own schedule export), there is no equivalent source for
  player props anywhere in this project's data. External research: no credible academic
  literature on prop-market efficiency specifically; the one well-evidenced, narrow edge
  mechanism (secondary/backup props going stale after injury news, since pricing models
  react to the starter's line but lag on role changes) is exactly what §6.2's injury-
  reallocation system already targets structurally. Getting real player-prop odds requires a
  new paid data source — a genuine, separate decision. **Priced and scoped 2026-07**: The
  Odds API (the-odds-api.com) has a confirmed `player_rush_yds` market; its cheapest tier
  with player props is the $99/mo "Business" plan (200k requests/mo, NFL props during
  season) — real, current pricing, checked directly against their docs, not assumed. Player
  props require their per-event odds endpoint, not the bulk sport-level one. **The
  experiment's code is now built and tested end-to-end against mock data matching this
  real, documented response shape** (`src/ingest/fetch_prop_odds.py`,
  `src/models/prop_odds_experiment.py`) — identify a team with a real lead-RB-out
  (`identify_lead_rb_outs`, gated on the engine's own pre-injury carry share ≥15%), pull that
  team's reallocated backup projections straight from the live pipeline's own props output,
  match to the posted line by normalized name, snapshot (append-not-overwrite, same
  discipline as the CLV log, §11.2.1), and reconcile against real outcomes later. Confirmed
  working end-to-end on a synthetic scenario (Cleveland's real 2026 depth chart, one real RB
  marked Out, mock posted odds) before any subscription exists. **Still gated on the actual
  $99/mo purchase decision** — that's the only remaining step, not more engineering.
- **QB completion-rate wind adjustment**: real, correctly-signed raw correlation, doesn't
  clear significance at current sample size (p=0.34, n=1282) — plausible with more data,
  not shipped on a hunch (§8).
- **True per-kickoff-window refresh**: the Sunday gameday-refresh (§11.2) is a practical
  approximation, not literal T-90-minutes-before-every-game precision — building that would
  require several separate scheduled runs timed to each kickoff window across Thu/Sun/Mon.
  **Measured (2026-07), not just deferred**: the originally-proposed test (diff a Sunday-
  morning snapshot against final inactives) isn't directly answerable — `nfl_data_py`'s
  injury export has exactly one row per player-week (55,548/55,552 confirmed), i.e. no
  point-in-time history to diff against. Reframed proxy: for this project's 4 presumed-
  starter positions, how often does the FINAL injury report's "Out" designation disagree
  with what real stats/snap data shows actually happened? "Out" is extremely precise
  (0.0-0.25% of "Out"-flagged presumed starters had real recorded involvement anyway) but
  the false-negative rate (presumed starter NOT flagged Out, yet had zero real involvement
  that week) is substantial: 13.3% (WR/TE), 16.3% (RB), 26.5% (OL), 28.0% (CB). This is
  **not** clean evidence for building a per-kickoff refresh, though: most of that gap is very
  likely healthy scratches/committee rotation decisions with no official injury designation
  at all (categorically different from "report was stale, would have updated by kickoff"),
  plus real name-matching noise inflating the OL/CB numbers specifically. A closer-to-
  kickoff refresh wouldn't fix a healthy-scratch miss, since there's no injury report entry
  to refresh in the first place. Closing this idea on this basis rather than building it.

---

## 14. Genuinely open items for a future session

**Rewritten 2026-07 (review round 4, #9)** — the prior version of this section was stale by
three review rounds: every numbered item in its main list (market-implied totals, kicker/aDOT,
the Monte Carlo layer, empirical-Bayes priors, the draft-capital rookie prior) had already
shipped, and its closing paragraph cited "~30-40+" TEST-window experiments against §13.0's own
current count. Reviewed the same way §9.4's stale figures were caught this round: check the
canonical section describes *current* state, not history. This section now lists only what's
actually still open, with history and the full ledger living in §13.1.x.

**Gated on external data (unchanged for 4 rounds — needs explicit user go-ahead before any
download):**
1. Historical opening-line data + CLV tracking against it (§13.3) — the *self-generated*
   version (current-week snapshots, §11.2.1) is live and needs no gate; this item is
   specifically about acquiring a paid/external historical archive for a longer baseline.
2. A real player-prop market data source (§13.3) — reframed by review round 2 as a small,
   decisive experiment (a month of a cheap current-week odds API testing the validated RB
   reallocation result against posted lines, MAE 4.49→3.71, p=0.0003 — the project's strongest
   single result) rather than a large commitment. Round 3's props-simulator tail-calibration
   work (§9.6) found the bootstrap-vs-naive probability gap is comparably sized to a realistic
   vig hold — real reason to think this experiment could resolve something, not just a
   nice-to-have. **All the engineering for this is done** (§13.3) — pricing confirmed ($99/mo,
   The Odds API), the full identify→match→snapshot→reconcile pipeline built and tested
   end-to-end against mock data. The only remaining step is the actual subscription decision.

**Open questions from this project's own work, not yet resolved:**
1. **The CB adjustment's edge is held provisionally** (§6.1.1) — real-looking (permutation test
   against the full specification search puts it at ~95th percentile, not the ~97.5th a
   narrower test suggested), lookahead-clean, but not proven as strongly as round 3 first
   reported. The self-generated CLV reconciliation log (§11.2.1) is now live and will
   accumulate real, prospective evidence on this specific claim as the season plays out — the
   next honest update to this section should use that data once enough of it exists.
2. **The permutation null's off-center mean isn't fully explained** (§6.1.1) — walk-forward
   fitting moved it from 54.3% toward 52.4%, confirming resubstitution as a real partial cause,
   but a residual gap from 50% remains. Confirmed real team/season variation in CB-flag
   incidence exists (a plausible contributor); the exact mechanism isn't pinned down further.
3. **The CB snap-rank mechanism is unresolved** (§6.1.1) — the "specific cornerback's absence"
   story isn't supported by the rank discriminator (two-thirds of flags fire on the third
   corner), but the per-rank samples (37-103) are too small to rule it out either.
4. **QB completion-rate wind effect** — real, correctly-signed, underpowered; re-test once
   another season or two of data accumulates.
5. **Props simulator extension** (§9.6) — built and validated for target count only; the same
   architecture (`build_standardized_residuals`, `TouchCountSimulator`) directly reapplies to
   carries and, further downstream, receptions/yards/TDs conditioned on a simulated touch
   count, whenever a future session extends the full compound chain.

**Before testing a new hypothesis**: check §13 first. The base rate of "this plausible-sounding
idea turns out to be null" is roughly 80% in this project's own history (~85-95+ distinct
TEST-window experiments run as of this writing, §13.0) — a new idea needs a specific,
non-generic mechanism argued for it, not just plausibility, before spending a validation cycle
on it. The pattern across all four review rounds has been consistent: real hits cluster in
measurement/auditing/plumbing, not new modelling ideas — weight where the next session's time
goes accordingly.
