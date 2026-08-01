# NBA Prediction Model — Research Log & Technical Documentation

Audience: an LLM (or engineer) with no prior context on this codebase, needing a complete
mental model of what data this model is built on, what was tried, what worked, what didn't,
and why. Following the practice established in the sibling NHL/MLB projects, this log keeps
dead ends and rejected data sources visible rather than erasing them — a source that looked
promising and turned out unusable is worth recording so it isn't re-investigated later.

All paths are relative to the repo root: `/Users/brettscully/Desktop/sports-models/nba`.

---

## 0. Current state — read this first

Build started 2026-07-24, first full dev-range results landed 2026-07-25. Architecture (see
Sec1-2): a closed-form pace x rating combine (NOT a Monte Carlo simulator like the MLB sibling,
NOT a Poisson count model like the NHL sibling -- NBA scores aggregate over ~100 possessions and
are well-modeled by a continuous distribution around a point projection), with a play-by-play-
derived RAPM-lite layer for the active-lineup adjustment. Dev = 2015-16 through 2023-24 (season
start-year 2015-2023), holdout = 2024-25 and 2025-26 (start-year 2024-2025), frozen as
`DEV_MAX_SEASON=2024` in `src/models/final_holdout_check.py`.

**STATUS (updated 2026-07-25): Phases 0-5 of the original game-score plan are ALL DONE**, ending
with a real, honest, partially-mixed Phase 4 holdout result that changes what "done" means here --
see Sec9 for the full writeup. Summary, each with a real decision-worthy number in
`metrics_ledger.parquet`:

- **Phase 0 (ingestion): COMPLETE.** Schedule, advanced box scores, play-by-play, and traditional
  box scores are all fully backfilled 2015-16 through 2025-26 (~14,500 games). `GameRotation` was
  abandoned entirely partway through (see Sec5) after its reliability degraded to a multi-day ETA
  for one data source alone -- replaced with a substitution-event-based lineup reconstruction
  using `BoxScoreTraditionalV3` + `PlayByPlayV3` instead, which finished cleanly.
- **Phase 1 (team-strength baseline): REAL RESULT, CONFIRMED ADOPTED on dev; REAL margin_mae
  regression found on holdout (Sec9) -- diagnosed as scoring-era drift, not a Phase 2 problem.**
  10,737 dev games, pace x rating beats the naive own-average floor on every metric (total_mae
  15.08 vs. 15.26, margin_mae 10.47 vs. 11.28, straight-up accuracy **64.0% vs. 57.3%**), all
  bootstrap CIs excluding zero. Kept in production despite the holdout finding -- see Sec9 for why.
- **Phase 2 (RAPM-lite active-lineup adjustment): REAL RESULT in BOTH oracle AND predictive-minutes
  mode, on BOTH dev and holdout -- ADOPTED and LIVE-WIRED.** Oracle mode (ceiling test, 10,594 dev
  games): real improvement on total_mae/margin_mae. Predictive mode (the actually-deployable
  version, same dev games): ALSO real, capturing ~95% of oracle's ceiling value -- see Sec9.1.
  Wired into `generate_predictions.py`'s live `run()` (Sec9.2) and confirmed via Sec9.3's isolation
  check to generalize to holdout, independent of the Phase 1 margin issue above.
- **Phase 3 (score distribution): REAL RESULT, mixed/honest.** Calibration is excellent (95%
  nominal interval → ~94-95% empirical coverage). Pace-scaled variance and Student-t tails are
  each NOT yet detectably better than the simpler flat-variance/Normal baseline at this sample
  size -- adopted Normal for simplicity, kept pace-scaling as physically motivated but unproven.
  See Sec6.
- **Phase 4 (holdout check): RUN, ONCE, 2026-07-25 -- mixed, honest result, fully diagnosed.** See
  Sec9.3-9.5: total_mae NOISE (fine), margin_mae REAL REGRESSION (a genuine, confirmed dev/holdout
  gap -- but shown via a SEPARATE isolation check to originate entirely in Phase 1, not Phase 2, and
  explained by a real, confirmed scoring-era drift, not a modeling defect), su REAL IMPROVEMENT on
  holdout (unusual, not a veto trigger). **Net call: Phase 2 stands, Phase 1's margin/spread
  calibration carries a real, open, documented caveat pending a future scoring-era-drift fix --
  exactly the "diagnose and log, don't silently deploy or silently ignore" pattern the NHL sibling's
  own scoring-drift cycle established.**
- **Phase 5 (live pipeline): FULLY WIRED, confirmed against a real historical slate.** Team-strength
  + Phase 2 lineup adjustment (predictive minutes, RotoWire-informed) + score distribution all run
  together in `generate_predictions.py`; confirmed live against 2018-01-15's real 11-game slate.

**What's actually left on the game-score side**: the Phase 1 margin/scoring-era-drift issue found in
Sec9 is a new, real, OPEN problem (not blocking, not silently accepted) -- a candidate for a future
cycle analogous to the NHL sibling's own Cycle 13/Sec20 scoring-drift fix. Everything else in the
original 5-phase plan is done.

**STATUS (updated 2026-08-01): the player-props subsystem (Sec8, 10-12) is ALSO DONE, v1 scope,
all 6 planned tasks complete.** Six per-player rate-category models (minutes, 2PT/3PT/FT
scoring, rebounding, playmaking, steals/blocks), each independently validated at full dev-range
scale with a real improvement over its naive floor -- along the way, found FOUR separate instances
of two superficially-similar stats needing OPPOSITE smoothing families (minutes vs. shot attempts,
2PT/3PT vs. FT, steals vs. blocks, and a team-scheme-vs-individual-skill split inside the matchup-
difficulty layer itself), confirmed empirically every time, never assumed by analogy. A 3-level
matchup-difficulty shrinkage hierarchy (defender-specific -> position-group-vs-team -> league
floor), validated on the 2017-2023 subset. A macro-anchor + micro-reallocation composition rule
(`usage_allocation.py`) that anchors points to Phase 1+2's already-validated team total without
double-counting. A player-level predictive-distribution layer (`prop_distribution.py`) reusing
`score_distribution.py`'s math for high-count stats plus a new Poisson/NegBin branch for low-count
ones. A live `generate_props.py` pipeline, confirmed against a real historical slate (points
reconcile exactly to the game-prediction pipeline's own team totals; no NaN/negative projections).
**Known, explicitly-flagged v1 gap**: matchup difficulty is built and validated but NOT yet wired
into the live `generate_props.py` call (every output row is tagged `matchup_adjusted: False`) --
a real fast-follow, not silently dropped. See Sec8/10/11/12 for the full writeup, including two
more real bugs found and fixed (an `inf`-poisoned log-score at a zero-mean count projection, and a
NaN-poisons-the-team-sum risk in the live usage-allocation wiring). 23 regression tests (45
assertions) passing in `tests/test_regression_bugs.py`.

**Phase 1 (team-strength engine) -- DONE. Real, full 9-season dev-range result, confirmed
adopted.** `validate_team_strength_baseline.py` ran against the complete dev range (2015-16
through 2023-24, 10,737 games after dropping 1 game with no prior history yet) once the box-score
backfill finished (2026-07-24). Result: pace x rating REAL IMPROVEMENT over the naive own-average
floor on all three metrics, 5,000-resample paired bootstrap, all CIs excluding zero --
total_mae 15.08 vs. 15.26 (naive), margin_mae 10.47 vs. 11.28, straight-up accuracy **64.0% vs.
57.3%**. This closely matches the earlier 3-season preliminary read (64.6%), confirming that
smaller sample wasn't a fluke. Home-court multiplier is stable and correctly > 1.0 across every
dev season (1.004-1.016, no concerning drift). The rest/back-to-back diagnostic also came back
real on the full range (p=8.0e-7, n=3,869 B2B games, -1.02 point effect) -- a strong candidate
for a tested Phase 1b increment, intentionally NOT adopted yet per the one-variable-at-a-time
discipline.

Three more real bugs were found and fixed getting this real run to actually complete (on top of
the home-court column bug from the 3-season smoke test, which is what caught the earlier,
backwards-sign multiplier -- see Sec4 for that one's full writeup):

1. **NaN silently poisoned the whole bootstrap result.** A single game (the very first one in
   the entire dev range, which has no trailing history yet) has a NaN prediction -- harmless on
   its own, but `bootstrap_significance.bootstrap_compare` converts to numpy arrays before
   averaging, and a raw numpy array's `.mean()` does NOT skip NaN the way a pandas Series does
   (confirmed directly: `np.array([1,2,nan,4]).mean()` is `nan`; `pd.Series([...]).mean()` is
   `2.33`). The 3-season smoke test never hit this because that slice didn't happen to include
   the dev range's very first game. Fixed by dropping NaN rows in the calling script before
   bootstrapping, and printing how many were dropped (1, exactly the expected count) rather than
   silently absorbing it.
2. **A duplicate-column crash in the second `metrics_ledger.append_run` call.** Renaming
   `naive_home`/`naive_away` to `pred_home`/`pred_away` on the FULL predictions frame (which
   already has its OWN `pred_home`/`pred_away` from the model arm) creates two columns sharing the
   same name -- `metrics_ledger.compute_run_metrics` crashed with "Cannot set a DataFrame with
   multiple columns to the single column pred_total" the moment it tried to sum them. Fixed by
   selecting the naive-only columns into their own frame FIRST, then renaming.

Both new bugs are covered by `tests/test_regression_bugs.py` alongside the original five.

`team_strength.py`, `shrinkage.py`, `home_court.py`, `rest_schedule.py`, `metrics_ledger.py`,
`bootstrap_significance.py`, `final_holdout_check.py`, and `validate_team_strength_baseline.py`
are all confirmed working end-to-end on the real, full dev range -- this run and its ledger entry
are the FIRST real, decision-worthy result in this project, not a smoke test.

**Phase 2a (lineup stint construction) -- core algorithm built and validated on 2 real games
from 2 different seasons (2023-24 and 2019-20), both reconciling EXACTLY (summed stint point
differentials matched the real box-score final score to the point) with zero warnings.**
`src/ingest/build_stints.py` is written. Two real bugs were found and fixed while validating
(see Sec4 for detail): (1) the ingest scripts' exception handlers didn't catch
`json.JSONDecodeError`, so a single bad response crashed an entire multi-hour background
backfill with zero games saved past the last checkpoint -- fixed by widening the caught
exception types; (2) the initial possession-counter naively substring-matched "Def:1" in a
Rebound row's description, which is that PLAYER's CUMULATIVE game-total defensive rebounds, not
a per-event flag -- silently undercounted total possessions by ~30%. Fixed by comparing each
Rebound event's `teamId` against the immediately-preceding `Missed Shot` event's `teamId`.
Also discovered: `GameRotation` returns a real HTTP 500/empty-body response for a small but
real fraction of games (confirmed reproducible, not transient -- same game_id fails 5/5 retries
across multiple minutes, while a neighboring game_id in the same season succeeds reliably) --
this is a genuine stats.nba.com data gap for those specific games, not a bug in this project's
code; `validate_data_coverage.py` is designed to surface exactly which games this affects once
the full backfill completes, and a gap-filling re-run against just the missing games is planned
as a follow-up (fetch functions already skip already-cached games, so this is cheap).

**Phase 2b (RAPM-lite ridge fit) -- code complete, core math validated on synthetic data.**
`rapm_lite.py` (design matrix + generalized-ridge fit with per-player experience-based
penalties, `player_priors.py`), `garbage_time.py`, `lineup_rating.py` (active-roster ->
team-rating adjustment, oracle minutes mode), and `validate_rapm_lineup_adjustment.py` are all
written. A controlled synthetic test (two players with a known, large true offensive-skill gap,
600 stints) confirmed the ridge fit correctly recovers the sign and approximate magnitude of the
true difference (+20.9 vs. -21.0 fitted `off_rapm`, `def_rapm` correctly near zero for both since
only offense was varied in the synthetic generator), and the walk-forward periodic-refit driver
(`compute_walkforward_player_ratings`) correctly produces one rating snapshot per checkpoint
using only strictly-prior data, with ratings stabilizing as more history accumulates. One real
bug found and fixed: `_career_games_played`'s first version counted a player's home-game
appearances and away-game appearances SEPARATELY and took the max, silently reporting roughly
half a typical player's true games-played count (a player who splits home/away roughly evenly
would show ~half their real total) -- fixed by pooling both columns before counting distinct
games. The real dev-range fit (needed for the oracle-mode validation run) awaits the
play-by-play/rotation backfill.

**Phase 3 (score-distribution model) -- code complete, validated on synthetic data.**
`score_distribution.py` (pace-scaled variance regression, empirical home/away correlation,
Normal-vs-Student-t via method-of-moments kurtosis, log-score + closed-form Normal CRPS, closed-
form win-probability/spread/total from the joint) and `validate_score_distribution.py`
(chronological fit/eval split within dev, distinct from the Phase 4 dev/holdout split) are
written. A synthetic test (5,000 games, genuine pace-scaling variance + correlated home/away
residuals) confirmed: fitted variance model's PREDICTIONS closely track the true variance across
the tested pace range (individual intercept/slope terms are only weakly identified in isolation
when pace never approaches 0 -- a known, harmless linear-regression extrapolation artifact, not
a bug -- but predictions in-range are accurate); fitted correlation (0.294) matched the true
generative correlation (0.3) closely; empirical interval coverage at 50/80/95% nominal matched
almost exactly (0.493/0.807/0.952); log-score correctly preferred the true mean over an
off-target one.

**Phase 4 (dev/holdout harness) -- code complete.** `validate_holdout_bootstrap.py`
(confirmatory-veto-only protocol, dev-vs-holdout gap bootstrapped rather than either set judged
in isolation) is written; genuinely not runnable yet since it requires a full dev-validated
model configuration to check, which doesn't exist until Phase 1-3's real dev-range runs
complete.

**Phase 5 (live daily pipeline) -- partially wired, confirmed working against real historical
dates.** `refresh_data.py` (incremental current-season refresh) and `generate_predictions.py`
are written. `generate_predictions.py` currently runs the Phase 1 team-strength projection +
Phase 3 score distribution live (confirmed against real historical slates, e.g. 2018-01-15's
11-game slate produced plausible 95-120-point projections); the Phase 2 RAPM-lite lineup
adjustment is NOT yet wired into this live entry point (documented honestly in the module's own
docstring, not silently omitted) -- it needs the full historical stint backfill first. Two real
integration pieces were built and confirmed while wiring this up: (1) `player_name_crosswalk.py`
-- RotoWire identifies players by name string, everything else in this project by `nba_api`'s
numeric PERSON_ID; confirmed live that `nba_api.stats.static.players` (a free, no-HTTP-call
5,103-player list) plus a generational-suffix-stripping fallback (RotoWire omits "Jr."/"III"/etc.
that the static list carries) matches 126/152 of a real live RotoWire injury report, with the
remainder being recent rookies/two-way players not yet in the static list; (2) `games_on_date`'s
home/away team resolution -- `ScoreboardV3`'s team-rows dataset has no explicit home/away column,
but `gameCode` (e.g. `"20260115/MEMORL"`) reliably encodes AWAY+HOME tricodes in its 6 characters
after the date, confirmed against a real slate rather than trusted from row order (which was
tried first and was NOT a reliable signal).

**Standing conventions carried over from the NHL/MLB projects** (see those projects'
`MODEL_DOCUMENTATION.md` for the full rationale behind each):

- **Walk-forward validation, no lookahead** — every feature/constant used to score a game
  must be computable from data strictly prior to that game.
- **A genuine holdout split**, carved out early and touched only for confirmatory checks
  under a pre-registered protocol (see the NHL project's §15 for the corrected version of
  this rule — holdout is a veto on a dev-confirmed improvement, not a second attempt to find
  one).
- **Paired bootstrap (5,000 resamples) as the standard significance gate** for every
  adoption decision, applied identically to dev and holdout.
- **An append-only metrics ledger** recording every validated model configuration.
- **This document updated honestly as work happens** — including in-sample scoring defects,
  reverted adoptions, and retracted findings. The NHL project's own history (a walk-forward-
  discipline bug found in review, a GBM stacking layer adopted then reverted once its
  adoption bootstrap was found to be scored in-sample) is the reason these conventions exist
  in their current, stricter form — start from that standard rather than re-discovering it.

## 1. Data sources — confirmed live 2026-07-24

**`nba_api` / stats.nba.com** (`src/ingest/fetch_schedule.py`, `fetch_boxscores.py`,
`fetch_playbyplay.py`, `fetch_rotation.py`):
- Plain `requests` gets a 403 from stats.nba.com; `nba_api`'s endpoint wrappers work with no
  auth (they set browser-like headers internally). No published rate limit; used the
  conventional ~0.6s/request courtesy delay.
- **`BoxScoreAdvancedV2` (and `PlayByPlayV2`) are dead** — confirmed by direct call: return an
  empty `{}` JSON body for a real 2023-24 game, not an error. The package's own docstrings
  confirm `PlayByPlayV2` is deprecated as of 2024-25 and no longer returns data. **Only the V3
  family works across the full 2015-16..2025-26 range** — this project standardizes on V3
  everywhere for consistency rather than mixing versions by season.
- `BoxScoreAdvancedV3` returns player-level and team-level datasets; the team-level rows already
  carry `pace`/`offensiveRating`/`defensiveRating`/`possessions` directly — no need to fetch
  `BoxScoreTraditionalV3` or re-derive pace by hand for Phase 1. Traditional box scores are
  deliberately NOT fetched in v1 (cuts ingestion request volume in half) since nothing built so
  far needs point/rebound/DNP detail beyond what advanced-v3 and `GameRotation` already provide.
- `GameRotation` gives `IN_TIME_REAL`/`OUT_TIME_REAL` per player per continuous on-court interval
  — confirmed empirically these are TENTHS OF A SECOND of cumulative elapsed game time from
  tip-off (a real 2023-24 game's max `OUT_TIME_REAL` was exactly 28800 = 48min * 60s * 10),
  spanning all periods including OT. Also carries `PLAYER_PTS`/`PT_DIFF` per interval, a free
  cross-check signal for the Phase 2 stint-reconciliation gate.
- `PlayByPlayV3`'s `actionType=="Substitution"` rows have `personId` set to the OUTGOING player
  and `description` formatted as `"SUB: <incoming> FOR <outgoing>"` — usable as a stint
  cross-check (see Sec1 plan), though `GameRotation` is the primary stint-construction source.
- `nba_api.stats.static.teams` gives the 30 current franchises with no HTTP call — confirmed
  zero relocations/rebrands across the entire 2015-2026 window (`TEAM_ID` stable throughout,
  unlike the NHL sibling which had to handle real relocations).

**Active-lineup / injury data** (`src/ingest/fetch_rotowire_lineups.py`, live-pipeline-only, no
historical archive):
- RotoWire's NBA injury report page has no server-rendered table — it's a client-side fetch to
  `GET https://www.rotowire.com/basketball/tables/injury-report.php?team=ALL&pos=ALL`, found via
  the rendered page's own network traffic. Returns clean public JSON (`player`, `team`,
  `position`, `injury`, `status` — Out/Doubtful/Questionable/Probable/Day-To-Day; `rDate` is
  subscriber-gated but not needed). Materially easier than the MLB sibling's RotoWire scrape
  (that one parses real HTML tables).
- NBA's official injury report exists at a predictable but rotating URL pattern
  (`ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>_<MM><AM|PM>.pdf`,
  confirmed via web search of real historical filenames) but is PDF-based, published multiple
  times/day per league rule (5pm day-before, then 8-10am/11am-1pm gameday depending on tip
  time). **Decision: not wired up for v1** — would add a `pdfplumber`-style dependency and
  brittle table-parsing for a source whose main advantage over RotoWire (official authority)
  isn't worth the engineering cost yet, since RotoWire already gives a workable Out/Doubtful
  signal. Revisit if RotoWire's status field ever proves unreliable in practice.

**Historical odds (calibration/benchmark only, not a training feature) — acquired for 2015-16
through 2022-23** via `src/ingest/fetch_odds_sbro.py`. Confirmed live (2026-07-24):
SportsbookReviewsOnline's `/scoresoddsarchives/nba-odds-<season>/` pages looked like a
client-rendered SPA when navigated interactively, but are actually plain server-rendered
WordPress pages — a bare `requests.get` with a browser User-Agent returns the full HTML table
directly, no browser automation needed (much simpler than the NHL sibling's dead scrape target,
which needed the Wayback Machine). Real parsing quirk found and fixed: the classic SBRO
Open/Close columns don't consistently carry "spread" on one row and "total" on the other —
some real rows mix conventions WITHIN the same row (e.g. a real 2015-16 game's away-side row
read Open=2 [a spread value] but Close=191 [a total value]) — fixed by classifying each of the
four cells (away open/close, home open/close) independently by magnitude (>=100 = total) rather
than assuming one whole row is one type. Coverage caveat: the site's own text says this archive
"will not be updated" — confirmed it stops at 2022-23, and that last season is itself only
partial (664 games, through 4pm-ish mid-January 2023) — so this cannot benchmark 2023-24 onward,
including any holdout-season comparison. Fine as-is: this is a nice-to-have calibration check,
not a requirement, and 8 full/near-full dev seasons of real closing lines is already useful.

## 2. Decisions made before the first cycle (formerly "open questions")

- **Target: game-level score** (both teams' points), matching the NHL/MLB siblings' convention,
  with derived win probability/spread/total from Phase 3's score-distribution layer — not
  player props (unlike the MLB sibling's daily props deliverable; revisit only after the
  game-score model itself is validated).
- **Market-odds benchmarking is deferred**, not blocking — see Sec1. The model is validated
  against its own naive baselines and paired bootstrap first; a market comparison is a
  calibration sanity-check to add later, not a dependency of Phase 1-4.
- **Holdout split: most recent 2 full seasons** (2024-25, 2025-26 — `season` start-year 2024 and
  2025), dev = 2015-16..2023-24 (start-year 2015-2023). Frozen as `DEV_MAX_SEASON=2024` in
  `src/models/final_holdout_check.py`, confirmed against the live schedule via
  `current_nba_season()` returning 2025 on 2026-07-24 (i.e. 2025-26 is the most recently
  completed season) — see that module's docstring for why this is a frozen literal, not
  recomputed at runtime.

## 3. Phase 1 architecture as built

`team_strength.build_team_game_log` merges the cached schedule (real final scores, home/away)
with cached advanced-box-score team rows into one row per team per game, regular season only
(playoffs excluded from training — a prediction target later, not dev signal, matching the
principle behind excluding spring training from the MLB sibling's fit). `shrinkage.py`
(reimplemented independently, not imported from nhl/) applies walk-forward Bayesian shrinkage
toward a trailing league average, with the same same-game-leak guard the NHL sibling's own
history found necessary (`_trailing_league_stat` collapses to one row per real `gameId` before
`shift(1)`, so a game's two team-rows can never see each other regardless of row order).

Exposure unit is GAMES not minutes/possessions (unlike NHL's PP/PK TOI-based rate, which needs a
TOI-weighted approach because power-play time varies a lot game to game) — `OFF_RATING`/
`DEF_RATING`/`PACE` are already possession-normalized per game by stats.nba.com, so one game is a
consistent unit of exposure regardless of pace.

**Untested hypothesis flagged, not assumed**: `add_walk_forward_rate`/`add_walk_forward_mean`
default to a full within-season reset (`cross_season_weight=0.0`, matching the NHL sibling's own
default) but expose a blend-toward-prior-season-rate option, since NBA rosters keep 60-80% of
minutes year over year (real continuity signal a hard reset discards at game 1 of each season)
— this is a candidate for testing in `validate_team_strength_baseline.py`'s grid, not shipped as
a default until shown real via bootstrap, exactly like the NHL sibling's own analogous
`cross_season_weight` parameter had to earn its place empirically (r=0.15 early-season
correlation, confirmed before adoption).

Home-court advantage (`home_court.py`) is fit empirically, not assumed as a flat constant: the
log-ratio of actual-to-mult-free-baseline rating on each side estimates `log(mult)` (home) and
`-log(mult)` (away) independently, averaged. `fit_home_court_by_season` is a temporal-drift
diagnostic (each season fit independently — informational only, not itself walk-forward);
`fit_home_court_walk_forward` (trailing EWMA, half-life 400 games, un-calibrated placeholder) is
what's actually used per-game in predictions.

Rest/back-to-back (`rest_schedule.py`) is NOT folded into predictions yet — `run_rest_diagnostic`
correlates each team-side's residual (actual minus the Phase 1 model's own prediction) against
`rest_days`/`is_b2b`, reported alongside the main validation run but not adopted until a real
(low-p-value) effect is confirmed via its own paired-bootstrap increment.

The naive floor the Phase 1 model must beat is a team's own trailing scoring average with NO
opponent adjustment at all (`shrinkage.add_walk_forward_mean` on raw `actualScore`, a small
5-game prior, deliberately unshrunk-and-untuned since it's meant to be a weak floor, not a
competitor).

**Status**: all of the above is written and smoke-tested for correctness (confirmed the
identical home/away point predictions seen on a tiny single-partial-season sample were the
expected artifact of most teams still having 0 prior games played that early, not a bug — see
`games_played_before` in the debug trace). The real dev-range bootstrap comparison has not been
run yet; results will be logged here once the box-score backfill completes.

## 4. Bugs found and fixed while validating

**Phase 1 Bug — home-court multiplier fit from the wrong columns (see Sec0 for the full
writeup).** `home_court._baseline_log_ratios` read `home_oRtg`/`home_dRtg`/`away_oRtg`/
`away_dRtg`/`home_pace`/`away_pace` off the wide per-game frame -- names that exist there, but
are that SAME game's own raw realized box-score values (renamed straight through from
`offRtg_home` etc. by `_to_wide_games`), not the walk-forward PRE-game prediction columns
(`rtg_attack_rate_home`, `pace_shrunk_mean_home`, etc.) that `project_game` itself actually
uses. Caught because the fitted multiplier came out below 1.0 in every tested season despite the
same underlying data showing a completely ordinary positive home-court edge (106.2 vs. 103.5
average points, 58.4% home win rate) -- a physically implausible result flagged it immediately
rather than being quietly accepted. Fixed by reading the correct walk-forward columns; a
follow-on 3-season preliminary run then showed a clean real improvement over naive on every
metric (Sec0).

**Phase 2b Bug — `validate_rapm_lineup_adjustment.py` referenced `row.home_court_mult` without
ever computing it** (an `AttributeError` waiting to happen the first time this script actually
ran against real stint data) -- `run_oracle_lineup_backtest` built the wide games frame via
`_to_wide_games` but never called `fit_home_court_walk_forward` on it the way
`validate_team_strength_baseline.build_dev_predictions` does. Fixed by adding that same call
before the merge/loop. Caught by code review while fixing the Phase 1 home-court bug above,
before ever getting a chance to fail at runtime -- worth noting as a reminder to re-check every
OTHER caller of a shared column/convention once one caller is found to have gotten it wrong.

## 4a. Phase 2a bugs found and fixed while validating `build_stints.py`

**Bug 1 — silent-crash exception handling.** All three per-game ingest scripts
(`fetch_boxscores.py`, `fetch_playbyplay.py`, `fetch_rotation.py`) originally caught only
`(requests.exceptions.RequestException, KeyError, IndexError)` around each per-game fetch.
`nba_api` occasionally returns a 500 with an empty body for a given game (see Bug 3 below),
which surfaces as `json.JSONDecodeError` (a `ValueError` subclass) when the library tries to
parse it — NOT one of the caught types. Confirmed real: the first full rotation backfill attempt
crashed the entire background process on an early game with zero games saved past the last
50-game checkpoint, silently losing hours of intended progress had it gone unnoticed. Fixed by
widening every fetch script's per-game exception tuple to include `ValueError`. Lesson: an
unattended multi-hour ingest job needs its retry wrapper to catch the ACTUAL exception types the
underlying library raises on a bad response, not just the ones that seemed obvious in advance —
worth spot-checking this whenever a new endpoint wrapper is added.

**Bug 2 — possession counter undercounted by ~30%, confirmed against a real game.** The first
version of `build_stints._count_possessions` detected defensive rebounds by checking whether a
`Rebound` row's `description` contained the substring `"Def:1"`. This is WRONG: the "Off:N Def:M"
suffix in a rebound description is that player's CUMULATIVE offensive/defensive rebound total so
far in the game (confirmed by inspecting real rows: `"Jokic REBOUND (Off:2 Def:6)"` etc. increase
monotonically across the game), not a per-event indicator — so the substring check only ever
matched a player's exact FIRST defensive rebound of the game, silently missing every later one.
Confirmed on a real 2023-24 game (0022300061): combined possession count came out to 132 against
an expected ~191 (the box score's own `possessions` field, summed both teams) — a ~30% gap.
Fixed by comparing each `Rebound` event's `teamId` against the immediately-preceding `Missed
Shot` event's `teamId` (same team = offensive rebound, possession continues; different team =
defensive rebound, possession ends) — this also correctly handles team-credited (non-player,
e.g. `"Lakers Rebound"`) rebounds, which still carry a real `teamId` even though `description`
has no Off/Def breakdown. Also added detection of MADE final free throws of a trip (subType
`"N of N"`, not a MISS) as a possession-ending event — a made final free throw has no following
rebound row at all, unlike a miss. After both fixes: 203 (2023-24 test game) and 197 (2019-20
test game) against expected ~191 and ~189 respectively — within ~5-7%, and directionally
consistent with a known, expected difference between a raw discrete PBP event count and the
NBA's own official `possessions` field (which is itself a continuity-adjusted ESTIMATE formula —
`0.5*((FGA+0.4*FTA-1.07*(OREB/(OREB+OppDREB))*(FGA-FGM)+TOV)` summed both teams — not a literal
count of discrete plays, so an exact match was never really the right target; internal
consistency between a stint's own points and possessions matters more for RAPM fitting than
matching NBA's specific estimator).

**Finding 3 — `GameRotation` returns a real, reproducible HTTP 500/empty-body for a nontrivial
fraction of games.** Discovered while trying to pull extra real games for build_stints testing:
game `0022100101` (2021-22) and `0021600101` (2016-17) both returned `500` with a 0-byte body on
every one of 5 retries with 3s backoff, checked directly with raw `requests` (not just through
`nba_api`) to rule out a client-side bug — while neighboring/different game_ids
(`0022300061`, `0021900101`) succeeded reliably in the same test run. This is NOT the same
failure mode as Bug 1 (which was about mishandling a transient/legitimate empty response) — it's
a genuine per-game data gap on stats.nba.com's own side for `GameRotation` specifically. Early
observation from the live rotation backfill: roughly 3 such permanent failures in the first ~30
games of the 2015-16 season (~10% rate), which is high enough to require a deliberate
gap-filling follow-up pass (re-running `fetch_rotation_season` after the bulk backfill
completes — the existing done-games cache means this only re-attempts exactly the missing
games) rather than being an ignorable edge case. `validate_data_coverage.py` will report the
exact scope once the full backfill finishes.

## 5. The `GameRotation` → substitution-event pivot (2026-07-24)

**`GameRotation` was abandoned entirely as the stint-construction input.** What started as "a
nontrivial failure rate" (Finding 3 above) got progressively worse over the course of one
evening: a direct timing test found individual requests taking a consistent ~30s whether they
succeeded OR failed (not a fast rejection), and a live backfill at that point was on pace for
roughly 1-3+ days just to finish one data source. A second timing test a while later found the
SAME endpoint had degraded further still (5 of 6 requests taking ~30s each). This is a real
stats.nba.com server-side degradation for this specific endpoint, not something fixable
client-side by retrying smarter.

**The pivot**: `BoxScoreTraditionalV3` (confirmed live: 8/8 requests succeeded in under 0.5s
each, same reliability profile as the already-solid `BoxScoreAdvancedV3`) plus
`PlayByPlayV3`'s already-fetched `Substitution` events can reconstruct the same on-court lineup
information `GameRotation` would have provided, without depending on it at all: seed each
team's opening lineup from the box score, then walk chronologically through substitution events
applying each one. `fetch_boxscore_traditional.py` was added (mirrors the existing reliable
fetch scripts) and its backfill started immediately; the `GameRotation` backfill was abandoned
mid-run.

**Four real, distinct bugs were found and fixed getting this new path to work**, each confirmed
against real games before and after the fix (not assumed):

1. **Starter detection**: a non-blank `position` field looked like a clean starter indicator on
   the one 2023-24 game checked first, but on a real 2015-16 game, 7 "home" and 9 "away" players
   all had non-blank positions -- not just the 5 real starters. Testing against only one
   season/era was not enough. Fixed by using ROW ORDER instead (the first 5 rows per team in
   `BoxScoreTraditionalV3`'s player-level dataset) -- cross-validated directly against the same
   response's own team-level `startersBench=="Starters"` points total, which matched exactly on
   every game checked.
2. **Score forward-fill used the wrong placeholder value**: `scoreHome`/`scoreAway` are blank on
   non-scoring PBP rows in recent seasons (safe to forward-fill past), but are the LITERAL STRING
   `"0"` on non-scoring rows in this same 2015-16 game -- and `pd.to_numeric("0")` parses that as
   a real zero, not something to skip. This silently produced negative/garbage point totals for
   older games. Fixed by only trusting `scoreHome`/`scoreAway` on rows confirmed to be a real MADE
   scoring event (`Made Shot`, or a `Free Throw` whose description doesn't start with `MISS`) --
   forcing every other row to NaN before forward-filling, regardless of what its raw string
   happened to contain.
3. **Name-collision disambiguation**: two teammates sharing a family name (confirmed real: Jeff
   Green and JaMychal Green, same team, same game) are disambiguated by `PlayByPlayV3` using
   NBA's own convention -- first two letters of the first name plus a period ("Je. Green" /
   "Ja. Green") -- rather than a bare, genuinely ambiguous "Green". A plain-familyName lookup
   couldn't match that form at all. Fixed by also indexing every player under that disambiguated
   form.
4. **Suffix mismatch, opposite direction from the RotoWire crosswalk bug**: a real 2015-16 game
   (Jimmy Butler) has `familyName == "Butler III"` in the box score, but the substitution
   description uses the bare "Butler" -- the opposite mismatch direction from
   `player_name_crosswalk.py`'s RotoWire problem (which strips a suffix the STATIC PLAYER LIST
   carries but RotoWire's report omits). Fixed the same way: index every suffixed family name
   under its suffix-stripped form too.

**Current state, honestly**: even after all four fixes, stint reconciliation is not yet perfect
on every game -- a real, recurring "phantom substitution" pattern remains (an incoming player
already on court with no matching prior exit logged, or an outgoing player not on court to
begin with), confirmed on multiple different games and not resolved by any of the four fixes
above. This looks like genuine data-entry noise in the underlying NBA play-by-play feed itself
(a known characteristic of this class of data, not unique to this project), not a remaining
code bug -- attempting a "treat the whole event as a no-op" heuristic for this specific pattern
was tried and made results WORSE on the one game tested, so it was reverted rather than kept on
a hunch. **Design decision**: `build_season_stints` does not exclude a whole game for an
imperfect reconciliation -- every stint that IS kept already individually passed its own 5-vs-5
on-court check, so a nonzero gap means some minutes are uncovered, not that included minutes are
wrong. Coverage (fraction of a game's real total points captured by kept stints) is reported as
an honest per-season diagnostic instead of a hard pass/fail gate. On a small 6-game manual test
sample spanning 2015-16 and 2023-24: one exact reconciliation, one within 5 points, and four
in the 60-75% coverage range -- averaging roughly 75% coverage.

**Updated with the real full-season number (2026-07-24, once 2015-16 finished backfilling):**
`build_season_stints(2015)` on all 1220 available regular-season games gives **63.5% coverage**
(15,161 interval warnings) -- meaningfully lower than the small-sample estimate above, an honest
downward revision worth recording rather than quietly keeping the more flattering number.
Investigated whether this points to a remaining code bug: sampled 30 games' worth of dropped
intervals (354 of them) and found the on-court player-count drift splits roughly evenly between
undercounts (4 players, most common) and overcounts (6-7 players) rather than skewing one
direction -- a systematic bug in this project's own logic would be expected to show a consistent
directional bias (e.g. always losing a player, never gaining one); a roughly balanced split is
more consistent with scattered "phantom substitution" noise genuinely present in the underlying
NBA play-by-play feed (matching the Reddish/Green/Butler-type anomalies already documented above)
than with an undiscovered deterministic bug. Not chasing this further right now: a majority-but-
incomplete stint sample across thousands of games is still workable RAPM training signal (that's
what the ridge penalty and large N are for); the real open risk is whether the DROPPED minutes are
randomly distributed or systematically correlated with something (e.g. garbage time, specific
team patterns) -- not yet checked, flagged as a follow-up rather than blocking the current cycle.

**Also discovered while pivoting**: partway through the evening, ALL THREE running backfills
(box scores, play-by-play, and the new traditional-box-score fetch) started hitting read
timeouts simultaneously -- a broader stats.nba.com slowdown, not specific to `GameRotation` this
time. Unfortunate timing: the box-score backfill was on its very last season (2025-26) when this
hit. No code fix applies here; this is an external, transient host-side condition to wait out.

## 6. Phase 3 (score-distribution) -- DONE. Real, full dev-range result.

`validate_score_distribution.py` ran on the real full dev range once Phase 1's predictions were
available (7,515 fit-set games, 3,222 eval-set games, chronological 70/30 split). Two honest,
different-flavored findings, not both wins:

**Calibration is excellent.** Empirical interval coverage tracks nominal closely at every level
checked -- 50% nominal gives ~47-50% empirical, 80% gives ~78-80%, 95% gives ~94-95% (Normal and
Student-t nearly identical). The pace-scaled variance model (`fit_residual_variance_model`) is
producing genuinely well-calibrated uncertainty, not just a plausible-looking point estimate.

**But: neither refinement over the simplest baseline is PROVEN at this sample size.** The
pace-scaled variance model is NOT detectably better than a flat-variance naive baseline on
log-score or CRPS (both bootstrap CIs include zero) -- i.e., modeling variance as a function of
pace, while physically motivated (more possessions aggregated -> more variance) and not WORSE,
isn't yet confirmed to matter at n=3,222. Likewise Student-t vs. Normal: NOT detectably better
(CI includes zero) -- consistent with the earlier finding of low excess kurtosis (effectively
Normal). **Decision: adopt Normal (simpler, no worse than Student-t) as the distribution family
for now; keep the pace-scaled variance model as designed (it's not harmful and is the physically
correct approach) but do NOT claim it as a proven improvement over flat variance -- an honest
"NOISE" verdict, not inflated into a false win.** Worth revisiting with more data or a different
metric (e.g. pace-stratified calibration specifically, rather than an aggregate score) if this
matters more later.

## 7. Phase 2b (RAPM-lite lineup adjustment) -- DONE. Real, full dev-range result. THE core
hypothesis of this whole project is confirmed.

All 9 dev seasons' stints were built via `build_season_stints` (2015-16 through 2023-24, ~1,220
games/season, 63-68% coverage in every single season -- confirmed stable across eras, not
degrading on older data, by spot-checking 2022-23 independently and getting 63.0% against
2015-16's 63.5%). One more real bug was found and fixed getting the full-range RAPM fit to run:
`rapm_lite.prepare_stints` never converted the cached schedule's `gameDate` (a plain string) to a
real datetime -- `compute_walkforward_player_ratings` compares it against `pd.Timestamp`
checkpoints from `pd.date_range` (always real Timestamps, even given string bounds), which raised
`TypeError: Invalid comparison between dtype=str and Timestamp` the first time this ran at real
scale (a small synthetic smoke test earlier had already-parsed dates on both sides, so it never
hit this). Fixed by explicitly parsing `gameDate` with `pd.to_datetime` in `prepare_stints`; now
covered by `tests/test_regression_bugs.py` (14 checks total, all passing).

**The real result** (`validate_rapm_lineup_adjustment.py`, oracle minutes-share mode -- see
`lineup_rating.py`'s docstring for why oracle mode is a ceiling test, not yet a deployable live
prediction -- 10,594 games, 5,000-resample paired bootstrap): adding the RAPM-lite active-lineup
adjustment on top of Phase 1's team-strength baseline gives a REAL IMPROVEMENT on both continuous
score-error metrics -- total_mae 15.058 vs. 15.080 (baseline), margin_mae 10.432 vs. 10.467, both
CIs excluding zero -- but NOT a statistically detectable improvement on straight-up win/loss
accuracy (64.19% vs. 64.03%, CI includes zero, NOISE). This is an honest, expected shape for a
refinement layer on top of an already-good baseline: the improvement is real but modest (roughly
1/8th the size of Phase 1's own jump over the naive floor), and win/loss is a coarser, noisier
target than continuous point-error, so a small real effect on MAE isn't guaranteed to show up as
a detectable SU move at this sample size. **This confirms the project's central premise --
knowing WHO is on the floor tonight measurably improves score prediction beyond team-level
averages alone -- in the oracle/ceiling sense.** The PREDICTIVE-minutes deployment (each player's
own trailing-average minutes, not real known minutes) is still a follow-up integration task for
`generate_predictions.py`, not yet wired into the live pipeline -- this result establishes that
the underlying signal is real and worth deploying, not that deployment is finished.

**UPDATE**: `generate_predictions.py` has since been updated to PREDICTIVE minutes mode (see that
module's own docstring) -- the follow-up noted above is done.

## 8. Player-props subsystem (started 2026-07-25) -- MLB-sibling-style per-player stat-line
projection (points split 2PT/3PT/FT, rebounds split OREB/DREB, assists, turnovers, steals,
blocks), matchup-and-defense-aware. Full design in the approved plan; building per the
sequencing there. Real, new data confirmed available: `BoxScoreMatchupsV3`/`BoxScoreDefensiveV2`
(per-defender-matchup stats, but ONLY 2017-18 onward -- confirmed live, fails for 2015-16/2016-17,
matching the real Second Spectrum tracking rollout), `BoxScorePlayerTrackV3` (touches, rebound
CHANCES, contested/uncontested FG -- full range, no boundary), `CommonTeamRoster` (per-season
`POSITION`, full range).

**`player_rate_shrinkage.py`** (the shared primitive, generalizing `shrinkage.py` to arbitrary
exposure + player-level grouping) and **`player_minutes.py`** (the foundational model) are done,
with a real, important finding along the way:

**Real finding: minutes need recency-weighting, not infinite-memory shrinkage.** The first version
of `add_minutes_rating` mirrored `shrinkage.add_walk_forward_rate`'s team-level pattern (expanding
average shrunk toward a trailing league prior) -- validated against a naive floor on the full dev
range (275,138 player-games) and found to be a **REAL REGRESSION** (CI excludes zero, MAE got
monotonically WORSE from 6.43 at prior_games=1 up to 7.99 at prior_games=20). Investigated rather
than just picking the least-bad prior: a direct comparison of rolling-window and EWMA alternatives
confirmed why -- a player's ROLE can change within a season (rotation change, return from injury,
a trade) in a way team-level PACE never does, so recent games are genuinely more informative than
season-to-date average. EWMA with halflife=2 games won outright (MAE 5.41, beating every rolling-
window and expanding-shrinkage variant tested). Added `add_walk_forward_player_mean_ewm` to the
shared primitive and switched `player_minutes.py` to it -- re-validated against a genuinely naive
season-average floor: **REAL IMPROVEMENT, MAE 5.41 vs. 10.87**, CI excluding zero. Also extracted
`metrics_ledger.append_generic_run` (shared timestamp/git-hash/parquet-append plumbing) so
non-score-shaped validation runs (per-player-stat MAE, not home/away totals) can log to the same
ledger without forcing `compute_run_metrics`'s score-specific contract to bend.

**`player_scoring_rates.py`** (2PT/3PT/FT attempts + makes) is done and validated, with two more
real findings on top of the minutes one above -- confirming, twice more, that smoothing-family
choice must be checked empirically per stat, never assumed by analogy:

**Real finding: shot attempts and make-rate need OPPOSITE smoothing families.** A first version
using expanding-shrinkage for BOTH attempts and makes was a REAL REGRESSION on all of 2PT/3PT/FT.
Diagnosed by testing each half separately on real 3PT data: attempts are a volume/role metric
(shot selection depends on hot streaks, coaching trust) -- EWMA halflife=10 games (MAE 1.173) beat
every expanding-shrinkage prior tested (best 1.198, and it got WORSE as the prior grew, the same
monotonic pattern minutes showed). Make-rate is a stable shooting-skill metric -- expanding-
shrinkage prior_attempts=150 (MAE 0.553) beat every EWMA halflife by ~40%. Fixed: EWMA for
attempts, expanding-shrinkage for makes, for 2PT and 3PT. Re-validated on the full dev range
(267,124 player-games): 2PT makes REAL IMPROVEMENT (MAE 1.0449 vs naive 1.0504), 3PT makes REAL
IMPROVEMENT (MAE 0.6155 vs naive 0.6188).

**Real finding: FT doesn't follow the 2PT/3PT pattern at all.** Applying that same fix to FT
(EWMA attempts halflife=10, expanding-shrinkage makes prior=100) was ALSO a REAL REGRESSION (shrunk
MAE 1.0726 vs naive 1.0538, CI (+0.0121,+0.0279) -- excludes zero the wrong way). Re-swept FT
specifically rather than assuming the 2PT/3PT split transfers: FTA per-minute wants
expanding-shrinkage, not EWMA (best expanding prior_minutes=20 -> MAE 1.5303, beating every EWMA
halflife tested, best 1.5615 at halflife=20) -- getting to the line is a stable per-minute
foul-drawing tendency (how a player attacks the rim), not a volume/role shot-selection decision,
so it behaves like a skill metric despite being called "attempts" like 2PT/3PT. FT make-rate also
wants a much smaller expanding-shrinkage prior than 2PT/3PT: prior_attempts=15 (MAE 0.3724) beat
both the old prior=100 (MAE 0.3778) and the naive floor (0.3773) -- FT-shooting skill stabilizes
over a far smaller attempt sample than 2PT/3PT accuracy does. Fixed both FT constants
(`PRIOR_MINUTES_FTA=20`, `PRIOR_ATTEMPTS_FTM=15`, both expanding-shrinkage). Re-validated on the
full dev range: FT makes REAL IMPROVEMENT (MAE 1.0425 vs naive 1.0434). All three categories now
pass. Regression test added (`test_ft_scoring_rate_uses_expanding_shrinkage_not_ewma`) checking
FT's shrunk rate against the closed-form cross-player-pooled formula, so a future "just make FT
match 2PT/3PT" refactor can't silently revert this.

**`player_defensive_event_rates.py`** (steals + blocks) is done and validated, with a THIRD
instance of the same lesson -- steals and blocks are both "rare defensive events" but need
opposite smoothing:

**Real finding: steals and blocks need OPPOSITE smoothing families.** Steals confirmed
expanding-shrinkage (prior_minutes=200, a persistent gambling/anticipation skill, MAE
~0.657-0.660 beating every EWMA halflife 5-40 at ~0.668-0.675). Assuming blocks would work the
same way (both "rare defensive events") was a REAL REGRESSION on the full dev range (shrunk MAE
0.4224 vs naive MAE 0.4102, CI (+0.0117,+0.0128)). Direct re-sweep on blocks specifically: every
expanding-shrinkage prior tested (1-1500) was flat-to-worse than the naive floor (best was
prior->0, degenerating to naive, MAE 0.4952), while EWMA halflife=10-12 games clearly won (MAE
0.4938). Makes basketball sense despite being the opposite of steals: block volume is driven by
matchup and rim-protection role (which players you're guarding, recent minutes at center) -- a
volume/role metric like shot attempts or minutes, not a stable individual skill like steals.
Fixed: `BLOCK_HALFLIFE_GAMES=10.0` (EWMA), steals unchanged (expanding-shrinkage prior=200).
Re-validated on the full dev range: steals REAL IMPROVEMENT (MAE 0.5445 vs naive 0.5501), blocks
now REAL IMPROVEMENT too (MAE 0.4126 vs naive 0.4140). Regression test added
(`test_block_rate_uses_ewma_not_expanding_shrinkage`).

This is now the THIRD time in this subsystem alone (minutes vs. shot attempts, 2PT/3PT vs. FT,
steals vs. blocks) that two superficially-similar stats needed opposite smoothing families,
confirming empirically each time rather than assumed by analogy -- treat every new prop category
as an open empirical question, never a copy-paste of the nearest existing category.

**`player_rebounding_rates.py`/`player_playmaking_rates.py`** now have their own `validate_*.py`
scripts (written this session, matching the other categories' pattern), and `BoxScorePlayerTrackV3`'s
full 2015-2025 backfill has since completed. A preliminary single-season smoke test (2015-16 only,
31,141 rows) found OREB/DREB/TOV already beating naive with their existing priors, while AST's
initial `PRIOR_TOUCHES_AST=300` (copy-pasted from TOV's, untested) was a real regression -- fixed
by lowering to 50, the fourth instance in this subsystem of a copy-pasted constant being wrong
when actually tested per-category (minutes vs. attempts, 2PT/3PT vs. FT, steals vs. blocks, and
AST vs. TOV's shared prior).

**Full dev-range re-validation (275,138 player-games, matching every other category's scale)
CONFIRMS all four categories, no further changes needed:**
- OREB: shrunk MAE 0.4401 vs naive 0.4475 (CI (-0.0081,-0.0066)) -- REAL IMPROVEMENT
- DREB: shrunk MAE 0.7211 vs naive 0.7395 (CI (-0.0195,-0.0173)) -- REAL IMPROVEMENT
- AST: shrunk MAE 0.9286 vs naive 0.9402 (CI (-0.0139,-0.0096)) -- REAL IMPROVEMENT (confirms the
  single-season prior=50 fix generalizes to the full range)
- TOV: shrunk MAE 0.7001 vs naive 0.7182 (CI (-0.0194,-0.0169)) -- REAL IMPROVEMENT

**Task #11 (`player_rate_shrinkage.py` primitive through all six per-player rate-category models)
is now fully DONE, every category validated at full dev-range scale with a real, CI-excludes-zero
improvement over its naive floor:** minutes (EWMA), 2PT/3PT/FT scoring (mixed EWMA/expanding-
shrinkage per category), STL/BLK defensive events (opposite smoothing families), OREB/DREB
rebounding, AST/TOV playmaking (all expanding-shrinkage). 17 regression tests (32 assertions)
passing in `tests/test_regression_bugs.py`.

Next: the matchup-difficulty layer (`fetch_boxscore_matchups.py` + `fetch_team_rosters.py` +
`matchup_difficulty.py`, 2017-18+ only), then usage allocation + prop distribution + the live
`generate_props.py` pipeline -- see the approved plan for the full sequencing. Done as of Sec10.

## 9. Game-score model finished: predictive-mode Phase 2, live wiring, and the Phase 4 holdout check (2026-07-25)

Closes out the original 5-phase game-score plan. Four things happened in order: a real bug found
and fixed, predictive-minutes mode validated (the actual deployability gate oracle mode couldn't
answer), Phase 2 wired into the live pipeline and confirmed, and the one-time Phase 4 holdout
check run -- which came back genuinely mixed and required one more diagnostic pass to understand
correctly rather than either accepting or dismissing at face value.

### 9.0 Bug found and fixed: `garbage_time.py`'s exposure weight used a season-wide max, not per-game

`add_garbage_time_weight` computed `total_length = max(28800, stints["endTenths"].max())` across
the ENTIRE `stints` frame passed in -- correct only if that frame is exactly one game, but every
real caller (`prepare_stints`) passes a full season at once. Confirmed on 2015-16: the season-wide
max is 40800 (a real multi-OT game) while a regulation game ends at 28800 -- every OTHER game's
`elapsed_fraction` was being computed against that one game's inflated length, systematically
understating how far along those games actually were and therefore under-flagging garbage time
almost everywhere. Fixed to compute `total_length` per `gameId` (`groupby(...).transform("max")`).
Regression test added (`test_garbage_time_uses_per_game_length_not_season_wide_max`, 15/15 passing).

**Re-ran Phase 2b's oracle-mode result under the fix to confirm it wasn't invalidated**: essentially
unchanged -- total_mae -0.0217 (was -0.0217), margin_mae -0.0350 (was -0.0350), su still NOISE. The
bug existed the whole time Sec7's original oracle result was computed; the fix re-confirms that
result rather than overturning it.

### 9.1 Predictive-minutes mode validated -- captures ~95% of oracle mode's ceiling

Oracle mode (Sec7) proved lineup-awareness carries real signal, but used REAL, already-known
minutes -- a ceiling test, not a deployable prediction. Built `predictive_minutes_shares`
(`lineup_rating.py`): same active-player SET as the real historical game (a fair backtest proxy
for "who was on the active/available roster" -- knowable pre-game in real life too, same logic as
the live pipeline's own RotoWire-based exclusion), but each player's MINUTES value is their own
trailing average over the team's last 10 games, not the real minutes they happened to play that
specific night. `validate_predictive_lineup_adjustment.py` mirrors Sec7's oracle backtest exactly,
swapping only this one input.

**Result (10,594 dev games, same games as oracle mode)**:

| metric | oracle mode (ceiling) | predictive mode (deployable) |
|---|---|---|
| total_mae | -0.0217, CI [-0.0294,-0.0139] REAL | -0.0206, CI [-0.0268,-0.0145] REAL |
| margin_mae | -0.0350, CI [-0.0434,-0.0266] REAL | -0.0336, CI [-0.0403,-0.0265] REAL |
| su | +0.0019, CI includes zero, NOISE | +0.0025, CI [-0.0001,+0.0052], NOISE (barely) |

Predictive mode captures 95% of oracle's total_mae improvement and 96% of its margin_mae
improvement -- trailing-average minutes is nearly as good as knowing real in-game minutes. This is
the real, clean, deployable result oracle mode alone couldn't provide. Logged as
`phase2_rapm_lineup_predictive`.

### 9.2 Phase 2 wired into the live pipeline (Phase 5), confirmed against a real historical slate

`generate_predictions.py`'s `run()` now fits a single fresh RAPM-lite snapshot from all cached
history through the most recent season (`_fit_latest_player_ratings` -- one fit "as of right now",
not the periodic backtest series), builds each team's recent-roster composite
(`team_recent_roster_rapm`) and tonight's active-lineup minutes shares (the already-written but
previously unused `resolve_active_lineup` -- RotoWire Out/Doubtful exclusion + trailing-average
minutes), and applies the resulting off/def adjustment on top of Phase 1's rating before
projecting. Confirmed live against 2018-01-15's real 11-game slate (matching Sec1's earlier Phase-
1-only smoke test): plausible 106-128 point projections, win probabilities spanning 15.6%-91.0%,
both crosswalk-miss warnings surfaced correctly (documented behavior, not a bug), lineup-source tag
printed per game (`predictive (trailing 10-game minutes, N player(s) excluded via RotoWire)`).

**Also fixed while wiring this**: `refresh_data.py` still called the abandoned `fetch_rotation_season`
(Sec5's `GameRotation` pivot never propagated here) and never rebuilt the current season's stints
cache at all -- meaning the live pipeline had no current-season lineup data to compute predictive
minutes from. Fixed: fetches `fetch_traditional_season` instead, and force-rebuilds
`build_season_stints(current, force=True)` every run (the stints cache is DERIVED from the raw
fetches above, not itself fetched, so there's no cost saving to skipping it and a real risk of
serving stale lineup data otherwise).

### 9.3 Phase 4: the one-time holdout check, run 2026-07-25 -- mixed result

Full dev+holdout predictions (13,033 games, Phase 1 + predictive-mode Phase 2) built via the new
`run_final_holdout_check.py` (correctness verified BEFORE the real run: its dev-only subset
reproduced Sec9.1's already-known predictive-mode dev numbers EXACTLY -- total_mae 15.0592,
margin_mae 10.4331, n=10,594 -- before ever touching holdout). Then the one-time confirmatory read:

| metric | dev | holdout | gap 95% CI | verdict |
|---|---|---|---|---|
| total_mae | 15.0592 | 15.3419 | [-0.2350, +0.7797] | NOISE -- no real regression |
| margin_mae | 10.4331 | 11.3100 | [+0.4810, +1.2742] | **REAL REGRESSION -- VETO** |
| su | 0.6428 | 0.6773 | [+0.0127, +0.0551] | REAL IMPROVEMENT on holdout (unusual, not a veto trigger) |

A real margin_mae regression on the exact metric Phase 2 was adopted for. Read at face value, this
looks like Phase 2's dev-confirmed gain didn't generalize -- exactly the shape of a spurious,
overfit result the confirmatory-veto protocol exists to catch. But Phase 1 had never been
holdout-tested on its own before this run, so this bundled check couldn't yet distinguish "Phase 2
caused this" from "Phase 1 already had this problem and Phase 2 is riding along."

### 9.4 Isolation check: the regression is entirely Phase 1's, not Phase 2's

Ran ONE additional, distinct one-time check -- Phase 1 ALONE, a genuinely different configuration
never previously holdout-tested (not a re-run of Sec9.3's same configuration; the confirmatory-veto
protocol is per-configuration, and the NHL sibling's own history of running §26/§32/§34 as separate
one-time checks on separate configurations is the precedent this follows):

| metric | dev | holdout | gap 95% CI | verdict |
|---|---|---|---|---|
| total_mae | 15.0829 | 15.3591 | [-0.2234, +0.7719] | NOISE |
| margin_mae | 10.4721 | 11.3506 | [+0.4856, +1.2672] | **REAL REGRESSION -- VETO** |
| su | 0.6399 | 0.6732 | [+0.0125, +0.0539] | REAL IMPROVEMENT (unusual) |

**Nearly identical to the bundled result on every metric, gap magnitude, and CI.** Phase 1 alone
already carries the exact same real margin_mae regression -- Phase 2 didn't cause it and doesn't
worsen it. Comparing Phase 1-alone to the bundle directly: Phase 2's OWN incremental margin_mae
contribution is a consistent ~0.04-point improvement on BOTH dev (10.4721 -> 10.4331) AND holdout
(11.3506 -> 11.3100) -- Phase 2's dev-confirmed gain DOES generalize to holdout; it's just riding on
top of a pre-existing Phase 1 issue that generalizes (badly) right along with it.

### 9.5 What's actually causing Phase 1's margin regression: confirmed real scoring-era drift

Checked directly against real, already-known outcomes (a pure data description, not a repeated
holdout performance read -- doesn't touch the one-time-read budget): mean actual points/team-game
rose from 102.7 (2015-16) to 115.6 (2025-26), a ~13-point-per-team, largely monotonic rise across
the whole window, with holdout's two seasons (113.8, 115.6) sitting at the high end. Pace rose too
but far less (98.9 -> 99.9) -- the scoring rise is mostly an efficiency/shooting-rate story, not a
pace story. A model whose rating shrinkage adapts to this over time but was dev-validated mostly
against the LOWER-scoring 2015-2023 era would systematically understate the ABSOLUTE SCALE of
margins in the higher-scoring holdout era -- exactly consistent with the observed pattern
(margin_mae real regression, total_mae fine, SU actually improving -- getting the WINNER right
doesn't require correct margin-magnitude calibration). This is structurally the same finding as the
NHL sibling's own Cycle 13/Sec20 scoring-era-drift discovery, arrived at independently here.

**Net verdict**: Phase 2 (RAPM-lite predictive lineup adjustment) **stands, live-wired** -- its own
claimed improvement is real and demonstrated to generalize to holdout, independent of Phase 1's
issue. Phase 1's margin/spread calibration carries a **real, open, honestly-documented caveat**:
margin and spread predictions should be treated with reduced confidence in the current (higher-
scoring) era specifically, pending a future cycle that adapts the rating combine's margin-scale
calibration to scoring-era drift (a decayed/walk-forward-adaptive fix, analogous to the NHL
sibling's own Cycle 13 fix) -- not silently deployed as fully trustworthy, and not blocking
everything else (win probability and totals show no such regression).

Both Phase 4 runs logged: `phase4_final_with_lineup` (the bundle, Sec9.3) and
`phase4_final_team_strength_only` (the isolation check, Sec9.4).

## 10. Matchup-difficulty layer -- DONE. Real, validated result on the 2017-2023 dev subset.

New ingest: `fetch_boxscore_matchups.py` (`BoxScoreMatchupsV3` + `BoxScoreDefensiveV2`, confirmed
live to work 2017-18 onward and fail cleanly before that -- `IndexError`/`AttributeError` from an
empty payload, not a clean 404, matching the real Second Spectrum tracking rollout, not a bug) and
`fetch_team_rosters.py` (`CommonTeamRoster`, full 2015-2025 range, fetched PER SEASON to avoid a
look-ahead leak from a player's later-career position change). Both fully backfilled: 9 seasons of
matchups/defensive data (2017-18..2025-26), 11 seasons of rosters (2015-16..2025-26).

`matchup_difficulty.py` implements the plan's 3-level shrinkage hierarchy (defender-specific ->
position-group-vs-team -> league-average floor) plus the "tonight's merge" weighting function.
`CommonTeamRoster`'s `POSITION` field is used for position-group bucketing, NOT
`BoxScoreMatchupsV3`'s own `positionOff` column -- confirmed live on real 2022-23 data to be BLANK
for fully 49.99% of matchup rows, the same kind of unreliable box-score position field
`build_stints.py` already found and avoided for starters detection.

**Real finding: the two rating levels need OPPOSITE smoothing families** -- the fourth instance of
this project's now-familiar pattern (minutes vs. attempts, 2PT/3PT vs. FT, steals vs. blocks):
- **Defender-specific difficulty** (points allowed per matchup-minute, individual defender) is a
  persistent SKILL -- expanding-shrinkage clearly wins (best MAE 3.978 at prior_matchup_minutes=50
  vs. naive's 4.056, beating every EWMA halflife tested, worst 4.29 at halflife=3).
- **Position-group-vs-team difficulty** (team defensive scheme against a position group) is a
  volatile TEAM-SCHEME metric -- personnel changes, coaching adjustments, and trades shift how a
  team defends a position group far faster than one player's individual defense changes. EWMA
  halflife=10 games (MAE 8.181) beat every expanding-shrinkage prior tested (best 8.227, barely
  better than naive's 8.288) and every other halflife tried.

**Full validation on the 2017-2023 dev subset** (`validate_matchup_difficulty.py`, the matchup
layer's own narrower dev range vs. the six per-player rate models' full 2015-2023 range):
- Defender-specific: shrunk MAE 3.978 vs naive 4.056 (CI (-0.0823,-0.0734), n=175,355) -- REAL
  IMPROVEMENT
- Position-group-vs-team: shrunk MAE 8.181 vs naive 8.485 (CI (-0.3401,-0.2681), n=47,022) -- REAL
  IMPROVEMENT

`opponent_defense_adjustment` implements the actual merge: given tonight's opposing roster, weight
each defender's rating by their SHARE of recent matchup-minutes at the offensive player's position
group (handles switch-heavy defenses -- never a hard 1:1 defender assignment), falling back
whole-cloth to the position-group-vs-team rating (not a partial blend) when the roster's total
matchup-minute exposure at that position group doesn't clear `MIN_MATCHUP_MINUTES_TO_TRUST=20.0`
(a placeholder pending real-slate testing once `generate_props.py` exists, not yet empirically
tuned). Two regression tests added confirming the weighted-blend behavior and the trust-floor
fallback. 19 regression tests (38 assertions) now passing in `tests/test_regression_bugs.py`.

**Task #12 (fetch_boxscore_matchups.py + fetch_team_rosters.py + matchup_difficulty.py) is now
DONE.**

## 11. Composition rule + player-level distributions -- DONE. `usage_allocation.py` + `prop_distribution.py`.

**`usage_allocation.py`** implements the plan's macro-anchor + micro-reallocation rule exactly:
`raw_projected_points` sums each player's 2PT/3PT/FT projected makes (weighted by point value)
from `player_scoring_rates.py`; `compute_usage_shares` normalizes a team's players' (matchup-
adjusted) raw points to shares summing to 1 (clipping negatives to 0 first -- a matchup adjustment
can in principle push a low-usage player's number below zero, which a real usage share can never
be; falls back to equal shares if a whole group clips to 0, avoiding a divide-by-zero);
`allocate_team_points` multiplies those shares by Phase 1+2's ALREADY-RAPM-adjusted team total --
the ONLY place the team total enters. Confirmed on a real historical game (8-player roster): shares
summed to exactly 1.0, allocated points summed to exactly the team total, and the ranking matched
each player's own raw signal (more minutes/raw points -> proportionally more of the total). Two
regression tests confirm the sum-to-team-total invariant and the negative-clip/all-zero edge cases.
REB/AST/TOV/STL/BLK are deliberately left UNANCHORED in v1 (each rate model's own `_proj` column
used directly, no share-based rescaling) -- Phase 1 has no team-level projection for those five
categories, so there's no second signal to conflict with yet; a team-level anchor for them is a
flagged fast-follow, not a silent omission.

**`prop_distribution.py`** is the player-level analog of `score_distribution.py`, reusing its
variance/Student-t math directly (not reimplemented) for high-count stats, plus a new
Poisson-vs-Negative-Binomial branch for low-count discrete stats picked via a variance-to-mean
overdispersion check. Validated on the full dev range (`validate_prop_distribution.py`, chronological
70/30 fit/eval split, mirroring `validate_score_distribution.py`'s own convention):
- **Points** (continuous): Student-t df=9.5 fitted (real, meaningful excess kurtosis), but mean
  log-score on the eval set came back essentially tied (normal=2.9028 vs t=2.9052) -- Normal
  adopted, matching this project's "don't add complexity that isn't clearly earning its keep"
  discipline even when a fitted parameter LOOKS like it should matter.
- **Blocks** (discrete): the overdispersion check did NOT trigger NB (variance/mean ratio came in
  under the threshold on real data) -- Poisson adopted, log-score 0.8736 on 66,937 eval rows.

**Real bug found and fixed**: a player with an expanding-shrunk `blk_rate_per_min` of exactly 0.0
(real -- a perimeter player who has genuinely never recorded a block) projects `mean=0.0`; Poisson
at mu=0 is a genuine degenerate point-mass at 0, so `logpmf(k>0, mu=0)` is EXACTLY `-inf` --
one real garbage-time block for that player poisoned the ENTIRE 66,937-row eval set's mean
log-score to `+inf` on the first real run. Fixed with `MIN_COUNT_MEAN=1e-3`, a floor applied before
evaluating pmf/logpmf in both `over_under_prob` and `log_score` -- a live projection can never be a
certainty that something "never happens". Two regression tests added: one confirming the fix (a
0-mean, nonzero-actual count no longer produces an infinite log-score for either family), one
confirming the NB2-to-scipy-`nbinom` parameter translation actually produces the right mean/variance
(checked against scipy's own computed moments, not trusted by algebra inspection alone).

23 regression tests (45 assertions) now passing in `tests/test_regression_bugs.py`.

**Task #13 (usage_allocation.py + prop_distribution.py) is now DONE.**

## 12. Live props pipeline -- DONE (v1 scope). `active_roster.py` + `generate_props.py`.

**`active_roster.py`**: extracted `games_on_date`, `resolve_active_lineup`, and `build_team_history`
out of `generate_predictions.py` into their own shared module -- both live entry points now call
the exact same active-lineup-resolution logic, so they can't silently drift apart. Confirmed the
refactor changed nothing behaviorally: re-ran `generate_predictions.py` against the same real
2018-01-15 historical slate used to originally validate it (11 games) both immediately after the
extraction and again later from inside `generate_props.py`'s own call -- all three runs produced
identical predictions for every game (e.g. CHA @ DET: 113.6/116.7 every time). All 23 regression
tests still pass.

**`generate_props.py`**: the live daily player-props entry point, run as
`python -m src.pipeline.generate_props [YYYY-MM-DD]`. Reuses `generate_predictions.py`'s own
already-RAPM-adjusted team point totals as the macro anchor (calls it directly rather than
re-deriving team ratings independently -- guarantees props and game predictions can never disagree
on a team's total), resolves both teams' active rosters via `active_roster.py`, projects every
active player's full stat line from all six rate-category models, composes points via
`usage_allocation.py`, and writes one row per (game, player, stat) to
`data/processed/daily_props_{date}.parquet`.

**Real bug found and fixed while wiring this up** (caught by reasoning through the code before
running it, not by a failed run): a player missing from one category's rate log (e.g. a brand-new
call-up with no scoring history yet) produces `NaN` for that category's projected points. Passing
that `NaN` straight into `usage_allocation.compute_usage_shares` would poison the `.sum()` for the
WHOLE team, turning every other player's share into `NaN` too -- the exact same
NaN-poisons-an-aggregate failure mode already found twice elsewhere in this project (the bootstrap
harness, and the `player_scoring_rates.py` validation script). Fixed with an explicit
`raw_projected_points(proj).fillna(0.0)` before computing shares, documented inline rather than
left to be rediscovered by a future crash.

**Verified against the same real 2018-01-15 slate** (matching the plan's own stated verification
method -- spot-check plausibility against a real historical date, not a synthetic one): 3,337
output rows across 334 players, zero negative or NaN `proj_mean` values, and points reconciled
EXACTLY -- summing all of one team's projected `points` rows reproduces `generate_predictions.py`'s
own team total for that game to 4+ decimal places (e.g. CHA @ DET: 113.55/116.65 vs. the printed
113.6/116.7). Per-player point distributions within a team are proportionally sane (top scorer
~14 projected points down to a fringe rotation player ~1, no implausible outliers).

**Known, explicitly-flagged v1 gaps (not silently shipped as complete):**
- **Matchup difficulty is NOT wired into this live call yet**, even though `matchup_difficulty.py`
  is fully built and validated (Sec10). Every output row carries `matchup_adjusted: False`. Wiring
  it live needs each offensive player's own position group (from `CommonTeamRoster`) and the full
  opposing active roster's defender ratings resolved for tonight specifically -- real additional
  live-data-assembly work, a flagged fast-follow.
- REB/AST/TOV/STL/BLK remain UNANCHORED (each rate model's own projection used directly, per
  `usage_allocation.py`'s documented v1 scope) and use an approximated projected-chances/
  projected-touches unit conversion (each player's own trailing chances-per-minute /
  touches-per-minute x projected minutes) rather than a dedicated validated exposure model.
- Active-lineup resolution stays the existing 2-tier design (RotoWire Out/Doubtful + trailing-
  minutes fallback) already documented in `generate_predictions.py` -- unchanged, not revisited here.

**All 6 tasks in the approved player-props plan (#9-#14) are now DONE.** The props subsystem has a
working, validated, real-data-tested v1 pipeline end to end: shared shrinkage primitive -> minutes
-> six per-player rate categories -> matchup difficulty (built, not yet live-wired) -> usage
allocation -> prop distribution -> live daily output. 23 regression tests (45 assertions) passing.
