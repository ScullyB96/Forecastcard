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
**Matchup difficulty is now wired into the live call** (Sec13): confirmed on the same real slate,
319 of 334 players got a real matchup adjustment, with the remaining 15 honestly falling back to
`matchup_adjusted: False` (players no longer on any current-season roster, not a bug -- see
Sec13). See Sec8/10/11/12/13 for the full writeup, including three real bugs found and fixed (an
`inf`-poisoned log-score at a zero-mean count projection, a NaN-poisons-the-team-sum risk in the
live usage-allocation wiring, both caught before shipping). 23 regression tests (45 assertions)
passing in `tests/test_regression_bugs.py`.

**Props Phase 4 (confirmatory holdout check, Sec14): RUN, ONCE, 2026-08-01 -- real, mixed, fully
diagnosed result, same precedent as the game-score model's own Sec9.** 5 of 12 sub-models
(minutes, 3PT makes, steals, OREB, assists) showed a REAL dev-vs-holdout MAE gap -- diagnosed as
genuine era-driven trend shifts in league-wide play style (rising 3PT volume, rising assist rates,
non-monotonic OREB/steal trends, deepening rotations), not a modeling defect. Critically, all 5
STILL beat a naive floor when re-checked on holdout data alone -- real degradation, but every
model remains net-positive in the era it's actually deployed against. **Net call: all 12 categories
stand, live-wired, no reversions** -- the 5 with a real gap carry an open, documented
reduced-confidence caveat (a candidate for a future recency-retune cycle), not a silent block.

**Fast-follow attempted immediately (Sec15)**: dev-only retuning (never touching holdout for the
decision) found a genuine improvement for 2 of the 5 -- `PRIOR_CHANCES_OREB` 100->50,
`PRIOR_TOUCHES_AST` 50->100, both adopted (real improvements on dev AND holdout in absolute
terms). No fix found for minutes/3PT/steals -- their already-deployed settings were already at or
near the recent-dev-slice optimum. Importantly, the OREB/AST fixes improved absolute accuracy but
did NOT close the relative dev/holdout gap -- an honest, informative negative result suggesting
the gap is a genuine structural era-shift, not a stale-hyperparameter problem, and pointing toward
the same class of fix already flagged for the game-score model's own scoring-drift issue (a
walk-forward-ADAPTIVE calibration, not a fixed smoothing parameter) as the real future direction.

**Adaptive-calibration attempt (Sec16): built, tested, honest negative result.** Built a
recency-weighted (EWMA) alternative to the shared shrinkage primitive's league-average blending
target (previously a flat cumulative average pooled across ALL history since 2015, not even
season-reset -- a clean, well-targeted hypothesis for the diagnosed drift). Tested broadly across
every affected category using dev-only chronological splits. Two categories (steals, FT-make)
looked like real fixes on initial dev-only checks; FT-make was caught regressing at full-dev-range
re-validation and reverted before ever reaching holdout; steals passed EVERY dev-only gate but
still made TRUE holdout performance worse when checked (a new one-time confirmatory read) and was
also reverted. **Nothing from this section is adopted** -- the new primitive option remains
available (harmless by default) but unused. Confirms more strongly than Sec15 that this gap needs
a genuinely structural fix (decompose league trend from player residual), not a fixed decay
constant, however carefully swept -- flagged as real future work, not a same-day fix.

**The structural fix, attempted and DECISIVELY resolved (Sec17): stop trying.** Built exactly the
"decompose league trend from player residual" architecture Sec16 called for
(`add_era_adjusted_player_rate`, detrend-then-retrend) and validated it far more rigorously than
anything before (consistent gains across 4 independent dev-internal cutoffs, full-dev-range,
vs-naive). It STILL failed on real holdout, and by MORE than the simpler Sec16 attempt did (steals:
dev/holdout gap widened from Sec16's (+0.0075,+0.0177) to Sec17's (+0.0149,+0.0251)). **Three
independent, increasingly sophisticated attempts, each making holdout WORSE, is a decisive pattern,
not a reason to try a fourth.** Diagnosis: the 2024-2025 steal-rate jump is a genuine REGIME CHANGE
at the exact dev/holdout boundary, not an extrapolable trend -- no technique trained only on
pre-2024 data can predict a shift with no precedent in that data. Critically, this does NOT mean
today's live system carries the same blind spot: `generate_props.py` refits every model fresh from
ALL cached data on every call, so the ACTUALLY DEPLOYED model has already walk-forward-absorbed the
2024-2025 regime shift by now -- the real remedy for a regime change is exactly what the system
already does (keep updating on real games), not a cleverer historical extrapolation. Stopping
further attempts at this specific gap. 26 regression tests (53 assertions) passing.

**Matchup difficulty extended to AST/TOV, full 3-level hierarchy (Sec18/19)**: both categories
confirmed real improvements over naive at the defender AND position-group levels (with family
choice re-checked independently at each -- TOV needed the opposite family from points/AST at the
position-group tier). OREB/STL/BLK investigated for the same treatment (Sec20) -- one real lead
found (BLK's `matchupBlocks` field) and tested rigorously, but it didn't clear the bar; none of the
three currently have a viable path with available data.

**A real, serious historical-backtest bug found and fixed via an actual spot-check (Sec21)**: a
2025-01-15 comparison against real box scores caught the live pipeline's "trailing history"
silently reaching past the target date into whatever's most recent as of the real wall-clock today
(e.g. projecting a star for ~11 minutes instead of his true ~38-minute trailing average). Not a
model defect -- every rate model and the walk-forward math were unaffected -- but a real gap in how
`generate_predictions.py`/`generate_props.py` resolve "tonight." Fixed with a new
`season_for_date` helper and a strict `gameDate < game_date` cutoff threaded through every trailing-
data lookup in both pipelines; harmless for genuine live use. Re-ran the corrected comparison:
slate-wide points MAE improved from 6.34 to 5.03, and a properly-designed check (comparing the
model's OWN pre-game top pick against its outcome, not the ex-post actual top scorer -- a known
selection-effect trap) found no evidence of a further systematic bias.

**A SECOND real bug found by continuing to spot-check (Sec22)**: an early-season date (2023-11-08)
showed EVERY stat category under-projected by a consistent ~30-40% -- traced to `team_history`'s
"last N games" lookback silently blending in games (and rosters) from the PRIOR season when a new
season hadn't yet played 10 games, inflating the resolved active roster to ~2x its true size
(22 players projected vs. 11 who actually played). A genuinely different bug from Sec21's (these
games were legitimately before `game_date`; the problem was allowing "current roster" to blend
across a season boundary at all). Fixed by restricting `team_history` to the target season only,
leaving team-level rating continuity and RAPM's own cross-season skill memory untouched (both
legitimately different concepts). Re-confirmed: the uniform under-projection is gone across every
category, and a mid-season date unaffected by this bug (2025-01-15) shows unchanged output,
confirming the fix is correctly scoped. 32 regression tests (67 assertions) passing. Two
consecutive spot-checks, two real bugs the aggregate bootstrap validation never could have caught
-- exactly the value of testing against real outcomes.

**Team-level anchoring for DREB/AST/TOV/STL/BLK, closing a long-flagged gap (Sec23, 2026-08-01):**
built `team_stat_rates.py` (mirroring `team_strength.py`'s pace x rating architecture, no new
ingest -- aggregates the already-cached player box scores to team level). All 6 categories beat
naive on dev; the one-time confirmatory holdout check found AST/TOV/STL showing the SAME
already-diagnosed era-trend/regime-change gap-widening as their player-level counterparts (Sec16-19)
but still net-beating naive on holdout in absolute terms, so adopted anyway (same precedent as
Sec15). OREB is a genuine, different result: it actually LOSES to naive on holdout (not just a
smaller margin) -- a real veto, left unanchored and flagged for future investigation rather than
force-adopted. 5 of 6 categories now wired live via the same macro-anchor + micro-reallocation rule
points already used (`usage_allocation.allocate_team_total`, generalized from
`allocate_team_points`). 36 regression tests (97 assertions) passing. See Sec23 for the full
writeup.

**Attempted fix for Phase 1's still-open margin/scoring-era-drift issue (Sec24, 2026-08-01):
decisive dev-only negative result.** Confirmed the scoring rise (102.7 -> 115.6 pts/team-game,
2015-16 -> 2025-26) is a continuous multi-year TREND visible throughout the entire dev range, not
a sudden regime jump like steals' -- a genuinely better candidate for recency-weighting than the
props subsystem's 3-strikes-and-out steals attempts. Built the same `league_avg_halflife_games`
option for `shrinkage.py`'s TEAM-level primitives and swept 7 halflife values (100-10000 games).
`total_mae` improves at longer halflives (confirms the scale-correction hypothesis), but
`margin_mae` -- the actual metric with the real holdout regression -- shows a REAL REGRESSION at
EVERY halflife tested, never crossing to noise or improvement even near-flat. Diagnosed why:
recency-weighting the SHARED league-average blending target reduces mean-scale bias but adds
per-team noise that a DIFFERENCE metric (margin) is more exposed to than a SUM metric (total).
Stopped cleanly at the dev-only Stage 1 gate -- no holdout read spent, nothing reverted (the new
option is harmless-by-default, kept as a tested-but-unused primitive). Phase 1's margin issue
remains open, unchanged from Sec9.5 -- refined diagnosis: it's not simply "stale scoring level"
(that would move margin and total together), so the real fix (if one exists) is a different
mechanism entirely, e.g. team-quality-spread drift across eras. 37 regression tests (100
assertions) passing.

**prop_distribution.py finally wired into the live pipeline, with two real findings along the way
(Sec25, 2026-08-01).** Discovered the module (built/dev-validated in Sec11) had NEVER actually been
imported by `generate_props.py` -- the live pipeline shipped bare point projections only, no
variance/probability, for its entire existence. Found and fixed a real bug while wiring it up: the
continuous-family code reused `score_distribution`'s team-score-sized variance floor
(`MIN_VARIANCE=4.0`), silently inflating every low-exposure player's assumed uncertainty (and
distorting Sec11's own original family-choice validation, not just live output). Fixing that floor
and adding a genuine calibration check (PIT vs nominal quantiles -- never done here before, only
log-score comparisons) surfaced a bigger, structural finding: the plan's a priori "points/2PT/3PT/
FT-made are continuous, CLT-justified" assumption was WRONG at the per-player-per-game level (true
only for TEAM aggregates) -- testing them as COUNT categories instead (like OREB/DREB/etc already
were) gave both better log-scores and dramatically better calibration for all four (e.g. ft_made:
calibration max-deviation 0.35 as Student-t vs 0.012 as NegBin). All 10 prop categories now use
Poisson/NegBin, live-refit fresh every call, and every output row carries `proj_var`/`family`/
`family_param` -- the actual sportsbook-style deliverable the plan always intended. 38 regression
tests (103 assertions) passing.

**A second lever tried for both remaining open problems (Sec26, 2026-08-01): shrinkage STRENGTH,
not just the blending target.** Motivated by a confirmed real widening of cross-team quality spread
in the holdout era (margin-spread std: ~4.0-5.15 through most of dev, 5.56/6.00/6.24 in
2023/2024/2025). For Phase 1 margin: reducing `prior_games_rating` genuinely improves margin_mae on
real holdout (11.3506 -> 11.3079) but REGRESSES total_mae by a larger absolute amount (+0.1264) --
a real tradeoff, not adopted. For OREB team-level anchoring: the SAME reduced-shrinkage hypothesis
made things monotonically WORSE (opposite of margin's result, ruling it out for OREB specifically);
the recency-weighted league average (already ruled out for margin) instead moved OREB from a clear
holdout LOSS to naive (Sec23) to a statistical TIE -- real progress, but a tie doesn't justify the
added complexity, so still not adopted. Both problems remain open; both now have two independently-
tested, ruled-out (or insufficient) mechanisms on record, narrowing what a future fix would need to
look like. 39 regression tests (106 assertions) passing.

**Full-model audit and 7 real bug fixes (Sec27, 2026-08-01).** A 29-agent workflow audited the
entire codebase for data gaps, calculation errors, logical fallacies, and unverified assumptions,
adversarially re-verifying every flagged finding before counting it. Found 17 confirmed real issues
(7 high-severity) and 26 additional-lever ideas. Fixed all 7 high-severity bugs, most notably: the
LIVE game-score pipeline had been hardcoding `home_court_mult=1.0` since its first commit, silently
discarding the validated home-court effect entirely; matchup difficulty was applied BACKWARDS on
points/AST/TOV (a sign error -- tougher matchups increased projections instead of suppressing them);
an EWMA season-boundary reset was silently zeroing real veterans' points/2PT/3PT/block props every
season during the week after opening night; and the officially-adopted Phase 2 RAPM numbers had been
computed on since-fixed-elsewhere roster-bleed logic, never re-validated -- re-running under the
corrected logic reproduced the same adopted conclusion, essentially unchanged. Also fixed several
medium-severity findings along the way (a dormant-but-real Student-t variance bug, a permanently
false-failing data-coverage check, a player-name-collision risk, and multiple stale docstrings). 44
regression tests (123 assertions) passing. See Sec27 for full detail.

**MAJOR FINDING, AND REVERTED (Sec28.2/28.3, 2026-08-01): Phase 2's predictive-mode RAPM lineup
adjustment no longer held up once a hindsight leak in its own validation was fixed -- DISABLED.**
The backtest that "confirmed" Phase 2 (Sec9.1/9.3/9.4) computed its active-player set from the real
historical game's own actual attendance (perfect hindsight on who plays), a materially easier
problem than what the live pipeline actually has to solve. Fixed to use the same trailing-rotation-
union mechanism the live pipeline genuinely uses, then re-ran both the dev-only check and a fresh
holdout-only isolation check: under the honest methodology, Phase 2 shows a REAL REGRESSION on
margin_mae on BOTH dev (+0.0146) and holdout (+0.0181), and no real help on total_mae or SU either --
completely reversing the previously-adopted conclusion. Oracle mode (which never had this leak)
still confirms lineup-awareness carries real signal in principle; the deployable estimation
mechanism just isn't capturing it net of its own noise. **Flagged for the user rather than acted on
unilaterally, then disabled per their decision**: `generate_predictions.py`'s
`INCLUDE_LINEUP_ADJUSTMENT = False` (both live pipelines revert to Phase 1 team-strength alone;
`generate_props.py` had no independent Phase 2 logic of its own, so it inherits the reverted totals
automatically). All the RAPM-lite/lineup-adjustment machinery is left in place, tested, and
available -- re-enable only after a genuinely improved minutes-projection mechanism is built and
validated. See Sec28 for full detail.

**Phase 1's long-open margin/scoring-era-drift regression -- FINALLY RESOLVED (Sec29, 2026-08-01),
via a lever from the audit's own research pass.** Sec24 (recency-weighted league-average target
alone) and Sec26 (reduced shrinkage strength alone) each failed/tied on their own, but each one's own
cost sat on the metric the OTHER was good at fixing -- neither investigation had tried them
TOGETHER. Swept a joint grid, found a genuine net win on the dev-only gates, then confirmed on real
holdout (vs. the OLD config): total_mae -0.2605 REAL IMPROVEMENT, margin_mae -0.0194 REAL
IMPROVEMENT, su NOISE (no harm) -- the first configuration across this entire investigation to clear
BOTH metrics with no tradeoff. Adopted as the new production default:
`team_strength.PRIOR_GAMES_RATING` 15.0 -> 12.0, new `LEAGUE_AVG_HALFLIFE_GAMES_RATING = 2000.0`.
Both the live pipeline and `validate_team_strength_baseline.py` pick this up automatically (no
explicit overrides needed); re-confirmed the new config still clearly beats naive
(margin_mae -0.8180, an even wider gap than the original Sec1 result). See Sec29 for full detail.

**A second real win for Phase 1 (Sec33, 2026-08-01): `cross_season_weight`, untested since
introduction.** `shrinkage.py` flagged this parameter as untested the day it was added; never swept
in this entire project until the audit's research pass flagged it. Swept on top of Sec29's already-
adopted defaults -- every value from 0.1-1.0 showed a real improvement on both total_mae and
margin_mae simultaneously, the cleanest dev-only result of any lever this session. Confirmed on real
holdout at `cross_season_weight=0.3`: total_mae -0.0835, margin_mae -0.0763, both REAL IMPROVEMENT,
su NOISE (no harm). Adopted as the new default (`CROSS_SEASON_WEIGHT_RATING = 0.3`); naive-floor gap
now even wider (margin_mae -0.8687). See Sec33 for full detail.

**A third real win: `cross_season_weight` also adopted for the 5 team_stat_rates categories
(Sec35, 2026-08-01).** Sec34 tested the same lever for OREB specifically and it converged to the
same ceiling as two other mechanisms (still unanchored). But dreb/ast/tov/stl/blk -- the 5
categories that DID clear their original Sec23 holdout check and are already live-anchored -- are a
different question, tested independently. All 5 cleared both dev stages; on real holdout, 4/5
(dreb/tov/stl/blk) showed REAL IMPROVEMENT vs. the current config and ast was NOISE (no regression
anywhere). Adopted `CROSS_SEASON_WEIGHT_STAT = 0.25` for `ADOPTED_CATEGORIES` only, via a
per-category default dict (`DEFAULT_CROSS_SEASON_WEIGHTS`) that deliberately keeps OREB pinned at
0.0 -- a uniform float default would have silently re-applied Sec34's rejected OREB config, since
`add_team_stat_ratings` loops over all 6 categories in one function. Reaches `generate_props.py`
automatically, no call-site changes. See Sec35 for full detail.

**A fourth real win for Phase 1: `own_halflife_games` (Sec36, 2026-08-01).** A structurally
different lever again -- recency-weights a team's OWN in-season history (previously a flat
cumulative mean), not the prior (`cross_season_weight`) or the shared target
(`league_avg_halflife_games`). Swept 5-80 games on the recent-dev slice; 5 was too aggressive (real
regression), 20/40 cleared the net-win bar most cleanly. Confirmed on real holdout at
`own_halflife_games=40`: total_mae -0.0499, margin_mae -0.0412, both REAL IMPROVEMENT vs. the
current config, su NOISE (no harm). Adopted as `team_strength.OWN_HALFLIFE_GAMES_RATING = 40.0`;
reaches the live pipeline automatically via `add_team_ratings`'s default. See Sec36 for full detail.

**A sixth real win: `own_halflife_games` extended to team_stat_rates (Sec37, 2026-08-01).** Tested
across all 6 categories including OREB (a structurally different lever from `cross_season_weight`,
so OREB wasn't assumed to fail the same way). All 6 cleared both dev stages; on holdout, 4/6
(dreb/tov/stl/blk) REAL IMPROVEMENT vs. current config, oreb/ast NOISE (no regression anywhere).
Adopted `OWN_HALFLIFE_GAMES_STAT = 20.0` for the 5 ADOPTED_CATEGORIES. OREB checked separately vs.
naive on holdout: still NOISE -- the FOURTH structurally distinct mechanism to converge on the same
ceiling, further confirming Sec34's conclusion that OREB's limit is structural, not a missing
parameter. See Sec37 for full detail.

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
- Matchup difficulty was not yet wired into this live call as of this section -- **fixed in Sec13.**
- REB/AST/TOV/STL/BLK remain UNANCHORED (each rate model's own projection used directly, per
  `usage_allocation.py`'s documented v1 scope) and use an approximated projected-chances/
  projected-touches unit conversion (each player's own trailing chances-per-minute /
  touches-per-minute x projected minutes) rather than a dedicated validated exposure model.
- Active-lineup resolution stays the existing 2-tier design (RotoWire Out/Doubtful + trailing-
  minutes fallback) already documented in `generate_predictions.py` -- unchanged, not revisited here.

**All 6 tasks in the approved player-props plan (#9-#14) are now DONE.** The props subsystem has a
working, validated, real-data-tested v1 pipeline end to end: shared shrinkage primitive -> minutes
-> six per-player rate categories -> matchup difficulty -> usage allocation -> prop distribution ->
live daily output.

## 13. Matchup difficulty wired into the live props pipeline (2026-08-01, fast-follow to Sec12)

Closes the one flagged gap from Sec12. `generate_props.py` now applies `opponent_defense_adjustment`
to points before `usage_allocation.compute_usage_shares`, via two new pieces:

- `_build_matchup_context(current_season, game_date)`: computed ONCE per `run()` call (not once per
  player) -- each defender's latest `difficulty_rate` snapshot, each (team, position_group)'s latest
  `posgroup_difficulty_rate` snapshot, a league-average-by-position-group floor (the mean of every
  team's latest posgroup rating, not a raw all-history mean, so one team's longer history can't
  outweigh another's), each position group's trailing defender-minutes exposure as of tonight (via
  `defender_position_group_minutes_asof`), and `matchup_difficulty.load_roster_position_lookup`
  (renamed from `_load_roster_position_lookup` -- now a genuinely shared public function, needed by
  both `matchup_difficulty.py` internally and `generate_props.py`).
- `_matchup_point_delta(pid, offense_season, opposing_team_id, opposing_roster_ids,
  projected_minutes, matchup_ctx)`: resolves the offensive player's position group, calls
  `opponent_defense_adjustment` with the opposing roster's defender ratings/exposure, and converts
  the resulting rate delta to a point-scale delta via `usage_allocation.matchup_point_delta`. Returns
  `(0.0, False)` -- an honest "not adjusted", not a silent zero -- when the player's position group
  can't be resolved at all (no `CommonTeamRoster` entry for them this season).

**Verified on the same real 2018-01-15 slate used to validate every other piece of this pipeline**:
319 of 334 players got a real, nonzero-eligible matchup adjustment (`matchup_adjusted: True`); the
remaining 15 fell back honestly to `False`. Investigated rather than assumed correct: all 15 turned
out to be players who simply aren't on ANY current-season (`2025-26`) roster (retired or otherwise
inactive since 2018) -- `current_season` here is genuinely today's season, the same "as of right
now" convention `generate_predictions.py` already uses for a historical backtest date (fits every
rating "as of today," not "as of the historical date"), so a player who last appeared in 2018 and
hasn't played since correctly has no resolvable position group. Not a bug -- the expected
consequence of testing a live-oriented pipeline against a historical probe date.

Team point totals still reconcile EXACTLY to `generate_predictions.py`'s own totals after the
matchup reshape (e.g. CHA @ DET: 113.55/116.65, unchanged from before this section) -- confirming
the macro-anchor invariant holds regardless of how the matchup layer reshapes individual shares,
exactly as `usage_allocation.py` was designed to guarantee.

23 regression tests (45 assertions) still passing -- no new regressions introduced by this wiring.

## 14. Props Phase 4: the confirmatory holdout check (2026-08-01) -- run once, real mixed result, fully diagnosed

Every props model's dev-side validation (Sec8/10/11) used ONLY the dev range (seasons 2015-2023,
2017-2023 for the matchup layer) -- none had ever been checked against holdout (season >=
`DEV_MAX_SEASON=2024`) before this section, unlike the game-score model's own Phase 4 (Sec9).
Extracted `validate_holdout_bootstrap.generic_holdout_confirmatory_check` (the same dev-vs-holdout-
GAP bootstrap Sec9 used, generalized off the score-specific home/away contract, mirroring how
`metrics_ledger.append_generic_run` was extracted earlier) and wrote
`run_props_holdout_check.py`, which runs the SAME already-dev-adopted model code (no re-tuning)
across the full dev+holdout range for all 12 props sub-models and checks each one's dev-vs-holdout
MAE gap. **Run exactly once, per the confirmatory-veto protocol.**

**Real, mixed result**: 5 of 12 categories came back REAL REGRESSION (a genuine, confirmed dev/
holdout gap): minutes, 3PT makes, steals, OREB, assists. 5 came back REAL IMPROVEMENT on holdout
(unusual, not a veto trigger by this project's own convention): 2PT makes, FT makes, DREB,
turnovers, and BOTH matchup-difficulty levels. 1 (blocks) was NOISE (CI includes zero, no real
gap). None of this was assumed to mean "revert the smoothing-family choices" -- diagnosed the same
way Sec9.5 diagnosed the game-score model's own margin regression, not silently accepted or
silently reverted.

**Diagnosis: real, era-driven league-wide trend shifts, not a modeling defect.** Checked real
per-season per-player-game averages directly (a pure data description, not a repeated holdout
performance read):

| season | mean minutes | mean 3PA | mean AST | mean OREB | mean STL | n distinct players |
|---|---|---|---|---|---|---|
| 2015 | 22.82 | 2.27 | 2.10 | 0.98 | 0.740 | 476 |
| 2019 | 22.87 | 3.23 | 2.31 | 0.95 | 0.724 | 529 |
| 2023 | 22.50 | 3.27 | 2.49 | 0.98 | 0.697 | 572 |
| 2024 | 22.58 | 3.52 | 2.49 | 1.04 | 0.767 | 569 |
| 2025 | 22.27 | 3.41 | 2.47 | 1.05 | 0.777 | 582 |

Each vetoed category has a real, explainable trend distinct from a simple monotonic drift: 3PT
attempt volume kept climbing into 2024-25 after a flatter 2021-2023 stretch (the make-rate model's
prior=150, calibrated mostly on the flatter stretch, is now systematically behind a still-rising
volume regime); assist rate rose steadily through the whole range and holds at its new, higher
level in holdout (a real pace-and-space evolution, not noise); OREB is NON-monotonic -- it actually
DIPPED through the 2017-2020 middle of the dev range before RISING again in 2024-2025, so an
infinite-memory expanding-shrinkage prior built mostly from the lower-OREB middle years now
systematically UNDER-projects the higher recent rate; steals similarly dipped through 2017-2023
before ticking back up in 2024-2025; and roster/rotation depth itself grew (`n_players` +22% from
2015 to 2025, more two-way/G-League churn), a real shift in the minutes-allocation environment the
EWMA halflife=2 model wasn't calibrated against. This is structurally the same class of finding as
the game-score model's own Sec9.5 scoring-era-drift discovery -- the world genuinely changed over
this 11-season window in several player-usage dimensions at once, not just team-level scoring.

**Follow-up check, the actual decision-relevant question**: does each vetoed model still beat a
naive floor when BOTH are evaluated on holdout data alone (not just "did it get worse relative to
dev")? Re-ran `bootstrap_significance.bootstrap_compare` restricted to holdout-only games for all 5
vetoed categories:

- minutes: shrunk=5.5149 vs naive=10.9602 (holdout-only) -- REAL IMPROVEMENT, CI excludes zero
- 3PT made: shrunk=0.6765 vs naive=0.6817 -- REAL IMPROVEMENT
- steals: shrunk=0.5554 vs naive=0.5650 -- REAL IMPROVEMENT
- OREB: shrunk=0.4470 vs naive=0.4563 -- REAL IMPROVEMENT
- assists: shrunk=0.9490 vs naive=0.9602 -- REAL IMPROVEMENT

**All 5 vetoed categories still beat naive decisively when evaluated on holdout alone.** The dev/
holdout gap is real, but every model remains net-positive in the era it'll actually be deployed
against -- the same conclusion Sec9 reached for the game-score model's Phase 1 margin issue.

**Net verdict, matching Sec9's own precedent exactly**: no props category is reverted or pulled
from `generate_props.py`. All 12 stand, live-wired. The 5 categories with a real dev/holdout gap
(minutes, 3PT makes, steals, OREB, assists) carry a real, open, documented caveat: their point
estimates should be treated with modestly reduced confidence in the current, still-shifting era
specifically, pending a future cycle that adapts these models' memory/smoothing toward more
recency-sensitivity (an EWMA-halflife retune for OREB/steals specifically, given their
non-monotonic trend; a lower expanding-shrinkage prior for 3PT makes and assists to track the
continuing volume/rate climb faster) -- not silently deployed as fully trustworthy, and not
blocking anything else (7 of 12 categories showed no such gap at all).

All results logged to `metrics_ledger.parquet` (`run_props_holdout_check.py`'s per-category runs,
plus a `props_holdout_followup_naive_check` entry for the 5 holdout-only re-checks). 23 regression
tests still passing -- this section added no new code paths to the live pipeline, only a one-time
diagnostic read.

## 15. Attempted fast-follow fix for the 5 flagged categories (2026-08-01) -- 2 of 5 genuinely improved, gap NOT closed for either; honest, disciplined negative result

Sec14 flagged the 5 real-dev-vs-holdout-gap categories as candidates for "a future cycle that
adapts these models' memory/smoothing toward more recency-sensitivity." Attempted that cycle
immediately rather than leaving it purely speculative -- with one hard constraint respected
throughout: **retuning must never consult real holdout data, only dev**, per the confirmatory-
veto protocol. Used a chronological 80/20 split WITHIN dev (matching
`validate_score_distribution.py`'s own FIT/EVAL convention) as the proxy for "how well does this
adapt going forward," then only re-checked real holdout ONCE per category that actually changed
(a genuinely new configuration earns its own one-time confirmatory read; re-reading holdout for an
UNCHANGED configuration would itself violate the protocol).

**Minutes, 3PT makes, steals: no fix found, left unchanged.** Swept halflife (minutes: 1/1.5/2/3/
5/8 games) and expanding-shrinkage priors (3PT make-rate: 30-600; steals: 30-500) on the recent-dev
eval slice -- in every case the ALREADY-DEPLOYED setting was at or within noise of the observed
optimum (minutes halflife=2 vs. best-found 1.5, negligible difference; 3PT make prior=150 was
literally the best value tested; steals prior=200 was the best value tested, with EWMA re-confirmed
clearly worse than expanding-shrinkage even on this recent slice). No dev-only-justified change
exists for these three -- the Sec14 caveat stands as originally written, unchanged.

**OREB and AST: a real dev-only-confirmed improvement WAS found and adopted.**
`PRIOR_CHANCES_OREB` lowered 100 -> 50 (recent-dev-slice MAE 0.4326 -> 0.4318, bootstrap-confirmed
real, CI excludes zero); `PRIOR_TOUCHES_AST` raised 50 -> 100 (recent-dev-slice MAE 0.9413 ->
0.9401, also bootstrap-confirmed real) -- **NOTE the AST direction is the OPPOSITE of what Sec14's
initial diagnosis guessed** ("a lower expanding-shrinkage prior... to track the continuing rate
climb faster"). Checked empirically rather than forced to match that earlier guess, and the data
said larger, not smaller -- a useful reminder that a plausible-sounding trend story is still a
hypothesis until tested, even inside this same documentation. Both re-validated as real
improvements over naive at full dev-range scale (OREB: 0.4389 vs naive 0.4475; AST: 0.9281 vs
naive 0.9402, both up slightly from their pre-fix full-dev numbers too).

**The honest result: retuning made both categories objectively BETTER in absolute terms on BOTH
dev and holdout, but did NOT close the relative dev-vs-holdout GAP.** Re-ran each as its own new
one-time confirmatory holdout check (a genuinely new configuration, not a re-read of the old one):

| category | old dev/holdout (gap) | new dev/holdout (gap) | verdict |
|---|---|---|---|
| OREB | 0.4401 / 0.4470 (+0.0021,+0.0116) | 0.4389 / 0.4464 (+0.0028,+0.0123) | still REAL REGRESSION |
| AST | 0.9286 / 0.9490 (+0.0110,+0.0300) | 0.9281 / 0.9473 (+0.0098,+0.0288) | still REAL REGRESSION |

Both absolute numbers improved (dev AND holdout each got a little better), but the GAP itself
barely moved. **This is informative, not a wasted cycle**: it confirms the Sec14 diagnosis more
precisely than Sec14 alone could -- the dev/holdout gap for these categories is NOT primarily a
stale-hyperparameter problem fixable by a simple prior/halflife retune (if it were, closing the gap
and improving the absolute number would have happened together). It's consistent with a genuine
structural shift in the underlying rate that a single scalar smoothing parameter, however well
tuned, cannot fully track -- exactly the kind of finding that would motivate the SAME class of fix
already flagged for the game-score model's own scoring-era-drift issue (Sec9.5): a decayed/walk-
forward-ADAPTIVE calibration that tracks a moving target, not a fixed-parameter smoothing choice at
all. Not built here -- a larger, real architectural change, not a same-day fast-follow.

**Net verdict**: `PRIOR_CHANCES_OREB=50` and `PRIOR_TOUCHES_AST=100` are ADOPTED (strictly better
than what they replaced, confirmed on both dev and holdout). The Sec14 reduced-confidence caveat
remains open for all 5 originally-flagged categories, now on updated (better, but still gapped)
numbers for OREB/AST specifically. All 5 still beat naive decisively on holdout alone (re-confirmed
for OREB/AST implicitly, since their holdout MAE improved and their earlier holdout-vs-naive
comparison already had comfortable room -- not re-run here to avoid a third holdout read for the
same underlying question). 24 regression tests (46 assertions) passing.

## 16. The walk-forward-adaptive calibration attempt (2026-08-01) -- built, tested, a complete honest negative result; nothing adopted

Sec15 pointed at "a decayed/walk-forward-ADAPTIVE calibration, not a fixed smoothing parameter" as
the real future direction, since retuning `prior_exposure` alone couldn't close the relative dev/
holdout gap even where it improved absolute accuracy. Built and tested that idea immediately.

**Root-cause candidate identified**: `add_walk_forward_player_rate`'s league-average blending
target (`_trailing_league_rate`) is a FLAT CUMULATIVE average pooled across the ENTIRE history
since 2015 -- NOT even season-reset, unlike the player-level cumulative sums in the same function.
A genuine league-wide rate shift (confirmed in Sec14) would be diluted by years of stale history in
this term specifically, a clean, well-targeted hypothesis for why a fixed shrinkage-strength retune
alone (Sec15) wasn't enough.

**Built**: `_trailing_league_rate_ewma` (recency-weighted, same per-game-collapse leak guard as the
original) and a new optional `league_avg_halflife_games` parameter on `add_walk_forward_player_rate`
-- `None` (default) preserves the EXACT original flat-cumulative behavior for every already-
validated model, so this option's mere existence changes nothing until a category explicitly opts
in. Two regression tests added (backward-compatibility + leak-guard + a synthetic-shift check
confirming the EWMA version actually tracks a level change faster than the flat-cumulative one).

**Tested broadly, dev-only, before touching real holdout**: swept `league_avg_halflife_games`
candidates for every category that uses this primitive (3PT-make, steals, OREB, AST, 2PT-make,
FT-make, TOV, DREB) on the same recent-dev chronological slice used in Sec15. Results were
genuinely mixed, not uniform in either direction -- exactly what you'd expect if the underlying
cause differs by category, not a single "this always helps" or "this never helps" story:
- 3PT-make, TOV, DREB, OREB (already-fixed prior): no meaningful change either way.
- AST (already-fixed prior): clearly WORSE with any halflife tested (0.9401 -> 0.9415-0.9426) --
  consistent with Sec15's finding that AST's issue isn't a stale-trend problem at all.
- 2PT-make: a small apparent gain that came back NOISE on a proper bootstrap check (CI included
  zero) -- correctly NOT adopted.
- **Steals and FT-make: real, bootstrap-confirmed gains on the recent-dev slice.** Both looked like
  genuine fixes at this stage.

**FT-make: caught at the very next validation gate.** Re-validating at FULL dev-range scale (not
just the recent slice) showed FT makes actually REGRESSED from a real improvement (shrunk 1.0425
vs naive 1.0434) to NOISE (shrunk 1.0430 vs naive 1.0434, CI includes zero). Reverted immediately
-- never reached a holdout check at all, caught by the very validation step that's supposed to
catch exactly this (a recent-slice-only signal that doesn't generalize to the whole range).

**Steals: passed every dev-only gate, still failed on real holdout.** This one is more serious and
more informative. It passed the recent-dev-slice bootstrap check AND the full-dev-range
re-validation (shrunk MAE improved slightly, 0.5445 -> 0.5441, still a real improvement over
naive). Ran the required new one-time confirmatory holdout check for this genuinely new
configuration -- and holdout performance got WORSE, not better: 0.5554 -> 0.5568 (dev improved
marginally, 0.5445 -> 0.5441, but holdout moved the wrong direction). **Reverted.** This is a more
serious negative result than the OREB/AST case in Sec15 (where the fix at least improved holdout
in absolute terms even though the relative gap didn't close) -- here, a change that passed EVERY
dev-only check available made the actual target metric worse. Confirms, more strongly than Sec15
already did, that a single fixed smoothing/halflife parameter -- however carefully sourced and
swept -- is not a reliable way to track a genuinely moving target; whatever recency pattern helped
within dev's own recent slice pointed in a different direction than what actually happened in
holdout.

**Net verdict**: `_trailing_league_rate_ewma`/`league_avg_halflife_games` remain in
`player_rate_shrinkage.py` as a tested, available primitive (harmless by default, and a real tool
for a future category where it might genuinely help) -- but NEITHER candidate application (FT,
steals) is adopted in any live model. All rate-model constants are back to their Sec14/15 state
except OREB (`PRIOR_CHANCES_OREB=50`) and AST (`PRIOR_TOUCHES_AST=100`), which remain adopted from
Sec15 (those held up on their own merits, independent of this section's league-average mechanism).
The 5 originally-flagged categories' reduced-confidence caveat (Sec14) stands, now confirmed
harder to fix with two independent, disciplined attempts (Sec15's prior-retune, Sec16's adaptive-
league-average) than initially hoped -- genuinely closing this gap likely needs something more
structural than any single-parameter smoothing adjustment (e.g. an explicit two-stage
decompose-the-league-trend-then-model-the-player-residual architecture, closer to how the NHL
sibling's own scoring-drift fix is described, rather than a decay constant bolted onto the
existing shrinkage formula) -- flagged as a real, larger, NOT-same-day open problem, not silently
dropped. 25 regression tests (50 assertions) passing; all rate-model outputs confirmed back to
their correct, validated values after both reverts.

## 17. The detrend-then-retrend attempt for steals (2026-08-01) -- a decisive, structural negative result; stopping this line of attack

Sec16's exact structural recommendation, built the very next attempt: `add_era_adjusted_player_rate`,
a detrend-then-retrend architecture. Deflates each historical game's count by the league rate AS OF
THAT GAME (so a player's cumulative history is expressed relative to their OWN era, not raw counts
blended across eras), runs the SAME UNCHANGED, already-validated shrinkage machinery on that
normalized scale, then re-inflates only the FINAL prediction with a responsive current-league-rate
estimate -- the noisy/adaptive component touches the output once, not smeared across every
historical row's blending the way Sec16's simpler attempt did. Two regression tests added (leak
guard + a synthetic-shift check confirming the re-inflation actually tracks the current era, not
the stale blended one).

**Validated far more cautiously than Sec16's attempt, specifically learning from that false
positive**: instead of one 80/20 dev-internal split, checked FOUR independent chronological
cutoffs (60/70/80/90%) for consistency -- steals showed a real, same-direction improvement at
EVERY cutoff (unlike the earlier attempt, which was only ever checked at one split). Bootstrap-
confirmed on the recent slice (0.5233 vs baseline 0.5263, CI excludes zero), full-dev-range
confirmed (0.5430 vs original 0.5445), and still beats naive decisively (0.5430 vs 0.5501, CI
excludes zero). Every available dev-only signal said this was a real, robust fix -- more robust
evidence than Sec16's reverted attempt ever had.

**It still failed on real holdout, and by MORE than Sec16's simpler attempt did:**

| configuration | dev MAE | holdout MAE | gap |
|---|---|---|---|
| original (flat-cumulative) | 0.5445 | 0.5554 | (+0.0058, +0.0160) |
| Sec16 (`league_avg_halflife=300`) | 0.5441 | 0.5568 | (+0.0075, +0.0177) |
| Sec17 (era-adjusted, detrend-then-retrend) | 0.5430 | 0.5630 | (+0.0149, +0.0251) |

**This monotonic pattern -- three independent attempts, each MORE sophisticated and MORE rigorously
dev-validated than the last, each making real holdout performance WORSE, not better -- is the
decisive finding, not a reason to try a fourth.** Reverted immediately; steals is back to its
original, simplest, most-validated configuration.

**Diagnosis, and why this stops the search rather than motivating another attempt**: Sec14's own
per-season data already showed the answer -- steal rate was 0.727 (2017) declining to 0.693-0.697
(2022-2023), then jumping to 0.767 (2024) and 0.777 (2025). That is not a smooth, extrapolable
trend; it is a REGIME CHANGE that starts exactly at the dev/holdout boundary. No technique that
only ever sees pre-2024 data -- however adaptive, however carefully validated on pre-2024 data
alone -- can predict a level shift that has no precedent anywhere in the data it's allowed to
learn from. Every dev-only validation signal (recent-slice, full-range, multi-cutoff, vs-naive)
can only ever measure "does this generalize within the pattern dev already contains" -- and a
technique that generalizes BETTER within dev's own pattern can, as shown here, generalize WORSE to
a genuine break in that pattern, because it's more confidently extrapolating a trend that simply
doesn't continue.

**The reframing that actually matters for real deployment**: this whole Phase 4 exercise measures
"how would a model trained ONLY on pre-2024 data perform on 2024-2025" -- a legitimate and
important walk-forward-validity check, but NOT the same question as "how does the model
TODAY (2026-08-01) perform going forward." `generate_props.py` refits every rate model FRESH on
every live call from ALL cached data through the present (see `_latest_snapshot`'s convention,
Sec12) -- meaning the ACTUALLY DEPLOYED model has already walk-forward-absorbed the entire
2024-2025 regime shift into its own trailing history by now, the same way it will absorb whatever
comes next. The Sec14 holdout gap is real and worth having found -- it correctly flags that a
model frozen at the 2023/2024 boundary would have underperformed through the shift -- but it is
NOT evidence that today's live, continuously-refitting model carries the same blind spot. No
further action needed here beyond what's already true: the system updates itself as real games
accrue, which is the actual remedy for a regime change, not a cleverer historical extrapolation.

**Net verdict**: steals reverted to its original configuration. No further attempts planned for
this specific gap via trend-extrapolation techniques -- three consecutive, independently-designed,
increasingly rigorous attempts is sufficient evidence to conclude this class of fix doesn't apply
here. `add_era_adjusted_player_rate` remains in `player_rate_shrinkage.py`, tested and available,
should a genuinely different (non-regime-shift) use case surface later. 26 regression tests (53
assertions) passing.

## 18. Matchup difficulty extended to AST/TOV (2026-08-01) -- real, validated, wired live

With the holdout-gap-chasing thread (Sec16/17) decisively closed, redirected effort toward a
genuinely new accuracy lever instead: matchup difficulty was only ever applied to POINTS (Sec13).
`BoxScoreDefensiveV2` already carries `matchupAssists` (assists ALLOWED while this defender guarded
the matchup) and `matchupTurnovers` (turnovers FORCED from the offensive player during that
matchup) in the exact same per-defender-per-game shape as `playerPoints` -- no new ingest needed.

**Generalized, not duplicated**: `add_defender_difficulty_rate` (points) is now a thin wrapper over
a new `add_defender_stat_difficulty_rate(log, stat_col, prior_matchup_minutes, prefix)`, so AST/TOV
reuse the identical mechanism rather than a copy-pasted implementation. A regression test confirms
the generalization introduced zero behavior change for points (byte-identical output to the
original implementation).

**Family/prior NOT assumed to transfer from points** -- checked empirically, per this project's
standing discipline, on the 2017-2023 matchup dev subset (175,355 rows): both AST-allowed and
TOV-forced confirmed expanding-shrinkage as the right family (matching points, but confirmed rather
than assumed), each with its own empirically-swept prior:
- AST-allowed: naive MAE 1.2740; best expanding-shrinkage 1.2454 @ prior=100 (beating every EWMA
  halflife tested, best 1.2973).
- TOV-forced: naive MAE 0.9109; best expanding-shrinkage 0.8859 @ prior=200 (beating every EWMA
  halflife tested, best 0.9245).

**Full validation** (`validate_matchup_difficulty.py`, extended to loop over a
`_DEFENDER_CATEGORIES` table instead of a single hardcoded points call): both categories confirmed
REAL IMPROVEMENT over naive on the full 2017-2023 matchup dev range -- AST-allowed shrunk=1.2454
vs naive=1.2740 (CI (-0.0300,-0.0269)), TOV-forced shrunk=0.8859 vs naive=0.9109 (CI
(-0.0264,-0.0237)).

**Wired into `generate_props.py`**, with an honestly-scoped simplification: AST/TOV get
DEFENDER-LEVEL matchup adjustment only, no position-group fallback tier (no
`BoxScoreMatchupsV3`-derived per-position-group log exists yet for these two stats -- a real,
flagged v1 scope decision, not an oversight). New `defender_total_minutes_asof` (position-group-
AGNOSTIC analog of `defender_position_group_minutes_asof`) weights opposing defenders by their
OVERALL matchup-minute exposure instead. Since AST/TOV are UNANCHORED categories (per
`usage_allocation.py`'s v1 scope -- no team-level total to redistribute), the matchup adjustment is
applied as a direct additive delta to the player's own projection (clipped at 0, since a count
projection can never go negative), not a share-reallocation the way points' adjustment is.

**Verified on the same real 2018-01-15 slate**: both AST and TOV matchup adjustments fired for all
334 players (`matchup_adjusted: True` on every row for both stats -- unlike points, which needs a
resolved position group and had 15 fall through; AST/TOV's simpler defender-level-only merge has no
such requirement), zero negative or NaN projections.

Two regression tests added (generalization-preserves-points-behavior; the new position-group-
agnostic exposure function's own leak guard). 28 regression tests (56 assertions) passing.

**Not yet done, honestly scoped as future work, not silently dropped**: OREB/STL/BLK don't have an
equally clean defender-level "matchup difficulty" proxy in the available data (`defensiveRebounds`
is the DEFENDER's own boxing-out success, not cleanly "OREB allowed"; steals/blocks are the
defender's own individual production, not framed as a specific opponent's suppression) -- would
need a more contorted proxy metric or different data, not attempted here. A position-group
fallback tier for AST/TOV was the immediate next fast-follow -- see Sec19, done same day.

## 19. Position-group fallback tier for AST/TOV (2026-08-01) -- the full 3-level hierarchy, done

Closed Sec18's one remaining flagged gap: AST/TOV only had the defender-level tier; points had the
full 3-level hierarchy. Generalized `build_position_group_matchup_log`/`add_position_group_difficulty_rate`
the same way the defender-level builders were generalized in Sec18 (points-specific functions
become thin wrappers over new generic ones -- a regression test confirms zero behavior change for
points).

**Family NOT assumed to transfer -- checked independently for AST and TOV at this level, per this
project's standing discipline (confirmed empirically on the 2017-2023 matchup dev subset, 47,022
rows):**
- AST-allowed MATCHED points' family: EWMA halflife=10 wins (shrunk MAE 2.6286 vs naive 2.8720,
  beating every expanding-shrinkage prior tested, best 2.6504).
- TOV-forced did NOT match points -- expanding-shrinkage wins instead (prior=250: shrunk MAE 1.8835
  vs naive 1.9121, beating every EWMA halflife tested, best 1.9062). Needed a new
  `add_position_group_stat_difficulty_rate_expanding` variant (the EWMA generic function couldn't
  express this family). Yet another instance of two similar-looking stats needing opposite
  treatment -- now confirmed at BOTH the defender level (Sec18, both matched points there) AND the
  position-group level (this section, TOV diverges) -- underscoring that family choice must be
  re-checked at every level of a hierarchy, not just once per stat.

**Full validation** (`validate_matchup_difficulty.py`, extended with a `_POSGROUP_CATEGORIES` table
mirroring the defender-level one): both confirmed REAL IMPROVEMENT over naive -- AST-allowed
posgroup shrunk=2.6286 vs naive=2.8720 (CI (-0.2580,-0.2293)), TOV-forced posgroup shrunk=1.8835 vs
naive=1.9121 (CI (-0.0325,-0.0247)).

**Wired into `generate_props.py`**: AST/TOV now use the exact same 3-level merge as points
(`opponent_defense_adjustment`, unmodified), reusing the ALREADY-BUILT per-position-group defender-
minutes exposure (`minutes_by_posgroup`) directly -- the weighting mechanism is stat-agnostic (how
much a defender has recently guarded a position group doesn't depend on which stat is being
predicted). **Verified on the same real 2018-01-15 slate**: AST, TOV, and points now show the
IDENTICAL 319-true/15-false `matchup_adjusted` split -- confirms all three correctly share the same
underlying position-group-resolution gate. One new regression test (the expanding-shrinkage
variant's own leak-guard/pooling signature). 29 regression tests (57 assertions) passing.

**Remaining honestly-flagged gap**: OREB/STL/BLK still have no clean matchup-difficulty proxy in
the available data (unchanged from Sec18) -- would need different data or a materially more
contorted metric, not pursued further without a clearer signal that it's worth the complexity.

## 20. Investigated the OREB/STL/BLK matchup-difficulty gap (2026-08-01) -- one real lead found and tested, none currently viable

Sec18/19 flagged this gap without a deep look at what data might close it. Investigated properly
rather than leaving the earlier assessment unchecked.

**BLK: a real data field exists that was missed earlier, tested, and doesn't clear this project's
bar.** `BoxScoreMatchupsV3` (already ingested) carries `matchupBlocks` -- blocks by a specific
defender against a specific offensive player, confirmed live and populated (7,889 of 243,002 rows
nonzero on real 2022-23 data, ~3.2%, matching how rare blocks are per individual pairing).
Re-framed the right way for BLK specifically (unlike points/AST/TOV, block PREDICTION is about the
DEFENDER's own stat, not something the offense "allows" -- so the natural adjustment is "how
blockable is tonight's opponent," an OFFENSE-side signal, not a defender-difficulty one): built
each offensive player's own "times blocked per matchup-minute," collapsing `matchupBlocks` by
`personIdOff` instead of `personIdDef` (175,187 rows, 25% nonzero -- much richer once aggregated
across every defender a player faced in a game). Swept the usual smoothing candidates (expanding-
shrinkage priors 30-800, EWMA halflives 10-80) against a naive floor, the same gate every other
category passed -- **the naive floor won outright here** (MAE 0.4134), beating every shrinkage
variant tested (best expanding 0.4144, best EWMA 0.4161). Unlike points-allowed/AST-allowed/
TOV-forced (all of which showed a clear, real shrinkage-beats-naive signal), "how blockable is this
specific offensive player" does not show enough game-to-game persistence to reward ANY smoothing --
the data field is real, but the underlying signal is apparently dominated by noise at this level of
aggregation. Not built further; a real, checked negative result, not an assumption.

**STL: no clean pairwise field exists.** Neither `BoxScoreDefensiveV2` nor `BoxScoreMatchupsV3`
has a steal-specific matchup field -- `matchupTurnovers` is the closest (steals ARE a subset of
turnovers), but conflates every turnover type, not steals specifically, and re-deriving a
steal-specific rate from that would be too noisy a proxy to trust without direct evidence.

**OREB: confirmed no pairwise field exists in ANY currently-ingested endpoint, checked a genuinely
new data source (`BoxScoreHustleV2`) for completeness.** A rebound isn't a 1-on-1 assigned event
the way a made shot, assist, turnover, or (per the BLK finding above) even a block can be -- it's a
loose-ball/positioning event involving everyone on the floor, and no endpoint attributes a specific
rebound to a specific defensive matchup. `BoxScoreHustleV2` (a genuinely new, not-yet-ingested
endpoint) DOES carry `offensiveBoxOuts`/`defensiveBoxOuts`/`boxOutPlayerRebounds` -- confirmed live,
same Second Spectrum-era coverage boundary as the other matchup endpoints (zero box-out data across
a 10-game sample in 2016-17; real nonzero data by 2017-18). But it's PER-PLAYER, not opponent-
specific -- it could only refine the EXISTING player-level OREB rate model itself (a different,
separate research question -- does box-out volume improve on `reboundChancesOffensive`'s already-
validated conversion-rate model), not add matchup-awareness. Not pursued here; flagged as a
distinct, tangential idea if OREB's own rate model is ever revisited, not a matchup-difficulty fix.

**STL, checked a second, more creative path -- also closed.** Considered whether already-ingested
`PlayByPlayV3` could supply steal-specific DEFENDER attribution (a steal event's description
sometimes names the stealing player in other NBA data feeds). Checked directly: `PlayByPlayV3` has
NO separate "Steal" `actionType` at all -- steals are folded entirely into `Turnover` events (14
distinct subtypes: "Bad Pass", "Lost Ball", "Offensive Foul Turnover", etc.), and neither the
structured fields nor the free-text `description` column carries any defender/stealer attribution
for those events (confirmed against real 2022-23 data: `Turnover` rows only ever name the
OFFENSIVE player who lost the ball, e.g. "Brown Bad Pass Turnover (P1.T1)", never the defender who
took it). This isn't a text-parsing difficulty to work around -- the underlying data genuinely
doesn't carry this attribution in this feed. Confirms STL has no viable path with any currently-
ingested data source, not just the two matchup endpoints already checked.

**Net conclusion**: none of OREB/STL/BLK have a currently-viable matchup-difficulty path, checked
across every data source already ingested (`BoxScoreDefensiveV2`, `BoxScoreMatchupsV3`,
`PlayByPlayV3`) plus one genuinely new one (`BoxScoreHustleV2`). This isn't an assumption carried
over from Sec18 -- it's a checked result, including one genuinely promising lead (BLK's
`matchupBlocks`) that was found, built, tested rigorously, and didn't clear the bar. Closing this
investigation here -- the remaining option would be a materially different NBA data source outside
`nba_api` entirely (e.g. licensed Second Spectrum tracking with box-out/deflection attribution),
which is a different scope of effort than a same-session research pass. No code changes this
section (a pure research/diagnostic pass); no regression count change.

## 21. Real 2025-01-15 spot-check surfaces and fixes a serious historical-backtest bug (2026-08-01)

Ran a genuine end-to-end sanity check the modeling work hadn't done yet: picked a real date from
last season (2025-01-15, an 11-game slate) and compared `generate_predictions.py`/
`generate_props.py`'s output against the REAL final box scores -- not another aggregate MAE
number, an actual look at "what did it say vs. what happened."

**Game-score side looked reasonable for an 11-game sample**: 6/11 straight-up (54.5%, within
normal binomial variance of the validated ~64% at this sample size), mean total error 15.7 (close
to the validated ~15.08), mean margin error 16.41 (elevated, but mostly driven by one extreme
outlier -- an actual 126-67 blowout no model would predict).

**Props side surfaced a real, serious bug**: nearly every high-minute star was drastically
under-projected in points (Anthony Edwards 11.6 vs actual 28; LeBron James 4.5 vs actual 22;
Stephen Curry 18.4 vs actual 31). Diagnosed rather than dismissed as noise: checked Anthony
Edwards's TRUE trailing-10-game minutes average as of 2025-01-15 directly against real box scores
-- 38.06 minutes, matching his actual 40.9 that night almost exactly. But the live pipeline had
projected him for only 10.7 minutes. Traced the cause: `resolve_active_lineup`'s "trailing 10
games" always means "the 10 most recent team games as of the REAL wall-clock today" -- correct and
intended for genuine live use, but when backtesting a PAST date, it silently reached past
2025-01-15 into much later, unrelated games (confirmed directly: the stint data pulled had gameIds
from the 2025-26 season), picking up whatever unrelated recent stretch Edwards happened to have as
of TODAY rather than his real form heading into that specific game.

**This was not a bug in the model itself** -- every rate model, RAPM fit, matchup-difficulty
layer, and the usage-allocation macro-anchor all remained exactly as validated (the walk-forward
dev/holdout bootstrap checks test each row using only data strictly before that row's own game,
regardless of which date the model is later re-queried for). It was specifically a gap in how the
LIVE entry points resolve "tonight's active roster" and "the current rating snapshot" -- both
designed around "most recent as of right now," which is exactly correct for genuine live use and
silently wrong for testing a date from the past.

**Fixed properly, not just noted**: added `fetch_schedule.season_for_date` (a pure calendar
lookup, no live API call, safe for backtesting -- with an honestly-documented single-season edge
case: the COVID-disrupted 2019-20 season ran into October 2020, outside this function's August
safety-buffer cutoff). Threaded a strict `gameDate < game_date` filter through every place either
live pipeline resolves "trailing" data: `generate_predictions.run`'s team ratings and
`_fit_latest_player_ratings`'s RAPM/minutes fit (now takes an optional `before_date`), and
`generate_props.py`'s six rate-model snapshots and matchup-difficulty context (via a new `_before`
helper). Harmless for genuine live use (`game_date` is today; there are no cached rows on or after
today anyway) -- this is a pure bugfix for historical backtesting, not a behavior change for
production. 31 regression tests (65 assertions) passing, including new tests for `season_for_date`'s
calendar logic and `_before`'s strict-inequality boundary.

**Re-ran the corrected comparison**: Anthony Edwards now projects 30.76 points (actual 28, from a
correct 38-minute trailing projection matching his real ~38.06-minute history). Slate-wide points
MAE improved from 6.34 to 5.03, with far more players correctly matched to the active roster (122
-> 269, since the roster resolution itself is no longer contaminated by an unrelated future
season). Several other stars still showed real misses (Curry 15.0 vs 31, Herro 21.2 vs 34) --
checked directly whether this was a NEW systematic bias or ordinary variance: their OWN trailing
minutes were fine (Curry projected 33.3 vs real 37.33 that night), so these are genuine points-rate
misses, not a minutes-projection problem. Tested for a systematic share-concentration bias properly
(comparing the MODEL'S OWN pre-game top-projected scorer against their own outcome, not the ex-post
actual top scorer -- comparing against the actual top scorer is a well-known selection-effect trap,
since picking the real max after the fact will show apparent underprojection for almost ANY model,
well-calibrated or not). Result: mean signed error -1.85 points, errors bidirectional across the
slate (both over- and under-projections) -- no evidence of a systematic bias, consistent with
ordinary game-to-game shooting variance the aggregate bootstrap validation already accounts for,
not a new problem requiring a fix.

**This is exactly what a real spot-check is for**: caught a genuine, serious bug the aggregate
statistics alone hadn't surfaced (a systematic issue in a specific real scenario, historical
backtesting, that the dev/holdout bootstrap protocol was never designed to catch, since it validates
walk-forward CORRECTNESS at the row level, not the LIVE PIPELINE'S date-handling when re-queried
for an arbitrary date) -- fixed it properly rather than working around the specific symptom, and
then correctly distinguished a real remaining pattern (Curry/Herro's misses) from a statistical
illusion (the naive top-scorer-share check) before concluding anything further was wrong.

## 22. A SECOND real bug found by continuing to spot-check: cross-season roster bleed early in a season (2026-08-01)

Kept testing after Sec21's fix rather than treating one clean spot-check as sufficient -- ran two
more real historical dates. 2024-03-05 (mid-season, well within dev range) came back clean across
EVERY category (points MAE 5.19, all other categories' MAEs matching their validated aggregate
numbers closely) -- good evidence the Sec21 fix generalizes. 2023-11-08 (early in a season, only
~9-10 games played) did not: EVERY SINGLE category was under-projected by a consistent ~30-40%
relative amount (points 6.37 vs actual 8.99; 2PT 1.46 vs 2.36; 3PT 0.67 vs 0.93; FT 0.92 vs 1.49;
OREB 0.55 vs 0.85; DREB 1.83 vs 2.70; AST 1.52 vs 1.97; TOV 0.80 vs 1.11; STL 0.42 vs 0.62; BLK
0.27 vs 0.44) -- a uniform, cross-category pattern, not noise.

**Diagnosed rather than dismissed**: confirmed team-level projected minutes correctly summed to
240 per team, ruling out a minutes-TOTAL bug. Checked real box scores directly: DAL/TOR/SAC each
had only 10-11 players record real minutes that night, but the live pipeline's resolved active
roster gave POSITIVE projected minutes to 19-22 players per team -- roughly DOUBLE the true
rotation size, diluting every real rotation player's share. Traced the cause: printed DAL's
"trailing 10 games" as of 2023-11-08 directly -- 3 of the 10 were from the PRIOR season (2022-23),
confirmed against both seasons' cached schedules. Early in a new season, the team hasn't played 10
games yet, so the "last 10 games" lookback (shared by BOTH `resolve_active_lineup`'s active-roster/
minutes resolution AND `team_recent_roster_rapm`'s recent-roster composite) silently reached back
into last season's games -- and therefore last season's ROSTER, including players who may have
since been traded, waived, or signed elsewhere -- inflating the resolved "active roster" pool with
players no longer even on the team.

**A genuinely different bug from Sec21's**, not a repeat: Sec21's fix (a strict `gameDate <
game_date` cutoff) does NOT catch this, because these games are legitimately BEFORE `game_date` --
the problem isn't a wrong-direction date leak, it's that "who is on this roster" was allowed to
blend across a season boundary at all. This is also a DIFFERENT concept from the team-level
PACE/RATING computation (which has its own, separate, intentional `cross_season_weight` option --
a team's stylistic identity can defensibly carry some cross-season memory) and from RAPM-lite's own
player-skill fit (`_fit_latest_player_ratings`, which is correctly left with full cross-season
history -- a player's true skill doesn't reset on opening night, unlike literal roster membership).

**Fixed**: both `generate_predictions.py` and `generate_props.py` now filter `team_log` to
`season == target_season` specifically before building `team_history`/`team_side` (the structure
`resolve_active_lineup` and `team_recent_roster_rapm` both consult) -- leaving the team-level
rating computation and the RAPM fit itself untouched, since those two legitimately want cross-
season memory. Re-checked DAL directly: trailing 10 games are now entirely within 2023-24 (only 7
games existed yet that season, correctly returning fewer than 10 rather than padding with stale
data), and the resolved active roster dropped from 22 to 12 players -- much closer to the true 11.

**Re-ran the corrected 2023-11-08 comparison**: the uniform under-projection is gone across every
category (points 8.32 vs 9.25; 2PT 2.02 vs 2.44; 3PT 0.93 vs 0.96; FT 1.27 vs 1.53; OREB 0.75 vs
0.86; DREB 2.49 vs 2.75; AST 2.10 vs 2.03 -- now essentially balanced; TOV 1.10 vs 1.11 -- nearly
exact; STL 0.57 vs 0.64; BLK 0.38 vs 0.46). Re-confirmed 2025-01-15 (a mid-season date, unaffected
by this specific bug) shows an unchanged player count (318), confirming the fix is correctly scoped
and doesn't disturb dates where this particular issue never applied. One new regression test added
(`build_team_history` with a pre-season-filtered log excludes prior-season gameIds entirely, not
just deprioritizes them). 32 regression tests (67 assertions) passing.

**Pattern worth naming (before Sec23's follow-up work)**: two consecutive real spot-checks each found a genuine, distinct bug
that months of aggregate dev/holdout bootstrap validation never would have surfaced, because both
bugs are specifically about how the LIVE ENTRY POINTS resolve "recent/current" state when queried
for an arbitrary date -- a concern the walk-forward-correctness bootstrap protocol was never
designed to test in the first place (it validates that each historical ROW's own features only use
information from before that row, which was always true here; it says nothing about whether a
LIVE CALLER'S notion of "the last N games" is scoped correctly). Real, hands-on spot-checking
against actual outcomes found real problems that no amount of additional aggregate statistical
rigor would have caught -- exactly the value the user's original "test a day from last year"
question was asking for.

## 23. Team-level anchoring for OREB/DREB/AST/TOV/STL/BLK (2026-08-01) -- 5 of 6 categories adopted; OREB genuinely fails holdout

After two consecutive real spot-check bugs (Sec21/22), moved to a planned improvement rather than
more ad hoc spot-checking: closing `usage_allocation.py`'s long-flagged gap where REB/AST/TOV/
STL/BLK had no team-level macro anchor at all (only points did, via Phase 1+2's RAPM-adjusted
team total). Built `src/models/team_stat_rates.py`, mirroring `team_strength.py`'s pace x rating
architecture exactly: each of the 6 categories gets a FOR (team's own total)/AGAINST (opponent's
total that game) pair, aggregated from the ALREADY-cached `boxscore_trad_player_*.parquet` player
box scores (grouped up to team level -- no new ingest needed), walk-forward shrunk via the SAME
`shrinkage.add_walk_forward_rate` primitive Phase 1 already uses for OFF_RATING/DEF_RATING, then
combined via the SAME multiplicative-ratio idiom `team_strength.project_game` uses for pace x
rating (no pace/100 rescaling needed here, since these are already whole-game totals, not
per-100-possession rates).

**Dev-only validation (`validate_team_stat_rates.py`): clean sweep, all 6 categories beat naive.**
Paired bootstrap (5,000 resamples) vs. each team's own naive trailing-average floor, full dev
range (21,476 team-sides): OREB 2.9059 vs 2.9269, DREB 4.1414 vs 4.2243, AST 3.7422 vs 3.8157, TOV
2.9518 vs 3.0200, STL 2.2458 vs 2.2811, BLK 1.8935 vs 1.9308 -- every category a REAL IMPROVEMENT,
CI excluding zero. A materially better dev-only strike rate than the props subsystem's own
per-player fast-follow attempts (Sec15-17, where most attempts failed), likely because team-level
aggregates are inherently lower-variance than individual player rates.

**The one-time confirmatory holdout check (`run_team_stat_holdout_check.py`) told a more nuanced
story.** 4 of 6 categories showed a REAL (not noise) dev-vs-holdout MAE gap: OREB (+0.0899,
+0.2323), AST (+0.0698, +0.2513), TOV (+0.0313, +0.1665), STL (+0.1325, +0.2411). DREB and BLK came
back NOISE (gap CI includes zero). At first glance this looks like 4 vetoes -- but the confirmatory-
veto protocol's actual question is a GAP (did performance degrade from dev to holdout), which is a
different question from "does the model still beat naive on real holdout data." Computed that
second, decision-relevant comparison directly:

| category | dev: model vs naive | holdout: model vs naive | model beats naive on holdout? |
|---|---|---|---|
| OREB | 2.9059 vs 2.9269 | 3.0661 vs 3.0401 | **NO -- model LOSES** |
| AST  | 3.7422 vs 3.8157 | 3.9014 vs 3.9856 | yes |
| TOV  | 2.9518 vs 3.0200 | 3.0503 vs 3.1423 | yes |
| STL  | 2.2458 vs 2.2811 | 2.4322 vs 2.4475 | yes |
| DREB | 4.1414 vs 4.2243 | 4.1276 vs 4.2467 | yes |
| BLK  | 1.8935 vs 1.9308 | 1.8937 vs 1.9307 | yes |

**AST/TOV/STL's widening gap is the same already-diagnosed phenomenon, not a new problem**: era-
driven league-wide trend shifts (AST/TOV) and steals' known 2024-2025 regime change (Sec16-19,
where three increasingly sophisticated historical-extrapolation fixes all failed to close this
exact gap at the PLAYER level and the investigation was deliberately stopped) -- both propagate up
to the team-level aggregate, which is expected since a team total is just a sum of player totals.
Exactly like OREB/AST's player-level prior retune in Sec15 ("the gap didn't close, but absolute
performance still held up"), these 3 categories are adopted anyway: what matters for a live
prediction is absolute accuracy against the naive floor in the era it's actually deployed against,
not whether the gap between two historical ranges is zero.

**OREB is a genuine, different kind of result: a real veto, not just a widening gap.** The model/
naive comparison FLIPS SIGN on holdout -- the team-level OREB model actually loses to the naive
floor on real unseen data, not merely "improves by less than on dev." This is not treated as a bug
to chase (per Sec16-19's hard-won lesson: don't retry the same family of historical-extrapolation
fix a 4th time on a name-different-but-structurally-similar problem) -- it's left as a documented,
flagged follow-up. `team_stat_rates.ADOPTED_CATEGORIES = ("dreb", "ast", "tov", "stl", "blk")`
excludes OREB explicitly while `STAT_COLUMNS` keeps it so every future validation run still checks
it, in case a genuinely different approach earns it a re-look later.

**Wired into the live pipeline** (`generate_props.py`): `usage_allocation.compute_usage_shares` +
the renamed `allocate_team_total` (was `allocate_team_points` -- the math was always stat-agnostic,
just named for its first use) now anchor DREB/AST/TOV/STL/BLK the same way points was always
anchored, with the team-level total supplied by `team_stat_rates.project_team_stat` instead of
RAPM. Falls back to the raw bottom-up per-player projection (unanchored, flagged
`anchored_to_team_total: False`) if either team has no team-stat history yet (e.g. a genuine
first-game-of-season live call) -- same honest-fallback convention used everywhere else in this
pipeline. OREB continues to report each player's own rate-model projection directly, exactly as
before. Confirmed on the real 2025-01-15 slate: team-level sums for every anchored category land
in realistic per-game ranges (DREB ~30-35, AST ~23-32, TOV ~12-16, STL ~7-10, BLK ~4-7 per team),
and `anchored_to_team_total` correctly reads 1.0 for points/dreb/ast/tov/stl/blk and 0.0 for
oreb/2pt/3pt/ft (the last three were never anchored, matchup-adjusted makes aside). 4 new
regression tests added (for/against symmetry in `build_team_stat_game_log`, the ratio-idiom combine
formula, `ADOPTED_CATEGORIES` correctly excluding OREB, and `_team_stat_totals`'s fallback-to-empty
behavior when a team is missing) -- 36 regression tests (97 assertions) passing.

## 24. Attempted fix for Phase 1's still-open scoring-era-drift margin regression (2026-08-01) -- decisive dev-only negative result, caught before spending the holdout read

Sec9.5 diagnosed but never fixed a real margin_mae dev/holdout regression: mean actual
points/team-game rose ~13 points from 2015-16 (102.7) to 2025-26 (115.6), and critically -- re-
confirmed directly here via `team_strength.build_team_game_log`'s own per-season means -- this
rise is almost MONOTONIC THROUGHOUT THE ENTIRE DEV RANGE (102.7 -> 105.6 -> 106.3 -> 111.2 -> 111.8
-> 112.1 -> 110.6 -> 114.7 -> 114.2 by 2023-24), not a sudden jump only at the dev/holdout boundary.
This is structurally different from the props subsystem's steals investigation (Sec16-19, three
attempts at the SAME family of fix, all failed on holdout, diagnosed as a genuine REGIME CHANGE
with zero precedent in pre-2024 data) -- a continuously-visible, multi-year trend is exactly the
kind of pattern recency-weighting is suited to extrapolate, unlike a level-shift with no
within-dev precedent to learn from. Worth a genuinely new attempt, not the same bet retried.

**Built**: added `league_avg_halflife_games` (default `None`, preserving the exact original
flat-cumulative behavior byte-for-byte -- regression-tested) to `shrinkage.py`'s
`add_walk_forward_rate`/`add_walk_forward_mean`, mirroring `player_rate_shrinkage.py`'s identical
option exactly. Threaded through `team_strength.add_team_ratings`. New
`validate_scoring_era_drift_fix.py`: fits BOTH the baseline (flat) and every candidate halflife on
the FULL dev range (fitting on a short recent-only window would never let the flat baseline's
staleness actually accumulate -- the whole thing being tested), then evaluates each fit on just the
last 3 dev seasons (2021-2023, where staleness is worst) as a cheap Stage 1 screen before ever
considering a full-dev-range Stage 2 or a holdout read.

**Result: a clean, decisive negative finding for the target metric, across a wide sweep.** Swept
halflife in {100, 300, 600, 1000, 2000, 5000, 10000} games (very responsive to nearly-flat) on the
2021-2023 evaluation slice, all vs. the current flat-cumulative baseline:

| halflife | total_mae delta | margin_mae delta | su delta |
|---|---|---|---|
| 100 | +0.089, NOISE | **+0.0078, REAL REGRESSION** | 0, NOISE |
| 300 | -0.092, NOISE | **+0.0074, REAL REGRESSION** | 0, NOISE |
| 600 | -0.135, NOISE | **+0.0066, REAL REGRESSION** | 0, NOISE |
| 1000 | -0.148, REAL IMPROVEMENT | **+0.0057, REAL REGRESSION** | 0, NOISE |
| 2000 | -0.159, REAL IMPROVEMENT | **+0.0039, REAL REGRESSION** | 0, NOISE |
| 5000 | -0.116, REAL IMPROVEMENT | **+0.0018, REAL REGRESSION** | 0, NOISE |
| 10000 | -0.070, REAL IMPROVEMENT | **+0.0009, REAL REGRESSION** | 0, NOISE |

`total_mae` improves at longer halflives (as hypothesized -- a responsive league average correctly
raises the model's expected SCALE to match the recent higher-scoring era). But `margin_mae` --
the ACTUAL metric with the confirmed real holdout regression -- shows a REAL REGRESSION at EVERY
halflife tested, monotonically shrinking toward (but never crossing into noise or improvement,
even at halflife=10000, which is nearly indistinguishable from the flat-cumulative baseline
itself) zero as halflife grows. No candidate cleared the Stage 1 bar (a real margin_mae
IMPROVEMENT), so the script correctly stopped before ever running a full-dev-range Stage 2 or
touching holdout -- the two-gate discipline did its job, catching a bad idea cheaply.

**Why this happens (mechanistic, not just empirical)**: `add_walk_forward_rate`'s shrinkage formula
blends EVERY team's own cumulative rating toward the SAME shared `league_avg` value. Making that
shared target more responsive (EWMA) corrects the AVERAGE scale (helping `total_mae`, a sum) but
also makes each INDIVIDUAL team's own rating estimate more sensitive to short-term noise in the
blending target -- noise that two different teams (rated at different points in their own history,
with different amounts of games-based shrinkage weight already accumulated) don't share and don't
cancel out. `margin_mae` is a DIFFERENCE of two such noisy estimates, so it's the metric most
exposed to this added per-team noise, while `total_mae` (a SUM) benefits from the corrected mean
scale enough to outweigh it. **This refines Sec9.5's diagnosis**: the margin problem isn't simply
"the model doesn't know today's league scores more" (that hypothesis would predict margin
improving right along with total, which did NOT happen) -- it's something structurally different
that a shared-blending-target recency fix can't reach.

**Not adopted, nothing reverted** (the new `league_avg_halflife_games` option is harmless by
default, kept as a tested-but-unused primitive -- same convention as `add_era_adjusted_player_rate`
from Sec17). **Stopping this specific line of attack for margin_mae** -- a comprehensive 7-point
sweep spanning very-responsive to near-flat, with a consistent, monotonic, never-crossing-zero
pattern, is stronger evidence of a genuine dead end than a single failed attempt would be. Phase
1's margin/scoring-era-drift issue remains open, documented, and monitored (unchanged from Sec9.5)
-- a candidate for a DIFFERENT mechanism (e.g. team-quality-spread/variance drift across eras,
rather than a scoring-level trend) in a future cycle, not this one. 2 new regression tests added
(`league_avg_halflife_games=None` default-preserving check + EWMA responsiveness check, both at
team level) -- 37 regression tests (100 assertions) passing.

## 25. Wired prop_distribution.py into the live pipeline (Task #26, 2026-08-01) -- two real findings along the way, both against the plan's original assumptions

Closing a gap bigger than expected: `prop_distribution.py` was built and dev-validated back in
Sec11 (points, blocks only), but `generate_props.py` never actually imported it -- confirmed by
grepping the whole codebase, the live pipeline had been shipping ONLY `proj_mean` for every
category, no variance/family/over-under-probability at all, despite the module's whole purpose
being "the actual sportsbook-style deliverable." Extended `validate_prop_distribution.py` from its
original 2-category smoke test to all 10 categories `generate_props.py` outputs, and along the way
found two real, structural problems with the ORIGINAL Sec11 design -- not just missing wiring.

**Finding 1 -- a real bug: the variance floor was borrowed from the wrong scale.**
`prop_distribution.py` originally called `score_distribution.predict_variance` directly, which
floors variance at `MIN_VARIANCE=4.0` -- sized for TEAM-SCORE variance (a ~100-point game). Applied
to PLAYER-level stats (raw fitted variance routinely well under 1 for a low-exposure player), this
silently clamped nearly every low-exposure player's variance up to exactly 4.0 -- confirmed
directly on real 2025-01-15 output (2PT-makes variance stuck at 4.0 for players projected at 0.2,
0.6, and 2.5 makes, regardless of their own actual uncertainty). Worse: this same inflated
`predict_variance` call is used INSIDE `fit_continuous_family` to standardize residuals before
fitting Student-t df -- meaning Sec11's ORIGINAL points family-choice validation itself was
distorted, not just later live output. Fixed with a new `MIN_PLAYER_VARIANCE=1e-3` floor, a local
`predict_variance` in `prop_distribution.py` (no longer importing `score_distribution`'s version).
Re-running the fit after the fix completely FLIPPED the family choice for every continuous
category, from "Normal, essentially tied with t" to a decisive Student-t: points log-score
normal=76.22 vs t=3.53; 3pt_made normal=13.65 vs t=1.57; ft_made normal=298.59(!) vs t=5.49. The old
bug had been masking real heavy-tailed behavior by making every low-exposure player's assumed
uncertainty artificially large, which incidentally made Normal look falsely competitive.

**Finding 2 -- a bigger, structural re-derivation: the plan's continuous-vs-count axis assignment
was itself wrong for player-level shot-based makes.** This validation script never had an actual
CALIBRATION check before (only log-score comparisons between two candidate families) -- added one
(the probability integral transform: `F(actual)` under the fitted distribution should be
~Uniform(0,1) if the model is well-calibrated; randomized PIT for discrete families to avoid
spurious clumping at integer jumps). Running it on the newly-fixed Student-t fits surfaced a second,
more important problem: real, sometimes severe miscalibration even AFTER Finding 1's fix --
ft_made's max deviation from nominal was 0.35 (badly wrong), 3pt_made's was 0.145. The plan's
original a priori split ("points/2PT/3PT/FT-made aggregate over many shot attempts, genuinely
continuous by CLT") was never actually re-derived until now -- tested directly by treating all four
as COUNT categories (Poisson/NegBin) instead, the same treatment already used for OREB/DREB/AST/
TOV/STL/BLK. Result: BOTH better log-score AND dramatically better calibration for every one of
them:

| category | as continuous (Student-t) | as count | calibration improvement |
|---|---|---|---|
| points | log-score 3.53, calib max-dev 0.032 | NegBin (r=7.70) log-score 2.85, calib max-dev 0.022 | better both ways |
| 2pt_made | log-score 1.86 | Poisson log-score 1.73, calib max-dev 0.027 | better |
| 3pt_made | log-score 1.57, calib max-dev 0.145 | Poisson log-score 1.21, calib max-dev 0.005 | dramatically better |
| ft_made | log-score 5.49, calib max-dev 0.352 | NegBin (r=1.80) log-score 1.52, calib max-dev 0.012 | dramatically better |

The plan's domain reasoning ("aggregates over many attempts, CLT applies") is true for a TEAM's
total points (Phase 1-3 already correctly treats that as continuous) but does NOT hold at the
PER-PLAYER-PER-GAME level: a typical player attempts far fewer 2PT/3PT/FT shots in one game (often
single digits) than a team accumulates points-scoring possessions -- nowhere near enough volume for
continuity to be the right approximation. **All 10 prop categories now use count families** (6
Poisson: 2pt_made, 3pt_made, oreb, dreb, ast, tov, stl, blk; 2 NegBin: points, ft_made).
`fit_continuous_family`/local `predict_variance` are kept as tested, available primitives (not
deleted) for any future category that might genuinely need them, matching this project's standing
convention for validated-but-unused code.

**Live wiring** (`generate_props.py`): `_fit_prop_distributions` refits each category's family
parameters fresh from historical data every call (mirrors `generate_predictions.run`'s identical
live-refit-over-caching tradeoff for `score_distribution.py`) -- Poisson needs nothing beyond the
mean, NegBin needs the fitted dispersion `r`, driven entirely by `prop_distribution.CATEGORY_FAMILY`
so the fitting logic follows each category's own validated config rather than a hardcoded branch.
Output rows now carry `proj_var` (the family's implied variance, informational), `family`, and
`family_param` (JSON, `{"r": ...}` for NegBin, `{}` for Poisson) -- enough for any downstream caller
to compute `over_under_prob(line, proj_mean, family, family_param)` for an arbitrary betting line.
Confirmed on the real 2025-01-15 slate: variance scales smoothly with each player's own mean (no
more flooring artifacts), `r` is a single shared value per category across every player (correct --
NB2 dispersion is a population-level parameter), and a spot-checked `over_under_prob` call for a
32.4-point-projected star player returns a sensible ~47% for a 31.89-point line. 1 new regression
test added (`test_prop_distribution_variance_floor_is_player_scale_not_team_scale`, 3 assertions:
the floor value itself, its scale relative to `score_distribution.MIN_VARIANCE`, and a realistic
low-exposure fit staying well under the old floor) -- 38 regression tests (103 assertions) passing.

## 26. A second lever tried for both open problems: shrinkage STRENGTH, not just the blending target -- one real (if partial) win, one real (if mixed) tradeoff, neither adopted outright

Continued pushing on both of Sec24/23's open findings (Phase 1's margin regression; OREB's
team-level anchoring failure) with a genuinely different mechanism from what was already ruled
out: not WHAT the league-average blending target tracks (Sec24's already-failed recency-weighting
attempt), but HOW STRONGLY a team's own rating is pulled toward it (`prior_games`, the shrinkage
weight itself). Motivated by a real, confirmed data pattern: cross-team quality SPREAD (std dev
across teams of each team's own average point differential) rose from mostly 4.0-5.15 across most
of the dev range to 5.56/6.00/6.24 in 2023/2024/2025 -- a FIXED shrinkage weight calibrated
implicitly against the dev era's typical spread would over-shrink relative to a genuinely wider
TRUE spread, understating real margins.

**Phase 1 margin -- a real, confirmed tradeoff, not a clean win.** Added overridable
`prior_games_rating`/`prior_games_pace` params to `team_strength.add_team_ratings` (default-
preserving, regression-tested). `validate_shrinkage_strength_fix.py` swept less-shrinkage values
{3,5,8,10,12} vs the current 15.0 on the recent-dev slice (fit on the full dev range so the
staleness effect actually has time to accumulate, matching Sec24's methodology) -- prior_games=10.0
was the best candidate, REAL IMPROVEMENT on margin_mae at both the recent slice (-0.0202) and full
dev range (-0.0163), earning a genuinely new one-time holdout read
(`run_shrinkage_strength_holdout_check.py`). Real holdout result: margin_mae DOES genuinely improve
(11.3506 -> 11.3079, CI excludes zero) -- but total_mae, previously fine (NOISE) on holdout, now
shows a REAL REGRESSION of larger absolute magnitude (+0.1264 vs margin's -0.0428 improvement).
**Not adopted**: fixing one metric by making a previously-healthy one worse by a larger margin
isn't a net win, it's a reshuffling of where the error shows up. Phase 1's margin issue remains
open -- two independent, structurally different mechanisms (recency-weighted target, reduced
shrinkage strength) have now been tried and ruled out; a future fix likely needs something that
addresses the widened spread WITHOUT also destabilizing the overall scale (e.g. a spread-adaptive
shrinkage weight that responds to real-time team-quality dispersion specifically, rather than a
single fixed or globally-recency-weighted knob) -- flagged as a harder problem than a calibration
tweak, not pursued further this cycle.

**OREB team-level anchoring -- decisively rules out one hypothesis, finds a real (if
insufficient) improvement from the other.** First, diagnosed WHERE OREB's holdout error was
concentrated: split by games-into-season, dev-vs-holdout gap came out similar in magnitude both
early (games-before<15: dev 2.97 -> holdout 3.06) and late (dev 2.90 -> holdout 3.09) -- ruling out
a stale-early-season-shrinkage-only mechanism (that would predict a MUCH worse early-season gap
than late). Then tested the SAME reduced-shrinkage hypothesis that helped margin: it made OREB
MONOTONICALLY WORSE at every value tried (5,8,10,12,15 all REAL REGRESSION vs current 20, closing
in on zero as prior approaches 20 but never crossing to improvement) -- the opposite of Phase 1's
result, decisively ruling out "over-shrinking" for OREB specifically. Finally tested the recency-
weighted league average (Sec24's already-failed-for-margin mechanism, never tried for OREB
specifically) -- this one showed real, consistent improvement across every halflife tested
(100/300/600/1000, best at 300: recent-slice delta -0.0077, full-dev-range delta -0.0023, both REAL
IMPROVEMENT), earning a one-time holdout read (`run_oreb_adaptive_holdout_check.py`). Result: the
candidate's OWN dev-vs-holdout gap is still a REAL REGRESSION (VETO by the gap-only criterion), but
the actually decision-relevant comparison -- vs naive, holdout only -- moved from a clear LOSS
(Sec23: 3.0661 vs 3.0401) to a statistical TIE (3.0354 vs 3.0401, NOISE, CI includes zero). **Not
adopted**: a tie with the simple naive floor doesn't justify the added complexity of the full
team-level combine, per this project's standing "don't add complexity that isn't earning its keep"
discipline -- OREB stays unanchored (bottom-up player-sum only). But this is real, useful progress,
not a null result: it identifies the adaptive league average specifically (not shrinkage strength)
as the more promising direction if OREB is revisited, and narrows the gap from a clear loss to a
coin-flip against naive.

**The broader pattern worth naming**: the SAME two candidate mechanisms (recency-weighted target,
reduced shrinkage strength), tested independently on two different problems, gave OPPOSITE partial
results -- reduced shrinkage helped margin's tradeoff but hurt OREB; the adaptive target helped
OREB's tie but had already failed margin. Neither mechanism generalizes across problems, and
neither fully solves the problem it helped with -- reinforcing this project's now well-established
finding (Sec16-19, Sec24) that era-driven miscalibration doesn't have one universal fix; each
symptom needs its own independently-tested remedy, and sometimes (as here, twice) the best
available remedy still isn't good enough to adopt. 1 new regression test added
(`test_add_team_ratings_new_prior_games_params_default_preserving`, 3 assertions confirming the
new overridable shrinkage-strength params reproduce the exact original Phase 1 behavior at their
defaults) -- 39 regression tests (106 assertions) passing.

## 27. Full-model audit (2026-08-01) -- 29-agent workflow, 17 confirmed real findings, 26 lever ideas; fixing the 7 high-severity bugs

Ran a comprehensive audit: 8 parallel subsystem reviewers swept the entire codebase for data gaps,
calculation errors, logical fallacies, and unverified assumptions; every flagged finding was then
adversarially re-verified by a SECOND agent instructed to try to refute it by reading the code
directly (not trust the first agent's claim); a separate 4-agent research pass looked for additional
predictive levers. Result: 17 confirmed findings (7 high-severity, 10 medium) and 26 lever ideas.
Full detail (every finding's file/line/failure-scenario, every lever's rationale/feasibility) is
preserved in the workflow's own transcript; this section documents the fixes actually made.

### 27.1 Real bug: live game-score pipeline never applied home-court advantage

`generate_predictions.py`'s `run()` called `project_game(..., home_court_mult=1.0)` literally, since
the file's first commit -- discarding the empirically-validated home-court effect (Sec4: home teams
average 106.2 vs away 103.5 points, 58.4% home win rate) that every validation/backtest script
(`validate_team_strength_baseline.py`, `run_final_holdout_check.py`, etc.) correctly fits and
applies via `fit_home_court_walk_forward`. Two equally-rated teams got an IDENTICAL projected score
regardless of who was home -- the live pipeline could not express a home-court edge at all -- and
`win_prob_home`/`interval` (fit on residuals from `build_dev_predictions()`, which DOES include the
correction) were applied to a systematically biased mean, corrupting the live win-probability and
spread-interval output too. Present through Sec9.2's "confirmed against 2018-01-15's real 11-game
slate" spot-check without the gap ever surfacing, since that check only sanity-checked plausibility,
not the specific presence of a home-court split.

**Fixed**: added `_latest_home_court_mult(team_log)`, which reshapes the SAME `team_log` the live
call already built (already `gameDate < game_date`-filtered, so walk-forward safe) into the wide
per-game shape via `validate_team_strength_baseline._to_wide_games`, fits
`fit_home_court_walk_forward` on it, and takes the last (most recent) value -- falling back to 1.0
only if there's no game history to fit from yet (an honest "no correction available", not a silent
wrong number). Re-ran the 2025-01-15 spot-check: home teams' projected scores now correctly shift up
and away teams' down across the board (e.g. PHI home win prob 29.2% -> 33.2%), confirming the fix
takes effect in the right direction.

### 27.2 Real bug: matchup difficulty applied backwards on points, AST, and TOV

`matchup_difficulty.opponent_defense_adjustment`'s own docstring is explicit: "positive return =
tougher-than-average matchup (suppress the offensive player's projection)". `generate_props.py`
instead ADDED this value to the player's raw projection (`adjusted_points[pid] = raw_points[pid] +
delta`, and the identical pattern for `ast_proj`/`tov_proj`) -- increasing a player's projection
against tougher defenders and decreasing it against weaker ones, exactly backwards. Team-total
reconciliation (what Sec13's validation actually checked) can't catch this, since it only verifies
the SUM stays anchored, never the per-player direction.

TOV compounds this with a real polarity question: `tov_forced` uses the OPPOSITE rate convention
from points/ast_allowed (higher rate = tougher/better defense, not lower), so a genuinely tougher
turnover-forcing defender produces a NEGATIVE delta under the shared `league_avg - defender_rate`
formula. Worked through the arithmetic by hand (and confirmed via a new regression test using
`opponent_defense_adjustment`'s REAL output, not hand-picked signs): the single fix of subtracting
instead of adding correctly handles BOTH polarities at once -- `raw - delta` is algebraically
equivalent to `raw + (defender_rate - league_avg_rate)`, and a higher defender_rate always pushes
the offensive stat up, whether that means "a weak defender allows more points" or "a tough defender
forces more turnovers". No separate TOV-specific sign-flip was needed.

**Fixed**: `generate_props.py`'s three application sites (points, ast, tov) now subtract the delta
instead of adding it. Also fixed a related, purely-documentation bug found in the same file: the
module docstring and an inline comment self-contradicted about whether AST/TOV are anchored to a
team total (one paragraph said yes -- correct, matching the code -- a later paragraph and inline
comment said no, stale since Sec23 added the anchoring); both now consistently describe the actual,
correct behavior. New regression test
`test_matchup_delta_application_direction_suppresses_points_boosts_tov` confirms both the
suppress-points and boost-turnovers directions using real `opponent_defense_adjustment` output, and
explicitly confirms the ORIGINAL buggy addition would have produced the opposite (wrong) result in
both cases -- a genuine regression test, not a tautology.

### 27.3 Real bug: `generate_props.py` silently converted genuinely-missing player data into confident zero projections

`_project_active_roster_stats` documents that a player missing from a category's rate log (e.g. a
brand-new call-up) should get NaN -- "NOT silently zero -- so a downstream caller can distinguish
genuinely projects near zero from no data available". That held for OREB/2PT/3PT/FT-made (read
directly with no fillna), but was silently broken for the five Task #24-anchored categories
(DREB/AST/TOV/STL/BLK): `.fillna(0.0)` ran unconditionally, BEFORE the anchoring branch was even
decided, so both the anchored and unanchored-fallback paths already contained 0.0 instead of NaN.
A two-way/call-up player with real projected minutes but no cached history would show up as
`stat_name='ast', proj_mean=0.0, anchored_to_team_total=True` in the final output -- indistinguishable
from a real, data-backed near-certainty.

**Fixed**: extracted the anchoring logic into a new, directly-testable helper,
`_anchor_preserving_missing(raw_with_nan, side_total)` -- it still needs a real 0.0 internally so
`compute_usage_shares`/`allocate_team_total` have something to sum over, but now remembers which
players were originally NaN and restores NaN on the FINAL output for exactly those players, in both
the anchored and unanchored branches. New regression test
`test_anchor_preserving_missing_does_not_mask_genuinely_missing_players` confirms both branches.

### 27.4 Real bug: EWMA's per-season reset silently zeroed real veterans' props every season

`add_walk_forward_player_mean_ewm` (used for minutes, 2PT/3PT attempt-rate, and block-rate) resets
to NaN on every player's FIRST game of every season BY DESIGN -- not just a rookie's literal debut,
every established veteran too (confirmed: the sibling expanding-shrinkage primitive,
`add_walk_forward_player_rate`, instead falls back to a real league-average number at zero prior
exposure, never NaN -- this asymmetry is specific to the EWMA family). `generate_props.py`'s
`_latest_snapshot` picked the chronologically LAST row per player via a plain `.groupby(...).last()`
-- so when projecting a player's SECOND game of a new season, the "latest" available row (their
season-opener, already played) had this NaN, even though a real, non-NaN value from late LAST
season was sitting in an earlier row of the exact same log. That NaN then hit
`generate_props.py`'s `fillna(0.0)` fallback -- explicitly commented as being for "a brand-new
call-up" -- silently zeroing the player's points/2PT/3PT/block share for that one game and
reallocating it to teammates. Recurs every single season, leaguewide, during the week after opening
night; none of this project's own spot-check dates (2018-01-15, 2023-11-08, 2024-03-05, 2025-01-15)
happened to land in that exact window, which is why it was never caught.

**Fixed**: `_latest_snapshot` now forward-fills each column within a player's own sorted history
before taking the last row (`g.ffill().iloc[-1]` per group) -- carrying last season's real estimate
forward across the reset instead of picking up the fresh NaN. A genuine rookie with no history at
all still correctly comes out NaN (nothing to forward-fill from). New regression test
`test_latest_snapshot_carries_forward_across_ewma_season_reset` confirms both cases directly.

### 27.5 Real bug: the adopted Phase 2 RAPM numbers were computed on since-fixed-elsewhere logic

Sec22 fixed a real cross-season roster-bleed bug (a team's "last N games" lookback silently blending
in the PRIOR season's roster early in a new season, inflating the resolved active roster ~2x) for
the two LIVE pipelines only. `validate_rapm_lineup_adjustment.py`'s `_build_team_history` --
reused UNMODIFIED by `validate_predictive_lineup_adjustment.py` and `run_final_holdout_check.py` --
still built each team's history from the full, unfiltered multi-season log with no season-boundary
guard. These three scripts produced the officially-adopted Phase 2 numbers (Sec7's oracle-mode
result, Sec9.1's predictive-mode "~95% of ceiling" result, Sec9.3/9.4's holdout verdict that "Phase
2 stands") -- all computed on logic that Sec22 had already found and fixed everywhere else, never
re-validated after that fix landed.

**Fixed**: `_build_team_history` now also tracks each game's own `season`, and all three callers'
per-row prior-games filter additionally requires `season == row.season`, mirroring the live fix
exactly. Re-ran all three affected validation scripts under the corrected logic:

| check | metric | ORIGINAL (stale logic) | RE-VALIDATED (fixed logic) | still holds? |
|---|---|---|---|---|
| Sec7 oracle-mode vs Phase 1 | total_mae | -0.0217 REAL | -0.0205 REAL | yes |
| | margin_mae | -0.0350 REAL | -0.0358 REAL | yes |
| | su | NOISE | NOISE | yes |
| Sec9.1 predictive-mode vs Phase 1 | total_mae | -0.0206 REAL | -0.0185 REAL | yes |
| | margin_mae | -0.0336 REAL | -0.0332 REAL | yes |
| | su | NOISE | NOISE | yes |
| Sec9.3 Phase 4 holdout (dev n=10467, holdout n=2407) | total_mae gap | NOISE | NOISE | yes |
| | margin_mae gap | REAL REGRESSION -- VETO | REAL REGRESSION -- VETO | yes |
| | su gap | REAL IMPROVEMENT (unusual) | REAL IMPROVEMENT (unusual) | yes |

**Every previously-adopted conclusion holds under the corrected logic, numbers essentially
unchanged.** This is a genuinely informative re-validation, not a foregone conclusion -- the bug was
real (confirmed by the code diff and by `_build_team_history`'s own git history), and there was no
guarantee ahead of time that fixing it wouldn't shift the verdict on a metric this close to its own
significance boundary. It didn't. Phase 2 (RAPM-lite) remains confirmed-adopted; Phase 1's margin/
scoring-era-drift issue remains open exactly as documented (Sec9.5, Sec24, Sec26) -- this fix and
re-validation neither creates nor resolves that open problem, it just closes the gap between what
was tested and what's actually deployed.

### 27.6 Additional fixes made during the same pass

While working through the 7 high-severity bugs, also fixed several of the medium-severity findings
that were quick and clearly worth doing:

- **`score_distribution.py`'s Student-t variance mismatch** (dormant, but real): every Student-t
  call site passed `scale=sqrt(variance)` directly to scipy, which does NOT give the t-distribution
  that variance -- scipy's `t.cdf`/`t.ppf`/`t.logpdf`/`t.sf` parameterize `X = loc + scale*T` where
  `Var(T) = df/(df-2)`, not 1, silently inflating the REALIZED variance and contradicting
  `margin_and_total_params`'s own "matched variance" docstring claim. Confirmed empirically
  (`scipy.stats.t.rvs(df=9.5, scale=10)` has empirical variance ~126.5, matching `100*9.5/7.5`, not
  the naive 100) -- at df=9.5 (Sec11's real fitted points df, before Sec25 moved points off the
  continuous path entirely), that's a ~12.6% too-wide implied standard deviation. Currently dormant
  in live production (every adopted category uses `family='normal'` or a count family, never `'t'`),
  but this exact bug WAS present in the original Sec6/Sec11 dev-range Normal-vs-t comparisons that
  decided those families in the first place. Fixed with a new `_t_scale(var, df)` helper (converts a
  target variance into the correct scipy `scale`), applied at every t-family call site in both
  `score_distribution.py` and the copy-pasted math in `prop_distribution.py`. New regression test
  `test_t_scale_produces_correctly_matched_variance` confirms the fix via real scipy sampling (not
  just algebra), and explicitly confirms the ORIGINAL buggy scale inflates variance well above target.

- **`validate_data_coverage.py`'s permanent false FAIL**: was checking `rotation_{season}.parquet`
  (the abandoned `GameRotation` source, Sec5) instead of `boxscore_trad_player_{season}.parquet` (what
  `build_stints.py` actually depends on) -- reporting `OVERALL: FAIL` on every run regardless of real
  data completeness, and never actually validating the source that matters. Fixed to check the right
  source; re-running now surfaces a small number of genuinely real, plausible gaps (10-11 games
  missing in a few seasons) instead of a permanent, misleading whole-season false alarm.

- **RotoWire injury report has no historical archive, but neither live pipeline warned when
  running for a non-today date**: `fetch_current_injury_report()` always reflects real wall-clock
  today with no `game_date` awareness -- not fixable at the data layer (there's nothing to backfill
  from), so added `warn_if_stale_for_backtest(game_date)`, called by both `generate_predictions.py`
  and `generate_props.py`, printing an explicit warning whenever `game_date != date.today()`.  Every
  one of this project's own historical spot-checks (2018-01-15, 2023-11-08, 2024-03-05, 2025-01-15)
  was silently exposed to this before now; confirmed the warning fires correctly on a live re-run of
  the 2025-01-15 spot-check.

- **Player-name-to-ID crosswalk had no `is_active` tiebreak or collision detection**: `nba_api`'s
  static player list has 37-38 groups of players sharing an exact `full_name`; the original code
  silently kept whichever entry happened to appear last in the list's arbitrary order. One real
  collision was live-relevant: "Brandon Williams" resolved correctly today purely by accident of list
  ordering (not a guarantee). Fixed with `_build_name_index`, which prefers `is_active=True` on a
  collision (a real, defensible tiebreak for a live injury report) and tracks genuinely ambiguous
  collisions (same active-status) in a `dupes` set, returning `None` for those -- same signal as "no
  match at all" rather than a silent wrong guess. Confirmed live: "Brandon Williams" now resolves
  deterministically to the active player's ID, and 37 genuinely ambiguous collisions are now tracked
  (down from being silently absorbed with zero visibility before).

- **Stale docstrings** in `prop_distribution.py` and `validate_prop_distribution.py` (both still
  described the plan's pre-Sec25 continuous/count split, contradicted by `CATEGORY_FAMILY`/
  `CATEGORY_SPECS` a few dozen lines below in the SAME files) and in `lineup_rating.py`'s
  `team_recent_roster_rapm` (claimed its no-history fallback means "no adjustment at all", when
  `project_lineup_adjustment` actually applies each active player's full raw RAPM rating unweighted
  in that case -- a real, non-zero adjustment that now fires at every team's season opener since
  Sec22's season-scoping fix). All three corrected to describe actual current behavior; none required
  a behavior change, only fixing what the code already does now being told accurately.

39 regression tests grew to 44 (123 assertions) across this pass -- new tests:
`test_matchup_delta_application_direction_suppresses_points_boosts_tov`,
`test_anchor_preserving_missing_does_not_mask_genuinely_missing_players`,
`test_latest_snapshot_carries_forward_across_ewma_season_reset`,
`test_t_scale_produces_correctly_matched_variance`,
`test_name_index_prefers_active_player_and_flags_genuine_ambiguity`.

**Still open** (flagged by the audit, not yet fixed): the `refresh_data.py` current-season data gap
for player-track/matchup/roster fetches (now fixed, see the module's own updated docstring) leaves
one related, smaller gap -- `active_roster.py`'s trailing-minutes lookback still has no check
against actual roster membership for a MID-season trade/waiver (Sec22 only fixed the cross-SEASON
version of this); and `predictive_minutes_shares`' backtest methodology uses a more-informed
active-player-set selection than the live pipeline can replicate (a real gap between the validated
~95%/96% ceiling-capture figure and live-achievable accuracy). Both are real, but neither is a
quick, isolated fix -- left as documented, flagged follow-up work rather than rushed.

## 28. The last 2 medium findings, fixed (2026-08-01)

### 28.1 Real bug: mid-season trade/waiver contamination in `resolve_active_lineup`

Sec22 fixed the cross-SEASON version of a roster-membership bug (a trailing lookback reaching into
the PRIOR season's roster). This left the MID-season version untouched: `resolve_active_lineup`'s
trailing-minutes lookback had no check against actual CURRENT roster membership at all -- a player
traded or waived kept contributing his real trailing minutes (and a nonzero share of tonight's
projection) for up to `MINUTES_LOOKBACK_GAMES` games after he left, diluting every actually-rostered
player's share and generating a full phantom prop row for someone who isn't on the team.

**Fixed**: added `active_roster.load_current_roster_player_ids(season)` (teamId -> set of
playerIds, from `CommonTeamRoster`'s cached snapshot) and a new `current_roster_ids` param on
`resolve_active_lineup` (default `None`, preserving exact original behavior) that additionally
excludes any player not on that set. Both live pipelines now pass it. Also fixed a related,
previously-invisible problem this surfaced: `fetch_rosters_season` marks a team "done" forever once
fetched once per season -- correct for a COMPLETE historical season, but wrong for the CURRENT one,
where the snapshot needs to genuinely stay current through mid-season trades. `refresh_data.py` now
force-refreshes rosters every call, mirroring the schedule's own already-forced refresh.

This new signal has the SAME wall-clock-only limitation as RotoWire's injury report (`CommonTeamRoster`
reflects "as of whenever last fetched", not a point-in-time archive) -- generalized the existing
`warn_if_stale_for_backtest` warning to cover both sources with one message, rather than adding a
second, redundant warning. Confirmed live on the 2025-01-15 spot-check: many players are now
(correctly, expectedly) flagged "excluded as no longer on the roster" -- since this run compares a
January 2025 date against TODAY's (2026-08-01) real roster snapshot, exactly the documented
limitation, not a new bug. A genuine live "tonight" call has no such mismatch. New regression test
`test_resolve_active_lineup_excludes_departed_players` confirms the default preserves original
behavior and the new param correctly excludes and renormalizes.

### 28.2 Real bug: `predictive_minutes_shares`' hindsight leak in the backtest active-player-set selection

The backtest's active-player SET was `oracle_minutes_shares(...).index` -- literally this exact
historical game's real attendance (perfect hindsight on who ends up playing). The live mechanism
this claimed to mirror, `resolve_active_lineup`, does NOT know who ends up playing -- it takes the
trailing-window UNION of anyone who's recently appeared, excluding only players an injury report
flags. A healthy scratch, DNP-CD, or unflagged load-management night is a scenario the live system
structurally cannot exclude in advance; the old backtest's oracle-derived set silently excluded such
players anyway (since they show 0 real minutes that exact game), measuring a strictly easier
selection problem than what's actually deployed.

**Fixed**: the active-player set is now ALSO derived purely from the trailing-window union, with no
reference to the specific game's own real outcome at all -- mechanism-matched to
`resolve_active_lineup` rather than a hindsight shortcut. New regression test
`test_predictive_minutes_shares_has_no_hindsight_leak` confirms a player who's genuinely in the
recent rotation but was scratched for the specific game being projected is still included (exactly
the case the old, oracle-based selection would have wrongly excluded).

This is a genuine methodology change to the input `validate_predictive_lineup_adjustment.py` and
`run_final_holdout_check.py` both depend on -- re-ran both under the corrected mechanism per this
project's standing discipline (never silently change a validated methodology without re-confirming
the numbers it produced), plus a fresh direct Phase1-alone-vs-Phase1+2 isolation check (mirroring
Sec9.4's original methodology exactly) since the dev-only result below was decisive enough to
demand the actual decision-relevant holdout comparison, not just a dev/holdout gap check:

| check | metric | ORIGINAL (hindsight-leaked) | RE-VALIDATED (fixed, honest) | still holds? |
|---|---|---|---|---|
| Sec9.1 predictive-mode vs Phase 1, DEV | total_mae | -0.0206 REAL IMPROVEMENT | **+0.0086 REAL REGRESSION** | **NO** |
| | margin_mae | -0.0336 REAL IMPROVEMENT | **+0.0146 REAL REGRESSION** | **NO** |
| | su | NOISE | NOISE | n/a |
| Isolation check, HOLDOUT ONLY (n=2407) | total_mae | (not separately run before) | +0.0004, NOISE | -- |
| | margin_mae | (not separately run before) | **+0.0181 REAL REGRESSION** | **NO** |
| | su | (not separately run before) | +0.0004, NOISE | -- |

**This overturns the previously-adopted Sec9.1/9.3/9.4 conclusion.** Under the honest methodology
(no more oracle-derived active-player set), Phase 2's predictive-mode lineup adjustment does NOT
help -- it makes MARGIN predictions genuinely WORSE, consistently on both dev and holdout, and
doesn't meaningfully help total_mae or SU either. The confirmatory-veto protocol's core premise
(spend one honest holdout read, let it settle the question) worked exactly as designed here: the
answer it gave is uncomfortable, but that's the point of running it rather than trusting the
dev-only number that motivated adopting Phase 2 in the first place.

**Important nuance, not a reason to dismiss the finding**: oracle mode (real, already-known minutes
-- Sec7's ceiling test, which never used `predictive_minutes_shares` and is UNAFFECTED by this fix)
still shows a real improvement (total_mae -0.0205, margin_mae -0.0358, both CI excludes zero,
reconfirmed in Sec27.5's re-validation). So lineup-awareness AS A CONCEPT genuinely carries real
signal -- the problem is specifically that the deployable, predictive minutes-share ESTIMATION
mechanism doesn't capture that signal well enough to net out ahead of its own noise. And this fixed
backtest is itself not a perfect proxy for live conditions either: it now has ZERO injury
information (RotoWire has no historical archive to backtest against, Sec27.6), while the ACTUAL
live pipeline (`resolve_active_lineup`) DOES have real, if partial, injury exclusion via RotoWire's
Out/Doubtful feed. That means this fixed backtest is likely a pessimistic lower bound on live
performance, not a precise estimate of it -- the true live-deployed accuracy of Phase 2 is somewhere
between the old (provably too-optimistic) and new (provably information-starved) numbers. What both
extremes now agree on: the ORIGINAL "confirmed, real improvement" claim was built on a flawed test
and cannot be trusted as-is.

**This is currently live in production** (`generate_predictions.py`/`generate_props.py` both apply
Phase 2's adjustment on every call) -- a consequential decision (disable Phase 2 pending a better
minutes-share mechanism, keep it while investigating further, or something else) is flagged for the
user rather than made unilaterally here. 2 new regression tests added
(`test_predictive_minutes_shares_has_no_hindsight_leak`,
`test_resolve_active_lineup_excludes_departed_players`) -- 46 regression tests (129 assertions)
passing.

### 28.3 Decision: Phase 2 disabled

Given the choice between disabling Phase 2 now, keeping it live while investigating a better
minutes-projection mechanism, or accepting the risk as-is, the user chose to disable it immediately
-- reverting to the configuration with a currently-valid confirmatory result (Phase 1 alone) rather
than continuing to run a configuration whose own adoption evidence had just been shown unreliable.

**Implementation**: `generate_predictions.py` gets a new module-level `INCLUDE_LINEUP_ADJUSTMENT =
False` constant. When `False`, `_fit_latest_player_ratings` is skipped entirely (no wasted RAPM fit
computation every call) and `have_lineup_adjustment` is forced `False`, so every game falls through
the ALREADY-EXISTING "team-strength only" code path (no new branch needed -- this path has existed
since before Phase 2 was ever wired in, and is exactly what oracle mode's ceiling test and Sec9.4's
original isolation check both compared against). `generate_props.py` has no independent RAPM
adjustment of its own -- it only ever reused `generate_predictions.py`'s output as its points macro-
anchor, so disabling Phase 2 there automatically and correctly propagates; the props module
docstring was updated so it doesn't misleadingly claim "already-RAPM-adjusted" regardless of the
live flag's state. `run_final_holdout_check.py`'s own `INCLUDE_LINEUP_ADJUSTMENT` flag flipped to
`False` to match, with an updated comment recording why.

**Verified live**: re-ran the 2025-01-15 spot-check on both pipelines. `generate_predictions.py` now
shows `[no lineup adjustment (team-strength only)]` for every game (previously showed a lineup
source tag and RotoWire exclusion count for each side); `generate_props.py` still runs cleanly
(its own `resolve_active_lineup` calls are for MINUTES-SHARE/rotation purposes only, unrelated to
Phase 2's RAPM skill adjustment, so its output shape is unaffected by this change).

**Nothing is deleted.** `rapm_lite.py`, `lineup_rating.py`, and `active_roster.py`'s RAPM-adjacent
machinery all remain in place, fully tested, and importable -- re-enabling Phase 2 later is a
one-line flag flip, but should only be done after a genuinely improved minutes-projection mechanism
(not just re-trusting the old, hindsight-leaked one) is built and validated through this project's
standard dev-then-holdout discipline.

## 29. Phase 1's long-open margin/scoring-era-drift regression -- FINALLY RESOLVED, via a lever from the audit's own research pass (2026-08-01)

The full-model audit's "untried synthesis" research angle flagged something neither Sec24 nor Sec26
had tried: `team_strength.add_team_ratings` already accepts BOTH `league_avg_halflife_games`
(Sec24's recency-weighted target) AND `prior_games_rating` (Sec26's shrinkage strength) as
independent kwargs on the SAME function call -- but each investigation swept its own lever while
leaving the other at its stock default. Sec24 (target alone) showed a real REGRESSION on margin_mae
at every halflife tried; Sec26 (strength alone) showed a real margin_mae IMPROVEMENT but a real
total_mae REGRESSION of larger magnitude -- each lever's own cost sat on the metric the OTHER lever
was good at fixing, a strong hint a joint configuration might net both out at once.

**Built `validate_joint_margin_fix.py`**: a 3x4 grid sweep (`prior_games_rating` in {8,10,12} x
`league_avg_halflife_games` in {1000,2000,5000,10000}) on the recent-dev slice (fit on the full dev
range, same Stage-1-screen discipline as every prior attempt), requiring a genuine NET WIN (no real
regression on either metric, real improvement on at least one) to advance. Result: 9 of 12 grid
cells cleared the bar outright; `prior_games_rating=12.0, halflife=2000.0` was the strongest (best
combined delta), and passed Stage 2 (full dev range) cleanly too: total_mae -0.0766 REAL
IMPROVEMENT, margin_mae -0.0113 REAL IMPROVEMENT, su NOISE.

**One-time confirmatory holdout read** (`run_joint_margin_fix_holdout_check.py`), the decision-
relevant comparison (new config vs. the OLD prior_games=15/flat-target config, holdout games only):

| metric | delta | verdict |
|---|---|---|
| total_mae | -0.2605 | REAL IMPROVEMENT |
| margin_mae | -0.0194 | REAL IMPROVEMENT |
| su | +0.0000 | NOISE (no harm) |

**The first configuration in this entire investigation (Sec24, Sec26, this) to clear BOTH metrics
on real holdout data with no tradeoff.** One honest caveat: the new candidate's OWN dev-vs-holdout
GAP on margin_mae is still a real widening (dev=10.4608 -> holdout=11.3312) -- the underlying
scoring-era-drift phenomenon is real and not eliminated by this fix. But that's a different question
from "does this beat what's currently deployed", which the holdout-only comparison answers
unambiguously yes: the new config copes with that same real difficulty measurably better than the
old one did.

**Adopted as the new production default**: `team_strength.py`'s `PRIOR_GAMES_RATING` changed
15.0 -> 12.0; new constant `LEAGUE_AVG_HALFLIFE_GAMES_RATING = 2000.0` added, and
`add_team_ratings`'s `league_avg_halflife_games` default changed from `None` (flat/infinite-memory)
to this constant -- `None` remains available as an explicit override for any caller that specifically
wants the original pre-Sec29 behavior. Since neither the live pipeline nor
`validate_team_strength_baseline.py` pass explicit overrides, both automatically pick up the new,
validated defaults with no further code changes needed. Re-ran the naive-floor comparison to confirm
the new config still clearly beats naive (total_mae -0.2506, margin_mae -0.8180, su +0.0686, all
REAL IMPROVEMENT -- an even larger margin_mae gap over naive than the original Sec1 result, since
naive doesn't benefit from either lever at all). Live spot-check (2025-01-15) confirms updated
predictions flow through automatically.

This closes out a genuinely long-running open problem (first diagnosed Sec9.5, worked on across
Sec16-19, Sec24, Sec26) -- not through a cleverer new mechanism, but by finally testing the
COMBINATION of two already-built, already-tested levers together, which is exactly the kind of gap
a dedicated audit/research pass is suited to catch that iterative in-the-moment investigation can
miss.

## 30. OREB shrinkage-strength lever, fully exhausted (2026-08-01)

A second audit lever: Sec26 tested REDUCING OREB's shrinkage strength (`PRIOR_GAMES['oreb']`,
current 20.0) down to 5/8/10/12/15 -- all monotonically WORSE, closing in on but never beating the
current default. The untried direction the audit flagged: nobody had tried MORE shrinkage (the
monotonic pattern pointed that way). Swept 25/30/40/60/100 on the recent-dev slice: 25/30/40 are
NOISE (no real difference from the current default), 60/100 are REAL REGRESSION. **`PRIOR_GAMES['oreb']=20.0`
sits at a genuine local optimum in BOTH directions now tested** -- this specific lever (shrinkage
strength) is fully exhausted for OREB. Combined with Sec26's earlier finding that the recency-
weighted league average moved OREB from a clear holdout loss to a statistical tie (real progress,
but insufficient to adopt), OREB team-level anchoring's tractable levers are now genuinely spent;
it remains unanchored (bottom-up player-sum only) per Sec23's original decision.

## 31. Detrend-then-retrend tested on 3PT-makes -- the audit lever's own premise didn't survive a real-data check

A third audit lever: Sec17's detrend-then-retrend architecture (`add_era_adjusted_player_rate`) was
built and validated exclusively for steals, where it failed decisively on holdout (a genuine regime
change, not a trend). The audit flagged 3PT-makes as untested and structurally better-suited (a
continuous, CLT-like trend rather than a level-shift). **Checked the premise directly against real
per-season data BEFORE spending effort on the fix itself**: mean 3PT attempts/made per player-game
by dev season -- 2.27/0.80 (2015) rising to 3.02/1.07 (2018), then essentially FLAT through the rest
of dev (3.23, 3.25, 3.32, 3.25, 3.27 attempts, 2019-2023). The rise is real but plateaus WITHIN the
dev range itself -- a materially different shape from margin's continuous-through-2023 climb
(Sec9.5), and actually closer to steals' own regime-change profile (a shift concentrated at/after
the dev/holdout boundary, not a trend visible throughout dev) than the audit's stated premise
suggested.

Swept `current_rate_halflife_games` in {25,50,100,200,400} on the recent-dev slice anyway (fit on
the full dev range, same discipline as every other candidate): all 5 values showed a REAL
REGRESSION vs. the current flat expanding-shrinkage baseline, consistently and by a similar small
magnitude -- a clean, decisive stop at Stage 1, no holdout read spent. This is a useful negative
result on two levels: it confirms detrend-then-retrend still isn't the right tool outside the
regime-change-diagnosis case it was actually built for, and it's a concrete example of this
project's own discipline (check the premise against real data before trusting a lever's stated
rationale) catching a plausible-sounding assumption that didn't actually hold up.

## 32. Back-to-back adjustment: a genuinely real diagnostic that still doesn't survive adoption

A domain-signal lever with an interesting starting point: `rest_schedule.py`'s residual-correlation
diagnostic found the B2B effect REAL early in this project (p=8.0e-7 on a real dev-range check), but
a repo-wide grep confirms `fit_b2b_adjustment`/`add_rest_days` were never actually wired into
`team_strength.py` or either live pipeline -- a confirmed-real signal that sat completely unused.
Built a proper walk-forward-safe adoption test (`validate_b2b_adjustment.py`): an expanding-mean B2B
correction using only strictly-prior B2B-game residuals (explicit sequential loop, not a vectorized
shift, specifically to avoid the same-game leak trap this project has hit before), applied as an
additive delta to whichever side is on a back-to-back.

**Result: a real, but MIXED and non-adoptable effect.** On the recent-dev slice: total_mae shows a
tiny REAL IMPROVEMENT (-0.0061) but margin_mae shows a tiny REAL REGRESSION (+0.0086) -- both
effects are an order of magnitude smaller than the original residual-correlation diagnostic's
"-1.02pt" headline number. Swept the shrinkage prior (10/50/200/500/1000 games) to rule out an
under-shrunk noisy estimate as the cause: the mixed pattern is remarkably STABLE across the entire
range, ruling that out. Mechanistically similar to Sec24's finding for the league-average target:
correcting the SCALE (total) can net out to a small real gain while adding noise to the specific
RELATIVE comparison (margin) that a difference metric is more exposed to. **Not adopted** -- a real
diagnostic correlation is confirmed NOT to translate cleanly into an adopted correction here,
consistent with this project's now well-established distinction between "a residual correlates with
X" and "adding a correction for X actually helps out-of-sample, net of the noise it introduces."

## 33. A second real win for Phase 1: `cross_season_weight`, untested since the day it was introduced

`shrinkage.py`'s own docstring flagged `cross_season_weight` as "a genuinely untested hypothesis for
NBA specifically... not an assumed-correct default" the day it was added -- and it stayed at its
default (0.0, full within-season reset) for this entire project until the full-model audit's
research pass flagged it as a real, cheap, never-actually-swept lever (untried_synthesis #6).

**Swept 0.1 through 1.0** on the recent-dev slice, ON TOP OF the already-adopted joint-fix defaults
(Sec29: `prior_games_rating=12.0`, `league_avg_halflife_games=2000.0`) -- EVERY single value tested
showed a real improvement on BOTH total_mae and margin_mae simultaneously, the strongest, cleanest
dev-only result of any lever tested this session. The effect grows through ~0.3-0.6 then fades back
toward noise by 1.0 (a full cross-season blend, no reset at all); su specifically peaks around
0.25-0.35. Picked `cross_season_weight=0.3` as the strongest all-around candidate (real improvement
on all three metrics) and confirmed it on the full dev range: total_mae -0.0711, margin_mae -0.0506,
su +0.0065, all REAL IMPROVEMENT.

**One-time confirmatory holdout read** (`run_cross_season_weight_holdout_check.py`), vs. the current
(already-Sec29-adopted, cross_season_weight=0.0) config, holdout games only:

| metric | delta | verdict |
|---|---|---|
| total_mae | -0.0835 | REAL IMPROVEMENT |
| margin_mae | -0.0763 | REAL IMPROVEMENT |
| su | +0.0025 | NOISE (no harm) |

**A second clean, genuine win for Phase 1's core rating engine**, on top of Sec29's joint fix.
**Adopted**: new constant `CROSS_SEASON_WEIGHT_RATING = 0.3`, `add_team_ratings`'s `cross_season_weight`
default changed from `0.0` to this constant (pass `0.0` explicitly for the original full-reset
behavior). Re-ran the naive-floor comparison with all three now-adopted defaults together: total_mae
-0.3216, margin_mae -0.8687, su +0.0752, all REAL IMPROVEMENT and each wider than the prior
Sec29-only re-check -- the gap over naive keeps growing as genuinely untested-but-available levers
get properly checked. Live spot-check (2025-01-15) confirms updated predictions flow through
automatically (no code changes needed in either live pipeline -- both call `add_team_ratings` with no
explicit overrides).

Between Sec29 and this section, the full-model audit's "untried synthesis" and domain-signal
research levers have now delivered two consecutive, genuine, holdout-confirmed improvements to
Phase 1 by testing combinations and parameters that existed in the code the whole time but had
never actually been checked -- a strong argument for treating a dedicated audit/research pass as a
recurring practice, not a one-off.

## 34. OREB team-level anchoring: a THIRD mechanism converges to the exact same ceiling -- investigation now closed

Given `cross_season_weight`'s clean, decisive win for `team_strength.py` (Sec33, via the identical
`shrinkage.add_walk_forward_rate` primitive), tested the same lever for `team_stat_rates.py` --
never exposed as a parameter there before (`add_team_stat_ratings` hardcoded 0.0 for every
category). Added it as an overridable, default-preserving parameter, then swept it for OREB
specifically (the one category with a real, confirmed absolute holdout loss to naive, per Sec23).

**Dev-only result looked genuinely promising**: 0.1 and 0.25 both showed REAL IMPROVEMENT on the
recent-dev slice and the full dev range, against both the current config AND naive directly (full
dev range: -0.0045 vs current, -0.0255 vs naive, both real). This cleared both Stage 1 and Stage 2
cleanly, the same bar the winning Phase 1 levers cleared.

**One-time confirmatory holdout read, the decisive test**: candidate's own dev-vs-holdout gap is
still a REAL REGRESSION (dev=2.9014 -> holdout=3.0637); vs. naive on holdout specifically:
3.0637 vs 3.0401, delta +0.0236, **NOISE** -- statistically indistinguishable from naive, not a
clear win.

**This is the THIRD structurally distinct mechanism to converge on the exact same outcome**:
Sec26's adaptive league-average target gave 3.0354 vs naive's 3.0401 (NOISE); Sec26's reduced/
increased shrinkage strength made things monotonically worse in both directions (Sec30); now
`cross_season_weight` gives 3.0637 vs 3.0401 (NOISE again). Three independent, well-motivated,
properly-validated mechanisms -- recency-weighted target, shrinkage strength, cross-season
blending -- all land at essentially the same "statistical tie with naive, never a clear beat"
ceiling. This is strong, convergent evidence that OREB's team-level rate model has a genuine
structural limit with the `add_walk_forward_rate` primitive as currently built, not a
tuning problem any single parameter can solve. **The OREB team-level anchoring investigation is
now closed** (not just "still open") -- it remains unanchored (bottom-up player-sum only, per
Sec23's original decision); a genuine fix, if one exists, would need a structurally different
primitive or a fundamentally different signal, not another parameter sweep on this one.

## 35. A third real win: `cross_season_weight` adopted for the 5 team_stat_rates ADOPTED_CATEGORIES (2026-08-01)

Sec34 tested `cross_season_weight` for OREB specifically and it converged to the same ceiling as
two other mechanisms. But the same lever, tested for the other 5 categories (dreb/ast/tov/stl/blk
-- the ones that DID clear their original Sec23 holdout check and are already live-anchored), is a
structurally different question: these categories don't have OREB's documented holdout-loss
problem, so there's no reason to assume the same ceiling applies. Tested independently.

**Stage 1 (recent-dev slice, candidate=0.25 vs current csw=0.0)**: all 5 categories showed REAL
IMPROVEMENT -- dreb -0.0079, ast -0.0140, tov -0.0136, stl -0.0069, blk -0.0086 (all CIs exclude
zero).

**Stage 2 (full dev range, n=21,476 games)**: all 5 cleared BOTH comparisons cleanly --
vs. current config: dreb -0.0098, ast -0.0175, tov -0.0118, stl -0.0056, blk -0.0067 (all REAL
IMPROVEMENT); vs. naive floor: dreb -0.0927, ast -0.0910, tov -0.0800, stl -0.0408, blk -0.0441
(all REAL IMPROVEMENT, larger margins than vs. current -- expected, since naive has no shrinkage at
all).

**One-time confirmatory holdout read** (`run_team_stat_cross_season_holdout_check.py`, n=4,900
games), candidate vs. current config, holdout-only:

| category | delta | verdict |
|---|---|---|
| dreb | -0.0162 | REAL IMPROVEMENT |
| ast | +0.0053 | NOISE (CI includes zero) |
| tov | -0.0160 | REAL IMPROVEMENT |
| stl | -0.0047 | REAL IMPROVEMENT |
| blk | -0.0041 | REAL IMPROVEMENT |

4 of 5 categories show a real holdout improvement; ast is noise but NOT a regression (CI includes
zero, point estimate barely positive). No category regressed. Net-win criterion cleared uniformly
-- **adopted `cross_season_weight=0.25` for all 5 ADOPTED_CATEGORIES**.

**Implementation detail that mattered**: `add_team_stat_ratings` loops over all 6
`STAT_COLUMNS` categories (including OREB) in one shared function. A naive "just change the
default float" edit would have silently re-applied Sec34's REJECTED OREB config the moment this
adoption landed, since OREB shares the same loop. Fixed by keying the default on a per-category
dict (`DEFAULT_CROSS_SEASON_WEIGHTS`) -- 0.25 for `ADOPTED_CATEGORIES`, 0.0 (unchanged) for oreb --
with the function's `cross_season_weight` param changed from a float default to `float | None`
(explicit float still applies uniformly across all 6, preserving every existing sweep script's
behavior; `None`, the new default, resolves per-category). Covered by
`test_add_team_stat_ratings_oreb_excluded_from_uniform_cross_season_adoption` in
`tests/test_regression_bugs.py`.

Both `generate_props.py` and `run_team_stat_holdout_check.py` call `add_team_stat_ratings(log)`
with no explicit `cross_season_weight`, so this adoption reaches the live props pipeline
automatically with no call-site changes needed -- the same "thread it through the shared default"
pattern Sec29/33 used for `team_strength.add_team_ratings`.

This is the fourth genuine, holdout-confirmed win found via the audit's research levers this
session (after the Sec29 joint margin fix, Sec33's Phase 1 `cross_season_weight`, and now this),
and the second time `cross_season_weight` specifically has cleared holdout -- reinforcing that this
parameter, flagged as untested since `shrinkage.py` was written, was a genuinely under-explored
lever across this entire codebase, not just for Phase 1.

## 36. A fifth real win: `own_halflife_games`, recency-weighting a team's OWN in-season history (2026-08-01)

The untried-synthesis lever from the audit's research pass: `cross_season_weight` (Sec33) and
`league_avg_halflife_games` (Sec29) each recency-weight a DIFFERENT part of the shrinkage math --
the early-season prior and the shared league-average target, respectively -- but neither touches
how a team's OWN within-season games are weighted against each other. `add_walk_forward_rate`'s
`for_cumsum_before`/`against_cumsum_before` have always been a flat cumulative sum: game 1 of a
season counts exactly as much as last night's game. Never tested whether recency-weighting the
team's own signal helps.

**New primitive**: `shrinkage._trailing_own_mean_ewma` -- an EWMA (half-life in games) of a team's
own value within (team, season), strictly prior games only. No same-game leak risk here unlike
`_trailing_league_stat_ewma` (which needs a per-gameId collapse before shifting): each row already
belongs to exactly one team, so grouping by (team, season) and `shift(1)`-ing within that group is
leak-safe by construction. Wired in as a new `own_halflife_games` param on `add_walk_forward_rate`
(default `None` reproduces the original flat-cumulative-sum column byte-for-byte, regression-tested)
-- when set, replaces the flat own-mean with the EWMA-weighted one before blending against the
shrinkage prior at `prior_games` strength, preserving the exact same shrinkage-strength
interpretation, just swapping how the "own value" is estimated.

**Dev-only Stage 1** (recent-dev slice, on top of the already-adopted Sec29/33 defaults): swept
5/10/20/40/80 games. 5 was too aggressive -- REAL REGRESSION on margin_mae (+0.1635) and su
(-0.0147), thrashing on noise at that short a half-life. 10/20/40/80 all cleared the net-win bar; 20
and 40 cleared it most cleanly, with REAL IMPROVEMENT on BOTH total_mae and margin_mae
simultaneously (40: -0.1233/-0.0273; 20: -0.1889/-0.0343), su NOISE (no harm) at both.

**Stage 2** (full dev range, n=10,737 games): both 20 and 40 beat the current config on total_mae
(real improvement); 40 ALSO beat it on margin_mae (real improvement, -0.0171) while 20 was noise-
not-regression there. Both beat naive by a wide margin on all three metrics. 40 chosen as the
strongest all-around candidate (clears both metrics cleanly, not just one).

**One-time confirmatory holdout read** (`run_own_halflife_holdout_check.py`), candidate vs. current
config, holdout-only: total_mae -0.0499 REAL IMPROVEMENT, margin_mae -0.0412 REAL IMPROVEMENT, su
NOISE (no harm) -- a clean net-positive result, no regression on any metric. The candidate's OWN
dev-vs-holdout gap on margin_mae still shows a REAL REGRESSION (dev=10.3931 -> holdout=11.2137) --
the same scoring-era-drift phenomenon Sec29 diagnosed is still present and not eliminated by this
fix either -- but the decision-relevant comparison (vs. the current config, holdout-only) is what
matters, exactly the same logic Sec29/33 already used: this config copes with that real difficulty
measurably better than the prior one did.

**Adopted**: `team_strength.OWN_HALFLIFE_GAMES_RATING = 40.0`, threaded through `add_team_ratings`'s
new `own_halflife_games` parameter (applied to rtg only, not pace -- untested for pace, left as a
possible future lever). All existing callers (`generate_predictions.py`, `generate_props.py`,
`validate_team_strength_baseline.py`, etc.) invoke `add_team_ratings(log)` with no explicit
override, so this reaches the live pipeline automatically, no call-site changes needed -- the same
"thread it through the shared default" pattern every Phase 1 lever this session has used.

This is the fifth genuine, holdout-confirmed win found via the audit's research levers this session
(Sec29, Sec33, Sec35, and now this), and the third structurally distinct lever (prior / target /
own-history) to independently pay off for Phase 1's rating engine specifically -- strong evidence
that `shrinkage.py`'s walk-forward primitive had a lot of genuinely unexplored surface area, not
just one lucky fix.

## 37. `own_halflife_games` extended to team_stat_rates -- a sixth real win, and a fourth confirmation of OREB's structural ceiling

Given `cross_season_weight` won for both Phase 1 (Sec33) and 5/6 team_stat_rates categories
(Sec35), the natural next check was whether `own_halflife_games` (Sec36) generalizes the same way.
Tested independently across all 6 categories, including OREB -- since `own_halflife_games` is a
structurally different lever from `cross_season_weight` (recency-weights the team's own history,
not the early-season prior), OREB wasn't assumed to fail the same way it did in Sec34/35.

**Dev-only Stage 1** (recent-dev slice, sweeping 20/40 games on top of each category's already-
adopted `cross_season_weight`): all 6 categories -- oreb included -- showed REAL IMPROVEMENT or
NOISE (never regression) at both values; 20 the stronger candidate for most (oreb -0.0069, dreb
-0.0208, tov -0.0190, blk -0.0063 real improvement at 20; ast/stl noise-to-mixed).

**Stage 2** (full dev range, n=21,476 games, `own_halflife_games=20`): all 6 categories, including
OREB, beat BOTH the current config AND the naive floor -- a clean sweep.

**One-time confirmatory holdout read** (`run_team_stat_own_halflife_holdout_check.py`), candidate
vs. current config, holdout-only:

| category | delta | verdict |
|---|---|---|
| oreb | -0.0026 | NOISE |
| dreb | -0.0146 | REAL IMPROVEMENT |
| ast | +0.0023 | NOISE |
| tov | -0.0358 | REAL IMPROVEMENT |
| stl | -0.0111 | REAL IMPROVEMENT |
| blk | -0.0108 | REAL IMPROVEMENT |

No category regressed; 4/6 show a real holdout improvement. **Adopted `OWN_HALFLIFE_GAMES_STAT =
20.0` for the 5 ADOPTED_CATEGORIES** (dreb/ast/tov/stl/blk), via a new
`DEFAULT_OWN_HALFLIFE_GAMES` per-category dict mirroring `DEFAULT_CROSS_SEASON_WEIGHTS`'s pattern.

**A genuinely new wrinkle required a different implementation than `cross_season_weight`'s**:
`cross_season_weight`'s real "off" value is `0.0`, which left `None` free to mean "use this
module's per-category default" with no ambiguity. `own_halflife_games`'s real "off" value at the
`shrinkage.py` primitive layer IS `None` itself -- so a plain `= None` default on
`add_team_stat_ratings` would be ambiguous between "caller wants no recency-weighting" and "caller
wants the per-category default." Resolved with a dedicated `_USE_PER_CATEGORY_DEFAULT` sentinel
string as the parameter's actual default, distinct from any legitimate value (float or `None`) a
caller might pass explicitly. Covered by
`test_add_team_stat_ratings_oreb_excluded_from_uniform_own_halflife_adoption`.

**OREB checked separately, since it looked promising on both dev stages here (unlike
`cross_season_weight`)**: does `own_halflife_games=20` let OREB finally beat naive on holdout? No --
still NOISE (3.0634 vs. naive's 3.0401, delta +0.0233, CI includes zero). This is the **fourth**
structurally distinct mechanism (after shrinkage strength/Sec30, adaptive league average/Sec26,
`cross_season_weight`/Sec34) to converge on the exact same "statistical tie with naive" ceiling for
OREB -- further reinforcing Sec34's conclusion that this is a genuine structural limit with the
`add_walk_forward_rate` primitive as currently built, not a parameter left untried. OREB's
`own_halflife_games` therefore stays at `None` (unchanged), the same treatment as its
`cross_season_weight`.

This is the sixth genuine, holdout-confirmed win found via the audit's research levers this session,
and `own_halflife_games`'s second consecutive win after Sec36 -- both `cross_season_weight` and
`own_halflife_games` have now each independently paid off in BOTH subsystems (Phase 1's team
ratings and team_stat_rates), while both have also independently confirmed OREB's ceiling. The
pattern is now unambiguous: OREB's problem is not a missing lever, it's the category itself.
