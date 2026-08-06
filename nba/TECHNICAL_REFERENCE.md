# NBA Prediction Model — Complete Technical Reference

**Audience note**: this document is written for a reader (human or model) with no prior context on
this codebase, who needs to (a) fully understand every mathematical and probabilistic assumption
this system makes, (b) be able to critique it with technical precision, and (c) help decide what to
build next. It is organized by SYSTEM ARCHITECTURE, not chronologically — for the chronological
research log (every experiment tried, including negative results and reverted decisions, with dates
and exact numbers), see `MODEL_DOCUMENTATION.md` in this same directory. This document is a
synthesis of that log plus a fresh read of the current source code as of 2026-08-02; where the two
disagree, the source code is authoritative and this document follows it.

---

## 0. System overview

Two independent-but-coupled prediction layers, both built from scratch (no reuse of code from the
sibling `nhl/`/`mlb/`/`nfl/` projects, though the *conventions* — walk-forward validation, a genuine
held-out test set, paired-bootstrap significance testing — are shared discipline):

1. **Game-outcome model**: predicts a final score (hence margin, total, win probability) for every
   NBA game. Architecture: a closed-form **pace × rating** combine (team-level, walk-forward
   shrunk), originally paired with a play-by-play-derived **RAPM-lite** active-lineup adjustment
   layer (**currently DISABLED in production** — see §6), then a **score-distribution** layer that
   turns the point projection into calibrated win probability / spread / total probabilities.

2. **Player-props model**: predicts a full per-player stat line (points split into 2PT/3PT/FT
   makes+attempts, rebounds split OREB/DREB, assists, turnovers, steals, blocks) for every active
   player in a game. Architecture: six independently-validated per-player walk-forward rate models,
   a 3-level matchup-difficulty adjustment layer, a composition rule that reconciles player-level
   sums back to the game-outcome model's own team totals, and a per-category predictive-distribution
   layer (Poisson/Negative-Binomial for every category, empirically — not assumed).

**Current production status (2026-08-02)**: the codebase and its validation are complete and
extensively stress-tested (see §7). The live Railway deployment (`nba-worker`) is **currently
CRASHED** — root cause is a stats.nba.com datacenter-IP block on Railway's egress range, unrelated to
any modeling defect; see §9 for the full incident writeup and what's needed to restore it.

**Core philosophical commitment threaded through every component below**: nothing is assumed to
work by analogy. Every smoothing-family choice, every prior, every distributional family is
empirically tested against a naive floor and/or a competing family, on real historical data, with a
paired bootstrap significance test — and the log of *rejected* hypotheses (see `MODEL_DOCUMENTATION.md`)
is kept as visible as the log of adopted ones. This produced a recurring, important empirical
finding: **superficially similar stats routinely need opposite treatment** (minutes vs. shot-attempt
volume, 2PT/3PT-accuracy vs. FT-accuracy, steals vs. blocks, team-scheme vs. individual-defender
skill) — confirmed independently at least four separate times, never assumed to transfer.

---

## 1. Data foundation

Every source below is `nba_api` (a Python wrapper around stats.nba.com's undocumented JSON API)
unless noted otherwise. `FIRST_DEV_SEASON = 2015` (the 2015-16 season, "start of the modern
3PT/pace-and-space era," a user-confirmed cutoff) is the universal lower bound for anything not
explicitly noted as more restricted. All fetchers share a common shape: `REQUEST_DELAY_SECONDS = 0.6`
courtesy delay between calls, `MAX_FETCH_RETRIES = 3` with linear backoff, `timeout=30`, per-game
resumable caching to parquet (checkpointed every 50 games so a killed process resumes cleanly),
never re-fetching a season already fully cached unless `force=True`.

### 1.1 Sources and endpoints

| Source file | Endpoint(s) | Provides | Coverage |
|---|---|---|---|
| `fetch_schedule.py` | `LeagueGameLog` | Schedule, final scores, home/away | 2015-16+ |
| `fetch_boxscores.py` | `BoxScoreAdvancedV3` | Team-level `pace`/`offensiveRating`/`defensiveRating` (Phase 1's core input) | 2015-16+ |
| `fetch_boxscore_traditional.py` | `BoxScoreTraditionalV3` | Player box scores (minutes, all counting stats) | 2015-16+ |
| `fetch_playbyplay.py` | `PlayByPlayV3` | Play-by-play events, substitutions | 2015-16+ |
| `fetch_player_track.py` | `BoxScorePlayerTrackV3` | `touches`, `reboundChancesOffensive`/`Defensive` | 2015-16+, **no era gap** |
| `fetch_boxscore_matchups.py` | `BoxScoreMatchupsV3` + `BoxScoreDefensiveV2` | Per-defender/per-matchup difficulty data | **2017-18+ only** |
| `fetch_team_rosters.py` | `CommonTeamRoster` | Per-season `POSITION` field | 2015-16+ |
| `fetch_rotowire_lineups.py` | RotoWire scrape (not `nba_api`) | Live injury report (Out/Doubtful/etc.) | **today only, no archive** |
| `fetch_rotation.py` | `GameRotation` | **ABANDONED** — superseded by PBP-substitution reconstruction | n/a |

### 1.2 Coverage boundaries that matter architecturally

- **`BoxScoreMatchupsV3`/`BoxScoreDefensiveV2` fail entirely (`IndexError`/`AttributeError`) for
  2015-16 and 2016-17**, succeeding cleanly from 2017-18 onward — confirmed live, and matching the
  real industry-wide Second Spectrum player-tracking rollout, not a bug to retry around. Frozen as
  `MATCHUP_DATA_START_SEASON = 2017` — the six per-player rate models still validate/run on the full
  2015-2025 range; only the matchup-difficulty *layer* on top is restricted to 2017-2025.
- **`BoxScorePlayerTrackV3` has NO coverage gap** — confirmed working across the full 2015-2025
  range, same reliability profile as the advanced/traditional box scores.
- **`GameRotation` was abandoned entirely** after being found to be "not just erroring, but
  genuinely SLOW right now" — direct timing tests showed several 25-30s round-trips for successful
  200 responses and 30s hard timeouts for failures, at any request rate. Replaced by reconstructing
  lineup stints from `BoxScoreTraditionalV3` starters (row order, not the `position` field — see
  §1.4) plus a chronological walk through `PlayByPlayV3` substitution events (`build_stints.py`).
  `fetch_rotation.py` is confirmed dead code — not imported by any pipeline entry point.

### 1.3 The one genuinely un-fixable live-only dependency: RotoWire

`fetch_rotowire_lineups.py` scrapes `rotowire.com/basketball/tables/injury-report.php` directly via
`requests` (not `nba_api`) with a browser-like header set, **no retry loop at all** (a single
attempt, `resp.raise_for_status()`), and **no parquet caching whatsoever** — by design, since there
is nothing to cache: RotoWire's report only ever reflects the CURRENT day, with no historical
archive. This is documented as a real, unfixable-at-the-data-layer gap:

> "REAL GAP FOUND (2026-08-01, full-model audit): every one of this project's own historical
> spot-checks ... was silently exposed to this exact contamination — a player genuinely out on the
> target historical date but healthy today wouldn't be excluded, and a player flagged Out today
> (for an unrelated, much later injury) would be wrongly excluded from a past game he actually
> played real minutes in. Not fixable at the data layer (there is nothing to backfill from), so the
> honest fix is an explicit, loud warning rather than a silent, invisible distortion."

Mitigation: `warn_if_stale_for_backtest(game_date)` prints a WARNING whenever `game_date` isn't
literally today. `CommonTeamRoster`'s per-team roster snapshot (used for mid-season trade/waiver
detection, §1.4) carries the identical wall-clock-only limitation. **Both are genuinely live-only
signals — there is no way to backtest injury exclusion or trade contamination on a historical date
with the data sources available**, and this should be treated as an accepted, permanent modeling
boundary, not an open bug.

### 1.4 Real data-quality bugs found and fixed during ingestion/transform (`build_stints.py`)

Four independent, confirmed-on-real-games bugs in the stint-reconstruction pipeline (which feeds
Phase 2's RAPM-lite fit, §6) — each is a good illustration of why this project insists on spot-
checking against real games rather than trusting an aggregate metric alone:

1. **Rebound Off/Def cumulative-count bug**: the play-by-play description string `"Off:N Def:M"` on
   a Rebound row is that player's CUMULATIVE rebound total *so far in the game*, not a per-event
   flag. Naively checking for the substring `"Def:1"` only matched a player's exact first defensive
   rebound, silently missing every later one — confirmed to undercount total possessions by ~30% on
   a real 2023-24 game (132 vs. an expected ~191). Fixed by comparing each Rebound row's team
   against the team of the immediately preceding "Missed Shot" row instead.
2. **Score-placeholder "0" era-dependent bug**: `scoreHome`/`scoreAway` are blank on non-scoring
   rows in recent seasons but are the **literal string `"0"`** on non-scoring rows in older games
   (confirmed on a real 2015-16 game) — `pd.to_numeric("0")` parses that as a real value, so a naive
   forward-fill silently used the wrong, too-low placeholder instead of carrying the real cumulative
   score forward.
3. **Starter-detection via the `position` field is unreliable across eras**: confirmed on a real
   2015-16 game where 7 home and 9 away players had a non-blank `position` value (should be exactly
   5 starters). Row order, by contrast, checked out on every game tested — the first 5 rows per
   team always sum to exactly the team-level dataset's own "Starters" points total. Fixed by using
   row order, not the `position` field, for starter detection (a separate, more reliable source —
   `CommonTeamRoster`'s per-season `POSITION` — is used for the *matchup-difficulty* position-group
   logic instead, a deliberately different purpose).
4. **Diacritic name-matching cascade failure**: `PlayByPlayV3`'s substitution text is ASCII-
   normalized (`"SUB: Murray FOR Jokic"`) while `BoxScoreTraditionalV3`'s `familyName` retains the
   real diacritic (`"Jokić"`) — an exact-string lookup failed to resolve `"Jokic"`, and because that
   substitution's incoming player couldn't be re-added to the on-court set, **every subsequent stint
   for that team was short one player for the rest of the game**. Confirmed on a real 2023-24 game:
   reconciliation was off by −44/−41 points before the fix, exact after.

Every reconstructed game is reconciled against the real box-score final score
(`reconcile_game`), and `build_season_stints` reports an aggregate **coverage** percentage (total
points covered by clean, 5-vs-5-verified stints ÷ total actual points) rather than silently trusting
every game — a transparent data-quality gate, not a black box.

### 1.5 Orchestration (`refresh_data.py`) and a second real gap found this session

`refresh_all_data()` re-touches **only the current season** on every call (every prior, complete
season is fetched once by the Phase 0 backfill and cached permanently) — in order: force-refetch the
schedule → advanced box scores → play-by-play → traditional box scores → player-track → **force**-
refetch rosters (current season only, since mid-season trades need the live snapshot) →
conditionally (`if current >= MATCHUP_DATA_START_SEASON`) matchups + defensive box scores → force-
rebuild the season's lineup stints. A real gap was found and fixed here:

> "SECOND BUG FOUND AND FIXED (2026-08-01, full-model audit): this function still never fetched
> player-tracking ... matchup-difficulty ... or team-roster ... data for the CURRENT season at
> all ... this was silently invisible because 2025-26's files already happened to exist on disk from
> a one-time/manual backfill, but the moment the NEXT season (2026-27) begins, the live props
> pipeline would have silently degraded to NaN rebounding/playmaking rates and zero matchup
> adjustment for the entire new season, with no error or warning anywhere."

**This is now fixed in code** — but is exactly the class of gap worth re-verifying is *actually
exercised* once the 2026-27 season genuinely begins (see §10).

---

## 2. The game-outcome model — Phase 1 (team strength: pace × rating)

### 2.1 Core data structure

`team_strength.build_team_game_log(start_year, end_year)` produces **one row per team per game**
(so every real game contributes two rows — a home-team row and an away-team row), by joining the
cached schedule (final scores, home/away, date/season) with the cached team-level *advanced* box
score (that game's realized `pace`, `offensiveRating`, `defensiveRating`, `possessions`). **Regular
season only** — playoff games are excluded from training data on the reasoning that playoff
rotations/intensity are a downstream *prediction target* (the model will eventually be asked to
predict a playoff game), not a distribution the training signal should be contaminated by, mirroring
the same choice in the MLB sibling project (excluding spring training).

### 2.2 The walk-forward shrinkage primitive (`shrinkage.py`)

This is the single most-reused piece of math in the entire codebase — Phase 1's PACE/OFF_RATING/
DEF_RATING, and later `team_stat_rates.py`'s DREB/AST/TOV/STL/BLK, are both built directly on top of
it (two independent applications of the identical function, not two reimplementations).

**The leak-guard problem it solves.** A trailing league-average statistic needs to use only
*strictly prior* real games. But the log has **two rows per game** (home + away). If you naively
sort the log and take a flat `.shift(1)` over it, there is no guarantee that a given game's away-row
doesn't land immediately after that SAME game's home-row in the sort order — in which case the away
row's "trailing average" would leak the home team's own result from the very game currently being
scored. The fix: `_trailing_league_stat` collapses to **one row per real `gameId` first** (taking
the mean of the two teams' values for that game), computes the trailing statistic on that
game-level series (so `.shift(1)` always skips a whole game, never one side of it), then broadcasts
the result back to both team-rows via a merge on `gameId`. This makes the leak structurally
impossible regardless of row order, not just "unlikely" — a hard invariant, not a tuned parameter.

**The exposure unit.** Games, not minutes/possessions: OFF_RATING/DEF_RATING/PACE are *already*
possession-normalized per game by stats.nba.com, so unlike (say) NHL power-play time-on-ice (which
varies hugely game to game, making a low-TOI game's noisy rate as influential as a high-TOI one if
weighted by games), one NBA game is already a consistent, comparable unit of signal regardless of
how many possessions it contained.

**The core formula** (`add_walk_forward_rate`, for a FOR/AGAINST pair like OFF_RATING/DEF_RATING):

For team *t* in season *s*, at the point of its *n*-th game that season (0-indexed, so *n* games
have already been played strictly before this one):

```
games_before(t,s,n)     = n
for_cumsum_before        = Σ (that team's own "for" value) over its first n games this season
against_cumsum_before    = Σ (that team's own "against" value) over its first n games this season

attack_rate(t,s,n)  = [ for_cumsum_before      + prior_games × prior_mean_for(t,s,n)     ] / (n + prior_games)
defense_rate(t,s,n) = [ against_cumsum_before  + prior_games × prior_mean_against(t,s,n) ] / (n + prior_games)
```

This is a **Bayesian-flavored shrinkage estimator**: `prior_games` acts as a pseudo-count of
"imaginary games" pulling the raw in-season average toward `prior_mean` — at `n=0` (a team's first
game of the season) the rate equals `prior_mean` exactly; as `n → ∞` the prior's influence vanishes
and the rate converges to the team's own raw in-season average. This is mathematically the same
structure as a Beta-Binomial or Normal-Normal conjugate shrinkage estimator, though it is not
derived from an explicit likelihood model — it's a pragmatic, empirically-validated approximation to
one (no distributional assumption is placed on `for_col`/`against_col` themselves; the shrinkage
weight is exposure-count-based, not variance-based, unlike a true empirical-Bayes estimator that
would weight by the inverse of each team's own residual variance).

Three independently-adjustable levers control what `prior_mean` actually is and how the team's own
signal is estimated — **all three are now adopted, non-default values**, each found and validated
completely independently, in three separate rounds of testing over this project's history:

**Lever 1 — `league_avg_halflife_games` (target recency)**: `prior_mean` is either
1. `_trailing_league_stat` — a flat, infinite-memory cumulative mean across the ENTIRE historical
   range (the original default, `None`), or
2. `_trailing_league_stat_ewma` — an EWMA (exponentially-weighted moving average, half-life measured
   in games not calendar time) over the same per-game-collapsed, leak-guarded series.

   Motivation: mean points/team-game rose from 102.7 (2015-16) to 115.6 (2025-26), *almost
   monotonically throughout the entire dev range* — a real, continuous multi-year scoring-inflation
   trend, not a sudden jump only at the dev/holdout boundary. A flat infinite-memory average of the
   league-wide rating dilutes this trend across nine seasons of history, systematically
   under-predicting recent, higher-scoring eras. **Adopted value: `LEAGUE_AVG_HALFLIFE_GAMES_RATING
   = 2000.0` games**. This runs on the per-`gameId`-collapsed series (`_trailing_league_stat_ewma`
   collapses each real game to one row before computing the EWMA), and the dev range averages
   ~1,193 real games/season — so 2000 games ≈ **1.7 NBA seasons**, not a small number, since even a
   "responsive" target here still needs to average over enough of the league to not be dominated by
   noise. (An earlier draft of this document mis-stated this as "~2.4 seasons," implicitly assuming
   team-ROW units — 2× the real-game count — rather than the real-game units the EWMA actually runs
   on; corrected here after review caught the discrepancy.)

**Lever 2 — `cross_season_weight` (early-season prior)**: at the very first game of a new season
(`n=0`), what should the prior actually be? The original default (`0.0`) resets fully to the
league-wide average, discarding everything the model learned about *that specific team* the
previous season. A nonzero weight blends in the team's own prior-season average instead:

```
prior_mean(t,s,n=0) = w × last_season_avg(t, s-1) + (1-w) × trailing_league_avg(s)
```

   (only applied when a prior season exists for that team; new/relocated franchises still fall back
   to the pure league average). Roster/scheme continuity between one season and the next carries
   real signal a full reset throws away. **Adopted value: `CROSS_SEASON_WEIGHT_RATING = 0.3`.**

**Lever 3 — `own_halflife_games` (own-history recency)**: `for_cumsum_before`/`against_cumsum_before`
above are a **flat, equally-weighted cumulative sum** — game 1 of a team's own season counts exactly
as much toward its rating as last night's game, no matter how far into the season. This is
structurally distinct from both levers above (Lever 1 changes the shared TARGET everyone shrinks
toward; Lever 2 changes only the very-first-game PRIOR; neither touches how a team's own in-season
games are weighted against each other). When set, the flat sum is replaced by
`_trailing_own_mean_ewma` — an EWMA of the team's own value, scaled back up by `games_before` to
preserve the same blending-weight semantics as the un-recency-weighted formula (i.e. the "how much
do we trust the team's own signal vs. the prior" balance is unchanged; only *which* estimate of the
team's own signal is used changes). **Adopted value: `OWN_HALFLIFE_GAMES_RATING = 40.0` games**
(applied to rtg only, not pace — untested for pace).

All three levers were swept via the SAME two-stage-then-one-time-holdout protocol (§7) and are
individually holdout-confirmed real, non-overlapping wins — full numeric detail in
`MODEL_DOCUMENTATION.md` Sec29/33/36. **Current adopted constants** (in `team_strength.py`):

| Constant | Value | Was |
|---|---|---|
| `PRIOR_GAMES_RATING` | 12.0 | 15.0 |
| `PRIOR_GAMES_PACE` | 15.0 | (never changed — untested for pace) |
| `CROSS_SEASON_WEIGHT_RATING` | 0.3 | 0.0 |
| `OWN_HALFLIFE_GAMES_RATING` | 40.0 | None (flat) |
| `LEAGUE_AVG_HALFLIFE_GAMES_RATING` | 2000.0 | None (flat, infinite memory) |

PACE uses `add_walk_forward_mean` (the single-value analog — both teams in a game share one
realized pace, so it isn't a FOR/AGAINST pair) with the SAME `cross_season_weight`/
`league_avg_halflife_games` levers but has never been tested with `own_halflife_games`.

### 2.3 Home-court advantage (`home_court.py`)

Fit **empirically as a multiplicative factor**, not assumed as a fixed additive constant ("home
team +3", NBA folklore). For every game, the log-ratio of actual-to-baseline rating on each side
should equal `+log(mult)` for home and `-log(mult)` for away (baseline = the pace×rating combine
with `mult` forced to exactly 1.0). Averaging the home-side log-ratio and the negated away-side
log-ratio (two independent estimates of the same `log(mult)`) uses the whole game's information:

```
log_mult = ( home_log_ratio − away_log_ratio ) / 2
home_court_mult = exp(log_mult)
```

Checked for temporal drift (home-court advantage is publicly documented as having compressed
league-wide over the last decade) via `fit_home_court_by_season`. Production uses
`fit_home_court_walk_forward` — a trailing EWMA (half-life in games, default 400.0, an
**un-calibrated placeholder**, swept 100–1600 in a Stage-1 test this session and found to make
**no detectable difference at any value tested** — the home-court effect itself is simply too small
relative to noise for this specific parameter to matter; see §8 for the honest negative result).

**Real bug found and fixed in this file's history**: the original version read raw REALIZED
box-score columns for "baseline" (i.e. that game's own actual final rating), making "actual" and
"baseline" circularly derived from the same outcome — this produced a home-court multiplier
*below* 1.0 in every season tested, despite the same data showing a textbook, unambiguous real
home-court edge (home teams averaging 106.2 vs. away's 103.5 points, 58.4% home win rate) — an
unambiguous sign the fitting function itself was broken, not that home advantage was negative.
Fixed by reading the correct walk-forward PRE-game columns instead.

### 2.4 The combine (`project_game`)

A **multiplicative-ratio** idiom (not additive), applied to both pace and rating so the two combine
cleanly through a shared mathematical structure:

```
projected_pace = league_avg_pace × (home_pace / league_avg_pace) × (away_pace / league_avg_pace)

projected_home_rating = league_avg_rating × (home_offRtg / league_avg_rating) × (away_defRtg / league_avg_rating) × home_court_mult
projected_away_rating = league_avg_rating × (away_offRtg / league_avg_rating) × (home_defRtg / league_avg_rating) / home_court_mult

projected_home_score = projected_home_rating × projected_pace / 100
projected_away_score = projected_away_rating × projected_pace / 100
```

The home-court multiplier is applied **symmetrically** (home's rating scaled up, away's scaled down
by the same factor) rather than as a flat point bonus, so it composes cleanly with the ratio
structure rather than needing a separate additive term bolted on afterward.

`project_team_stat` (`team_stat_rates.py`) is the exact same multiplicative-ratio idiom, generalized
to an arbitrary FOR/AGAINST stat pair (used for DREB/AST/TOV/STL/BLK, §3).

**A real, previously-unchecked mis-calibration, found via external review (2026-08-02)**: the
combine's algebra implicitly assumes a team's rating deviation from league average carries a
**slope of exactly 1.0** into the projection. Never checked until an external technical review of
this document flagged it as an untested assumption. Regressing realized (home-court-adjusted)
log-rating deviation on the walk-forward-predicted log-deviation terms, pooled across both sides,
full dev+holdout range (n=26,352):

```
slope_off = 0.9163 (se=0.0264)      slope_def = 0.9066 (se=0.0297)
```

Both are real, ~3.1–3.2-standard-error departures from 1.0 — **real outcomes regress toward league
average MORE than the combine's implicit slope assumes**, meaning the model is mildly overconfident
for teams far from average. The identical check for pace found the OPPOSITE sign
(`slope_home=1.0721, slope_away=1.1034`, both above 1.0 — pace slightly UNDER-extrapolates). Not yet
built into a corrected model — this is a validated diagnostic finding, queued as a candidate lever
(a fitted damping/amplification coefficient per deviation term, tested through the same two-stage-
then-holdout protocol as everything else) rather than adopted on the strength of the diagnostic
alone. See `MODEL_DOCUMENTATION.md` Sec44.3.

### 2.5 Headline validated results

**vs. naive floor** (dev range, 2015-16 through 2023-24, 10,737 games): that team's own trailing-5-
game scoring average, **no opponent adjustment at all** — total_mae 15.08 vs. 15.26 (naive),
margin_mae 10.47 vs. 11.28, straight-up accuracy **64.0% vs. 57.3%** — all three real improvements.

**vs. the real market** (dev range only, 2015-16 through 2022-23, 8,841 games matched against
SportsbookReviewsOnline closing lines — see §9.5 for the full writeup): the model has a real,
statistically significant gap on every metric checked — margin_mae 10.29 vs. the market's 9.90 (real
regression), total_mae 14.76 vs. 14.14 (real regression), straight-up accuracy 65.2% vs. the market's
67.5% (real regression). Not a red flag — beating closing lines is a famously hard bar most models
never clear — but this is the project's first and only external, non-naive benchmark, and it
recalibrates what "the model is doing well" actually means: comfortably ahead of a weak floor,
genuinely behind the real market.

### 2.6 The scoring-era-drift finding, reframed: likely mostly a scale artifact, not model decay

Three adopted levers (this section) each measurably improved how the model copes with rising raw
margin_mae over time — but none of them eliminate the underlying trend, and `own_halflife_games`'s
own dev-vs-holdout gap on margin_mae still shows a real widening. **A follow-up diagnostic (2026-08-02,
prompted by external review) puts this in a materially different light.** Margin_mae is not
scale-invariant, and the REALIZED margin standard deviation (true game-to-game unpredictability) is
independently rising almost in lockstep with the documented MAE trend (r=+0.846 vs. margin_mae's own
r=+0.828 against season). Two normalized views:

- **model_margin_mae ÷ naive_margin_mae** (relative skill vs. floor): essentially flat across all 11
  dev+holdout seasons (r=+0.046, p=0.89) — holdout's mean (0.909) is slightly BETTER than dev's
  (0.922), not worse.
- **model_margin_mae ÷ realized_margin_std** (does model error scale with true task difficulty):
  flat-to-improving (r=−0.375) — holdout's mean (0.694) is again slightly BETTER than dev's (0.726).

**The two holdout seasons post the best-normalized margin performance of the entire 11-season
range, not the worst.** This strongly suggests the raw margin_mae trend that's motivated three
separate adopted fixes is predominantly explained by genuinely rising, irreducible game-to-game
unpredictability across NBA eras — not by the model falling further behind a fixed level of
difficulty. This does not retroactively undo any of the three adopted levers (each cleared its own
proper paired-bootstrap test against the prior config on identical games, which remains valid
regardless) — but it reframes whether continuing to chase "margin drift" as a distinct, unsolved
mechanism is the right framing, vs. accepting it as a real, already-coped-with characteristic of the
sport. Held with appropriate caution (only 11 season-level data points). See
`MODEL_DOCUMENTATION.md` Sec44.2 for the full numbers — **still worth the P1 attention in §10, but
the priority ordering there should account for this reframing.**

---

## 3. Team-level anchoring for DREB/AST/TOV/STL/BLK (`team_stat_rates.py`)

Built to close a long-flagged gap: `usage_allocation.py`'s original v1 design deliberately left
REB/AST/TOV/STL/BLK **unanchored** at the team level (bottom-up player-sum only, no macro total to
reconcile against) — a documented fast-follow, not an oversight, since Phase 1 never built a
team-level projection for these categories the way it did for points via pace×rating.

**Architecture**: an EXACT mirror of `team_strength.py` — each category gets a FOR (this team's own
tendency) / AGAINST (what this team allows the opponent) pair, walk-forward shrunk via the
identical `shrinkage.add_walk_forward_rate` primitive Phase 1 uses, then combined via the identical
multiplicative-ratio idiom (`project_team_stat`). No new data ingest needed — built entirely from
`BoxScoreTraditionalV3` player box scores already cached for other purposes, aggregated (summed
across every player on a team in a game) to the team level.

**Priors** (`PRIOR_GAMES`, games): oreb=20, dreb=20, ast=20, tov=20, stl=20, blk=20 — each
independently swept and confirmed, not assumed to transfer from OFF/DEF_RATING's own prior or from
each other (this project's standing discipline: every category earns its own check).

**`ADOPTED_CATEGORIES = ("dreb", "ast", "tov", "stl", "blk")`** — **OREB is deliberately excluded**.
On its original confirmatory holdout check, OREB's model genuinely LOSES to naive (3.0661 vs. 3.0401
MAE) — a real veto, not just a narrower margin. This is the single most-investigated open item in
the whole codebase: **four independently-designed mechanisms** have since been tried specifically to
fix OREB team-level anchoring, and every one converges to the exact same ceiling (a statistical tie
with naive at best, never a clean beat):
1. Reduced/increased shrinkage strength — monotonically worse in both directions.
2. Adaptive (EWMA) league-average target — real progress, moved OREB from a clear loss to a
   statistical tie, but a tie doesn't justify the added complexity.
3. `cross_season_weight` — real regression/veto on its own holdout check.
4. `own_halflife_games` — statistical tie with naive again (delta +0.0233, CI includes zero).

**This is now treated as decisive, convergent evidence of a genuine structural limit** in the
`add_walk_forward_rate` primitive as applied to OREB specifically — not a tuning problem any further
parameter sweep is likely to solve. OREB stays unanchored (bottom-up player-sum only). A genuine fix,
if one exists, needs either a structurally different primitive (e.g. explicit possession-share or
shot-miss-location modeling) or a fundamentally different signal — flagged as a real open research
question, not a "just try more values" item.

The other 5 categories all cleared their confirmatory holdout checks and are live-wired via
`usage_allocation.allocate_team_total` (generalized from the points-specific
`allocate_team_points`). Two further real wins have since been layered on top for these 5:
`CROSS_SEASON_WEIGHT_STAT = 0.25` and `OWN_HALFLIFE_GAMES_STAT = 20.0` — both via the same
per-category-default-with-sentinel design pattern documented in `team_stat_rates.py` (see §7 for why
`own_halflife_games` specifically needed a dedicated sentinel rather than reusing `None` as "use the
default", the way `cross_season_weight` could).

---

## 4. The score-distribution layer (`score_distribution.py`) — Phase 3

Turns Phase 1(+2)'s point projection (a mean) into a full outcome distribution — win probability,
spread, total — rather than stopping at a point estimate.

**Distributional family choice, justified by domain reasoning (not assumed a priori without
checking)**: an NBA score aggregates over roughly 100 quasi-independent possessions per team, so
(unlike the NHL sibling's low-scoring, discrete Poisson/Dixon-Coles approach) a **continuous**
distribution around the point projection is the natural fit by the Central Limit Theorem — the CLT
argument holds at the TEAM level (many possessions aggregated) even though, as discovered later
(§5), the *identical* CLT argument does NOT hold for an individual PLAYER's own shot attempts in one
game (nowhere near enough volume).

**Variance model**: fit as a function of projected PACE (more possessions → more aggregated
variance, not a flat constant) via OLS: `variance ≈ intercept + slope × pace`, floored at
`MIN_VARIANCE = 4.0` so a pathological low-pace fit can't predict a near-zero variance.

**Home/away residual correlation**: fit empirically (`fit_home_away_correlation`, a single global
Pearson correlation, not pace-conditional — a deliberately simpler first cut, flagged as a candidate
refinement not assumed necessary) rather than assumed zero. Rationale: a fast, high-possession game
inflates BOTH sides' variance simultaneously via a shared pace-driven component, even after
controlling for each side's own mean — treating home/away residuals as independent would understate
the true joint uncertainty.

**Margin/total closed form**, from the joint home/away Normal(-family) parameters:
```
cov        = corr × sqrt(var_home × var_away)
margin_mean = mean_home − mean_away        margin_var = var_home + var_away − 2·cov
total_mean  = mean_home + mean_away        total_var  = var_home + var_away + 2·cov
```

**Family choice (Normal vs. Student-t)**: both are fit and compared via calibration coverage (PIT —
probability-integral-transform vs. nominal quantiles) + log-score (a proper scoring rule that works
uniformly across families, chosen over CRPS specifically because CRPS only has a simple closed form
for Normal). **Adopted: Normal**, for simplicity — at the tested sample size, neither pace-scaled
variance nor Student-t tails were *detectably* better than the simpler flat-variance/Normal
baseline. Pace-scaling is kept anyway as physically motivated (more possessions should mean more
variance) even though not yet proven superior at this sample size; Student-t machinery is kept as a
tested, available, currently-unused primitive (df fit via method-of-moments on excess kurtosis:
`df = 6/excess_kurtosis + 4`, floored at 2.5, falling back to `df=200` — effectively Normal — when
excess kurtosis is non-positive, since fitting a Student-t to thinner-than-normal tails is not
meaningful).

**A real, non-obvious bug found and fixed here (`_t_scale`)**: scipy parameterizes a Student-t as
`X = loc + scale·T` where `Var(T) = df/(df−2)` for `df>2` — NOT 1. Passing `scale = sqrt(variance)`
directly (what every t-family call site here, and the copy-pasted math in `prop_distribution.py`,
originally did) silently **inflates the realized variance by a factor of `df/(df−2)`**, contradicting
this module's own explicit "matched variance, only tail shape differs" design intent. Confirmed
empirically: `t.rvs(df=9.5, scale=10, N=3e6)` has empirical variance ≈126.5, matching
`100 × 9.5/7.5 ≈ 126.67`, not the naive 100. This bug was silently present in the ORIGINAL dev-range
Normal-vs-t comparison that decided the family in the first place (though dormant in live production
today, since every adopted category ended up using `family='normal'` anyway, which never calls this
path). Fixed via a shared `_t_scale(var, df)` helper, applied everywhere a t-family scale is
computed, in both `score_distribution.py` and `prop_distribution.py`.

**Calibration result**: 95% nominal interval → ~94–95% empirical coverage — well-calibrated.

---

## 5. The player-props subsystem

Predicts a full per-player stat line: points (split 2PT/3PT/FT makes+attempts), rebounds (split
OREB/DREB), assists, turnovers, steals, blocks. Six independently-built and independently-validated
per-player rate models, a 3-level matchup-difficulty adjustment, a composition rule reconciling
player sums back to team totals, and a per-category predictive-distribution layer. The single
biggest recurring finding across this entire subsystem: **superficially similar stats need
opposite smoothing treatment**, confirmed empirically at least four separate times — never assumed
by analogy, always checked.

### 5.1 The shared primitive (`player_rate_shrinkage.py`)

Generalizes `shrinkage.py`'s team-level pattern to (a) an arbitrary EXPOSURE unit (attempts,
rebound chances, touches, minutes — not just "1 game = 1 unit") and (b) player-level grouping.

**Core formula** (`add_walk_forward_player_rate`):
```
shrunk_rate = (numerator_cumsum_before + prior_exposure × league_avg_rate)
              / (exposure_cumsum_before + prior_exposure)
```
`numerator_before`/`exposure_before` are per-`(playerId, season)` cumulative sums strictly before
the current row (season-reset, no cross-season blending in v1 — "a deliberate simplification, not
an oversight"). `league_avg_rate` defaults to a **pooled** trailing rate (`_trailing_league_rate`):
collapse to one row per `gameId` (SUMMING numerator and exposure across every player-row sharing
that game, not averaging — there are ~20–26 player-rows per game, not 2, so a mean-of-rows approach
like the team-level primitive uses would be wrong here), `shift(1)`, expanding sum, divide. This is
structurally the NHL sibling's `add_walk_forward_toi_rate` pattern, not `shrinkage.py`'s own
per-row-mean approach — a deliberate, documented difference between the two "walk-forward shrinkage"
primitives in this codebase.

A parallel `add_walk_forward_player_mean_ewm` handles a single per-game VALUE (not a
numerator/exposure pair) with pure EWMA — no expanding-shrinkage prior at all — used for minutes and
several EWMA-family categories below (`min_periods=1`: a player's 2nd career game projects from
their 1st game alone; explicitly tested and found to already outperform adding any league-average
floor on top).

`add_era_adjusted_player_rate` — a "detrend-then-retrend" architecture (deflate each historical
numerator by the trailing league rate AT THAT TIME, run the unchanged shrinkage machinery on the
deflated numerator, re-inflate at the end by a responsive current-era EWMA estimate) — is built,
tested, and **validated-but-NOT-adopted** (see steals below; it made real holdout performance worse,
twice, in increasingly sophisticated attempts).

### 5.2 The volume-vs-skill split, confirmed independently across every category

| Category | Metric type | Winning family | Losing family (how much worse) |
|---|---|---|---|
| Minutes | pure volume/role | EWMA, halflife=**2.0 games** | expanding-shrinkage — *monotonically* worse as prior grows (MAE 6.43→7.99 for prior 1→20 games) |
| 2PT/3PT attempts | volume/role | EWMA, halflife=**10.0 games** | expanding (best 1.198 vs EWMA's 1.173) |
| 2PT/3PT make rate | stable skill | expanding, prior=**150 attempts** each | EWMA — "actively harmful, not just unhelpful" (worst EWMA 0.77 vs expanding's 0.553, ~40% worse) |
| FT attempts | stable skill (draws fouls) | expanding, prior=**20 minutes** | EWMA (1.5303 vs 1.5615) — opposite of 2PT/3PT attempts |
| FT make rate | stable skill, smaller sample | expanding, prior=**15 attempts** | prior=100 over-shrinks (0.3778 vs 0.3724) |
| OREB conversion | stable skill | expanding, prior=**50 chances** | (no EWMA variant built — expanding always wins here) |
| DREB conversion | stable skill | expanding, prior=**150 chances** | (unchanged — no real dev/holdout gap) |
| AST rate | stable-ish skill | expanding, prior=**100 touches** | prior=300 (copied from TOV) was a REAL REGRESSION even though 300 helps TOV |
| TOV rate | stable skill | expanding, prior=**300 touches** | (unchanged — no real dev/holdout gap) |
| STL rate | persistent individual skill (anticipation) | expanding, prior=**200 minutes** | EWMA (0.668–0.675 vs expanding's 0.657–0.660) |
| BLK rate | **volatile**, unlike STL | EWMA, halflife=**10.0 games** | expanding (0.4224 vs naive's 0.4102 — a real regression, not just worse) |

FT specifically needed its own from-scratch re-sweep rather than inheriting either 2PT/3PT's or a
shared treatment — applying 2PT/3PT's exact recipe to FT was itself a confirmed real regression
(shrunk MAE 1.0726 vs. naive 1.0538) before the FT-specific priors above were found.

### 5.3 Exposure units, not just "per game"

Each category is normalized by the exposure unit that actually drives it, not a one-size-fits-all
per-game or per-minute rate: shooting attempts by minutes, make-rates by own attempts, rebounding by
**rebound chances** (opportunity, from player-tracking — not raw minutes, since two equal-minutes
players can have wildly different rebound opportunity depending on team shot profile/pace),
AST/TOV by **touches** ("fundamentally events that happen ON a touch, not per minute of standing
around"), STL/BLK by minutes.

### 5.4 Matchup difficulty (`matchup_difficulty.py`) — 3-level shrinkage hierarchy

Restricted to `season >= MATCHUP_DATA_START_SEASON (2017)` — the six rate models above still run on
the full 2015-2025 range; only this adjustment *layer* is boundary-restricted.

1. **Defender-specific rate** (`BoxScoreDefensiveV2`, keyed by `playerId` ALONE — not
   `(playerId, team)`, since a defender's skill is portable across a trade): a persistent individual
   skill, expanding-shrinkage wins clearly (points: prior=50 matchup-minutes, MAE 3.978 vs. naive
   4.056; every EWMA halflife tested was worse, up to 4.29).
2. **Position-group-vs-team rate** (`BoxScoreMatchupsV3`, collapsed by the OFFENSIVE player's roster
   position group via `CommonTeamRoster`'s `POSITION` field — **not** the matchups endpoint's own
   `positionOff` column, confirmed blank on fully half of all real 2022-23 rows): a volatile
   TEAM-SCHEME metric, the OPPOSITE family wins — points: EWMA halflife=10 (8.181) clearly beats
   every expanding prior tested (best 8.227, barely better than naive's 8.288 — expanding almost
   doesn't help at all at this level).
3. **League-average difficulty at that position group**: fixed floor, always available, no
   shrinkage.

AST matches points' family choice at both levels (expanding then EWMA); **TOV does not** — expanding-
shrinkage wins at BOTH levels for TOV (level 1 prior=200, level 2 prior=250 matchup-minutes),
confirmed independently, not assumed to match AST just because they're often discussed together.

`MIN_MATCHUP_MINUTES_TO_TRUST` gates whether a specific defender/position-group signal is trusted at
all (below the floor, falls back to the next, more pooled level) — explicitly flagged as an
un-tuned placeholder pending real-slate testing, not a validated cutoff.

**Sign convention**: relative difficulty = `league_avg_rate − this_rate` (positive = tougher than
average). A real, serious sign bug was found and fixed in the live wiring (`generate_props.py`, not
this file): the delta was being ADDED to a player's raw projection instead of SUBTRACTED, which
INCREASED points against tougher defenders and decreased them against weaker ones — exactly
backwards. The single subtraction fix is correct for BOTH points/AST (higher opponent rate = weaker
defense = should boost the offensive player) and TOV (higher `tov_forced` rate = TOUGHER defense,
opposite polarity) simultaneously, because the underlying delta already encodes each stat's own
convention before the sign is applied.

### 5.5 Composition rule (`usage_allocation.py`) — macro-anchor + micro-reallocation

The single most important design decision in the whole props subsystem, resolving a real
double-counting risk: RAPM/team-strength already answers "how many total points will this lineup
score" (a MACRO question); matchup difficulty answers a different, MICRO question ("given that
total, which player gets how much, and does the specific matchup shift the mix"). These compose as
**macro-anchor + micro-reallocation**, never as two independent additive stacks on the same number:

```
raw_points  = 2×2pt_proj_made + 3×3pt_proj_made + 1×ft_proj_made          (before matchup adjustment)
adjusted_points[p] = raw_points[p] − matchup_delta[p]                      (micro-reallocation)
usage_share[p] = clip(adjusted_points[p], 0) / Σ clip(adjusted_points, 0)  (renormalize; falls back
                                                                             to equal shares if the
                                                                             whole team clips to ≤0)
final_points[p] = usage_share[p] × team_total                             (macro-anchor — team_total
                                                                             comes from the game-score
                                                                             pipeline's own output)
```

Matchup difficulty reshapes shares; the team-strength/RAPM total sets the total. **The exact same
mechanism generalizes to DREB/AST/TOV/STL/BLK**, anchored against `team_stat_rates.py`'s team-level
totals (§3) instead of RAPM's points total — 5 of the 6 team-stat categories (all but OREB) are
anchored this way; **OREB is deliberately left unanchored** (bottom-up per-player rate-model sum,
no team-level rescaling at all), the direct consequence of OREB's team-level model genuinely losing
to naive on holdout (§3) — "honestly unanchored, not silently gold-plated."

### 5.6 Predictive distributions (`prop_distribution.py`)

**All 10 categories currently use COUNT families (Poisson or Negative-Binomial)** — this directly
overturned the plan's original a priori assumption that points/2PM/3PM/FTM should be continuous
(Normal/Student-t) by CLT reasoning:

```python
CATEGORY_FAMILY = {
    "points":   negbin,   "2pt_made": poisson,  "3pt_made": poisson,  "ft_made": negbin,
    "oreb":     poisson,  "dreb":     poisson,   "ast":      poisson,  "tov":     poisson,
    "stl":      poisson,  "blk":      poisson,
}
```

**Why the plan's original assumption was wrong, confirmed by an actual calibration check (not just
log-score) that had never been run before**: the CLT argument is genuinely valid for a TEAM's
aggregate points (dozens of shot attempts across 5 players) but NOT for one player's own made shots
in one game (5–10 2PT attempts, 2–5 3PT attempts, 1–4 FTs is nowhere near enough volume). Treating
all four high-count categories as COUNT distributions instead gave both a better log-score and
dramatically better calibration — e.g. FT-made calibration max-deviation dropped from 0.35 (badly
miscalibrated, Student-t) to 0.012 (NegBin).

Poisson-vs-NegBin selection: method-of-moments overdispersion check
(`r = mean² / (variance − mean)`, NB2 parameterization `var = mean + mean²/r`), used only when
empirical variance clears `1.2 × mean` — otherwise Poisson (the simpler, correct default). Continuous
(Normal/Student-t) machinery is kept as a fully tested, currently-unreachable primitive for any
future category that might genuinely need it.

**Two real bugs found and fixed here**:
1. **Variance-floor bug**: originally reused `score_distribution.predict_variance`'s team-scale
   floor (`MIN_VARIANCE=4.0`) directly on player-level stats whose real fitted variance is routinely
   under 1 (e.g. 2PT-makes variance at 1 projected attempt ≈ 0.54) — silently clamping nearly every
   player's variance up to exactly 4.0, and (more seriously) distorting the ORIGINAL family-choice
   validation itself, not just live output, since the same inflated-variance call was used
   internally to standardize residuals before fitting Student-t df. Fixed with a player-scale floor
   (`MIN_PLAYER_VARIANCE = 1e-3`).
2. **Degenerate Poisson mu=0 bug**: a player with an expanding-shrunk rate of exactly 0.0 (real — a
   perimeter player who's simply never recorded a block) projects `mean=0.0`; Poisson at μ=0 is a
   genuine point-mass at 0, so `logpmf(k>0, μ=0) = −∞` exactly — one real garbage-time event for that
   player poisons an entire evaluation set's mean log-score to `+∞`. Fixed with `MIN_COUNT_MEAN = 1e-3`
   as a floor — "a live projection can never be a certainty that something 'never happens'."

### 5.7 Two further real bugs found in the live wiring (`generate_props.py`)

- **EWMA NaN-on-season-reset bug**: EWMA-based columns (minutes, 2PT/3PT attempt-rate, block-rate)
  correctly reset to NaN on a player's first game of every new season (a deliberate season-boundary
  reset, not a rookie-only edge case) — but a plain `.last()` picked up that NaN row as the "latest
  snapshot" for a player's SECOND game of a new season too, even though a real, non-NaN value from
  late last season sat in an earlier row of the exact same log. This silently zeroed real,
  established veterans' points/2PT/3PT/block props (reallocating their share to teammates) every
  single season, leaguewide, during the week after opening night. Fixed by forward-filling before
  taking the last row.
- **Unconditional-fillna masking-missing-players bug**: an unconditional `.fillna(0.0)` made a
  genuinely missing player (e.g. a two-way/call-up with no cached rebounding/playmaking/defensive
  history) indistinguishable from a real, data-backed near-zero projection. Fixed with a helper that
  preserves the missing/NaN distinction explicitly.

### 5.8 A separate, related RAPM-adjacent primitive: `player_priors.py`

Not a player-prop rate model itself, but the ridge-penalty schedule for Phase 2's RAPM-lite fit
(§6): a variable-penalty (not variable-mean) empirical-Bayes formulation —
`lambda(games) = 2000 × (0.15 + 0.85 × 0.5^(games/120))` — low-experience players get a LARGER ridge
penalty (shrunk harder toward the shared league-average-zero prior), decaying toward a floor of
`2000 × 0.15 = 300` (never fully unregularized, since a fully unregularized column is not
well-posed under teammate multicollinearity). Chosen over a variable-prior-MEAN formulation
specifically to avoid a self-referential bootstrap problem ("you'd need RAPM ratings to fit the
curve that initializes RAPM ratings").

---

## 6. Phase 2 — RAPM-lite active-lineup adjustment (CURRENTLY DISABLED)

**`INCLUDE_LINEUP_ADJUSTMENT = False`** in both live pipeline entry points, since 2026-08-01. This
section documents the full mechanism (all of it remains in place, tested, and importable — nothing
was deleted) and exactly why it's off.

### 6.1 The math (`rapm_lite.py`)

**Regularized Adjusted Plus-Minus, lite**: a generalized RIDGE regression with a **per-player-column
penalty** (not one global alpha, so `sklearn.Ridge` can't be used directly — solved via the normal
equations):

```
beta = solve( XᵀWX + diag(λ),  Xᵀ W y_centered )
```

- Design matrix `X`: two rows per physical stint (one per team's offensive half). For the offensive
  team's row: `off_p = 1` for each of its 5 on-court players' offensive columns, `def_p = 1` for the
  opposing 5 players' defensive columns, everything else 0. `2 × n_players` columns total.
- Target `y = 100 × pts / possessions` — points per 100 possessions, same units as
  `team_strength`'s ratings — **centered on its weighted mean first**, so the fitted `off_rapm`/
  `def_rapm` are directly interpretable as points above/below average per 100 possessions.
- Row weight `w = possessions × exposureWeight` (garbage-time down-weight, §6.2, folded directly
  into the regression weight, not applied as a separate filter).
- Per-column penalty `λ(games)` from `player_priors.lambda_for_experience` (§5.8) — the SAME penalty
  applied to both a player's offensive and defensive column.
- **Walk-forward refit cadence**: every 14 days (`REFIT_PERIOD_DAYS = 14`), not per individual game
  date — "a pragmatic tractability compromise... refitting a several-hundred-to-a-few-thousand-
  column ridge system for each of ~1,500+ individual game dates ... would be far more compute than a
  biweekly refit, while still being fully walk-forward-safe" (each checkpoint fit uses only stints
  strictly before that checkpoint date). `MIN_STINTS_TO_FIT = 200` guards against fitting on a tiny,
  unstable early sample.

### 6.2 Garbage-time down-weighting (`garbage_time.py`)

`exposureWeight` = 1.0 normally, `DOWNWEIGHT_FACTOR = 0.25` (an un-calibrated placeholder) once a
stint's absolute pre-stint margin exceeds a time-decaying threshold: `25.0 − 15.0 × elapsed_fraction`
(i.e. a 25-point allowed margin at tip-off, narrowing linearly to 10 points by end of regulation —
"follows the same shape as Cleaning the Glass's public garbage-time definition ... but with
placeholder, un-calibrated cutoffs"). A real bug was found and fixed here: `total_length` (used to
compute `elapsed_fraction`) was originally computed as a single max across an ENTIRE SEASON's worth
of stints passed in at once, rather than per individual game — one multi-OT game in 2015-16
(40,800 tenths) silently understated garbage-time down-weighting for ~1,185 of that season's 1,220
OTHER games, each divided by the wrong game's length. Fixed by computing `total_length` per game.
This weight is a genuine no-op (1.0) unless a stint actually crosses the threshold — "safe to leave
wired in and simply find it doesn't matter if that's what validation shows."

### 6.3 Active-lineup projection (`lineup_rating.py`) — where the disabled bug lived

```
adjusted_oRtg = team_baseline_oRtg + Σ_active_players( minutes_share_p × (rapm_off_p − team_recent_avg_rapm_off) )
```
(symmetric formula for dRtg). An unrated player (too new/insufficient stints) contributes exactly 0
deviation — treated as league-average, the safest default. `team_recent_avg_rapm_off/def` is a
minutes-weighted average of each rotation player's RAPM-lite rating over the trailing
`DEFAULT_LOOKBACK_GAMES = 10` games, using REAL historical minutes internally regardless of which
mode below is used for the outer projection.

**Oracle mode** (`oracle_minutes_shares`): uses the real, actual minutes that specific historical
game's players ended up playing — a ceiling/ceiling-test only, explicitly not deployable live (the
live system cannot know tonight's actual minutes in advance).

**Predictive mode** (`predictive_minutes_shares`): takes the trailing-`lookback_games` union of
players who've recently appeared, computes each one's own trailing-average minutes, renormalizes —
the mechanism meant to mirror what the live pipeline can actually do.

**The bug that got Phase 2 disabled**, quoted in full because it's the single most consequential
finding of this project's history:

> "REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the active-player SET used to be
> `oracle_minutes_shares(...).index` — literally the real historical game's own actual attendance,
> i.e. PERFECT HINDSIGHT on who ends up playing that night. The live mechanism this claimed to
> mirror (`resolve_active_lineup`) does NOT know who ends up playing — it takes the trailing-
> `lookback_games` UNION of anyone who's recently appeared, then excludes only players an injury
> report flags. A healthy scratch/DNP-CD/unflagged load-management night is a scenario the live
> system structurally cannot exclude in advance, but the old backtest's oracle-derived active set
> silently excluded such players anyway ... measuring an easier selection problem than the live
> pipeline actually solves."

**Re-validated result after the fix, completely reversing the prior conclusion**:

| check | metric | ORIGINAL (hindsight-leaked) | RE-VALIDATED (honest) |
|---|---|---|---|
| dev | total_mae | −0.0206 REAL IMPROVEMENT | **+0.0086 REAL REGRESSION** |
| dev | margin_mae | −0.0336 REAL IMPROVEMENT | **+0.0146 REAL REGRESSION** |
| holdout-only | margin_mae | (not previously checked) | **+0.0181 REAL REGRESSION** |

**Oracle mode itself is unaffected and still confirms real signal exists in principle**
(total_mae −0.0205, margin_mae −0.0358, both real improvements, reconfirmed after an unrelated
season-scoping fix to `_build_team_history` was also applied retroactively) — the problem is
specifically that the deployable, predictive minutes-share ESTIMATION doesn't capture that signal
net of its own noise, not that lineup-awareness is a dead end conceptually.

**Flagged to the user rather than silently reverted** — disabled per explicit user decision
(`INCLUDE_LINEUP_ADJUSTMENT = False`), which skips the RAPM-lite fit computation entirely (not
computed-then-discarded) and falls through to the pre-existing team-strength-only code path that
predates Phase 2 ever being wired in. `generate_props.py` has no independent Phase 2 logic of its
own — it only ever reused `generate_predictions.py`'s output as its points macro-anchor, so
disabling Phase 2 there propagates automatically.

### 6.4 What would be needed to safely re-enable Phase 2

Per the code's own documentation: a **genuinely improved minutes-projection mechanism** must be
built and pass the full dev → holdout confirmatory-veto pipeline again — not just re-trusting the
old, hindsight-leaked backtest. One additional asymmetry worth weighing when judging any new
mechanism: the honest re-validated backtest has ZERO injury information (RotoWire has no historical
archive, §1.3), while the live pipeline DOES have real, if partial, injury exclusion — meaning the
current honest backtest is likely a *pessimistic* lower bound on live performance, not a precise
estimate of it. Mechanically, re-enabling is a one-line flag flip in two files once a new mechanism
clears validation — nothing structural needs to change.

### 6.5 Rest/back-to-back (`rest_schedule.py`) — a related, separately-tested, NOT-adopted signal

Not part of RAPM/lineup adjustment at all — a standalone diagnostic tool
(`correlate_residual_with_rest`) that correlates a team-side's RESIDUAL (never raw score, to avoid
recovering "good teams tend to have better rest management" as a confound) against `rest_days`/
`is_b2b`. A materially more careful walk-forward version of this exact idea was built and tested
this session (§8) and found real-but-not-adoptable; this simpler diagnostic remains unwired,
informational only.

---

## 7. Validation framework

### 7.1 Dev/holdout split and the confirmatory-veto protocol

`final_holdout_check.py` freezes `DEV_MAX_SEASON = 2024` as a **literal constant**, deliberately
never recomputed from a live API call at runtime — recomputing it on every import would silently
shift games from holdout into dev as future seasons complete, quietly widening the "already looked
at" set without anyone deciding that on purpose. Dev = seasons 2015–2023 inclusive (10,737 games);
holdout = seasons 2024–2025 (the two most recently completed full seasons as of when this was
frozen).

**The protocol**: holdout is read **exactly once per genuinely new model configuration**, only to
VETO a dev-confirmed improvement — never to rank candidates or pick between configurations. Every
candidate fix goes through a mandatory two-stage DEV-ONLY screen first:
- **Stage 1**: a cheap chronological slice of the most-recent dev seasons (fast, catches obviously
  bad candidates early).
- **Stage 2**: the full dev range (10,737 games), both vs. the current config AND vs. a naive floor.

Only a candidate that clears BOTH dev stages ever earns the one-time holdout read. This session
introduced an explicit **"net win" criterion** for the holdout decision: a candidate must show *no
real regression* on any key metric AND *real improvement* on at least one — distinguishing a genuine
win from a tradeoff (several historical candidates improved one metric while regressing another by a
larger amount, and were correctly NOT adopted under this rule).

`generic_holdout_confirmatory_check` computes the paired-bootstrap gap between dev and holdout
performance (not holdout performance in isolation — some gap is expected even for a genuinely good
model evaluated on a different, more recent set of games; only a REAL regression in that gap should
veto).

### 7.2 Standard paired bootstrap (`bootstrap_significance.bootstrap_compare`)

The universal significance-testing primitive: resamples real **games** (not simulated trials) with
replacement — the correct unit for comparing two deterministic point-prediction models on the same
historical games. 5,000 resamples by default, 95% percentile CI on the paired delta. Memory-batched
(a fixed `_MAX_IDX_MATRIX_BYTES` cap) so it scales safely from team-level counts (~10,000 games) to
player-level counts (hundreds of thousands of player-game rows) without materializing one giant
index matrix upfront (a real bug found and fixed earlier this project: the original version did
materialize it upfront and would thrash/swap at player-level scale).

### 7.3 Block/cluster bootstrap (`block_bootstrap_compare`) — NEW this session

The standard bootstrap above implicitly assumes every game-row is an independent draw. This is
questionable if a team's residuals are *serially correlated* across its own consecutive games (e.g.
a systematic rating mispricing that persists for a few games before the walk-forward fit catches
up) — a naive per-row bootstrap would understate the true sampling variability in that case,
inflating apparent significance (a real Type-I-error risk). `block_bootstrap_compare` resamples
whole BLOCKS (e.g. one block per team) with replacement instead of individual rows — a block drawn
by a resample contributes every one of its rows together, preserving whatever real within-block
correlation exists rather than assuming it away. Regression-tested with synthetic data proving the
exact failure mode it exists to catch: 20 blocks × 20 rows, each block sharing a random "delta bias"
(so within-block deltas are correlated) but a TRUE average delta of exactly zero across all 20
blocks — the naive bootstrap is fooled into a spurious "REAL IMPROVEMENT" verdict; the block
bootstrap correctly shows NOISE, with a CI more than 2× wider. Applied to the flagship Phase 1
result with `block_col="team"` (only 30 blocks — one per team across the entire 9-season dev range,
about as conservative a clustering as reasonably justified): both total_mae (delta −0.2323, CI
(−0.2654, −0.2001)) and margin_mae (delta −0.0790, CI (−0.0927, −0.0663)) remain clearly REAL
IMPROVEMENT even under this much stricter test.

### 7.4 Rolling-window walk-forward backtest (`rolling_window_backtest.py`) — NEW this session

Because this project's rate models are already fully walk-forward-safe end to end (every prediction
across the whole dev range already uses only strictly-prior games — there is no leakage to guard
against by literally re-fitting per rolling origin the way a non-walk-forward model would need),
this tool is a REPORTING enhancement, not a re-fitting one: it slices an already-computed prediction
set by season and runs an independent bootstrap comparison per season, instead of one pooled
aggregate — surfacing whether a candidate's apparent win is stable across time or hidden inside a
favorable average. Applied to the 3 adopted Phase 1 wins across all 8 dev seasons: **zero real
regressions** on either total_mae or margin_mae in any season (6/8 and 7/8 seasons respectively show
real improvement, the rest show noise — never a regression).

### 7.5 Multiple-comparisons correction

This session ran roughly 9 independent one-time confirmatory holdout reads. A family-wise Bonferroni
correction (per-test alpha = 0.05/9 ≈ 0.00556, a 99.444% CI) was applied retroactively to every
ADOPTED win: all 3 Phase 1 levers and all 4 checked `own_halflife_games` team_stat_rates categories
survive; `cross_season_weight`'s weakest two results (stl/blk, already the smallest-magnitude of that
group at the standard 95% bar) do not survive this much stricter threshold — still directionally
correct (not reversed to a regression), just not statistically distinguishable from zero under a
deliberately conservative worst-case bound. Documented honestly as the least statistically robust of
this session's adopted changes, not silently smoothed over.

### 7.6 A real methodological trap caught and fixed mid-session

Re-testing an OLDER adopted lever (the Sec29 joint margin fix) under the Bonferroni correction
initially showed its margin_mae result **not surviving** — but this was a confound, not a real
finding: the re-check script implicitly let the TWO LATER-adopted levers (`cross_season_weight`,
`own_halflife_games`, neither of which existed when the joint margin fix was originally tested)
apply to both comparison arms equally, artificially narrowing the measured gap. Fixed by explicitly
pinning both later levers OFF on both arms to faithfully reproduce the original test conditions —
after the fix, the delta exactly matched the originally-documented number and cleared the stricter
bar cleanly. **Lesson generalized**: any later re-test of an isolated historical result must
explicitly pin every lever that didn't exist at the time of the original test, not just accept
whatever the current module-level defaults happen to be.

---

## 8. Honest negative results (closed investigations, not silently abandoned)

- **Home-court EWMA halflife**: swept 100–1600 games on top of the fully-adopted current config —
  every value statistically indistinguishable from every other. The home-court effect itself is too
  small relative to noise for this parameter to matter at any value tested. No holdout read spent.
- **3PT-makes detrend-then-retrend**: the lever's own stated premise (a continuing 3PT-volume trend
  through the dev range, mirroring margin's real scoring-era drift) did not survive a direct data
  check — 3PT attempts/player-game actually PLATEAU mid-dev-range (2019–2023: 3.23, 3.25, 3.32,
  3.25, 3.27), contradicting the premise before any validation effort was spent building the
  correction.
- **Back-to-back adjustment**: a genuinely strong DIAGNOSTIC correlation (p=8.0×10⁻⁷) between
  back-to-back status and residual error — but a proper walk-forward-safe ADOPTION test (built with
  an explicit sequential loop specifically to avoid a same-game leak trap a naive vectorized
  shift/expanding approach would have hit) found it does NOT survive out-of-sample: the correction
  doesn't help net of the noise it introduces. Kept as the clearest example in this codebase of the
  distinction between "a residual correlates with X" (diagnostic) and "correcting for X actually
  helps out-of-sample" (adoption claim) — these are not the same question.
- **Opponent-familiarity/rematch effects**: a prior meeting's margin correlates with the CURRENT
  game's raw actual margin at r=0.169 (p=9.8×10⁻⁴⁵) — but correlates with the model's own RESIDUAL at
  only r=−0.025 (p=0.039, borderline given how many tests ran this session) — meaning the ratings
  model already absorbs nearly all of a prior meeting's real information. Not pursued further; no
  validation effort spent building a correction on a diagnostic this weak.
- **Player-level own-history recency-weighting**: skipped without building new infrastructure, after
  finding the identical family of idea already tested and found actively HARMFUL (~40% worse) for
  shooting make-rates, and independently confirmed harmful for rebound-conversion rate too — both
  are stable, persistent SKILL metrics where the existing expanding-shrinkage (long memory) approach
  is already known to beat any recency-weighted alternative. Team-level RATINGS (§2.2's win) and
  player-level SKILL RATES are analogous-looking but empirically opposite cases — exactly the
  "don't assume by analogy" lesson this project keeps re-confirming.
- **OREB team-level anchoring**: see §3 — four independent mechanisms all converge to the same
  ceiling; formally closed as a structural limitation, not an open parameter search.

---

## 9. Production infrastructure

### 9.1 Live pipeline flow

**`generate_predictions.py run(game_date)`**: `refresh_all_data()` → resolve `target_season` purely
from `game_date` via `season_for_date` (never wall-clock "now" — a past bug: the live pipeline once
resolved "current season" from real today even for a historical backtest date, silently pulling a
star's trailing minutes from whatever's most recent as of the real today instead of the target
date) → build and rate the team-game log, filtered strictly to `gameDate < game_date` → resolve each
team's latest rating and the latest walk-forward home-court multiplier (`_latest_home_court_mult`,
§2.3) → build team history and load the current roster (both season-restricted, see below/§3) → (skipped
entirely while `INCLUDE_LINEUP_ADJUSTMENT = False`) fit RAPM-lite → fit the score-distribution
residual/variance/correlation model → per game, `project_game(...)` → win probability / 80% spread
interval if the distribution fit succeeded → write `data/processed/daily_predictions_{date}.parquet`.

A real bug lived here until 2026-08-01: `run()` called `project_game(..., home_court_mult=1.0)`
**literally hardcoded**, discarding the entire empirically-validated home-court effect — two equally
-rated teams got an identical projected score regardless of who was home, and the score-distribution
layer's win-probability/interval math (fit on residuals that DO assume the correction) was applied
to a systematically biased mean. Fixed via `_latest_home_court_mult`, which fits
`fit_home_court_walk_forward` on the same already-filtered team log and takes its most recent value.

**`generate_props.py run(game_date)`**: calls `generate_predictions.run(game_date)` **directly** for
team totals — "props and game predictions can never silently disagree on a team's total, and this
file has no separate lineup-adjustment logic of its own." Then: refreshes data, resolves the active
roster per team via `resolve_active_lineup` (the trade-contamination guard below applies), builds all
six per-player rate-model snapshots plus team_stat_rates totals, applies matchup difficulty (gated
on `season >= MATCHUP_DATA_START_SEASON` and a minutes-trust floor), runs the macro-anchor +
micro-reallocation composition rule, fits per-category predictive distributions, and writes
`data/processed/daily_props_{date}.parquet` — one row per `(gameId, playerId, stat_name)` with
`proj_mean, proj_var, family, family_param, minutes_tier_tag, anchored_to_team_total,
matchup_adjusted`.

**`active_roster.resolve_active_lineup`**: trailing-rotation-union over the last
`MINUTES_LOOKBACK_GAMES = 10` team games (every player who appeared in ANY of them, not just the
most recent lineup), averaged and renormalized, minus anyone RotoWire flags Out/Doubtful, minus
anyone not on the CURRENT roster snapshot (`current_roster_ids`, a trade/waiver fix — a real
bug where a traded/waived player kept contributing a phantom minutes share and a full phantom prop
row for up to 10 games after leaving).

**`export_to_site_db.py`** is the only boundary between the model and the public site — reads the
two parquet files above (never model internals) and upserts into a shared multi-sport Postgres DB.
No fixed betting line exists anywhere in the pipeline's own output — `over_prob`/`under_prob`/`line`
are left NULL; `proj_mean` (and now `proj_var`/`family`/`family_param`) are the only populated
numeric fields.

### 9.2 The proxy mechanism (`nba_proxy.py`)

```python
def configure_proxy() -> None:
    NBAStatsHTTP.set_session(_curl_cffi_requests.Session(impersonate="chrome124"))
    proxy_url = os.environ.get("NBA_STATS_PROXY_URL")
    if proxy_url:
        import nba_api.library.http as _nba_http
        _nba_http.PROXY = proxy_url
```

Two independently-necessary fixes bundled here, confirmed via a systematic live test sweep from
inside the Railway container:
1. **A non-blocked egress IP** — Railway's own IP is confirmed blanket-blocked by stats.nba.com (a
   silent TCP/TLS-connects-but-HTTP-never-arrives black hole, a datacenter-IP reputation block, not
   a rate-limit or header issue). A residential proxy is needed.
   **NOT SUFFICIENT ALONE** — tested through 20 different residential-proxy backend IPs, plain
   `requests`/curl still gets the identical silent black hole.
2. **A browser-realistic TLS/HTTP fingerprint** — `curl_cffi` (`impersonate="chrome124"`) through the
   SAME residential IP succeeds with a full real response, confirming stats.nba.com's WAF also
   fingerprints the handshake itself, independent of IP reputation. **Both are independently
   required; neither alone is sufficient.**

`configure_proxy()` is called unconditionally at the top of both live entry points' `__main__`
blocks — safe to call with `NBA_STATS_PROXY_URL` unset (e.g. local dev), since `curl_cffi`'s session
behaves like a normal direct client with `proxies=None`.

### 9.3 Deployment configuration (`railway.toml`)

```toml
[deploy]
startCommand = "python -m src.pipeline.generate_predictions && python -m src.pipeline.generate_props && python -m src.pipeline.export_to_site_db && curl ... /internal/notify/nba || true"
cronSchedule = "0 1,14 * * *"
restartPolicyType = "NEVER"
```

**Dual-run schedule**, fires twice daily: 01:00 UTC (~9pm ET the night before, EDT) populates
tomorrow's slate the moment the site's displayed day flips at midnight Eastern; 14:00 UTC (~10am ET)
refreshes it. **A known, documented, unresolved DST gap**: these literal UTC values are EDT-accurate
— during EST months (Nov–Mar, squarely inside the NBA season), each fires one Eastern hour earlier
than intended. Low severity (the pipeline still runs, just at a slightly different local time), not
yet fixed.

### 9.4 Current incident (as of 2026-08-02)

`nba-worker` on Railway is **CRASHED**. Root cause, confirmed by direct observation of live Railway
logs across two consecutive deploys: stats.nba.com **blanket-blocks Railway's datacenter IP** — a
watched redeploy showed the fallback logic (fixed this session) correctly trying BOTH the newest
season (2026-27, empty, times out on every attempt) AND the prior season (2025-26, has real data,
ALSO times out identically) before genuinely raising — proving the block is blanket across every
stats.nba.com call from that IP, not specific to querying an empty/new season. A real, separate CODE
bug was found and fixed alongside this (`current_nba_season()`'s candidate-fallback loop wasn't
catching per-candidate failures, so it crashed before ever trying the fallback year at all) — that
fix is correct, deployed, and confirmed working (the fallback IS now correctly attempted), but
cannot restore service on its own since the underlying block is blanket. **The only real fix is
activating the already-built (§9.2) residential-proxy support with real credentials from a
residential-proxy provider AND ensuring `curl_cffi` fingerprint impersonation is used (confirmed
both are independently necessary)** — obtaining those credentials requires a third-party account and
payment that must happen on the user's end, not something this assistant can do directly. Once a
real `http://username:password@host:port` URL exists, setting it as the `NBA_STATS_PROXY_URL`
Railway environment variable is a regular, low-risk config change.

### 9.5 External market benchmark (2026-08-02) — the project's first non-naive calibration point

Every validation in this project compared the model against a naive floor or its own prior config —
never against the real market. Partial infrastructure for this already existed, dormant:
`fetch_odds_sbro.py` (built early in the project as a "nice-to-have," §1) had cached real closing
lines/totals/moneylines for 2015-16 through 2022-23, never read by any validation script. Found and
fixed a real sign-convention documentation error while wiring it up: `homeSpreadClose` is the
market's own predicted home margin directly (positive = home favored), confirmed empirically
(r=0.445 with real outcomes, mean matches the real average home-court margin almost exactly) — the
opposite of what the ingest script's own docstring claimed.

New script `validate_vs_market.py` matches 8,841 dev-range model predictions to a real closing line
(a NaN-poisons-the-bootstrap trap hit and fixed along the way — `bootstrap_compare`'s vectorized
mean doesn't skip NaN the way pandas does). **Result: a real, significant gap on every metric**:

| metric | model | market | delta | verdict |
|---|---|---|---|---|
| margin_mae | 10.289 | 9.895 | +0.394 | REAL REGRESSION |
| total_mae | 14.758 | 14.145 | +0.613 | REAL REGRESSION |
| straight-up accuracy | 65.15% | 67.45% | −2.30pp | REAL REGRESSION |

Not a red flag — beating closing lines is a famously hard bar — but this is the first honest external
reference point this project has ever had, and it materially changes what "the model performs well"
should be taken to mean (comfortably ahead of naive, genuinely behind the real market). Cannot be
extended to the 2024-2025 holdout range without new (paid) data, since SBRO's archive stops at
2022-23. See `MODEL_DOCUMENTATION.md` Sec44.4 for the complete writeup including the sign-convention
fix.

---

## 10. Gaps checklist — prioritized, for next-season prep

### P0 — blocking or near-blocking, fix before/at 2026-27 tip-off

1. **Railway `nba-worker` is CRASHED right now** (§9.4). Nothing else in this checklist matters if
   the live pipeline can't run at all. Needs real residential-proxy credentials (user action) +
   confirmation that `curl_cffi` fingerprint impersonation is actually wired through every call path
   (it's set once via `configure_proxy()`'s shared session swap, so this should already cover every
   `nba_api` call — but this has never been confirmed end-to-end against the real block, only
   individually against a test sweep).
2. **`refresh_data.py`'s current-season fetches were fixed this session but never actually exercised
   against a genuinely NEW season boundary.** The fix (player-track/matchups/rosters now fetched for
   the current season, not just historical backfill) was validated by code review and by the fact
   the 2025-26 season's files already existed from a prior manual backfill — it has not yet been
   watched succeed against a real "day 1 of 2026-27, zero cached files" scenario. **Recommend**: the
   first live run after 2026-27's opening night should be manually watched (logs, not just
   assumed-successful cron) to confirm player-track/matchup/roster data actually populates for the
   new season, not silently NaN.
3. **`MATCHUP_DATA_START_SEASON = 2017` frozen constant is fine and doesn't need touching** — but
   worth explicitly re-confirming `BoxScoreMatchupsV3`/`BoxScoreDefensiveV2` still work identically
   for 2026-27 specifically once real games exist (an untested but low-risk assumption that this
   season, like every one since 2017-18, will behave the same way).
4. **DST cron mistiming** (§9.3): `railway.toml`'s `cronSchedule = "0 1,14 * * *"` is EDT-accurate
   only — fires one Eastern hour early throughout EST months (Nov–Mar, i.e. most of a real NBA
   season). Cheap, mechanical fix (two seasonal cron expressions, or a UTC-offset-aware scheduler)
   that's been sitting unaddressed; worth just fixing rather than re-deferring again.

### P1 — real, open modeling questions (not blocking, but the most valuable next research)

5. **Phase 1's margin/scoring-era-drift — REFRAMED (§2.6), downgraded from "urgent open problem" to
   "worth one clean follow-up, not a fire."** The normalization diagnostic (§2.6) found the raw
   margin_mae trend tracks almost exactly with rising REALIZED margin variance (r=0.846 for true
   variance vs. r=0.828 for model MAE, against season), while the model's performance NORMALIZED by
   either the naive floor or the season's own true variance shows no degradation at all — holdout is
   if anything the best-normalized stretch in the whole range. **Recommended next step, in order**:
   (a) formalize the normalization diagnostic into a real `validate_*.py` script rather than leaving
   it ad hoc; (b) if it holds up, the combine log5-slope finding (§2.4) is the more promising concrete
   lever — a fitted damping coefficient on the rating deviation terms (slopes measured at 0.916/0.907
   vs. the assumed 1.0) is a genuinely new, well-motivated, already-diagnosed mechanism, distinct from
   the three shrinkage-primitive levers already tried; (c) only pursue a dedicated era/quality-spread
   regressor if (a) and (b) don't close the gap — don't jump straight to new-mechanism-building without
   re-checking the metric itself first.
6. **OREB team-level anchoring is closed, not open** — four independent mechanisms converged to the
   exact same "statistical tie with naive" ceiling. Don't re-open this with a fifth parameter sweep
   on `add_walk_forward_rate`'s existing knobs. **One concrete structurally-different primitive worth
   trying, from external review**: the current approach fits team OREB and opponent DREB as
   independent FOR/AGAINST count rates, conflating shot volume, shot accuracy, and rebounding skill
   — the first two of which this system already projects separately and better via the pace×rating
   and scoring-rate machinery. Try instead: `projected_team_OREB = projected_misses ×
   projected_OREB_share`, where `projected_misses` comes directly from the existing attempt/make-rate
   projections (already validated), and `OREB_share` is a walk-forward FOR/AGAINST rate (own OREB%
   vs. opponent's DREB%-allowed, combined via the same log5-style ratio idiom as everything else) —
   removing miss-VOLUME noise from the rebounding signal specifically, and enforcing game-level
   coherence (home OREB + away DREB should reconcile to home's own projected misses) that the current
   independent-fit approach doesn't guarantee. A genuinely different primitive, not a parameter
   choice — worth the two-stage-then-holdout protocol if pursued.
7. **Phase 2 (RAPM-lite lineup adjustment) is disabled and should stay disabled** until a genuinely
   improved minutes-projection mechanism is built (§6.4) — the core signal (oracle mode) is real and
   confirmed; the deployable minutes-estimation piece specifically is the gap. This is real,
   substantial modeling work (not a quick parameter tune) if pursued.
8. **`cross_season_weight`'s stl/blk results for team_stat_rates are the least statistically robust
   adopted change this session** (§7.5) — they don't survive a conservative Bonferroni correction,
   though they're not reversed to a regression either. As next season's real games accumulate, this
   is a natural, cheap thing to re-check with a larger sample before assuming it's settled forever.
9. **A real market benchmark now exists for dev range only (§2.5/§9.5)** — the model shows a real,
   quantified gap vs. real closing lines (margin_mae +0.39, total_mae +0.61, SU accuracy −2.3pp), the
   project's first external (non-naive) reference point. Cannot be extended to the 2024-2025 holdout
   range without a new paid historical-odds source (SBRO stops at 2022-23) — not pursued unless
   closing the market gap becomes an explicit project goal, but worth keeping in mind as the honest
   current ceiling estimate whenever "how good is this model" comes up.

### P2 — known-tested, currently low-value (don't re-litigate without new information)

9. Home-court EWMA halflife (`fit_home_court_walk_forward`'s default 400.0): swept 100–1600, no
   detectable difference at any value. Leave as-is; the effect itself is too small for this
   parameter to matter.
10. Garbage-time downweight factor (`DOWNWEIGHT_FACTOR = 0.25`) and the margin-threshold shape
    (`25.0 − 15.0×elapsed_fraction`): both un-calibrated placeholders, but currently moot — they only
    affect the RAPM-lite fit, and Phase 2 is disabled. Revisit only alongside item 7.
11. `MIN_MATCHUP_MINUTES_TO_TRUST` and `player_priors.py`'s RAPM ridge schedule
    (`BASE_LAMBDA`/`MIN_LAMBDA_FRACTION`/`EXPERIENCE_HALFLIFE_GAMES`): both explicitly flagged in
    their own code as un-tuned placeholders pending real-slate testing. Low priority while Phase 2
    is disabled and matchup difficulty already clears its own validation gate without them being
    precisely tuned.
12. Rest/back-to-back adjustment: a genuinely strong diagnostic (p=8×10⁻⁷) that does not survive a
    proper walk-forward adoption test (§8). Don't re-attempt without a structurally different
    mechanism than "flat additive correction on B2B games" — that specific family has been tried and
    found wanting.

### P3 — new-season-specific hygiene (cheap, mechanical, worth doing once at tip-off)

13. **Roster churn**: `CommonTeamRoster` is fetched fresh (forced) every `refresh_data.py` call for
    the current season specifically for this reason — confirm this actually captures 2026-27's
    real offseason trades/draft picks/free-agency moves once rosters are set, not stale preseason
    data.
14. **True rookies with zero history**: every rate model already handles this correctly by design
    (NaN until enough exposure accumulates, no assumed prior beyond the shared league-average
    shrinkage target) — nothing to build, just worth remembering this is intentional, not a gap, so
    a rookie's early-season NaN outputs aren't mistaken for a bug.
15. **Team relocations/rebrands**: none have occurred in this dataset's history (2015-16–present),
    unlike the NHL sibling project which had to handle a real one — currently unmonitored (no
    explicit test or handling exists) since it's never come up. Worth a brief sanity check if any
    franchise news breaks before the season, otherwise not worth building for a hypothetical.
16. **RotoWire/current-roster live-only dependencies (§1.3) are a permanent, accepted limitation**,
    not something to "fix" — just worth remembering neither injury exclusion nor trade-contamination
    guarding can ever be backtested on a historical date, only exercised live.

### What NOT to build (explicitly, so it isn't re-proposed without new information)

- Joint/correlated cross-category variance modeling for parlay-style combo props — no confirmed
  product need; the current independently-fit, calibration-checked per-category distributions are
  already methodologically sound for standalone props.
- Player-level "recency-weight own skill-rate history" — already found actively harmful (twice,
  independently) for exactly this family of stat.
- Any further OREB team-level anchoring parameter sweep on the existing `add_walk_forward_rate`
  primitive — closed, see item 6.
