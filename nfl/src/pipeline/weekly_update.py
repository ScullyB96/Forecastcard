"""Automated weekly pipeline: refresh data -> rebuild every rating engine ->
auto-detect the next unplayed week -> predict margin/total and player props
for it. Run this once a week (or on demand) and it does the right thing
whether the season hasn't started yet or is midway through.

Replaces the hardcoded "Week 1, 2026" scripts (predict_2026.py,
predict_props_2026.py) for ongoing use -- those remain useful as references
and for the one-time preseason forecast, but this is what should run every
week once real games start, since it reads the schedule natively (confirmed
available in full from nfl_data_py once the league releases it) instead of
depending on a manually re-extracted PDF.

Design choice: refresh ALL seasons with force=True every run rather than
attempting incremental updates. A few minutes of extra runtime buys a lot of
robustness -- no partial-file merge logic to get wrong, and it's the same
well-tested code path fetch.py already uses. import_weekly_data() (nfl_data_py's
own weekly aggregate) is skipped deliberately: it lags the season by design
(confirmed multiple times this project), so weekly stats are always derived
from our own PBP data instead (build_weekly_stats_from_pbp.py).
"""

import datetime as _dt

import numpy as np
import pandas as pd

from src.ingest.build_weekly_stats_from_pbp import build_weekly_stats
from src.ingest.fetch import fetch_draft_picks, fetch_injuries, fetch_pbp, fetch_schedules, fetch_snap_counts, fetch_weekly_rosters
from src.ingest.name_matching import (
    build_lastname_pos_to_id,
    build_lastname_team_pos_to_id,
    build_name_to_id,
    build_qb_name_to_id,
    find_qb_changes,
    norm_name,
)
from src.models.game_environment import (
    PassRateEngine,
    apply_game_script,
    build_full_game_pass_rate,
    build_pass_rate_table,
    fit_game_script,
)
from src.models.drive_transitions import implied_total_quantile_bin
from src.models.game_simulator import build_simulator_for_season_range
from src.models.injury_adjustment import JOINT_COEFS_FORWARD, apply_joint_adjustment, compute_injury_flags
from src.models.injury_reallocation import out_player_ids_for_team, reallocate_shares, rookie_fallback_rb_rates
from src.models.rookie_prior import build_rookie_season_shares, fit_draft_capital_prior, rookie_fallback_from_draft_capital
from src.models.player_usage import ShareEngine, TdRateEngine, build_player_week_shares, fit_empirical_bayes_prior_weight
from src.models.qb_adjustment import QbRatingEngine, build_qb_week_table, build_starter_sequence
from src.models.qb_passing_stats import build_qb_passing_engines, presumed_starter_qb_id, project_passing_stats
from src.models.ratings import PowerRatingEngine, build_dataset
from src.models.weather_adjustment import apply_weather_adjustment, fit_weather_adjustment
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

CALIBRATION_SEASONS = {2022, 2023, 2024, 2025}  # most recent, non-COVID-anomalous -- see predict_2026.py
# LEAGUE_AVG_PLAYS is computed fresh each run below (review round 2, #4.2) -- see that comment.
# points per unit of QB rating gap. Refit 2026-07 against the BLEND residual
# (this adjustment is applied on top of the market-blended margin below, not
# raw Layer-1 -- see injury_adjustment.py's module docstring for why fitting
# against the pure-Layer-1 residual, as the original 6.616 was, double-counts
# information the market already prices in). New value: 2.970, pooled
# 8-season era-consistent-residual fit, t=+1.52, refit jointly with the
# symmetric CB injury coefficient (v3) in injury_adjustment.py.
# Named _MARKET (review round 2, #1.3): predict_2026.py has its own SWAP_B_LAYER1=6.616,
# same physical quantity fit against a different residual basis -- same-named constants
# with different meanings is exactly the bug class np.polyfit's arg-order bit this project
# three times, so the two are named for their basis rather than left as bare "SWAP_B" in
# both files. Do NOT sync the two values; each is correct for its own file's residual basis.
# Frozen, not auto-refit every run -- same reasoning as JOINT_COEFS_FORWARD's own comment in
# injury_adjustment.py: this took real, multi-round scrutiny to arrive at, and a silent
# weekly auto-recompute would re-run that judgment-laden process with no human review.
SWAP_B_MARKET = 2.970

# Win-probability calibration for the game simulator's recentered win prob (review round 4's
# recentering fix, re-validated + this correction added 2026-08 -- see
# validate_win_prob_calibration.py). Recentering was verified exact by construction and,
# separately, shown to genuinely improve Brier/CRPS over the raw simulator on TEST -- but the
# recentered probability itself still shows a real, out-of-sample-confirmed miscalibration
# pattern (overconfident near 0.5, underconfident in its most lopsided games). A strict
# CALIB=2022-2023 -> HOLDOUT=2024-2025 split (never overlapping either the TRAIN=2018-2021
# drive pools or each other) found linear-refit narrowly beat isotonic and Beta calibration on
# HOLDOUT Brier (0.2064 vs 0.2065 vs 0.2067, vs. 0.2084 uncalibrated) -- same decision rule as
# TD_PROB_CALIB (§9.4): linear ships since nothing beat it outright. FROZEN, not auto-refit
# every run, unlike TD_PROB_CALIB -- doing this safely without leakage would require
# re-simulating years of history walk-forward every pipeline run (this simulator's drive pools
# are a single fixed TRAIN-period pool, not internally walk-forward like the share/rate
# engines TD_PROB_CALIB's inputs come from), not worth the runtime cost for a ~1% Brier gain.
# Fit on all of 2022-2025 (the strict split above already answered the out-of-sample question).
WIN_PROB_CALIB_A = -0.1467
WIN_PROB_CALIB_B = 1.2790

# TD-probability calibration: TD_PROB_CALIB_A/B is now AUTO-REFIT LINEAR every pipeline run
# (review round 4, #7; see the computation below, near calib_bucket_means) -- not a frozen
# literal, and not isotonic (round 3's choice, reverted -- isotonic underperformed a strict
# walk-forward comparison and was kept anyway on resubstitution-flavored evidence, the same
# evidence structure this project correctly rejected for the CB coefficient). Refitting the
# linear map every run from real historical (raw_td_prob, actual_td) pairs retires the
# stale-coupling bug class (R0.1) just as well as isotonic would have, with better
# demonstrated out-of-sample behavior. predict_props_2026.py keeps its own frozen linear
# TD_PROB_CALIB_A/B=(0.046,0.769), paired with that script's own frozen swept
# prior_weight=30 -- internally self-consistent as a historical reference, same pattern as
# SWAP_B_LAYER1.


def american_odds_from_prob(p: float) -> int:
    """Standard American-odds conversion from an implied win probability.
    p>=0.5 -> favorite (negative); p<0.5 -> underdog (positive). Clipped away
    from the exact 0/1 boundary to avoid a division blowup on a degenerate
    simulated win probability."""
    p = min(max(p, 1e-4), 1 - 1e-4)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def current_nfl_season() -> int:
    """The NFL season year is the year it *starts* in (a 2026 season runs
    Sep 2026 - Feb 2027). Games Jan-Jun belong to the PRIOR season year."""
    today = _dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def _fetch_with_fallback(fetch_fn, seasons: list[int], name: str) -> list[int]:
    """Try the full season range; if the current (possibly-not-yet-started)
    season has no published data yet, some nfl_data_py importers (import_injuries,
    import_snap_counts) throw instead of degrading gracefully. Fall back to
    everything except the newest season in that case -- expected behavior
    before that season's data starts getting published week by week."""
    try:
        fetch_fn(seasons, force=True)
        return seasons
    except Exception as e:
        fallback = seasons[:-1]
        print(f"  {name}: no data yet for {seasons[-1]} ({e}); using {min(fallback)}-{fallback[-1]} instead")
        fetch_fn(fallback, force=True)
        return fallback


def refresh_all_data(seasons: list[int]) -> dict[str, list[int]]:
    covered = {}
    fetch_schedules(seasons, force=True)  # always has the upcoming season's schedule once released
    covered["schedules"] = seasons
    covered["pbp"] = _fetch_with_fallback(fetch_pbp, seasons, "pbp")
    covered["weekly_rosters"] = _fetch_with_fallback(fetch_weekly_rosters, seasons, "weekly_rosters")
    covered["injuries"] = _fetch_with_fallback(fetch_injuries, seasons, "injuries")
    covered["snap_counts"] = _fetch_with_fallback(fetch_snap_counts, seasons, "snap_counts")
    covered["draft_picks"] = _fetch_with_fallback(fetch_draft_picks, seasons, "draft_picks")

    # our own PBP-derived weekly stats, not nfl_data_py's import_weekly_data (lags the season)
    pbp_seasons = covered["pbp"]
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet",
                          columns=["season", "week", "season_type", "posteam", "play_type", "receiver_player_id",
                                   "rusher_player_id", "passer_player_id", "epa", "complete_pass", "receiving_yards",
                                   "rushing_yards", "pass_touchdown", "rush_touchdown", "passing_yards", "interception",
                                   "air_yards"])
    ros_seasons = covered["weekly_rosters"]
    rosters = pd.read_parquet(DATA_RAW / f"weekly_rosters_{min(ros_seasons)}_{max(ros_seasons)}.parquet")
    weekly = build_weekly_stats(pbp, rosters)
    weekly.to_parquet(DATA_RAW / f"weekly_player_stats_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet", index=False)
    covered["weekly_player_stats"] = pbp_seasons
    return covered


def find_next_week(schedules: pd.DataFrame, season: int) -> tuple[int | None, int | None]:
    """Returns (season, week) of the next game with no score yet, or
    (None, None) if the whole loaded schedule is already played."""
    reg = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    upcoming = reg[reg["home_score"].isna()]
    if upcoming.empty:
        return None, None
    return season, int(upcoming["week"].min())


def fit_calibration(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    return fit_linear(x, y)


if __name__ == "__main__":
    season = current_nfl_season()
    seasons = list(range(2016, season + 1))
    print(f"=== weekly pipeline run: target season {season} ===")

    print("refreshing raw data...")
    covered = refresh_all_data(seasons)

    schedules = pd.read_parquet(DATA_RAW / f"schedules_{min(seasons)}_{max(seasons)}.parquet")
    pred_season, pred_week = find_next_week(schedules, season)
    if pred_week is None:
        print(f"no unplayed games found for season {season} in the loaded schedule -- nothing to predict yet.")
        raise SystemExit(0)
    print(f"next unplayed week: {pred_season} week {pred_week}")

    # --- rebuild every engine, walking forward through everything real we have ---
    print("rebuilding Layer 1 power ratings...")
    raw = build_dataset(covered["pbp"], schedule_seasons=seasons)
    matchups = raw.rename(columns={"home_team_x": "team_a", "away_team_x": "team_b"})
    rating_engine = PowerRatingEngine()
    rated = rating_engine.run_walk_forward(matchups)
    rating_engine._maybe_new_season(pred_season)  # preseason mean-reversion if this is week 1

    train = rated[rated["season"].isin(CALIBRATION_SEASONS)]
    base_a, base_b = fit_calibration(train["pregame_rating_diff"], train["actual_margin"])

    reg_sched = schedules[schedules["game_type"] == "REG"]
    team_roof = (
        reg_sched.sort_values(["season", "week"]).groupby("home_team").last()["roof"]
        .isin(["dome", "closed"]).astype(float).to_dict()
    )
    # `rated` (built two lines above via the exact same PowerRatingEngine
    # walk-forward this file also uses) already carries everything the
    # totals-calibration step below needs -- home_score/away_score/game_id
    # from build_dataset's schedule merge, pregame_total_signal from
    # run_walk_forward. Previously this re-read a `layer1_games_with_
    # ratings.parquet` file that's only ever written by ratings.py's own
    # standalone `__main__` block -- a file that never existed on a fresh
    # Railway deploy (nothing here calls ratings.py), which crashed every
    # real run with a FileNotFoundError. Reusing the in-memory frame
    # removes that phantom dependency entirely instead of requiring a
    # separate manual bootstrap step.
    layer1_full = rated.copy()
    layer1_full["actual_total"] = layer1_full["home_score"] + layer1_full["away_score"]
    roof_lookup = reg_sched[["game_id", "roof"]].assign(is_indoor=lambda d: d["roof"].isin(["dome", "closed"]).astype(float))
    layer1_full = layer1_full.merge(roof_lookup[["game_id", "is_indoor"]], on="game_id", how="left")
    total_train = layer1_full[layer1_full["season"].isin(CALIBRATION_SEASONS)]
    X_train = np.column_stack([np.ones(len(total_train)), total_train["pregame_total_signal"], total_train["is_indoor"]])
    total_coefs, *_ = np.linalg.lstsq(X_train, total_train["actual_total"].to_numpy(), rcond=None)

    # MARGIN: no blend (review round 1, #1.2/#1.3). A proper scoring-panel
    # comparison (ATS%, CRPS, signed bias with bootstrap CI -- see
    # src/models/scoring.py) on TEST=2022-2025, matching this exact
    # growing-window methodology, found the blend's ATS win rate against the
    # closing line was 49.6% (95% CI [46.6%, 52.6%], indistinguishable from a
    # coin flip and below the ~52.4% breakeven against standard -110 vig),
    # AND the blend carries a real, statistically distinguishable negative
    # bias (-0.99, CI excludes 0) that using the market line alone does not
    # (-0.57, CI includes 0) -- a direct consequence of blending in our own
    # model's own real bias. The market line is used directly as the margin
    # base now, with the validated QB-swap/injury adjustments (Phase 0.1,
    # refit against this exact residual so they aren't double-counting the
    # same information) layered on top. base_a/base_b (Layer 1's own margin
    # calibration, above) is kept only as a fallback for the rare case a
    # market line isn't published yet for a game.
    #
    # TOTAL: blend ALSO removed (review round 2, #2.4). Round 1 kept it on
    # MAE + a single p-value alone (10.20 vs 10.19, p=0.43) -- the exact same
    # thin evidence state that hid the margin blend's harm until the full
    # panel was run. Running that same panel on totals (src/models/
    # validate_market_blend_totals.py), matching this exact growing-window
    # methodology: O/U% = 50.1% (95% CI [47.2%, 53.0%]) -- indistinguishable
    # from a coin flip and spanning well below the ~52.4% breakeven, the same
    # failure pattern that killed the margin blend (signed bias itself was
    # NOT distinguishable from 0 here, unlike margin's real negative bias --
    # but the O/U% test is the decisive one, since it's the only one that
    # actually converts to betting edge). The market total_line is now used
    # directly, with Layer 1's own total calibration (total_coefs, above)
    # kept only as a fallback for the rare case a market total isn't
    # published yet for a game.

    print("rebuilding QB ratings...")
    wps = covered["weekly_player_stats"]
    weekly = pd.read_parquet(DATA_RAW / f"weekly_player_stats_{min(wps)}_{max(wps)}.parquet")
    qb_weeks = build_qb_week_table(weekly)
    qb_starters = build_starter_sequence(schedules)
    qb_engine = QbRatingEngine()
    qb_engine.run_with_starters(qb_starters, qb_weeks)

    print("rebuilding injury flags...")
    injury_flags = compute_injury_flags()

    print("rebuilding player usage (target/carry share, TD rate)...")
    shares = build_player_week_shares(weekly)
    tgt_engine = ShareEngine()
    tgt_result = tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    carry_engine = ShareEngine()
    carry_result = carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")

    # draft-capital rookie usage prior (review #2.6): a true rookie (zero real
    # engine history) with no Clay-PDF projection available -- which is EVERY
    # WR/TE rookie today (the Clay-based fallback below only ever covered RB)
    # -- previously got zero projected volume. Validated via leave-one-season-
    # out (fit on all other seasons' rookies, predict the held-out season):
    # beats a naive position-wide-mean baseline by ~18% on WR/TE target share
    # and ~16% on RB carry share, both p<0.0001. Works automatically every
    # season (nfl_data_py's draft data refreshes every pipeline run, no manual
    # PDF re-extraction needed) and links by real gsis_id, no name-matching.
    ds = covered["draft_picks"]
    draft_picks = pd.read_parquet(DATA_RAW / f"draft_picks_{min(ds)}_{max(ds)}.parquet")
    rookie_season_shares = build_rookie_season_shares(draft_picks, weekly)
    target_share_prior = fit_draft_capital_prior(rookie_season_shares, "target_share")
    carry_share_prior = fit_draft_capital_prior(rookie_season_shares, "carry_share")
    # prior_weight fit via empirical-Bayes variance decomposition (review 2026-07
    # #2.3), replacing the old swept constants (30 for both TD rates, raised from
    # an unvalidated 15; 15 for catch rate, never validated at all). Validated:
    # the EB-fit value (much higher, ~150-210 for TD rates) tames a 5-TD/22-target
    # outlier to ~1.35x league average instead of ~2.6x, with BETTER top-decile
    # calibration, at a small aggregate-MAE cost consistent with this project's
    # own established finding that the metric is flat across a wide prior_weight
    # range -- see player_usage.py's fit_empirical_bayes_prior_weight docstring.
    involved_targets = shares[shares["targets"] > 0]
    involved_carries = shares[shares["carries"] > 0]
    career_rec_td = involved_targets.groupby("player_id").agg(touches=("targets", "sum"), hits=("receiving_tds", "sum"))
    career_rush_td = involved_carries.groupby("player_id").agg(touches=("carries", "sum"), hits=("rushing_tds", "sum"))
    career_catch = involved_targets.groupby("player_id").agg(touches=("targets", "sum"), hits=("receptions", "sum"))

    league_rec_td_rate = involved_targets["receiving_tds"].sum() / involved_targets["targets"].sum()
    league_rush_td_rate = involved_carries["rushing_tds"].sum() / involved_carries["carries"].sum()
    league_catch_rate = involved_targets["receptions"].sum() / involved_targets["targets"].sum()

    rec_td_pw = fit_empirical_bayes_prior_weight(career_rec_td["touches"], career_rec_td["hits"], league_rec_td_rate * (1 - league_rec_td_rate))["prior_weight"]
    rush_td_pw = fit_empirical_bayes_prior_weight(career_rush_td["touches"], career_rush_td["hits"], league_rush_td_rate * (1 - league_rush_td_rate))["prior_weight"]
    catch_pw = fit_empirical_bayes_prior_weight(career_catch["touches"], career_catch["hits"], league_catch_rate * (1 - league_catch_rate))["prior_weight"]
    print(f"  EB-fit prior_weight: rec_td={rec_td_pw:.1f}  rush_td={rush_td_pw:.1f}  catch_rate={catch_pw:.1f}")

    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=rec_td_pw)
    rec_td_result = rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")
    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=rush_td_pw)
    rush_td_result = rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    # yards/catch-rate: real but weaker signal than share or TD-rate (validated via
    # prior_weight sweep -- see predict_props_2026.py). Included for a complete
    # statline, not because it's a strong differentiator. ypt/ypc prior_weight
    # not yet re-derived via empirical-Bayes (continuous, not Bernoulli-shaped --
    # would need a play-level variance estimate rather than mu*(1-mu); left as
    # swept values for now, see MODEL_DOCUMENTATION.md).
    league_ypt = shares["receiving_yards"].sum() / shares["targets"].sum()
    league_ypc = shares["rushing_yards"].sum() / shares["carries"].sum()
    ypt_engine = TdRateEngine(league_rate=league_ypt, prior_weight=80.0)
    ypt_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_yards", "targets")
    ypc_engine = TdRateEngine(league_rate=league_ypc, prior_weight=300.0)
    ypc_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_yards", "carries")
    catch_rate_engine = TdRateEngine(league_rate=league_catch_rate, prior_weight=catch_pw)
    catch_rate_engine.run_walk_forward(shares[shares["targets"] > 0], "receptions", "targets")

    # aDOT (review #2.1): air_yards is already in PBP, previously unused. Reuses
    # TdRateEngine generically (air_yards/targets is just another Bayesian-shrunk
    # per-touch rate). NOT applied as a receiving-yards correction below -- an
    # initial test found a false-positive improvement (p<0.0001) that turned out
    # to be an artifact of a flawed pass-attempts proxy in the validation
    # methodology; once properly re-tested against real historical pregame
    # pass-rate predictions, aDOT added no significant point-estimate signal
    # beyond target_share+ypt (MAE 19.206->19.198, p=0.51). See
    # MODEL_DOCUMENTATION.md and player_usage.py's compute_wopr()/
    # fit_adot_calibration() docstrings for the full, corrected writeup --
    # kept here only as a building block (real per-player rate, computed
    # correctly) for potential future distribution/variance work, not as a
    # live correction.
    league_adot = shares["air_yards"].sum() / shares["targets"].sum()
    adot_engine = TdRateEngine(league_rate=league_adot, prior_weight=80.0)
    adot_engine.run_walk_forward(shares[shares["targets"] > 0], "air_yards", "targets")

    print("rebuilding QB passing statline engines (completions, yards, TDs, INTs)...")
    qb_passing_engines = build_qb_passing_engines(weekly)

    print("fitting weather adjustment (QB yards/attempt, WR/TE yards/target -- wind + temp, jointly)...")
    sched_reg_wind = schedules[schedules["game_type"] == "REG"][["game_id", "season", "week", "home_team", "away_team", "wind", "temp", "roof"]]
    wind_home = sched_reg_wind.rename(columns={"home_team": "recent_team"})[["season", "week", "recent_team", "wind", "temp", "roof"]]
    wind_away = sched_reg_wind.rename(columns={"away_team": "recent_team"})[["season", "week", "recent_team", "wind", "temp", "roof"]]
    team_game_wind = pd.concat([wind_home, wind_away])

    qb_weeks_wind = weekly[(weekly["season_type"] == "REG") & (weekly["attempts"] >= 5)].merge(
        team_game_wind, on=["season", "week", "recent_team"], how="left"
    )
    league_ypa = qb_weeks_wind["passing_yards"].sum() / qb_weeks_wind["attempts"].sum()
    ypa_wind_engine = TdRateEngine(league_rate=league_ypa, prior_weight=150.0)
    ypa_wind_result = ypa_wind_engine.run_walk_forward(qb_weeks_wind, "passing_yards", "attempts")
    qb_weather_coefs = fit_weather_adjustment(ypa_wind_result, "passing_yards", "attempts", "pregame_passing_yards_rate")
    print(f"  QB yards/attempt weather coefs (intercept, wind_slope, temp_slope): {qb_weather_coefs}")

    shares_wind = shares.merge(team_game_wind, on=["season", "week", "recent_team"], how="left")
    ypt_wind_engine = TdRateEngine(league_rate=league_ypt, prior_weight=80.0)  # fresh instance -- do NOT reuse ypt_engine, it's already been walked forward once for real prop predictions
    wr_wind_result = ypt_wind_engine.run_walk_forward(shares_wind[shares_wind["targets"] > 0], "receiving_yards", "targets")
    wr_weather_coefs = fit_weather_adjustment(wr_wind_result, "receiving_yards", "targets", "pregame_receiving_yards_rate")
    print(f"  WR/TE yards/target weather coefs (intercept, wind_slope, temp_slope): {wr_weather_coefs}")

    print("rebuilding pass rate...")
    pbp_seasons = covered["pbp"]
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet")

    # LEAGUE_AVG_PLAYS (review round 2, #4.2 stale-coupling sweep): was a hardcoded literal
    # (62.859), duplicated identically across three files, and never recomputed as new
    # seasons accumulate -- unlike every other rate/coefficient in this pipeline, which is
    # rebuilt fresh every run. Real play volume has been trending down (confirmed: full-
    # history 2016-2025 mean 62.40 plays/team-game vs. 2024-2025-only mean 61.04, a real
    # ~2.2% drop, plausibly a real pace-of-play trend) -- computed fresh here from
    # CALIBRATION_SEASONS, matching this project's own forward-looking-calibration
    # convention (§3.3), instead of a frozen constant. predict_props_2026.py's own copy is
    # left frozen (a one-time historical reference for that exact preseason forecast, same
    # pattern as SWAP_B_LAYER1/TD_PROB_CALIB there).
    plays_pbp = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run"]) & pbp["season"].isin(CALIBRATION_SEASONS)]
    LEAGUE_AVG_PLAYS = plays_pbp.groupby(["season", "week", "posteam"]).size().mean()
    print(f"  league avg plays/team-game ({min(CALIBRATION_SEASONS)}-{max(CALIBRATION_SEASONS)}): {LEAGUE_AVG_PLAYS:.2f}")

    # Monte Carlo drive-state game simulator (review round 2, #3.1): margin/win-probability
    # distribution at parity with the point-estimate approach on MAE/CRPS/straight-up/Brier
    # (§9.5 v2), while additionally producing a full outcome distribution the point-estimate
    # pipeline structurally cannot express -- unlocks moneyline pricing. DISPLAY-ONLY addition:
    # does not replace our_margin/our_total above. Total is deliberately NOT surfaced from the
    # simulator -- v2's total is known-weaker than the top-down model (MAE 11.63 vs 10.20).
    print("building game simulator (margin/win-probability distribution, display-only)...")
    game_sim, sim_bin_edges = build_simulator_for_season_range(pbp, schedules)

    pr_engine = PassRateEngine()
    pr_result = pr_engine.run_walk_forward(build_pass_rate_table(pbp))

    full_pr = build_full_game_pass_rate(pbp)
    sched_reg = schedules[schedules["game_type"] == "REG"][
        ["game_id", "home_team", "away_team", "home_score", "away_score", "spread_line", "total_line"]
    ]
    full_pr = full_pr.merge(sched_reg, on="game_id", how="inner").dropna(subset=["home_score"])
    full_pr["team_margin"] = np.where(
        full_pr["team"] == full_pr["home_team"], full_pr["home_score"] - full_pr["away_score"],
        full_pr["away_score"] - full_pr["home_score"],
    )
    full_pr = full_pr.merge(
        pr_result[["game_id", "team", "pregame_pass_rate_pred"]], on=["game_id", "team"], how="inner"
    )
    full_pr["pass_rate_residual"] = full_pr["full_pass_rate"] - full_pr["pregame_pass_rate_pred"]
    # BUG FIX 2026-07 (review #3.4 housekeeping audit): this used to read
    # `script_slope, script_intercept = np.polyfit(...)[::-1]`, which actually
    # assigns script_slope=intercept and script_intercept=slope (np.polyfit
    # returns [slope, intercept]; reversing without swapping the names is the
    # exact bug documented in src/utils/stats.py). Confirmed against real data
    # this flipped the sign and roughly doubled the magnitude of the "a team
    # that's winning passes less" game-script effect used to adjust every
    # prop prediction's pass attempts. fit_linear() returns (intercept, slope)
    # and asserts the sign is sane, so this can't silently regress again.
    #
    # v2 (review #1.5): market-implied team total added as a second regressor
    # -- validated real, if modest, improvement beyond team_margin alone (MAE
    # 0.07141->0.07080, p=0.0406, same-direction across all 4 TEST seasons
    # individually -- see game_environment.py's fit_game_script docstring).
    full_pr_market = full_pr.dropna(subset=["spread_line", "total_line"]).copy()
    full_pr_market["implied_team_total"] = np.where(
        full_pr_market["team"] == full_pr_market["home_team"],
        full_pr_market["total_line"] / 2 + full_pr_market["spread_line"] / 2,
        full_pr_market["total_line"] / 2 - full_pr_market["spread_line"] / 2,
    )
    train_mask = full_pr_market["season"].isin(CALIBRATION_SEASONS)
    game_script_coefs = fit_game_script(
        full_pr_market.loc[train_mask, "team_margin"],
        full_pr_market.loc[train_mask, "implied_team_total"],
        full_pr_market.loc[train_mask, "pass_rate_residual"],
    )

    # --- TD-probability calibration, AUTO-REFIT LINEAR every run (review round 4, #7) ---
    # Round 3 replaced the hand-fit TD_PROB_CALIB_A/B linear map with isotonic regression,
    # reasoning that isotonic's extra flexibility could fix the residual top-decile overshoot
    # a linear refit couldn't (0.494 predicted vs. 0.439 actual). Round 4 caught a real
    # self-consistency problem: round 3's own STRICT walk-forward comparison (both fit on
    # TRAIN, scored on TEST) showed isotonic UNDERPERFORMING linear by ~2x (+0.032 vs. +0.015
    # top-decile overshoot) -- isotonic was kept anyway on a full-history-fit comparison where
    # TEST sits inside the fit set, the same resubstitution-flavored evidence this project
    # correctly rejected for the CB coefficient ten sections earlier in this same document.
    # The genuinely strong argument for isotonic -- retiring the stale-coupling bug class --
    # comes from REFITTING EVERY RUN, not from isotonic specifically: a two-parameter linear
    # map refit every run (via fit_linear on the same real historical (raw_td_prob, actual_td)
    # pairs below, decile-bucketed the same way the original R0.1 refit was) captures 100% of
    # that structural benefit with the better demonstrated out-of-sample behavior. Shipped as
    # the default; validate_props_pipeline.py's strict walk-forward comparison also tests Beta
    # calibration as a small-df monotone upgrade candidate -- promote only if it wins that
    # comparison outright, not a full-history one.
    # Reuses every engine/table already built above (tgt_result, carry_result, rec_td_result,
    # rush_td_result, full_pr_market's real historical team_margin/implied_team_total) --
    # same construction validate_props_pipeline.py's decile-calibration check uses, just
    # applied to all available history instead of a single TRAIN/TEST split, matching this
    # pipeline's own "use everything for forward production" convention (e.g. the EB
    # prior_weight fit above).
    calib = tgt_result.rename(columns={"recent_team": "team"})[
        ["season", "week", "player_id", "team", "pregame_target_share_calc", "targets"]
    ].merge(
        carry_result.rename(columns={"recent_team": "team"})[["season", "week", "player_id", "team", "pregame_carry_share_calc", "carries"]],
        on=["season", "week", "player_id", "team"], how="outer",
    )
    calib = calib.merge(
        rec_td_result[["season", "week", "player_id", "pregame_receiving_tds_rate", "receiving_tds"]],
        on=["season", "week", "player_id"], how="left",
    )
    calib = calib.merge(
        rush_td_result[["season", "week", "player_id", "pregame_rushing_tds_rate", "rushing_tds"]],
        on=["season", "week", "player_id"], how="left",
    )
    calib = calib.merge(
        full_pr_market[["season", "week", "team", "pregame_pass_rate_pred", "team_margin", "implied_team_total"]],
        on=["season", "week", "team"], how="inner",
    )
    calib["adj_pass_rate"] = np.clip(
        calib["pregame_pass_rate_pred"] + apply_game_script(game_script_coefs, calib["team_margin"], calib["implied_team_total"]),
        0.30, 0.80,
    )
    calib["proj_targets"] = LEAGUE_AVG_PLAYS * calib["adj_pass_rate"] * calib["pregame_target_share_calc"].fillna(0)
    calib["proj_carries"] = LEAGUE_AVG_PLAYS * (1 - calib["adj_pass_rate"]) * calib["pregame_carry_share_calc"].fillna(0)
    calib["proj_rec_tds"] = calib["proj_targets"] * calib["pregame_receiving_tds_rate"].fillna(league_rec_td_rate)
    calib["proj_rush_tds"] = calib["proj_carries"] * calib["pregame_rushing_tds_rate"].fillna(league_rush_td_rate)
    calib["raw_td_prob"] = 1 - np.exp(-(calib["proj_rec_tds"] + calib["proj_rush_tds"]))
    calib["actual_td"] = ((calib["receiving_tds"].fillna(0) + calib["rushing_tds"].fillna(0)) >= 1).astype(float)
    calib = calib[(calib["pregame_target_share_calc"] > 0.03) | (calib["pregame_carry_share_calc"] > 0.03)]

    calib_bucketed = calib.copy()
    calib_bucketed["bucket"] = pd.qcut(calib_bucketed["raw_td_prob"], 10, labels=False, duplicates="drop")
    calib_bucket_means = calib_bucketed.groupby("bucket").agg(
        mean_predicted=("raw_td_prob", "mean"), actual_td_rate=("actual_td", "mean")
    )
    TD_PROB_CALIB_A, TD_PROB_CALIB_B = fit_linear(calib_bucket_means["mean_predicted"], calib_bucket_means["actual_td_rate"])
    print(f"  TD-probability calibration refit on {len(calib)} real historical player-weeks: "
          f"actual = {TD_PROB_CALIB_A:.3f} + {TD_PROB_CALIB_B:.3f}*raw")

    # --- offseason QB-swap bootstrap (Week 1 only, and only for the one season we
    # have a manually-extracted incoming-starter source for). nfl_data_py's schedule
    # never carries a starter QB for an unplayed game (confirmed: home_qb_id/
    # away_qb_id are NaN for every 2026 week-1 row), so a team whose real starter
    # changed this offseason (trade, FA signing, retirement) is invisible to every
    # engine here until that team actually plays a game. Clay's PDF is the only
    # source in this project that has that offseason information -- once Week 1 is
    # actually played, build_starter_sequence(schedules) picks up the real starter
    # from then on and this whole block naturally stops mattering (and in a future
    # season with no new PDF extraction, it silently no-ops rather than guessing).
    team_qb_swap_delta = {}
    team_new_starter_name = {}  # feeds presumed_starter_qb_id below so props use the real new starter, not a stale one
    qb_name_to_id = {}
    clay_path = DATA_RAW / "clay_2026_player_projections.parquet"
    if pred_season == 2026 and pred_week == 1 and clay_path.exists():
        clay_players = pd.read_parquet(clay_path)
        qb_name_to_id = build_qb_name_to_id(schedules)

        def qb_rating_by_name(name: str) -> float:
            qb_id = qb_name_to_id.get(name)
            return qb_engine.predict(qb_id) if qb_id is not None else 0.0

        qb_changes = find_qb_changes(schedules, clay_players)
        for row in qb_changes.itertuples():
            if row.likely_change:
                team_qb_swap_delta[row.team] = qb_rating_by_name(row.projected_2026) - qb_rating_by_name(row.starter_2025)
                team_new_starter_name[row.team] = row.projected_2026
        print(f"offseason QB-swap bootstrap: applying to {len(team_qb_swap_delta)} teams (Clay's 2026 projections) -- "
              f"{', '.join(sorted(team_qb_swap_delta))}")
    else:
        print("no offseason QB-swap source for this week -- ratings reflect only real starter data seen so far")

    # --- predict the next week's games ---
    print(f"\n=== predicting {pred_season} week {pred_week} ===\n")
    week_games = schedules[
        (schedules["season"] == pred_season) & (schedules["week"] == pred_week) & (schedules["game_type"] == "REG")
    ]

    game_rows = []
    for g in week_games.itertuples():
        home, away = g.home_team, g.away_team
        home_net, away_net = rating_engine.nets(home, away)
        rating_diff = home_net - away_net
        total_signal = home_net + away_net
        is_indoor = team_roof.get(home, 0.0)

        # MARGIN and TOTAL: use the real market line directly (no blend for
        # either, as of review round 2 -- see comment above), falling back to
        # our own Layer 1 calibration only if a line isn't published yet for
        # this game.
        base_margin = base_a + base_b * rating_diff
        base_total = total_coefs[0] + total_coefs[1] * total_signal + total_coefs[2] * is_indoor
        market_spread = getattr(g, "spread_line", None)
        market_total = getattr(g, "total_line", None)
        market_margin_valid = market_spread is not None and not (isinstance(market_spread, float) and np.isnan(market_spread))
        market_total_valid = market_total is not None and not (isinstance(market_total, float) and np.isnan(market_total))
        margin_base = market_spread if market_margin_valid else base_margin
        total_base = market_total if market_total_valid else base_total

        home_qb_swap = team_qb_swap_delta.get(home, 0.0)
        away_qb_swap = team_qb_swap_delta.get(away, 0.0)
        our_margin = margin_base + SWAP_B_MARKET * (home_qb_swap - away_qb_swap)
        our_total = total_base

        injury_adj = apply_joint_adjustment(
            pd.DataFrame([{"season": pred_season, "week": pred_week, "team_a": home, "team_b": away}]),
            injury_flags, coefs=JOINT_COEFS_FORWARD,
        ).iloc[0]
        our_margin += injury_adj

        # raw flags, for the CLV line-snapshot log below (review round 3, #2) -- separate
        # from injury_adj itself so the log records WHICH side is flagged, not just the
        # combined point adjustment.
        home_cb_rows = injury_flags[(injury_flags["season"] == pred_season) & (injury_flags["week"] == pred_week) & (injury_flags["team"] == home)]
        away_cb_rows = injury_flags[(injury_flags["season"] == pred_season) & (injury_flags["week"] == pred_week) & (injury_flags["team"] == away)]
        home_cb_flag = bool((home_cb_rows["cb_out"] >= 1).any())
        away_cb_flag = bool((away_cb_rows["cb_out"] >= 1).any())
        qb_swap_flag = (home_qb_swap - away_qb_swap) != 0.0

        # game simulator: margin/win-probability distribution + moneyline, display-only
        # (see comment above build_simulator_for_season_range). Needs a real market line
        # for both sides of the implied-total quantile conditioning; skip gracefully
        # (None fields) for the rare case a line isn't published yet, same fallback
        # discipline as margin/total above.
        sim_margin_std = sim_home_win_prob = sim_home_moneyline = sim_away_moneyline = sim_total_std = None
        if market_margin_valid and market_total_valid:
            home_implied_total = market_total / 2 + market_spread / 2
            away_implied_total = market_total / 2 - market_spread / 2
            home_q = implied_total_quantile_bin(home_implied_total, sim_bin_edges)
            away_q = implied_total_quantile_bin(away_implied_total, sim_bin_edges)
            # n_trials=5000 (raised from 300, review 2026-08): at 300 trials the Monte Carlo
            # standard error on a win probability is sqrt(0.25/300) ~= 2.9 percentage points --
            # material noise on a number converted directly into a moneyline, and larger than
            # the ~1% Brier improvement WIN_PROB_CALIB_A/B bought above. At 5000 trials the MC
            # error drops to ~0.6pp. Cost is real but small (a per-game simulation loop, not the
            # dominant runtime cost of a weekly pipeline run already fetching years of PBP data).
            sim_result = game_sim.simulate_game(home_q, away_q, n_trials=5000)

            # Recenter on our_margin before deriving win probability/moneyline (review round 4,
            # #4): the raw simulated margin distribution centers wherever the drive dynamics land
            # it (margin MAE 9.69 vs. the market-based our_margin's 9.49 -- worse at LOCATION),
            # while our_margin is the better location estimate (same "market for location, model
            # for shape" pattern already used to remove the margin market blend in Phase 0).
            # Confirmed live: mean |implied simulated margin - our_margin| = 1.58 points across a
            # full slate before this fix -- the displayed spread and the derived moneyline
            # implied different games. Shifting preserves the simulator's variance/skew/key-number
            # shape (the entire reason it exists) while inheriting the market-based margin's
            # location.
            recentered_margins = sim_result["margins"] - sim_result["margin_mean"] + our_margin
            recentered_win_prob = float((recentered_margins > 0).mean())
            # Calibration correction on top of recentering (see WIN_PROB_CALIB_A/B's own
            # comment above) -- recentering fixes LOCATION consistency with our_margin;
            # this fixes a separate, out-of-sample-confirmed miscalibration in the resulting
            # probability itself.
            calibrated_win_prob = min(max(WIN_PROB_CALIB_A + WIN_PROB_CALIB_B * recentered_win_prob, 0.0), 1.0)

            sim_margin_std = round(sim_result["margin_std"], 2)
            sim_home_win_prob = round(calibrated_win_prob, 3)
            sim_home_moneyline = american_odds_from_prob(calibrated_win_prob)
            sim_away_moneyline = american_odds_from_prob(1 - calibrated_win_prob)

            # Ship the simulated total distribution (review round 4, #5), unblocked by the same
            # recentering trick: total was held back because its LOCATION trails the top-down
            # model (sim MAE 12.28 vs. our_total's 10.20), but location is exactly what our_total
            # already provides -- the simulator's value is its SHAPE (a real distribution over
            # totals, enabling alt-totals/team-totals/O-U-at-any-number pricing the top-down
            # point estimate structurally cannot do). Only sim_total_std is surfaced -- the shape
            # is real and validated (§9.5); the (now-corrected) location is our_total's, not a new
            # independent estimate, so there's no separate "sim_total_mean" to publish.
            sim_total_std = round(sim_result["total_std"], 2)

        game_rows.append({
            "game_id": g.game_id, "home": home, "away": away,
            "our_home_pts": round((our_total + our_margin) / 2, 1),
            "our_away_pts": round((our_total - our_margin) / 2, 1),
            "our_margin": round(our_margin, 1), "our_total": round(our_total, 1),
            "market_spread": market_spread, "market_total": market_total,
            "sim_margin_std": sim_margin_std, "sim_home_win_prob": sim_home_win_prob,
            "sim_home_moneyline": sim_home_moneyline, "sim_away_moneyline": sim_away_moneyline,
            "sim_total_std": sim_total_std,
            "home_cb_flag": home_cb_flag, "away_cb_flag": away_cb_flag, "qb_swap_flag": qb_swap_flag,
            "wind": getattr(g, "wind", None), "roof": getattr(g, "roof", None), "temp": getattr(g, "temp", None),
        })

    games_df = pd.DataFrame(game_rows)
    pd.set_option("display.width", 160)
    print(games_df.to_string(index=False))
    games_df.to_parquet(DATA_PROCESSED / f"predictions_{pred_season}_wk{pred_week}.parquet", index=False)

    # --- CLV line-snapshot log (review round 3, #2) -- self-generated, zero-cost CLV
    # tracking: APPEND one row per game per run rather than overwriting, so the Tuesday
    # early-week line and the Sunday gameday-refresh line (this pipeline already runs
    # both, §11.2) both get preserved. This is the only evaluation instrument that works
    # on this project's real timescale (round 2, #2.3's own statistical-efficiency
    # argument) and it was previously being thrown away every run.
    #
    # bet_side (review round 4, #8): which side the model's disagreement with the market
    # implies betting, computed directly from this row's own our_margin/market_spread --
    # deterministic, so recording it now (rather than reconstructing after the fact) just
    # removes any later ambiguity about sign convention. "home" if our_margin is more
    # home-favorable than the market line, "away" if more away-favorable, "push" if equal.
    snapshot_cols = [
        "game_id", "home", "away", "market_spread", "market_total", "our_margin", "our_total",
        "home_cb_flag", "away_cb_flag", "qb_swap_flag", "sim_home_win_prob",
    ]
    snapshot = games_df[snapshot_cols].copy()
    snapshot["bet_side"] = np.select(
        [snapshot["our_margin"] > snapshot["market_spread"], snapshot["our_margin"] < snapshot["market_spread"]],
        ["home", "away"], default="push",
    )
    snapshot["run_timestamp"] = _dt.datetime.now().isoformat()
    snapshot["season"] = pred_season
    snapshot["week"] = pred_week
    snapshot_path = DATA_PROCESSED / "line_snapshots.parquet"
    if snapshot_path.exists():
        existing = pd.read_parquet(snapshot_path)
        snapshot = pd.concat([existing, snapshot], ignore_index=True)
    snapshot.to_parquet(snapshot_path, index=False)
    print(f"appended {len(games_df)} rows to line_snapshots.parquet ({len(snapshot)} total)")

    # --- post-game reconciliation (review round 4, #8): the log above captures the line
    # AT RUN TIME, not the CLOSING line CLV actually requires, and has no final score to
    # grade against -- neither CLV nor realized ATS was computable from it yet. A completed
    # game's own spread_line/total_line in nfl_data_py's schedule data IS the closing line
    # (already fetched every run); join it back onto every snapshot row whose game has since
    # finished. Runs on EVERY row each pipeline run (cheap: a merge, not a refetch), so a
    # game reconciles automatically the very next time this pipeline runs after it completes.
    all_sched = schedules[schedules["game_type"] == "REG"][
        ["game_id", "spread_line", "total_line", "home_score", "away_score"]
    ].rename(columns={"spread_line": "closing_spread", "total_line": "closing_total"})
    completed = all_sched.dropna(subset=["home_score", "away_score"])
    snapshot = snapshot.drop(columns=["closing_spread", "closing_total", "home_score", "away_score"], errors="ignore")
    snapshot = snapshot.merge(completed, on="game_id", how="left")
    snapshot.to_parquet(snapshot_path, index=False)
    n_reconciled = snapshot["home_score"].notna().sum()
    print(f"reconciled {n_reconciled}/{len(snapshot)} line_snapshots.parquet rows against completed games")

    # --- player props for the same week, using CURRENT rosters (not a static PDF) ---
    print(f"\n=== player props, {pred_season} week {pred_week} ===\n")
    ros_seasons = covered["weekly_rosters"]
    rosters = pd.read_parquet(DATA_RAW / f"weekly_rosters_{min(ros_seasons)}_{max(ros_seasons)}.parquet")
    latest_roster_week = rosters[rosters["season"] == pred_season]["week"].max()
    if pd.isna(latest_roster_week):
        latest_roster_season = rosters["season"].max()
        # BUG FIX (2026-07, exposed by the nflreadpy migration -- see fetch.py's module
        # docstring): this used to be rosters["week"].max() unconditionally, which picks the
        # roster table's OWN raw max week -- for a season already finished, that's whatever
        # postseason week the eventual Super Bowl participants played (week 22 for a recent
        # season). That's doubly wrong as a "current roster" proxy: (1) it's severely
        # degenerate -- every non-playoff team has already had its real players stripped from
        # the table (confirmed: week 22 of a recent season has only 165 rows total, vs.
        # ~2,650 at a normal midseason week); (2) even a real regular-season-ending week
        # (e.g. week 18) carries that week's real, IN-SEASON injury/reserve STATUSES (RES/INA/
        # etc.) -- confirmed directly: Patrick Mahomes shows status="RES" at week 18 of a
        # recent season from a real midseason IR stint, which would incorrectly carry over
        # into "currently out" if used as a stand-in for a brand-new season's roster. Fix:
        # use that season's own WEEK 1 snapshot instead -- confirmed directly (Mahomes,
        # Jayden Daniels both "ACT" at week 1 of the same season) that a season-opening
        # roster is the closest real historical analog to "a fresh, mostly-healthy, full
        # 32-team roster," which is exactly what's needed as a stand-in for a season not yet
        # underway. Previously masked entirely because nfl_data_py could already return real,
        # if early, NEXT-season preseason roster data, so this fallback path rarely fired;
        # nflreadpy currently validates season ranges more strictly (raises rather than
        # returning early preseason data), which exposed this latent bug on the first real
        # run under the new library.
        latest_roster_week = rosters[rosters["season"] == latest_roster_season]["week"].min()
    else:
        latest_roster_season = pred_season
    current_roster = rosters[(rosters["season"] == latest_roster_season) & (rosters["week"] == latest_roster_week)]
    current_roster = current_roster[current_roster["position"].isin(["WR", "TE", "RB", "QB"])]

    # true rookies (drafted/signed after our roster snapshot's season) are invisible to
    # weekly_rosters/weekly_player_stats entirely -- Clay's projections are the only
    # source in this project with real information about them. Same fallback pattern
    # already validated in predict_props_2026.py, ported here since it was never wired
    # into the live pipeline. Degrades gracefully (no rookie fallback) if no Clay
    # extraction exists for this season, same as the QB-swap bootstrap above.
    clay_path_props = DATA_RAW / "clay_2026_player_projections.parquet"
    if clay_path_props.exists():
        clay_players = pd.read_parquet(clay_path_props)
        clay_players["games"] = clay_players["games"].clip(lower=1)
        clay_name_to_id = build_name_to_id(rosters)
        clay_lastname_team_pos_to_id = build_lastname_team_pos_to_id(rosters, season=latest_roster_season)
        clay_lastname_pos_to_id = build_lastname_pos_to_id(rosters, season=latest_roster_season)
    else:
        clay_players = None

    prop_rows = []
    for g in game_rows:
        margin = g["our_margin"]
        m_spread, m_total = g["market_spread"], g["market_total"]
        market_valid = all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in (m_spread, m_total))
        home_implied_total = m_total / 2 + m_spread / 2 if market_valid else 0.0
        away_implied_total = m_total / 2 - m_spread / 2 if market_valid else 0.0
        for team, team_margin, implied_total in [
            (g["home"], margin, home_implied_total), (g["away"], -margin, away_implied_total)
        ]:
            script_adj = apply_game_script(game_script_coefs, team_margin, implied_total)
            pass_rate = np.clip(pr_engine.predict(team) + script_adj, 0.30, 0.80)
            pass_attempts = LEAGUE_AVG_PLAYS * pass_rate
            rush_attempts = LEAGUE_AVG_PLAYS * (1 - pass_rate)
            team_roster = current_roster[current_roster["team"] == team]

            # players ruled out this week (real roster status + verified manual overrides
            # for breaking news the automated data hasn't caught up to yet -- see
            # injury_reallocation.py). RB carry share gets proportionally reallocated
            # among the remaining active backs (validated: real improvement, ~17% lower
            # MAE on historical OUT instances). WR/TE targets are NOT reallocated --
            # validated to make predictions worse; excluding the player is already the
            # better default.
            out_ids = out_player_ids_for_team(team, current_roster)
            team_rbs = team_roster[team_roster["position"] == "RB"]
            rb_raw_shares = {pid: carry_engine.predict(pid) for pid in team_rbs["player_id"]}
            # draft-capital fallback first (always available, every season), Clay's
            # real current-year projection second so it overrides when present --
            # Clay's scheme-fit/landing-spot-specific info is likely more precise
            # than a historical round/position prior when it actually exists.
            rb_draft_fallback = rookie_fallback_from_draft_capital(
                team, "RB", current_roster, draft_picks, pred_season, carry_engine, carry_share_prior,
            )
            rookie_rbs = {}
            if clay_players is not None:
                rookie_rbs = rookie_fallback_rb_rates(
                    team, clay_players,
                    clay_name_to_id, clay_lastname_team_pos_to_id, clay_lastname_pos_to_id,
                    carry_engine, rush_attempts, league_ypc, league_rush_td_rate,
                )
                # BUG FIX (2026-07, found verifying the nflreadpy migration): rookie_fallback_rb_rates
                # doesn't check out_ids at all (its own trigger is "zero real engine history", unrelated
                # to this week's roster status) -- a true rookie who is BOTH a zero-history player AND
                # currently marked Out/inactive gets rendered TWICE: once by the main per-roster-row loop
                # (status_note="OUT", correctly zeroed) and again here (a real, nonzero fallback row for
                # the same person). Confirmed reproducing (Jordan James, SF, 2026 wk1). Drop any Clay
                # fallback entry whose name matches an Out player on this team, by normalized name (same
                # matching convention as the draft-capital-vs-Clay dedup just below, for the same reason:
                # resolved_pid isn't reliable enough alone).
                out_names = {norm_name(current_roster.loc[current_roster["player_id"] == pid, "player_name"].iloc[0])
                             for pid in out_ids if (current_roster["player_id"] == pid).any()}
                rookie_rbs = {k: v for k, v in rookie_rbs.items() if norm_name(v["player_name"]) not in out_names}
            # BUG FIX 2026-07 (self-caught in a full-codebase review): the two
            # rookie fallbacks are keyed differently -- draft-capital by the
            # real player_id, Clay by a synthetic "clay:<name>" string -- so
            # a true first-year rookie RB covered by BOTH (real draft slot +
            # a Clay projection) could get double-counted in reallocate_shares
            # AND rendered as two separate prop rows for the same real person.
            # Drop the draft-capital entry for anyone Clay ALSO covers, so
            # exactly one fallback (Clay, when available; draft-capital
            # otherwise) ever contributes a share for a given real player.
            # De-dupe by NORMALIZED NAME (not resolved_pid) since both sources
            # have a player_name regardless of whether either one's own ID
            # resolution succeeded.
            #
            # (A real 2026 case -- Quinshon Judkins, CLE -- initially looked
            # like this exact bug but wasn't: it was a separate name-COLLISION
            # bug inside rookie_fallback_rb_rates itself, since fixed there --
            # see injury_reallocation.py. This dedup is still kept as a real,
            # independent safety net for genuine both-sources-cover-the-same-
            # true-rookie cases.)
            clay_covered_names = {norm_name(v["player_name"]) for v in rookie_rbs.values()}
            pid_to_name = current_roster.set_index("player_id")["player_name"].to_dict()
            rb_draft_fallback = {
                pid: share for pid, share in rb_draft_fallback.items()
                if norm_name(pid_to_name.get(pid, "")) not in clay_covered_names
            }
            rb_raw_shares.update(rb_draft_fallback)
            rb_raw_shares.update({k: v["share"] for k, v in rookie_rbs.items()})
            rb_share_overrides = reallocate_shares(rb_raw_shares, out_ids)

            # WR/TE: draft-capital fallback only (no Clay equivalent exists for
            # these positions today -- a real, previously-unfilled gap). NOT run
            # through reallocate_shares -- that mechanism is specifically
            # validated-harmful for WR/TE (§6.2), unrelated to this fallback.
            wr_te_draft_fallback = {
                **rookie_fallback_from_draft_capital(team, "WR", current_roster, draft_picks, pred_season, tgt_engine, target_share_prior),
                **rookie_fallback_from_draft_capital(team, "TE", current_roster, draft_picks, pred_season, tgt_engine, target_share_prior),
            }
            # BUG FIX (2026-07, found investigating the roster-snapshot fallback bug above):
            # this used to be current_roster (ALL 32 teams) minus out_ids, not team-filtered --
            # harmless whenever presumed_starter_qb_id's real-history match succeeds, but its
            # own fallback (roster_qbs.iloc[0]) then silently returns THE SAME first QB row in
            # the entire league-wide table for every team that falls through, rather than a
            # real roster QB for that specific team. Team-filtering here makes that fallback
            # (if it ever fires again) at least return one of this team's own real QBs.
            qb_eligible_roster = team_roster[~team_roster["player_id"].isin(out_ids)]

            starter_id = presumed_starter_qb_id(
                team, qb_eligible_roster, weekly,
                override_name=team_new_starter_name.get(team), override_name_to_id=qb_name_to_id,
            )
            if starter_id is not None:
                passing = project_passing_stats(starter_id, pass_attempts, qb_passing_engines)
                # weather adjustment (wind + temp, fit jointly): validated (p=0.0086) -- no-op
                # indoors or when weather isn't known yet (far-in-advance predictions; this is
                # the main thing a near-kickoff refresh gains once a real forecast exists)
                adj_ypa = apply_weather_adjustment(
                    qb_passing_engines["passing_yards"].predict(starter_id), g["wind"], g["temp"], g["roof"], qb_weather_coefs
                )
                passing["proj_passing_yards"] = round(adj_ypa * pass_attempts, 1)
                starter_row = current_roster[current_roster["player_id"] == starter_id]
                starter_name = starter_row["player_name"].iloc[0] if not starter_row.empty else starter_id
                prop_rows.append({
                    "game_id": g["game_id"], "team": team, "player": starter_name, "position": "QB",
                    **passing,
                    "proj_targets": 0.0, "proj_carries": 0.0, "proj_receptions": 0.0,
                    "proj_rec_yards": 0.0, "proj_rush_yards": 0.0, "td_probability": None,
                    "status_note": "",
                })

            for p in team_roster.itertuples():
                pid = p.player_id
                if p.position == "QB":
                    continue  # starter handled above with a full passing statline; backups add no real prop value
                if pid in out_ids:
                    prop_rows.append({
                        "game_id": g["game_id"], "team": team, "player": p.player_name, "position": p.position,
                        "proj_pass_attempts": 0.0, "proj_completions": 0.0, "proj_passing_yards": 0.0,
                        "proj_passing_tds": 0.0, "proj_interceptions": 0.0,
                        "proj_targets": 0.0, "proj_carries": 0.0, "proj_receptions": 0.0,
                        "proj_rec_yards": 0.0, "proj_rush_yards": 0.0, "td_probability": 0.0,
                        "status_note": "OUT",
                    })
                    continue
                tgt_share = wr_te_draft_fallback.get(pid, tgt_engine.predict(pid))
                carry_share = rb_share_overrides.get(pid, carry_engine.predict(pid))
                if tgt_share == 0 and carry_share == 0:
                    continue
                proj_targets = pass_attempts * tgt_share
                proj_carries = rush_attempts * carry_share
                proj_receptions = proj_targets * catch_rate_engine.predict(pid)
                adj_ypt = apply_weather_adjustment(ypt_engine.predict(pid), g["wind"], g["temp"], g["roof"], wr_weather_coefs)
                proj_rec_yards = proj_targets * adj_ypt
                proj_rush_yards = proj_carries * ypc_engine.predict(pid)
                proj_rec_tds = proj_targets * rec_td_engine.predict(pid)
                proj_rush_tds = proj_carries * rush_td_engine.predict(pid)
                raw_td_prob = 1 - np.exp(-(proj_rec_tds + proj_rush_tds))
                td_prob = float(np.clip(TD_PROB_CALIB_A + TD_PROB_CALIB_B * raw_td_prob, 0.0, 1.0))
                prop_rows.append({
                    "game_id": g["game_id"], "team": team, "player": p.player_name, "position": p.position,
                    "proj_pass_attempts": 0.0, "proj_completions": 0.0, "proj_passing_yards": 0.0,
                    "proj_passing_tds": 0.0, "proj_interceptions": 0.0,
                    "proj_targets": round(proj_targets, 1), "proj_carries": round(proj_carries, 1),
                    "proj_receptions": round(proj_receptions, 1),
                    "proj_rec_yards": round(proj_rec_yards, 1), "proj_rush_yards": round(proj_rush_yards, 1),
                    "td_probability": round(td_prob, 3),
                    "status_note": "",
                })

            for key, info in rookie_rbs.items():
                carry_share = rb_share_overrides.get(key, 0.0)
                if carry_share <= 0:
                    continue
                proj_carries = rush_attempts * carry_share
                proj_rush_yards = proj_carries * info["ypc_rate"]
                proj_rush_tds = proj_carries * info["rush_td_rate"]
                raw_td_prob_rb = 1 - np.exp(-proj_rush_tds)
                td_prob = float(np.clip(TD_PROB_CALIB_A + TD_PROB_CALIB_B * raw_td_prob_rb, 0.0, 1.0))
                prop_rows.append({
                    "game_id": g["game_id"], "team": team, "player": info["player_name"], "position": "RB",
                    "proj_pass_attempts": 0.0, "proj_completions": 0.0, "proj_passing_yards": 0.0,
                    "proj_passing_tds": 0.0, "proj_interceptions": 0.0,
                    "proj_targets": 0.0, "proj_carries": round(proj_carries, 1), "proj_receptions": 0.0,
                    "proj_rec_yards": 0.0, "proj_rush_yards": round(proj_rush_yards, 1),
                    "td_probability": round(td_prob, 3),
                    "status_note": "rookie fallback (Clay projection, no NFL history)",
                })

    props_df = pd.DataFrame(prop_rows)
    props_df.to_parquet(DATA_PROCESSED / f"props_{pred_season}_wk{pred_week}.parquet", index=False)

    # --- print organized BY GAME: every game score comes with both teams' expected statlines ---
    print(f"\n=== full expected statlines by game, {pred_season} week {pred_week} ===\n")
    pd.set_option("display.width", 200)
    skill_cols = ["player", "position", "proj_targets", "proj_receptions", "proj_rec_yards",
                  "proj_carries", "proj_rush_yards", "td_probability"]
    qb_cols = ["player", "position", "proj_pass_attempts", "proj_completions", "proj_passing_yards",
               "proj_passing_tds", "proj_interceptions"]
    for g in game_rows:
        print(f"--- {g['away']} @ {g['home']}  (proj {g['our_away_pts']}-{g['our_home_pts']}, total {g['our_total']}) ---")
        game_props = props_df[props_df["game_id"] == g["game_id"]]
        for team in (g["away"], g["home"]):
            team_props = game_props[game_props["team"] == team]
            qb_row = team_props[team_props["position"] == "QB"]
            skill = team_props[team_props["position"] != "QB"]
            active = skill[skill["status_note"] != "OUT"]
            out_rows = skill[skill["status_note"] == "OUT"]
            skill_rows = active.sort_values("td_probability", ascending=False).head(6)
            print(f"  {team}:")
            if not qb_row.empty:
                print("   " + qb_row[qb_cols].to_string(index=False, header=False))
            if not out_rows.empty:
                for r in out_rows.itertuples():
                    print(f"    {r.player} ({r.position}): OUT -- touches reallocated to teammates below")
            print("   " + skill_rows[skill_cols].to_string(index=False, header=False))
        print()

    print(f"saved predictions_{pred_season}_wk{pred_week}.parquet, props_{pred_season}_wk{pred_week}.parquet")
