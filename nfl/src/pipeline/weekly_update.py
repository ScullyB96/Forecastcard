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
from src.ingest.fetch import fetch_injuries, fetch_pbp, fetch_schedules, fetch_snap_counts, fetch_weekly_rosters
from src.models.game_environment import PassRateEngine, build_full_game_pass_rate, build_pass_rate_table
from src.models.injury_adjustment import JOINT_COEFS_FORWARD, apply_joint_adjustment, compute_injury_flags
from src.models.injury_reallocation import out_player_ids_for_team, reallocate_shares, rookie_fallback_rb_rates
from src.models.market_blend import MARKET_FIT_START_SEASON, apply_blend, fit_market_blend
from src.models.player_usage import ShareEngine, TdRateEngine, build_player_week_shares
from src.models.predict_2026 import build_qb_name_to_id, find_qb_changes
from src.models.predict_props_2026 import build_lastname_pos_to_id, build_lastname_team_pos_to_id, build_name_to_id
from src.models.qb_adjustment import QbRatingEngine, build_qb_week_table, build_starter_sequence
from src.models.qb_passing_stats import build_qb_passing_engines, presumed_starter_qb_id, project_passing_stats
from src.models.ratings import PowerRatingEngine, build_dataset
from src.models.weather_adjustment import apply_wind_adjustment, fit_wind_adjustment
from src.utils.paths import DATA_PROCESSED, DATA_RAW

CALIBRATION_SEASONS = {2022, 2023, 2024, 2025}  # most recent, non-COVID-anomalous -- see predict_2026.py
LEAGUE_AVG_PLAYS = 62.859
SWAP_B = 6.616  # points per unit of QB rating gap, validated in qb_adjustment.py


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

    # our own PBP-derived weekly stats, not nfl_data_py's import_weekly_data (lags the season)
    pbp_seasons = covered["pbp"]
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet",
                          columns=["season", "week", "season_type", "posteam", "play_type", "receiver_player_id",
                                   "rusher_player_id", "passer_player_id", "epa", "complete_pass", "receiving_yards",
                                   "rushing_yards", "pass_touchdown", "rush_touchdown", "passing_yards", "interception"])
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
    b, a = np.polyfit(x, y, 1)
    return a, b


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
    layer1_full = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    layer1_full["actual_total"] = layer1_full["home_score"] + layer1_full["away_score"]
    roof_lookup = reg_sched[["game_id", "roof"]].assign(is_indoor=lambda d: d["roof"].isin(["dome", "closed"]).astype(float))
    layer1_full = layer1_full.merge(roof_lookup[["game_id", "is_indoor"]], on="game_id", how="left")
    total_train = layer1_full[layer1_full["season"].isin(CALIBRATION_SEASONS)]
    X_train = np.column_stack([np.ones(len(total_train)), total_train["pregame_total_signal"], total_train["is_indoor"]])
    total_coefs, *_ = np.linalg.lstsq(X_train, total_train["actual_total"].to_numpy(), rcond=None)

    print("fitting market blend (real Vegas closing lines, already in schedules data)...")
    market_fit_seasons = set(range(MARKET_FIT_START_SEASON, season))  # grows every year, excludes the season being predicted
    market = schedules[["game_id", "spread_line", "total_line"]]

    margin_fit = rated.merge(market, on="game_id", how="inner").dropna(subset=["spread_line"])
    margin_fit = margin_fit[margin_fit["season"].isin(market_fit_seasons)]
    margin_fit_our_pred = base_a + base_b * margin_fit["pregame_rating_diff"]
    margin_blend = fit_market_blend(margin_fit_our_pred, margin_fit["spread_line"], margin_fit["actual_margin"])
    print(f"  margin blend (n={len(margin_fit)}): {margin_blend}")

    total_fit = layer1_full.merge(market, on="game_id", how="inner").dropna(subset=["total_line"])
    total_fit = total_fit[total_fit["season"].isin(market_fit_seasons)]
    total_fit_our_pred = total_coefs[0] + total_coefs[1] * total_fit["pregame_total_signal"] + total_coefs[2] * total_fit["is_indoor"]
    total_blend = fit_market_blend(total_fit_our_pred, total_fit["total_line"], total_fit["actual_total"])
    print(f"  total blend (n={len(total_fit)}): {total_blend}")

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
    tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    carry_engine = ShareEngine()
    carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")
    # prior_weight=30 (raised from 15 -- that value was only ever validated via
    # aggregate decile calibration, never actually swept. A proper sweep found
    # aggregate MAE/calibration are flat from prior_weight=10 to 130, but 15
    # under-regularizes extreme small-sample outliers: a rookie with a real
    # 5-TD-on-22-target stretch still projected at 3.3x league average even
    # after shrinkage. 30 costs nothing in validated aggregate accuracy while
    # meaningfully taming that case (3.3x -> 2.7x).)
    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()
    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=30.0)
    rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")
    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=30.0)
    rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    # yards/catch-rate: real but weaker signal than share or TD-rate (validated via
    # prior_weight sweep -- see predict_props_2026.py). Included for a complete
    # statline, not because it's a strong differentiator.
    league_ypt = shares["receiving_yards"].sum() / shares["targets"].sum()
    league_ypc = shares["rushing_yards"].sum() / shares["carries"].sum()
    league_catch_rate = shares["receptions"].sum() / shares["targets"].sum()
    ypt_engine = TdRateEngine(league_rate=league_ypt, prior_weight=80.0)
    ypt_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_yards", "targets")
    ypc_engine = TdRateEngine(league_rate=league_ypc, prior_weight=300.0)
    ypc_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_yards", "carries")
    catch_rate_engine = TdRateEngine(league_rate=league_catch_rate, prior_weight=15.0)
    catch_rate_engine.run_walk_forward(shares[shares["targets"] > 0], "receptions", "targets")

    print("rebuilding QB passing statline engines (completions, yards, TDs, INTs)...")
    qb_passing_engines = build_qb_passing_engines(weekly)

    print("fitting wind adjustment (QB yards/attempt, WR/TE yards/target)...")
    sched_reg_wind = schedules[schedules["game_type"] == "REG"][["game_id", "season", "week", "home_team", "away_team", "wind", "roof"]]
    wind_home = sched_reg_wind.rename(columns={"home_team": "recent_team"})[["season", "week", "recent_team", "wind", "roof"]]
    wind_away = sched_reg_wind.rename(columns={"away_team": "recent_team"})[["season", "week", "recent_team", "wind", "roof"]]
    team_game_wind = pd.concat([wind_home, wind_away])

    qb_weeks_wind = weekly[(weekly["season_type"] == "REG") & (weekly["attempts"] >= 5)].merge(
        team_game_wind, on=["season", "week", "recent_team"], how="left"
    )
    league_ypa = qb_weeks_wind["passing_yards"].sum() / qb_weeks_wind["attempts"].sum()
    ypa_wind_engine = TdRateEngine(league_rate=league_ypa, prior_weight=150.0)
    ypa_wind_result = ypa_wind_engine.run_walk_forward(qb_weeks_wind, "passing_yards", "attempts")
    qb_wind_coefs = fit_wind_adjustment(ypa_wind_result, "passing_yards", "attempts", "pregame_passing_yards_rate")
    print(f"  QB yards/attempt wind coefs (intercept, slope): {qb_wind_coefs}")

    shares_wind = shares.merge(team_game_wind, on=["season", "week", "recent_team"], how="left")
    ypt_wind_engine = TdRateEngine(league_rate=league_ypt, prior_weight=80.0)  # fresh instance -- do NOT reuse ypt_engine, it's already been walked forward once for real prop predictions
    wr_wind_result = ypt_wind_engine.run_walk_forward(shares_wind[shares_wind["targets"] > 0], "receiving_yards", "targets")
    wr_wind_coefs = fit_wind_adjustment(wr_wind_result, "receiving_yards", "targets", "pregame_receiving_yards_rate")
    print(f"  WR/TE yards/target wind coefs (intercept, slope): {wr_wind_coefs}")

    print("rebuilding pass rate...")
    pbp_seasons = covered["pbp"]
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet")
    pr_engine = PassRateEngine()
    pr_result = pr_engine.run_walk_forward(build_pass_rate_table(pbp))

    full_pr = build_full_game_pass_rate(pbp)
    sched_reg = schedules[schedules["game_type"] == "REG"][["game_id", "home_team", "away_team", "home_score", "away_score"]]
    full_pr = full_pr.merge(sched_reg, on="game_id", how="inner").dropna(subset=["home_score"])
    full_pr["team_margin"] = np.where(
        full_pr["team"] == full_pr["home_team"], full_pr["home_score"] - full_pr["away_score"],
        full_pr["away_score"] - full_pr["home_score"],
    )
    full_pr = full_pr.merge(
        pr_result[["game_id", "team", "pregame_pass_rate_pred"]], on=["game_id", "team"], how="inner"
    )
    full_pr["pass_rate_residual"] = full_pr["full_pass_rate"] - full_pr["pregame_pass_rate_pred"]
    train_mask = full_pr["season"].isin(CALIBRATION_SEASONS)
    script_slope, script_intercept = np.polyfit(
        full_pr.loc[train_mask, "team_margin"], full_pr.loc[train_mask, "pass_rate_residual"], 1
    )[::-1]

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

        # blend our base Layer 1 prediction with the real market line (if published
        # yet for this game), THEN layer our own validated corrections (QB swap,
        # injury) on top -- matches exactly what was validated in market_blend.py
        base_margin = base_a + base_b * rating_diff
        base_total = total_coefs[0] + total_coefs[1] * total_signal + total_coefs[2] * is_indoor
        market_spread = getattr(g, "spread_line", None)
        market_total = getattr(g, "total_line", None)
        blended_margin = apply_blend(base_margin, market_spread, margin_blend)
        blended_total = apply_blend(base_total, market_total, total_blend)

        home_qb_swap = team_qb_swap_delta.get(home, 0.0)
        away_qb_swap = team_qb_swap_delta.get(away, 0.0)
        our_margin = blended_margin + SWAP_B * (home_qb_swap - away_qb_swap)
        our_total = blended_total

        injury_adj = apply_joint_adjustment(
            pd.DataFrame([{"season": pred_season, "week": pred_week, "team_a": home, "team_b": away}]),
            injury_flags, coefs=JOINT_COEFS_FORWARD,
        ).iloc[0]
        our_margin += injury_adj

        game_rows.append({
            "game_id": g.game_id, "home": home, "away": away,
            "our_home_pts": round((our_total + our_margin) / 2, 1),
            "our_away_pts": round((our_total - our_margin) / 2, 1),
            "our_margin": round(our_margin, 1), "our_total": round(our_total, 1),
            "market_spread": market_spread, "market_total": market_total,
            "wind": getattr(g, "wind", None), "roof": getattr(g, "roof", None),
        })

    games_df = pd.DataFrame(game_rows)
    pd.set_option("display.width", 160)
    print(games_df.to_string(index=False))
    games_df.to_parquet(DATA_PROCESSED / f"predictions_{pred_season}_wk{pred_week}.parquet", index=False)

    # --- player props for the same week, using CURRENT rosters (not a static PDF) ---
    print(f"\n=== player props, {pred_season} week {pred_week} ===\n")
    ros_seasons = covered["weekly_rosters"]
    rosters = pd.read_parquet(DATA_RAW / f"weekly_rosters_{min(ros_seasons)}_{max(ros_seasons)}.parquet")
    latest_roster_week = rosters[rosters["season"] == pred_season]["week"].max()
    if pd.isna(latest_roster_week):
        latest_roster_week = rosters["week"].max()
        latest_roster_season = rosters["season"].max()
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
        for team, team_margin in [(g["home"], margin), (g["away"], -margin)]:
            pass_rate = np.clip(pr_engine.predict(team) + script_intercept + script_slope * team_margin, 0.30, 0.80)
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
            rookie_rbs = {}
            if clay_players is not None:
                rookie_rbs = rookie_fallback_rb_rates(
                    team, clay_players,
                    clay_name_to_id, clay_lastname_team_pos_to_id, clay_lastname_pos_to_id,
                    carry_engine, rush_attempts, league_ypc, league_rush_td_rate,
                )
                rb_raw_shares.update({k: v["share"] for k, v in rookie_rbs.items()})
            rb_share_overrides = reallocate_shares(rb_raw_shares, out_ids)
            qb_eligible_roster = current_roster[~current_roster["player_id"].isin(out_ids)]

            starter_id = presumed_starter_qb_id(
                team, qb_eligible_roster, weekly,
                override_name=team_new_starter_name.get(team), override_name_to_id=qb_name_to_id,
            )
            if starter_id is not None:
                passing = project_passing_stats(starter_id, pass_attempts, qb_passing_engines)
                # wind adjustment: validated (p=0.0022) -- no-op indoors or when wind isn't
                # known yet (far-in-advance predictions; this is the main thing a
                # near-kickoff refresh gains once a real forecast exists)
                adj_ypa = apply_wind_adjustment(
                    qb_passing_engines["passing_yards"].predict(starter_id), g["wind"], g["roof"], qb_wind_coefs
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
                tgt_share = tgt_engine.predict(pid)
                carry_share = rb_share_overrides.get(pid, carry_engine.predict(pid))
                if tgt_share == 0 and carry_share == 0:
                    continue
                proj_targets = pass_attempts * tgt_share
                proj_carries = rush_attempts * carry_share
                proj_receptions = proj_targets * catch_rate_engine.predict(pid)
                adj_ypt = apply_wind_adjustment(ypt_engine.predict(pid), g["wind"], g["roof"], wr_wind_coefs)
                proj_rec_yards = proj_targets * adj_ypt
                proj_rush_yards = proj_carries * ypc_engine.predict(pid)
                proj_rec_tds = proj_targets * rec_td_engine.predict(pid)
                proj_rush_tds = proj_carries * rush_td_engine.predict(pid)
                raw_td_prob = 1 - np.exp(-(proj_rec_tds + proj_rush_tds))
                td_prob = np.clip(0.046 + 0.769 * raw_td_prob, 0.0, 1.0)
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
                td_prob = np.clip(0.046 + 0.769 * (1 - np.exp(-proj_rush_tds)), 0.0, 1.0)
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
