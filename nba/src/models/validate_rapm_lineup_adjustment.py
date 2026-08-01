"""Phase 2b's acceptance gate: does adding the RAPM-lite active-lineup
adjustment on top of Phase 1's team-strength baseline actually improve
predictions -- using the ORACLE minutes-share mode (real, already-known
per-game minutes; see `lineup_rating.py`'s docstring for why this measures
a ceiling, not yet a deployable live prediction).

Dev-only, same confirmatory-veto discipline as Phase 1's validation script.
Run as `python -m src.models.validate_rapm_lineup_adjustment`.
"""

import pandas as pd

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_str
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.lineup_rating import (
    oracle_minutes_shares, player_minutes_from_stints, project_lineup_adjustment, team_recent_roster_rapm,
)
from src.models.rapm_lite import compute_walkforward_player_ratings, prepare_stints
from src.models.team_strength import project_game
from src.models.validate_team_strength_baseline import _rated_dev_log, _to_wide_games, build_dev_predictions


def load_dev_stints() -> pd.DataFrame:
    frames = []
    for y in range(FIRST_DEV_SEASON, DEV_MAX_SEASON):
        schedule_path = f"data/raw/schedule_{season_str(y)}.parquet"
        try:
            schedule = pd.read_parquet(schedule_path)
        except FileNotFoundError:
            continue
        stints = build_season_stints(y)
        if stints.empty:
            continue
        frames.append(prepare_stints(stints, schedule[schedule["seasonType"] == "Regular Season"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_team_history(long_log: pd.DataFrame) -> tuple[dict, dict]:
    """From the long (one row per team per game) team-game log: for each
    team, the ordered-by-date list of that team's own (gameDate, season,
    gameId) tuples, plus a per-team {gameId: 'home'/'away'} side lookup --
    what `lineup_rating.team_recent_roster_rapm` needs for its lookback.

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): Sec22 fixed a
    real cross-season roster-bleed bug (a team's "last N games" lookback
    silently blending in the PRIOR season's roster early in a new season,
    inflating the resolved roster ~2x) for the two LIVE pipelines only,
    by restricting `team_log` to `season == target_season` before
    building team history. This function -- reused unmodified by
    `validate_predictive_lineup_adjustment.py` and
    `run_final_holdout_check.py` -- produced the officially-adopted Phase
    2 dev/holdout numbers (Sec7/Sec9.1/Sec9.3/Sec9.4) using the SAME
    unfixed, season-blending logic, and was never re-validated after the
    live fix landed. Now also tracks each game's own `season` so every
    caller's per-row filter can additionally require `season ==
    row.season`, mirroring the live fix exactly -- see
    MODEL_DOCUMENTATION.md for the re-validation this enabled."""
    long_log = long_log.sort_values("gameDate")
    team_history: dict[int, list] = {}
    team_side: dict[int, dict] = {}
    for row in long_log.itertuples(index=False):
        team_history.setdefault(row.team, []).append((row.gameDate, row.season, row.gameId))
        team_side.setdefault(row.team, {})[row.gameId] = "home" if row.is_home else "away"
    return team_history, team_side


def run_oracle_lineup_backtest() -> pd.DataFrame:
    stints = load_dev_stints()
    if stints.empty:
        raise RuntimeError("no dev-range stints available yet -- backfill not far enough along")

    player_ratings = compute_walkforward_player_ratings(stints)
    if player_ratings.empty:
        raise RuntimeError("not enough stint history yet to fit any RAPM checkpoint")
    player_minutes = player_minutes_from_stints(stints)

    baseline_log = _rated_dev_log()
    games = _to_wide_games(baseline_log)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)
    games = games.merge(stints[["gameId"]].drop_duplicates(), on="gameId", how="inner")

    team_history, team_side = _build_team_history(baseline_log)
    checkpoints = player_ratings["asOfDate"].sort_values().unique()

    records = []
    for row in games.itertuples(index=False):
        prior_checkpoints = checkpoints[checkpoints < pd.Timestamp(row.gameDate)]
        if len(prior_checkpoints) == 0:
            continue
        as_of = prior_checkpoints[-1]
        ratings_snapshot = player_ratings[player_ratings["asOfDate"] == as_of]

        home_id, away_id = row.team_home, row.team_away
        home_prior_games = [gid for gdate, gseason, gid in team_history.get(home_id, [])
                             if gdate < row.gameDate and gseason == row.season]
        away_prior_games = [gid for gdate, gseason, gid in team_history.get(away_id, [])
                             if gdate < row.gameDate and gseason == row.season]

        home_recent_off, home_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, home_prior_games, team_side[home_id])
        away_recent_off, away_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, away_prior_games, team_side[away_id])

        home_shares = oracle_minutes_shares(player_minutes, row.gameId, "home")
        away_shares = oracle_minutes_shares(player_minutes, row.gameId, "away")
        if home_shares.empty or away_shares.empty:
            continue

        home_off_adj, home_def_adj = project_lineup_adjustment(home_shares, ratings_snapshot, home_recent_off, home_recent_def)
        away_off_adj, away_def_adj = project_lineup_adjustment(away_shares, ratings_snapshot, away_recent_off, away_recent_def)

        _, pred_home_lineup, pred_away_lineup = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home + home_off_adj, home_dRtg=row.rtg_defense_rate_home + home_def_adj,
            away_oRtg=row.rtg_attack_rate_away + away_off_adj, away_dRtg=row.rtg_defense_rate_away + away_def_adj,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        records.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
            "pred_home_lineup": pred_home_lineup, "pred_away_lineup": pred_away_lineup,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Loading dev stints and fitting walk-forward RAPM-lite checkpoints...", flush=True)
    result = run_oracle_lineup_backtest()
    print(f"built {len(result)} oracle lineup-adjusted predictions", flush=True)

    baseline = build_dev_predictions()[["gameId", "pred_home", "pred_away"]].rename(
        columns={"pred_home": "pred_home_baseline", "pred_away": "pred_away_baseline"})
    result = result.merge(baseline, on="gameId", how="inner")

    result["abs_total_err_lineup"] = ((result["pred_home_lineup"] + result["pred_away_lineup"]) - (result["actual_home"] + result["actual_away"])).abs()
    result["abs_margin_err_lineup"] = ((result["pred_home_lineup"] - result["pred_away_lineup"]) - (result["actual_home"] - result["actual_away"])).abs()
    result["su_lineup"] = ((result["pred_home_lineup"] > result["pred_away_lineup"]).astype(int) == (result["actual_home"] > result["actual_away"]).astype(int)).astype(float)

    result["abs_total_err_baseline"] = ((result["pred_home_baseline"] + result["pred_away_baseline"]) - (result["actual_home"] + result["actual_away"])).abs()
    result["abs_margin_err_baseline"] = ((result["pred_home_baseline"] - result["pred_away_baseline"]) - (result["actual_home"] - result["actual_away"])).abs()
    result["su_baseline"] = ((result["pred_home_baseline"] > result["pred_away_baseline"]).astype(int) == (result["actual_home"] > result["actual_away"]).astype(int)).astype(float)

    arm_lineup = result[["gameId", "abs_total_err_lineup", "abs_margin_err_lineup", "su_lineup"]].rename(
        columns={"abs_total_err_lineup": "abs_total_err", "abs_margin_err_lineup": "abs_margin_err", "su_lineup": "su"})
    arm_baseline = result[["gameId", "abs_total_err_baseline", "abs_margin_err_baseline", "su_baseline"]].rename(
        columns={"abs_total_err_baseline": "abs_total_err", "abs_margin_err_baseline": "abs_margin_err", "su_baseline": "su"})

    bootstrap_compare(
        arm_lineup, arm_baseline, game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="rapm_lineup_oracle", label_b="team_strength_only",
    )

    metrics_ledger.append_run(
        result.rename(columns={"pred_home_lineup": "pred_home", "pred_away_lineup": "pred_away"})[
            ["gameId", "season", "actual_home", "actual_away", "pred_home", "pred_away"]],
        model_name="phase2_rapm_lineup_oracle",
        config_flags={"dev_max_season": DEV_MAX_SEASON, "mode": "oracle_minutes"},
        notes="Phase 2b RAPM-lite lineup adjustment (oracle minutes-share mode) vs Phase 1 baseline",
    )
