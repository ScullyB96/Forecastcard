"""Phase 2 (predictive-minutes mode)'s acceptance gate: does the RAPM-lite
active-lineup adjustment still improve predictions when minutes are
PROJECTED from each active player's own trailing average, rather than
taken from the real, already-known minutes they happened to play that
specific night (`validate_rapm_lineup_adjustment.py`'s ORACLE mode, a
ceiling test -- see `lineup_rating.py`'s docstring for the oracle-vs-
predictive distinction).

This is the honest question oracle mode's real dev-range result (Sec7)
could NOT answer on its own: oracle mode proved lineup-awareness carries
real signal IN PRINCIPLE, but a live/backtest system never has real
in-game minutes ahead of time. Predictive mode is what could actually be
deployed -- if it doesn't ALSO show a real improvement, oracle mode's
result would be a proof of a ceiling that isn't practically reachable, not
evidence this specific deployable design captures real value.

Dev-only, same confirmatory-veto discipline as Phase 1/2b's validation
scripts. Run as `python -m src.models.validate_predictive_lineup_adjustment`.
"""

import pandas as pd

from src.models.bootstrap_significance import bootstrap_compare
from src.models.home_court import fit_home_court_walk_forward
from src.models.lineup_rating import (
    player_minutes_from_stints, predictive_minutes_shares, project_lineup_adjustment, team_recent_roster_rapm,
)
from src.models.rapm_lite import compute_walkforward_player_ratings
from src.models.team_strength import project_game
from src.models.validate_rapm_lineup_adjustment import _build_team_history, load_dev_stints
from src.models.validate_team_strength_baseline import _rated_dev_log, _to_wide_games, build_dev_predictions

LOOKBACK_GAMES = 10  # matches lineup_rating.DEFAULT_LOOKBACK_GAMES and generate_predictions.py's live default


def run_predictive_lineup_backtest() -> pd.DataFrame:
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
        home_prior_games = [gid for gdate, gid in team_history.get(home_id, []) if gdate < row.gameDate]
        away_prior_games = [gid for gdate, gid in team_history.get(away_id, []) if gdate < row.gameDate]

        home_recent_off, home_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, home_prior_games, team_side[home_id])
        away_recent_off, away_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, away_prior_games, team_side[away_id])

        # The one substantive difference from the oracle backtest: PROJECTED minutes shares
        # (each active player's own trailing average, same active-player set as the real game),
        # not the real in-game minutes they happened to play that night.
        home_shares = predictive_minutes_shares(
            player_minutes, row.gameId, "home", home_prior_games, team_side[home_id], LOOKBACK_GAMES)
        away_shares = predictive_minutes_shares(
            player_minutes, row.gameId, "away", away_prior_games, team_side[away_id], LOOKBACK_GAMES)
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
    print("Loading dev stints and fitting walk-forward RAPM-lite checkpoints "
          "(same fits as oracle mode -- only the minutes-share input differs)...", flush=True)
    result = run_predictive_lineup_backtest()
    print(f"built {len(result)} predictive lineup-adjusted predictions", flush=True)

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
        label_a="rapm_lineup_predictive", label_b="team_strength_only",
    )

    from src.models import metrics_ledger
    from src.models.final_holdout_check import DEV_MAX_SEASON
    metrics_ledger.append_run(
        result.rename(columns={"pred_home_lineup": "pred_home", "pred_away_lineup": "pred_away"})[
            ["gameId", "season", "actual_home", "actual_away", "pred_home", "pred_away"]],
        model_name="phase2_rapm_lineup_predictive",
        config_flags={"dev_max_season": DEV_MAX_SEASON, "mode": "predictive_minutes", "lookback_games": LOOKBACK_GAMES},
        notes="Phase 2 RAPM-lite lineup adjustment (PREDICTIVE minutes-share mode -- trailing "
              "average, not real in-game minutes) vs Phase 1 baseline",
    )
