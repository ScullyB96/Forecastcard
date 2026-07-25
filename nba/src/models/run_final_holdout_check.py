"""Phase 4: builds the full dev+holdout predictions frame from the FINAL
dev-adopted model configuration, then calls `validate_holdout_bootstrap.
holdout_confirmatory_check` -- the ONE-TIME confirmatory-veto read. See
that module's own docstring for the protocol this must respect: run once,
after every dev-side adoption decision is settled, never re-run after
seeing the result and tweaking anything.

INCLUDE_LINEUP_ADJUSTMENT controls whether Phase 2 (RAPM-lite, predictive-
minutes mode) is part of the scored configuration -- flip this only after
`validate_predictive_lineup_adjustment.py`'s own dev-only bootstrap has
already confirmed a real improvement; this script does not re-derive that
decision, it implements whatever Sec0 of MODEL_DOCUMENTATION.md says is
currently adopted.

Run as `python -m src.models.run_final_holdout_check` -- ONLY ONCE.
"""

import pandas as pd

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models.home_court import fit_home_court_walk_forward
from src.models.lineup_rating import (
    player_minutes_from_stints, predictive_minutes_shares, project_lineup_adjustment, team_recent_roster_rapm,
)
from src.models.rapm_lite import compute_walkforward_player_ratings, prepare_stints
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
from src.models.validate_holdout_bootstrap import holdout_confirmatory_check
from src.models.validate_predictive_lineup_adjustment import LOOKBACK_GAMES
from src.models.validate_rapm_lineup_adjustment import _build_team_history
from src.models.validate_team_strength_baseline import _to_wide_games

INCLUDE_LINEUP_ADJUSTMENT = True  # predictive-minutes mode dev-adopted 2026-07-25: real improvement
# on total_mae (-0.0206, CI excludes zero) and margin_mae (-0.0336, CI excludes zero), matching
# oracle mode's ceiling almost exactly -- see validate_predictive_lineup_adjustment.py / MODEL_
# DOCUMENTATION.md Sec7/Sec9.


def _full_range_long_log() -> pd.DataFrame:
    last_season = current_nba_season()
    return add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, last_season))


def build_phase1_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Phase 1 (team-strength) only, across the FULL dev+holdout range --
    the same construction as `validate_team_strength_baseline.build_dev_predictions`,
    just not truncated to dev-only."""
    games = _to_wide_games(log)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)

    preds = []
    for row in games.itertuples(index=False):
        _, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        preds.append({"gameId": row.gameId, "season": row.season,
                      "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
                      "pred_home": pred_home, "pred_away": pred_away})
    return pd.DataFrame(preds).dropna(subset=["pred_home", "pred_away"])


def build_phase1_plus_lineup_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Phase 1 + Phase 2 (predictive-minutes mode), across the FULL
    dev+holdout range -- same construction as
    `validate_predictive_lineup_adjustment.run_predictive_lineup_backtest`,
    just spanning holdout seasons too (needs their stints built first --
    see `build_season_stints`)."""
    last_season = current_nba_season()
    frames = []
    for y in range(FIRST_DEV_SEASON, last_season + 1):
        from src.ingest.fetch_schedule import season_str
        sched_path = f"data/raw/schedule_{season_str(y)}.parquet"
        try:
            schedule = pd.read_parquet(sched_path)
        except FileNotFoundError:
            continue
        stints = build_season_stints(y)
        if stints.empty:
            continue
        frames.append(prepare_stints(stints, schedule[schedule["seasonType"] == "Regular Season"]))
    if not frames:
        raise RuntimeError("no stints available across the full range")
    all_stints = pd.concat(frames, ignore_index=True)

    player_ratings = compute_walkforward_player_ratings(all_stints)
    if player_ratings.empty:
        raise RuntimeError("not enough stint history yet to fit any RAPM checkpoint")
    player_minutes = player_minutes_from_stints(all_stints)

    games = _to_wide_games(log)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)
    games = games.merge(all_stints[["gameId"]].drop_duplicates(), on="gameId", how="inner")

    team_history, team_side = _build_team_history(log)
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

        home_shares = predictive_minutes_shares(
            player_minutes, row.gameId, "home", home_prior_games, team_side[home_id], LOOKBACK_GAMES)
        away_shares = predictive_minutes_shares(
            player_minutes, row.gameId, "away", away_prior_games, team_side[away_id], LOOKBACK_GAMES)
        if home_shares.empty or away_shares.empty:
            continue

        home_off_adj, home_def_adj = project_lineup_adjustment(home_shares, ratings_snapshot, home_recent_off, home_recent_def)
        away_off_adj, away_def_adj = project_lineup_adjustment(away_shares, ratings_snapshot, away_recent_off, away_recent_def)

        _, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home + home_off_adj, home_dRtg=row.rtg_defense_rate_home + home_def_adj,
            away_oRtg=row.rtg_attack_rate_away + away_off_adj, away_dRtg=row.rtg_defense_rate_away + away_def_adj,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        records.append({"gameId": row.gameId, "season": row.season,
                        "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
                        "pred_home": pred_home, "pred_away": pred_away})
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=== PHASE 4: ONE-TIME CONFIRMATORY HOLDOUT CHECK ===", flush=True)
    print("If this has already been run once for this exact configuration, STOP.", flush=True)
    print(f"INCLUDE_LINEUP_ADJUSTMENT = {INCLUDE_LINEUP_ADJUSTMENT}\n", flush=True)

    long_log = _full_range_long_log()
    full_preds = (build_phase1_plus_lineup_predictions(long_log) if INCLUDE_LINEUP_ADJUSTMENT
                  else build_phase1_predictions(long_log))
    print(f"built {len(full_preds)} full-range (dev+holdout) predictions, "
          f"seasons {sorted(full_preds['season'].unique())}\n", flush=True)

    results = holdout_confirmatory_check(full_preds)

    from src.models import metrics_ledger
    model_name = "phase4_final_with_lineup" if INCLUDE_LINEUP_ADJUSTMENT else "phase4_final_team_strength_only"
    metrics_ledger.append_run(
        full_preds, model_name=model_name,
        config_flags={"include_lineup_adjustment": INCLUDE_LINEUP_ADJUSTMENT},
        notes="Phase 4 one-time confirmatory holdout check -- final adopted configuration, dev+holdout full range",
        extra_metrics={f"holdout_{k}_verdict": v["verdict"] for k, v in results.items()},
    )
    print("\nlogged to metrics_ledger.py", flush=True)
