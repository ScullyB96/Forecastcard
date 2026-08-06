"""Diagnostic (NOT an adoption-gate script): decomposes Phase 2's predictive-mode gap
(`validate_predictive_lineup_adjustment.py`, real margin_mae REGRESSION on both dev and holdout,
Sec28.2) into two separately-measurable failure modes, using `lineup_rating.semi_oracle_minutes_shares`
as a bridge between full ORACLE (real attendance + real minutes) and full PREDICTIVE (trailing-window
union + trailing-average shares):

- ORACLE vs. SEMI-ORACLE gap = pure SHARE-REDISTRIBUTION error: even with perfect knowledge of who's
  active tonight, how much value is lost from not knowing each player's EXACT minutes (foul trouble,
  garbage time, a coach's specific call)?
- SEMI-ORACLE vs. PREDICTIVE gap = pure ATTENDANCE-PREDICTION error: how much value is lost
  specifically from not knowing WHO is active (the injury/rest/healthy-scratch problem)?

This tells us WHICH problem a future minutes-model rebuild actually needs to solve, before
committing to build one. This script spends no holdout read and adopts nothing -- it is pure
dev-only research to inform the SCOPE of a future rebuild, not a validation gate itself.

Run as `python -m src.models.validate_semi_oracle_lineup_adjustment`.
"""

import pandas as pd

from src.models.bootstrap_significance import bootstrap_compare
from src.models.home_court import fit_home_court_walk_forward
from src.models.lineup_rating import (
    oracle_minutes_shares, player_minutes_from_stints, predictive_minutes_shares,
    project_lineup_adjustment, semi_oracle_minutes_shares, team_recent_roster_rapm,
)
from src.models.rapm_lite import compute_walkforward_player_ratings
from src.models.team_strength import project_game
from src.models.validate_rapm_lineup_adjustment import _build_team_history, load_dev_stints
from src.models.validate_team_strength_baseline import _to_wide_games, _rated_dev_log, build_dev_predictions

LOOKBACK_GAMES = 10


def run_all_three_modes() -> pd.DataFrame:
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

        modes = {}
        modes["oracle_home"] = oracle_minutes_shares(player_minutes, row.gameId, "home")
        modes["oracle_away"] = oracle_minutes_shares(player_minutes, row.gameId, "away")
        modes["semi_home"] = semi_oracle_minutes_shares(
            player_minutes, row.gameId, "home", home_prior_games, team_side[home_id], LOOKBACK_GAMES)
        modes["semi_away"] = semi_oracle_minutes_shares(
            player_minutes, row.gameId, "away", away_prior_games, team_side[away_id], LOOKBACK_GAMES)
        modes["pred_home"] = predictive_minutes_shares(
            player_minutes, row.gameId, "home", home_prior_games, team_side[home_id], LOOKBACK_GAMES)
        modes["pred_away"] = predictive_minutes_shares(
            player_minutes, row.gameId, "away", away_prior_games, team_side[away_id], LOOKBACK_GAMES)

        # Only keep games where ALL SIX share vectors are non-empty -- apples-to-apples comparison
        # across all three modes on the exact same game set, not three different game populations.
        if any(v.empty for v in modes.values()):
            continue

        record = {"gameId": row.gameId, "season": row.season,
                   "actual_home": row.actual_home_score, "actual_away": row.actual_away_score}
        for mode_name, home_key, away_key in (
            ("oracle", "oracle_home", "oracle_away"), ("semi", "semi_home", "semi_away"), ("pred", "pred_home", "pred_away"),
        ):
            home_off_adj, home_def_adj = project_lineup_adjustment(modes[home_key], ratings_snapshot, home_recent_off, home_recent_def)
            away_off_adj, away_def_adj = project_lineup_adjustment(modes[away_key], ratings_snapshot, away_recent_off, away_recent_def)
            _, pred_home, pred_away = project_game(
                home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
                league_avg_pace=row.league_avg_pace,
                home_oRtg=row.rtg_attack_rate_home + home_off_adj, home_dRtg=row.rtg_defense_rate_home + home_def_adj,
                away_oRtg=row.rtg_attack_rate_away + away_off_adj, away_dRtg=row.rtg_defense_rate_away + away_def_adj,
                league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
            )
            record[f"pred_home_{mode_name}"] = pred_home
            record[f"pred_away_{mode_name}"] = pred_away
        records.append(record)

    return pd.DataFrame(records)


def _arms(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    d = df[["gameId"]].copy()
    d["abs_total_err"] = ((df[f"pred_home_{mode}"] + df[f"pred_away_{mode}"]) - (df["actual_home"] + df["actual_away"])).abs()
    d["abs_margin_err"] = ((df[f"pred_home_{mode}"] - df[f"pred_away_{mode}"]) - (df["actual_home"] - df["actual_away"])).abs()
    d["su"] = ((df[f"pred_home_{mode}"] > df[f"pred_away_{mode}"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return d


if __name__ == "__main__":
    print("Loading dev stints and fitting walk-forward RAPM-lite checkpoints "
          "(same fits used by all three modes -- only the minutes-share input differs)...", flush=True)
    result = run_all_three_modes()
    print(f"built {len(result)} games with ALL THREE modes available (apples-to-apples comparison)", flush=True)

    baseline = build_dev_predictions()[["gameId", "pred_home", "pred_away"]].rename(
        columns={"pred_home": "pred_home_baseline", "pred_away": "pred_away_baseline"})
    result = result.merge(baseline, on="gameId", how="inner")

    metrics = [
        {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
        {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
        {"name": "su", "col": "su", "higher_is_better": True},
    ]
    arm_oracle = _arms(result, "oracle")
    arm_semi = _arms(result, "semi")
    arm_pred = _arms(result, "pred")
    arm_baseline = _arms(result.rename(columns={"pred_home_baseline": "pred_home_base", "pred_away_baseline": "pred_away_base"}), "base")

    print("\n=== Check 1: each mode vs. Phase 1 baseline (team-strength only) ===", flush=True)
    for name, arm in (("oracle", arm_oracle), ("semi-oracle", arm_semi), ("predictive", arm_pred)):
        print(f"--- {name} vs baseline ---", flush=True)
        bootstrap_compare(arm, arm_baseline, game_id_col="gameId", metrics=metrics, label_a=name, label_b="baseline")

    print("\n=== Check 2 (THE decomposition): oracle vs semi-oracle -- pure SHARE-REDISTRIBUTION error ===", flush=True)
    bootstrap_compare(arm_oracle, arm_semi, game_id_col="gameId", metrics=metrics, label_a="oracle", label_b="semi_oracle")

    print("\n=== Check 3 (THE decomposition): semi-oracle vs predictive -- pure ATTENDANCE-PREDICTION error ===", flush=True)
    bootstrap_compare(arm_semi, arm_pred, game_id_col="gameId", metrics=metrics, label_a="semi_oracle", label_b="predictive")

    from src.models import metrics_ledger
    from src.models.final_holdout_check import DEV_MAX_SEASON
    metrics_ledger.append_generic_run(
        metrics={"n_games": len(result)},
        model_name="phase2_semi_oracle_decomposition",
        config_flags={"dev_max_season": DEV_MAX_SEASON, "lookback_games": LOOKBACK_GAMES},
        notes="Dev-only diagnostic decomposing Phase 2's predictive-mode gap into "
              "share-redistribution error (oracle vs semi-oracle) and attendance-prediction "
              "error (semi-oracle vs predictive). No holdout read, no adoption decision.",
    )
