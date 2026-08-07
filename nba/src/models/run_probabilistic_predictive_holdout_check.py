"""The one-time confirmatory holdout read for the probabilistic-attendance
predictive-minutes mechanism (task #66, MODEL_DOCUMENTATION.md Sec65):
does `lineup_rating.probabilistic_predictive_minutes_shares` (trailing-
average minutes weighted by `attendance_model.predict_attendance_probability`
instead of a hard include/exclude union) recover a real net win over Phase 1
alone -- the exact thing the CURRENT live predictive-minutes mechanism
(`predictive_minutes_shares`) fails to do, which is why Phase 2 predictive
mode is disabled live (INCLUDE_LINEUP_ADJUSTMENT=False in
generate_predictions.py / run_final_holdout_check.py).

Chosen via dev-only Stage 1 (recent-dev slice sweep of streak_decay=
0.5/0.6/0.7/0.8/0.9): 0.5-0.8 all clear (REAL IMPROVEMENT on total_mae AND
margin_mae, no real regression); 0.9 shows a real su regression. streak_decay=
0.5 picked (largest effect size, delta=-0.0073 total_mae). Stage 2 (full dev
range, n=10,467) confirmed: total_mae delta=-0.0069 (CI entirely negative),
margin_mae delta=-0.0117 (CI entirely negative), su noise.

Run as `python -m src.models.run_probabilistic_predictive_holdout_check` --
ONLY ONCE.
"""

import pandas as pd

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, season_str
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import split_dev_holdout
from src.models.home_court import fit_home_court_walk_forward
from src.models.lineup_rating import (
    player_minutes_from_stints, probabilistic_predictive_minutes_shares,
    project_lineup_adjustment, team_recent_roster_rapm,
)
from src.models.rapm_lite import compute_walkforward_player_ratings, prepare_stints
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check
from src.models.validate_predictive_lineup_adjustment import LOOKBACK_GAMES
from src.models.validate_rapm_lineup_adjustment import _build_team_history
from src.models.validate_team_strength_baseline import _to_wide_games

STREAK_DECAY = 0.5


def _full_range_long_log() -> pd.DataFrame:
    last_season = current_nba_season()
    return add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, last_season))


def build_phase1_predictions(log: pd.DataFrame) -> pd.DataFrame:
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


def build_probabilistic_predictive_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """Phase 1 + Phase 2 (probabilistic predictive-minutes mode), across
    the FULL dev+holdout range -- mirrors `run_final_holdout_check.
    build_phase1_plus_lineup_predictions` exactly, swapping
    `predictive_minutes_shares` for `probabilistic_predictive_minutes_shares`."""
    last_season = current_nba_season()
    frames = []
    for y in range(FIRST_DEV_SEASON, last_season + 1):
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
        home_prior_games = [gid for gdate, gseason, gid in team_history.get(home_id, [])
                             if gdate < row.gameDate and gseason == row.season]
        away_prior_games = [gid for gdate, gseason, gid in team_history.get(away_id, [])
                             if gdate < row.gameDate and gseason == row.season]

        home_recent_off, home_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, home_prior_games, team_side[home_id])
        away_recent_off, away_recent_def = team_recent_roster_rapm(
            player_minutes, ratings_snapshot, away_prior_games, team_side[away_id])

        home_shares = probabilistic_predictive_minutes_shares(
            player_minutes, row.gameId, "home", home_prior_games, team_side[home_id],
            LOOKBACK_GAMES, streak_decay=STREAK_DECAY)
        away_shares = probabilistic_predictive_minutes_shares(
            player_minutes, row.gameId, "away", away_prior_games, team_side[away_id],
            LOOKBACK_GAMES, streak_decay=STREAK_DECAY)
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


def _arms(df: pd.DataFrame, pred_home_col: str, pred_away_col: str) -> pd.DataFrame:
    d = df[["gameId", "season"]].copy()
    d["abs_total_err"] = ((df[pred_home_col] + df[pred_away_col]) - (df["actual_home"] + df["actual_away"])).abs()
    d["abs_margin_err"] = ((df[pred_home_col] - df[pred_away_col]) - (df["actual_home"] - df["actual_away"])).abs()
    d["su"] = ((df[pred_home_col] > df[pred_away_col]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return d


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: probabilistic predictive-minutes mode "
          f"(streak_decay={STREAK_DECAY}) vs Phase 1 baseline ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    long_log = _full_range_long_log()
    print("Building Phase 1 baseline predictions (full dev+holdout range)...", flush=True)
    baseline_full = build_phase1_predictions(long_log)
    print("Building probabilistic-predictive predictions (full dev+holdout range, this is the slow step)...", flush=True)
    candidate_full = build_probabilistic_predictive_predictions(long_log)

    merged = candidate_full.merge(
        baseline_full[["gameId", "pred_home", "pred_away"]].rename(
            columns={"pred_home": "pred_home_base", "pred_away": "pred_away_base"}),
        on="gameId", how="inner")
    print(f"n={len(merged)} games with both predictions, seasons {sorted(merged['season'].unique())}\n", flush=True)

    dev, holdout = split_dev_holdout(merged)

    print("--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    dev_arms = _arms(dev, "pred_home", "pred_away")
    holdout_arms = _arms(holdout, "pred_home", "pred_away")
    generic_holdout_confirmatory_check(dev_arms["abs_total_err"].to_numpy(), holdout_arms["abs_total_err"].to_numpy(),
                                        metric_name="abs_total_err", higher_is_better=False)
    generic_holdout_confirmatory_check(dev_arms["abs_margin_err"].to_numpy(), holdout_arms["abs_margin_err"].to_numpy(),
                                        metric_name="abs_margin_err", higher_is_better=False)

    print("\n--- Check 2: candidate vs Phase 1 baseline, HOLDOUT GAMES ONLY (THE decision-relevant comparison) ---", flush=True)
    holdout_result = bootstrap_compare(
        _arms(holdout, "pred_home", "pred_away"), _arms(holdout, "pred_home_base", "pred_away_base"),
        game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="probabilistic_predictive", label_b="phase1_baseline",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(holdout),
                 "total_mae_delta": holdout_result["total_mae"]["delta"], "total_mae_verdict": holdout_result["total_mae"]["verdict"],
                 "margin_mae_delta": holdout_result["margin_mae"]["delta"], "margin_mae_verdict": holdout_result["margin_mae"]["verdict"],
                 "su_delta": holdout_result["su"]["delta"], "su_verdict": holdout_result["su"]["verdict"]},
        model_name="phase2_probabilistic_predictive_vs_phase1",
        config_flags={"streak_decay": STREAK_DECAY, "lookback_games": LOOKBACK_GAMES},
        notes="One-time confirmatory holdout-only comparison: probabilistic-attendance predictive-minutes mode vs Phase 1 alone",
    )

    print("\n=== SUMMARY ===", flush=True)
    real_improvements = sum(1 for m in ("total_mae", "margin_mae", "su") if "REAL IMPROVEMENT" in holdout_result[m]["verdict"])
    real_regressions = sum(1 for m in ("total_mae", "margin_mae", "su") if "REAL REGRESSION" in holdout_result[m]["verdict"])
    for m in ("total_mae", "margin_mae", "su"):
        print(f"{m}: {holdout_result[m]['verdict']}", flush=True)
    if real_improvements > 0 and real_regressions == 0:
        print("\n-> NET POSITIVE on real holdout. ADOPT (re-enable Phase 2 predictive mode live, using "
              "probabilistic_predictive_minutes_shares, streak_decay=0.5, in place of predictive_minutes_shares).", flush=True)
    elif real_regressions > 0:
        print("\n-> REAL REGRESSION on at least one metric on real holdout -- NOT adopted.", flush=True)
    else:
        print("\n-> NEUTRAL on real holdout (noise) -- NOT adopted (no real out-of-sample gain confirmed).", flush=True)
