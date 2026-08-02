"""The one-time confirmatory holdout read for `own_halflife_games` (full-
model-audit untried-synthesis lever, 2026-08-01) -- recency-weights a team's
OWN within-season history (previously a flat, equally-weighted cumulative
mean), structurally distinct from `cross_season_weight` (early-season prior
only, Sec33) and `league_avg_halflife_games` (shared target only, Sec29):
this is the first lever to touch how a team's own in-season games are
weighted against each other.

Chosen via dev-only Stage 1 (recent-dev slice sweep of 5/10/20/40/80 games,
on top of the already-adopted Sec29/33 defaults): 5 was a real regression (too
aggressive), 10/20/40/80 all cleared the net-win bar, 20 and 40 cleared it most
cleanly (real improvement on BOTH total_mae and margin_mae). Stage 2 (full dev
range, n=10,737 games) confirmed both 20 and 40 beat the current config on
total_mae; 40 ALSO beat it on margin_mae (20 was noise-not-regression there).
40 chosen as the strongest all-around candidate for the one-time holdout read.

Run as `python -m src.models.run_own_halflife_holdout_check` -- ONLY ONCE.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.home_court import fit_home_court_walk_forward
from src.models.shrinkage import add_walk_forward_mean, add_walk_forward_rate
from src.models.team_strength import (CROSS_SEASON_WEIGHT_RATING, LEAGUE_AVG_HALFLIFE_GAMES_RATING,
                                       PRIOR_GAMES_PACE, PRIOR_GAMES_RATING, build_team_game_log, project_game)
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check

CANDIDATE_OWN_HALFLIFE_GAMES = 40.0


def _to_wide_games(log: pd.DataFrame) -> pd.DataFrame:
    home = log[log["is_home"]].copy()
    away = log[~log["is_home"]].copy()
    games = home.merge(away, on=["gameId", "gameDate", "season"], suffixes=("_home", "_away"))
    games = games.rename(columns={
        "actualScore_home": "actual_home_score", "actualScore_away": "actual_away_score",
        "pace_league_avg_home": "league_avg_pace", "rtg_league_avg_home": "league_avg_rtg",
    })
    return games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def _build_predictions(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
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
    return pd.DataFrame(preds)


def _full_dev_and_holdout_fit(own_halflife_games: float | None) -> pd.DataFrame:
    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    log = add_walk_forward_mean(log, "pace", PRIOR_GAMES_PACE, prefix="pace",
                                 cross_season_weight=CROSS_SEASON_WEIGHT_RATING,
                                 league_avg_halflife_games=LEAGUE_AVG_HALFLIFE_GAMES_RATING)
    log = add_walk_forward_rate(log, "offRtg", "defRtg", PRIOR_GAMES_RATING, prefix="rtg",
                                 cross_season_weight=CROSS_SEASON_WEIGHT_RATING,
                                 league_avg_halflife_games=LEAGUE_AVG_HALFLIFE_GAMES_RATING,
                                 own_halflife_games=own_halflife_games)
    games = _to_wide_games(log)
    df = _build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


def _arms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_total_err"] = ((df["pred_home"] + df["pred_away"]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err"] = ((df["pred_home"] - df["pred_away"]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su"] = ((df["pred_home"] > df["pred_away"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return df[["gameId", "season", "abs_total_err", "abs_margin_err", "su"]]


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: own_halflife_games={CANDIDATE_OWN_HALFLIFE_GAMES} "
          f"vs current (own_halflife_games=None) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    print("Building full dev+holdout predictions for BOTH configs...", flush=True)
    candidate_full = _full_dev_and_holdout_fit(CANDIDATE_OWN_HALFLIFE_GAMES)
    current_full = _full_dev_and_holdout_fit(None)

    cand_arm = _arms(candidate_full)
    cand_dev, cand_holdout = split_dev_holdout(cand_arm)
    current_arm = _arms(current_full)
    current_dev, current_holdout = split_dev_holdout(current_arm)

    print("\n--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    gap_results = {}
    for metric, higher_is_better in [("abs_total_err", False), ("abs_margin_err", False), ("su", True)]:
        gap_results[metric] = generic_holdout_confirmatory_check(
            cand_dev[metric].to_numpy(), cand_holdout[metric].to_numpy(), metric_name=metric, higher_is_better=higher_is_better)

    print("\n--- Check 2: candidate vs CURRENT config, holdout games only (THE decision-relevant comparison) ---", flush=True)
    holdout_result = bootstrap_compare(
        cand_holdout, current_holdout, game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a=f"own_halflife_{CANDIDATE_OWN_HALFLIFE_GAMES}", label_b="current_config",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(cand_holdout),
                 "total_mae_delta": holdout_result["total_mae"]["delta"], "total_mae_verdict": holdout_result["total_mae"]["verdict"],
                 "margin_mae_delta": holdout_result["margin_mae"]["delta"], "margin_mae_verdict": holdout_result["margin_mae"]["verdict"],
                 "su_delta": holdout_result["su"]["delta"], "su_verdict": holdout_result["su"]["verdict"]},
        model_name="phase1_own_halflife_vs_current",
        config_flags={"own_halflife_games": CANDIDATE_OWN_HALFLIFE_GAMES},
        notes="One-time confirmatory holdout-only comparison: own_halflife_games=40 vs current Phase 1 config (own_halflife_games=None)",
    )

    print("\n=== SUMMARY ===", flush=True)
    print("Holdout-only, new config vs current config (THE decision-relevant comparison):", flush=True)
    for name in ("total_mae", "margin_mae", "su"):
        print(f"  {name}: {holdout_result[name]['verdict']}", flush=True)

    total_ok = "REAL REGRESSION" not in holdout_result["total_mae"]["verdict"]
    margin_ok = "REAL REGRESSION" not in holdout_result["margin_mae"]["verdict"]
    su_ok = "REAL REGRESSION" not in holdout_result["su"]["verdict"]
    any_real_gain = any("REAL IMPROVEMENT" in holdout_result[m]["verdict"] for m in ("total_mae", "margin_mae", "su"))
    if total_ok and margin_ok and su_ok and any_real_gain:
        print("\n-> NET POSITIVE on real holdout: no regression on any metric, real improvement on "
              "at least one. ADOPT.", flush=True)
    elif total_ok and margin_ok and su_ok:
        print("\n-> NEUTRAL on real holdout: no regression, but no clear real improvement either.", flush=True)
    else:
        print("\n-> REAL REGRESSION on at least one metric on real holdout -- NOT adopted.", flush=True)
