"""Honest holdout re-check for the cross-season team-strength prior (Sec12),
mirroring check_holdout_drift.py's own established methodology exactly: the
walk-forward situational/goalie lambdas are computed across the FULL range
(dev + holdout -- legitimate, since the cross-season prior itself has no
separate "fit" step beyond what's already baked into each game's own
walk-forward computation), but the away-B2B adjustment and the OT/SO
win-rate are fit on DEVELOPMENT-SET-ONLY data, then applied to BOTH splits
without refitting -- an honest look at how this generalizes to games the
constants never saw.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate, full_win_probability
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_current_best
from src.utils.paths import DATA_RAW


def run_full_range_with_dev_fit_constants(cross_season_weight: float | None, min_season: int = 20102011,
                                           prior_minutes_multiplier: float = 1.0) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule_reg_only = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule_reg_only)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)

    base = run_current_best(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=cross_season_weight,
                             prior_minutes_multiplier=prior_minutes_multiplier)
    base = base[base["season"] >= min_season].copy()

    dev_only = base[base["season"] < DEV_MAX_SEASON]
    away_b2b_adj = fit_away_b2b_adjustment(schedule, dev_only["actual_away"], dev_only["lambda_away"],
                                            dev_only["gameId"])

    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(away_rest, on="gameId", how="left")
    # Sec22: symmetric embedded-bias credit (corrected from Sec19.5/Sec20's wrong version).
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    base = base.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(base["p_b2b_walk_forward"], away_b2b_adj)
    base["lambda_home"] = base["lambda_home"] + credit
    base["lambda_away_adj"] = base["lambda_away"] + credit + (base["away_rest"] == 0) * away_b2b_adj

    results = []
    for row in base.itertuples():
        joint = score_distribution(row.lambda_home, row.lambda_away_adj)
        home_win_reg, away_win_reg, tie_prob = home_win_prob_regulation(joint)
        home_win_full, _ = full_win_probability(home_win_reg, tie_prob, p_home_wins_ot)
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": row.lambda_home, "lambda_away": row.lambda_away_adj,
            "home_win_prob": home_win_full, "away_win_prob": 1 - home_win_full, "tie_prob": 0.0,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    mult = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    r = run_full_range_with_dev_fit_constants(cross_season_weight=weight, prior_minutes_multiplier=mult)
    dev = r[r["season"] < DEV_MAX_SEASON]
    holdout = r[r["season"] >= DEV_MAX_SEASON]
    print(f"cross_season_weight={weight}, prior_minutes_multiplier={mult}: development set {len(dev)} games, "
          f"holdout set {len(holdout)} games")

    dev_row = append_run(
        dev, model_name=f"cross_season_prior_w{weight}_m{mult}_DEV_SET_ONLY",
        config_flags={"halflife_games": HALFLIFE_GAMES, "cross_season_weight": weight,
                       "prior_minutes_multiplier": mult, "split": "development"},
        notes=f"Sec13 holdout re-check: development-set-only metrics for cross_season_weight={weight}, "
              f"prior_minutes_multiplier={mult}.",
    )
    holdout_row = append_run(
        holdout, model_name=f"cross_season_prior_w{weight}_m{mult}_HOLDOUT_CHECK",
        config_flags={"halflife_games": HALFLIFE_GAMES, "cross_season_weight": weight,
                       "prior_minutes_multiplier": mult, "split": "holdout"},
        notes=f"Sec13 holdout re-check: honest holdout reading for cross_season_weight={weight}, "
              f"prior_minutes_multiplier={mult}, all constants fit dev-only.",
    )
    print("\n--- development set ---")
    for k, v in dev_row.items():
        print(f"  {k}: {v}")
    print("\n--- holdout set ---")
    for k, v in holdout_row.items():
        print(f"  {k}: {v}")
