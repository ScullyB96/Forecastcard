"""Re-checks the genuine 2024-25/2025-26 holdout (Sec4.8 policy) against the
current best model (`drift_adjusted_poisson`, Sec7.5) -- the holdout
comparison used throughout Sec4.8-4.9 and the drift investigation in Sec7
was all still based on the earlier `goalie_overlay_poisson` snapshot, from
before the OT/SO resolution, rest adjustment, and drift fix existed. This
was the natural, flagged-as-overdue next step once Sec7.5 landed, since
Cycle 13 was specifically motivated by trying to close this gap.

Structural note: every constant used here (away-B2B adjustment,
P(home wins OT/SO), the halflife/prior-scale drift fix itself) is fit on
DEVELOPMENT-SET-ONLY data, then applied to score BOTH the development set
and the holdout -- `validate_rest.run_validation()`'s own chain filters its
returned games to the fit range, so this script re-derives the same
pipeline with the fit/score ranges deliberately decoupled, rather than
reusing that function directly.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate, full_win_probability
from src.models.rest_schedule import add_rest_days, fit_away_b2b_adjustment
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_current_best
from src.utils.paths import DATA_RAW


def run_full_range_with_dev_fit_constants(min_season: int = 20102011) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule_reg_only = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule_reg_only)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)

    # Full range (dev + holdout) with the drift fix baked in, but constants below fit dev-only.
    base = run_current_best(league_avg_halflife_games=HALFLIFE_GAMES)
    base = base[base["season"] >= min_season].copy()

    dev_only = base[base["season"] < DEV_MAX_SEASON]
    away_b2b_adj = fit_away_b2b_adjustment(schedule, dev_only["actual_away"], dev_only["lambda_away"],
                                            dev_only["gameId"])

    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(away_rest, on="gameId", how="left")
    base["lambda_away_adj"] = base["lambda_away"] + (base["away_rest"] == 0) * away_b2b_adj

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
    r = run_full_range_with_dev_fit_constants()
    dev = r[r["season"] < DEV_MAX_SEASON]
    holdout = r[r["season"] >= DEV_MAX_SEASON]
    print(f"development set: {len(dev)} games, holdout set: {len(holdout)} games")

    dev_row = append_run(
        dev, model_name="drift_adjusted_poisson_DEV_SET_ONLY",
        config_flags={"halflife_games": HALFLIFE_GAMES, "prior_minutes_scale": 0.6,
                       "split": "development", "dev_max_season": DEV_MAX_SEASON},
        notes="Sec7.6: development-set-only metrics for drift_adjusted_poisson, re-checking the holdout "
              "gap this cycle was motivated by closing.",
    )
    holdout_row = append_run(
        holdout, model_name="drift_adjusted_poisson_HOLDOUT_CHECK",
        config_flags={"halflife_games": HALFLIFE_GAMES, "prior_minutes_scale": 0.6,
                       "split": "holdout", "dev_max_season": DEV_MAX_SEASON},
        notes="Sec7.6: honest holdout reading (2024-25 + 2025-26) of drift_adjusted_poisson -- all "
              "constants fit dev-only, applied to holdout without refitting. Compare against "
              "goalie_overlay_poisson_HOLDOUT_CHECK (Sec4.8) to see whether the drift fix narrowed the gap.",
    )

    print("\n--- development set ---")
    for k, v in dev_row.items():
        print(f"  {k}: {v}")
    print("\n--- holdout set ---")
    for k, v in holdout_row.items():
        print(f"  {k}: {v}")
