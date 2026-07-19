"""End-to-end prediction pipeline: Layer 1 (EPA power ratings) + QB-swap
adjustment, fully calibrated. This is the same code path that will predict
2026-27 week 1 once that schedule is released -- it just walks the full
history through to "now" and predicts whatever games come next.

Run as a module to see a concrete demonstration: predictions for a real,
already-played week, compared against both the actual result and the Vegas
closing line for that week.
"""

import numpy as np
import pandas as pd

from src.models.injury_adjustment import apply_joint_adjustment, compute_injury_flags
from src.models.qb_adjustment import (
    QbRatingEngine,
    build_qb_week_table,
    build_starter_sequence,
)
from src.models.ratings import PowerRatingEngine, build_dataset
from src.utils.paths import DATA_RAW, DATA_PROCESSED

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
DEMO_SEASON = 2025
DEMO_WEEK = 18


def fit_calibration(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    return a, b


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    raw = build_dataset(seasons)
    matchups = raw.rename(columns={"home_team_x": "team_a", "away_team_x": "team_b"})

    rating_engine = PowerRatingEngine()
    rated = rating_engine.run_walk_forward(matchups)

    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    qb_weeks = build_qb_week_table(weekly)
    starters = build_starter_sequence(schedules)
    qb_engine = QbRatingEngine()
    starters = qb_engine.run_with_starters(starters, qb_weeks)
    starters["swap_delta"] = np.where(
        starters["qb_changed"], starters["pregame_qb_rating"] - starters["prev_qb_rating"], 0.0
    ).astype(float)
    starters["swap_delta"] = starters["swap_delta"].fillna(0.0)

    home_swap = starters.rename(columns={"team": "home_team_x2", "swap_delta": "home_swap_delta"})[
        ["game_id", "home_team_x2", "home_swap_delta"]
    ]
    away_swap = starters.rename(columns={"team": "away_team_x2", "swap_delta": "away_swap_delta"})[
        ["game_id", "away_team_x2", "away_swap_delta"]
    ]
    rated = rated.merge(
        home_swap, left_on=["game_id", "team_a"], right_on=["game_id", "home_team_x2"], how="left"
    ).merge(away_swap, left_on=["game_id", "team_b"], right_on=["game_id", "away_team_x2"], how="left")
    rated["home_swap_delta"] = rated["home_swap_delta"].fillna(0.0)
    rated["away_swap_delta"] = rated["away_swap_delta"].fillna(0.0)
    rated["net_swap_delta"] = rated["home_swap_delta"] - rated["away_swap_delta"]

    train = rated[rated["season"].isin(TRAIN_SEASONS)]
    base_a, base_b = fit_calibration(train["pregame_rating_diff"], train["actual_margin"])
    rated["base_pred"] = base_a + base_b * rated["pregame_rating_diff"]
    train_resid = train["actual_margin"] - rated.loc[train.index, "base_pred"]
    swap_a, swap_b = fit_calibration(train["net_swap_delta"], train_resid)
    rated["qb_adjusted_pred"] = rated["base_pred"] + swap_a + swap_b * rated["net_swap_delta"]

    injury_flags = compute_injury_flags()
    rated["injury_adjustment"] = apply_joint_adjustment(rated, injury_flags)
    rated["final_pred"] = rated["qb_adjusted_pred"] + rated["injury_adjustment"]

    # --- demonstration: a real, already-played week ---
    demo = rated[(rated["season"] == DEMO_SEASON) & (rated["week"] == DEMO_WEEK)].copy()
    sched_lines = schedules[schedules["game_type"] == "REG"][["game_id", "spread_line"]]
    demo = demo.merge(sched_lines, on="game_id", how="left")
    demo["vegas_pred"] = demo["spread_line"]
    demo["our_error"] = (demo["final_pred"] - demo["actual_margin"]).abs()
    demo["vegas_error"] = (demo["vegas_pred"] - demo["actual_margin"]).abs()

    print(f"=== {DEMO_SEASON} Week {DEMO_WEEK}: predicted vs actual (walk-forward, no lookahead) ===\n")
    cols = ["team_a", "team_b", "final_pred", "vegas_pred", "actual_margin", "our_error", "vegas_error"]
    display = demo[cols].rename(
        columns={
            "team_a": "home",
            "team_b": "away",
            "final_pred": "our_pred",
            "actual_margin": "actual",
        }
    )
    display = display.round(1)
    print(display.to_string(index=False))
    print(f"\nweek {DEMO_WEEK} our MAE: {demo['our_error'].mean():.2f}   vegas MAE: {demo['vegas_error'].mean():.2f}")

    # --- current, most-up-to-date ratings (this is what would seed 2026-27 week 1) ---
    print("\n=== current power ratings (as of end of loaded data) ===")
    ratings_df = pd.DataFrame(
        {
            "team": list(rating_engine.off_ratings.keys()),
            "off_rating": list(rating_engine.off_ratings.values()),
            "def_rating": [rating_engine.def_ratings[t] for t in rating_engine.off_ratings],
        }
    )
    ratings_df["net_rating"] = ratings_df["off_rating"] + ratings_df["def_rating"] * -1
    ratings_df = ratings_df.sort_values("net_rating", ascending=False).reset_index(drop=True)
    print(ratings_df.head(10).round(4).to_string(index=False))
    print("...")
    print(ratings_df.tail(5).round(4).to_string(index=False))

    ratings_df.to_parquet(DATA_PROCESSED / "current_team_ratings.parquet", index=False)
    print("\nsaved current_team_ratings.parquet")
