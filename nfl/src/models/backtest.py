"""Calibrate the EPA rating-diff -> point-margin scaling on a train window,
then evaluate walk-forward predictions on a held-out test window.

The rating_diff itself is already leakage-free (computed pre-game in
ratings.py). This script additionally holds the *calibration* (the linear
mapping from rating_diff to points) to a train/test split, and burns in the
first two seasons so ratings have converged from their season-1 cold start
before either window begins.

Superseded by tune.py (which additionally grid-searches the rating engine's
hyperparameters with a proper nested train/val/test split) -- kept as a
simpler, quick sanity check on the current tuned defaults.
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED

BURN_IN_SEASONS = {2016, 2017}
TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def fit_calibration(train: pd.DataFrame) -> tuple[float, float]:
    x = train["pregame_rating_diff"].to_numpy()
    y = train["actual_margin"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    return a, b


def evaluate(df: pd.DataFrame, a: float, b: float, label: str) -> dict:
    pred = a + b * df["pregame_rating_diff"]
    actual = df["actual_margin"]
    mae = (pred - actual).abs().mean()
    rmse = np.sqrt(((pred - actual) ** 2).mean())
    correct_sign = (np.sign(pred) == np.sign(actual)).mean()

    naive_zero_mae = actual.abs().mean()
    home_win_rate = (actual > 0).mean()

    print(f"\n--- {label} (n={len(df)}) ---")
    print(f"model MAE:        {mae:.2f} pts")
    print(f"model RMSE:       {rmse:.2f} pts")
    print(f"model straight-up win %: {correct_sign:.3f}")
    print(f"naive (predict 0) MAE:   {naive_zero_mae:.2f} pts")
    print(f"actual home win rate:    {home_win_rate:.3f}")
    return {"mae": mae, "rmse": rmse, "correct_sign": correct_sign}


if __name__ == "__main__":
    df = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]

    a, b = fit_calibration(train)
    print(f"calibration fit on {sorted(TRAIN_SEASONS)}: margin = {a:.2f} + {b:.2f} * rating_diff")
    print(f"(a = implied home-field advantage in points, b = EPA/play-diff-to-points scale)")

    evaluate(train, a, b, f"TRAIN {sorted(TRAIN_SEASONS)}")
    evaluate(test, a, b, f"TEST {sorted(TEST_SEASONS)} (held out, walk-forward)")

    # naive baseline for context: predict mean train margin for every test game (HFA-only, no team skill)
    naive_pred = train["actual_margin"].mean()
    naive_mae = (test["actual_margin"] - naive_pred).abs().mean()
    print(f"\nHFA-only baseline (predict {naive_pred:.2f} every game) TEST MAE: {naive_mae:.2f} pts")
