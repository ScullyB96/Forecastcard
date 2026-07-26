"""Grid search Layer 1 hyperparameters (alpha, off_shrink, def_shrink) using a
nested split so the final held-out test set (2022-2025) is never touched
during tuning:

  - 2016-2017: burn-in (ratings cold-start, not scored)
  - 2018-2020: tune-train (fit calibration a,b for each candidate)
  - 2021:      tune-validation (pick the candidate with lowest MAE here)
  - 2022-2025: final test, evaluated once with the chosen hyperparameters
"""

import itertools

import numpy as np
import pandas as pd

from src.models.ratings import PowerRatingEngine, build_dataset
from src.utils.paths import DATA_PROCESSED
from src.utils.stats import fit_linear

TUNE_TRAIN = {2018, 2019, 2020}
TUNE_VAL = {2021}
FINAL_TRAIN = {2018, 2019, 2020, 2021}
FINAL_TEST = {2022, 2023, 2024, 2025}

ALPHA_GRID = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
OFF_SHRINK_GRID = [0.20, 0.35, 0.50]
DEF_SHRINK_GRID = [0.35, 0.50, 0.65]


def fit_calibration(train: pd.DataFrame) -> tuple[float, float]:
    return fit_linear(train["pregame_rating_diff"], train["actual_margin"])


def mae_for(df: pd.DataFrame, a: float, b: float) -> float:
    pred = a + b * df["pregame_rating_diff"]
    return (pred - df["actual_margin"]).abs().mean()


def run_engine(raw: pd.DataFrame, alpha: float, off_shrink: float, def_shrink: float) -> pd.DataFrame:
    engine = PowerRatingEngine(alpha=alpha, off_shrink=off_shrink, def_shrink=def_shrink)
    matchups = raw.rename(columns={"home_team_x": "team_a", "away_team_x": "team_b"})
    return engine.run_walk_forward(matchups)


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    raw = build_dataset(seasons)

    results = []
    for alpha, off_shrink, def_shrink in itertools.product(ALPHA_GRID, OFF_SHRINK_GRID, DEF_SHRINK_GRID):
        df = run_engine(raw, alpha, off_shrink, def_shrink)
        train = df[df["season"].isin(TUNE_TRAIN)]
        val = df[df["season"].isin(TUNE_VAL)]
        a, b = fit_calibration(train)
        val_mae = mae_for(val, a, b)
        results.append((alpha, off_shrink, def_shrink, val_mae))

    results.sort(key=lambda r: r[3])
    print("Top 10 candidates by 2021 (tune-val) MAE:")
    for alpha, off_shrink, def_shrink, val_mae in results[:10]:
        print(f"  alpha={alpha:.2f} off_shrink={off_shrink:.2f} def_shrink={def_shrink:.2f}  val_MAE={val_mae:.3f}")

    best_alpha, best_off, best_def, _ = results[0]
    print(f"\nBest: alpha={best_alpha}, off_shrink={best_off}, def_shrink={best_def}")

    # Final, once-only evaluation on the untouched test seasons
    df = run_engine(raw, best_alpha, best_off, best_def)
    train = df[df["season"].isin(FINAL_TRAIN)]
    test = df[df["season"].isin(FINAL_TEST)]
    a, b = fit_calibration(train)
    test_mae = mae_for(test, a, b)
    pred = a + b * test["pregame_rating_diff"]
    correct_sign = (np.sign(pred) == np.sign(test["actual_margin"])).mean()
    print(f"\nFinal TEST {sorted(FINAL_TEST)} with tuned hyperparameters:")
    print(f"  calibration: margin = {a:.2f} + {b:.2f} * rating_diff")
    print(f"  MAE = {test_mae:.2f} pts, straight-up win% = {correct_sign:.3f}")

    df.to_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet", index=False)
    with open(DATA_PROCESSED / "layer1_best_hparams.txt", "w") as f:
        f.write(f"alpha={best_alpha}\noff_shrink={best_off}\ndef_shrink={best_def}\na={a}\nb={b}\n")
