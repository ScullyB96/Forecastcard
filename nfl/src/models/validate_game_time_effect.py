"""Does game timing -- day of week (Thursday/Monday short-week games) or primetime slot
(national TV night games) -- predict the market-line residual, independent of the rest-
differential mechanism already tested and rejected (validate_rest_effect.py)? Real,
testable candidates people commonly claim: primetime games run "sloppier" (less prep
focus, more media disruption) or conversely "sharper" (national-spotlight energy);
Thursday games specifically compound short rest with a truncated week of preparation.

Real data already in schedules (`weekday`, `gametime`), never used. Walk-forward:
TRAIN=2018-2021 fit, TEST=2022-2025 held out, same discipline as every other check in
this project. Tests MARGIN and TOTAL, and reports each flag's raw incidence so a null
result can't be blamed on a starved sample.
"""

import numpy as np
import pandas as pd

from src.models.scoring import crps_gaussian, fit_residual_sigma, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def fit_ols_tstat(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    a, b = fit_linear(x, y)
    resid = y.to_numpy() - (a + b * x.to_numpy())
    n = len(x)
    if n <= 2 or np.allclose(x, x.iloc[0]):
        return a, b, 0.0
    sigma2 = (resid ** 2).sum() / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se_b = np.sqrt(sigma2 / sxx) if sxx > 0 else np.inf
    return a, b, (b / se_b if se_b > 0 else 0.0)


def evaluate_flag(train, test, flag_col, resid_col, base_col, actual_col, sigma, label):
    a, b, t = fit_ols_tstat(train[flag_col], train[resid_col])
    n_train_flagged = int(train[flag_col].sum())
    n_test_flagged = int(test[flag_col].sum())
    print(f"\n-- {label} --  (n TRAIN flagged={n_train_flagged}, n TEST flagged={n_test_flagged})")
    print(f"TRAIN fit: intercept={a:+.4f}  slope={b:+.4f}  t-stat={t:+.2f}")
    test = test.copy()
    test["adj"] = test[base_col] + a + b * test[flag_col]
    mae_before = (test[base_col] - test[actual_col]).abs().mean()
    mae_after = (test["adj"] - test[actual_col]).abs().mean()
    crps_before = crps_gaussian(test[base_col], test[actual_col], sigma)["crps"]
    crps_after = crps_gaussian(test["adj"], test[actual_col], sigma)["crps"]
    print(f"TEST (all games, n={len(test)}): MAE {mae_before:.4f} -> {mae_after:.4f}   CRPS {crps_before:.4f} -> {crps_after:.4f}")
    flagged = test[test[flag_col] == 1.0]
    if len(flagged) > 5:
        bias = signed_bias(flagged[base_col], flagged[actual_col])
        print(f"signed_bias on flagged TEST subset (market alone, n={len(flagged)}) = {bias['bias']:+.3f}  "
              f"CI=[{bias['ci'][0]:+.3f},{bias['ci'][1]:+.3f}]")


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    reg = schedules[schedules["game_type"] == "REG"][
        ["game_id", "spread_line", "total_line", "weekday", "gametime"]
    ]
    df = layer1.merge(reg, on="game_id", how="left").dropna(subset=["spread_line", "total_line", "weekday", "gametime"])
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["market_margin"] = df["spread_line"]
    df["resid_margin"] = df["actual_margin"] - df["market_margin"]
    df["resid_total"] = df["actual_total"] - df["total_line"]

    df["kickoff_hour"] = df["gametime"].str.split(":").str[0].astype(int)
    df["is_thursday"] = (df["weekday"] == "Thursday").astype(float)
    df["is_monday"] = (df["weekday"] == "Monday").astype(float)
    df["is_primetime"] = (df["kickoff_hour"] >= 19).astype(float)
    df["is_early_intl"] = (df["kickoff_hour"] < 12).astype(float)  # London/international 9:30am ET games

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n TRAIN={len(train)}  n TEST={len(test)}")
    sigma_m = fit_residual_sigma(train["market_margin"], train["actual_margin"])
    sigma_t = fit_residual_sigma(train["total_line"], train["actual_total"])

    print("\n" + "=" * 70)
    print("MARGIN")
    print("=" * 70)
    for flag, label in [("is_thursday", "Thursday game"), ("is_monday", "Monday Night Football"),
                         ("is_primetime", "Any primetime (kickoff >= 19:00 ET)"),
                         ("is_early_intl", "International/London early game (<12:00 ET)")]:
        evaluate_flag(train, test, flag, "resid_margin", "market_margin", "actual_margin", sigma_m, label)

    print("\n" + "=" * 70)
    print("TOTAL")
    print("=" * 70)
    for flag, label in [("is_thursday", "Thursday game"), ("is_monday", "Monday Night Football"),
                         ("is_primetime", "Any primetime (kickoff >= 19:00 ET)"),
                         ("is_early_intl", "International/London early game (<12:00 ET)")]:
        evaluate_flag(train, test, flag, "resid_total", "total_line", "actual_total", sigma_t, label)

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
