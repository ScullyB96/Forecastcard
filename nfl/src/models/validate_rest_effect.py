"""Does rest differential (days since each team's last game -- `home_rest`/`away_rest`,
already in the real schedules data, never used anywhere in this project) predict the
market-line residual? Real, testable mechanism: a team on a short week (Thursday games,
4-6 days) has less practice/recovery time than a team coming off a normal week or a bye
(10-17 days) -- distinct from the already-tested-and-rejected "clinched/eliminated rest in
weeks 17-18" scenario (§13.2), which is about teams resting starters late in a lost season,
not this general week-to-week rest mechanism that applies to every game all year.

Walk-forward: TRAIN=2018-2021 fit, TEST=2022-2025 held out, matching this project's standard
discipline. Tests both MARGIN (does the better-rested team beat the market's expectation?)
and TOTAL (does a short week suppress pace/scoring?).
"""

import numpy as np
import pandas as pd

from src.models.scoring import crps_gaussian, fit_residual_sigma, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
SHORT_WEEK_MAX = 6   # <=6 days rest (Thursday/moved-up games)
LONG_REST_MIN = 10   # >=10 days rest (bye week or similar)


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


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    reg = schedules[schedules["game_type"] == "REG"][
        ["game_id", "spread_line", "total_line", "home_rest", "away_rest", "weekday"]
    ]
    df = layer1.merge(reg, on="game_id", how="left").dropna(subset=["spread_line", "total_line", "home_rest", "away_rest"])
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["market_margin"] = df["spread_line"]
    df["resid_margin"] = df["actual_margin"] - df["market_margin"]
    df["resid_total"] = df["actual_total"] - df["total_line"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n TRAIN={len(train)}  n TEST={len(test)}\n")

    print("=== MARGIN: resid_margin ~ rest_diff ===")
    a_m, b_m, t_m = fit_ols_tstat(train["rest_diff"], train["resid_margin"])
    print(f"TRAIN fit: intercept={a_m:+.4f}  slope={b_m:+.4f}  t-stat={t_m:+.2f}")
    test["adj_margin"] = test["market_margin"] + a_m + b_m * test["rest_diff"]
    sigma_m = fit_residual_sigma(train["market_margin"], train["actual_margin"])
    mae_before = (test["market_margin"] - test["actual_margin"]).abs().mean()
    mae_after = (test["adj_margin"] - test["actual_margin"]).abs().mean()
    crps_before = crps_gaussian(test["market_margin"], test["actual_margin"], sigma_m)["crps"]
    crps_after = crps_gaussian(test["adj_margin"], test["actual_margin"], sigma_m)["crps"]
    print(f"TEST (all games, n={len(test)}): MAE {mae_before:.4f} -> {mae_after:.4f}   CRPS {crps_before:.4f} -> {crps_after:.4f}")

    print("\n=== MARGIN, short-week subset (either team on <=%d days rest) ===" % SHORT_WEEK_MAX)
    sw = test[(test["home_rest"] <= SHORT_WEEK_MAX) | (test["away_rest"] <= SHORT_WEEK_MAX)]
    mae_before_sw = (sw["market_margin"] - sw["actual_margin"]).abs().mean()
    mae_after_sw = (sw["adj_margin"] - sw["actual_margin"]).abs().mean()
    bias_before_sw = signed_bias(sw["market_margin"], sw["actual_margin"])
    bias_after_sw = signed_bias(sw["adj_margin"], sw["actual_margin"])
    print(f"n={len(sw)}: MAE {mae_before_sw:.4f} -> {mae_after_sw:.4f}")
    print(f"signed_bias (market alone) = {bias_before_sw['bias']:+.3f}  CI=[{bias_before_sw['ci'][0]:+.3f},{bias_before_sw['ci'][1]:+.3f}]")
    print(f"signed_bias (rest-adjusted) = {bias_after_sw['bias']:+.3f}  CI=[{bias_after_sw['ci'][0]:+.3f},{bias_after_sw['ci'][1]:+.3f}]")

    print("\n=== MARGIN, long-rest subset (either team on >=%d days rest, e.g. bye week) ===" % LONG_REST_MIN)
    lr = test[(test["home_rest"] >= LONG_REST_MIN) | (test["away_rest"] >= LONG_REST_MIN)]
    bias_before_lr = signed_bias(lr["market_margin"], lr["actual_margin"])
    print(f"n={len(lr)}: signed_bias (market alone) = {bias_before_lr['bias']:+.3f}  CI=[{bias_before_lr['ci'][0]:+.3f},{bias_before_lr['ci'][1]:+.3f}]")

    print("\n=== TOTAL: resid_total ~ rest_diff (does short week suppress pace/scoring?) ===")
    a_t, b_t, t_t = fit_ols_tstat(train["rest_diff"], train["resid_total"])
    print(f"TRAIN fit: intercept={a_t:+.4f}  slope={b_t:+.4f}  t-stat={t_t:+.2f}")
    test["adj_total"] = test["total_line"] + a_t + b_t * test["rest_diff"]
    sigma_t = fit_residual_sigma(train["total_line"], train["actual_total"])
    mae_before_t = (test["total_line"] - test["actual_total"]).abs().mean()
    mae_after_t = (test["adj_total"] - test["actual_total"]).abs().mean()
    print(f"TEST (all games, n={len(test)}): MAE {mae_before_t:.4f} -> {mae_after_t:.4f}")

    print("\n=== TOTAL, using a symmetric 'either team short-rested' flag instead of signed diff ===")
    print("(fatigue should suppress scoring regardless of WHICH team is short-rested -- a signed")
    print(" rest_diff term can't capture that; test the flag-based mechanism directly)")
    train_flag = train.copy()
    train_flag["either_short"] = ((train_flag["home_rest"] <= SHORT_WEEK_MAX) | (train_flag["away_rest"] <= SHORT_WEEK_MAX)).astype(float)
    a_tf, b_tf, t_tf = fit_ols_tstat(train_flag["either_short"], train_flag["resid_total"])
    print(f"TRAIN fit: intercept={a_tf:+.4f}  slope={b_tf:+.4f}  t-stat={t_tf:+.2f}")
    test["either_short"] = ((test["home_rest"] <= SHORT_WEEK_MAX) | (test["away_rest"] <= SHORT_WEEK_MAX)).astype(float)
    flagged = test[test["either_short"] == 1.0]
    bias_flagged = signed_bias(flagged["total_line"], flagged["actual_total"])
    print(f"n={len(flagged)}: signed_bias (market alone, on flagged games) = {bias_flagged['bias']:+.3f}  "
          f"CI=[{bias_flagged['ci'][0]:+.3f},{bias_flagged['ci'][1]:+.3f}]")
    print("(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
