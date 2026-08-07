"""Does temperature (`temp`, real data already in schedules, never used -- only `wind` from
the same table is wired up in weather_adjustment.py) predict QB yards/attempt or WR/TE
yards/target beyond what wind already explains? Real mechanism: cold reduces grip/ball feel
for passers and receivers, same physical intuition already validated for wind, just a
different weather variable from the same real games.

Mirrors weather_adjustment.py's own wind methodology exactly: walk-forward TdRateEngine
(fit on ALL history, since the engine itself is walk-forward -- no leakage), fit the
weather-effect coefficient on TRAIN=2018-2021 residuals only, score on TEST=2022-2025.
Tests temp UNIVARIATE (mirrors wind's own original test) and JOINTLY with wind (temp and
wind have real, if weak, negative correlation -- r=-0.098 -- across real outdoor games, so a
univariate temp effect could partly be wind bleeding through; the joint fit isolates temp's
own independent contribution).
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.models.player_usage import TdRateEngine
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def fit_ols_multi(X_df: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(X_df))] + [X_df[c].to_numpy(dtype=float) for c in X_df.columns])
    yv = y.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    n, k = X.shape
    sigma2 = (resid ** 2).sum() / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, beta / se


def build_rate_table(weekly: pd.DataFrame, schedules: pd.DataFrame, actual_col: str, touch_col: str, min_touch: int):
    reg = schedules[schedules["game_type"] == "REG"][
        ["season", "week", "home_team", "away_team", "wind", "roof", "temp"]
    ]
    home = reg.rename(columns={"home_team": "recent_team"})[["season", "week", "recent_team", "wind", "roof", "temp"]]
    away = reg.rename(columns={"away_team": "recent_team"})[["season", "week", "recent_team", "wind", "roof", "temp"]]
    team_game = pd.concat([home, away])

    weeks = weekly[(weekly["season_type"] == "REG") & (weekly[touch_col] >= min_touch)].merge(
        team_game, on=["season", "week", "recent_team"], how="left"
    )
    league_rate = weeks[actual_col].sum() / weeks[touch_col].sum()
    engine = TdRateEngine(league_rate=league_rate, prior_weight=150.0)
    result = engine.run_walk_forward(weeks, actual_col, touch_col)
    return result


def evaluate(result: pd.DataFrame, actual_col: str, touch_col: str, pregame_col: str, label: str):
    fit_pool = result[(result["roof"] == "outdoors") & result["wind"].notna() & result["temp"].notna()].copy()
    fit_pool["resid"] = fit_pool[actual_col] / fit_pool[touch_col] - fit_pool[pregame_col]
    train = fit_pool[fit_pool["season"].isin(TRAIN_SEASONS)]
    test = fit_pool[fit_pool["season"].isin(TEST_SEASONS)].copy()
    print(f"\n=== {label}: n TRAIN={len(train)}  n TEST={len(test)} ===")

    print("-- univariate: resid ~ temp (mirrors wind's own original test) --")
    a_u, b_u = fit_linear(train["temp"], train["resid"])
    beta_u, t_u = fit_ols_multi(train[["temp"]], train["resid"])
    print(f"TRAIN fit: intercept={a_u:+.5f}  slope={b_u:+.5f}  t-stat={t_u[1]:+.2f}")
    test["adj_uni"] = np.maximum(test[pregame_col] + a_u + b_u * test["temp"], 0.0)
    mae_before = (test[pregame_col] * test[touch_col] - test[actual_col]).abs().mean()
    mae_after_uni = (test["adj_uni"] * test[touch_col] - test[actual_col]).abs().mean()
    t_test = stats.ttest_rel(
        (test[pregame_col] * test[touch_col] - test[actual_col]).abs(),
        (test["adj_uni"] * test[touch_col] - test[actual_col]).abs(),
    )
    print(f"TEST projected-total MAE: {mae_before:.3f} -> {mae_after_uni:.3f}  (paired t-test p={t_test.pvalue:.4f})")

    print("-- joint: resid ~ wind + temp (isolates temp's own contribution, controlling for wind) --")
    beta_j, t_j = fit_ols_multi(train[["wind", "temp"]], train["resid"])
    print(f"TRAIN fit: intercept={beta_j[0]:+.5f}  wind_slope={beta_j[1]:+.5f} (t={t_j[1]:+.2f})  "
          f"temp_slope={beta_j[2]:+.5f} (t={t_j[2]:+.2f})")
    test["adj_joint"] = np.maximum(test[pregame_col] + beta_j[0] + beta_j[1] * test["wind"] + beta_j[2] * test["temp"], 0.0)
    mae_after_joint = (test["adj_joint"] * test[touch_col] - test[actual_col]).abs().mean()
    t_test_j = stats.ttest_rel(
        (test[pregame_col] * test[touch_col] - test[actual_col]).abs(),
        (test["adj_joint"] * test[touch_col] - test[actual_col]).abs(),
    )
    print(f"TEST projected-total MAE: {mae_before:.3f} -> {mae_after_joint:.3f}  (paired t-test p={t_test_j.pvalue:.4f})")


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")

    qb_result = build_rate_table(weekly, schedules, "passing_yards", "attempts", 5)
    evaluate(qb_result, "passing_yards", "attempts", "pregame_passing_yards_rate", "QB yards/attempt")

    wr_result = build_rate_table(weekly, schedules, "receiving_yards", "targets", 1)
    evaluate(wr_result, "receiving_yards", "targets", "pregame_receiving_yards_rate", "WR/TE yards/target")

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
