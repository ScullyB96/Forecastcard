"""Validate the Monte Carlo drive-state simulator (game_simulator.py) against
real historical games, walk-forward: the drive-resampling pools are built
ONLY from TRAIN=2018-2021 (matching this project's dominant TRAIN/TEST
convention -- see market_blend.py, injury_adjustment.py, weather_adjustment.py),
evaluated on TEST=2022-2025, never touching TEST data when building the pools.
(A live/production version would use a growing window like market_blend.py's
MARKET_FIT_START_SEASON, refit each season on all prior years -- this fixed
split is for comparability with the rest of this project's validated numbers.)

Compared against the CURRENT model's Gaussian point-estimate + fixed-sigma
approximation (src/models/scoring.py) on: CRPS (margin, total), win-probability
calibration (Brier), and straight-up/ATS-style sanity checks.
"""

import numpy as np
import pandas as pd

from src.models.drive_transitions import implied_total_quantile_bin
from src.models.game_simulator import build_simulator_for_season_range
from src.models.market_blend import MARKET_FIT_START_SEASON, apply_blend, fit_market_blend
from src.models.scoring import crps_gaussian, fit_residual_sigma
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
N_TRIALS = 300


def empirical_crps(samples: np.ndarray, y: float) -> float:
    """Unbiased empirical CRPS estimator from a finite sample of the
    predictive distribution: E|X-y| - 0.5*E|X-X'| (X, X' iid draws)."""
    n = len(samples)
    term1 = np.abs(samples - y).mean()
    diffs = np.abs(samples[:, None] - samples[None, :])
    term2 = diffs.sum() / (2 * n * n)
    return float(term1 - term2)


if __name__ == "__main__":
    print("loading data...")
    pbp = pd.read_parquet(f"{DATA_RAW}/pbp_2016_2025.parquet")
    schedules = pd.read_parquet(f"{DATA_RAW}/schedules_2016_2025.parquet")
    layer1 = pd.read_parquet(f"{DATA_PROCESSED}/layer1_games_with_ratings.parquet")

    print("building drive-resampling pools from TRAIN=2018-2021 only (no TEST lookahead)...")
    pbp_train = pbp[pbp["season"].isin(TRAIN_SEASONS)]
    schedules_train = schedules[schedules["season"].isin(TRAIN_SEASONS)]
    sim, bin_edges = build_simulator_for_season_range(pbp_train, schedules_train)
    print(f"  implied-total quantile bin edges (fit on TRAIN): {bin_edges}")

    # --- current model's margin/total base predictions on TEST, for comparison ---
    base_train = layer1[layer1["season"].isin(TRAIN_SEASONS)]
    base_a, base_b = fit_linear(base_train["pregame_rating_diff"], base_train["actual_margin"])
    layer1["base_pred_margin"] = base_a + base_b * layer1["pregame_rating_diff"]

    market = schedules[["game_id", "spread_line", "total_line"]]
    layer1_m = layer1.merge(market, on="game_id", how="left")
    mf = layer1_m[layer1_m["season"].isin(TRAIN_SEASONS)].dropna(subset=["spread_line"])
    margin_blend = fit_market_blend(mf["base_pred_margin"], mf["spread_line"], mf["actual_margin"])

    layer1_m["actual_total"] = layer1_m["home_score"] + layer1_m["away_score"]
    roof = schedules[schedules["game_type"] == "REG"][["game_id", "roof"]]
    layer1_m = layer1_m.merge(roof, on="game_id", how="left")
    layer1_m["is_indoor"] = layer1_m["roof"].isin(["dome", "closed"]).astype(float)
    total_train = layer1_m[layer1_m["season"].isin(TRAIN_SEASONS)]
    X_train_t = np.column_stack([np.ones(len(total_train)), total_train["pregame_total_signal"], total_train["is_indoor"]])
    total_coefs, *_ = np.linalg.lstsq(X_train_t, total_train["actual_total"].to_numpy(), rcond=None)
    layer1_m["base_pred_total"] = total_coefs[0] + total_coefs[1] * layer1_m["pregame_total_signal"] + total_coefs[2] * layer1_m["is_indoor"]
    tf = layer1_m[layer1_m["season"].isin(TRAIN_SEASONS)].dropna(subset=["total_line"])
    total_blend = fit_market_blend(tf["base_pred_total"], tf["total_line"], tf["actual_total"])

    layer1_m["blended_margin"] = layer1_m.apply(lambda r: apply_blend(r["base_pred_margin"], r["spread_line"], margin_blend), axis=1)
    layer1_m["blended_total"] = layer1_m.apply(lambda r: apply_blend(r["base_pred_total"], r["total_line"], total_blend), axis=1)

    sigma_margin = fit_residual_sigma(base_a + base_b * base_train["pregame_rating_diff"], base_train["actual_margin"])
    sigma_total = fit_residual_sigma(tf["base_pred_total"], tf["actual_total"])

    test_games = layer1_m[layer1_m["season"].isin(TEST_SEASONS)].dropna(subset=["team_a", "team_b", "spread_line", "total_line"]).copy()
    print(f"n TEST games: {len(test_games)}")
    test_games["home_implied_total"] = test_games["total_line"] / 2 + test_games["spread_line"] / 2
    test_games["away_implied_total"] = test_games["total_line"] / 2 - test_games["spread_line"] / 2

    print(f"\nsimulating all TEST games ({N_TRIALS} trials each)...")
    sim_margin_mean, sim_margin_std, sim_total_mean, sim_total_std, sim_home_wp = [], [], [], [], []
    sim_margin_crps, sim_total_crps = [], []
    for i, row in enumerate(test_games.itertuples()):
        # each team's OWN market-implied total for THIS specific game (already
        # opponent-adjusted -- no separate offense/defense lookup needed),
        # binned using TRAIN-fit quantile edges
        home_q = implied_total_quantile_bin(row.home_implied_total, bin_edges)
        away_q = implied_total_quantile_bin(row.away_implied_total, bin_edges)

        result = sim.simulate_game(home_q, away_q, n_trials=N_TRIALS)
        sim_margin_mean.append(result["margin_mean"])
        sim_margin_std.append(result["margin_std"])
        sim_total_mean.append(result["total_mean"])
        sim_total_std.append(result["total_std"])
        sim_home_wp.append(result["home_win_prob"])
        sim_margin_crps.append(empirical_crps(result["margins"], row.actual_margin))
        sim_total_crps.append(empirical_crps(result["totals"], row.actual_total))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test_games)} games simulated...")

    test_games["sim_margin_mean"] = sim_margin_mean
    test_games["sim_margin_std"] = sim_margin_std
    test_games["sim_total_mean"] = sim_total_mean
    test_games["sim_total_std"] = sim_total_std
    test_games["sim_home_wp"] = sim_home_wp
    test_games["sim_margin_crps"] = sim_margin_crps
    test_games["sim_total_crps"] = sim_total_crps
    valid = test_games.dropna(subset=["sim_margin_mean"])
    print(f"\nn valid simulated games: {len(valid)}")

    # --- MARGIN comparison ---
    print("\n=== MARGIN ===")
    mae_sim = (valid["sim_margin_mean"] - valid["actual_margin"]).abs().mean()
    mae_current = (valid["blended_margin"] - valid["actual_margin"]).abs().mean()
    print(f"MAE: simulator={mae_sim:.4f}  current blend={mae_current:.4f}")

    crps_sim = valid["sim_margin_crps"].mean()
    crps_current = crps_gaussian(valid["blended_margin"], valid["actual_margin"], sigma_margin)["crps"]
    print(f"CRPS: simulator={crps_sim:.4f}  current (Gaussian approx)={crps_current:.4f}")

    su_sim = (np.sign(valid["sim_margin_mean"]) == np.sign(valid["actual_margin"])).mean()
    su_current = (np.sign(valid["blended_margin"]) == np.sign(valid["actual_margin"])).mean()
    print(f"straight-up: simulator={su_sim:.4f}  current={su_current:.4f}")

    print(f"\nsimulator predicted margin_std (avg): {valid['sim_margin_std'].mean():.2f}  "
          f"(real historical margin std ~13-14)")

    # --- TOTAL comparison ---
    print("\n=== TOTAL ===")
    mae_sim_t = (valid["sim_total_mean"] - valid["actual_total"]).abs().mean()
    mae_current_t = (valid["blended_total"] - valid["actual_total"]).abs().mean()
    print(f"MAE: simulator={mae_sim_t:.4f}  current blend={mae_current_t:.4f}")
    crps_sim_t = valid["sim_total_crps"].mean()
    crps_current_t = crps_gaussian(valid["blended_total"], valid["actual_total"], sigma_total)["crps"]
    print(f"CRPS: simulator={crps_sim_t:.4f}  current (Gaussian approx)={crps_current_t:.4f}")

    # --- WIN PROBABILITY calibration ---
    print("\n=== WIN PROBABILITY CALIBRATION ===")
    from scipy.stats import norm
    valid = valid.copy()
    valid["current_wp"] = norm.cdf(valid["blended_margin"] / sigma_margin)
    valid["actual_home_win"] = (valid["actual_margin"] > 0).astype(float)
    brier_sim = ((valid["sim_home_wp"] - valid["actual_home_win"]) ** 2).mean()
    brier_current = ((valid["current_wp"] - valid["actual_home_win"]) ** 2).mean()
    print(f"Brier: simulator={brier_sim:.4f}  current (Gaussian-implied)={brier_current:.4f}")

    valid["wp_bucket"] = pd.cut(valid["sim_home_wp"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    calib = valid.groupby("wp_bucket", observed=True).agg(
        mean_pred_wp=("sim_home_wp", "mean"), actual_win_rate=("actual_home_win", "mean"), n=("actual_home_win", "size")
    )
    print("\nreliability (simulator win prob vs actual):")
    print(calib)
