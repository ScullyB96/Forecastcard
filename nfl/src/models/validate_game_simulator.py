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

**2026-08 re-validation**: the "current model" comparison baseline below was
originally `blended_margin`/`blended_total` (the market/own-model blend fit via
`fit_market_blend`) -- both blends were REMOVED from production in Phase 0/§4.3
(the real shipped margin/total base is now the market `spread_line`/`total_line`
directly), so that baseline had gone stale relative to what's actually shipped.
Also added: the exact recentering the live pipeline applies before deriving
win probability/moneyline (`weekly_update.py`, review round 4) was verified
"correct by construction" on a single live game but never actually re-run
through this TEST-set walk-forward harness -- the Brier/CRPS numbers on record
predate that fix entirely. Both fixed here: baseline is now real
`spread_line`/`total_line`, and `sim_home_wp`/`sim_margin_crps`/`sim_total_crps`
are computed BOTH raw and recentered so the recentering's actual effect on
calibration (not just its internal consistency) is measured directly.
"""

import numpy as np
import pandas as pd

from src.models.drive_transitions import implied_total_quantile_bin
from src.models.game_simulator import build_simulator_for_season_range
from src.models.scoring import crps_gaussian, fit_residual_sigma
from src.utils.paths import DATA_PROCESSED, DATA_RAW

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
    # Real shipped baseline (§4.3): market spread_line/total_line DIRECTLY, not a blend --
    # both blends were removed from production. sigma fit on TRAIN residuals only.
    market = schedules[["game_id", "spread_line", "total_line"]]
    layer1_m = layer1.merge(market, on="game_id", how="left")
    layer1_m["actual_total"] = layer1_m["home_score"] + layer1_m["away_score"]

    train_mask = layer1_m["season"].isin(TRAIN_SEASONS)
    mf = layer1_m[train_mask].dropna(subset=["spread_line"])
    tf = layer1_m[train_mask].dropna(subset=["total_line"])
    sigma_margin = fit_residual_sigma(mf["spread_line"], mf["actual_margin"])
    sigma_total = fit_residual_sigma(tf["total_line"], tf["actual_total"])

    test_games = layer1_m[layer1_m["season"].isin(TEST_SEASONS)].dropna(subset=["team_a", "team_b", "spread_line", "total_line"]).copy()
    print(f"n TEST games: {len(test_games)}")
    test_games["home_implied_total"] = test_games["total_line"] / 2 + test_games["spread_line"] / 2
    test_games["away_implied_total"] = test_games["total_line"] / 2 - test_games["spread_line"] / 2

    print(f"\nsimulating all TEST games ({N_TRIALS} trials each)...")
    sim_margin_mean, sim_margin_std, sim_total_mean, sim_total_std = [], [], [], []
    raw_home_wp, recentered_home_wp = [], []
    raw_margin_crps, recentered_margin_crps, raw_total_crps, recentered_total_crps = [], [], [], []
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

        # Recenter EXACTLY as weekly_update.py does (review round 4): shift the raw simulated
        # margin/total samples so their mean lands on the real market line (spread_line/
        # total_line -- the exact "current model" baseline this row is being compared
        # against), keeping the simulator's own variance/skew/key-number shape intact.
        recentered_margins = result["margins"] - result["margin_mean"] + row.spread_line
        recentered_totals = result["totals"] - result["total_mean"] + row.total_line

        raw_home_wp.append(result["home_win_prob"])
        recentered_home_wp.append(float((recentered_margins > 0).mean()))
        raw_margin_crps.append(empirical_crps(result["margins"], row.actual_margin))
        recentered_margin_crps.append(empirical_crps(recentered_margins, row.actual_margin))
        raw_total_crps.append(empirical_crps(result["totals"], row.actual_total))
        recentered_total_crps.append(empirical_crps(recentered_totals, row.actual_total))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test_games)} games simulated...")

    test_games["sim_margin_mean"] = sim_margin_mean
    test_games["sim_margin_std"] = sim_margin_std
    test_games["sim_total_mean"] = sim_total_mean
    test_games["sim_total_std"] = sim_total_std
    test_games["raw_home_wp"] = raw_home_wp
    test_games["recentered_home_wp"] = recentered_home_wp
    test_games["raw_margin_crps"] = raw_margin_crps
    test_games["recentered_margin_crps"] = recentered_margin_crps
    test_games["raw_total_crps"] = raw_total_crps
    test_games["recentered_total_crps"] = recentered_total_crps
    valid = test_games.dropna(subset=["sim_margin_mean"])
    print(f"\nn valid simulated games: {len(valid)}")

    # --- MARGIN comparison (current model = real spread_line, direct, per §4.3) ---
    print("\n=== MARGIN ===")
    mae_sim = (valid["sim_margin_mean"] - valid["actual_margin"]).abs().mean()
    mae_current = (valid["spread_line"] - valid["actual_margin"]).abs().mean()
    print(f"MAE: simulator (raw)={mae_sim:.4f}  current (market spread_line)={mae_current:.4f}")

    crps_raw = valid["raw_margin_crps"].mean()
    crps_recentered = valid["recentered_margin_crps"].mean()
    crps_current = crps_gaussian(valid["spread_line"], valid["actual_margin"], sigma_margin)["crps"]
    print(f"CRPS: simulator raw={crps_raw:.4f}  simulator recentered={crps_recentered:.4f}  "
          f"current (Gaussian approx)={crps_current:.4f}")

    su_sim = (np.sign(valid["sim_margin_mean"]) == np.sign(valid["actual_margin"])).mean()
    su_current = (np.sign(valid["spread_line"]) == np.sign(valid["actual_margin"])).mean()
    print(f"straight-up: simulator (raw)={su_sim:.4f}  current={su_current:.4f}")

    print(f"\nsimulator predicted margin_std (avg): {valid['sim_margin_std'].mean():.2f}  "
          f"(real historical margin std ~13-14)")

    # --- TOTAL comparison (current model = real total_line, direct, per §4.3) ---
    print("\n=== TOTAL ===")
    mae_sim_t = (valid["sim_total_mean"] - valid["actual_total"]).abs().mean()
    mae_current_t = (valid["total_line"] - valid["actual_total"]).abs().mean()
    print(f"MAE: simulator (raw)={mae_sim_t:.4f}  current (market total_line)={mae_current_t:.4f}")
    crps_raw_t = valid["raw_total_crps"].mean()
    crps_recentered_t = valid["recentered_total_crps"].mean()
    crps_current_t = crps_gaussian(valid["total_line"], valid["actual_total"], sigma_total)["crps"]
    print(f"CRPS: simulator raw={crps_raw_t:.4f}  simulator recentered={crps_recentered_t:.4f}  "
          f"current (Gaussian approx)={crps_current_t:.4f}")

    # --- WIN PROBABILITY calibration: raw sim vs. recentered sim (the actual shipped
    # mechanism, review round 4) vs. current model's Gaussian-implied wp ---
    print("\n=== WIN PROBABILITY CALIBRATION ===")
    from scipy.stats import norm
    valid = valid.copy()
    valid["current_wp"] = norm.cdf(valid["spread_line"] / sigma_margin)
    valid["actual_home_win"] = (valid["actual_margin"] > 0).astype(float)
    brier_raw = ((valid["raw_home_wp"] - valid["actual_home_win"]) ** 2).mean()
    brier_recentered = ((valid["recentered_home_wp"] - valid["actual_home_win"]) ** 2).mean()
    brier_current = ((valid["current_wp"] - valid["actual_home_win"]) ** 2).mean()
    print(f"Brier: simulator (raw)={brier_raw:.4f}  simulator (recentered, shipped)={brier_recentered:.4f}  "
          f"current (Gaussian-implied)={brier_current:.4f}")

    valid["wp_bucket"] = pd.cut(valid["recentered_home_wp"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    calib = valid.groupby("wp_bucket", observed=True).agg(
        mean_pred_wp=("recentered_home_wp", "mean"), actual_win_rate=("actual_home_win", "mean"), n=("actual_home_win", "size")
    )
    print("\nreliability (RECENTERED, shipped win prob vs actual):")
    print(calib)
