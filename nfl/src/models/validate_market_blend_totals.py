"""Review round 2, #2.4: the margin market blend was removed after a full ATS%/CRPS/
signed-bias panel found it statistically indistinguishable from a coin flip with a real
negative bias (§4.3) -- but the SAME panel was never run on the total blend, which was kept
on MAE + a single p-value alone (MAE 10.20 vs 10.19, p=0.43, "no evidence of harm"). Totals
now carry a LARGER own-model weight (0.11-0.13) than margin's pre-removal 0.02-0.17, so the
asymmetry in rigor runs exactly the wrong way. This closes that gap with the identical
panel and the identical growing-window methodology weekly_update.py actually uses (not a
single fixed-window approximation), plus a rolling-origin fold check on the blend's own
weight (mirroring §10.5's margin demo, never previously done for totals).

Real kill criterion (mirrors §4.3's margin precedent exactly): if O/U% is statistically
indistinguishable from a coin flip and/or shows a real distinguishable negative bias, the
total blend should be removed the same way margin's was.
"""

import numpy as np
import pandas as pd

from src.models.market_blend import MARKET_FIT_START_SEASON, apply_blend, fit_market_blend
from src.models.scoring import crps_gaussian, fit_residual_sigma, ou_win_rate, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.rolling_origin_cv import sign_consistency

TRAIN_BASE_SEASONS = {2018, 2019, 2020, 2021}  # honest out-of-sample base total_coefs fit
TEST_SEASONS = [2022, 2023, 2024, 2025]
ALL_SEASONS = list(range(2018, 2026))


def fit_total_coefs(layer1_full: pd.DataFrame, train_seasons: set) -> np.ndarray:
    train = layer1_full[layer1_full["season"].isin(train_seasons)]
    X = np.column_stack([np.ones(len(train)), train["pregame_total_signal"], train["is_indoor"]])
    coefs, *_ = np.linalg.lstsq(X, train["actual_total"].to_numpy(), rcond=None)
    return coefs


def predict_total(layer1_full: pd.DataFrame, coefs: np.ndarray) -> pd.Series:
    return coefs[0] + coefs[1] * layer1_full["pregame_total_signal"] + coefs[2] * layer1_full["is_indoor"]


if __name__ == "__main__":
    layer1_full = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    layer1_full["actual_total"] = layer1_full["home_score"] + layer1_full["away_score"]
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    reg_sched = schedules[schedules["game_type"] == "REG"]
    roof_lookup = reg_sched[["game_id", "roof"]].assign(is_indoor=lambda d: d["roof"].isin(["dome", "closed"]).astype(float))
    layer1_full = layer1_full.merge(roof_lookup[["game_id", "is_indoor"]], on="game_id", how="left")
    market = schedules[["game_id", "total_line"]]
    layer1_full = layer1_full.merge(market, on="game_id", how="left")

    # honest out-of-sample base calibration for scoring TEST (matches predict.py's own
    # TRAIN=2018-2021 convention -- NOT weekly_update.py's CALIBRATION_SEASONS, which would
    # leak TEST into the base fit for this backtest specifically)
    base_coefs = fit_total_coefs(layer1_full, TRAIN_BASE_SEASONS)
    layer1_full["our_total_pred"] = predict_total(layer1_full, base_coefs)

    # per-TEST-season growing-window blend, exactly reproducing weekly_update.py's own
    # market_fit_seasons = range(MARKET_FIT_START_SEASON, season) mechanism
    scored_rows = []
    for test_season in TEST_SEASONS:
        market_fit_seasons = set(range(MARKET_FIT_START_SEASON, test_season))
        fit_pool = layer1_full[layer1_full["season"].isin(market_fit_seasons)].dropna(subset=["total_line"])
        blend = fit_market_blend(fit_pool["our_total_pred"], fit_pool["total_line"], fit_pool["actual_total"])

        score_pool = layer1_full[layer1_full["season"] == test_season].dropna(subset=["total_line"]).copy()
        score_pool["blended_pred"] = score_pool.apply(
            lambda r: apply_blend(r["our_total_pred"], r["total_line"], blend), axis=1
        )
        scored_rows.append(score_pool)

    test = pd.concat(scored_rows, ignore_index=True)
    print(f"TEST=2022-2025, n={len(test)}\n")

    print("=== R1.5: full panel on TOTALS, matching margin's rigor ===\n")
    for label, pred_col in [("Our model alone", "our_total_pred"), ("Blend (production)", "blended_pred")]:
        pred = test[pred_col]
        ou = ou_win_rate(pred, test["total_line"], test["actual_total"])
        bias = signed_bias(pred, test["actual_total"])
        sigma = fit_residual_sigma(pred, test["actual_total"])
        crps = crps_gaussian(pred, test["actual_total"], sigma)
        mae = (pred - test["actual_total"]).abs().mean()
        print(f"{label:20s} MAE={mae:.3f}  O/U%={ou['win_rate']:.4f} (n={ou['n']}, CI=[{ou['ci'][0]:.3f},{ou['ci'][1]:.3f}])"
              f"  bias={bias['bias']:+.3f} (CI=[{bias['ci'][0]:+.3f},{bias['ci'][1]:+.3f}])  CRPS={crps['crps']:.3f}")

    sigma_market = fit_residual_sigma(test["total_line"], test["actual_total"])
    crps_market = crps_gaussian(test["total_line"], test["actual_total"], sigma_market)
    bias_market = signed_bias(test["total_line"], test["actual_total"])
    mae_market = (test["total_line"] - test["actual_total"]).abs().mean()
    print(f"{'Market alone':20s} MAE={mae_market:.3f}  O/U%=-- (trivial)"
          f"  bias={bias_market['bias']:+.3f} (CI=[{bias_market['ci'][0]:+.3f},{bias_market['ci'][1]:+.3f}])  CRPS={crps_market['crps']:.3f}")

    print("\n=== R1.5 / S5.1 (review round 3, #4): rolling-origin CV on the total blend's own-model weight ===")
    print("Round 2 reported sign consistency only. Round 3 correctly pointed out this doesn't")
    print("match how margin's removal was actually decided (§10.5's own demo used 'beats")
    print("market-alone in X of 8 folds' as the decisive statistic) -- added here for parity.")
    rows = []
    for i in range(4, len(ALL_SEASONS)):
        train_seasons = set(ALL_SEASONS[:i])
        test_season = ALL_SEASONS[i]
        base = fit_total_coefs(layer1_full, train_seasons)
        layer1_full["_tmp_pred"] = predict_total(layer1_full, base)
        fit_pool = layer1_full[layer1_full["season"].isin(train_seasons)].dropna(subset=["total_line"])
        blend = fit_market_blend(fit_pool["_tmp_pred"], fit_pool["total_line"], fit_pool["actual_total"])

        fold_test = layer1_full[layer1_full["season"] == test_season].dropna(subset=["total_line"]).copy()
        fold_test["_blended_pred"] = fold_test.apply(lambda r: apply_blend(r["_tmp_pred"], r["total_line"], blend), axis=1)
        blend_mae = (fold_test["_blended_pred"] - fold_test["actual_total"]).abs().mean()
        market_mae = (fold_test["total_line"] - fold_test["actual_total"]).abs().mean()
        rows.append({
            "train_seasons": f"{min(train_seasons)}-{max(train_seasons)}", "test_season": test_season,
            "our_weight": blend["our_weight"], "blend_mae": blend_mae, "market_mae": market_mae,
            "blend_beats_market": blend_mae < market_mae,
        })
    fold_df = pd.DataFrame(rows)
    print(fold_df.round(4).to_string(index=False))
    print(f"\nsign consistency: {sign_consistency(fold_df, 'our_weight'):.3f}")
    print(f"magnitude range: [{fold_df['our_weight'].min():.4f}, {fold_df['our_weight'].max():.4f}]  median={fold_df['our_weight'].median():.4f}")
    n_beats = int(fold_df["blend_beats_market"].sum())
    print(f"blend beats market-alone on MAE in {n_beats} of {len(fold_df)} folds")

    print("\n=== S5.1: statistical-power check on the O/U% kill criterion ===")
    print("Before treating the O/U% test above as decisive: what's the best-case O/U% this")
    print("component's own weight could possibly produce, even with a PERFECTLY informative signal?")
    disagreement = test["blended_pred"] - test["total_line"]
    disagreement_sd = disagreement.std()
    avg_weight = fold_df["our_weight"].mean()
    best_case_ou = 0.5 + 0.02 * disagreement_sd  # ~2%/point O/U-probability sensitivity near the line
    print(f"SD(blended_pred - total_line) = {disagreement_sd:.2f} points  (avg blend weight ~{avg_weight:.3f})")
    print(f"best-case O/U% at a PERFECTLY informative signal of this size: ~{best_case_ou:.3f}")
    print(f"observed O/U% CI from the panel above should be checked against this -- if the CI")
    print(f"contains this best-case value, the test had no power to distinguish real signal from")
    print(f"noise, and the decision must rest on bias/CRPS/fold-consistency instead (see §4.3).")
