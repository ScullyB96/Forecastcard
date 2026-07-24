"""Corrected holdout protocol (per external review, 2026-07-23, MODEL_DOCUMENTATION.md
Sec15): the holdout comparisons in Sec12.4 and Sec13.2 used raw point
estimates with no confidence intervals -- on 2,624 holdout games, several of
the differences treated as decisive were on the order of 1-2 games, while
dev-set verdicts are held to a 5,000-resample paired-bootstrap standard.
This script re-runs the SAME holdout comparisons with the identical paired
bootstrap standard used everywhere else in this project, to answer: which of
the holdout-based rejections were actually statistically real, and which
were noise dressed up as a verdict?

Holdout's corrected role (adopted here, going forward): CONFIRMATORY VETO
ONLY. A dev-set-confirmed real improvement is rejected on holdout grounds
if and only if the holdout paired bootstrap shows a REAL regression (CI
does not cross zero) on a metric that was real on the dev set. Holdout
point estimates are NEVER used to rank or choose between multiple
dev-confirmed candidates -- that is what the dev-set bootstrap is for, and
using holdout for ranking too collapses it into a second dev set, defeating
its purpose as a genuinely held-out check.
"""

import numpy as np
import pandas as pd

from src.models.check_holdout_cross_season import run_full_range_with_dev_fit_constants
from src.models.final_holdout_check import DEV_MAX_SEASON


def _per_game_metrics(df: pd.DataFrame) -> dict:
    actual_home_win = (df["actual_home"] > df["actual_away"]).astype(int).values
    p = df["home_win_prob"].clip(1e-6, 1 - 1e-6).values
    su_correct = ((p > 0.5).astype(int) == actual_home_win).astype(float)
    brier = (p - actual_home_win) ** 2
    total_ae = ((df["lambda_home"] + df["lambda_away"]) - (df["actual_home"] + df["actual_away"])).abs().values
    margin_ae = ((df["lambda_home"] - df["lambda_away"]) - (df["actual_home"] - df["actual_away"])).abs().values
    return {"su": su_correct, "brier": brier, "total_ae": total_ae, "margin_ae": margin_ae}


def paired_bootstrap(base: pd.DataFrame, treated: pd.DataFrame, n_resamples: int = 5000,
                      seed: int = 20260723) -> dict:
    base_m = _per_game_metrics(base)
    treated_m = _per_game_metrics(treated)
    n = len(base)
    rng = np.random.default_rng(seed)
    idx_draws = rng.integers(0, n, size=(n_resamples, n))
    out = {}
    for key in ("su", "brier", "total_ae", "margin_ae"):
        diff = treated_m[key] - base_m[key]
        boot_means = diff[idx_draws].mean(axis=1)
        out[key] = {"mean_diff": float(diff.mean()), "ci_low": float(np.percentile(boot_means, 2.5)),
                    "ci_high": float(np.percentile(boot_means, 97.5))}
    return out


def _holdout_only(weight, mult=1.0):
    r = run_full_range_with_dev_fit_constants(cross_season_weight=(None if weight == 0.0 else weight),
                                               prior_minutes_multiplier=mult)
    return r[r["season"] >= DEV_MAX_SEASON].reset_index(drop=True)


if __name__ == "__main__":
    print("Computing holdout-only per-game results for each configuration "
          "(this re-derives the full pipeline several times -- may take a few minutes)\n")

    holdout_w0 = _holdout_only(0.0)
    holdout_w05 = _holdout_only(0.5)
    holdout_w10 = _holdout_only(1.0)
    print(f"n holdout games: {len(holdout_w0)}\n")

    print("=== Cycle 18 (Sec12.4): weight=0.5 vs weight=0.0, HOLDOUT paired bootstrap ===")
    boot = paired_bootstrap(holdout_w0, holdout_w05)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b = boot[key]
        print(f"  {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")

    print("\n=== Cycle 18 (Sec12.4): weight=1.0 vs weight=0.0, HOLDOUT paired bootstrap ===")
    boot = paired_bootstrap(holdout_w0, holdout_w10)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b = boot[key]
        print(f"  {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")

    print("\n=== Cycle 18 (Sec12.4): weight=1.0 vs weight=0.5 (direct), HOLDOUT paired bootstrap ===")
    boot = paired_bootstrap(holdout_w05, holdout_w10)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b = boot[key]
        print(f"  {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")

    print("\n=== Cycle 19 (Sec13.2): weight=0.75/mult=X vs weight=0.5/mult=1.0 (current best), "
          "HOLDOUT paired bootstrap ===")
    for mult in (1.5, 2.0, 3.0):
        holdout_candidate = _holdout_only(0.75, mult)
        boot = paired_bootstrap(holdout_w05, holdout_candidate)
        print(f"\n  weight=0.75, mult={mult}:")
        for key in ("su", "brier", "total_ae", "margin_ae"):
            b = boot[key]
            print(f"    {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")
