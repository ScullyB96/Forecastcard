"""Phase 4: the one-time confirmatory holdout check. Read ONCE, at the very
end of a dev-validated cycle, and used ONLY to veto a dev-confirmed
improvement -- never to rank candidates or pick between configurations.
Mirrors the NHL sibling's corrected protocol (that project's own history
records not carving out and respecting a holdout early enough as a real
process gap -- this project's holdout split was frozen in Phase 0/1,
before any dev-side tuning happened, specifically to avoid repeating it).

Reports the same dev-set metrics recomputed on the holdout set (seasons
>= DEV_MAX_SEASON), with a paired bootstrap on the DELTA between dev and
holdout performance -- not holdout performance in isolation, since some
gap between dev and holdout is expected even for a genuinely good model
(a different, more recent set of games) and only a REAL regression should
trigger a veto. Uses the same `"REAL IMPROVEMENT"/"REAL REGRESSION"/"NOISE"`
verdict language as the MLB sibling's `ab_significance.py`.

Run as `python -m src.models.validate_holdout_bootstrap` only once a full
dev-validated model (Phase 1 + adopted Phase 2/3 increments) exists -- not
before.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout


def _per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["abs_total_err"] = ((d["pred_home"] + d["pred_away"]) - (d["actual_home"] + d["actual_away"])).abs()
    d["abs_margin_err"] = ((d["pred_home"] - d["pred_away"]) - (d["actual_home"] - d["actual_away"])).abs()
    d["su"] = ((d["pred_home"] > d["pred_away"]).astype(int) == (d["actual_home"] > d["actual_away"]).astype(int)).astype(float)
    return d


def generic_holdout_confirmatory_check(dev_arr: np.ndarray, holdout_arr: np.ndarray, metric_name: str,
                                        higher_is_better: bool, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """The same dev-vs-holdout-GAP bootstrap as `holdout_confirmatory_check`
    (the ONE-TIME confirmatory-veto read, see module docstring), extracted
    to operate on a single already-computed per-row metric array rather
    than home/away score columns -- so the props subsystem's ~10 different
    per-player-stat categories (each its own MAE, not a score-shaped
    home/away pair) can share this exact protocol without forcing this
    function's score-specific contract to bend, mirroring how
    `metrics_ledger.append_generic_run` was extracted from `append_run` for
    the identical reason."""
    rng = np.random.default_rng(seed)
    dev_point, holdout_point = dev_arr.mean(), holdout_arr.mean()

    dev_boot = np.array([dev_arr[rng.integers(0, len(dev_arr), len(dev_arr))].mean() for _ in range(n_bootstrap)])
    holdout_boot = np.array([holdout_arr[rng.integers(0, len(holdout_arr), len(holdout_arr))].mean() for _ in range(n_bootstrap)])
    gap = holdout_boot - dev_boot
    ci = np.percentile(gap, [2.5, 97.5])
    excludes_zero = ci[0] > 0 or ci[1] < 0
    if not excludes_zero:
        verdict = "NOISE -- dev/holdout gap CI includes zero, no real regression detected"
    else:
        regressed = (holdout_point - dev_point > 0) if not higher_is_better else (holdout_point - dev_point < 0)
        verdict = "REAL REGRESSION -- VETO" if regressed else "REAL IMPROVEMENT on holdout (unusual but not a veto trigger)"

    result = {"dev": float(dev_point), "holdout": float(holdout_point),
              "gap_ci95": (float(ci[0]), float(ci[1])), "verdict": verdict}
    print(f"{metric_name:24s} dev={dev_point:.4f}  holdout={holdout_point:.4f}  "
          f"gap_ci95=({ci[0]:+.4f}, {ci[1]:+.4f})  -> {verdict}", flush=True)
    return result


def holdout_confirmatory_check(full_predictions: pd.DataFrame, n_bootstrap: int = 5000, seed: int = 0) -> dict:
    """full_predictions must span BOTH dev and holdout seasons (gameId,
    season, actual_home, actual_away, pred_home, pred_away) -- produced by
    literally the same, already dev-validated model code, just run across
    the full season range for the first and only time here."""
    dev, holdout = split_dev_holdout(full_predictions)
    if holdout.empty:
        raise RuntimeError("no holdout-range predictions present -- did you actually score seasons >= DEV_MAX_SEASON?")

    dev_m = _per_game_metrics(dev)
    holdout_m = _per_game_metrics(holdout)

    print(f"=== Phase 4 confirmatory holdout check (dev n={len(dev_m)}, holdout n={len(holdout_m)}) ===", flush=True)
    print("THIS IS A ONE-TIME READ. If this has already been run once for this model configuration, "
          "STOP -- re-running after seeing the result and tweaking anything is exactly what the "
          "confirmatory-veto protocol exists to prevent.", flush=True)

    rng = np.random.default_rng(seed)
    results = {}
    for name, col, higher_is_better in (
        ("total_mae", "abs_total_err", False), ("margin_mae", "abs_margin_err", False), ("su", "su", True),
    ):
        dev_arr, holdout_arr = dev_m[col].to_numpy(), holdout_m[col].to_numpy()
        dev_point, holdout_point = dev_arr.mean(), holdout_arr.mean()

        # Paired-in-spirit only within each set (dev and holdout are different games, so this
        # bootstraps each set's own sampling uncertainty around its mean, then compares the two
        # independent bootstrap distributions for the dev-vs-holdout gap).
        dev_boot = np.array([dev_arr[rng.integers(0, len(dev_arr), len(dev_arr))].mean() for _ in range(n_bootstrap)])
        holdout_boot = np.array([holdout_arr[rng.integers(0, len(holdout_arr), len(holdout_arr))].mean() for _ in range(n_bootstrap)])
        gap = holdout_boot - dev_boot
        ci = np.percentile(gap, [2.5, 97.5])
        excludes_zero = ci[0] > 0 or ci[1] < 0
        if not excludes_zero:
            verdict = "NOISE -- dev/holdout gap CI includes zero, no real regression detected"
        else:
            regressed = (holdout_point - dev_point > 0) if not higher_is_better else (holdout_point - dev_point < 0)
            verdict = "REAL REGRESSION -- VETO" if regressed else "REAL IMPROVEMENT on holdout (unusual but not a veto trigger)"

        results[name] = {"dev": float(dev_point), "holdout": float(holdout_point),
                          "gap_ci95": (float(ci[0]), float(ci[1])), "verdict": verdict}
        print(f"{name:12s} dev={dev_point:.4f}  holdout={holdout_point:.4f}  "
              f"gap_ci95=({ci[0]:+.4f}, {ci[1]:+.4f})  -> {verdict}", flush=True)

    return results


if __name__ == "__main__":
    print("This script requires a full-range (dev + holdout) predictions DataFrame from an "
          "already dev-validated model pipeline -- not runnable standalone until Phase 1-3 have "
          "a dev-confirmed configuration to check. See MODEL_DOCUMENTATION.md for current status.", flush=True)
