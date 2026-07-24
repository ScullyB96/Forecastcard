"""Paired-bootstrap significance testing for full-stack A/B comparisons.

Added 2026-07-22 after an external methodology critique of this project's
own A/B process: at n~600 games, straight-up (SU) accuracy has a binomial
SE of ~2pp (95% CI +/-4pp on a single arm) -- and because `rng` in
validate_game_simulator.py is created once and consumed sequentially across
the whole run, two arms that differ in even one game's outcome diverge in
every subsequent random draw. They are NOT a paired comparison of identical
random draws over the same games, just two independent Monte Carlo runs
that happen to start aligned. Several keep/revert calls this session
(jet-lag -1.2pp, pitch_walk_multiplier -1.4pp, whiff-rate -1.3pp) were
decided on deltas smaller than this combined noise floor.

This module loads two saved `*_validation.parquet` outputs (one per arm,
matched by game_pk) and reports a bootstrap CI on the delta for both SU and
Brier score, so a keep/revert decision can be made on "does the CI exclude
zero" rather than trusting a raw point estimate.
"""

import numpy as np
import pandas as pd

N_BOOTSTRAP = 10000


def _per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["actual_home_win"] = (df["actual_home"] > df["actual_away"]).astype(int)
    # SU picked from trial-level home-win share (sim_home_win_prob > 0.5), not
    # sign of the mean simulated margin -- see validate_game_simulator.py's own
    # 2026-07-22 note (critique claim 6): walk-off truncation skews the mean
    # margin's distribution in a way the direct trial-level win share doesn't.
    df["su"] = ((df["sim_home_win_prob"] > 0.5).astype(int) == df["actual_home_win"]).astype(float)
    df["brier"] = (df["sim_home_win_prob"] - df["actual_home_win"]) ** 2
    return df[["game_pk", "su", "brier"]]


def bootstrap_compare(path_a: str, path_b: str, label_a: str = "A", label_b: str = "B",
                       n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """path_a/path_b: paths to two validate_game_simulator.py-style parquet
    outputs (must share the same game_pk set -- same random_state=1 sample).
    Returns a dict with point estimates, bootstrap CIs, and a plain-English
    verdict for both SU and Brier score. label_a is treated as the "on"/
    treatment arm, label_b as the "off"/baseline arm (delta = a - b, so a
    positive SU delta or NEGATIVE Brier delta both mean "a is better")."""
    a = _per_game_metrics(pd.read_parquet(path_a))
    b = _per_game_metrics(pd.read_parquet(path_b))
    merged = a.merge(b, on="game_pk", suffixes=("_a", "_b"), how="inner")
    n_games = len(merged)
    if n_games == 0:
        raise ValueError("no overlapping game_pk between the two arms -- were they run on the same sample?")

    rng = np.random.default_rng(seed)
    su_deltas = np.empty(n_bootstrap)
    brier_deltas = np.empty(n_bootstrap)
    su_a_arr, su_b_arr = merged["su_a"].to_numpy(), merged["su_b"].to_numpy()
    brier_a_arr, brier_b_arr = merged["brier_a"].to_numpy(), merged["brier_b"].to_numpy()
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_games, n_games)
        su_deltas[i] = su_a_arr[idx].mean() - su_b_arr[idx].mean()
        brier_deltas[i] = brier_a_arr[idx].mean() - brier_b_arr[idx].mean()

    su_point = su_a_arr.mean() - su_b_arr.mean()
    brier_point = brier_a_arr.mean() - brier_b_arr.mean()
    su_ci = np.percentile(su_deltas, [2.5, 97.5])
    brier_ci = np.percentile(brier_deltas, [2.5, 97.5])

    def verdict(point, ci, higher_is_better):
        excludes_zero = ci[0] > 0 or ci[1] < 0
        if not excludes_zero:
            return "NOISE -- CI includes zero, point estimate is not distinguishable from no effect"
        better = (point > 0) if higher_is_better else (point < 0)
        return f"REAL {'IMPROVEMENT' if better else 'REGRESSION'} -- CI excludes zero"

    result = {
        "n_games": n_games,
        "su_a": su_a_arr.mean(), "su_b": su_b_arr.mean(), "su_delta": su_point, "su_delta_ci95": tuple(su_ci),
        "su_verdict": verdict(su_point, su_ci, higher_is_better=True),
        "brier_a": brier_a_arr.mean(), "brier_b": brier_b_arr.mean(), "brier_delta": brier_point,
        "brier_delta_ci95": tuple(brier_ci),
        "brier_verdict": verdict(brier_point, brier_ci, higher_is_better=False),
    }
    print(f"=== paired bootstrap comparison: {label_a} vs {label_b} (n={n_games} games, {n_bootstrap} resamples) ===")
    print(f"SU:    {label_a}={result['su_a']:.3f}  {label_b}={result['su_b']:.3f}  "
          f"delta={su_point:+.4f}  95% CI=({su_ci[0]:+.4f}, {su_ci[1]:+.4f})")
    print(f"       -> {result['su_verdict']}")
    print(f"Brier: {label_a}={result['brier_a']:.4f}  {label_b}={result['brier_b']:.4f}  "
          f"delta={brier_point:+.4f}  95% CI=({brier_ci[0]:+.4f}, {brier_ci[1]:+.4f})")
    print(f"       -> {result['brier_verdict']}")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python -m src.models.ab_significance <arm_a.parquet> <arm_b.parquet>")
        sys.exit(1)
    bootstrap_compare(sys.argv[1], sys.argv[2], label_a="A", label_b="B")
