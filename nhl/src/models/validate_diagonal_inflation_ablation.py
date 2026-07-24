"""theta=1 ablation (external review, follow-up 2026-07-23): runs the IDENTICAL
Sec10 pipeline (regulation-ratio reconstruction + OT/SO layer) with the
diagonal-inflation term switched OFF (theta=1, a pure no-op in
inflate_diagonal), to separate two questions Sec10 conflated:
1. Does the regulation-ratio reconstruction pathway itself (converting
   final-score lambda to a properly-scaled regulation lambda, then adding
   back exactly one goal only for games reaching OT) improve total-MAE on
   its own, with no inflation at all?
2. How much of the model-implied Var(T)/E(T) gap (Sec10.4's corrected,
   population-level version) does theta actually close, versus how much is
   already present (or absent) at theta=1?
"""

import numpy as np

from src.models.validate_diagonal_inflation import _per_game_metrics, paired_bootstrap, run_validation

if __name__ == "__main__":
    base, r_theta1, theta_used, p_home_wins_ot = run_validation(theta_override=1.0)
    print(f"validated {len(r_theta1)} dev-set games, theta FORCED to {theta_used:.4f} (no inflation), "
          f"p_home_wins_ot={p_home_wins_ot:.4f}")

    for label, df in [("BASELINE (drift_adjusted_poisson, production)", base),
                       ("RECONSTRUCTION PATHWAY ONLY (theta=1, no inflation)", r_theta1)]:
        actual_home_win = (df["actual_home"] > df["actual_away"]).astype(int)
        p = df["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
        su = ((p > 0.5).astype(int) == actual_home_win).mean()
        brier = ((p - actual_home_win) ** 2).mean()
        total_mae = ((df["lambda_home"] + df["lambda_away"]) - (df["actual_home"] + df["actual_away"])).abs().mean()
        margin_mae = ((df["lambda_home"] - df["lambda_away"]) - (df["actual_home"] - df["actual_away"])).abs().mean()
        print(f"[{label}] SU={su:.4f} Brier={brier:.5f} total_mae={total_mae:.5f} margin_mae={margin_mae:.5f}")

    print("\npaired bootstrap (5000 resamples), theta=1 reconstruction-only MINUS baseline:")
    boot = paired_bootstrap(base, r_theta1)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b = boot[key]
        print(f"  {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")

    actual_total = r_theta1["actual_home"] + r_theta1["actual_away"]
    real_var_over_mean = actual_total.var(ddof=0) / actual_total.mean()
    pop_var_total = r_theta1["var_total"].mean() + r_theta1["exp_total"].var(ddof=0)
    pop_exp_total = r_theta1["exp_total"].mean()
    print(f"\nVar(T)/E(T), theta=1 (no inflation): model={pop_var_total/pop_exp_total:.4f} "
          f"vs. real={real_var_over_mean:.4f}")
    print("(compare against Sec10.4's theta=1.3338 model figure of 1.0193 on the same real target "
          "0.8913 -- shows how much of the gap theta actually closes)")
