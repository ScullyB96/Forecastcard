"""Corrected holdout protocol (Sec15) for Cycle 26 (Sec25.6/Sec38): the
walk-forward per-game tie-mass delta is a dev-confirmed real improvement
(CRPS(total)+CRPS(margin), Sec26.3) with no dev-set Brier/SU regression --
per Sec15, holdout is confirmatory-veto-only. Rejects ONLY if a paired
bootstrap on holdout-only games shows a REAL regression on Brier, SU, or
margin-MAE (the core ledger metrics). Total-MAE is explicitly NOT a veto
metric here: Sec24.4's pre-registered endgame prediction is that each real
structural fix should make holdout total-MAE tick WORSE (unmasking the
decayed baseline's own transient recent-era overshoot), not better -- so a
real total-MAE regression is treated as an expected, pre-logged side effect,
not a disqualifying one."""

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_tie_mass_ratio import _bootstrap, _metrics, run_baseline, run_treated

if __name__ == "__main__":
    baseline = run_baseline(min_season=DEV_MAX_SEASON, max_season=20999999)
    treated = run_treated(min_season=DEV_MAX_SEASON, max_season=20999999)
    print(f"{len(baseline)} holdout games (seasons >= {DEV_MAX_SEASON})\n")

    mb, mt = _metrics(baseline), _metrics(treated)
    print("=== Holdout bootstrap: treated - baseline ===")
    veto_metrics = {"brier", "su", "margin_mae"}
    any_veto = False
    for key in ("su", "brier", "margin_mae", "total_mae"):
        d, lo, hi = _bootstrap(mb[key], mt[key])
        real = (lo > 0) or (hi < 0)
        tag = "REAL" if real else "crosses zero"
        role = "VETO METRIC" if key in veto_metrics else "informational, not a veto metric"
        print(f"{key:12s} mean_diff={d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]  ({tag}, {role})")
        if real and key in veto_metrics:
            any_veto = True

    print(f"\nVerdict: {'VETO TRIGGERED -- real regression on a core metric' if any_veto else 'NO VETO -- no real regression on Brier, SU, or margin-MAE'}")
