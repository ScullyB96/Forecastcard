"""Tests for the degradation-budget check script (Sec37.4/RUNBOOK.md),
using synthetic data with a known ground truth -- confirms the verdict
logic actually fires correctly (escalation / routine breach / within
budget / live-outperforming), not just that the script runs without
crashing on real (currently tiny-sample) data.

Run: `.venv/bin/python -m tests.test_check_live_degradation`
"""

import numpy as np

from src.models.check_live_degradation import _bootstrap_gap, _verdict

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_within_budget_verdict():
    """Live and holdout drawn from the SAME distribution -- the gap CI
    should comfortably include zero and stay under the routine budget."""
    rng = np.random.default_rng(0)
    live = rng.normal(2.0, 0.5, 300)
    holdout = rng.normal(2.0, 0.5, 2624)
    _, lo, hi = _bootstrap_gap(live, holdout)
    verdict = _verdict("margin_mae", lo, hi)
    check("identical-distribution live/holdout verdicts as within budget", "within budget" in verdict, verdict)


def test_routine_breach_verdict():
    """Live margin_mae real-worse than holdout by an amount between the
    routine budget (0.00382) and the escalation ceiling (0.01371). Effect
    size and noise chosen so the 95% CI's lower bound reliably clears
    0.00382 while the upper bound stays comfortably under 0.01371 (worked
    out analytically: SE dominated by the live arm, ~300 games at sd=0.03
    gives a CI half-width of ~0.0036 around an 0.008 point gap)."""
    rng = np.random.default_rng(1)
    holdout = rng.normal(2.0, 0.03, 2624)
    live = rng.normal(2.008, 0.03, 300)
    _, lo, hi = _bootstrap_gap(live, holdout)
    verdict = _verdict("margin_mae", lo, hi)
    check("a real gap between routine budget and escalation ceiling triggers ROUTINE BREACH",
          "ROUTINE BREACH" in verdict, f"verdict={verdict!r}, CI=[{lo:.5f},{hi:.5f}]")
    check("the routine-breach verdict for margin_mae specifically names the goalie pipeline",
          "goalie" in verdict.lower(), verdict)


def test_escalation_verdict():
    """Live margin_mae real-worse than holdout by MORE than the escalation
    ceiling (0.01371) -- a genuine, unbudgeted anomaly."""
    rng = np.random.default_rng(2)
    holdout = rng.normal(2.0, 0.05, 2624)
    live = rng.normal(2.05, 0.05, 300)  # ~0.05 gap, well past escalation
    _, lo, hi = _bootstrap_gap(live, holdout)
    verdict = _verdict("margin_mae", lo, hi)
    check("a gap beyond the escalation ceiling triggers ESCALATION", "ESCALATION" in verdict, verdict)


def test_live_outperforming_verdict():
    """Live REAL-BETTER than holdout -- should be noted, not treated as a
    problem."""
    rng = np.random.default_rng(3)
    holdout = rng.normal(2.0, 0.05, 2624)
    live = rng.normal(1.95, 0.05, 300)  # live has LOWER (better) error
    _, lo, hi = _bootstrap_gap(live, holdout)
    verdict = _verdict("margin_mae", lo, hi)
    check("live real-outperforming holdout is noted, not flagged as a problem",
          "outperforming" in verdict, verdict)


if __name__ == "__main__":
    test_within_budget_verdict()
    test_routine_breach_verdict()
    test_escalation_verdict()
    test_live_outperforming_verdict()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    else:
        print("ALL CHECKS PASSED")
