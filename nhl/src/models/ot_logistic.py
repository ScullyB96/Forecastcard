"""Cycle 22 (Sec17): team-specific OT win probability, fit ONLY for
OT-decided games (lastPeriodType=="OT") -- the residuals-first check
(validate_ot_logistic_check.py) confirmed real signal there (r=0.0842,
p=0.0001, n=2209) but NONE in shootout-decided games (r=0.0140, p=0.576,
n=1601), exactly matching the prior expectation that real 3v3 hockey skill
should predict OT outcomes while a shootout is a narrower, more
luck-driven skills competition. Per that finding, shootouts keep the
existing flat empirical rate (no team-specific component would be honest
to add given the null result); only the OT-decided share of tie mass gets
the new team-specific logistic.

Single global logistic (a, b), fit ONCE on dev-set OT-decided games --
same "fit once, not walk-forward per-game" precedent as
overtime_shootout.fit_ot_home_win_rate itself (each game's own lambda_diff
IS already walk-forward; only the two logistic coefficients are fit as a
single dev-set constant)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ot_logistic(lambda_diff: np.ndarray, home_won: np.ndarray) -> tuple[float, float]:
    """Returns (a, b) for P(home wins OT) = sigmoid(a + b*lambda_diff), fit
    by MLE (statsmodels Logit) on real OT-decided games."""
    X = sm.add_constant(lambda_diff)
    model = sm.Logit(home_won, X).fit(disp=0)
    a, b = model.params[0], model.params[1]
    return float(a), float(b)


def predict_ot_logistic_prob(a: float, b: float, lambda_diff) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * lambda_diff)))


def fit_ot_so_split(schedule_with_ot: pd.DataFrame, max_season_exclusive: int) -> dict:
    """Empirical, fixed constants (fit once on the dev set): the share of
    OT/SO games actually decided in OT vs. needing a shootout, and the
    flat (non-team-specific) home win rate in shootouts specifically."""
    dev = schedule_with_ot[(schedule_with_ot["gameType"] == 2) & (schedule_with_ot["went_to_ot"])
                            & (schedule_with_ot["season"] >= 20102011)
                            & (schedule_with_ot["season"] < max_season_exclusive)]
    n_total = len(dev)
    ot_only = dev[dev["lastPeriodType"] == "OT"]
    so_only = dev[dev["lastPeriodType"] == "SO"]
    return {
        "p_decided_in_ot": len(ot_only) / n_total,
        "p_needs_so": len(so_only) / n_total,
        "p_home_wins_so_flat": float((so_only["homeScore"] > so_only["awayScore"]).mean()),
    }
