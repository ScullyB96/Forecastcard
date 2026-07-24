"""Cycle 20 (re-scoped tie-mass fix, per Sec10.6.3's residuals-confirmed
local-transfer shape): replaces Cycle 17's rejected GLOBAL diagonal-inflation
theta (src/models/diagonal_inflation.py, which taxed every off-diagonal cell
uniformly -- including blowouts that needed MORE mass, not less -- and
measurably hurt margin-MAE) with a LOCAL transfer that moves mass only
between each diagonal cell (x,x) and its two immediate neighbors
(x+1,x)/(x,x+1), leaving every |margin|>=2 cell untouched.

Motivating measurement (Sec10.6.3, confirmed before building anything): the
model's excess mass at |margin|=1 (0.1036, both signs) is MORE than the
entire tie deficit (0.0577) -- 180.8% of the offsetting excess concentrated
at |margin|=1 alone, with |margin|>=2 bins net NEGATIVE (the model actually
under-predicts several of them). A single scalar theta that taxes every
off-diagonal cell uniformly was therefore structurally the wrong shape; a
local transfer that ONLY touches the (x,x)/(x+1,x)/(x,x+1) triplet is the
shape the data itself indicates.

Design: for every diagonal level x, move a FRACTION delta of the mass
currently at (x+1,x) and (x,x+1) into (x,x):

    new P(x,x)     = P(x,x) + delta * (P(x+1,x) + P(x,x+1))
    new P(x+1,x)   = P(x+1,x) * (1 - delta)
    new P(x,x+1)   = P(x,x+1) * (1 - delta)

Mass is moved, not rescaled-and-renormalized (unlike Cycle 17's theta), so
the joint is automatically still a valid distribution -- no separate
normalization step needed, and no risk of pushing any cell negative for
delta in [0, 1). delta=0 is an exact no-op.

Closed-form fit, same style as Cycle 17's theta: the average TOTAL diagonal
mass across the fitting population increases by exactly
`delta * (average mass at |margin|=1)`, so matching the real average tie
rate has a direct algebraic solution:

    delta = (real_tie_rate - mean_diag_mass_before) / mean_margin1_mass_before
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson


def _indep_joint(lambda_home: float, lambda_away: float, max_goals: int = 12) -> np.ndarray:
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_pmf, away_pmf)


def diagonal_mass(joint: np.ndarray) -> float:
    return float(np.trace(joint))


def margin1_mass(joint: np.ndarray) -> float:
    """Total mass at |home - away| == 1 (both the (x+1,x) and (x,x+1) cells,
    all x)."""
    n = joint.shape[0]
    total = 0.0
    for x in range(n - 1):
        total += joint[x + 1, x] + joint[x, x + 1]
    return float(total)


def local_transfer(joint: np.ndarray, delta: float) -> np.ndarray:
    """delta=0 is a no-op. delta in [0, 1) moves that fraction of each
    |margin|=1 cell's own mass into its adjacent diagonal cell -- cells with
    |margin|>=2 are completely untouched, unlike Cycle 17's global rescale."""
    if not (0 <= delta < 1):
        raise ValueError(f"delta={delta} outside valid range [0, 1)")
    n = joint.shape[0]
    out = joint.copy()
    for x in range(n - 1):
        moved = delta * (joint[x + 1, x] + joint[x, x + 1])
        out[x, x] += moved
        out[x + 1, x] *= (1 - delta)
        out[x, x + 1] *= (1 - delta)
    return out


def fit_delta(lambda_home: np.ndarray, lambda_away: np.ndarray, reg_home: np.ndarray, reg_away: np.ndarray,
              max_goals: int = 12) -> float:
    """Closed-form: delta = (real average tie rate - average model-implied
    diagonal mass) / average model-implied |margin|=1 mass, since
    local_transfer moves exactly `delta * margin1_mass` of probability mass
    from |margin|=1 onto the diagonal for every game."""
    real_tie_rate = float((reg_home == reg_away).mean())
    diag_masses = []
    m1_masses = []
    for lh, la in zip(lambda_home, lambda_away):
        joint = _indep_joint(lh, la, max_goals)
        diag_masses.append(diagonal_mass(joint))
        m1_masses.append(margin1_mass(joint))
    mean_diag = float(np.mean(diag_masses))
    mean_m1 = float(np.mean(m1_masses))
    return (real_tie_rate - mean_diag) / mean_m1
