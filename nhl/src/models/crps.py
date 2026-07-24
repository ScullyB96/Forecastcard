"""Discrete CRPS (continuous ranked probability score, a.k.a. ranked
probability score/RPS on a discrete outcome grid) -- added per external
review, MODEL_DOCUMENTATION.md Sec16: total-goals/margin MAE score only the
POINT prediction (the distribution's mean), so a future dependence/
dispersion fix (the off-diagonal term Sec10.6/Sec14 showed is still needed)
could close the entire real Var(T)/E(T) gap (Sec10.6.1: real 0.8913 vs.
model ~1.02) and still show up as a wash in MAE terms -- CRPS is what makes
a distributional-width correction measurable at all.

CRPS(F, y) = sum_k [F(k) - 1{y <= k}]^2, summed over every outcome k in the
distribution's support -- the discrete analogue of the standard continuous
CRPS integral, using a discrete CDF instead of a density."""

import numpy as np


def discrete_crps(pmf: np.ndarray, actual: int) -> float:
    """pmf: array of P(outcome = k) for k = 0, 1, ..., len(pmf)-1 (or, for a
    margin distribution, index 0 corresponds to whatever the caller's own
    minimum margin offset is -- the caller is responsible for the index<->
    value mapping; `actual` must already be expressed in that same index
    space)."""
    cdf = np.cumsum(pmf)
    indicator = (np.arange(len(pmf)) >= actual).astype(float)
    return float(np.sum((cdf - indicator) ** 2))


def total_pmf_from_joint(joint: np.ndarray) -> np.ndarray:
    """joint[x, y] = P(home=x, away=y) -> P(total = x+y), indexed 0..2N."""
    n = joint.shape[0]
    out = np.zeros(2 * n - 1)
    for x in range(n):
        for y in range(n):
            out[x + y] += joint[x, y]
    return out


def margin_pmf_from_joint(joint: np.ndarray) -> tuple[np.ndarray, int]:
    """joint[x, y] = P(home=x, away=y) -> P(margin = x-y), indexed 0..2N-2
    with an offset (returned second) so index 0 corresponds to margin
    -(N-1). Caller must add the offset to a real margin value before
    indexing/calling discrete_crps."""
    n = joint.shape[0]
    offset = n - 1
    out = np.zeros(2 * n - 1)
    for x in range(n):
        for y in range(n):
            out[x - y + offset] += joint[x, y]
    return out, offset
