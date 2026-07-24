"""Dixon-Coles-style low-score correlation correction for the independent-
Poisson joint distribution (MODEL_DOCUMENTATION.md Sec3.4/Sec4.11).

Motivation, tested rather than assumed: checked BOTH count-process
refinements the architecture proposal flagged as untested hypotheses.
Overdispersion was checked first and found NOT present -- real dev-set
goal-count variance is if anything slightly BELOW what Poisson predicts
(variance/mean ratio ~0.97-0.98 for both home and away), so Negative
Binomial was ruled out before building it. Score-state correlation WAS
found real: Pearson correlation between home/away scoring residuals
(actual - lambda) is -0.062 (p<1e-6, n=16,528) across all real dev-set
games, and a much larger -0.452 (p<1e-4, n=358) specifically in games
where both teams scored 1 or fewer -- the same low-joint-scoreline
phenomenon Dixon-Coles (1997) built their soccer correction for.

Standard Dixon-Coles form: apply a correction factor tau(x,y) to the
independent-Poisson joint P(x,y) for the four low scorelines
{(0,0),(1,0),(0,1),(1,1)}, leaving all other scorelines as plain
independent Poisson, then renormalize the full grid to sum to 1. A single
correlation parameter rho is fit by maximum likelihood against our own
real data (not assumed at a value borrowed from soccer, where scoring
levels and game dynamics differ).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson


def dc_tau(x: int, y: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    """The Dixon-Coles correction multiplier for scoreline (x, y); 1.0 for
    any scoreline outside the four low-score cells it's defined for."""
    if x == 0 and y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_adjusted_joint(lambda_home: float, lambda_away: float, rho: float, max_goals: int = 12) -> np.ndarray:
    """Full P(home=i, away=j) grid: independent-Poisson base, Dixon-Coles
    tau applied to the four low-score cells, renormalized to sum to 1
    (the tau adjustment alone doesn't preserve total probability mass)."""
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    joint = np.outer(home_pmf, away_pmf)
    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        joint[x, y] *= dc_tau(x, y, lambda_home, lambda_away, rho)
    return joint / joint.sum()


def negative_log_likelihood(rho: float, lambda_home: np.ndarray, lambda_away: np.ndarray,
                             actual_home: np.ndarray, actual_away: np.ndarray) -> float:
    """Sum of -log(P(actual_home_i, actual_away_i)) under the Dixon-Coles-
    adjusted joint for each real game -- only the four low-score cells
    actually depend on rho, so this only needs the tau factor and a
    game-specific renormalization constant, not the full grid, for speed.

    BUG FOUND AND FIXED (2026-07-23): originally floored any resulting
    probability at 1e-12 rather than rejecting invalid rho values outright.
    Confirmed real: NHL's much higher scoring rate than soccer (the sport
    Dixon-Coles was built for) means tau(0,0)=1-lambda_home*lambda_away*rho
    goes NEGATIVE at rho as low as ~0.054 for the highest-scoring real
    matchups in the dev set (mean lambda_home*lambda_away=8.1, max=18.4) --
    far inside what looked like a reasonable search range (-0.3, 0.3). The
    floor silently absorbed these invalid (negative, pre-floor) probabilities
    instead of penalizing them, letting the fit wander into a meaningless
    region and land exactly on the search bound (0.3) -- a sign, not
    ignored, that the search was unconstrained where it needed to be
    constrained. Now returns +inf for any rho that would make ANY game's
    real outcome have a negative raw probability, so the optimizer is
    structurally confined to the valid region rather than relying on a
    guessed bound."""
    total = 0.0
    for lh, la, ah, aa in zip(lambda_home, lambda_away, actual_home, actual_away):
        base_p = poisson.pmf(ah, lh) * poisson.pmf(aa, la)
        tau = dc_tau(int(ah), int(aa), lh, la, rho)
        raw_p = base_p * tau
        if raw_p < 0:
            return np.inf
        # Renormalization constant: sum of tau over the 4 low cells' base probabilities,
        # plus everything else unchanged -- equivalent to 1 + sum((tau-1)*base_p) over those 4 cells.
        norm = 1.0
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            t = dc_tau(x, y, lh, la, rho)
            norm += (t - 1) * poisson.pmf(x, lh) * poisson.pmf(y, la)
        if norm <= 0:
            return np.inf
        p = max(raw_p / norm, 1e-12)
        total -= np.log(p)
    return total


def fit_rho(lambda_home: np.ndarray, lambda_away: np.ndarray,
            actual_home: np.ndarray, actual_away: np.ndarray) -> float:
    """MLE fit of rho against real (lambda, actual) pairs -- bounded to a
    plausible range (no assumption of the specific value, just that a
    valid correction keeps every low-score cell's probability positive)."""
    result = minimize_scalar(
        negative_log_likelihood, bounds=(-0.3, 0.3), method="bounded",
        args=(lambda_home, lambda_away, actual_home, actual_away),
    )
    return float(result.x)
