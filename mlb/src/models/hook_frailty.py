"""Pure, dependency-free hook-frailty math (task #145), split out of
bullpen_usage_policy.py specifically so game_simulator.py can import it without a
circular import: bullpen_usage_policy.py -> bullpen.py -> game_simulator.py (for
OUTCOMES), so game_simulator.py itself can never import bullpen_usage_policy.py or
bullpen.py at module scope. Nothing in this module touches either of those -- it's
just the bucketing conventions and the mean-preserving logistic-frailty transform,
usable identically by the FITTING side (bullpen_usage_policy.py) and the SIMULATING
side (game_simulator.py).

See bullpen_usage_policy.py's module docstring for the full story of why this
mechanism exists (a real per-start selection effect in real hook timing) and
MODEL_DOCUMENTATION.md sec 11.20 for the complete fit/validation writeup.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit, roots_hermitenorm

HOOK_FRAILTY_SIGMA1 = 3.8   # task #145: fitted per-start frailty magnitude at inning 1
HOOK_FRAILTY_DECAY = 0.65   # per additional inning, sigma(inning) = SIGMA1 * DECAY**(inning-1)

_GH_NODES, _GH_WEIGHTS = roots_hermitenorm(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()  # Gauss-Hermite quadrature, normalized to a
                                                # probability measure, for exact E_h[sigmoid(x+h)]


def hook_frailty_sigma(inning: int, sigma1: float = HOOK_FRAILTY_SIGMA1,
                        decay: float = HOOK_FRAILTY_DECAY) -> float:
    return sigma1 * (decay ** (inning - 1))


def _marginal_mean(x: float, sigma: float) -> float:
    """E_h[sigmoid(x+h)], h~N(0,sigma^2), via Gauss-Hermite quadrature."""
    return float(np.sum(_GH_WEIGHTS * expit(x + _GH_NODES * sigma)))


_pre_noise_logit_cache: dict = {}


def solve_pre_noise_logit(p: float, sigma: float) -> float:
    """Exact (not closed-form-approximate) mean-preserving correction: returns x such
    that E_h[sigmoid(x+h)] == p for h~N(0,sigma^2). The closed-form (Zeger-Liang-Albert)
    approximation was checked and found biased at low p -- exactly where most hook
    cells live -- so this project uses an exact numerical root-find instead, cached
    per (p, sigma) since there are only ~300 distinct table cells."""
    key = (round(p, 6), round(sigma, 4))
    if key in _pre_noise_logit_cache:
        return _pre_noise_logit_cache[key]
    if sigma < 1e-6:
        x = logit(np.clip(p, 1e-6, 1 - 1e-6))
        _pre_noise_logit_cache[key] = x
        return x
    p = np.clip(p, 1e-6, 1 - 1e-6)
    target_logit = logit(p)
    lo, hi = target_logit - 6 * sigma - 5, target_logit + 6 * sigma + 5
    x = brentq(lambda x: _marginal_mean(x, sigma) - p, lo, hi, xtol=1e-8)
    _pre_noise_logit_cache[key] = x
    return x


def apply_hook_frailty(p_baseline: float, z: float, inning: int,
                        sigma1: float = HOOK_FRAILTY_SIGMA1,
                        decay: float = HOOK_FRAILTY_DECAY) -> float:
    """Adjusts a baseline (already-validated, per-cell) hook hazard for one PA using
    this start/trial's own frailty draw `z` (a single standard normal drawn ONCE per
    start per trial and reused across every inning of that start -- its INFLUENCE
    shrinks with inning via `sigma`, but it is the same underlying latent draw
    throughout, representing one correlated propensity for that start). Mean-preserving:
    averaging this over the frailty distribution reproduces p_baseline exactly."""
    sigma = hook_frailty_sigma(inning, sigma1, decay)
    if sigma < 1e-6:
        return p_baseline
    x = solve_pre_noise_logit(p_baseline, sigma)
    return float(expit(x + z * sigma))


def bucket_inning(i) -> int:
    return min(int(i), 9)


def bucket_pa_count(n) -> str:
    if n <= 6: return "1-6"
    if n <= 12: return "7-12"
    if n <= 18: return "13-18"
    return "19+"


def bucket_runs_allowed(r) -> int:
    return min(int(r), 4)


def bucket_margin(m) -> str:
    if m <= -4: return "trail_big"
    if m <= -1: return "trail"
    if m == 0: return "tied"
    if m <= 3: return "lead"
    return "lead_big"
