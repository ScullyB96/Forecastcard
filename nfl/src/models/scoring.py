"""Betting-relevant scoring metrics, replacing pooled MAE as the primary
decision metric everywhere a margin/total prediction is evaluated against the
market (review 2026-07 #1.3): a betting model is graded on whether it beats
the closing line and how well-calibrated its implied probabilities are, not
on how close its point estimate is to the final score. MAE rewards a model
that's usually close and occasionally very wrong the same as one that's
always moderately off -- ATS/O-U win rate and CRPS are what actually convert
to edge.

The current live pipeline still emits point estimates, not full distributions.
CRPS/Brier here use a Gaussian approximation: N(point_estimate, sigma), with
sigma the residual std from a training fit, fixed rather than per-game -- an
honest placeholder, not a claim of calibrated per-game uncertainty.
`game_simulator.py` (Phase 2, added 2026-07) now provides a real, validated
Monte Carlo predictive distribution for margin/win-probability specifically,
but is not yet wired into the live pipeline -- once it (or a future totals
fix for it) is, swap in its actual empirical distribution here instead of
this Gaussian approximation.
"""

import numpy as np
import pandas as pd
from scipy import stats


def ats_win_rate(pred: pd.Series, market_line: pd.Series, actual: pd.Series) -> dict:
    """Against-the-spread win rate: bet the side our prediction disagrees
    with the market on, see if the actual outcome landed on that side of the
    market line. Pushes (exact tie with the line) are excluded from the
    denominator, matching standard sports-betting convention."""
    predicted_edge = pred - market_line
    actual_edge = actual - market_line
    push = np.isclose(actual_edge, 0.0)
    decided = ~push
    win = np.sign(predicted_edge[decided]) == np.sign(actual_edge[decided])
    n = int(decided.sum())
    rate = float(win.mean()) if n else float("nan")
    return {"win_rate": rate, "n": n, "n_push": int(push.sum()), "ci": _bootstrap_ci(win.to_numpy().astype(float))}


def ou_win_rate(pred: pd.Series, market_line: pd.Series, actual: pd.Series) -> dict:
    """Over/under win rate -- same mechanics as ats_win_rate, applied to totals."""
    return ats_win_rate(pred, market_line, actual)


def signed_bias(pred: pd.Series, actual: pd.Series) -> dict:
    """Mean signed error (pred - actual). Unlike MAE, this reveals systematic
    over/under-prediction on a subset -- the thing that actually costs money
    if you bet the same direction repeatedly, and what MAE structurally
    cannot show (it's symmetric by construction)."""
    err = (pred - actual).to_numpy()
    return {"bias": float(err.mean()), "n": len(err), "ci": _bootstrap_ci(err)}


def crps_gaussian(pred: pd.Series, actual: pd.Series, sigma: float) -> dict:
    """Continuous Ranked Probability Score under a Gaussian(pred, sigma)
    forecast distribution -- proper scoring rule that rewards both accuracy
    and honest uncertainty, unlike MAE (which only sees the point estimate).
    Closed form: sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)], z=(y-mu)/sigma."""
    z = (actual.to_numpy() - pred.to_numpy()) / sigma
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return {"crps": float(crps.mean()), "n": len(crps), "ci": _bootstrap_ci(crps)}


def brier_score(pred: pd.Series, actual: pd.Series, sigma: float) -> dict:
    """Brier score for the implied home-win probability Phi(pred_margin/sigma)
    under the same Gaussian approximation as crps_gaussian. Parameter order
    (pred, actual, sigma) matches crps_gaussian/fit_residual_sigma -- kept
    consistent across this module's sibling functions after a review found
    brier_score/log_loss_gaussian had drifted to a different (pred, sigma,
    actual) order, an easy positional-argument trap (nothing called them
    incorrectly yet, but it was only a matter of time)."""
    p_home_win = stats.norm.cdf(pred.to_numpy() / sigma)
    outcome = (actual.to_numpy() > 0).astype(float)
    sq_err = (p_home_win - outcome) ** 2
    return {"brier": float(sq_err.mean()), "n": len(sq_err), "ci": _bootstrap_ci(sq_err)}


def log_loss_gaussian(pred: pd.Series, actual: pd.Series, sigma: float) -> dict:
    p_home_win = np.clip(stats.norm.cdf(pred.to_numpy() / sigma), 1e-6, 1 - 1e-6)
    outcome = (actual.to_numpy() > 0).astype(float)
    ll = -(outcome * np.log(p_home_win) + (1 - outcome) * np.log(1 - p_home_win))
    return {"log_loss": float(ll.mean()), "n": len(ll), "ci": _bootstrap_ci(ll)}


def fit_residual_sigma(pred: pd.Series, actual: pd.Series) -> float:
    """Residual std from a TRAIN fit, for use as the fixed sigma in the
    Gaussian approximations above. Fit on TRAIN only, apply to TEST -- same
    walk-forward discipline as everything else in this project."""
    return float((actual - pred).std(ddof=1))


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lo = (1 - ci) / 2 * 100
    hi = (1 - (1 - ci) / 2) * 100
    return (float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi)))


def summarize(label: str, pred: pd.Series, market_line: pd.Series, actual: pd.Series, sigma: float) -> None:
    """Print the full scoring panel for one candidate prediction, replacing a
    bare MAE print statement -- the standard report format going forward."""
    ats = ats_win_rate(pred, market_line, actual)
    bias = signed_bias(pred, actual)
    crps = crps_gaussian(pred, actual, sigma)
    mae = (pred - actual).abs().mean()
    print(
        f"  {label:32s}  MAE={mae:.3f}  ATS%={ats['win_rate']:.4f} (n={ats['n']}, "
        f"95% CI [{ats['ci'][0]:.3f},{ats['ci'][1]:.3f}])  "
        f"bias={bias['bias']:+.3f} (CI [{bias['ci'][0]:+.3f},{bias['ci'][1]:+.3f}])  "
        f"CRPS={crps['crps']:.3f}"
    )
