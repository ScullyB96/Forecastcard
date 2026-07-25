"""Turns Phase 1/2's point projections (projected_home_score,
projected_away_score) into a full outcome distribution -- win probability,
spread, and total -- rather than just a point estimate.

NBA scores aggregate over ~100 quasi-independent possessions, so (unlike
the NHL sibling's low-scoring Poisson/Dixon-Coles approach) a continuous
distribution around the point projection is the natural fit, not a discrete
count model. Variance is fit as a function of projected PACE (more
possessions -> more aggregated variance, not a flat constant), and the
home/away residual correlation is fit empirically rather than assumed zero
(a fast, high-possession game inflates both sides' variance simultaneously
via a shared pace-driven component even after controlling for the mean).

Both Normal and Student-t candidate families are fit and compared via
calibration coverage + log-score (a proper scoring rule that works
uniformly for any parametric family, unlike CRPS which only has a simple
closed form for Normal) -- see `validate_score_distribution.py` for the
comparison and adoption decision. Nothing here assumes Normal wins a
priori.
"""

import numpy as np
import pandas as pd
from scipy import stats

MIN_VARIANCE = 4.0  # floor so a pathological low-pace fit can't predict near-zero variance


def fit_residual_variance_model(residuals: np.ndarray, pace: np.ndarray) -> dict:
    """OLS fit of squared residual ~ a + b*pace (variance should scale
    with pace -- more possessions aggregated -> more variance, not less).
    Returns {"intercept": a, "slope": b} for `predict_variance`."""
    X = np.column_stack([np.ones_like(pace), pace])
    sq_resid = residuals ** 2
    coefs, *_ = np.linalg.lstsq(X, sq_resid, rcond=None)
    return {"intercept": float(coefs[0]), "slope": float(coefs[1])}


def predict_variance(variance_model: dict, pace: np.ndarray) -> np.ndarray:
    raw = variance_model["intercept"] + variance_model["slope"] * pace
    return np.maximum(raw, MIN_VARIANCE)


def fit_home_away_correlation(home_residuals: np.ndarray, away_residuals: np.ndarray) -> float:
    """A single global correlation (not pace-conditional -- a simpler first
    cut; pace-conditional correlation is a candidate refinement, not
    assumed necessary here)."""
    return float(np.corrcoef(home_residuals, away_residuals)[0, 1])


def fit_student_t_df(standardized_residuals: np.ndarray) -> float:
    """Degrees of freedom via method-of-moments on excess kurtosis (simpler
    and more robust than full MLE for a first cut): df = 6/excess_kurtosis + 4
    for positive excess kurtosis (heavier-than-normal tails); falls back to
    a large df (effectively Normal) if excess kurtosis is <= 0 (fitting a
    Student-t to THINNER-than-normal-tailed data is not meaningful -- df
    would diverge to infinity anyway)."""
    excess_kurtosis = float(stats.kurtosis(standardized_residuals, fisher=True))
    if excess_kurtosis <= 0.05:
        return 200.0  # effectively Normal
    return max(2.5, 6.0 / excess_kurtosis + 4.0)


def margin_and_total_params(mean_home: float, mean_away: float, var_home: float, var_away: float,
                             corr: float) -> dict:
    """Closed-form margin/total mean+variance from the joint home/away
    normal-family parameters (works identically whether the marginals are
    treated as Normal or Student-t with matched variance -- only the tail
    shape differs, not these first two moments)."""
    cov = corr * np.sqrt(var_home * var_away)
    return {
        "margin_mean": mean_home - mean_away, "margin_var": var_home + var_away - 2 * cov,
        "total_mean": mean_home + mean_away, "total_var": var_home + var_away + 2 * cov,
    }


def win_prob_home(mean_home: float, mean_away: float, var_home: float, var_away: float,
                   corr: float, family: str = "normal", df: float = 200.0) -> float:
    params = margin_and_total_params(mean_home, mean_away, var_home, var_away, corr)
    margin_std = np.sqrt(params["margin_var"])
    if family == "t":
        return float(1 - stats.t.cdf(0, df=df, loc=params["margin_mean"], scale=margin_std))
    return float(1 - stats.norm.cdf(0, loc=params["margin_mean"], scale=margin_std))


def interval(mean: float, var: float, coverage: float, family: str = "normal", df: float = 200.0) -> tuple[float, float]:
    alpha = (1 - coverage) / 2
    std = np.sqrt(var)
    if family == "t":
        return float(stats.t.ppf(alpha, df=df, loc=mean, scale=std)), float(stats.t.ppf(1 - alpha, df=df, loc=mean, scale=std))
    return float(stats.norm.ppf(alpha, loc=mean, scale=std)), float(stats.norm.ppf(1 - alpha, loc=mean, scale=std))


def log_score(actual: float, mean: float, var: float, family: str = "normal", df: float = 200.0) -> float:
    """Negative log-likelihood (lower is better) -- a proper scoring rule
    that works uniformly for any parametric family, used as the primary
    Normal-vs-Student-t comparison metric instead of CRPS (which only has a
    simple closed form for Normal)."""
    std = np.sqrt(var)
    if family == "t":
        return float(-stats.t.logpdf(actual, df=df, loc=mean, scale=std))
    return float(-stats.norm.logpdf(actual, loc=mean, scale=std))


def crps_normal(actual: float, mean: float, var: float) -> float:
    """Closed-form CRPS for a Normal distribution (Gneiting & Raftery 2007)
    -- an additional Normal-only diagnostic alongside log-score."""
    std = np.sqrt(var)
    z = (actual - mean) / std
    return float(std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


def empirical_coverage(actuals: np.ndarray, means: np.ndarray, variances: np.ndarray, coverage: float,
                        family: str = "normal", df: float = 200.0) -> float:
    """Fraction of real outcomes falling inside the nominal `coverage`
    interval -- should be close to `coverage` itself for a well-calibrated
    distribution."""
    alpha = (1 - coverage) / 2
    stds = np.sqrt(variances)
    if family == "t":
        lo = stats.t.ppf(alpha, df=df, loc=means, scale=stds)
        hi = stats.t.ppf(1 - alpha, df=df, loc=means, scale=stds)
    else:
        lo = stats.norm.ppf(alpha, loc=means, scale=stds)
        hi = stats.norm.ppf(1 - alpha, loc=means, scale=stds)
    return float(((actuals >= lo) & (actuals <= hi)).mean())
