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


REFIT_PERIOD_DAYS = 14  # mirrors rapm_lite.py's own biweekly cadence -- the closest existing
# walk-forward-refit analog in this codebase (see that module's docstring for the tractability
# rationale: refitting for every individual game date would be far more compute than a periodic
# refit, while still being fully walk-forward-safe).
MIN_RESIDUALS_TO_FIT = 200  # don't attempt an OLS fit on a tiny, unstable early sample --
# same role as rapm_lite.MIN_STINTS_TO_FIT, tuned for this much simpler 2-parameter fit.


def compute_walkforward_variance_model(residuals: pd.DataFrame, refit_period_days: int = REFIT_PERIOD_DAYS) -> pd.DataFrame:
    """`residuals` must have columns `gameDate`, `residual`, `pace` (one row
    per team-side residual). Refits `fit_residual_variance_model` every
    `refit_period_days`, using ONLY residuals STRICTLY BEFORE each
    checkpoint -- fixes Sec45.3's finding that a single static fit over the
    entire historical range produces a real, significant declining margin-
    coverage trend (r=-0.749 to -0.782 by season) that a global recency-
    weighted refit does NOT resolve (Sec45.3 also tested that; it makes the
    trend slightly worse). Mirrors `rapm_lite.compute_walkforward_player_ratings`'s
    exact checkpoint-loop shape. Returns one row per checkpoint that had
    enough prior data to fit: `asOfDate`, `intercept`, `slope`."""
    residuals = residuals.sort_values("gameDate").reset_index(drop=True)
    first_date, last_date = residuals["gameDate"].min(), residuals["gameDate"].max()
    checkpoints = pd.date_range(first_date, last_date, freq=f"{refit_period_days}D")

    rows = []
    for checkpoint in checkpoints:
        prior = residuals[residuals["gameDate"] < checkpoint]
        if len(prior) < MIN_RESIDUALS_TO_FIT:
            continue
        model = fit_residual_variance_model(prior["residual"].to_numpy(), prior["pace"].to_numpy())
        rows.append({"asOfDate": checkpoint, "intercept": model["intercept"], "slope": model["slope"]})
    return pd.DataFrame(rows, columns=["asOfDate", "intercept", "slope"])


def predict_variance_walk_forward(variance_checkpoints: pd.DataFrame, game_dates: pd.Series,
                                   pace: np.ndarray) -> np.ndarray:
    """For each row, looks up the latest checkpoint strictly before that
    row's `gameDate` (same "most recent prior checkpoint" lookup pattern
    used throughout this project's RAPM-lite validation scripts, e.g.
    `validate_predictive_lineup_adjustment.py`) and applies
    `predict_variance` with that checkpoint's fit. A date before EVERY
    checkpoint falls back to the EARLIEST available checkpoint (matches
    this project's standing convention of using whatever's genuinely
    available rather than refusing a prediction -- e.g.
    `team_stat_rates`'s `games_played_before=0` fallback to the league
    prior) instead of raising or returning NaN."""
    if variance_checkpoints.empty:
        raise ValueError("no variance checkpoints available -- not enough residual history to fit even one")
    checkpoints = variance_checkpoints.sort_values("asOfDate").reset_index(drop=True)
    game_dates = pd.to_datetime(game_dates).reset_index(drop=True)

    idx = checkpoints["asOfDate"].to_numpy().searchsorted(game_dates.to_numpy(), side="left") - 1
    idx = np.clip(idx, 0, len(checkpoints) - 1)
    intercept = checkpoints["intercept"].to_numpy()[idx]
    slope = checkpoints["slope"].to_numpy()[idx]
    raw = intercept + slope * np.asarray(pace)
    return np.maximum(raw, MIN_VARIANCE)


def fit_home_away_correlation(home_residuals: np.ndarray, away_residuals: np.ndarray) -> float:
    """A single global correlation (not pace-conditional -- a simpler first
    cut; pace-conditional correlation is a candidate refinement, not
    assumed necessary here)."""
    return float(np.corrcoef(home_residuals, away_residuals)[0, 1])


def _t_scale(var, df: float):
    """Converts a target VARIANCE into the scipy `scale` parameter for a
    Student-t with `df` degrees of freedom.

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): scipy's
    `t.cdf`/`t.ppf`/`t.logpdf`/`t.sf` parameterize `X = loc + scale*T`
    where T is a STANDARD Student-t with `Var(T) = df/(df-2)` for df>2 --
    NOT 1 -- so passing `scale=sqrt(var)` directly (what every t-family
    call site in this module, and the copy-pasted math in
    `prop_distribution.py`, originally did) silently inflates the
    REALIZED variance by a factor of `df/(df-2)`, contradicting this
    module's own "matched variance -- only the tail shape differs, not
    these first two moments" design intent (see `margin_and_total_params`'s
    docstring). Confirmed empirically: `scipy.stats.t.rvs(df=9.5, scale=10,
    size=3e6)` has empirical variance ~126.5, matching `100 * 9.5/7.5 ~
    126.67`, not the naive 100. Currently dormant in live production
    (every adopted category uses `family='normal'`), but this bug was
    silently present in the ORIGINAL Sec6/Sec11 dev-range Normal-vs-t
    comparisons that decided the family in the first place -- df is
    floored at 2.001 here (variance is undefined at df<=2, though
    `fit_student_t_df` already floors its own output at 2.5)."""
    df = max(df, 2.001)
    return np.sqrt(var * (df - 2) / df)


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
    if family == "t":
        return float(1 - stats.t.cdf(0, df=df, loc=params["margin_mean"], scale=_t_scale(params["margin_var"], df)))
    return float(1 - stats.norm.cdf(0, loc=params["margin_mean"], scale=np.sqrt(params["margin_var"])))


def interval(mean: float, var: float, coverage: float, family: str = "normal", df: float = 200.0) -> tuple[float, float]:
    alpha = (1 - coverage) / 2
    if family == "t":
        scale = _t_scale(var, df)
        return float(stats.t.ppf(alpha, df=df, loc=mean, scale=scale)), float(stats.t.ppf(1 - alpha, df=df, loc=mean, scale=scale))
    std = np.sqrt(var)
    return float(stats.norm.ppf(alpha, loc=mean, scale=std)), float(stats.norm.ppf(1 - alpha, loc=mean, scale=std))


def log_score(actual: float, mean: float, var: float, family: str = "normal", df: float = 200.0) -> float:
    """Negative log-likelihood (lower is better) -- a proper scoring rule
    that works uniformly for any parametric family, used as the primary
    Normal-vs-Student-t comparison metric instead of CRPS (which only has a
    simple closed form for Normal)."""
    if family == "t":
        return float(-stats.t.logpdf(actual, df=df, loc=mean, scale=_t_scale(var, df)))
    return float(-stats.norm.logpdf(actual, loc=mean, scale=np.sqrt(var)))


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
    if family == "t":
        scales = _t_scale(variances, df)
        lo = stats.t.ppf(alpha, df=df, loc=means, scale=scales)
        hi = stats.t.ppf(1 - alpha, df=df, loc=means, scale=scales)
    else:
        stds = np.sqrt(variances)
        lo = stats.norm.ppf(alpha, loc=means, scale=stds)
        hi = stats.norm.ppf(1 - alpha, loc=means, scale=stds)
    return float(((actuals >= lo) & (actuals <= hi)).mean())
