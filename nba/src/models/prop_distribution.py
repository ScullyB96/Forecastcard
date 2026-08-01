"""Player-level analog of `score_distribution.py`: turns a player's
rate-model point projection (a mean) into a full predictive distribution,
enabling over/under probabilities the same way `score_distribution.py`
does for game totals/margins -- the actual sportsbook-style deliverable.

HIGH-COUNT stats (points, minutes, 2PM/3PM/FTM) -> Normal or Student-t,
REUSING `score_distribution.fit_residual_variance_model`/`predict_variance`/
`fit_student_t_df` directly (not reimplemented) -- refit on player
residuals against the player's own projected EXPOSURE (minutes or
attempts) as the variance-scaling regressor, in place of game-level
`score_distribution.py`'s "pace" (more exposure -> more aggregated
variance, the identical logic, just at player scale).

LOW-COUNT stats (OREB, STL, BLK, low-usage AST/TOV) -> Poisson or
Negative-Binomial, picked via a variance-to-mean overdispersion check --
the same "fit both, let data decide" discipline `score_distribution.py`
already uses for Normal-vs-Student-t, extended to a NEW family axis
(discrete vs. continuous) that game-level totals never needed: an
individual player's low-count stat in a single game genuinely IS a
discrete count (a player either blocks 0, 1, 2... shots), unlike a
~100-possession-aggregated team score where a continuous approximation is
well justified by the central limit theorem.
"""

import numpy as np
from scipy import stats

from src.models.score_distribution import fit_residual_variance_model, fit_student_t_df, predict_variance

# Variance/mean ratio above which Negative-Binomial is preferred over Poisson (which assumes
# variance == mean exactly). A placeholder pending empirical confirmation on real per-player
# residuals (see validate_prop_distribution.py) -- not asserted to be exactly the right cutoff,
# just a reasonable starting gate so mild noise doesn't trigger NB's extra parameter needlessly.
OVERDISPERSION_THRESHOLD = 1.2

# Floor on NB's dispersion parameter r -- guards against a pathological near-zero r (from a
# near-zero variance-minus-mean denominator) producing a degenerate, wildly overdispersed fit.
MIN_NB_DISPERSION = 1.0

# Floor on a count family's mean (mu) before evaluating pmf/logpmf. BUG FOUND (2026-08-01,
# validate_prop_distribution.py's real run on 66,937 blocks eval rows): a player with an
# expanding-shrunk `blk_rate_per_min` of exactly 0.0 (real -- a perimeter player who has simply
# never recorded a block) projects `mean=0.0`; Poisson at mu=0 is a genuine degenerate point-mass
# at 0, so `logpmf(k>0, mu=0)` is EXACTLY -inf, not just very negative -- one real garbage-time
# block for that player poisons the entire eval set's mean log-score to +inf. A live projection
# can never be a certainty that something "never happens", so mu is floored here rather than left
# to produce a real -inf.
MIN_COUNT_MEAN = 1e-3


def fit_continuous_family(residuals: np.ndarray, exposure: np.ndarray) -> dict:
    """For high-count stats: fits the variance-vs-exposure model and
    Student-t degrees of freedom, EXACTLY `score_distribution.py`'s own
    math with "pace" replaced by "player exposure" (projected minutes for
    points/minutes, projected attempts for 2PM/3PM/FTM -- whichever this
    stat's own rate model already uses as its exposure unit). Family
    choice (normal vs. t) is made downstream by
    `validate_prop_distribution.py`'s calibration comparison, mirroring
    `score_distribution.py`'s own convention of not assuming the answer."""
    variance_model = fit_residual_variance_model(residuals, exposure)
    std_resid = residuals / np.sqrt(np.maximum(predict_variance(variance_model, exposure), 1e-6))
    df = fit_student_t_df(std_resid)
    return {"variance_model": variance_model, "df": df}


def fit_count_family(actuals: np.ndarray, means: np.ndarray) -> dict:
    """Poisson vs. Negative-Binomial via a variance-to-mean overdispersion
    check: Poisson assumes `variance == mean`; real box-score counts
    routinely show MORE variance than that (a player's blocks/steals swing
    game to game for reasons beyond pure Poisson noise -- foul trouble,
    blowouts, matchup, garbage time). Method-of-moments NB dispersion:
    `r = mean^2 / (variance - mean)` (NB2 parameterization: `var = mean +
    mean^2/r`), used only when empirical variance clears
    `OVERDISPERSION_THRESHOLD x mean` -- below that, `r` would be negative
    or undefined, and Poisson is already the simpler, correct choice."""
    residuals = actuals - means
    mean_of_means = float(np.mean(means))
    empirical_variance = float(np.var(residuals))
    if mean_of_means <= 0 or empirical_variance <= OVERDISPERSION_THRESHOLD * mean_of_means:
        return {"family": "poisson"}
    r = mean_of_means ** 2 / (empirical_variance - mean_of_means)
    return {"family": "negbin", "r": max(r, MIN_NB_DISPERSION)}


def over_under_prob(line: float, mean: float, family: str, params: dict) -> float:
    """P(actual > line) under the fitted family -- the sportsbook-style
    deliverable. `params` needs "variance" for normal/t (this player's own
    `predict_variance`-scaled variance at their projected exposure) and
    "df" for t; needs "r" (NB dispersion, from `fit_count_family`) for
    negbin; needs nothing extra for poisson (mean alone parameterizes it).
    scipy's discrete `.sf()` correctly handles a half-integer betting line
    (e.g. 2.5) without any manual flooring."""
    if family == "normal":
        return float(stats.norm.sf(line, loc=mean, scale=np.sqrt(params["variance"])))
    if family == "t":
        return float(stats.t.sf(line, df=params["df"], loc=mean, scale=np.sqrt(params["variance"])))
    if family == "poisson":
        return float(stats.poisson.sf(line, mu=max(mean, MIN_COUNT_MEAN)))
    if family == "negbin":
        r = params["r"]
        mean = max(mean, MIN_COUNT_MEAN)
        p = r / (r + mean)  # scipy's nbinom(n, p) has mean = n(1-p)/p; solving for p at n=r gives this
        return float(stats.nbinom.sf(line, n=r, p=p))
    raise ValueError(f"unknown family: {family}")


def log_score(actual: float, mean: float, family: str, params: dict) -> float:
    """Negative log-likelihood (lower is better) for ANY of the four
    families -- extends `score_distribution.log_score` (normal/t only) to
    also cover poisson/negbin, so Normal-vs-t AND Poisson-vs-NB can both be
    compared via the exact same proper-scoring-rule discipline
    `score_distribution.py` established, rather than inventing a separate
    ad hoc comparison metric for the discrete-family case."""
    if family == "normal":
        return float(-stats.norm.logpdf(actual, loc=mean, scale=np.sqrt(params["variance"])))
    if family == "t":
        return float(-stats.t.logpdf(actual, df=params["df"], loc=mean, scale=np.sqrt(params["variance"])))
    if family == "poisson":
        return float(-stats.poisson.logpmf(actual, mu=max(mean, MIN_COUNT_MEAN)))
    if family == "negbin":
        r = params["r"]
        mean = max(mean, MIN_COUNT_MEAN)
        p = r / (r + mean)
        return float(-stats.nbinom.logpmf(actual, n=r, p=p))
    raise ValueError(f"unknown family: {family}")
