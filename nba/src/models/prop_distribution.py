"""Player-level analog of `score_distribution.py`: turns a player's
rate-model point projection (a mean) into a full predictive distribution,
enabling over/under probabilities the same way `score_distribution.py`
does for game totals/margins -- the actual sportsbook-style deliverable.

STALE DOCSTRING FIXED (2026-08-01, full-model audit): this used to
describe the plan's ORIGINAL a priori split (points/2PM/3PM/FTM as
"HIGH-COUNT... Normal or Student-t", OREB/STL/BLK/AST/TOV as "LOW-COUNT...
Poisson or Negative-Binomial") -- directly contradicted by `CATEGORY_FAMILY`
below in this SAME file, where every one of the 10 categories has
`family_type: "count"`. See `CATEGORY_FAMILY`'s own docstring for the full
story: Sec25 found that a genuine calibration check (not just log-score)
showed points/2PT/3PT/FT-made are dramatically BETTER calibrated as count
categories too -- the plan's CLT-based domain reasoning holds for a TEAM's
aggregate points (already correctly continuous in `score_distribution.py`)
but not for an individual player's own shot-based makes in one game, which
never see enough volume for that approximation to hold. Every category
below picks its family from a genuine, data-driven decision
(`validate_prop_distribution.py`), never assumed.

`fit_continuous_family`/local `predict_variance`/the "continuous"
`family_type` branch are kept as tested, available primitives (REUSING
`score_distribution.fit_residual_variance_model`/`fit_student_t_df`
directly, not reimplemented -- `predict_variance` itself is a LOCAL
reimplementation, see `MIN_PLAYER_VARIANCE`'s docstring for why) in case a
future category genuinely needs them -- not reachable for any
currently-adopted category.
"""

import numpy as np
from scipy import stats

from src.models.score_distribution import _t_scale, fit_residual_variance_model, fit_student_t_df

# Floor on a CONTINUOUS category's predicted variance. NOT `score_distribution.MIN_VARIANCE`
# (=4.0) -- that constant is sized for TEAM-SCORE variance (a ~100-point game's variance is
# naturally in the dozens, so 4.0 is a negligible floor there). REAL BUG FOUND (2026-08-01): this
# module originally imported and called `score_distribution.predict_variance` directly, which
# applies that SAME 4.0 floor to PLAYER-level stats whose raw fitted variance is routinely well
# under 1 (e.g. 2PT-makes variance at 1 projected attempt fits to ~0.54 on real dev data) --
# confirmed directly: nearly every player's variance was being silently clamped up to exactly 4.0
# regardless of their own actual exposure, and (more seriously) this SAME inflated-variance
# `predict_variance` call is also used internally by `fit_continuous_family` to standardize
# residuals before fitting Student-t df, meaning the original family-choice validation itself was
# distorted, not just the live output. Fixed with a player-scale floor instead, re-validated after
# the fix (see MODEL_DOCUMENTATION.md).
MIN_PLAYER_VARIANCE = 1e-3


def predict_variance(variance_model: dict, exposure: np.ndarray) -> np.ndarray:
    """Player-scale analog of `score_distribution.predict_variance` --
    same linear-in-exposure formula, floored at `MIN_PLAYER_VARIANCE`
    instead of that module's team-score-sized `MIN_VARIANCE`."""
    raw = variance_model["intercept"] + variance_model["slope"] * exposure
    return np.maximum(raw, MIN_PLAYER_VARIANCE)

# Variance/mean ratio above which Negative-Binomial is preferred over Poisson (which assumes
# variance == mean exactly). A placeholder pending empirical confirmation on real per-player
# residuals (see validate_prop_distribution.py) -- not asserted to be exactly the right cutoff,
# just a reasonable starting gate so mild noise doesn't trigger NB's extra parameter needlessly.
OVERDISPERSION_THRESHOLD = 1.2

# Floor on NB's dispersion parameter r -- guards against a pathological near-zero r (from a
# near-zero variance-minus-mean denominator) producing a degenerate, wildly overdispersed fit.
MIN_NB_DISPERSION = 1.0

# Per-category family choice, confirmed empirically (not assumed) by `validate_prop_distribution.py`
# on the full dev range. Two rounds of real findings here, both against the plan's original a
# priori "continuous vs count" axis assignment:
#
# ROUND 1 (variance-floor bug fix): the continuous branch originally reused
# `score_distribution.predict_variance` directly, which floors variance at that module's
# team-score-sized `MIN_VARIANCE=4.0` -- silently inflating every low-exposure player's assumed
# uncertainty. Fixed with a player-scale `MIN_PLAYER_VARIANCE` floor (see above).
#
# ROUND 2 (the bigger finding): once a genuine CALIBRATION check was added (PIT values vs nominal
# quantiles -- something this module never actually verified before, only log-score comparisons),
# points/2PT/3PT/FT-made ALL showed real, sometimes severe miscalibration even after Round 1's fix
# (e.g. ft_made's Student-t: max deviation from nominal 0.35 -- badly wrong). Directly tested the
# plan's assumed axis itself (never re-derived until now): treating ALL FOUR as COUNT categories
# (Poisson/NegBin, same treatment as OREB/DREB/etc, since a player's shot-based makes in one game
# are genuinely low-count -- 5-10 2PT attempts, 2-5 3PT attempts, 1-4 FTs is nowhere near enough
# volume for the CLT-based continuity the plan assumed) gives BOTH a better log-score AND a
# dramatically better calibration for every one of them (e.g. ft_made count-family calibration max
# deviation: 0.012, vs 0.35 as continuous; points: negbin log-score 2.85 vs t's 3.53, calibration
# 0.022 vs 0.032). The plan's original domain-reasoning split ("points/makes aggregate over many
# attempts, genuinely continuous by CLT") does NOT hold at the PER-PLAYER-PER-GAME level -- it
# would hold for a TEAM's aggregate points (Phase 1-3 already treats that correctly as continuous),
# but an individual player's own makes in one game is a much smaller, genuinely discrete sample.
# ALL 10 categories now use count families -- `family_type: "continuous"` and its supporting
# machinery (`fit_continuous_family`, `predict_variance`) are kept as tested, available primitives
# (not deleted) in case a future category genuinely needs them, following this project's standing
# convention for validated-but-currently-unused code (see `add_era_adjusted_player_rate`).
CATEGORY_FAMILY = {
    "points": {"family_type": "count", "family": "negbin", "exposure_col": None},
    "2pt_made": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "3pt_made": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "ft_made": {"family_type": "count", "family": "negbin", "exposure_col": None},
    "oreb": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "dreb": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "ast": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "tov": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "stl": {"family_type": "count", "family": "poisson", "exposure_col": None},
    "blk": {"family_type": "count", "family": "poisson", "exposure_col": None},
}

# Categories where `fit_count_family_mean_dependent` (task #57, Sec52) showed a REAL log-score
# improvement for the near-zero-mean population with NO regression for the high-mean population,
# confirmed via paired bootstrap on a real chronological eval split (not assumed from the Sec46.2
# diagnostic alone): oreb, blk, ast. `tov` was ALSO diagnosed with a bucket-dependent pattern in
# Sec46.2 but did NOT clear a real improvement on this same validation (NOISE both for its low-mean
# subset and overall) -- deliberately excluded here despite sharing the diagnostic, per this
# project's per-category "never assume a related category transfers" discipline. dreb/stl showed no
# bucket-dependence at all in Sec46.2 and were never tested here. A category in this set still keeps
# its normal single-family entry in `CATEGORY_FAMILY` above (used as the fallback/description), but
# `generate_props.py._fit_prop_distributions`/`_distribution_fields` route it through
# `fit_count_family_mean_dependent`/`family_for_mean` instead of the pooled single-family fit.
MEAN_DEPENDENT_CATEGORIES = {"oreb", "blk", "ast"}

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


MEAN_DEPENDENT_N_BUCKETS = 4  # quantile buckets, matching the diagnostic (MODEL_DOCUMENTATION.md
# Sec46.2) that motivated this -- not re-tuned here, just reused as the same granularity that
# revealed the effect in the first place.
MIN_BUCKET_SIZE = 500  # don't attempt a separate overdispersion fit on a tiny bucket -- same role
# as `rapm_lite.MIN_STINTS_TO_FIT`, sized for this project's typical multi-season eval-set counts.


def fit_count_family_mean_dependent(actuals: np.ndarray, means: np.ndarray,
                                     n_buckets: int = MEAN_DEPENDENT_N_BUCKETS) -> dict:
    """Mean-DEPENDENT generalization of `fit_count_family` (task #57,
    Sec46.2's finding): a single POOLED family choice for a whole category
    hides that overdispersion needing Negative-Binomial is concentrated in
    the NEAR-ZERO-projected-mean bucket specifically (deep-bench players
    projected essentially 0 for that category) -- NOT spread evenly, and
    NOT concentrated in the high-mean/star tier the original hypothesis
    expected. High-mean buckets are consistently fine with Poisson.

    Splits players into `n_buckets` quantile buckets of their own
    projected `means` (same quantile-bucketing Sec46.2's diagnostic used),
    fits `fit_count_family` (UNMODIFIED -- reuses the exact already-
    validated overdispersion math) separately within each bucket. A
    bucket with fewer than `MIN_BUCKET_SIZE` rows falls back to Poisson
    rather than fitting an unstable NB on too little data.

    Returns `{"bucket_edges": [...], "bucket_fits": [...]}` --
    `bucket_edges` has `n_buckets - 1` interior cutpoints (quantiles of
    `means` from the FIT set); `family_for_mean` below is the prediction-
    time lookup counterpart, keyed by a player's own projected mean (not
    by which bucket they happened to fall in historically -- the cutpoints
    generalize prospectively, the same way any other trained threshold
    does)."""
    means = np.asarray(means)
    actuals = np.asarray(actuals)
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    bucket_edges = np.quantile(means, quantiles).tolist()
    bucket_ids = np.searchsorted(bucket_edges, means)

    bucket_fits = []
    for b in range(n_buckets):
        mask = bucket_ids == b
        if mask.sum() < MIN_BUCKET_SIZE:
            bucket_fits.append({"family": "poisson"})
        else:
            bucket_fits.append(fit_count_family(actuals[mask], means[mask]))
    return {"bucket_edges": bucket_edges, "bucket_fits": bucket_fits}


def family_for_mean(model: dict, mean: float) -> dict:
    """Prediction-time counterpart to `fit_count_family_mean_dependent`:
    looks up which bucket a projected `mean` falls into (via that model's
    own quantile `bucket_edges`) and returns that bucket's fitted family
    params -- exactly the shape `over_under_prob`/`log_score` already
    expect from `fit_count_family`'s single-bucket output."""
    bucket_id = int(np.searchsorted(model["bucket_edges"], mean))
    return model["bucket_fits"][bucket_id]


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
        return float(stats.t.sf(line, df=params["df"], loc=mean, scale=_t_scale(params["variance"], params["df"])))
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
        return float(-stats.t.logpdf(actual, df=params["df"], loc=mean, scale=_t_scale(params["variance"], params["df"])))
    if family == "poisson":
        return float(-stats.poisson.logpmf(actual, mu=max(mean, MIN_COUNT_MEAN)))
    if family == "negbin":
        r = params["r"]
        mean = max(mean, MIN_COUNT_MEAN)
        p = r / (r + mean)
        return float(-stats.nbinom.logpmf(actual, n=r, p=p))
    raise ValueError(f"unknown family: {family}")
