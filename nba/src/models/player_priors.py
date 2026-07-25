"""Experience-based shrinkage strength for the RAPM-lite ridge fit
(`rapm_lite.py`), so a rookie/call-up with a handful of stints doesn't get an
arbitrarily extreme rating just because ridge regression is otherwise
under-determined for them.

Simplification vs. the original plan (documented honestly, not silently
substituted): rather than fitting a nonzero, experience-dependent PRIOR MEAN
per player (which would need a self-referential bootstrapped curve-fit --
you'd need RAPM ratings to fit the curve that initializes RAPM ratings), this
uses a variable ridge PENALTY (precision) by experience instead: low
career-games-played players get a LARGER lambda (shrunk harder toward the
shared 0 = league-average prior), high-experience players get a smaller
lambda (allowed to deviate further from average once they've actually shown
it over enough stints). This is an equally valid empirical-Bayes
formulation -- variable prior variance instead of variable prior mean -- and
is simpler to implement and reason about walk-forward-safely. Flagged as a
candidate to revisit (e.g. testing an actual experience-curve prior mean)
once the base RAPM-lite layer is validated.
"""

import numpy as np

BASE_LAMBDA = 2000.0  # placeholder, un-calibrated -- see MODEL_DOCUMENTATION.md
MIN_LAMBDA_FRACTION = 0.15  # even a fully-established veteran keeps at least this much shrinkage
EXPERIENCE_HALFLIFE_GAMES = 120.0  # games until shrinkage strength decays halfway to its floor


def lambda_for_experience(career_games_played_before: np.ndarray) -> np.ndarray:
    """Per-player ridge penalty as a function of career games played before
    the current fit date (0 for a player with no prior stint history at
    all). Decays from BASE_LAMBDA toward BASE_LAMBDA * MIN_LAMBDA_FRACTION
    with an exponential half-life in games -- smooth, monotonic, and never
    reaches exactly zero (a fully unregularized column is not well-posed
    when co-occurring teammates create multicollinearity, see
    MODEL_DOCUMENTATION.md's RAPM-identifiability risk note)."""
    decay = 0.5 ** (career_games_played_before / EXPERIENCE_HALFLIFE_GAMES)
    return BASE_LAMBDA * (MIN_LAMBDA_FRACTION + (1 - MIN_LAMBDA_FRACTION) * decay)
