"""Blend our own margin/total predictions with real Vegas closing lines.

nfl_data_py's schedule data already includes real historical closing lines
(spread_line, total_line, moneylines) -- 100% coverage 2016-2025, and lines
are typically published for the upcoming week well before kickoff, including
preseason Week 1 lines. This was previously unused.

Validated (walk-forward, TRAIN=2018-2021 fit / TEST=2022-2025 held out,
matching this project's standard split): market alone beats our model alone
on both margin (MAE 9.49 vs 9.99) and total (10.19 vs 10.65). A regularized
blend of the two is statistically indistinguishable from market alone in
practice but retains a small, real, positive contribution from our own model
-- plain OLS on the smaller CALIBRATION_SEASONS window (2022-2025 only, 4
seasons) is numerically unstable due to strong collinearity between our_margin
and spread_line (corr ~0.83): the "our model" weight swings from -0.06 to
+0.17 depending on the exact window, and even Ridge doesn't fully stabilize
it at that sample size. Fitting on a wider window (2018 through the season
before the one being predicted) with cross-validated Ridge gives consistent,
sensible, always-positive weights, so that's what's used here.

A dynamically-growing window is used (not a hardcoded 2018-2025) so this
keeps improving as more real seasons accumulate season over season, rather
than going stale.

An nfelo-inspired dynamic per-team-error-weighted blend was also tried and
UNDERPERFORMED this simple regularized blend (MAE 9.65 vs 9.55 on margin) --
not included here.

NOT used in live production as of review round 2 -- the margin blend was removed in round 1
(a full ATS%/CRPS/signed-bias panel found it indistinguishable from a coin flip with a real
negative bias, MODEL_DOCUMENTATION.md §4.3) and the total blend was removed in round 2 on the
identical panel (O/U%=50.1%, CI [47.2%,53.0%], the same coin-flip failure pattern -- §4.3,
src/models/validate_market_blend_totals.py). `fit_market_blend`/`apply_blend` are kept as
generic, still-correct utilities -- `validate_game_simulator.py` uses them for its own
historical comparison-baseline construction, and a future feature could reuse them -- but
`weekly_update.py` now uses the market spread_line/total_line directly for both margin and
total, falling back to Layer 1's own calibration only when no line is published yet.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

MARKET_FIT_START_SEASON = 2018  # excludes 2016-2017 burn-in, matches project convention


def fit_market_blend(x_ours: pd.Series, x_market: pd.Series, y: pd.Series) -> dict:
    """Fit a Ridge-regularized blend of our own prediction and the market
    line against the actual outcome. Returns {intercept, our_weight, market_weight}."""
    X = np.column_stack([x_ours.to_numpy(), x_market.to_numpy()])
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 40), cv=KFold(5, shuffle=True, random_state=0))
    ridge.fit(X, y.to_numpy())
    return {
        "intercept": float(ridge.intercept_),
        "our_weight": float(ridge.coef_[0]),
        "market_weight": float(ridge.coef_[1]),
    }


def apply_blend(our_pred: float, market_line: float | None, blend: dict) -> float:
    """Apply a fitted blend, falling back to our own prediction unblended if
    no market line is available yet for this specific game."""
    if market_line is None or (isinstance(market_line, float) and np.isnan(market_line)):
        return our_pred
    return blend["intercept"] + blend["our_weight"] * our_pred + blend["market_weight"] * market_line
