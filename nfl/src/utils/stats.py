"""Shared statistical helpers. fit_linear() replaces raw np.polyfit calls
project-wide (review 2026-07 #3.4): np.polyfit(x, y, 1) returns [slope,
intercept] -- highest-degree coefficient first -- which is easy to unpack
backwards when a call site wants (intercept, slope) order (as almost every
call site in this project does, matching the `pred = a + b*x` convention).
That exact bug has shipped three times in this project: a hyperparameter
re-optimization test, an early wind-adjustment draft (both caught and fixed
in-session), and a third, previously-undetected instance in this audit --
weekly_update.py's and validate_props_pipeline.py's game-script pass-rate
adjustment used `script_slope, script_intercept = np.polyfit(...)[::-1]`,
which actually assigns script_slope=intercept and script_intercept=slope.
Confirmed against real data: this flipped the sign and roughly doubled the
magnitude of the "a team that's winning passes less" effect used to adjust
projected pass attempts for every prop prediction.
"""

import numpy as np


def fit_linear(x, y) -> tuple[float, float]:
    """Fit y = intercept + slope*x. Returns (intercept, slope) -- matching
    this project's `pred = a + b*x` convention, NOT np.polyfit's native
    (slope, intercept) order. Asserts the fitted slope's sign matches the raw
    correlation sign, catching a swapped-unpacking bug at the source rather
    than silently shipping a sign-flipped adjustment."""
    slope, intercept = np.polyfit(x, y, 1)
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x_arr.std() > 0 and y_arr.std() > 0:
        raw_corr = np.corrcoef(x_arr, y_arr)[0, 1]
        if raw_corr != 0 and np.sign(slope) != np.sign(raw_corr):
            raise AssertionError(
                f"fit_linear: fitted slope ({slope:+.5f}) contradicts the raw "
                f"correlation sign ({raw_corr:+.4f}) -- check for a swapped "
                f"slope/intercept unpacking bug at the call site."
            )
    return float(intercept), float(slope)
