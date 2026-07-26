"""Wind adjustment for QB yards/attempt and WR/TE yards/target -- validated
walk-forward (TRAIN=2018-2021 fit / TEST=2022-2025 held out):

  QB yards/attempt:      MAE 44.72 -> 44.09 (p=0.0022, n=1282)
  WR/TE yards/target:    MAE 13.55 -> 13.31 (p<0.0001, n=9534, much larger
                         sample -- this is the stronger, more confident one)

Completion rate showed the same directionally-correct (negative) raw
correlation but did NOT clear significance (p=0.34, n=1282) -- not included
here. Don't add it without a lot more data or a different approach; a
correlation existing in the raw crosstab is not sufficient on its own, per
this project's standing discipline (this is exactly the mistake almost made
here the first time, from a slope/intercept unpacking bug that silently
flipped the sign of the fitted adjustment -- always sanity-check a fitted
sign against the raw correlation before trusting it).

Wind is only meaningfully knowable close to kickoff (forecasts aren't
reliable until Fri/Sat) -- this adjustment is a no-op for games far in
advance (wind is NaN until schedules data gets a real forecast), and is the
main thing a near-kickoff refresh run actually gains over the early-week
one, alongside fresher inactive-list data.
"""

import numpy as np
import pandas as pd

from src.utils.stats import fit_linear

FIT_SEASONS = set(range(2018, 2026))


def fit_wind_adjustment(rate_result: pd.DataFrame, actual_col: str, touch_col: str, pregame_rate_col: str) -> tuple[float, float]:
    """rate_result must already have wind/roof merged in (outdoor games only
    matter; dome/closed games have no wind and get filtered out here).
    Returns (intercept, slope) for: rate_residual = intercept + slope * wind."""
    fit = rate_result[
        (rate_result["roof"] == "outdoors") & rate_result["wind"].notna() & rate_result["season"].isin(FIT_SEASONS)
    ].copy()
    fit["resid"] = fit[actual_col] / fit[touch_col] - fit[pregame_rate_col]
    return fit_linear(fit["wind"], fit["resid"])


def apply_wind_adjustment(base_rate: float, wind: float | None, roof: str | None, coefs: tuple[float, float]) -> float:
    """No-op for indoor games or when wind isn't known yet (far-in-advance
    predictions, before a real forecast exists)."""
    if roof != "outdoors" or wind is None or (isinstance(wind, float) and np.isnan(wind)):
        return base_rate
    intercept, slope = coefs
    return max(base_rate + intercept + slope * wind, 0.0)
