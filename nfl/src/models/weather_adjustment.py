"""Weather adjustment for QB yards/attempt and WR/TE yards/target -- validated
walk-forward (TRAIN=2018-2021 fit / TEST=2022-2025 held out). The fit itself
operates on the RATE (yards/attempt, yards/target); the MAE numbers below are
on the resulting projected TOTAL yards for the game (rate * volume), not on
the rate itself -- caught and corrected 2026-08 (external review): a rate-MAE
of 44 is physically impossible at this scale (real QB ypa is ~7).

Two weather variables, fit JOINTLY (`fit_weather_adjustment`/`apply_weather_adjustment`):

  wind only (original, kept for the historical record):
    QB projected passing yards:      MAE 44.72 -> 44.09 (p=0.0022, n=1282)
    WR/TE projected receiving yards: MAE 13.55 -> 13.31 (p<0.0001, n=9534)

  temperature added 2026-08 (`src/models/validate_temperature_effect.py`; real, unused
  `temp` column already in schedules -- same physical intuition as wind, colder reduces
  grip/ball feel for passers and receivers). Wind and temp have a real but weak negative
  correlation across outdoor games (r=-0.098) -- fit JOINTLY, not as two independent
  univariate corrections, so temp's own contribution is isolated from wind bleeding
  through that correlation:
    QB projected passing yards:      MAE 44.72 -> 44.00 (temp t=+3.39 controlling for
                                      wind, wind t=-3.57; p=0.0086, n=1282)
    WR/TE projected receiving yards: MAE 13.56 -> 13.31 (temp t=+2.19 controlling for
                                      wind, wind t=-3.31; p<0.0001, n=9534)
  Joint beats either variable's own univariate-only correction on both statlines --
  temp is real, independent signal, not wind bleeding through the correlation.

Completion rate showed the same directionally-correct (negative, for wind) raw
correlation but did NOT clear significance (p=0.34, n=1282) -- not included
here. Don't add it without a lot more data or a different approach; a
correlation existing in the raw crosstab is not sufficient on its own, per
this project's standing discipline (this is exactly the mistake almost made
here the first time, from a slope/intercept unpacking bug that silently
flipped the sign of the fitted adjustment -- always sanity-check a fitted
sign against the raw correlation before trusting it).

Wind/temp are only meaningfully knowable close to kickoff (forecasts aren't
reliable until Fri/Sat) -- this adjustment is a no-op for games far in
advance (both are NaN until schedules data gets a real forecast, and always
missing together -- confirmed directly, zero games have one without the
other), and is the main thing a near-kickoff refresh run actually gains over
the early-week one, alongside fresher inactive-list data.
"""

import numpy as np
import pandas as pd

FIT_SEASONS = set(range(2018, 2026))


def fit_weather_adjustment(rate_result: pd.DataFrame, actual_col: str, touch_col: str, pregame_rate_col: str) -> tuple[float, float, float]:
    """rate_result must already have wind/temp/roof merged in (outdoor games only
    matter; dome/closed games have neither and get filtered out here).
    Returns (intercept, wind_slope, temp_slope) for:
    rate_residual = intercept + wind_slope * wind + temp_slope * temp."""
    fit = rate_result[
        (rate_result["roof"] == "outdoors") & rate_result["wind"].notna() & rate_result["temp"].notna()
        & rate_result["season"].isin(FIT_SEASONS)
    ].copy()
    resid = (fit[actual_col] / fit[touch_col] - fit[pregame_rate_col]).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(fit)), fit["wind"].to_numpy(dtype=float), fit["temp"].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])


def apply_weather_adjustment(base_rate: float, wind: float | None, temp: float | None, roof: str | None,
                              coefs: tuple[float, float, float]) -> float:
    """No-op for indoor games or when weather isn't known yet (far-in-advance
    predictions, before a real forecast exists)."""
    if (roof != "outdoors" or wind is None or temp is None
            or (isinstance(wind, float) and np.isnan(wind)) or (isinstance(temp, float) and np.isnan(temp))):
        return base_rate
    intercept, wind_slope, temp_slope = coefs
    return max(base_rate + intercept + wind_slope * wind + temp_slope * temp, 0.0)
