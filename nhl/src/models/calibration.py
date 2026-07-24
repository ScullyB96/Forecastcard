"""Isotonic calibration layer on top of the current best model's win
probability (Cycle 12, MODEL_DOCUMENTATION.md Sec6.9). Fit on a temporal
calibration-fit fold, evaluated on a held-out calibration-eval fold NEVER
used to fit the isotonic mapping -- Niculescu-Mizil & Caruana (2005, cited
in the external research report Sec5.4) found isotonic regression needs
roughly 1,000+ calibration points to be stable; this project's dev set
comfortably clears that regardless of how the temporal split is drawn.

Does NOT need market data -- only this project's own real game outcomes,
already available for every season (unlike the market-benchmark piece in
Sec6.8-6.9, which is genuinely limited to seasons with recovered odds data,
Sec6.1's gap).
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_calibration(train_probs: pd.Series, train_actual_home_win: pd.Series) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(train_probs, train_actual_home_win)
    return iso


def brier(p: pd.Series, actual: pd.Series) -> float:
    return float(((p.clip(1e-6, 1 - 1e-6) - actual) ** 2).mean())


def log_loss(p: pd.Series, actual: pd.Series) -> float:
    pc = p.clip(1e-6, 1 - 1e-6)
    return float(-(actual * np.log(pc) + (1 - actual) * np.log(1 - pc)).mean())
