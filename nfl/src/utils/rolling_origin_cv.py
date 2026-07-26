"""Rolling-origin (blocked walk-forward) cross-validation harness, reused
across future validation scripts (review 2026-07 #3.1). Every prior
validation in this project used a single fixed TRAIN/TEST split (2018-2021 /
2022-2025) -- honest and walk-forward-safe, but a single point estimate can't
distinguish "real, stable signal" from "happened to work on this one split."
Rolling-origin CV instead walks the origin forward one season at a time,
refitting on everything before it and scoring the next season, producing a
per-season fold that reveals whether an effect (an MAE improvement, a fitted
coefficient's sign) is CONSISTENT fold-to-fold or fragile.

This would have caught market_blend.py's own documented multicollinearity
instability (its "our model" coefficient swinging from -0.06 to +0.17
depending on the exact window, §4.2) directly from the fold-to-fold spread,
without needing RidgeCV as a separate diagnostic step.
"""

import numpy as np
import pandas as pd


def rolling_origin_folds(seasons: list[int], min_train_seasons: int = 4) -> list[tuple[set, int]]:
    """[(train_seasons, test_season), ...]: each fold trains on every season
    strictly before test_season, requiring at least min_train_seasons of
    history before the first fold (matching this project's own convention of
    a 2016-2017 burn-in before TRAIN starts at 2018)."""
    seasons = sorted(seasons)
    return [
        (set(seasons[:i]), seasons[i])
        for i in range(min_train_seasons, len(seasons))
    ]


def evaluate_rolling_origin(fit_fn, score_fn, seasons: list[int], min_train_seasons: int = 4) -> pd.DataFrame:
    """fit_fn(train_seasons: set) -> params (whatever shape score_fn expects).
    score_fn(params, test_season: int) -> dict of at least {"mae": float};
    include any fitted coefficient(s) under their own key(s) so
    fold-to-fold sign/magnitude consistency can be read directly off the
    returned table (e.g. {"mae": ..., "our_weight": ...}).

    Returns one row per fold plus the train/test season labels -- inspect
    the coefficient column's sign and magnitude across rows directly; a
    real, stable effect keeps the same sign fold-to-fold, a fragile one
    doesn't (see market_blend.py's own documented example, §4.2).
    """
    rows = []
    for train_seasons, test_season in rolling_origin_folds(seasons, min_train_seasons):
        params = fit_fn(train_seasons)
        result = score_fn(params, test_season)
        result = {
            "train_seasons": f"{min(train_seasons)}-{max(train_seasons)}",
            "n_train_seasons": len(train_seasons),
            "test_season": test_season,
            **result,
        }
        rows.append(result)
    return pd.DataFrame(rows)


def sign_consistency(df: pd.DataFrame, col: str) -> float:
    """Fraction of folds where `col`'s sign matches the modal sign across all
    folds -- 1.0 = perfectly consistent, 0.5 = coin-flip/fragile."""
    signs = np.sign(df[col])
    modal_sign = signs.mode().iloc[0]
    return float((signs == modal_sign).mean())
