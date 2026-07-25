"""Stint exposure-weight down-multiplier for likely garbage time, so a
15-point-lead mop-up stint doesn't count as much RAPM signal as a
competitive one. A testable hyperparameter (validated via its own
bootstrap comparison in `validate_rapm_lineup_adjustment.py`), not assumed
necessary -- `garbage_time_weight` returns 1.0 (no-op) unless a stint
actually crosses the threshold, so this is safe to leave wired in and simply
find it doesn't matter if that's what validation shows.

Threshold follows the same shape as Cleaning the Glass's public garbage-time
definition (margin scales down as the game gets later) but with placeholder,
un-calibrated cutoffs -- not copied from a specific published table, since
this project doesn't have a supporting citation for exact numbers and
un-calibrated placeholders are this project's standing convention for any
constant that hasn't been fit yet.
"""

import numpy as np
import pandas as pd

DOWNWEIGHT_FACTOR = 0.25  # placeholder, un-calibrated


def _margin_threshold(elapsed_fraction: np.ndarray) -> np.ndarray:
    """Wider allowed margin early in the game, narrowing as it progresses --
    a 20-point lead with 10 minutes left is not yet decided; the same margin
    with 2 minutes left usually is."""
    return 25.0 - 15.0 * elapsed_fraction


def add_garbage_time_weight(stints: pd.DataFrame, game_length_tenths: float = 28800.0) -> pd.DataFrame:
    """stints must have startTenths/endTenths and a running score margin
    available BEFORE the stint starts (`marginBeforeStint`, signed from the
    home team's perspective, abs() taken here since either side can be up
    big). Adds `exposureWeight` (1.0 normally, DOWNWEIGHT_FACTOR if the
    stint starts already past the margin threshold for that point in the
    game -- games that go to OT use `game_length_tenths` + realized OT time
    so the fraction still makes sense).

    BUG FOUND AND FIXED (2026-07-25): `total_length` used to be computed as
    a single `s["endTenths"].max()` across the ENTIRE `stints` frame passed
    in -- correct only if `stints` is ever exactly one game, but every real
    caller (`prepare_stints`) passes a whole season's stints at once.
    Confirmed on 2015-16: season-wide max is 40800 (a multi-OT game) while a
    regulation game ends at 28800, so EVERY other game's `elapsed_fraction`
    was being divided by that one game's inflated length instead of its own
    -- silently understating garbage-time down-weighting for ~1,185/1,220
    games that season alone. Fixed by computing `total_length` PER GAME."""
    s = stints.copy()
    if s.empty:
        s["exposureWeight"] = pd.Series(dtype=float)
        return s
    per_game_max = s.groupby("gameId")["endTenths"].transform("max")
    total_length = per_game_max.clip(lower=game_length_tenths)
    elapsed_fraction = (s["startTenths"] / total_length).clip(0, 1)
    threshold = _margin_threshold(elapsed_fraction)
    is_garbage = s["marginBeforeStint"].abs() > threshold
    s["exposureWeight"] = np.where(is_garbage, DOWNWEIGHT_FACTOR, 1.0)
    return s
