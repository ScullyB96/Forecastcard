"""Defines the dev/holdout split and enforces the confirmatory-veto
protocol: holdout is read ONCE, at the end, only to veto a dev-confirmed
improvement -- never to rank candidates or pick between them. Mirrors the
NHL sibling's `final_holdout_check.py` structure (reimplemented
independently, no import).

DEV_MAX_SEASON is a frozen literal, decided once and never recomputed from
`current_nba_season()` at runtime: recomputing it every time this module is
imported would silently shift games from holdout into dev as future seasons
complete, quietly widening the "already looked at" set without anyone
deciding that on purpose. Confirmed live (2026-07-24):
`fetch_schedule.current_nba_season()` returned 2025 (i.e. the 2025-26 season
is the most recently completed one) -- per the user-confirmed convention
(dev = 2015-16 through most-recent-completed-season minus 2, holdout = the
most recent 2 full seasons), that fixes holdout = {2024 (2024-25), 2025
(2025-26)} and dev = seasons 2015-2023 inclusive.
"""

import pandas as pd

DEV_MAX_SEASON = 2024  # exclusive -- dev is season < 2024; holdout is season >= 2024


def split_dev_holdout(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = results[results["season"] < DEV_MAX_SEASON]
    holdout = results[results["season"] >= DEV_MAX_SEASON]
    return dev, holdout
