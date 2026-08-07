"""Pull per-game referee-crew assignments (`BoxScoreSummaryV3`) from
stats.nba.com via `nba_api` and cache one parquet per season -- the raw
material for `referee_rates.py`'s per-official walk-forward tendency
signal (task #63: the earlier "no usable data source" assumption was
wrong, confirmed live 2026-08-06, see MODEL_DOCUMENTATION.md Sec58).

Uses V3 (not `BoxScoreSummaryV2`, which carries a documented data-
availability warning for games on/after 2025-04-10) -- same "prefer the V3
family" convention this project already follows for box scores
(`fetch_boxscore_traditional.py`'s own docstring). `BoxScoreSummaryV3.
get_data_frames()[3]` is the officials table (3 rows per game: gameId,
personId, name, nameI, firstName, familyName, jerseyNum) -- accessed by
POSITIONAL index, not `get_normalized_dict()`, because this nba_api
version's dataset-name metadata for V3 is empty (`get_available_data()`
returns no keys) even though the positional data frames themselves are
real and correct -- confirmed live across 2015-16/2018-19/2022-23, `personId`
values match `BoxScoreSummaryV2`'s `OFFICIAL_ID` for the same real games
(e.g. Zach Zarba=2534, Scott Foster=1162).

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscore_traditional.py`.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoresummaryv3

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 50
OFFICIALS_DATASET_INDEX = 3  # positional -- see module docstring for why not get_normalized_dict()


def _fetch_game_with_retry(game_id: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            box = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=30)
            df = box.get_data_frames()[OFFICIALS_DATASET_INDEX].copy()
            if df.empty:
                return None
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError,
                AttributeError, TypeError) as e:
            # REAL BUG FOUND AND FIXED (2026-08-07, full backfill run): nba_api's own V3 response
            # parser (`boxscoresummaryv3.py`'s `get_arena_info_data`) unconditionally calls
            # `.get()` on the parsed arena dict without a None-check, and raises a bare
            # `AttributeError` (not wrapped in any nba_api-specific exception type) when a game's
            # real API response has a null/missing arena section -- confirmed live: this crashed
            # the ENTIRE multi-hour backfill on one single malformed game near the end of the
            # in-progress 2025-26 season, an uncaught exception this function's original except
            # clause didn't list. `AttributeError`/`TypeError` added so a single library-parser
            # bug on one game degrades to a per-game WARNING (skip, retry, eventually give up on
            # just that game) instead of taking down every other already-fetched game's progress.
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} officials failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_officials_season(start_year: int, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"officials_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["gameId"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: officials already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching officials for {len(todo)} of {len(schedule)} games "
          f"({len(done_ids)} already cached){' [current season]' if is_current else ''}...", flush=True)

    new_frames = []
    fetched_since_checkpoint = 0
    for i, game_id in enumerate(todo["gameId"]):
        df = _fetch_game_with_retry(game_id)
        time.sleep(REQUEST_DELAY_SECONDS)
        if df is None:
            continue
        new_frames.append(df)
        fetched_since_checkpoint += 1

        if fetched_since_checkpoint >= CHECKPOINT_EVERY or i == len(todo) - 1:
            existing = pd.concat([existing, *new_frames], ignore_index=True) if new_frames else existing
            existing.to_parquet(path, index=False)
            new_frames = []
            fetched_since_checkpoint = 0
            print(f"  ...{i + 1}/{len(todo)} fetched, checkpointed "
                  f"({existing['gameId'].nunique()} games, {len(existing)} rows total)", flush=True)

    return existing


def fetch_officials_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_officials_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling officials {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_officials_all(current)
    print("done", flush=True)
