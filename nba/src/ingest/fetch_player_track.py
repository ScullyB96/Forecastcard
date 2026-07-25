"""Pull per-game player-tracking box scores (`BoxScorePlayerTrackV3`) from
stats.nba.com via `nba_api` and cache one parquet per season -- the source
for `reboundChancesOffensive`/`reboundChancesDefensive` (rebound
OPPORTUNITY, not just rebounds grabbed -- a much better exposure
denominator for a rebounding-rate model than raw minutes) and `touches`
(a materially better usage/playmaking-opportunity proxy than minutes).

Confirmed live (2026-07-25) across every era tested (2015-16 through
2023-24): unlike `BoxScoreMatchupsV3`/`BoxScoreDefensiveV2` (which fail for
2015-16/2016-17, see `fetch_boxscore_matchups.py`), this endpoint has NO
coverage boundary -- full range, same general reliability as
`BoxScoreAdvancedV3`/`BoxScoreTraditionalV3`.

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscores.py` -- see that module's docstring for the rationale.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoreplayertrackv3

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 50


def _fetch_game_with_retry(game_id: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            box = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id, timeout=30)
            df = box.get_data_frames()[0].copy()
            if df.empty:
                return None
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} player-track failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_player_track_season(start_year: int, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"player_track_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["gameId"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: player-track already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching player-track for {len(todo)} of {len(schedule)} games "
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


def fetch_player_track_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_player_track_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling player-track {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_player_track_all(current)
    print("done", flush=True)
