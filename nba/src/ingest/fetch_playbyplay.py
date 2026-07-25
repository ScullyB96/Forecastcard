"""Pull per-game play-by-play from stats.nba.com via `nba_api` (`PlayByPlayV3`
only -- `PlayByPlayV2` is deprecated and returns empty JSON for any
2024-25+ game, confirmed live 2026-07-24, so V3 is used across the entire
dev/holdout range for consistency) and cache one parquet per season.

Confirmed live (2026-07-24) against a real 2023-24 game (0022300061):
`clock` is a period-relative ISO-8601 duration string (e.g. "PT11M42.00S"),
`scoreHome`/`scoreAway` are populated only on scoring-relevant rows (blank
otherwise -- forward-fill needed downstream) and `actionType=="Substitution"`
rows exist with `personId` set to the OUTGOING player and `description`
formatted as "SUB: <incoming> FOR <outgoing>" -- this project's stint-builder
(`src/ingest/build_stints.py`) primarily uses `GameRotation`'s
IN_TIME_REAL/OUT_TIME_REAL instead (see that module), with these
Substitution rows used only as a cross-check.

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscores.py` -- see that module's docstring for the rationale.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import playbyplayv3

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 25  # PBP rows are much larger per game than box scores; checkpoint more often


def _fetch_game_with_retry(game_id: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
            df = pbp.get_data_frames()[0].copy()
            if df.empty:
                return None
            df["gameId"] = game_id
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} PBP failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_playbyplay_season(start_year: int, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"playbyplay_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["gameId"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: PBP already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching PBP for {len(todo)} of {len(schedule)} games "
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


def fetch_playbyplay_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_playbyplay_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling play-by-play {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_playbyplay_all(current)
    print("done", flush=True)
