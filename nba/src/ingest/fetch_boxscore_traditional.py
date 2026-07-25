"""Pull per-game traditional box scores from stats.nba.com via `nba_api` and
cache one parquet per season -- specifically for the `position` field
(non-blank exactly for that game's 5 starters, blank for bench/DNP,
confirmed live 2026-07-24 against a real game), which is what
`build_stints.py`'s substitution-event-based stint construction uses to
seed each game's opening lineup.

This endpoint replaces `GameRotation` as the primary stint-construction
input (see MODEL_DOCUMENTATION.md Sec4 Finding 3 / the pivot writeup) --
`GameRotation` was found to be persistently, severely unreliable (server-side
degradation, confirmed via direct timing: many requests taking a consistent
~30s regardless of outcome), while `BoxScoreTraditionalV3` -- same V3
family, same general reliability profile as `BoxScoreAdvancedV3` -- was
confirmed live to succeed 8/8 in well under 0.5s each. Combined with
`PlayByPlayV3`'s Substitution events (already fetched, already reliable),
starters + a chronological walk through substitutions reconstructs the same
on-court lineup information `GameRotation` would have provided, without
depending on it at all.

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscores.py` -- see that module's docstring for the rationale.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoretraditionalv3

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 50


def _fetch_game_with_retry(game_id: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30)
            df = box.get_data_frames()[0].copy()
            if df.empty:
                return None
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} traditional box score failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_traditional_season(start_year: int, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"boxscore_trad_player_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["gameId"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: traditional box scores already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching traditional box scores for {len(todo)} of {len(schedule)} games "
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


def fetch_traditional_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_traditional_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling traditional box scores {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_traditional_all(current)
    print("done", flush=True)
