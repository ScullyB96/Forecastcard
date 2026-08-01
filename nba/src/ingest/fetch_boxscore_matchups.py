"""Pull per-game defender-matchup box scores from stats.nba.com via
`nba_api`, for `matchup_difficulty.py`'s defender-difficulty rate model.
Two independent endpoints, each cached to its own per-season parquet:

- `BoxScoreMatchupsV3`: pairwise (offensive player, defensive player) rows --
  matchup minutes, points/FG/3PT/AST/TOV allowed while THAT specific pairing
  was on court. Finer-grained; used for the position-group-vs-opponent level
  of `matchup_difficulty.py`'s shrinkage hierarchy (collapsed by position
  group, not read pairwise).
- `BoxScoreDefensiveV2`: pre-aggregated per-defender rows (one row per
  defender per game, already summed across every offensive player they
  guarded) -- simpler, and the primary source for the defender-specific
  level of the hierarchy.

CONFIRMED LIVE (2026-07-25): both endpoints raise (`IndexError`/
`AttributeError` depending on the endpoint -- stats.nba.com returns an empty
payload, not a clean 404) for 2015-16 and 2016-17, and succeed from 2017-18
onward. This isn't a bug to retry around -- it matches the real Second
Spectrum player-tracking rollout (matchup data didn't exist industry-wide
before 2017-18). `MATCHUP_DATA_START_SEASON` is a frozen constant for this
reason, mirroring `FIRST_DEV_SEASON`'s pattern -- the six per-player rate
models still validate on the full 2015-2025 range; only the matchup-
difficulty LAYER on top is restricted to 2017-2025.

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscores.py`/`fetch_player_track.py`.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoredefensivev2, boxscorematchupsv3

from src.ingest.fetch_schedule import current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

MATCHUP_DATA_START_SEASON = 2017  # frozen -- see module docstring; do NOT lower to FIRST_DEV_SEASON

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 50


def _fetch_with_retry(endpoint_cls, game_id: str, label: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            box = endpoint_cls(game_id=game_id, timeout=30)
            df = box.get_data_frames()[0].copy()
            if df.empty:
                return None
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError, AttributeError, TypeError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} {label} failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def _fetch_season_generic(start_year: int, endpoint_cls, label: str, filename_prefix: str, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"{filename_prefix}_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["gameId"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: {label} already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching {label} for {len(todo)} of {len(schedule)} games "
          f"({len(done_ids)} already cached){' [current season]' if is_current else ''}...", flush=True)

    new_frames = []
    fetched_since_checkpoint = 0
    for i, game_id in enumerate(todo["gameId"]):
        df = _fetch_with_retry(endpoint_cls, game_id, label)
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


def fetch_matchups_season(start_year: int, force: bool = False) -> pd.DataFrame:
    return _fetch_season_generic(start_year, boxscorematchupsv3.BoxScoreMatchupsV3,
                                  "matchups", "boxscore_matchups_v3", force=force)


def fetch_defensive_season(start_year: int, force: bool = False) -> pd.DataFrame:
    return _fetch_season_generic(start_year, boxscoredefensivev2.BoxScoreDefensiveV2,
                                  "defensive", "boxscore_defensive_v2", force=force)


def fetch_boxscore_matchups_all(end_year: int, force: bool = False) -> None:
    for y in range(MATCHUP_DATA_START_SEASON, end_year + 1):
        fetch_matchups_season(y, force=force)
        fetch_defensive_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling boxscore matchups {season_str(MATCHUP_DATA_START_SEASON)}..{season_str(current)}", flush=True)
    fetch_boxscore_matchups_all(current)
    print("done", flush=True)
