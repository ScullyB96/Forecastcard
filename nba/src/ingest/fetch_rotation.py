"""Pull per-game on-court rotation intervals (`GameRotation`) from
stats.nba.com via `nba_api` and cache one parquet per season -- the primary
input to `src/ingest/build_stints.py`'s lineup-stint construction.

Confirmed live (2026-07-24) against a real 2023-24 game (0022300061):
`GameRotation` returns two datasets (away team, home team), each one row per
player per continuous on-court interval, with `IN_TIME_REAL`/`OUT_TIME_REAL`
in TENTHS OF A SECOND of cumulative elapsed game time from tip-off (a 48-min
regulation game's max observed OUT_TIME_REAL was 28800 = 48*60*10, confirming
the unit empirically rather than assuming it) -- spanning all periods
including overtime, not reset per period. `PLAYER_PTS`/`PT_DIFF` are also
present per interval (points scored / plus-minus while that player was on
the floor for that interval) and are cheap free cross-check signal for
`build_stints.py`'s box-score reconciliation gate.

Same resumable, checkpointed, per-season-parquet caching pattern as
`fetch_boxscores.py` -- see that module's docstring for the rationale.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import gamerotation

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 2.0  # `GameRotation` empirically fails much more often than the
# advanced-box-score/PBP endpoints at the conventional 0.6s delay -- confirmed (2026-07-24) via
# a live isolated test: even with NO other rotation fetcher running concurrently (box score/PBP
# backfills still running, but those hit different endpoints), GameRotation still failed 11/15
# real requests in a row, escalating from intermittent 500s to outright connection failures --
# see MODEL_DOCUMENTATION.md Sec4 Finding 3. A slower, more patient pace is used here specifically
# for this endpoint.
#
# UPDATED FINDING (2026-07-24, supersedes the note below): a direct per-request timing test
# revealed the real cause isn't (only) a dead/erroring endpoint -- it's that stats.nba.com's
# `gamerotation` backend is simply SLOW right now, full stop. Timing 10 consecutive real
# requests: several took 25-30s to return a genuine 200 with real data (not an error), and the
# ones that failed did so as a 30s read-timeout, not a fast rejection. A 30s timeout was
# therefore silently converting some real-but-slow successes into failures. TIMEOUT raised to 60s
# so a slow-but-real response has room to complete instead of being aborted -- this doesn't make
# the backfill faster (each game can still legitimately take up to ~30s), but it should recover
# some of what were previously being counted as permanent failures.
#
# MAX_FETCH_RETRIES was lowered from an initial 5 to 2 based on an EARLIER, narrower diagnostic
# (a single game_id failing 5/5 with an identical fast error) -- that observation is still real
# for genuinely dead game_ids, but the timing test above shows most of the current slowness is
# query latency, not a dead endpoint, so retries are kept modest rather than pushed back up (an
# already-slow 30-60s request being retried 5x would be far worse for total wall-clock time). A
# separate gap-filling re-run (see fetch_rotation_all's caching) remains the real fix for whatever
# still comes up permanently missing after this change.
MAX_FETCH_RETRIES = 2
CHECKPOINT_EVERY = 50


def _fetch_game_with_retry(game_id: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            gr = gamerotation.GameRotation(game_id=game_id, timeout=60)
            away_df, home_df = gr.get_data_frames()
            if away_df.empty and home_df.empty:
                return None
            away_df, home_df = away_df.copy(), home_df.copy()
            away_df["isHomeTeam"] = False
            home_df["isHomeTeam"] = True
            return pd.concat([away_df, home_df], ignore_index=True)
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    print(f"  WARNING: game {game_id} rotation failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_rotation_season(start_year: int, force: bool = False) -> pd.DataFrame:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    path = DATA_RAW / f"rotation_{season_str(start_year)}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_ids = set(existing["GAME_ID"].unique()) if not existing.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: rotation already complete ({len(done_ids)} games cached)", flush=True)
        return existing

    print(f"{season_str(start_year)}: fetching rotation for {len(todo)} of {len(schedule)} games "
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
                  f"({existing['GAME_ID'].nunique()} games, {len(existing)} rows total)", flush=True)

    return existing


def fetch_rotation_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_rotation_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling rotation data {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_rotation_all(current)
    print("done", flush=True)
