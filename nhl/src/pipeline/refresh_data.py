"""Refresh the cached data `predict_game.py` depends on, before a live
daily prediction run. Mirrors the sibling NBA/MLB projects' complete-vs-
current-season split where each source's own refresh mechanics allow it.

Confirmed live 2026-07-23 (MODEL_DOCUMENTATION.md Sec1): MoneyPuck's team-
situational CSV and goalie-shots archive are both "refreshed in place" at
the source -- a plain wholesale re-fetch always gets the current state, no
incremental logic needed or possible. The NHL API schedule is the one
source that needs incremental handling: `fetch_nhl_api.py`'s own `__main__`
hardcodes a fixed end date, which would silently stop picking up new games
if just re-run as-is.

**Known, flagged inefficiency, not silently ignored**: `fetch_moneypuck_
goalie_games.py`'s wholesale re-fetch downloads the FULL 2007-2024 shots
archive (a large file) every single call, even though the 2007-2024 season
range essentially never changes. Fine for the Sep 19-20 dress rehearsal and
early testing; if daily refresh latency becomes a real problem once this
runs every day of the season, worth adding real caching (e.g. skip the
historical zip once its own row count stops growing) -- not built now
since correctness came first and this hasn't been measured as a real
problem yet (see MODEL_DOCUMENTATION.md Sec42 for the canary/latency work
this should be measured alongside).
"""

from datetime import date, timedelta

import pandas as pd

from src.ingest.fetch_moneypuck import fetch_all_teams_gamelog
from src.ingest.fetch_moneypuck_goalie_games import fetch_and_build_goalie_game_log
from src.ingest.fetch_nhl_api import fetch_schedule_range
from src.ingest.schema_guards import assert_moneypuck_situational_schema, assert_schedule_schema
from src.utils.paths import DATA_RAW

SCHEDULE_PATH = DATA_RAW / "nhl_schedule_2008_2026.parquet"
REFETCH_BUFFER_DAYS = 14  # re-pull a couple weeks before the last cached date too, in case a
# late score correction/lastPeriodType update landed after that game was first cached


def refresh_schedule() -> pd.DataFrame:
    """Incremental: re-fetches from (last cached gameDate - buffer) through
    today, and merges into the existing cache -- NOT a full 2008-2026
    re-walk (correct either way, since `fetch_schedule_range` dedupes by
    gameId, but a full re-walk is ~1000 requests for no benefit once the
    cache already covers most of history)."""
    existing = pd.read_parquet(SCHEDULE_PATH) if SCHEDULE_PATH.exists() else pd.DataFrame()
    if existing.empty:
        start_date = "2008-08-01"
    else:
        last_date = pd.to_datetime(existing["gameDate"]).max()
        start_date = (last_date - timedelta(days=REFETCH_BUFFER_DAYS)).strftime("%Y-%m-%d")
    end_date = date.today().isoformat()

    fresh = fetch_schedule_range(start_date, end_date)
    # Validate the FRESH data before merging/overwriting the cache -- a schema break must not
    # silently corrupt the last known-good cached file (task #44's degradation budget and Sec38's
    # emergency-fix path both assume a provider break is noticed here, not discovered later as
    # silently-wrong predictions).
    assert_schedule_schema(fresh)
    if existing.empty:
        combined = fresh
    else:
        combined = pd.concat([existing, fresh], ignore_index=True).drop_duplicates(
            subset=["gameId"], keep="last").sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    combined.to_parquet(SCHEDULE_PATH, index=False)
    print(f"refresh_schedule: {len(combined)} total games cached (fetched {start_date}..{end_date}, "
          f"{len(fresh)} rows in that window)", flush=True)
    return combined


def refresh_moneypuck_situational() -> None:
    df = fetch_all_teams_gamelog()
    assert_moneypuck_situational_schema(df)
    df.to_parquet(DATA_RAW / "moneypuck_all_teams.parquet", index=False)
    print(f"refresh_moneypuck_situational: {len(df)} rows cached", flush=True)


def refresh_moneypuck_goalies() -> None:
    df = fetch_and_build_goalie_game_log()
    df.to_parquet(DATA_RAW / "moneypuck_goalie_games.parquet", index=False)
    print(f"refresh_moneypuck_goalies: {len(df)} rows cached", flush=True)


def refresh_all() -> None:
    refresh_schedule()
    refresh_moneypuck_situational()
    refresh_moneypuck_goalies()


if __name__ == "__main__":
    refresh_all()
