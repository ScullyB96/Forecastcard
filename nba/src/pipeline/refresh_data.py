"""Nightly incremental data refresh: only the CURRENT season is re-touched.
Every prior, complete season was already fetched once and cached
permanently by the Phase 0 backfill -- re-running this never re-fetches
them (mirrors the MLB sibling's `daily_update.py` complete-vs-current-season
split).

Schedule is force-refetched for the current season every run (cheap -- 1-2
`LeagueGameLog` calls return the WHOLE season's results so far, needed to
pick up newly-final scores). Box scores/play-by-play/traditional-box are
NOT force-refetched -- their existing done-games cache already only fetches
games missing from the cache, which naturally picks up newly-completed
games without ever re-requesting an already-cached one. The season's
STINTS cache, unlike the raw fetches, IS force-rebuilt every run -- it's
derived from the (possibly newly-completed) raw data above, not fetched
from an external source, so there's no cost saving to skipping it and a
real risk of serving stale lineup data otherwise.

BUG FOUND AND FIXED (2026-07-25): this function used to call
`fetch_rotation_season` -- the `GameRotation` endpoint that Sec5 of
MODEL_DOCUMENTATION.md documents as ABANDONED ENTIRELY (server-side
reliability degraded to a multi-day ETA) in favor of a substitution-event
reconstruction from `BoxScoreTraditionalV3` + `PlayByPlayV3`. The live
pipeline's nightly refresh was never updated to match that pivot -- it was
fetching data nothing downstream consumes anymore, while never fetching
the traditional box scores `build_stints.py`'s actual (current) primary
path needs, and never rebuilding the current season's stints cache at all.
Predictive-mode lineup adjustment (Sec7) has no current-season lineup data
to compute trailing minutes from without this fix."""

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_boxscore_traditional import fetch_traditional_season
from src.ingest.fetch_boxscores import fetch_boxscores_season
from src.ingest.fetch_playbyplay import fetch_playbyplay_season
from src.ingest.fetch_schedule import current_nba_season, fetch_season


def refresh_all_data() -> int:
    """Returns the current season's start year (so callers don't need a
    second call to `current_nba_season()`)."""
    current = current_nba_season()
    print(f"refreshing current season {current}...", flush=True)
    fetch_season(current, force=True)
    fetch_boxscores_season(current)
    fetch_playbyplay_season(current)
    fetch_traditional_season(current)
    build_season_stints(current, force=True)
    print("refresh complete", flush=True)
    return current


if __name__ == "__main__":
    refresh_all_data()
