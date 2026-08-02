"""Automated daily pipeline: refresh data -> (once built) rebuild every rating
engine -> predict today's games. Meant to run once a day (e.g. via a
scheduled task), unattended, hosted.

This is the DATA-REFRESH skeleton only for now -- the true-talent/matchup/
simulation engine (Phase 2-3 of the project plan) doesn't exist yet. Wiring
prediction logic in later should be additive: this module's job is to
guarantee that by the time prediction code runs, every data source is fresh
and verified, and it should stay that way even as models get added.

Key design difference from the NFL project's weekly pipeline: MLB plays
almost every day and has 10+ seasons of pitch-level Statcast history, so
"refresh everything with force=True every run" (fine for a small weekly
league) would be far too slow/wasteful here. Instead: complete past seasons
are fetched once and never touched again; only the current season is
refreshed, and only incrementally (see src/ingest/fetch.py).

`print(..., flush=True)` is used throughout rather than plain print --
learned the hard way *while building this*: a backgrounded/unattended process
that isn't attached to a terminal block-buffers stdout, so without flush=True
a scheduled daily run's logs can sit invisible until the process exits,
which defeats the point of logging progress for a hosted, unattended job.
"""

import datetime as _dt

import pandas as pd

from src.ingest.fetch import (
    current_mlb_season,
    fetch_schedule_season,
    fetch_statcast_season,
    fetch_stolen_base_stats_for_seasons,
    verify_and_backfill_statcast,
)
from src.utils.paths import DATA_RAW

HISTORY_START_SEASON = 2023  # earliest season currently cached; expand later if the model needs more depth


def refresh_all_data() -> dict:
    """Complete seasons are fetched only if missing; the current season is
    always refreshed incrementally. Returns a small status summary."""
    season = current_mlb_season()
    seasons = list(range(HISTORY_START_SEASON, season + 1))
    status = {"season": season, "seasons_covered": seasons, "errors": []}

    for s in seasons:
        try:
            fetch_schedule_season(s)
        except Exception as e:
            print(f"  ERROR fetching schedule for {s}: {e}", flush=True)
            status["errors"].append(f"schedule_{s}: {e}")
        try:
            fetch_statcast_season(s)
        except Exception as e:
            print(f"  ERROR fetching statcast for {s}: {e}", flush=True)
            status["errors"].append(f"statcast_{s}: {e}")
        # cross-source check: catches a silent partial-fetch that raised no
        # exception (the real failure mode that once cost the sibling NFL
        # project two whole missing seasons) -- see verify_and_backfill_statcast.
        try:
            verify_and_backfill_statcast(s)
        except Exception as e:
            print(f"  ERROR verifying statcast coverage for {s}: {e}", flush=True)
            status["errors"].append(f"verify_statcast_{s}: {e}")

    # task #163 (2026-08-02): this was NEVER called from anywhere in the
    # automated pipeline -- stolen_bases.parquet only existed because every
    # local dev machine had it from a one-time manual backfill run when
    # baserunning.py was first built (task #74). A genuinely fresh
    # environment (confirmed on Railway) has no such file, which crashes
    # build_pregame_sb_rates with a real ZeroDivisionError (see baserunning.py's
    # own fix for the defensive side of this). This call is already
    # per-game-checkpointed/resumable (only fetches games not already in the
    # cache, see fetch_stolen_base_stats_for_seasons's own docstring) -- same
    # "expensive once, cheap forever after" shape as the schedule/statcast
    # fetches above, not a new kind of cost. The FIRST run anywhere without a
    # persisted cache (e.g. a fresh Railway volume) will be slow -- one real
    # API call per historical game across every season in `seasons`
    # (thousands of calls) -- exactly like this same run's own first-ever
    # statcast backfill was slow, and for the identical reason.
    try:
        fetch_stolen_base_stats_for_seasons(seasons)
    except Exception as e:
        print(f"  ERROR fetching stolen-base stats: {e}", flush=True)
        status["errors"].append(f"stolen_bases: {e}")

    return status


def todays_games() -> pd.DataFrame:
    """The current season's schedule for today's date -- whatever teams are
    playing, with probable pitchers and (once posted, usually a few hours
    pre-game) confirmed lineups."""
    season = current_mlb_season()
    path = DATA_RAW / f"schedule_{season}.parquet"
    if not path.exists():
        return pd.DataFrame()
    sched = pd.read_parquet(path)
    today = str(_dt.date.today())
    return sched[sched["date"] == today]


if __name__ == "__main__":
    print(f"=== MLB daily pipeline run: {_dt.date.today()} ===", flush=True)
    print("refreshing raw data...", flush=True)
    status = refresh_all_data()
    if status["errors"]:
        print(f"\ncompleted with {len(status['errors'])} error(s): {status['errors']}", flush=True)
    else:
        print("\nall sources refreshed cleanly.", flush=True)

    games = todays_games()
    print(f"\n=== today's games ({_dt.date.today()}) ===", flush=True)
    if games.empty:
        print("no games scheduled today (or today's schedule hasn't been fetched yet).", flush=True)
    else:
        cols = ["home_team", "away_team", "home_probable_pitcher_name", "away_probable_pitcher_name", "status"]
        print(games[cols].to_string(index=False), flush=True)
        print("\n(prediction engine not built yet -- this is the data-refresh skeleton; "
              "score/prop predictions will be added here once the true-talent + simulation "
              "layers exist)", flush=True)
