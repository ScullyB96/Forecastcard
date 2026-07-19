"""Pull raw NFL data via nfl_data_py and cache it locally as parquet.

Every fetch_* function is idempotent: it skips the network call if a cached
file already exists, unless force=True. Run this module directly to (re)build
the full raw cache for a season range.
"""

import time

import nfl_data_py as nfl

from src.utils.paths import DATA_RAW

MAX_FETCH_RETRIES = 2


def _cached(name: str, seasons: list[int], loader, force: bool = False):
    path = DATA_RAW / f"{name}_{min(seasons)}_{max(seasons)}.parquet"
    if path.exists() and not force:
        return path

    # nfl_data_py's multi-season loaders can silently drop a season (a transient
    # per-year fetch hiccup inside its own loop) without raising -- confirmed
    # this project's own cache once went missing 2017 and 2023 from an 11-season
    # pbp pull with no exception at all. Retry, and only accept a result once
    # every requested season with a "season" column is actually present.
    df = None
    for attempt in range(MAX_FETCH_RETRIES + 1):
        df = loader()
        if "season" not in df.columns:
            break
        missing = set(seasons) - set(df["season"].unique())
        if not missing:
            break
        # missing only the newest requested season almost always means "that
        # season hasn't started yet" (no amount of retrying produces plays for
        # a season with zero games played) -- let the caller's own fallback
        # (e.g. _fetch_with_fallback) handle that by trimming the range instead
        # of burning retries on an expensive full refetch that can't succeed.
        if missing == {max(seasons)}:
            raise RuntimeError(f"{name}: no data yet for season {max(seasons)}")
        if attempt < MAX_FETCH_RETRIES:
            print(f"  {name}: fetch silently missing season(s) {sorted(missing)} "
                  f"(attempt {attempt + 1}/{MAX_FETCH_RETRIES + 1}), retrying...")
            time.sleep(2)
        else:
            raise RuntimeError(f"{name}: fetch still missing season(s) {sorted(missing)} "
                                f"after {MAX_FETCH_RETRIES + 1} attempts")

    df.to_parquet(path, index=False)
    return path


def fetch_schedules(seasons: list[int], force: bool = False):
    return _cached("schedules", seasons, lambda: nfl.import_schedules(seasons), force)


def fetch_pbp(seasons: list[int], force: bool = False):
    return _cached(
        "pbp", seasons, lambda: nfl.import_pbp_data(seasons, downcast=True, cache=False), force
    )


def fetch_weekly_rosters(seasons: list[int], force: bool = False):
    return _cached(
        "weekly_rosters", seasons, lambda: nfl.import_weekly_rosters(seasons), force
    )


def fetch_weekly_data(seasons: list[int], force: bool = False):
    return _cached(
        "weekly_player_stats",
        seasons,
        lambda: nfl.import_weekly_data(seasons, downcast=True),
        force,
    )


def fetch_snap_counts(seasons: list[int], force: bool = False):
    return _cached("snap_counts", seasons, lambda: nfl.import_snap_counts(seasons), force)


def fetch_injuries(seasons: list[int], force: bool = False):
    return _cached("injuries", seasons, lambda: nfl.import_injuries(seasons), force)


def fetch_ngs(stat_type: str, seasons: list[int], force: bool = False):
    """stat_type is one of 'passing', 'receiving', 'rushing'."""
    return _cached(
        f"ngs_{stat_type}", seasons, lambda: nfl.import_ngs_data(stat_type, seasons), force
    )


def fetch_all(seasons: list[int], force: bool = False) -> dict[str, str]:
    paths = {
        "schedules": fetch_schedules(seasons, force),
        "pbp": fetch_pbp(seasons, force),
        "weekly_rosters": fetch_weekly_rosters(seasons, force),
        "weekly_player_stats": fetch_weekly_data(seasons, force),
        "injuries": fetch_injuries(seasons, force),
        "ngs_passing": fetch_ngs("passing", seasons, force),
        "ngs_receiving": fetch_ngs("receiving", seasons, force),
        "ngs_rushing": fetch_ngs("rushing", seasons, force),
    }
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    import sys

    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    result = fetch_all(list(range(start, end + 1)))
    for name, path in result.items():
        print(f"{name}: {path}")
