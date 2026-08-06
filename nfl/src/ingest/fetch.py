"""Pull raw NFL data via nflreadpy and cache it locally as parquet.

Migrated off `nfl_data_py` (2026-07): confirmed directly against the real GitHub repo that it
was archived read-only on 2025-09-25, with the maintainers' own words -- "nfl_data_py has been
deprecated in favour of nflreadpy. All future development will occur in nflreadpy and users are
encouraged to switch immediately. No further nfl_data_py maintenance or updates are planned."
Running a live pipeline on a permanently unmaintained package (no further releases, ever) one
season before kickoff was flagged as this project's single highest-priority structural risk by
an external review, verified rather than taken on faith.

`nflreadpy` returns Polars DataFrames, not pandas -- every loader call below converts via
`.to_pandas()` immediately, so every downstream file (which only ever reads the cached parquet
output of this module, never imports nflreadpy directly) needs zero changes. Verified by direct,
hands-on comparison (real 2024 data, both libraries, columns AND values) before migrating:
schedules/pbp/snap_counts/draft_picks are schema- and value-identical to the old source for
every column this codebase actually reads. Weekly rosters rename `gsis_id`->`player_id` and
`full_name`->`player_name` right after fetching (nflreadpy's own column names) specifically so
every downstream consumer of the old column names keeps working unchanged; `age` is dropped
(confirmed unused anywhere in this project). Injuries drops `season_type` (confirmed unused
here too) but needs no renames -- `full_name`/`gsis_id`/`report_status` were already named
identically in both sources.

Every fetch_* function is idempotent: it skips the network call if a cached
file already exists, unless force=True. Run this module directly to (re)build
the full raw cache for a season range.
"""

import time

import nflreadpy as nfr

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
    return _cached("schedules", seasons, lambda: nfr.load_schedules(seasons).to_pandas(), force)


def fetch_pbp(seasons: list[int], force: bool = False):
    return _cached("pbp", seasons, lambda: nfr.load_pbp(seasons).to_pandas(), force)


def _load_weekly_rosters(seasons: list[int]):
    df = nfr.load_rosters_weekly(seasons).to_pandas()
    # nflreadpy's own column names -- renamed back to nfl_data_py's so every downstream
    # consumer of player_id/player_name (15+ files) needs zero changes. See module docstring.
    return df.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})


def fetch_weekly_rosters(seasons: list[int], force: bool = False):
    return _cached("weekly_rosters", seasons, lambda: _load_weekly_rosters(seasons), force)


def fetch_weekly_data(seasons: list[int], force: bool = False):
    return _cached(
        "weekly_player_stats",
        seasons,
        lambda: nfr.load_player_stats(seasons).to_pandas(),
        force,
    )


def fetch_snap_counts(seasons: list[int], force: bool = False):
    return _cached("snap_counts", seasons, lambda: nfr.load_snap_counts(seasons).to_pandas(), force)


def fetch_injuries(seasons: list[int], force: bool = False):
    return _cached("injuries", seasons, lambda: nfr.load_injuries(seasons).to_pandas(), force)


def fetch_ngs(stat_type: str, seasons: list[int], force: bool = False):
    """stat_type is one of 'passing', 'receiving', 'rushing'."""
    return _cached(
        f"ngs_{stat_type}", seasons,
        lambda: nfr.load_nextgen_stats(seasons, stat_type=stat_type).to_pandas(), force,
    )


def fetch_draft_picks(seasons: list[int], force: bool = False):
    """Draft round/pick/position, keyed by real gsis_id -- used by
    rookie_prior.py's draft-capital usage prior (review #2.6)."""
    return _cached("draft_picks", seasons, lambda: nfr.load_draft_picks(seasons).to_pandas(), force)


def fetch_all(seasons: list[int], force: bool = False) -> dict[str, str]:
    # NOTE: does NOT call fetch_weekly_data() (nfl.import_weekly_data) --
    # confirmed unused by any downstream script (housekeeping audit, review
    # 2026-07 #3.4). The live pipeline (weekly_update.py) deliberately never
    # uses it either: import_weekly_data() lags the season by design, so
    # weekly player stats are always derived from our own PBP data instead
    # (build_weekly_stats_from_pbp.py). fetch_weekly_data() itself is kept
    # (harmless, occasionally useful for an ad-hoc comparison), just not spent
    # on by default here.
    paths = {
        "schedules": fetch_schedules(seasons, force),
        "pbp": fetch_pbp(seasons, force),
        "weekly_rosters": fetch_weekly_rosters(seasons, force),
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
