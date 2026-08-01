"""Pull per-team, per-season rosters (`CommonTeamRoster`) from stats.nba.com
via `nba_api` -- the source of `matchup_difficulty.py`'s per-season
`POSITION` field (e.g. "G", "F", "C-F"), used for the position-group level
of its shrinkage hierarchy.

Fetched PER SEASON deliberately, not just once for the current roster: a
player's position can change across their career (a rookie guard who
bulks up into a forward, a positionless wing reclassified year to year),
and reusing a single current-day snapshot for every season in the dev range
would be a look-ahead leak (using information about a player's role that
wasn't true yet in an earlier season). 30 teams x 11 seasons = 330 calls
total -- cheap enough to just re-fetch a whole season on each resume rather
than tracking per-team completion within a season.

Confirmed live (2026-07-25): works across the full range, no boundary issue
unlike `BoxScoreMatchupsV3`/`BoxScoreDefensiveV2` (see
`fetch_boxscore_matchups.py`) -- `CommonTeamRoster` is a roster snapshot, not
tracking-derived, so it isn't gated by the Second Spectrum rollout.

NOTE: `build_stints.py` already found `BoxScoreTraditionalV3`'s own
`position` column isn't reliably populated across every era -- do NOT reuse
that column for position-group logic. This is a separate, more reliable
source specifically for that purpose.

Same resumable, checkpointed, per-season-parquet caching pattern as the
other `fetch_*.py` scripts, checkpointed per TEAM (not per game -- there is
no per-game concept here).
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import commonteamroster

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, season_str
from src.ingest.team_codes import ABBREV_TO_TEAM_ID
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3


def _fetch_team_with_retry(team_id: int, season_label: str) -> pd.DataFrame | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season_label, timeout=30)
            df = roster.get_data_frames()[0].copy()
            if df.empty:
                return None
            return df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: team {team_id} roster for {season_label} failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_rosters_season(start_year: int, force: bool = False) -> pd.DataFrame:
    season_label = season_str(start_year)
    path = DATA_RAW / f"team_rosters_{season_label}.parquet"
    existing = pd.read_parquet(path) if path.exists() and not force else pd.DataFrame()
    done_team_ids = set(existing["TeamID"].unique()) if not existing.empty else set()

    todo_team_ids = [tid for tid in ABBREV_TO_TEAM_ID.values() if tid not in done_team_ids]
    if not todo_team_ids:
        print(f"{season_label}: rosters already complete ({len(done_team_ids)} teams cached)", flush=True)
        return existing

    print(f"{season_label}: fetching rosters for {len(todo_team_ids)} of {len(ABBREV_TO_TEAM_ID)} teams "
          f"({len(done_team_ids)} already cached)...", flush=True)

    new_frames = []
    for i, team_id in enumerate(todo_team_ids):
        df = _fetch_team_with_retry(team_id, season_label)
        time.sleep(REQUEST_DELAY_SECONDS)
        if df is None:
            continue
        new_frames.append(df)
        existing = pd.concat([existing, df], ignore_index=True)
        existing.to_parquet(path, index=False)
        print(f"  ...{i + 1}/{len(todo_team_ids)} teams fetched, checkpointed "
              f"({existing['TeamID'].nunique()} teams, {len(existing)} rows total)", flush=True)

    return existing


def fetch_team_rosters_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_rosters_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling team rosters {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_team_rosters_all(current)
    print("done", flush=True)
