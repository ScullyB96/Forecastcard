"""Pull per-game advanced box scores (team + player PACE/OFF_RATING/DEF_RATING
etc.) from stats.nba.com via `nba_api` and cache one parquet per season.

Confirmed live (2026-07-24): `BoxScoreAdvancedV2` (and, by extension,
`PlayByPlayV2` per the package's own deprecation docstring) is dead --
returns an empty `{}` JSON body for a 2023-24 game, not an error. Only the V3
family is usable across the full dev/holdout range. `BoxScoreAdvancedV3`
returns two datasets: player-level (`dfs[0]`) and team-level (`dfs[1]`), and
the team-level rows already carry `pace`/`offensiveRating`/`defensiveRating`
directly -- no need to separately fetch `BoxScoreTraditionalV3` or re-derive
pace by hand for Phase 1's team-strength engine. `BoxScoreTraditionalV3` is
deferred: not needed until a phase actually requires point/rebound/DNP
detail beyond what V3-advanced and GameRotation already provide.

Caching is per-game, appended into one parquet per season, and resumable:
a completed season is fetched once and never re-touched; re-running this
script only fetches games missing from that season's cache. This matters
because full historical backfill (2015-16..present, ~14,500 games) at the
conventional ~0.6s/request stats.nba.com courtesy delay takes hours, so the
process must survive being killed and restarted without redoing work.
"""

import time

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoreadvancedv3

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, fetch_season, season_str
from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6
MAX_FETCH_RETRIES = 3
CHECKPOINT_EVERY = 50  # games between incremental parquet saves


def _fetch_game_with_retry(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            box = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id, timeout=30)
            dfs = box.get_data_frames()
            player_df, team_df = dfs[0].copy(), dfs[1].copy()
            if player_df.empty and team_df.empty:
                return None  # game truly has no advanced box score (e.g. postponed/vacated)
            return player_df, team_df
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    print(f"  WARNING: game {game_id} failed after {MAX_FETCH_RETRIES} attempts: {last_err}", flush=True)
    return None


def fetch_boxscores_season(start_year: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = fetch_season(start_year)
    is_current = start_year == current_nba_season()

    player_path = DATA_RAW / f"boxscore_adv_player_{season_str(start_year)}.parquet"
    team_path = DATA_RAW / f"boxscore_adv_team_{season_str(start_year)}.parquet"

    existing_player = pd.read_parquet(player_path) if player_path.exists() and not force else pd.DataFrame()
    existing_team = pd.read_parquet(team_path) if team_path.exists() and not force else pd.DataFrame()
    done_ids = set(existing_team["gameId"].unique()) if not existing_team.empty else set()

    todo = schedule[~schedule["gameId"].isin(done_ids)]
    if todo.empty:
        print(f"{season_str(start_year)}: box scores already complete ({len(done_ids)} games cached)", flush=True)
        return existing_player, existing_team

    print(f"{season_str(start_year)}: fetching {len(todo)} of {len(schedule)} games "
          f"({len(done_ids)} already cached){' [current season]' if is_current else ''}...", flush=True)

    new_player_frames, new_team_frames = [], []
    fetched_since_checkpoint = 0
    for i, game_id in enumerate(todo["gameId"]):
        result = _fetch_game_with_retry(game_id)
        time.sleep(REQUEST_DELAY_SECONDS)
        if result is None:
            continue
        p_df, t_df = result
        new_player_frames.append(p_df)
        new_team_frames.append(t_df)
        fetched_since_checkpoint += 1

        if fetched_since_checkpoint >= CHECKPOINT_EVERY or i == len(todo) - 1:
            combined_player = pd.concat([existing_player, *new_player_frames], ignore_index=True) if new_player_frames else existing_player
            combined_team = pd.concat([existing_team, *new_team_frames], ignore_index=True) if new_team_frames else existing_team
            combined_player.to_parquet(player_path, index=False)
            combined_team.to_parquet(team_path, index=False)
            existing_player, existing_team = combined_player, combined_team
            new_player_frames, new_team_frames = [], []
            fetched_since_checkpoint = 0
            print(f"  ...{i + 1}/{len(todo)} fetched, checkpointed ({len(existing_team)} team-game rows total)", flush=True)

    return existing_player, existing_team


def fetch_boxscores_all(end_year: int, force: bool = False) -> None:
    for y in range(FIRST_DEV_SEASON, end_year + 1):
        fetch_boxscores_season(y, force=force)


if __name__ == "__main__":
    current = current_nba_season()
    print(f"backfilling advanced box scores {season_str(FIRST_DEV_SEASON)}..{season_str(current)}", flush=True)
    fetch_boxscores_all(current)
    print("done", flush=True)
