"""Travel-distance/timezone-fatigue diagnostic (task #60). Same
residual-first discipline as `rest_schedule.py`: correlate against the
RESIDUAL (actual - `team_strength.project_game`'s baseline, no travel
term at all), never raw score, to avoid recovering a scheduling
artifact/confound instead of a real fatigue effect.
"""

import numpy as np
import pandas as pd

from src.ingest.team_codes import TEAM_ID_TO_ABBREV
from src.ingest.team_locations import TEAM_LOCATIONS

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_travel_fatigue(log: pd.DataFrame) -> pd.DataFrame:
    """`log`: `team_strength.build_team_game_log`'s own shape (one row per
    team per game -- `team`/`opponent` are numeric teamIds, `is_home`,
    `gameDate`, `season`). Adds:

    - `game_lat`/`game_lon`/`game_tz_zone`: this game's PHYSICAL location
      (the home team's arena, regardless of which side `team` is on).
    - `travel_km`/`tz_shift`: distance (haversine) / signed timezone-zone
      change from THIS TEAM's own immediately-prior game's location.
      Walk-forward safe by construction: a simple `groupby(team,
      season).shift(1)` on this team's OWN row sequence reads only that
      team's own strictly-prior game, no cross-team same-game leak risk
      (same safety argument as `shrinkage._trailing_own_mean_ewma`) --
      unlike `_trailing_league_stat`'s POOLED case, there's no shared
      gameId collapse needed here since each team's travel history is
      independent of the opponent's."""
    log = log.sort_values(["team", "season", "gameDate"]).reset_index(drop=True)
    home_abbrev = log["team"].map(TEAM_ID_TO_ABBREV).where(log["is_home"], log["opponent"].map(TEAM_ID_TO_ABBREV))
    log["game_lat"] = home_abbrev.map(lambda a: TEAM_LOCATIONS[a]["lat"])
    log["game_lon"] = home_abbrev.map(lambda a: TEAM_LOCATIONS[a]["lon"])
    log["game_tz_zone"] = home_abbrev.map(lambda a: TEAM_LOCATIONS[a]["tz_zone"])

    grp = log.groupby(["team", "season"])
    prev_lat = grp["game_lat"].shift(1)
    prev_lon = grp["game_lon"].shift(1)
    prev_tz = grp["game_tz_zone"].shift(1)

    log["travel_km"] = _haversine_km(prev_lat, prev_lon, log["game_lat"], log["game_lon"])
    log["tz_shift"] = log["game_tz_zone"] - prev_tz
    return log
