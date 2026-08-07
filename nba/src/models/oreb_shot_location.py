"""OREB shot-location decomposition (follow-up to task #55's oreb_decomposition.py,
prompted by external research confirming rebound-recovery rate varies
substantially by shot zone -- e.g. long rebounds off 3PT misses behave very
differently from short rebounds off interior misses -- a mechanism none of
the five previously-tested OREB mechanisms captured, since they all worked
from team-AGGREGATE miss counts with no zone information at all.

NO NEW INGEST NEEDED: `playbyplay_*.parquet` (already backfilled 2015-2025)
already has `shotDistance`/`shotValue` for every "Missed Shot" action, and
the immediately-following "Rebound" action's `teamId` tells us whether the
shooting team recovered it (OREB) or the opponent did (DREB) -- confirmed
live against real PBP data. `stats.nba.com`'s ShotChartDetail endpoint
(what the original research angle assumed was needed) is NOT required.

Real quirk found and handled: BLOCK/STEAL credit rows share the same
`actionNumber` as their parent event but have `actionType == ""` (an empty
string, not NaN) -- confirmed live these are secondary-credit rows, not
real chronological events. Naively taking "the next row after a miss"
without filtering these out grabs a block-credit row instead of the real
rebound. Fixed by filtering to `actionType != ""` FIRST, before finding
each miss's next event.
"""

import numpy as np
import pandas as pd

from src.models.oreb_decomposition import add_walk_forward_exposure_rate, project_oreb_share
from src.models.shrinkage import add_walk_forward_rate
from src.models.team_stat_rates import project_team_stat

PRIOR_GAMES_MISSES_ZONE = 20.0  # un-calibrated placeholder, same convention as oreb_decomposition.py's
# flat (non-zone) misses model -- swept before adoption.
PRIOR_MISSES_OREB_RATE_ZONE = 100.0  # exposure unit is MISSES-IN-THIS-ZONE, smaller prior than the
# flat model's 200 since each zone sees roughly 1/4 the exposure of the pooled total -- swept before adoption.
PRIOR_MISSES_DREB_RATE_ZONE = 100.0

# Zone bucketing: shotValue (2 vs 3, already correctly flags the 3PT line, more
# reliable than a distance-only cutoff since corner-3 range starts closer than
# the top-of-key arc) combined with shotDistance for the 2PT sub-buckets.
ZONE_RIM = "rim"           # 2PT, distance 0-4ft
ZONE_SHORT_MID = "short_mid"  # 2PT, distance 5-13ft
ZONE_LONG_MID = "long_mid"    # 2PT, distance 14ft+
ZONE_THREE = "three"          # any 3PT attempt


def _zone_for_row(shot_value: pd.Series, shot_distance: pd.Series) -> pd.Series:
    zone = pd.Series(ZONE_LONG_MID, index=shot_value.index)
    zone = zone.where(shot_value == 2, ZONE_THREE)
    zone = zone.where(~((shot_value == 2) & (shot_distance <= 13)), ZONE_SHORT_MID)
    zone = zone.where(~((shot_value == 2) & (shot_distance <= 4)), ZONE_RIM)
    return zone


def build_miss_rebound_log(pbp: pd.DataFrame) -> pd.DataFrame:
    """`pbp`: raw cached play-by-play (gameId, actionNumber, actionType,
    teamId, shotDistance, shotValue). Returns one row per real missed
    field-goal attempt that was followed by a REAL (non-dead-ball, teamId
    != 0) rebound: gameId, shooting_team, zone, is_oreb (1.0 if the
    shooting team itself recovered it, 0.0 if the opponent did). Dead-ball
    rebounds (teamId == 0, e.g. ball out of bounds) are excluded entirely
    -- genuinely ambiguous, not fabricated as either outcome."""
    p = pbp[pbp["actionType"] != ""].copy()
    p = p.sort_values(["gameId", "actionNumber"]).reset_index(drop=True)

    is_miss = (p["actionType"] == "Missed Shot").to_numpy()
    is_rebound = (p["actionType"] == "Rebound").to_numpy()
    same_game_as_next = (p["gameId"].to_numpy()[:-1] == p["gameId"].to_numpy()[1:])

    next_is_rebound = np.zeros(len(p), dtype=bool)
    next_is_rebound[:-1] = is_rebound[1:] & same_game_as_next
    miss_with_rebound = is_miss & next_is_rebound

    idx = np.flatnonzero(miss_with_rebound)
    shooting_team = p["teamId"].to_numpy()[idx]
    rebound_team = p["teamId"].to_numpy()[idx + 1]
    valid = rebound_team != 0  # exclude dead-ball/no-team-credited rebounds

    zone = _zone_for_row(p["shotValue"], p["shotDistance"])
    return pd.DataFrame({
        "gameId": p["gameId"].to_numpy()[idx][valid],
        "shooting_team": shooting_team[valid],
        "rebounding_team": rebound_team[valid],
        "zone": zone.to_numpy()[idx][valid],
        "is_oreb": (shooting_team[valid] == rebound_team[valid]).astype(float),
    })


ZONES = (ZONE_RIM, ZONE_SHORT_MID, ZONE_LONG_MID, ZONE_THREE)


def build_zone_team_game_log(miss_rebound_log: pd.DataFrame, base_log: pd.DataFrame) -> pd.DataFrame:
    """`miss_rebound_log`: `build_miss_rebound_log`'s output (has
    `rebounding_team`, needed here to attribute DREB-by-zone to whichever
    team recovered a miss that wasn't their own). `base_log`:
    `team_stat_rates.build_team_stat_game_log`'s own output (already one
    row per team per game, with `team`/`opponent`/`gameDate`/`season`) --
    reused as the join skeleton so this shares the exact same team-game
    universe/column conventions as the non-zone `oreb_decomposition.py`.

    Adds, per zone: `misses_{zone}_for`/`misses_{zone}_against` (this
    team's own misses in that zone this game / the opponent's, same
    for/against pairing convention `oreb_decomposition.py` already uses --
    a game's away-team misses_for IS its home-team's misses_against,
    looked up via a second merge keyed on `opponent` rather than
    recomputed), `oreb_{zone}_for` (this team's own OREB on its own
    misses), `dreb_{zone}_for` (this team's own DREB on the OPPONENT's
    misses in that zone this game)."""
    log = base_log[["gameId", "team", "opponent", "gameDate", "season"]].copy()
    for zone in ZONES:
        z = miss_rebound_log[miss_rebound_log["zone"] == zone]

        own_misses = z.groupby(["gameId", "shooting_team"], as_index=False).size().rename(
            columns={"shooting_team": "team", "size": "misses"})
        own_oreb = z[z["is_oreb"] == 1.0].groupby(["gameId", "shooting_team"], as_index=False).size().rename(
            columns={"shooting_team": "team", "size": f"oreb_{zone}_for"})
        own_dreb = z[z["is_oreb"] == 0.0].groupby(["gameId", "rebounding_team"], as_index=False).size().rename(
            columns={"rebounding_team": "team", "size": f"dreb_{zone}_for"})

        log = log.merge(own_misses.rename(columns={"misses": f"misses_{zone}_for"}), on=["gameId", "team"], how="left")
        log = log.merge(own_misses.rename(columns={"team": "opponent", "misses": f"misses_{zone}_against"}),
                         on=["gameId", "opponent"], how="left")
        log = log.merge(own_oreb, on=["gameId", "team"], how="left")
        log = log.merge(own_dreb, on=["gameId", "team"], how="left")
        for col in (f"misses_{zone}_for", f"misses_{zone}_against", f"oreb_{zone}_for", f"dreb_{zone}_for"):
            log[col] = log[col].fillna(0.0)
    return log


def add_zone_oreb_decomposition_ratings(log: pd.DataFrame) -> pd.DataFrame:
    """Per zone: walk-forward misses for/against rate (same primitive and
    prior as the flat `oreb_decomposition.py` model, applied separately
    per zone) plus exposure-normalized own-OREB-rate/own-DREB-rate (same
    `add_walk_forward_exposure_rate` primitive, reused unmodified per
    zone)."""
    for zone in ZONES:
        log = add_walk_forward_rate(log, f"misses_{zone}_for", f"misses_{zone}_against",
                                     PRIOR_GAMES_MISSES_ZONE, prefix=f"misses_{zone}")
        log = add_walk_forward_exposure_rate(log, f"oreb_{zone}_for", f"misses_{zone}_for",
                                              PRIOR_MISSES_OREB_RATE_ZONE, prefix=f"oreb_rate_{zone}")
        log = add_walk_forward_exposure_rate(log, f"dreb_{zone}_for", f"misses_{zone}_against",
                                              PRIOR_MISSES_DREB_RATE_ZONE, prefix=f"dreb_rate_{zone}")
    return log


def project_team_oreb_by_zone(row) -> tuple[float, float]:
    """Sums `oreb_decomposition.project_team_oreb`'s exact per-zone
    mechanism (projected_misses_in_zone x projected_oreb_share_in_zone)
    across all 4 zones -- `row` is one wide (home_/away_-suffixed) game
    row with every zone's walk-forward columns already attached (via
    `add_zone_oreb_decomposition_ratings` then a home/away merge, same
    shape `oreb_decomposition.py`'s own validation scripts use)."""
    home_total, away_total = 0.0, 0.0
    for zone in ZONES:
        home_misses, away_misses = project_team_stat(
            getattr(row, f"misses_{zone}_attack_rate_home"), getattr(row, f"misses_{zone}_defense_rate_home"),
            getattr(row, f"misses_{zone}_attack_rate_away"), getattr(row, f"misses_{zone}_defense_rate_away"),
            getattr(row, f"misses_{zone}_league_avg_home"))
        home_share, away_share = project_oreb_share(
            getattr(row, f"oreb_rate_{zone}_shrunk_rate_home"), getattr(row, f"dreb_rate_{zone}_shrunk_rate_home"),
            getattr(row, f"oreb_rate_{zone}_shrunk_rate_away"), getattr(row, f"dreb_rate_{zone}_shrunk_rate_away"),
            getattr(row, f"oreb_rate_{zone}_league_avg_rate_home"), getattr(row, f"dreb_rate_{zone}_league_avg_rate_home"))
        home_total += home_share * home_misses
        away_total += away_share * away_misses
    return home_total, away_total
