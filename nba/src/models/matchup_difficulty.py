"""Defender-difficulty rate model: how much a given defender/team suppresses
opposing scoring relative to league average, feeding `usage_allocation.py`'s
per-player scoring-share reallocation (the "micro-reallocation" half of the
macro-anchor + micro-reallocation composition rule -- RAPM-lite still sets
the team scoring total; this shifts how that total is split, and slightly
shades a player's own projected efficiency, based on tonight's specific
defensive matchup).

Restricted to `MATCHUP_DATA_START_SEASON=2017` onward (`BoxScoreMatchupsV3`/
`BoxScoreDefensiveV2` coverage boundary, confirmed live -- see
`fetch_boxscore_matchups.py`). The six per-player rate models validate on
the full 2015-2025 range; only this matchup layer is restricted.

THREE-LEVEL SHRINKAGE HIERARCHY, most-specific-available wins (mirrors
`player_priors.lambda_for_experience`'s "more granularity earns more weight
only once it has enough exposure" idea):
1. Defender-specific rate: this defender's own walk-forward difficulty,
   shrunk toward the league average as matchup-minute exposure accumulates
   -- same `add_walk_forward_player_rate` primitive as every other
   per-player rate model in this codebase. Keyed by `playerId` ALONE, not
   `(playerId, team)` -- a defender's skill is portable across a trade; a
   live caller joins TONIGHT's active roster to a defender's OWN historical
   rate, never resets it because they changed teams.
2. Position-group-vs-opponent-team rate: from `BoxScoreMatchupsV3`,
   collapsed by the OFFENSIVE player's position group using
   `CommonTeamRoster`'s per-season `POSITION` field -- NOT
   `BoxScoreMatchupsV3`'s own `positionOff` column, which is confirmed
   BLANK for fully HALF of all matchup rows on real 2022-23 data (49.99%),
   exactly the kind of unreliable box-score position field
   `build_stints.py` already found and avoided for starters detection.
   Captures team defensive SCHEME against a position group (e.g. "drops
   bigs off the perimeter"), available even when a specific defender
   hasn't guarded this specific player much.
3. League-average difficulty at that position group: a fixed floor,
   always available, no shrinkage needed (it's already a broad average).

"Difficulty" is measured as points allowed per matchup-minute (lower =
tougher defender/scheme); relative difficulty for the adjustment is
`league_avg_rate - this_rate` (positive = tougher than average).

CONFIRMED EMPIRICALLY (full 2017-2023 dev subset, not assumed by analogy --
this project's standing discipline): the two levels need OPPOSITE smoothing
families, a now-familiar pattern (minutes vs. attempts, 2PT/3PT vs. FT,
steals vs. blocks):
- Defender-specific difficulty (level 1) is a persistent INDIVIDUAL skill --
  expanding-shrinkage clearly wins (best MAE 3.978 at prior_matchup_
  minutes=50, vs. naive's 4.056 and every EWMA halflife tested, worst-case
  4.29 at halflife=3).
- Position-group-vs-team difficulty (level 2) is a volatile TEAM-SCHEME
  metric -- personnel changes, coaching adjustments, and trades shift how a
  team defends a position group far faster than one player's individual
  defensive skill changes. EWMA halflife=10 games (MAE 8.181) beat every
  expanding-shrinkage prior tested (best 8.227 at prior=75, still worse
  than naive's own 8.288 -- expanding-shrinkage barely helps at all here)
  and every other EWMA halflife tried.
"""

import numpy as np
import pandas as pd

from src.ingest.fetch_schedule import season_str
from src.models.player_rate_shrinkage import add_walk_forward_player_mean_ewm, add_walk_forward_player_rate
from src.utils.paths import DATA_RAW

MATCHUP_DATA_START_SEASON = 2017  # frozen -- see module docstring; must match fetch_boxscore_matchups.py

# Defender-specific difficulty: expanding-shrinkage, empirically-swept optimum (see module docstring).
PRIOR_MATCHUP_MINUTES_DIFFICULTY = 50.0

# Position-group-vs-team difficulty: EWMA (opposite family from level 1, see module docstring),
# empirically-swept optimum halflife.
POSGROUP_HALFLIFE_GAMES = 10.0


def _parse_minutes(minutes_str) -> float:
    """"MM:SS" (no leading zero on minutes, e.g. "7:59") -> float minutes. Blank -> 0.0."""
    if minutes_str is None or minutes_str == "":
        return 0.0
    m, s = str(minutes_str).split(":")
    return int(m) + int(s) / 60.0


def position_group(position_str) -> str:
    """`CommonTeamRoster`'s `POSITION` field ("G", "F", "C", "F-C", "G-F", ...)
    collapsed to a single-letter group by primary (first-listed) position.
    Blank/missing -> "" (caller should drop or floor-fallback these, never
    silently bucket into a real group)."""
    if position_str is None or position_str == "":
        return ""
    return str(position_str)[0]


def _load_roster_position_lookup(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per (season, playerId) -> position_group, from
    `CommonTeamRoster` (NOT any box-score's own position column -- see
    module docstring). A player who was traded mid-season and appears on
    two rosters that year keeps both rows (harmless: position rarely
    differs across a same-season trade, and the join below is on
    (season, playerId) only, first-match-wins)."""
    frames = []
    for y in range(start_year, end_year + 1):
        path = DATA_RAW / f"team_rosters_{season_str(y)}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        frames.append(pd.DataFrame({
            "season": y, "playerId": df["PLAYER_ID"], "position_group": df["POSITION"].apply(position_group),
        }))
    if not frames:
        return pd.DataFrame(columns=["season", "playerId", "position_group"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["season", "playerId"], keep="first")


def build_defender_game_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per (defender, game) across [max(start_year,
    MATCHUP_DATA_START_SEASON), end_year], from `BoxScoreDefensiveV2`
    (already team/opponent-aggregated per defender -- no pairwise
    reconstruction needed at this level). Requires that season's schedule
    (for `gameDate`) and cached defensive box score -- seasons missing
    either are skipped silently, same convention as
    `player_minutes.build_player_game_log`."""
    frames = []
    for y in range(max(start_year, MATCHUP_DATA_START_SEASON), end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_defensive_v2_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule = pd.read_parquet(sched_path)
        schedule = schedule[schedule["seasonType"] == "Regular Season"]
        box = pd.read_parquet(box_path)

        merged = schedule.merge(box, on="gameId", how="inner")
        # schedule has one row per (gameId, homeTeamId/awayTeamId); box's teamId is the DEFENDER's
        # team, which is either side -- merging on gameId alone (not team) then reading gameDate
        # off the schedule row is correct since gameDate doesn't depend on which side matched.
        merged = merged.drop_duplicates(subset=["gameId", "teamId", "personId"])
        frames.append(pd.DataFrame({
            "gameId": merged["gameId"], "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": y, "playerId": merged["personId"], "team": merged["teamId"],
            "matchup_minutes": merged["matchupMinutes"].apply(_parse_minutes),
            "points_allowed": merged["playerPoints"],
            "partial_possessions": merged["partialPossessions"],
        }))
    if not frames:
        return pd.DataFrame(columns=["gameId", "gameDate", "season", "playerId", "team",
                                      "matchup_minutes", "points_allowed", "partial_possessions"])
    return pd.concat(frames, ignore_index=True)


def add_defender_difficulty_rate(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `build_defender_game_log`'s columns. Adds
    `difficulty_rate` (shrunk points-allowed per matchup-minute; LOWER is a
    tougher defender) and `difficulty_proj` (this row's own realized
    matchup-minutes x rate -- a backtest convenience, see the pattern in
    every other `add_*_rate` function in this codebase)."""
    log = log.copy()
    log = add_walk_forward_player_rate(log, "points_allowed", "matchup_minutes",
                                        PRIOR_MATCHUP_MINUTES_DIFFICULTY, prefix="difficulty")
    log = log.rename(columns={"difficulty_shrunk_rate": "difficulty_rate"})
    log["difficulty_proj"] = log["difficulty_rate"] * log["matchup_minutes"]
    return log


def build_position_group_matchup_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per (defending team, offensive position group, game) across
    [max(start_year, MATCHUP_DATA_START_SEASON), end_year], from
    `BoxScoreMatchupsV3` collapsed by the OFFENSIVE player's roster
    position group (see module docstring for why NOT `positionOff`).
    Captures team-level defensive scheme against a position group, the
    level-2 fallback when a specific defender lacks enough matchup-minute
    history against tonight's opponent."""
    roster_lookup = _load_roster_position_lookup(start_year, end_year)

    frames = []
    for y in range(max(start_year, MATCHUP_DATA_START_SEASON), end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_matchups_v3_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule = pd.read_parquet(sched_path)
        schedule = schedule[schedule["seasonType"] == "Regular Season"]
        box = pd.read_parquet(box_path)

        box = box.merge(roster_lookup[roster_lookup["season"] == y], left_on="personIdOff", right_on="playerId", how="left")
        box = box[box["position_group"].notna() & (box["position_group"] != "")]
        box["matchup_minutes"] = box["matchupMinutes"].apply(_parse_minutes)

        grouped = box.groupby(["gameId", "teamId", "position_group"], as_index=False).agg(
            matchup_minutes=("matchup_minutes", "sum"), points_allowed=("playerPoints", "sum"))
        merged = schedule.merge(grouped, on="gameId", how="inner")
        frames.append(pd.DataFrame({
            "gameId": merged["gameId"], "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": y, "team": merged["teamId"], "position_group": merged["position_group"],
            "matchup_minutes": merged["matchup_minutes"], "points_allowed": merged["points_allowed"],
        }))
    if not frames:
        return pd.DataFrame(columns=["gameId", "gameDate", "season", "team", "position_group",
                                      "matchup_minutes", "points_allowed"])
    return pd.concat(frames, ignore_index=True)


def add_position_group_difficulty_rate(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `build_position_group_matchup_log`'s columns. Groups by
    (team, position_group) rather than a single `playerId` -- the shared
    `add_walk_forward_player_mean_ewm` primitive is generic over `group_col`,
    so this reuses it with a composite string key rather than reimplementing
    a team-level walk-forward rate from scratch. Adds `posgroup_difficulty_rate`
    (EWMA-smoothed points-allowed per matchup-minute for that team against
    that position group -- EWMA, not expanding-shrinkage, confirmed
    empirically, see module docstring)."""
    log = log.copy()
    log["team_posgroup"] = log["team"].astype(str) + "_" + log["position_group"]
    log["_pts_per_min_raw"] = log["points_allowed"] / log["matchup_minutes"].replace(0, np.nan)
    log = add_walk_forward_player_mean_ewm(log, "_pts_per_min_raw", POSGROUP_HALFLIFE_GAMES,
                                            prefix="posgroup", group_col="team_posgroup")
    log = log.rename(columns={"posgroup_ewm_rate": "posgroup_difficulty_rate"})
    return log.drop(columns=["team_posgroup", "_pts_per_min_raw"])


# "Tonight's merge" -- handles switch-heavy defenses (never a hard 1:1 defender assignment).
# Not yet empirically tuned (pending real-slate testing once `generate_props.py` exists, per the
# approved plan's own Verification section) -- a placeholder floor, not a guessed prediction input.
MIN_MATCHUP_MINUTES_TO_TRUST = 20.0


def build_defender_position_group_minutes(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per (defender, offensive position group, game) across
    [max(start_year, MATCHUP_DATA_START_SEASON), end_year], from
    `BoxScoreMatchupsV3` -- the per-DEFENDER (not team-collapsed) analog of
    `build_position_group_matchup_log`, needed to compute each defender's
    SHARE of their team's total matchup-minutes against a position group
    (the w(p,d) weight in the merge below -- a switch-heavy defense splits
    time guarding a position group across several players, so the
    adjustment must be a weighted blend, never a single hard-assigned
    defender)."""
    roster_lookup = _load_roster_position_lookup(start_year, end_year)

    frames = []
    for y in range(max(start_year, MATCHUP_DATA_START_SEASON), end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_matchups_v3_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule = pd.read_parquet(sched_path)
        schedule = schedule[schedule["seasonType"] == "Regular Season"]
        box = pd.read_parquet(box_path)

        box = box.merge(roster_lookup[roster_lookup["season"] == y], left_on="personIdOff", right_on="playerId", how="left")
        box = box[box["position_group"].notna() & (box["position_group"] != "")]
        box["matchup_minutes"] = box["matchupMinutes"].apply(_parse_minutes)

        grouped = box.groupby(["gameId", "personIdDef", "teamId", "position_group"], as_index=False).agg(
            matchup_minutes=("matchup_minutes", "sum"))
        merged = schedule.merge(grouped, on="gameId", how="inner")
        frames.append(pd.DataFrame({
            "gameId": merged["gameId"], "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": y, "playerId": merged["personIdDef"], "team": merged["teamId"],
            "position_group": merged["position_group"], "matchup_minutes": merged["matchup_minutes"],
        }))
    if not frames:
        return pd.DataFrame(columns=["gameId", "gameDate", "season", "playerId", "team",
                                      "position_group", "matchup_minutes"])
    return pd.concat(frames, ignore_index=True)


def defender_position_group_minutes_asof(defender_posgroup_log: pd.DataFrame, position_group_value: str,
                                          as_of_date) -> pd.Series:
    """Trailing (strictly before `as_of_date`) total matchup-minutes each
    defender has logged against `position_group_value`, from
    `build_defender_position_group_minutes`'s output. Returns a Series
    indexed by `playerId` -- the raw exposure a live caller turns into
    shares once it knows tonight's actual opposing roster (this function
    doesn't need the roster; it just answers "how much recent history does
    each defender who's EVER appeared have")."""
    subset = defender_posgroup_log[
        (defender_posgroup_log["position_group"] == position_group_value)
        & (defender_posgroup_log["gameDate"] < pd.Timestamp(as_of_date))
    ]
    return subset.groupby("playerId")["matchup_minutes"].sum()


def opponent_defense_adjustment(opposing_roster_player_ids: list,
                                 defender_ratings: pd.Series, defender_minutes_at_posgroup: pd.Series,
                                 posgroup_rating: float | None, league_avg_posgroup_rating: float,
                                 min_matchup_minutes_to_trust: float = MIN_MATCHUP_MINUTES_TO_TRUST) -> float:
    """The 3-level hierarchy's actual combination, for ONE offensive
    player's position group against ONE opposing roster tonight --
    `defender_minutes_at_posgroup` must already be pre-filtered to that
    position group (e.g. via `defender_position_group_minutes_asof`); this
    function doesn't re-filter, it only weights and combines. Positive
    return = tougher-than-average matchup (suppress the offensive player's
    projection); negative = easier-than-average (boost it); designed to be
    ADDED to `league_avg_posgroup_rating - <player's own projected rate>`
    upstream in `usage_allocation.py`, never applied twice.

    - `defender_ratings`: Series indexed by playerId, level-1 `difficulty_rate`
      (already walk-forward shrunk) for every defender league-wide.
    - `defender_minutes_at_posgroup`: Series indexed by playerId, this
      specific position group's trailing matchup-minutes per defender (from
      `defender_position_group_minutes_asof`) -- used ONLY to weight w(p,d)
      across `opposing_roster_player_ids`, not as a trust gate by itself.
    - `posgroup_rating`/`league_avg_posgroup_rating`: level 2 and level 3,
      always available as the fallback when the roster's total matchup-
      minute exposure at this position group doesn't clear
      `min_matchup_minutes_to_trust`.

    Falls back whole-cloth to level 2 (or level 3, if level 2 is also
    unavailable) rather than blending partial defender coverage with the
    team rating -- avoids double-counting the same signal at two levels at
    once, matching the macro-anchor + micro-reallocation "never both move
    the same number" discipline used for the RAPM/usage-share split."""
    roster_minutes = {pid: float(defender_minutes_at_posgroup.get(pid, 0.0)) for pid in opposing_roster_player_ids}
    total_minutes = sum(roster_minutes.values())

    if total_minutes < min_matchup_minutes_to_trust:
        if posgroup_rating is not None:
            return league_avg_posgroup_rating - posgroup_rating
        return 0.0

    adjustment = 0.0
    for pid, minutes in roster_minutes.items():
        if minutes <= 0:
            continue
        weight = minutes / total_minutes
        defender_rate = defender_ratings.get(pid)
        if defender_rate is None:
            defender_rate = posgroup_rating if posgroup_rating is not None else league_avg_posgroup_rating
        adjustment += weight * (league_avg_posgroup_rating - defender_rate)
    return adjustment
