"""The foundational player-props model: walk-forward shrunk minutes per
player, plus `build_player_game_log`, the SHARED player-level game-log
builder every other rate-category model (`player_scoring_rates.py`,
`player_rebounding_rates.py`, `player_playmaking_rates.py`,
`player_defensive_event_rates.py`) imports from -- mirrors how
`team_strength.build_team_game_log` is the one shared team-level builder
Phase 1's files all reuse, rather than each file re-merging box scores
against the schedule independently.

Built directly from `BoxScoreTraditionalV3`'s cached rows (already
confirmed to carry `personId`/`teamId`/`position`/`comment`/`minutes`
["MM:SS" string]/`fieldGoalsMade`.../`points` for every game). This is
deliberately NOT a separate "DNP vs. absent" special case: a player who
wasn't with the team that night (assigned to the G League, not on the
active roster) simply never generates a row in that game's box score at
all, so building the log purely from real box-score rows already excludes
them by construction -- no extra filtering needed. A real DNP-coach's-
decision player DOES have a row (0 real minutes, `comment` set,
`minutes==""`), which is exactly the correct 0-minute signal to keep.
"""

import pandas as pd

from src.ingest.fetch_schedule import season_str
from src.models.player_rate_shrinkage import add_walk_forward_player_mean_ewm
from src.utils.paths import DATA_RAW

MINUTES_HALFLIFE_GAMES = 2.0  # confirmed empirically (not assumed), see add_walk_forward_player_mean_ewm's
# docstring -- an infinite-memory expanding shrinkage was a REAL REGRESSION vs. a short-window recency
# average; halflife=2 games gave the best real MAE of every option tested.


def _parse_minutes(minutes_str) -> float:
    """"MM:SS" (no leading zero on minutes, e.g. "7:59") -> float minutes.
    Blank string (a real DNP row) -> 0.0."""
    if minutes_str is None or minutes_str == "":
        return 0.0
    m, s = str(minutes_str).split(":")
    return int(m) + int(s) / 60.0


def build_player_game_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per player per game across [start_year, end_year], regular
    season only (matches Phase 1's train-on-regular-season-only
    convention). Requires that season's schedule + traditional box score to
    already be cached -- seasons missing either are skipped silently (the
    caller decides how much history it actually needs; Phase 0's
    `validate_data_coverage.py` is the place to check for real gaps)."""
    frames = []
    for y in range(start_year, end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_trad_player_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule = pd.read_parquet(sched_path)
        schedule = schedule[schedule["seasonType"] == "Regular Season"]
        box = pd.read_parquet(box_path)

        for side, team_col, opp_col in (("home", "homeTeamId", "awayTeamId"), ("away", "awayTeamId", "homeTeamId")):
            merged = schedule.merge(box, left_on=["gameId", team_col], right_on=["gameId", "teamId"], how="inner")
            frames.append(pd.DataFrame({
                "gameId": merged["gameId"], "gameDate": pd.to_datetime(merged["gameDate"]),
                "season": merged["season"], "playerId": merged["personId"],
                "team": merged[team_col], "opponent": merged[opp_col], "is_home": side == "home",
                "minutes": merged["minutes"].apply(_parse_minutes),
                "fgMade2": merged["fieldGoalsMade"] - merged["threePointersMade"],
                "fgAtt2": merged["fieldGoalsAttempted"] - merged["threePointersAttempted"],
                "fg3Made": merged["threePointersMade"], "fg3Att": merged["threePointersAttempted"],
                "ftMade": merged["freeThrowsMade"], "ftAtt": merged["freeThrowsAttempted"],
                "oreb": merged["reboundsOffensive"], "dreb": merged["reboundsDefensive"],
                "assists": merged["assists"], "turnovers": merged["turnovers"],
                "steals": merged["steals"], "blocks": merged["blocks"], "points": merged["points"],
            }))

    if not frames:
        return pd.DataFrame()
    log = pd.concat(frames, ignore_index=True)
    return log.sort_values(["gameDate", "gameId", "playerId"]).reset_index(drop=True)


def attach_player_track(log: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Merges `BoxScorePlayerTrackV3`'s `touches`/`reboundChancesOffensive`/
    `reboundChancesDefensive` onto an existing player-game log (from
    `build_player_game_log`), keyed on `(gameId, playerId)`. Confirmed live
    2026-07-25: this endpoint has NO coverage-boundary issue (full range,
    unlike `BoxScoreMatchupsV3`/`BoxScoreDefensiveV2` -- see
    `fetch_boxscore_matchups.py`), but the backfill still takes real time
    to complete; games not yet fetched simply get NaN for these three
    columns rather than being dropped, so callers should `dropna` on
    whichever of these three columns they actually need before using them."""
    frames = []
    for y in range(start_year, end_year + 1):
        path = DATA_RAW / f"player_track_{season_str(y)}.parquet"
        if not path.exists():
            continue
        pt = pd.read_parquet(path, columns=["gameId", "personId", "touches",
                                             "reboundChancesOffensive", "reboundChancesDefensive"])
        frames.append(pt)
    if not frames:
        for col in ("touches", "reboundChancesOffensive", "reboundChancesDefensive"):
            log[col] = float("nan")
        return log

    pt_all = pd.concat(frames, ignore_index=True).rename(columns={"personId": "playerId"})
    return log.merge(pt_all, on=["gameId", "playerId"], how="left")


def add_minutes_rating(log: pd.DataFrame) -> pd.DataFrame:
    """Adds `minutes_ewm_rate`: this player's own recency-weighted
    (exponentially-decayed, halflife=`MINUTES_HALFLIFE_GAMES`) walk-forward
    minutes projection. See `player_rate_shrinkage.add_walk_forward_player_mean_ewm`'s
    docstring for why this recency-weighted approach replaced an initial
    infinite-memory-shrinkage attempt that was confirmed, via a real
    bootstrap run, to be a REAL REGRESSION vs. even a naive floor."""
    return add_walk_forward_player_mean_ewm(log, "minutes", MINUTES_HALFLIFE_GAMES, prefix="minutes")


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season

    current = current_nba_season()
    log = build_player_game_log(FIRST_DEV_SEASON, current)
    print(f"player-game log: {len(log)} rows ({log['playerId'].nunique()} distinct players, "
          f"{log['gameId'].nunique()} games)", flush=True)
    rated = add_minutes_rating(log)
    sample = rated.dropna(subset=["minutes_ewm_rate"]).tail(10)
    print(sample[["gameId", "playerId", "minutes", "minutes_ewm_rate"]].to_string(), flush=True)
