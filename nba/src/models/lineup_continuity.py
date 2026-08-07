"""Lineup continuity/chemistry diagnostic (task #62): does using more
FAMILIAR 5-man combinations (lineups that have played together many times
before, this season) predict team performance beyond what individual
player quality already explains -- a "chemistry from reps together" idea
distinct from Sec47's attendance question (which is about WHO plays, not
whether the specific 5-man grouping is a familiar one). Same residual-
first discipline as the other diagnostics this session.

Built from `rapm_lite.prepare_stints`'s existing stints (homePlayers/
awayPlayers tuples per stint) -- no new ingest.
"""

import pandas as pd


def _lineup_key(players) -> tuple:
    return tuple(sorted(players))


def add_lineup_continuity(stints: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """`stints`: one row per stint (has `homePlayers`/`awayPlayers` tuples,
    `gameId`, `gameDate`, `possessions`). `schedule`: needs `gameId`,
    `homeTeamId`, `awayTeamId`, `season`. Returns one row per
    (gameId, team, lineup_key) -- possessions THAT LINEUP played in THAT
    game (summed across every stint using it, since the same 5-man combo
    can recur in multiple non-contiguous stints within one game) and
    `games_together_before`: how many STRICTLY PRIOR games (same team,
    same season) this EXACT lineup has appeared in at all (0 for a
    brand-new combination). Walk-forward safe: collapses to one row per
    (gameId, team, lineup_key) FIRST, then `cumcount()` within
    (team, season, lineup_key) counts only DISTINCT PRIOR GAMES, never a
    same-game double-count."""
    sched = schedule[["gameId", "homeTeamId", "awayTeamId", "season"]]
    s = stints.merge(sched, on="gameId", how="inner")

    rows = []
    for side, players_col, team_col in (("home", "homePlayers", "homeTeamId"), ("away", "awayPlayers", "awayTeamId")):
        d = s[["gameId", "gameDate", "season", team_col, players_col, "possessions"]].copy()
        d = d.rename(columns={team_col: "team", players_col: "players"})
        d["lineup_key"] = d["players"].map(_lineup_key)
        rows.append(d[["gameId", "gameDate", "season", "team", "lineup_key", "possessions"]])
    long = pd.concat(rows, ignore_index=True)

    per_game_lineup = long.groupby(["gameId", "gameDate", "season", "team", "lineup_key"], as_index=False)["possessions"].sum()
    per_game_lineup = per_game_lineup.sort_values(["team", "season", "lineup_key", "gameDate", "gameId"]).reset_index(drop=True)
    per_game_lineup["games_together_before"] = per_game_lineup.groupby(["team", "season", "lineup_key"]).cumcount()
    return per_game_lineup


def team_game_continuity_score(lineup_continuity: pd.DataFrame) -> pd.DataFrame:
    """Collapses `add_lineup_continuity`'s per-(game, team, lineup) rows to
    one row per (gameId, team): `continuity_score` is the POSSESSION-
    WEIGHTED average of `games_together_before` across every lineup used
    that game -- "how much familiar-combination history did tonight's
    rotation, as a whole, carry in." A team debuting an entirely new
    rotation (e.g. after a trade) scores near 0; a team running its usual
    lineups all night scores high."""
    def _weighted_mean(g: pd.DataFrame) -> float:
        total_poss = g["possessions"].sum()
        if total_poss <= 0:
            return float("nan")
        return float((g["games_together_before"] * g["possessions"]).sum() / total_poss)

    result = lineup_continuity.groupby(["gameId", "team"]).apply(_weighted_mean, include_groups=False)
    return result.rename("continuity_score").reset_index()
