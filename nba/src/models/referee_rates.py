"""Referee-crew tendency diagnostic (task #64, MODEL_DOCUMENTATION.md
Sec58): does a game's officiating crew's trailing tendency (how many free
throws/fouls games they've worked have historically produced) correlate
with that game's own scoring residual, beyond what team_strength's
baseline already explains? Same residual-first discipline as every other
diagnostic this session.

Built from `fetch_officials.py`'s new per-game crew data + already-cached
player box scores (fieldGoalsAttempted-style columns already summed
elsewhere in this project) -- no additional ingest beyond `fetch_officials.py`
itself.
"""

import numpy as np
import pandas as pd

PRIOR_GAMES_REFEREE = 100.0  # un-calibrated placeholder, swept before any adoption decision


def build_official_game_log(officials: pd.DataFrame, box: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """`officials`: `fetch_officials.py`'s cached output (gameId, personId,
    name). `box`: player-level traditional box scores (has gameId, teamId,
    freeThrowsAttempted, foulsPersonal). `schedule`: gameId/gameDate/season.

    Returns one row per (gameId, personId): that GAME's own total_fta/
    total_fouls (BOTH teams summed -- a single shared game-level outcome,
    identical across every official who worked it, same as
    `travel_fatigue.py`'s per-game location being shared by both team-sides
    of a game)."""
    game_totals = box.groupby("gameId", as_index=False).agg(
        total_fta=("freeThrowsAttempted", "sum"), total_fouls=("foulsPersonal", "sum"))
    sched = schedule[["gameId", "gameDate", "season"]]

    log = officials[["gameId", "personId"]].merge(game_totals, on="gameId", how="inner")
    log = log.merge(sched, on="gameId", how="inner")
    log["gameDate"] = pd.to_datetime(log["gameDate"])
    return log


def build_official_game_log_differential(officials: pd.DataFrame, box: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Task #69 (external-review follow-up, MODEL_DOCUMENTATION.md Sec59):
    `build_official_game_log`'s total-fouls/FTA outcome tested a PACE/STYLE
    question ("does this crew call a lot of fouls overall") and failed
    holdout. Real referee-bias literature (Price & Wolfers; Scorecasting)
    instead focuses on a BIAS/LEVERAGE question: does a crew tend to call
    FEWER fouls on the home team specifically ("home cooking")? This
    builds that alternative outcome: `home_foul_diff`/`home_fta_diff` =
    (away team's own fouls/FTA) - (home team's own fouls/FTA) for that
    game -- positive means the home team got the benefit (fewer fouls
    called on them, or fewer FTA conceded) relative to the away team.
    Same `add_referee_tendency`/`crew_tendency` primitives apply
    unmodified to this new value_col, since they're already generic."""
    home_totals = box.merge(schedule[["gameId", "homeTeamId"]], on="gameId", how="inner")
    home_totals = home_totals[home_totals["teamId"] == home_totals["homeTeamId"]]
    home_totals = home_totals.groupby("gameId", as_index=False).agg(
        home_fta=("freeThrowsAttempted", "sum"), home_fouls=("foulsPersonal", "sum"))

    away_totals = box.merge(schedule[["gameId", "awayTeamId"]], on="gameId", how="inner")
    away_totals = away_totals[away_totals["teamId"] == away_totals["awayTeamId"]]
    away_totals = away_totals.groupby("gameId", as_index=False).agg(
        away_fta=("freeThrowsAttempted", "sum"), away_fouls=("foulsPersonal", "sum"))

    game_diff = home_totals.merge(away_totals, on="gameId", how="inner")
    game_diff["home_fta_diff"] = game_diff["away_fta"] - game_diff["home_fta"]
    game_diff["home_foul_diff"] = game_diff["away_fouls"] - game_diff["home_fouls"]

    sched = schedule[["gameId", "gameDate", "season"]]
    log = officials[["gameId", "personId"]].merge(game_diff[["gameId", "home_fta_diff", "home_foul_diff"]], on="gameId", how="inner")
    log = log.merge(sched, on="gameId", how="inner")
    log["gameDate"] = pd.to_datetime(log["gameDate"])
    return log


def _trailing_league_mean(log: pd.DataFrame, value_col: str) -> pd.Series:
    """Trailing (strictly prior games only) pooled mean of `value_col`
    across ALL officials -- collapses to one row per gameId FIRST (every
    official working a game shares the SAME value_col), so multiple
    officials' rows for one game can never see each other's "prior" value
    leak in from the same game, mirroring `shrinkage._trailing_league_stat`'s
    identical guard."""
    per_game = log.groupby("gameId", as_index=False).agg(gameDate=("gameDate", "first"), _val=(value_col, "first"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    per_game["_trailing"] = per_game["_val"].shift(1).expanding().mean()
    return log.merge(per_game[["gameId", "_trailing"]], on="gameId", how="left")["_trailing"]


def add_referee_tendency(log: pd.DataFrame, value_col: str, prior_games: float = PRIOR_GAMES_REFEREE) -> pd.DataFrame:
    """Adds `{value_col}_league_avg` and `{value_col}_official_tendency`:
    a walk-forward-shrunk trailing mean of `value_col` for games THIS
    OFFICIAL worked, blended toward the trailing league-wide mean at
    `prior_games` strength. Deliberately NOT season-reset (an official's
    calling style is a persistent trait across seasons, not reset by
    roster turnover the way team quality is -- same design choice as
    `home_court.fit_team_home_court_walk_forward`). No same-game leak risk
    for the official's OWN history (each officiated game is exactly one
    row per official, unlike the pooled league mean above)."""
    log = log.sort_values(["personId", "gameDate", "gameId"]).reset_index(drop=True)
    league_avg_col = f"{value_col}_league_avg"
    log[league_avg_col] = _trailing_league_mean(log, value_col)

    grp = log.groupby("personId")
    games_before = grp.cumcount()
    cumsum_before = grp[value_col].cumsum() - log[value_col]
    own_mean = cumsum_before / games_before.replace(0, np.nan)

    shrunk = (games_before * own_mean.fillna(0.0) + prior_games * log[league_avg_col]) / (games_before + prior_games)
    log[f"{value_col}_official_tendency"] = shrunk
    return log


def crew_tendency(log_with_tendency: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapses per-(gameId, personId) tendency rows to one row per
    gameId: the plain average of that game's 3 officials' own trailing
    tendencies (no per-official disentangling attempted -- a simple,
    honest crew-level combine, consistent with this project's "start with
    the simplest defensible mechanism" discipline)."""
    col = f"{value_col}_official_tendency"
    return log_with_tendency.groupby("gameId", as_index=False)[col].mean().rename(
        columns={col: f"crew_{value_col}_tendency"})
