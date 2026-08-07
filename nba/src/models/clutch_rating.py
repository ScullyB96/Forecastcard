"""Clutch-time performance diagnostic (task #61): is there a persistent
team-level "clutch skill" -- performing above/below one's own overall
level specifically in close-and-late situations -- that isn't already
captured by a team's overall walk-forward rating? A well-known question
in sports analytics generally found to be mostly noise/regression-to-the-
mean; tested here empirically rather than assumed either way, same
residual-first discipline as `rest_schedule.py`.

Reuses `rapm_lite.prepare_stints`'s existing `marginBeforeStint` (running
score margin before a stint) and `garbage_time.py`'s per-game elapsed-time
convention -- no new ingest needed, stints already have everything.
"""

import numpy as np
import pandas as pd

CLUTCH_MARGIN_THRESHOLD = 5.0  # points -- standard "close" cutoff, matches common clutch-time defs
CLUTCH_TIME_REMAINING_TENTHS = 3000.0  # last 5 game-minutes (300s x 10) of regulation/OT


def identify_clutch_stints(stints_with_margin: pd.DataFrame, game_length_tenths: float = 28800.0) -> pd.DataFrame:
    """`stints_with_margin`: `rapm_lite.prepare_stints`'s output (has
    `startTenths`/`endTenths`/`marginBeforeStint`). Adds `is_clutch`: True
    if the stint STARTS within the trailing `CLUTCH_TIME_REMAINING_TENTHS`
    of that game's OWN total length (per-game, so OT games are handled
    correctly -- same per-game-max convention `garbage_time.add_garbage_time_weight`
    already uses) AND the score margin before the stint is within
    `CLUTCH_MARGIN_THRESHOLD` points."""
    s = stints_with_margin.copy()
    per_game_max = s.groupby("gameId")["endTenths"].transform("max")
    total_length = per_game_max.clip(lower=game_length_tenths)
    time_remaining = total_length - s["startTenths"]
    s["is_clutch"] = (time_remaining <= CLUTCH_TIME_REMAINING_TENTHS) & (s["marginBeforeStint"].abs() <= CLUTCH_MARGIN_THRESHOLD)
    return s


def team_game_clutch_net_rating(stints_with_clutch_flag: pd.DataFrame) -> pd.DataFrame:
    """One row per (gameId, team_side in {'home','away'}): `clutch_net_rtg`
    (points per 100 possessions during CLUTCH stints only) and `clutch_poss`
    (how many clutch possessions this game had, for filtering thin
    samples). NaN clutch_net_rtg for a game with zero clutch possessions
    (most blowouts never reach a close-and-late situation at all) -- never
    fabricated as 0."""
    clutch = stints_with_clutch_flag[stints_with_clutch_flag["is_clutch"]]
    rows = []
    for side, pts_col in (("home", "homePts"), ("away", "awayPts")):
        g = clutch.groupby("gameId", as_index=False).agg(pts=(pts_col, "sum"), poss=("possessions", "sum"))
        g["team_side"] = side
        g["clutch_net_rtg"] = np.where(g["poss"] > 0, 100.0 * g["pts"] / g["poss"], np.nan)
        rows.append(g[["gameId", "team_side", "clutch_net_rtg", "poss"]].rename(columns={"poss": "clutch_poss"}))
    return pd.concat(rows, ignore_index=True)


def add_trailing_clutch_deviation(log: pd.DataFrame, min_prior_clutch_games: int = 10) -> pd.DataFrame:
    """`log`: one row per team per game, sorted by (team, season, gameDate),
    with `clutch_net_rtg` (NaN if that game had no clutch possessions) and
    `overall_net_rtg` (this team's own realized net rating that game, the
    "usual level" baseline). Adds `trailing_clutch_deviation`: this team's
    own trailing average (clutch_net_rtg - overall_net_rtg) across its
    STRICTLY PRIOR games that actually had clutch possessions (skip-NaN
    walk-forward mean, within season) -- NaN until at least
    `min_prior_clutch_games` such games have accumulated, to avoid an
    unstable 1-2-game estimate."""
    log = log.sort_values(["team", "season", "gameDate"]).reset_index(drop=True)
    dev = log["clutch_net_rtg"] - log["overall_net_rtg"]
    grp = dev.groupby([log["team"], log["season"]])

    cum_sum = grp.transform(lambda s: s.fillna(0.0).cumsum().shift(1))
    cum_count = grp.transform(lambda s: s.notna().cumsum().shift(1))
    trailing_mean = cum_sum / cum_count
    log["trailing_clutch_deviation"] = trailing_mean.where(cum_count >= min_prior_clutch_games)
    return log
