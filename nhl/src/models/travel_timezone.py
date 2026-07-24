"""Cycle 15: back-to-back x time-zone-change interaction (MODEL_DOCUMENTATION.md
Sec9), per the external research report (Sec5.3): RotoWire's own 11-season
study found raw time-zone crossings and travel direction do NOT clear
significance on their own -- only the back-to-back x time-zone-change
INTERACTION does. Checked for a real effect on this project's own data
before building anything, per the established discipline.

Team home time zones are real, stable facts (arena location), confirmed
against the actual 35 distinct team abbreviations present in this project's
own regular-season schedule data (including historical ATL/PHX and modern
UTA/SEA) rather than assumed from a generic list.
"""

import pandas as pd

TEAM_TIMEZONE = {
    # Eastern
    "BOS": "ET", "BUF": "ET", "CAR": "ET", "CBJ": "ET", "DET": "ET", "FLA": "ET",
    "MTL": "ET", "NJD": "ET", "NYI": "ET", "NYR": "ET", "OTT": "ET", "PHI": "ET",
    "PIT": "ET", "TBL": "ET", "TOR": "ET", "WSH": "ET", "ATL": "ET",
    # Central
    "CHI": "CT", "DAL": "CT", "MIN": "CT", "NSH": "CT", "STL": "CT", "WPG": "CT",
    # Mountain
    "ARI": "MT", "PHX": "MT", "UTA": "MT", "COL": "MT", "CGY": "MT", "EDM": "MT",
    # Pacific
    "ANA": "PT", "LAK": "PT", "SJS": "PT", "SEA": "PT", "VAN": "PT", "VGK": "PT",
}


def add_timezone_change(team_game_log: pd.DataFrame) -> pd.DataFrame:
    """log: one row per team per game (baseline_naive_poisson.build_team_game_log
    convention: gameId, gameDate, season, team, opponent, is_home). Adds
    `venue_timezone` (this game's, i.e. the HOME team's, timezone) and, per
    team walk-forward, `timezone_changed` (bool: did this team's own
    previous game's venue sit in a different time zone than this game's)."""
    log = team_game_log.copy()
    log["gameDate_dt"] = pd.to_datetime(log["gameDate"])
    log = log.sort_values(["team", "season", "gameDate_dt"]).reset_index(drop=True)

    home_team_of_game = log["team"].where(log["is_home"], log["opponent"])
    log["venue_timezone"] = home_team_of_game.map(TEAM_TIMEZONE)

    prev_tz = log.groupby(["team", "season"])["venue_timezone"].shift(1)
    log["timezone_changed"] = (log["venue_timezone"] != prev_tz) & prev_tz.notna()
    return log
