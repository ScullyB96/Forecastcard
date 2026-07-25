"""Team ID/abbreviation lookup.

Confirmed live (2026-07-24): nba_api ships a static list of the 30 current
franchises (`nba_api.stats.static.teams`) with no HTTP call required, and
TEAM_ID has been stable across every endpoint used by this project (box
score, schedule, rotation) for the entire 2015-16..present dev/holdout range
-- there have been zero relocations/rebrands in that window (unlike NHL's
project, which had to handle real relocations). This module is therefore a
thin, static lookup, not a fetched-and-cached one.
"""

from nba_api.stats.static import teams as _static_teams

TEAM_ID_TO_ABBREV: dict[int, str] = {t["id"]: t["abbreviation"] for t in _static_teams.get_teams()}
ABBREV_TO_TEAM_ID: dict[str, int] = {v: k for k, v in TEAM_ID_TO_ABBREV.items()}
TEAM_ID_TO_NAME: dict[int, str] = {t["id"]: t["full_name"] for t in _static_teams.get_teams()}

assert len(TEAM_ID_TO_ABBREV) == 30, f"expected 30 current NBA franchises, got {len(TEAM_ID_TO_ABBREV)}"
