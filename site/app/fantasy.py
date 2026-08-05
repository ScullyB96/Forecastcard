"""Yahoo-default fantasy point scoring for the Fantasy tab (2026-08-05).

Scores whatever stat lines main.py's _leaderboard/_group_props already
assemble (one dict per player, keyed by the same MARKET_ABBREV short codes
used everywhere else on the site: HR/H/TB/RBI/BB/K for batters, K/BB/H/R/BF
for pitchers) -- no new data pipeline, no new DB query, just a scoring
function applied to numbers the model already projects.

Key simplification: Yahoo's default batting point values (1B=1, 2B=2,
3B=3, HR=4) are, by construction, exactly what "Total Bases" (TB = 1*1B +
2*2B + 3*3B + 4*HR) already measures. So TB alone correctly captures the
full hit-type scoring in one number -- HR and H (raw hit count) must NOT
also be scored on top of TB, or hit-type value gets double-counted.

Honest, stated gaps (surfaced on the page, not hidden): batters' Runs
Scored and Stolen Bases aren't tracked per batter anywhere in this
project's pipeline; pitchers' Innings Pitched, Wins, Losses, and Saves
aren't tracked either, and "Runs Allowed" is used here as an earned-run
proxy since earned-vs-unearned isn't modeled. Both gaps mean this is a
real but partial fantasy score, not the full Yahoo default -- which is
exactly why batters and pitchers get separate boards (see the plan this
followed) rather than one combined rank that would bury real pitchers
under batters for data-completeness reasons, not fantasy value.
"""

BATTER_POINTS = {"TB": 1.0, "RBI": 1.0, "BB": 1.0, "K": -1.0}
PITCHER_POINTS = {"K": 1.0, "BB": -1.0, "H": -1.0, "R": -2.0}  # R used as an ER proxy

BATTER_OMITTED = ["Runs Scored", "Stolen Bases"]
PITCHER_OMITTED = ["Innings Pitched", "Wins", "Losses", "Saves"]

POINTS_BY_ROLE = {"batter": BATTER_POINTS, "pitcher": PITCHER_POINTS}
OMITTED_BY_ROLE = {"batter": BATTER_OMITTED, "pitcher": PITCHER_OMITTED}


def fantasy_points(role: str | None, stats: dict) -> float | None:
    """Yahoo-default fantasy points for one player's stat dict (keyed by
    the same short codes as MARKET_ABBREV). None for any role this module
    doesn't have a scoring table for (non-MLB sports have no `role` at
    all today; a future sport/role would just fall through to None here
    rather than a hardcoded sport check) -- the caller skips those."""
    weights = POINTS_BY_ROLE.get(role)
    if weights is None:
        return None
    return sum(weights[k] * v for k, v in stats.items() if k in weights and v is not None)
