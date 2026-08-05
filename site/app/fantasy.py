"""Yahoo-default fantasy point scoring for the Fantasy tab (2026-08-05).

Scores whatever stat lines main.py's _fantasy_leaderboard already assembles
(one dict per player, keyed by the same MARKET_ABBREV short codes used
everywhere else on the site) -- no new data pipeline, no new DB query, just
a scoring function applied to numbers the model already projects.

Weights below are Yahoo Fantasy Baseball's real Head-to-Head Points DEFAULT
league settings, verified 2026-08-05 against two independent live sources
(help.yahoo.com's own default-league-settings page and a Yahoo Sports
points-league-strategy article), not an invented simplification:
  - Batting: 1B=2.6, 2B=5.2, 3B=7.8, HR=10.4, R=1.9, RBI=1.9, BB=2.6,
    SB=4.2, HBP=2.6. No strikeout penalty for batters -- confirmed by both
    sources, so K is NOT scored here (the original implementation's
    `K: -1.0` was wrong and is now removed).
  - Pitching: Outs=1 (3 pts/inning), W=8, SV=8, K=3, ER=-3, BB=-1.3,
    H=-1.3, HBP=-1.3. No Losses or Quality Starts category.

Key simplification kept from the original design: Yahoo's per-base
batting values are exactly 2.6 x [1, 2, 3, 4] for 1B/2B/3B/HR, which is
by construction what "Total Bases" (TB = 1*1B + 2*2B + 3*3B + 4*HR)
already measures. So `2.6 * TB` alone correctly reproduces the full
hit-type scoring in one number -- HR and H (raw hit count) must NOT also
be scored on top of TB, or hit-type value gets double-counted.

Root cause of every pitcher previously showing a NEGATIVE total: Outs/
Innings-Pitched was completely absent from the old PITCHER_POINTS (only
negative categories -- BB/H/R -- were scored), even though it's normally
the single largest positive component of real Yahoo pitcher scoring. Fixed
via the "IP" key below (props.py now exports mean_innings, a real
per-trial simulated innings count -- see props._pitcher_props's
OUTS_PRODUCED comment -- not a pregame estimate).

Root cause of the missing Stolen Bases category: no per-batter SB estimate
existed anywhere in props.py's output. Fixed via "expected_sb" (props.py's
_attach_expected_sb, a display-only approximation from real walk-forward
SB rates -- see that function's docstring for why this doesn't reintroduce
the M5 double-counting bug the simulator itself avoids).

Honest, stated gaps still remaining (surfaced on the page, not hidden):
batters' Runs Scored isn't tracked per batter anywhere in this project's
pipeline (RBI, walks, total-base value, HBP, and now SB all are); pitchers'
Wins, Losses, and Saves aren't tracked (K, walks/hits/HBP allowed, runs
allowed, and now innings pitched all are), and "Runs Allowed" is used here
as an earned-run proxy since earned-vs-unearned isn't modeled. These gaps
are exactly why batters and pitchers get separate boards rather than one
combined rank that would bury real pitchers under batters for
data-completeness reasons, not fantasy value.
"""

BATTER_POINTS = {"TB": 2.6, "RBI": 1.9, "BB": 2.6, "SB": 4.2, "HBP": 2.6}
PITCHER_POINTS = {"IP": 3.0, "K": 3.0, "BB": -1.3, "H": -1.3, "HBP": -1.3, "R": -3.0}  # R used as an ER proxy

BATTER_OMITTED = ["Runs Scored"]
PITCHER_OMITTED = ["Wins", "Losses", "Saves"]

# Columns that are a real number but NOT a directly-simulated per-trial stat
# like every other column here -- flagged separately from OMITTED (2026-08-05
# audit) so the page doesn't imply SB carries the same footing as TB/RBI/BB/
# HBP. props._attach_expected_sb approximates game-level SB opportunities as
# non-HR times on base, but the attempt_rate/success_rate it's multiplied by
# were fit against a STRICTER "open base ahead" opportunity definition
# (baserunning.build_sb_opportunities) -- a real, if modest given the
# shrinkage, unit mismatch worth surfacing rather than presenting SB as if it
# were exactly as trustworthy as the simulated columns next to it.
BATTER_ESTIMATED = ["SB"]
PITCHER_ESTIMATED: list[str] = []

POINTS_BY_ROLE = {"batter": BATTER_POINTS, "pitcher": PITCHER_POINTS}
OMITTED_BY_ROLE = {"batter": BATTER_OMITTED, "pitcher": PITCHER_OMITTED}
ESTIMATED_BY_ROLE = {"batter": BATTER_ESTIMATED, "pitcher": PITCHER_ESTIMATED}


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
