"""Cycle 2: swap the naive baseline's raw-goals team-strength input for
MoneyPuck's whole-game xG (situation=="all"), keeping every other piece of
the model identical (same shrinkage/walk-forward machinery via shrinkage.py,
same Poisson combine, same validation harness) -- isolating exactly one
variable (xG vs. real goals as the input to team strength) per
MODEL_DOCUMENTATION.md Sec3.9's closing note and the project's stated
"don't stack multiple modeling ideas before checking each one" discipline.

Deliberately NOT yet doing in this cycle (left for a later, separately-
tested cycle, per Sec3.2): splitting by strength state (5v5/PP/PK), a
goalie overlay, or rest/schedule adjustments. Using situation=="all" (whole-
game xG, all strength states combined) is the closest apples-to-apples
comparison to the raw-goals baseline's whole-game goal counts.

Confirmed directly (2026-07-23): MoneyPuck's `gameId` uses the EXACT same
numeric scheme as the NHL API's own game id (verified on a real sample game,
non-obviously convenient -- both are apparently derived from the same
official schedule feed), so team-game rows join on (gameId, normalized team
code) directly -- no date-based fuzzy matching needed.
"""

import pandas as pd

from src.ingest.team_codes import normalize_moneypuck_team_code, normalize_nhl_api_team_code
from src.models.baseline_naive_poisson import build_team_game_log
from src.models.shrinkage import add_walk_forward_rate

PRIOR_GAMES_XG = 10  # same placeholder as the raw-goals baseline -- not yet separately calibrated


def build_team_game_xg_log(schedule: pd.DataFrame, moneypuck: pd.DataFrame) -> pd.DataFrame:
    """One row per team per real regular-season game: the actual-goals
    columns (goals_for/goals_against, the target every model is ultimately
    validated against) from the NHL API schedule, plus xg_for/xg_against
    (the new predictive input) from MoneyPuck's whole-game (situation=="all")
    row for that same team-game."""
    goals_log = build_team_game_log(schedule)
    # NHL API's own abbrev needs the same PHX->ARI normalization as MoneyPuck's
    # side -- confirmed real (2026-07-23): omitting this silently dropped
    # every Phoenix Coyotes game (2008-09 through 2013-14, 465 team-game rows)
    # from the join, since the schedule side still said "PHX" while
    # MoneyPuck's side (already normalized below) says "ARI".
    goals_log["season_start"] = goals_log["season"].astype(str).str[:4].astype(int)
    goals_log["team"] = [normalize_nhl_api_team_code(t, s) for t, s in zip(goals_log["team"], goals_log["season_start"])]
    goals_log = goals_log.drop(columns=["season_start"])

    mp_all = moneypuck[moneypuck["situation"] == "all"].copy()
    mp_all["team_norm"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(mp_all["team"], mp_all["season"])]
    mp_all = mp_all[["gameId", "team_norm", "xGoalsFor", "xGoalsAgainst"]].rename(
        columns={"team_norm": "team", "xGoalsFor": "xg_for", "xGoalsAgainst": "xg_against"})

    merged = goals_log.merge(mp_all, on=["gameId", "team"], how="inner")
    dropped = len(goals_log) - len(merged)
    if dropped:
        # Real, not assumed: report exactly how many team-game rows didn't
        # find an xG match rather than silently proceeding on a smaller set.
        print(f"team_strength_xg: {dropped}/{len(goals_log)} team-game rows had no MoneyPuck xG match, dropped")
    return merged


def add_walk_forward_xg_strength(log: pd.DataFrame, prior_games: float = PRIOR_GAMES_XG) -> pd.DataFrame:
    return add_walk_forward_rate(log, for_col="xg_for", against_col="xg_against",
                                  prior_games=prior_games, prefix="xg")
