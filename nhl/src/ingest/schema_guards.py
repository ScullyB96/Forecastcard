"""Data-dependency canaries for the two ingest sources everything else in
this project rests on: the NHL API schedule and MoneyPuck's team-
situational CSV. Task #44's degradation budget and §38's emergency-fix
path both assume a provider breaking gets NOTICED immediately, not
discovered a week later as silently-wrong predictions -- these assertions
are what makes that trigger on day one.

Same pattern as `fetch_rotowire_injuries.py`'s status-value warning
(Sec39.1): fail loudly on a genuinely new/missing value that this
project's own model logic depends on, rather than silently either
crashing downstream in a confusing way or quietly producing wrong numbers.
Mirrors this project's own history directly -- Cycle 3's original
`team_strength_situational.py` bug (assuming 5on5/5on4/4on5 was the
complete situation list, missing `other`, ~12.5% of real xG) is exactly
the failure mode `assert_moneypuck_situational_schema` exists to catch
automatically, rather than requiring another multi-day-gap discovery.

Deliberately NOT strict about every column value this project's model
doesn't actually consume: `schedule["gameType"]` has real, legitimate
values beyond {1,2,3} (confirmed live 2026-07-25: 1,2,3,4,6,7,8,9,12,19,20
all appear in the real cached schedule -- likely all-star/skills-
competition/exhibition variants this project has never needed to name
since it only ever filters to `gameType==2`) -- failing loudly on every
new gameType value seen would be noise, not signal. The checks below
scope strictly to what the model's own pipeline (`_load_schedule`'s
`gameType==2` filter) actually depends on being well-formed.
"""

REQUIRED_SCHEDULE_COLUMNS = {
    "gameId", "season", "gameType", "gameDate", "gameState", "lastPeriodType",
    "awayTeamAbbrev", "awayScore", "homeTeamAbbrev", "homeScore",
}
KNOWN_REGULAR_SEASON_GAME_STATES = {"FINAL", "OFF", "FUT"}
KNOWN_LAST_PERIOD_TYPES = {"REG", "OT", "SO"}

REQUIRED_MONEYPUCK_COLUMNS = {
    "gameId", "team", "season", "situation", "gameDate", "xGoalsFor", "xGoalsAgainst", "iceTime",
}
KNOWN_SITUATIONS = {"5on5", "5on4", "4on5", "all", "other"}


class SchemaGuardError(Exception):
    """Raised when a real ingest source no longer matches what this
    project's model logic assumes -- the exact trigger §38's emergency-fix
    path exists for, not a routine warning to print past."""


def assert_schedule_schema(schedule) -> None:
    missing_cols = REQUIRED_SCHEDULE_COLUMNS - set(schedule.columns)
    if missing_cols:
        raise SchemaGuardError(
            f"NHL API schedule is missing required column(s): {sorted(missing_cols)} -- "
            f"the API's own response shape has changed; every downstream model file that reads "
            f"this column will fail or silently produce wrong numbers. Investigate before any "
            f"prediction runs on this data.")

    real_games = schedule[schedule["gameType"] == 2]
    if real_games.empty:
        return  # nothing to check yet (e.g. pure off-season cache) -- not itself an error

    bad_states = set(real_games["gameState"].dropna().unique()) - KNOWN_REGULAR_SEASON_GAME_STATES
    if bad_states:
        raise SchemaGuardError(
            f"NHL API returned unrecognized gameState value(s) for real (gameType==2) games: "
            f"{sorted(bad_states)} -- known values are {sorted(KNOWN_REGULAR_SEASON_GAME_STATES)}. "
            f"Every model file filtering on gameState=='OFF' needs re-checking against whatever "
            f"this new value means before trusting predictions built from it.")

    decided = real_games[real_games["gameState"].isin(["FINAL", "OFF"])]
    bad_periods = set(decided["lastPeriodType"].dropna().unique()) - KNOWN_LAST_PERIOD_TYPES
    if bad_periods:
        raise SchemaGuardError(
            f"NHL API returned unrecognized lastPeriodType value(s) for decided games: "
            f"{sorted(bad_periods)} -- known values are {sorted(KNOWN_LAST_PERIOD_TYPES)}. The "
            f"entire OT/SO sub-model (overtime_shootout.py, ot_logistic.py) assumes exactly these "
            f"three; a new value (e.g. a rule change adding a new tiebreaker format) needs a real "
            f"model update, not silent misclassification.")


def assert_moneypuck_situational_schema(moneypuck) -> None:
    missing_cols = REQUIRED_MONEYPUCK_COLUMNS - set(moneypuck.columns)
    if missing_cols:
        raise SchemaGuardError(
            f"MoneyPuck team-situational CSV is missing required column(s): {sorted(missing_cols)} "
            f"-- the file's own schema has changed upstream; every model file reading this column "
            f"will fail or silently produce wrong numbers.")

    real_situations = set(moneypuck["situation"].unique())
    missing_situations = KNOWN_SITUATIONS - real_situations
    new_situations = real_situations - KNOWN_SITUATIONS
    if missing_situations or new_situations:
        raise SchemaGuardError(
            f"MoneyPuck's 'situation' column no longer matches the known set. "
            f"Missing: {sorted(missing_situations) or 'none'}. "
            f"New/unrecognized: {sorted(new_situations) or 'none'}. "
            f"This is EXACTLY the failure mode that produced Cycle 3's original bug (assuming "
            f"5on5/5on4/4on5 was complete, silently missing the 'other' situation, ~12.5% of "
            f"real xG) -- investigate before trusting any prediction built from this data, don't "
            f"silently drop or add a situation to team_strength_situational.py without re-deriving "
            f"the combine logic.")
