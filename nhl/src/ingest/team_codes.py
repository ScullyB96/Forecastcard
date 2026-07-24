"""Reconcile NHL API team abbreviations against MoneyPuck's team codes.

This turned out to be more than a static find-and-replace -- confirmed
directly against both live sources (2026-07-23), not assumed:

1. MoneyPuck changed ITS OWN abbreviation convention mid-history: seasons
   2008-2020 use a dotted style (L.A, N.J, S.J, T.B); seasons 2021-2025 use
   plain 3-letter codes (LAK, NJD, SJS, TBL) that already match the NHL API.
   Confirmed by checking which `season` values each variant appears under
   in the real downloaded `all_teams.csv` -- the switch lands exactly at
   the 2021 season, no overlap or gap.

2. The NHL API's OWN numeric team id is NOT stable across a market
   rename, which is a sharper gotcha than "the abbreviation changed":
   Phoenix Coyotes (abbrev PHX, id 27, through the 2013-14 season) and
   Arizona Coyotes (abbrev ARI, id 53, 2014-15 through 2023-24) are stored
   as two different team ids in the NHL API despite being the same
   continuous roster/organization -- confirmed by pulling real schedule
   weeks straddling the 2014 rename and reading the `id` field directly,
   not inferred from the abbreviation alone. MoneyPuck, by contrast, does
   NOT distinguish the Phoenix vs. Arizona eras -- it uses `ARI` uniformly
   for the entire 2008-2023 span. So for JOINING the two sources, NHL
   API's PHX rows map to MoneyPuck's ARI code -- confirmed correct, not
   the same thing as claiming PHX and ARI are "the same team" in any
   deeper sense.

3. Utah (abbrev UTA, id 59, 2024-25 on) is a genuinely distinct id from
   Arizona (53) in the NHL API, and MoneyPuck agrees -- UTA is a separate
   code starting at MoneyPuck's season 2024, with no ARI/UTA overlap.
   Both sources treat the 2024 Arizona-to-Utah transition as a boundary,
   unlike the Phoenix/Arizona case where only the NHL API draws a line.
4. Atlanta Thrashers (ATL, id 11) to Winnipeg Jets (WPG, id 52, 2011-12
   season on) is a clean relocation boundary both sources agree on.

IMPORTANT: this module answers "which rows join to which" -- a data
question. It does NOT answer "should a team-strength rating carry over
across one of these boundaries" -- that's a modeling decision (Utah
inherited most of Arizona's actual roster via the 2024 sale, which is a
real argument for treating it as continuous for a roster-driven strength
rating, unlike a true expansion team with no roster history at all) left
to the team-strength layer (Sec3.2 of MODEL_DOCUMENTATION.md), not baked
in here.
"""

import pandas as pd

# MoneyPuck's pre-2021-season dotted codes -> the plain code MoneyPuck itself
# switches to from the 2021 season on (which already matches the NHL API).
# Confirmed: this covers every dotted code found in the real file; no other
# team ever appeared under a MoneyPuck-only dotted variant.
MONEYPUCK_DOTTED_TO_PLAIN = {
    "L.A": "LAK",
    "N.J": "NJD",
    "S.J": "SJS",
    "T.B": "TBL",
}

# NHL API abbrev -> MoneyPuck code, for the one case where MoneyPuck merges
# two NHL-API-distinct eras under a single code. Every other NHL API abbrev
# is passed through unchanged (it already matches MoneyPuck's plain code,
# post-normalization above).
NHL_API_TO_MONEYPUCK = {
    "PHX": "ARI",  # Phoenix Coyotes (NHL API id 27) -> MoneyPuck's ARI (2008-2023)
}


def normalize_moneypuck_team_code(team: str, season: int) -> str:
    """MoneyPuck `team` column -> canonical (NHL-API-style) 3-letter code.
    `season` isn't actually needed for disambiguation here (each dotted code
    maps to exactly one plain code across all seasons it appears in) but is
    accepted for symmetry with normalize_nhl_api_team_code and in case a
    future MoneyPuck refresh introduces a season-dependent case."""
    return MONEYPUCK_DOTTED_TO_PLAIN.get(team, team)


def normalize_nhl_api_team_code(abbrev: str, season: int) -> str:
    """NHL API `abbrev` (already canonical/3-letter) -> the code MoneyPuck's
    `team` column uses for that season, so the two sources join cleanly."""
    return NHL_API_TO_MONEYPUCK.get(abbrev, abbrev)


def validate_crosswalk(schedule: pd.DataFrame, moneypuck: pd.DataFrame) -> dict:
    """Real validation, not a spot check: normalizes every team code on both
    sides and confirms every (normalized_code, season) pair that appears in
    the NHL API schedule also appears in MoneyPuck's data, and vice versa.
    Returns a dict of any mismatches found -- empty dict means a clean join.
    `schedule.season` is NHL API's YYYYYYYY form (e.g. 20232024); MoneyPuck's
    `season` is the single starting year (e.g. 2023) -- converted here."""
    sched = schedule.copy()
    sched["season_start"] = sched["season"].astype(str).str[:4].astype(int)
    sched_codes = set()
    for _, row in sched.iterrows():
        for col in ("awayTeamAbbrev", "homeTeamAbbrev"):
            sched_codes.add((normalize_nhl_api_team_code(row[col], row["season_start"]), row["season_start"]))

    mp = moneypuck.copy()
    mp["norm_team"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(mp["team"], mp["season"])]
    mp_codes = set(zip(mp["norm_team"], mp["season"]))

    only_in_schedule = sched_codes - mp_codes
    only_in_moneypuck = mp_codes - sched_codes
    return {
        "only_in_schedule": sorted(only_in_schedule),
        "only_in_moneypuck": sorted(only_in_moneypuck),
    }


if __name__ == "__main__":
    from src.utils.paths import DATA_RAW

    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[schedule["gameType"] == 2]  # regular season only, matches MoneyPuck's own game log
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")
    moneypuck = moneypuck[moneypuck["situation"] == "all"]

    result = validate_crosswalk(schedule, moneypuck)
    print("only_in_schedule (NHL API has this team/season, MoneyPuck doesn't):")
    print(result["only_in_schedule"])
    print("\nonly_in_moneypuck (MoneyPuck has this team/season, NHL API schedule doesn't):")
    print(result["only_in_moneypuck"])
