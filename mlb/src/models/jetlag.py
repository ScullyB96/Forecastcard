"""Team-level circadian jet-lag fatigue factor -- a genuinely new signal
CATEGORY (schedule-derived, team-level) distinct from every batter/pitcher
rate signal built so far this session.

Motivation (external research pass, PNAS 2017 jet-lag study, 1992-2011 data):
teams traveling east across multiple time zones with insufficient rest show
measurably worse offense. Rather than trust the external study's magnitudes
directly (pre-dates the 2023 rules changes and covers a different era),
this was independently validated on THIS project's own 2023-2025 data before
building anything.

VALIDATION (2026-07-22, own schedule + PA table, team-season-demeaned to
control for underlying team quality):
- >=2 timezone eastward travel showed NO significant effect on team runs
  scored (p=0.21-0.76 across several rest-day cutoffs) -- too weak a dose.
- >=3 timezone eastward travel (regardless of rest days) showed a real,
  statistically significant effect: team runs scored -0.49/game relative to
  that team's own season average (p=0.019, n=200 team-games) vs a
  near-zero baseline delta for non-traveling games.
- Negative control (WESTWARD travel, which the external study says should
  show no effect): >=2TZ westward showed no significant effect (p=0.21,
  and if anything trended the WRONG direction, i.e. pure noise) -- exactly
  the expected null result, a real confirmatory check that the eastward
  finding isn't an artifact of the demeaning method.
- At the per-PA level (n=191 team-games meeting the >=3TZ-eastward bar):
  hit_rate -4.15% relative (p=0.078, marginal), hr_rate -7.09% (p=0.31,
  not significant alone), total-bases-per-PA ("slg_proxy") -5.98%
  (p=0.043, significant). Individual splits are underpowered at this
  sample size, but every single metric points the same direction
  (suppressed offense), consistent with the team-runs-level finding.

Given the marginal-but-directionally-consistent per-category evidence, this
is built as a SINGLE uniform multiplier on all HIT_EVENTS categories (same
"contact quality"-style uniform scaling as expected_stats.py's
contact_quality_multiplier), sized to the most defensible measured
aggregate (-4.15% relative hit rate), rather than guessing at differentiated
per-category multipliers the sample can't actually support.

Scope, deliberately narrow: applies ONLY to the traveling team's OFFENSE
(the effect that was actually validated). The pitching-side effect (more
HR allowed when jet-lagged) trended the same direction in this project's
own data but was not independently significant here, so it is NOT modeled
separately -- doing so would be adding an unvalidated second effect on top
of an already-marginal first one.

FULL-STACK A/B RESULT (2026-07-22, n=597, wired into game_simulator.py's
combine_matchup_distribution/simulate_half_inning/simulate_game and
validate_game_simulator.py's per-game loop): despite the real, statistically
significant component-level effect above, this REGRESSED straight-up
accuracy -- SU 60.5%->59.3% (-1.2pp), while total MAE moved only marginally
(3.409->3.407) and margin MAE got slightly worse (3.445->3.457). REVERTED
from game_simulator.py (the jetlag_factors param on
combine_matchup_distribution, batting_jetlag_factor on simulate_half_inning,
home_jetlag_factor/away_jetlag_factor on simulate_game and both half-inning
call sites) and validate_game_simulator.py (the import, team_jetlag_flags
construction, and per-game resolution/pass-through) -- confirmed via source
inspection matching the pre-deployment state. This module is kept as a
documented-but-unused artifact, same treatment as every other
investigated-and-rejected idea this session -- this was the third
consecutive failure in a row after the auto-runner fix and pulled-air-rate
win (speed-conditioned transitions -3.2pp, pitch_walk_multiplier -1.4pp,
jet-lag -1.2pp), reinforcing that even statistically real, leakage-free-
validated signals frequently do not survive contact with this project's
full-stack Monte Carlo simulator -- the full-stack A/B remains the only
reliable arbiter, never the component-level test alone.
"""

import numpy as np
import pandas as pd

# Fixed, real, baseball-season (daylight-saving) UTC offset in hours for each
# team's home ballpark -- a static physical fact about geography, same
# "baked-in real-world constant" pattern as park_orientation.py's stadium
# facts. Arizona has no DST (fixed at -7 year-round); every other park
# observes DST, and the MLB season (Feb spring training through Oct/Nov
# postseason) falls almost entirely within DST for the whole US.
TEAM_UTC_OFFSET = {
    "NYY": -4, "NYM": -4, "BOS": -4, "TB": -4, "TOR": -4, "BAL": -4,
    "PHI": -4, "PIT": -4, "ATL": -4, "MIA": -4, "WSH": -4, "CLE": -4,
    "DET": -4, "CIN": -4,
    "CHC": -5, "CWS": -5, "MIL": -5, "STL": -5, "MIN": -5, "KC": -5,
    "TEX": -5, "HOU": -5,
    "COL": -6, "AZ": -7,
    "LAD": -7, "LAA": -7, "ATH": -7, "SD": -7, "SF": -7, "OAK": -7, "SEA": -7,
}

JETLAG_TZ_THRESHOLD = 3  # validated dose -- >=2 showed no significant effect on real data

# Uniform multiplier on HIT_EVENTS (single/double/triple/home_run) for the
# jet-lagged team's own batters -- sized to the measured -4.15% relative hit
# rate (the most defensible single aggregate this sample size supports).
JETLAG_HIT_MULTIPLIER = 0.9585


def build_team_travel_log(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): that team's own chronological game log
    (home AND away games combined, since a team travels between whatever
    cities it plays in, not just to/from its own park), with the prior
    game's venue/date already looked up. Entirely schedule-derived --
    walk-forward-safe by construction (a team's travel history is always
    fully known in advance of any future game, unlike a player rate)."""
    sched = schedule[(schedule["game_type"] == "R") & (schedule["status"] == "Final")].copy()
    sched["date"] = pd.to_datetime(sched["date"])

    home = sched[["game_pk", "date", "season", "home_team", "venue_name"]].rename(columns={"home_team": "team"})
    away = sched[["game_pk", "date", "season", "away_team", "venue_name"]].rename(columns={"away_team": "team"})
    team_games = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])

    venue_tz = home[["venue_name", "team"]].drop_duplicates("venue_name").set_index("venue_name")["team"].map(
        TEAM_UTC_OFFSET
    )
    team_games["tz"] = team_games["venue_name"].map(venue_tz)
    team_games["prev_tz"] = team_games.groupby("team")["tz"].shift(1)
    team_games["tz_delta"] = team_games["tz"] - team_games["prev_tz"]
    return team_games


def build_team_jetlag_flags(schedule: pd.DataFrame) -> dict[tuple[int, str], bool]:
    """{(game_pk, team): bool} -- True if that team traveled >=JETLAG_TZ_THRESHOLD
    timezones eastward since their immediately preceding game (their very
    first game of a season/dataset has no prior game to compare against,
    so it's never flagged -- a cold start, not a guess)."""
    log = build_team_travel_log(schedule)
    log["jetlagged"] = log["tz_delta"] >= JETLAG_TZ_THRESHOLD
    return {(int(r.game_pk), r.team): bool(r.jetlagged) for r in log.itertuples()}


HIT_EVENTS = ("single", "double", "triple", "home_run")


def resolve_jetlag_factor(is_jetlagged: bool) -> dict[str, float] | None:
    """{outcome: multiplier} for combine_matchup_distribution's jetlag_factors
    slot, or None (no-op) when the team isn't flagged -- same "None means
    skip this layer entirely" convention as every other optional factor in
    game_simulator.py."""
    if not is_jetlagged:
        return None
    return {o: JETLAG_HIT_MULTIPLIER for o in HIT_EVENTS}
