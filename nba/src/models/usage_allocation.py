"""The composition rule that resolves the RAPM-lite / matchup-difficulty
double-counting risk -- the single most important design decision in the
props subsystem (see the approved plan's Sec4). RAPM-lite already answers
the MACRO question ("how many points will this active lineup score/allow,
total") -- validated in Phase 2, not touched here. Matchup difficulty
(`matchup_difficulty.py`) answers a different, MICRO question ("given that
total, which player gets how much, and does tonight's specific matchup
shift the mix"). These compose as MACRO-ANCHOR + MICRO-REALLOCATION, never
as two independent additive stacks on the same number:

- POINTS (anchored): each player's raw (minutes -> attempts -> makes,
  matchup-adjusted) projection sets a `usage_share`;
  `final_points_p = usage_share_p * team_total_points`, where
  `team_total_points` comes from Phase 1+2's ALREADY-RAPM-adjusted
  `team_strength.project_game` output. Matchup difficulty reshapes SHARES
  (via an additive point-scale nudge before normalizing); RAPM sets the
  TOTAL. `team_total_points` is the ONLY place the team total enters --
  matchup difficulty must never also rescale it, or the same signal moves
  the number twice.
- DREB/AST/TOV/STL/BLK (ANCHORED, Task #24): `team_stat_rates.py` builds a
  team-level walk-forward for/against combine for all 6 of
  OREB/DREB/AST/TOV/STL/BLK, mirroring Phase 1's pace x rating
  architecture. 5 of the 6 cleared the one-time confirmatory holdout check
  (see `team_stat_rates.ADOPTED_CATEGORIES` and MODEL_DOCUMENTATION.md
  Sec23) and are anchored here via the exact same `compute_usage_shares` +
  `allocate_team_total` mechanism as points -- these two functions were
  already stat-agnostic in their math, just named for points originally.
- OREB (UNANCHORED): the one category whose team-level model genuinely
  lost to naive on real holdout data (not just a widening gap -- the
  comparison flips sign), so it's deliberately excluded from anchoring and
  falls back to each player's own rate-model projection (`oreb_proj`)
  DIRECTLY, with no share-based rescaling -- honestly unanchored, not
  silently gold-plated. Left as a flagged follow-up investigation (see
  MODEL_DOCUMENTATION.md Sec23), not force-adopted.
"""

import pandas as pd

POINTS_PER_MAKE = {"2pt": 2, "3pt": 3, "ft": 1}


def raw_projected_points(scoring_rated: pd.DataFrame) -> pd.Series:
    """Sum of `player_scoring_rates.add_scoring_rates`'s three categories'
    projected makes, weighted by point value -- BEFORE any matchup
    adjustment or team-total anchoring. This is the raw signal usage
    shares are computed from."""
    return (POINTS_PER_MAKE["2pt"] * scoring_rated["2pt_proj_made"]
            + POINTS_PER_MAKE["3pt"] * scoring_rated["3pt_proj_made"]
            + POINTS_PER_MAKE["ft"] * scoring_rated["ft_proj_made"])


def matchup_point_delta(rate_delta: float, projected_minutes: float) -> float:
    """Converts a `matchup_difficulty.opponent_defense_adjustment` RATE
    delta (points allowed per matchup-minute, relative to league average)
    into a POINT-scale delta for one player's projected game, by scaling
    over their own projected minutes -- the simplest defensible exposure
    proxy. A more precise version would scale by projected MATCHUP minutes
    specifically (this player's real on-court time against the specific
    defenders who'll guard them), which requires resolving tonight's
    actual defensive assignments -- deferred to `generate_props.py`'s live
    wiring, not built here."""
    return rate_delta * projected_minutes


def compute_usage_shares(adjusted_points: pd.Series) -> pd.Series:
    """Normalizes a group of players' (already matchup-adjusted) raw point
    projections, for ONE team in ONE game, to shares summing to 1.

    Clips negative values to 0 first -- a matchup adjustment can in
    principle push a very-low-usage player's raw+adjustment below zero
    (an aggressive downward nudge on an already-small number), but a
    "usage share" can never be negative in reality, so a clipped floor is
    the correct fix, not a symptom to silently propagate.

    Falls back to EQUAL shares if every player's (clipped) value is 0 (a
    genuine edge case only for a group with no offensive signal at all,
    e.g. testing on a single row) -- avoids a divide-by-zero and is a
    defensible neutral default rather than an arbitrary crash."""
    clipped = adjusted_points.clip(lower=0)
    total = clipped.sum()
    if total <= 0:
        n = len(clipped)
        return pd.Series(1.0 / n, index=clipped.index) if n > 0 else clipped
    return clipped / total


def allocate_team_total(usage_shares: pd.Series, team_total: float) -> pd.Series:
    """The macro anchor itself: `final_p = usage_share_p x team_total`.
    For points, `team_total` must already be Phase 1+2's RAPM-adjusted
    `team_strength.project_game` output; for DREB/AST/TOV/STL/BLK, it's
    `team_stat_rates.project_team_stat`'s output for that category -- this
    function is stat-agnostic and does not compute or adjust the total
    itself, only distributes whatever total it's given."""
    return usage_shares * team_total
