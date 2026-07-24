"""Cycle 3: split team strength by strength state (even strength / power play
/ penalty kill) using MoneyPuck's own situational rows, instead of the
whole-game ("all") xG used in xg_based_poisson (Cycle 2). See
MODEL_DOCUMENTATION.md Sec3.2 for why this split matters: a team's 5v5
offense, PP conversion, and PK suppression are different skills with
different sample sizes, and blending them into one whole-game number (as
Cycle 2 did) throws that away.

Note on naming: MoneyPuck's own situational rows are 5on5/5on4/4on5, but
this module calls the first one "ev" (even strength) rather than "5v5" --
"5v5" is not a valid Python identifier prefix (can't start a name with a
digit), which breaks pandas itertuples() attribute access silently
(confirmed: it falls back to positional "_1" names instead of raising,
which would have been a much easier bug to catch). Renamed to sidestep the
whole class of problem rather than working around it.

Deliberately scoped to isolate ONE new variable versus Cycle 2 (situational
rate splitting), not two: the TIME-ON-ICE weight used to combine each
situation's rate into a per-game total is a single LEAGUE-AVERAGE per
situation (not team-specific expected PP/PK minutes) -- whether a team's own
above/below-average penalty-drawing rate should also feed into its expected
PP time is a separate, real idea, but bundling it into this same cycle would
make it impossible to tell which change (rate-splitting vs. TOI-weighting)
was responsible for any observed improvement. Left as a candidate for a
later, separately-tested cycle.

Also deliberately NOT modeling shorthanded goals-for (a team scoring while
killing its own penalty) as a separate term -- a genuinely small-magnitude,
rare event league-wide (confirmed: ~0.07 xG/team-game, ~2.4% of all xG),
folding it in is a minor, low-priority addition.

CAUGHT AND FIXED DURING THIS CYCLE (2026-07-23): the first version of this
module only used MoneyPuck's `5on5`/`5on4`/`4on5` situational rows, which I
assumed (wrongly, not checked first) covered the whole game. Validating
against real games produced a predicted mean total goals of ~5.0 against a
real ~5.86 -- too large a gap to be explained by the deliberately-omitted
shorthanded-goals-for term alone. Investigated by comparing MoneyPuck's
directly-published `situation=="all"` xG total (used in Cycle 2) against
the sum of the three situations used here: they did NOT match (2.434 vs
2.867 mean xG/team-game) -- confirming a real, un-modeled fifth situation.
MoneyPuck's own data has a 5th situation, `other`, covering the strength
states 5v5/5v4/4v5 don't (4v4, 3v3 overtime, 6-on-5 empty net, etc.) --
confirmed NOT negligible, ~0.358 xG/team-game, ~12.5% of the whole-game
total. Added as a 4th additive component below (`other_*`), using the same
even-strength-style symmetric treatment (its for/against columns are
symmetric league-wide by construction, same as 5v5, unlike PP/PK) rather
than left out. Lesson: "the situations I already knew the names of" is not
the same claim as "all the situations there are" -- should have listed the
real distinct values in the `situation` column before assuming coverage,
the same category of mistake the project's own data-survey discipline
(MODEL_DOCUMENTATION.md Sec1) exists to prevent, just recurring one layer
deeper in a single column's own possible values.

Confirmed empirically before writing the combine logic (2026-07-23): 5v5
TOI averages ~48.7 min/game, PP/PK TOI averages ~4.9 min/game each (944 of
46,444 team-game situational rows have zero PP/PK TOI -- a team that drew
zero power plays that game -- handled correctly by construction since
cumulative-sum-based shrinkage never divides by a single row's TOI).
"""

import pandas as pd

from src.ingest.team_codes import normalize_moneypuck_team_code, normalize_nhl_api_team_code
from src.models.baseline_naive_poisson import build_team_game_log
from src.models.shrinkage import add_walk_forward_toi_rate, add_walk_forward_toi_rate_cross_season

# Recalibrated 2026-07-23 (MODEL_DOCUMENTATION.md Sec7.5) -- scaled to 0.6x the original
# "~10 games worth" placeholders (487/49/49/23 min) after adopting a decayed (not infinite-
# memory) league-average baseline (see shrinkage.py's league_avg_halflife_games): the
# original constants were implicitly tuned against the OLD expanding-mean baseline's own
# lag/bias pattern, and kept unchanged after switching baselines produced a real, bootstrap-
# confirmed margin-MAE regression (Sec7.4) even though the decay fix itself was correct.
# Grid-searching prior_minutes jointly with the new baseline found 0.6x resolves this --
# real margin-MAE improvement, neutral Brier, positive-trending total-MAE -- confirmed via
# bootstrap before adopting, not assumed from the point estimate alone.
PRIOR_MINUTES_EV = 487 * 0.6
PRIOR_MINUTES_PP = 49 * 0.6
PRIOR_MINUTES_PK = 49 * 0.6
PRIOR_MINUTES_OTHER = 23 * 0.6


def _situational_slice(moneypuck: pd.DataFrame, situation: str, prefix: str) -> pd.DataFrame:
    d = moneypuck[moneypuck["situation"] == situation].copy()
    d["team"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(d["team"], d["season"])]
    return d[["gameId", "team", "xGoalsFor", "xGoalsAgainst", "iceTime"]].rename(columns={
        "xGoalsFor": f"{prefix}_xgf", "xGoalsAgainst": f"{prefix}_xga", "iceTime": f"{prefix}_toi_sec",
    })


def build_team_game_situational_log(schedule: pd.DataFrame, moneypuck: pd.DataFrame) -> pd.DataFrame:
    """One row per team per real regular-season game: actual goals_for/
    goals_against (the target) plus even-strength/PP/PK xG-for, xG-against,
    and TOI (seconds) from MoneyPuck's own situational splits."""
    goals_log = build_team_game_log(schedule)
    goals_log["season_start"] = goals_log["season"].astype(str).str[:4].astype(int)
    goals_log["team"] = [normalize_nhl_api_team_code(t, s) for t, s in zip(goals_log["team"], goals_log["season_start"])]
    goals_log = goals_log.drop(columns=["season_start"])

    ev = _situational_slice(moneypuck, "5on5", "ev")
    pp = _situational_slice(moneypuck, "5on4", "pp")
    pk = _situational_slice(moneypuck, "4on5", "pk")
    other = _situational_slice(moneypuck, "other", "other")

    merged = goals_log.merge(ev, on=["gameId", "team"], how="inner") \
                       .merge(pp, on=["gameId", "team"], how="inner") \
                       .merge(pk, on=["gameId", "team"], how="inner") \
                       .merge(other, on=["gameId", "team"], how="inner")
    dropped = len(goals_log) - len(merged)
    if dropped:
        print(f"team_strength_situational: {dropped}/{len(goals_log)} team-game rows had no full situational "
              f"match (even-strength+PP+PK+other), dropped")
    return merged


def add_walk_forward_situational_strength(log: pd.DataFrame,
                                           league_avg_halflife_games: float | None = None,
                                           cross_season_weight: float | None = None,
                                           prior_minutes_multiplier: float = 1.0) -> pd.DataFrame:
    """`league_avg_halflife_games`: see shrinkage._trailing_league_stat --
    None (default) preserves every existing caller's exact behavior; a real
    number tests the Sec4.15 scoring-era-drift fix.

    `cross_season_weight`: see shrinkage.add_walk_forward_toi_rate_cross_season
    -- None (default) preserves every existing caller's exact behavior
    (uses add_walk_forward_toi_rate); a real number tests the Sec12
    cross-season team-strength prior on all four situational channels.

    `prior_minutes_multiplier`: further multiplies PRIOR_MINUTES_EV/PP/PK/OTHER
    (already ×0.6-scaled for the decayed baseline, Sec7.5) by this factor --
    1.0 (default) preserves every existing caller's exact behavior. Tests
    whether the prior-strength constants need re-tuning now that the prior
    they're weighting is a cross-season blend rather than pure league
    average (Sec7.5's own lesson: priors and any new carryover/decay term
    are entangled, don't assume the old scale stays optimal), see Sec13."""
    channels = [("ev_xgf", "ev_xga", "ev_toi_sec", PRIOR_MINUTES_EV * prior_minutes_multiplier, "ev"),
                ("pp_xgf", "pp_xga", "pp_toi_sec", PRIOR_MINUTES_PP * prior_minutes_multiplier, "pp"),
                ("pk_xgf", "pk_xga", "pk_toi_sec", PRIOR_MINUTES_PK * prior_minutes_multiplier, "pk"),
                ("other_xgf", "other_xga", "other_toi_sec", PRIOR_MINUTES_OTHER * prior_minutes_multiplier, "other")]
    for for_col, against_col, toi_col, prior_minutes, prefix in channels:
        if cross_season_weight is None:
            log = add_walk_forward_toi_rate(log, for_col, against_col, toi_col, prior_minutes, prefix,
                                             league_avg_halflife_games=league_avg_halflife_games)
        else:
            log = add_walk_forward_toi_rate_cross_season(
                log, for_col, against_col, toi_col, prior_minutes, prefix, cross_season_weight,
                league_avg_halflife_games=league_avg_halflife_games)
    return log


def _combine(attack_per60: float, opp_defense_per60: float, league_avg_attack: float, league_avg_defense: float,
             toi_min: float) -> float:
    return league_avg_attack * (attack_per60 / league_avg_attack) * (opp_defense_per60 / league_avg_defense) \
        * (toi_min / 60)


def predict_situational_lambda(
    home_ev_attack: float, home_ev_league_avg_for: float,
    home_ev_defense: float, home_ev_league_avg_against: float,
    away_ev_attack: float, away_ev_league_avg_for: float,
    away_ev_defense: float, away_ev_league_avg_against: float,
    home_pp_attack: float, home_pp_league_avg_for: float,
    away_pp_attack: float, away_pp_league_avg_for: float,
    home_pk_defense: float, home_pk_league_avg_against: float,
    away_pk_defense: float, away_pk_league_avg_against: float,
    home_other_attack: float, home_other_league_avg_for: float,
    home_other_defense: float, home_other_league_avg_against: float,
    away_other_attack: float, away_other_league_avg_for: float,
    away_other_defense: float, away_other_league_avg_against: float,
    league_avg_ev_toi_min: float, home_pp_toi_min: float, away_pp_toi_min: float, league_avg_other_toi_min: float,
    league_avg_sh_rate_per60: float = 0.0, home_pk_toi_min: float = 0.0, away_pk_toi_min: float = 0.0,
) -> tuple[float, float]:
    """Combines even-strength + PP/PK-against + other (4v4/3v3-OT/empty-net/
    etc., see module docstring) components for each side. Takes plain
    scalars (not row objects) so callers aren't coupled to a particular
    merge/suffix layout. Each rate is paired with ITS OWN matching league
    average -- a PP attack rate against the PP-for league average, a PK
    defense rate against the PK-AGAINST league average (which is the true
    symmetric partner of the PP-for league average: a power-play goal for
    one team is simultaneously a shorthanded-goal-against for the other,
    same clock window -- see shrinkage.py's bug writeup for why using a
    mismatched pair inflated ratios ~8x before this was fixed). "other" is
    treated symmetrically like even-strength (its for/against are symmetric
    league-wide by construction, unlike PP/PK).

    EV and "other" TOI weight are still a single shared league-average
    (out of scope for the team-specific-TOI cycle -- see
    MODEL_DOCUMENTATION.md Sec4.5). PP TOI weight is now `home_pp_toi_min`/
    `away_pp_toi_min` -- SEPARATE values, generalizing Cycle 3 (which is
    exactly the special case home_pp_toi_min == away_pp_toi_min == the
    shared league average) to let each side's own expected power-play time
    differ, per Sec4.5's cycle.

    `league_avg_sh_rate_per60`/`home_pk_toi_min`/`away_pk_toi_min` (Cycle 23,
    Sec18): shorthanded-goals-for, deliberately left unmodeled since Cycle 3
    ("a genuinely small-magnitude, rare event... folding it in is a minor,
    low-priority addition") -- Sec18 found it's the dominant, fully
    explained driver of the PP+PK bucket's own 16-season, zero-exception
    undershoot (real SH goals ~0.140/game league-wide, stable across the
    whole 2008-2025 window, SH-xG already well-calibrated to SH-goals in
    aggregate, ratio 1.008 -- no fitted conversion factor needed). Defaults
    to 0.0 (exact no-op, preserves every existing caller's behavior).
    LEAGUE-LEVEL only (not team-specific shrinkage) for this first version --
    at ~0.14/game league-wide, a team's own single-season SH-goal sample is
    tiny (~11 events/season), too small for a reliable team-specific rate
    without its own dedicated repeatability check."""
    home_ev = _combine(home_ev_attack, away_ev_defense, home_ev_league_avg_for, away_ev_league_avg_against,
                        league_avg_ev_toi_min)
    away_ev = _combine(away_ev_attack, home_ev_defense, away_ev_league_avg_for, home_ev_league_avg_against,
                        league_avg_ev_toi_min)

    home_pp = _combine(home_pp_attack, away_pk_defense, home_pp_league_avg_for, away_pk_league_avg_against,
                        home_pp_toi_min)
    away_pp = _combine(away_pp_attack, home_pk_defense, away_pp_league_avg_for, home_pk_league_avg_against,
                        away_pp_toi_min)

    # Shorthanded-goals-for: this team's own goal, scored while THIS team is
    # shorthanded (killing a penalty) -- a league-wide rate applied to this
    # team's own (already walk-forward-estimated) PK ice time.
    home_sh = league_avg_sh_rate_per60 * (home_pk_toi_min / 60)
    away_sh = league_avg_sh_rate_per60 * (away_pk_toi_min / 60)

    home_other = _combine(home_other_attack, away_other_defense, home_other_league_avg_for,
                           away_other_league_avg_against, league_avg_other_toi_min)
    away_other = _combine(away_other_attack, home_other_defense, away_other_league_avg_for,
                           home_other_league_avg_against, league_avg_other_toi_min)

    return home_ev + home_pp + home_other + home_sh, away_ev + away_pp + away_other + away_sh
