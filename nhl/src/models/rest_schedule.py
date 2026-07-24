"""Cycle 11: rest/schedule (back-to-back) adjustment (MODEL_DOCUMENTATION.md
Sec3.2/Sec4.14) -- flagged since the architecture proposal, tested here for
the first time.

Checked for a real effect BEFORE building anything (same discipline as the
NegBin rejection, Sec4.11): correlated rest days and a back-to-back
indicator against the current best model's own scoring residuals on the
development set. Results were mixed and mostly negligible --
correlation coefficients for continuous rest_days were tiny (r=0.004 to
0.027, i.e. well under 0.1% of variance explained) -- but a targeted
extreme-contrast check (back-to-back vs. 3+ days rested) found ONE real,
statistically significant effect: an AWAY team playing the second game of
a back-to-back scores about 0.16 fewer goals than an equally-rested away
team would (t-test p=0.0041, n=3,785 back-to-back away-side games). The
equivalent HOME-side effect was NOT significant (p=0.12), and the model
actually slightly under-predicts home-team win probability on home
back-to-backs rather than over-predicting it -- so only the away-side
effect is built here, not a general "back-to-back always hurts" rule.

Rest days are computed within-season only (reset at season boundary, same
convention as every other walk-forward feature in this project) -- the
gap between a team's last game of one season and first game of the next
(often 100+ days, all-star break, injury recovery, etc.) isn't a
meaningful "fatigue" signal in the same sense as a genuine in-season
back-to-back.
"""

import pandas as pd

from src.models.baseline_naive_poisson import build_team_game_log


def add_rest_days(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (via build_team_game_log), with rest_days
    added: days of rest since that team's own immediately preceding game
    THIS SEASON (0 = back-to-back; NaN for a season's first game for that
    team, left as NaN -- callers should treat a missing value as "unknown,
    don't adjust" rather than imputing a guess)."""
    log = build_team_game_log(schedule)
    log["gameDate_dt"] = pd.to_datetime(log["gameDate"])
    log = log.sort_values(["team", "season", "gameDate_dt"]).reset_index(drop=True)
    prev_date = log.groupby(["team", "season"])["gameDate_dt"].shift(1)
    log["rest_days"] = (log["gameDate_dt"] - prev_date).dt.days - 1
    return log


def fit_away_b2b_adjustment(schedule: pd.DataFrame, actual_away: pd.Series, lambda_away: pd.Series,
                             game_ids: pd.Series) -> float:
    """Empirical mean(actual_away - lambda_away | away team on a
    back-to-back) minus the same for all other away games -- a single
    global additive constant, fit once on the games passed in (callers
    should restrict to development-set games), same practice as every
    other global constant in this project (home_ice_multiplier, goalie
    shrinkage prior, etc.). This DIFFERENCE is a clean measurement of the
    real, tonight-specific causal effect (Sec22) -- both the B2B and
    non-B2B fitting groups' own `lambda_away` already carry the SAME
    embedded historical contamination (see `symmetric_b2b_bias_credit`
    below), so it cancels out of this subtraction and doesn't bias the fit
    itself. Applied here as `lambda_away += (rest_days==0) * this value`,
    unconditionally on the away side only -- kept exactly as originally
    shipped; see Sec22 for the separate, additional correction needed."""
    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]]
    df = pd.DataFrame({"gameId": game_ids, "actual_away": actual_away, "lambda_away": lambda_away}) \
        .merge(away_rest, on="gameId", how="left")
    resid = df["actual_away"] - df["lambda_away"]
    b2b_mean = resid[df["rest_days"] == 0].mean()
    other_mean = resid[df["rest_days"] != 0].mean()
    return float(b2b_mean - other_mean)


def add_walk_forward_b2b_incidence(schedule: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward (expanding-mean, strictly-prior-games-only) P(a
    league-wide away-side game is on a back-to-back), one row per gameId --
    Sec22's symmetric bias credit needs this to track any real drift in B2B
    scheduling frequency over time (e.g. load-management-driven schedule
    changes), rather than a single fixed dev-set constant."""
    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "gameDate", "rest_days"]].copy()
    away_rest["gameDate_dt"] = pd.to_datetime(away_rest["gameDate"])
    away_rest = away_rest.sort_values(["gameDate_dt", "gameId"]).reset_index(drop=True)
    away_rest["is_b2b"] = (away_rest["rest_days"] == 0).astype(float)
    away_rest["p_b2b_walk_forward"] = away_rest["is_b2b"].shift(1).expanding().mean().bfill()
    return away_rest[["gameId", "p_b2b_walk_forward"]]


def symmetric_b2b_bias_credit(p_b2b_walk_forward: pd.Series, away_b2b_adj: float) -> pd.Series:
    """Sec22 (corrected from Sec19.5/Sec20's WRONG, away-only-conditional
    "centered" version, which put the entire returned credit on the away
    side and thereby shifted every game's predicted MARGIN toward away by
    ~0.027 -- confirmed as the real, bootstrap-measured cause of that
    version's margin-MAE regression).

    The real mechanism: `away_b2b_adj` is a real, validated, tonight-
    specific causal effect (kept as `fit_away_b2b_adjustment` measures it,
    unchanged). But EVERY team's own walk-forward attack-rate estimate --
    whether that team is home or away TONIGHT -- pools that team's own
    entire history, INCLUDING the ~p_b2b fraction of ITS OWN away games
    that were themselves back-to-backs. Since roughly half of any team's
    own games are away games, the embedded contamination in each team's own
    rate is `0.5 * p_b2b * away_b2b_adj` -- present in BOTH sides of
    EVERY game (whichever team is home tonight, whichever is away), because
    it comes from each team's own history, not tonight's specific matchup.
    The fix is a SYMMETRIC, UNCONDITIONAL credit added to BOTH lambdas,
    every game, regardless of tonight's B2B status: `-0.5 * p_b2b *
    away_b2b_adj` (a positive quantity, since `away_b2b_adj` is negative).

    This cancels EXACTLY out of margin (added identically to both sides --
    not merely approximately mean-zero over the population, bit-identical
    per game) while raising the total mean by exactly this credit's full
    value on BOTH sides (2x the per-side credit) -- recovering precisely
    the ladder's missing rest-stage bias, since the existing (unchanged)
    conditional term's own population-mean effect is `p_b2b * away_b2b_adj`,
    and `2 * (-0.5*p_b2b*away_b2b_adj) + p_b2b*away_b2b_adj = 0`, i.e. the
    NET shift relative to the (still-contaminated) raw team-strength output
    is exactly the `+0.027`-sized correction the ladder needs."""
    return -0.5 * p_b2b_walk_forward * away_b2b_adj
