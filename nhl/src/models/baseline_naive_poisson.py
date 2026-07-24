"""Deliberately naive walk-forward baseline: season-to-date RAW GOALS (no xG,
no situational splits, no goalie overlay, no rest/travel adjustment) fed into
independent Poisson home/away goal distributions.

Purpose (see MODEL_DOCUMENTATION.md Sec3.9/Sec4): this is the floor every
later refinement (xG-based strength, goalie overlay, NegBin, correlation
term, ...) must beat on real held-out games. If a "smarter" signal can't
out-predict this, it doesn't get kept -- mirrors the MLB project's
incremental-validation discipline.

Walk-forward guarantee: a team's attack/defense rate for game G is computed
ONLY from that team's own games strictly before G's date, within the same
season (deliberately no cross-season carryover yet -- an explicit,
separately-validated improvement to try later, not assumed here). The
league-average rate used for shrinkage is a trailing expanding mean over ALL
games strictly before G's date, with no season reset -- this avoids a
cold-start problem at each season's opener (Oct games start with a real,
if noisy, trailing league average from the tail of the prior season)
at the cost of a single genuinely noisy stretch: the very first ~2-3 weeks
of the earliest season in the dataset (2008-09), where there is no prior
data at all and the expanding mean is thin. Accepted as a known, harmless
edge case of a deliberately naive baseline.

Home ice and the shrinkage pseudo-games constant (PRIOR_GAMES) are explicit,
clearly-marked placeholders -- not yet empirically fit on our own data, per
the same "don't hard-code a constant we haven't checked ourselves" standard
set in MODEL_DOCUMENTATION.md Sec3.2 for the xG stabilization constants.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.shrinkage import add_walk_forward_rate

PRIOR_GAMES = 10  # shrinkage pseudo-games toward league average -- PLACEHOLDER, not yet calibrated
HOME_ICE_GOAL_MULTIPLIER = 1.06  # PLACEHOLDER -- to be replaced by an empirical estimate in validate_baseline.py


def build_team_game_log(schedule: pd.DataFrame) -> pd.DataFrame:
    """schedule: NHL API schedule (regular season, final games only) ->
    one row per team per game (each real game produces 2 rows): gameId,
    gameDate, season, team, opponent, is_home, goals_for, goals_against."""
    finals = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")].copy()
    home = pd.DataFrame({
        "gameId": finals["gameId"], "gameDate": finals["gameDate"], "season": finals["season"],
        "team": finals["homeTeamAbbrev"], "opponent": finals["awayTeamAbbrev"], "is_home": True,
        "goals_for": finals["homeScore"], "goals_against": finals["awayScore"],
    })
    away = pd.DataFrame({
        "gameId": finals["gameId"], "gameDate": finals["gameDate"], "season": finals["season"],
        "team": finals["awayTeamAbbrev"], "opponent": finals["homeTeamAbbrev"], "is_home": False,
        "goals_for": finals["awayScore"], "goals_against": finals["homeScore"],
    })
    log = pd.concat([home, away], ignore_index=True)
    return log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def add_walk_forward_strength(log: pd.DataFrame, prior_games: float = PRIOR_GAMES) -> pd.DataFrame:
    """Thin wrapper over shrinkage.add_walk_forward_rate for the raw-goals
    case (prefix "goals" -> goals_league_avg/goals_attack_rate/
    goals_defense_rate). Row 0's goals_league_avg is NaN (no prior data at
    all) -- callers must ffill/bfill with a sensible constant before using
    it, documented rather than silently patched here."""
    return add_walk_forward_rate(log, for_col="goals_for", against_col="goals_against",
                                  prior_games=prior_games, prefix="goals")


def predict_game_lambdas(home_attack: float, home_defense: float, away_attack: float, away_defense: float,
                          league_avg: float, home_ice_multiplier: float = HOME_ICE_GOAL_MULTIPLIER) -> tuple[float, float]:
    """Standard Poisson-model combine (same family used publicly for
    low-scoring-sport goal models): each team's expected goals = league
    average * (its own attack strength ratio) * (opponent's defense
    weakness ratio), with a home-ice multiplier applied to the home side
    and its reciprocal applied to the away side (keeps the adjustment
    symmetric rather than only inflating the home total)."""
    home_attack_ratio = home_attack / league_avg
    home_defense_ratio = home_defense / league_avg
    away_attack_ratio = away_attack / league_avg
    away_defense_ratio = away_defense / league_avg

    lambda_home = league_avg * home_attack_ratio * away_defense_ratio * home_ice_multiplier
    lambda_away = league_avg * away_attack_ratio * home_defense_ratio / home_ice_multiplier
    return lambda_home, lambda_away


def score_distribution(lambda_home: float, lambda_away: float, max_goals: int = 12) -> np.ndarray:
    """Full joint P(home_goals=i, away_goals=j) grid under the independent-
    Poisson assumption (Sec3.4's starting point, before any NegBin/
    correlation refinement is tested)."""
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_pmf, away_pmf)


def home_win_prob_regulation(joint: np.ndarray) -> tuple[float, float, float]:
    """Returns (P(home wins in regulation), P(away wins in regulation),
    P(tie after regulation)) from the joint score grid. NOTE: does not yet
    include the Layer 4 OT/SO sub-model (Sec3.5, not built yet) -- ties are
    reported separately here rather than silently split 50/50, so callers
    validating THIS layer aren't accidentally also validating an assumed
    OT coin-flip they didn't ask for."""
    home_win = np.tril(joint, k=-1).sum()
    away_win = np.triu(joint, k=1).sum()
    tie = np.trace(joint)
    return home_win, away_win, tie
