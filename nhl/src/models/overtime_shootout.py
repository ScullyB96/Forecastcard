"""Architecture Layer 4 (MODEL_DOCUMENTATION.md Sec3.5): the OT/SO
tiebreaker sub-model, motivated directly by Sec4.11's finding that a naive
score-correlation term would confound genuine regulation-time score effects
with a mechanical artifact of OT/SO games always ending with matched
regulation scores.

Every prior cycle (1-8) validated win probability using
`home_win_prob / (1 - tie_prob)` as a workaround for not having a real
tiebreaker model -- explicitly flagged as a limitation since
`baseline_naive_poisson.home_win_prob_regulation`'s own docstring in
Cycle 1. This module replaces that workaround with a real one: the actual
empirical OT/SO home-ice win rate (checked, not assumed 50/50, per this
project's own standard), combined with the existing model's tie_prob to
produce a genuine unconditional win probability
`home_win_prob + tie_prob * p_home_wins_ot`.

Regulation-only scores are reconstructed from `lastPeriodType` (REG/OT/SO,
confirmed already present in the schedule endpoint -- no play-by-play
ingestion needed) and the final score: whichever team won an OT/SO-decided
game had exactly one bonus goal (the OT winner or the shootout's awarded
goal) added to its own tally, confirmed against real games in
fetch_nhl_api.py's docstring.
"""

import numpy as np
import pandas as pd


def add_regulation_score(schedule: pd.DataFrame) -> pd.DataFrame:
    """Adds reg_home_score, reg_away_score, went_to_ot (bool) to a
    schedule DataFrame with homeScore/awayScore/lastPeriodType. Includes a
    self-consistency check: for any went_to_ot game, reg_home_score must
    equal reg_away_score (both teams were genuinely tied after regulation)
    -- raises if that's ever violated, rather than silently trusting the
    reconstruction."""
    df = schedule.copy()
    df["went_to_ot"] = df["lastPeriodType"].isin(["OT", "SO"])
    home_won = df["homeScore"] > df["awayScore"]
    df["reg_home_score"] = df["homeScore"] - np.where(df["went_to_ot"] & home_won, 1, 0)
    df["reg_away_score"] = df["awayScore"] - np.where(df["went_to_ot"] & ~home_won, 1, 0)

    ot_rows = df[df["went_to_ot"]]
    mismatch = ot_rows[ot_rows["reg_home_score"] != ot_rows["reg_away_score"]]
    if len(mismatch):
        raise ValueError(f"{len(mismatch)} OT/SO games have a non-tied reconstructed regulation score -- "
                          f"the lastPeriodType/final-score reconstruction assumption is broken for these rows")
    return df


def add_walk_forward_reg_ratio(schedule_with_ot: pd.DataFrame, halflife_games: float) -> pd.DataFrame:
    """One row per game: walk-forward (strictly-prior, EWMA-decayed,
    same halflife family as every other decayed baseline in this project)
    home/away regulation ratio -- replaces the fixed dev-fit
    HOME_REG_RATIO/AWAY_REG_RATIO constants (Sec4.13) with a time-varying
    estimate (Sec21.2/Sec25), so any real drift in the regulation-vs-final
    scoring split (e.g. from OT/SO rule changes) is tracked rather than
    averaged away over the whole dev-set history."""
    df = schedule_with_ot.sort_values(["gameDate", "gameId"]).reset_index(drop=True).copy()
    for side, reg_col, final_col in (("home", "reg_home_score", "homeScore"), ("away", "reg_away_score", "awayScore")):
        trailing_reg = df[reg_col].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
        trailing_final = df[final_col].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
        df[f"{side}_reg_ratio_wf"] = (trailing_reg / trailing_final).bfill()
    return df[["gameId", "home_reg_ratio_wf", "away_reg_ratio_wf"]]


def fit_ot_home_win_rate(schedule_with_ot: pd.DataFrame, max_season_exclusive: int) -> float:
    """Empirical P(home team wins | game reached OT/SO), fit ONCE on the
    development set (seasons < max_season_exclusive) -- a single global
    constant, same practice as home_ice_multiplier and the goalie
    shrinkage prior, not walk-forward per-game (no reason to expect this
    rate drifts meaningfully game-to-game within the development window)."""
    dev = schedule_with_ot[(schedule_with_ot["gameType"] == 2) & (schedule_with_ot["went_to_ot"])
                            & (schedule_with_ot["season"] >= 20102011)
                            & (schedule_with_ot["season"] < max_season_exclusive)]
    return float((dev["homeScore"] > dev["awayScore"]).mean())


def full_win_probability(home_win_prob_reg: float, tie_prob: float, p_home_wins_ot: float) -> tuple[float, float]:
    """The genuine, unconditional (home_win, away_win) probability --
    replaces every prior cycle's `p / (1 - tie_prob)` renormalization
    workaround with an actual resolution of the tie mass."""
    home_win_total = home_win_prob_reg + tie_prob * p_home_wins_ot
    away_win_total = (1 - home_win_prob_reg - tie_prob) + tie_prob * (1 - p_home_wins_ot)
    return home_win_total, away_win_total
