"""QB passing statline projections: completions, passing yards, passing TDs,
interceptions -- given a projected attempts volume for that game.

Reuses TdRateEngine (Bayesian/Laplace shrinkage, already used elsewhere in
this project for TD rate, yards/target, yards/carry) rather than introducing
a new mechanism. Prior weights were chosen via an out-of-sample sweep
(TRAIN=2018-2021, TEST=2022-2025, QB-weeks with >=5 attempts to exclude
mop-up/trick-play appearances):

  completions:    prior_weight=300  (MAE 2.371 vs naive 2.424, ~2.2% better)
  passing_yards:  prior_weight=150  (MAE 43.838 vs naive 44.710, ~2.0% better)
  passing_tds:    prior_weight=150  (MAE 0.860 vs naive 0.868, ~0.9% better)
  interceptions:  prior_weight=300  (MAE 0.663 vs naive 0.664, ~0.15% better)

Honest framing: completions/yards carry the same weak-but-real signal
already documented for receiving/rushing yards in player_usage.py. TD rate is
weaker still. Interceptions are close to pure noise per-QB -- consistent
with turnover luck being non-predictive everywhere else it was tested this
project -- so the interception projection is really "close to league
average," not a differentiated skill estimate. Still included because a
complete statline should show it, with that caveat understood.
"""

import pandas as pd

from src.models.player_usage import TdRateEngine

COMPLETION_PRIOR_WEIGHT = 300.0
YARDS_PRIOR_WEIGHT = 150.0
TD_PRIOR_WEIGHT = 150.0
INT_PRIOR_WEIGHT = 300.0


def build_qb_passing_engines(qb_weeks: pd.DataFrame) -> dict:
    """qb_weeks: player-week rows (any attempts > 0 -- the engines update on
    real participation the same way TD/yards engines do elsewhere)."""
    qb_weeks = qb_weeks[qb_weeks["attempts"] > 0]
    league_comp = qb_weeks["completions"].sum() / qb_weeks["attempts"].sum()
    league_ypa = qb_weeks["passing_yards"].sum() / qb_weeks["attempts"].sum()
    league_td = qb_weeks["passing_tds"].sum() / qb_weeks["attempts"].sum()
    league_int = qb_weeks["interceptions"].sum() / qb_weeks["attempts"].sum()

    comp_engine = TdRateEngine(league_rate=league_comp, prior_weight=COMPLETION_PRIOR_WEIGHT)
    comp_engine.run_walk_forward(qb_weeks, "completions", "attempts")
    ypa_engine = TdRateEngine(league_rate=league_ypa, prior_weight=YARDS_PRIOR_WEIGHT)
    ypa_engine.run_walk_forward(qb_weeks, "passing_yards", "attempts")
    td_engine = TdRateEngine(league_rate=league_td, prior_weight=TD_PRIOR_WEIGHT)
    td_engine.run_walk_forward(qb_weeks, "passing_tds", "attempts")
    int_engine = TdRateEngine(league_rate=league_int, prior_weight=INT_PRIOR_WEIGHT)
    int_engine.run_walk_forward(qb_weeks, "interceptions", "attempts")

    return {
        "completions": comp_engine,
        "passing_yards": ypa_engine,
        "passing_tds": td_engine,
        "interceptions": int_engine,
    }


def presumed_starter_qb_id(
    team: str,
    current_roster: pd.DataFrame,
    qb_weeks: pd.DataFrame,
    override_name: str | None = None,
    override_name_to_id: dict | None = None,
) -> str | None:
    """Best-effort identification of a team's current starting QB: whoever
    on the current roster had the most real attempts in the most recent
    season on record for that team. override_name/override_name_to_id let a
    caller force a known offseason swap (this project's Clay-PDF Week-1
    bootstrap, see weekly_update.py) to win over pure history -- since a
    brand-new starter has no attempts for this team yet to rank by."""
    if override_name and override_name_to_id:
        qid = override_name_to_id.get(override_name)
        if qid is not None:
            return qid
    roster_qbs = current_roster[current_roster["position"] == "QB"]
    if roster_qbs.empty:
        return None
    recent = qb_weeks[qb_weeks["recent_team"] == team]
    if recent.empty:
        return roster_qbs.iloc[0]["player_id"]
    latest_season = recent["season"].max()
    totals = recent[recent["season"] == latest_season].groupby("player_id")["attempts"].sum()
    on_roster = totals[totals.index.isin(roster_qbs["player_id"])]
    if on_roster.empty:
        return roster_qbs.iloc[0]["player_id"]
    return on_roster.idxmax()


def project_passing_stats(qb_id: str, proj_attempts: float, engines: dict) -> dict:
    return {
        "proj_pass_attempts": round(proj_attempts, 1),
        "proj_completions": round(proj_attempts * engines["completions"].predict(qb_id), 1),
        "proj_passing_yards": round(proj_attempts * engines["passing_yards"].predict(qb_id), 1),
        "proj_passing_tds": round(proj_attempts * engines["passing_tds"].predict(qb_id), 2),
        "proj_interceptions": round(proj_attempts * engines["interceptions"].predict(qb_id), 2),
    }
