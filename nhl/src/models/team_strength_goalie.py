"""Cycle 5: a starting-goalie overlay on top of the current best team model
(situational_team_specific_pp_toi_poisson, Cycle 4) -- a specific starting
goalie's own recent performance is a well-documented large single-game
swing factor in hockey, distinct from the team's own shot-suppression
skill, and Cycles 2-4 have only moved total-goals accuracy, not the still-
untouched win-probability axis (see MODEL_DOCUMENTATION.md Sec4.5's closing
note for why this is the next candidate).

Built from src/ingest/fetch_moneypuck_goalie_games.py's per-(goalie, game)
aggregate (shots faced, xG faced, goals allowed), since MoneyPuck has no
bulk goalie GAME-log file (only season aggregates) -- see that module's
docstring for the real game-id reconstruction bug found and fixed while
building it.

Deliberately scoped as an ADDITIVE OVERLAY on the existing model's output
lambda, not a rebuild of the combine logic -- isolates "does a goalie
signal help" as the one new variable on top of an already-validated team
model, per this project's incremental discipline. A goalie's own walk-
forward-shrunk goals-saved-above-expected-per-appearance (GSAx/game) is
subtracted from the OPPONENT's predicted goals (a good goalie reduces how
much the other team scores).

Known, flagged simplification (not yet tested as its own variable): goalie
shrinkage resets at season boundaries, same convention as the team-level
models -- arguably wrong for goalies specifically (an established goalie's
skill doesn't reset over the summer the way a team's roster composition
can change), but kept consistent with the rest of this project for this
cycle rather than testing two ideas (goalie overlay + cross-season goalie
history) at once.
"""

import pandas as pd

from src.ingest.team_codes import normalize_moneypuck_team_code
from src.models.shrinkage import add_walk_forward_mean
from src.utils.paths import DATA_RAW

PRIOR_GAMES_GOALIE = 12  # Empirically calibrated (2026-07-23) -- but NOT from the split-half/
# Spearman-Brown "true talent stabilization" calculation (calibrate_goalie_prior.py), which measured
# K=~80 games. Directly grid-searched K against this project's own downstream validation metrics
# instead (the standard this project holds every constant to -- see MODEL_DOCUMENTATION.md Sec4.7)
# and found K=80 makes margin-MAE measurably WORSE (bootstrap 95% CI entirely negative) than the
# original untested placeholder of 25, even though it's the "textbook-correct" reliability-derived
# number. A real, informative tension: K=80 correctly describes how slowly a goalie's TRUE TALENT
# LEVEL stabilizes, but this overlay's whole justification (Cycle 5) was margin prediction
# specifically, which benefits from being MORE reactive to recent form than the pure-reliability
# number would suggest -- a goalie's short-term hot/cold state carries real next-game predictive
# value beyond what "how confidently do we know their true talent" alone captures. Grid search over
# K in {1,2,3,5,7,10,12,13,15,25,50,80,120,200} found margin-MAE is U-shaped with its minimum at
# K~=10-13 (essentially flat across that range); K=12 chosen from that flat region, trading a modest
# amount of total-goals-MAE and Brier (both prefer K~=50) for the margin improvement this overlay
# exists for. This constant may need revisiting if a future cycle's validation target changes.

GOALIE_ADJUSTMENT_FLOOR = 0.05  # keeps an adjusted lambda from going non-positive in extreme tails


def build_primary_goalie_per_team_game(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game): the goalie who faced the most shots for
    that team in that game (a reasonable proxy for "the goalie who mostly
    played," correctly favoring the starter over a late relief appearance
    in a pulled-goalie game), their own gsax (xg_faced - goals_allowed)
    for that specific appearance, and gameDate/season joined from the
    schedule for the walk-forward shrinkage below."""
    gg = pd.read_parquet(DATA_RAW / "moneypuck_goalie_games.parquet")
    gg["gsax"] = gg["xg_faced"] - gg["goals_allowed"]

    sched = schedule[["gameId", "gameDate", "season"]].drop_duplicates()
    merged = gg.merge(sched, on="gameId", how="inner")
    dropped = len(gg) - len(merged)
    if dropped:
        print(f"team_strength_goalie: {dropped}/{len(gg)} goalie-game rows had no schedule match, dropped")

    merged["team"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(merged["goalie_team"], merged["season"])]
    return merged.sort_values("shots_faced", ascending=False).drop_duplicates(subset=["gameId", "team"], keep="first")


def add_walk_forward_goalie_strength(log: pd.DataFrame, prior_games: float = PRIOR_GAMES_GOALIE,
                                      reset_each_season: bool = True,
                                      league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """log: one row per (team, game) with the PRIMARY goalie's own gsax --
    reuses add_walk_forward_mean with goalieIdForShot standing in for the
    usual "team" grouping key (goalie shrinkage is walk-forward per-GOALIE,
    not per-team, since a goalie can move between teams).

    `reset_each_season` (default True, Cycle 5's original behavior): whether
    a goalie's own history resets each season boundary or carries across
    the off-season -- see MODEL_DOCUMENTATION.md Sec4.10 for the test of
    turning this off.

    `league_avg_halflife_games` (Sec19, Cycle 24): BUG FOUND -- this call
    never threaded the Cycle 13 decay fix through to the goalie layer's own
    `goalie_league_avg` reference (the trailing GSAx baseline
    `goalie_relative = goalie_shrunk_mean - goalie_league_avg` is measured
    against), leaving it an infinite-memory expanding mean while every other
    league-average baseline in the pipeline has used the decayed version
    since Sec7.2/7.5 -- the same two-baselines-different-memory bug class as
    Sec7.1/7.3, now found a third time. None (default) preserves the
    original (buggy) behavior exactly until validated; see Sec19 for the
    real-data check of whether decaying it here is a real improvement."""
    log = log.rename(columns={"team": "_team_for_goalie", "goalieIdForShot": "team"})
    log = add_walk_forward_mean(log, value_col="gsax", prior_games=prior_games, prefix="goalie",
                                 reset_each_season=reset_each_season,
                                 league_avg_halflife_games=league_avg_halflife_games)
    return log.rename(columns={"team": "goalieIdForShot", "_team_for_goalie": "team"})
