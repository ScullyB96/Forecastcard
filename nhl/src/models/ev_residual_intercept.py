"""Cycle 28 (Sec18.3/Sec19.2/Sec28.5): walk-forward EV-bucket intercept, the
last piece of the mean-total chain. Redirected by Sec28.5's grid finding
(HALFLIFE_GAMES was never the "shim") to be scoped as a real, independently
load-bearing term, not a small mop-up.

Three spec requirements (Sec28's build instructions), each addressed here:
(a) symmetric across both lambdas, margin-neutral by the Sec22 algebra --
    `symmetric_ev_intercept_credit` adds the IDENTICAL value to both sides,
    exactly like `rest_schedule.symmetric_b2b_bias_credit`, so margin is
    unaffected to machine precision by construction, not by a fitted
    cancellation.
(b) the tracked quantity is realized minus PRE-intercept prediction --
    `add_walk_forward_ev_residual` is built entirely from
    `validate_bias_decomposition.run_decomposition`'s own pred_ev/actual_ev
    columns (the existing, already-validated Sec18.2/Sec23.1-bug-fixed
    combine, with no intercept applied), never from a post-intercept
    residual -- avoids the feedback loop where the term would chase its own
    output.
(c) decayed trailing window, with its OWN halflife as the one constant this
    term introduces (gridded in validate_ev_intercept.py, not reused from
    HALFLIFE_GAMES) -- everything else (which residual to track, how to
    apply it) is measured/derived, not fit.
"""

import numpy as np
import pandas as pd

from src.models.validate_bias_decomposition import run_decomposition


def compute_raw_ev_residual_log(min_season: int) -> pd.DataFrame:
    """One row per gameId: `game_ev_residual` (actual_ev_total -
    pred_ev_total, pre-intercept), sorted walk-forward. Split out from the
    EWMA step so a halflife grid can share this (expensive) computation
    once instead of rebuilding the decomposition per grid cell."""
    r = run_decomposition(min_season=min_season, use_sh_term=True)
    r = r.sort_values(["season", "gameId"]).reset_index(drop=True)
    r["pred_ev_total"] = r["pred_ev_home"] + r["pred_ev_away"]
    r["actual_ev_total"] = r["actual_ev_h"] + r["actual_ev_a"]
    r["game_ev_residual"] = r["actual_ev_total"] - r["pred_ev_total"]
    return r[["gameId", "season", "game_ev_residual"]]


def add_walk_forward_ev_residual(min_season: int, halflife_games: float,
                                  raw_log: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per gameId: `ev_residual_wf`, a walk-forward (strictly-prior,
    EWMA-decayed) trailing mean of (actual_ev_total - pred_ev_total),
    measured on the EXISTING (pre-intercept) EV-bucket combine. Pooled
    league-wide (not team-specific), matching the SH-term's own
    single-league-rate construction (Sec18.4) -- the EV residual (Sec18.2)
    is a league-wide, not team-specific, finding. `raw_log`: pass a
    pre-computed `compute_raw_ev_residual_log` result to skip rebuilding
    the decomposition when grid-searching `halflife_games`."""
    r = raw_log if raw_log is not None else compute_raw_ev_residual_log(min_season)
    r = r.copy()
    r["ev_residual_wf"] = r["game_ev_residual"].shift(1).ewm(halflife=halflife_games, min_periods=1).mean().bfill()
    return r[["gameId", "season", "ev_residual_wf"]]


def symmetric_ev_intercept_credit(ev_residual_wf: pd.Series) -> pd.Series:
    """Half the trailing whole-game EV residual, added to BOTH lambda_home
    and lambda_away unconditionally -- the total shifts by the full
    residual (restoring the whole-game deficit); the margin is unaffected
    to machine precision, since both sides receive the identical value."""
    return ev_residual_wf / 2.0
