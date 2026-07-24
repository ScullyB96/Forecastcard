"""Cycle 26 (Sec25.6/Sec38): walk-forward tie-probability calibration RATIO,
replacing Sec25's single dev-fit global delta with a per-game, time-varying
target. Sec25.6 flagged that the naive per-game redesign -- solving each
game's own delta so its diagonal mass equals the trailing empirical OT rate
directly -- is a LEVEL target that repeats Cycle 17's mistake in new
clothes: it forces every game toward the SAME absolute tie probability
regardless of how lopsided or close that specific matchup is, erasing
exactly the cross-game variation an independent-Poisson joint already
encodes correctly (and which the local-transfer/margin-bin work, Sec10.6.2,
confirmed is real signal, not noise).

Fix: compute a walk-forward RATIO (trailing real OT rate / trailing
model-implied mean diagonal mass) and multiply each game's OWN modeled
diagonal mass by it, so a mismatched game's already-lower tie probability
stays proportionally lower than a coin-flip game's -- the aggregate hits the
empirical rate by construction, while matchup-dependence is preserved
rather than erased. Same epistemic class as `overtime_shootout.
add_walk_forward_reg_ratio` (Sec21.2/Sec25): a measured quantity, not a
fitted constant.
"""

import numpy as np
import pandas as pd

from src.models.two_sided_diagonal_transfer import _indep_joint, above_mass, below_mass, diagonal_mass


def add_walk_forward_ot_calibration_ratio(base: pd.DataFrame, schedule_with_ot: pd.DataFrame,
                                           halflife_games: float, max_goals: int = 12) -> pd.DataFrame:
    """base: one row per game with gameId, gameDate, lambda_home_reg,
    lambda_away_reg (already reg-ratio-scaled). Returns gameId plus this
    game's OWN model_diag_mass/above_mass/below_mass (needed downstream by
    `fit_deltas_per_game`) and `ot_calibration_ratio` -- a walk-forward,
    strictly-prior, EWMA-decayed ratio of trailing real OT rate to trailing
    model-implied mean diagonal mass, same halflife family as every other
    decayed quantity in this project."""
    df = base[["gameId", "gameDate", "lambda_home_reg", "lambda_away_reg"]].merge(
        schedule_with_ot[["gameId", "went_to_ot"]], on="gameId", how="left")
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    diag_masses, above_masses, below_masses = [], [], []
    for lh, la in zip(df["lambda_home_reg"], df["lambda_away_reg"]):
        joint = _indep_joint(lh, la, max_goals)
        diag_masses.append(diagonal_mass(joint))
        above_masses.append(above_mass(joint))
        below_masses.append(below_mass(joint))
    df["model_diag_mass"] = diag_masses
    df["above_mass"] = above_masses
    df["below_mass"] = below_masses

    trailing_real_ot = df["went_to_ot"].astype(float).shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
    trailing_model_diag = df["model_diag_mass"].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
    df["ot_calibration_ratio"] = (trailing_real_ot / trailing_model_diag).bfill()
    return df[["gameId", "model_diag_mass", "above_mass", "below_mass", "ot_calibration_ratio"]]
