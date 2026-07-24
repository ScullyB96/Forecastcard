"""Cycle 17 (re-scoped per external review, 2026-07-23, MODEL_DOCUMENTATION.md
Sec10): diagonal (tie-mass) inflation, fit BEFORE any off-diagonal
dependence term -- because a negative-dependence copula (the originally-
scoped Cycle 17, Sec4.11/4.13) moves diagonal mass DOWN, which would make
the confirmed tie-mass deficit WORSE, not better. The two need to be
correctly sequenced, not layered naively.

Motivating number (checked directly on the current best model before
building anything, per this project's own discipline): model-implied
regulation-tie probability averages 0.169 across the dev set; the real
empirical regulation-tie (went-to-OT) rate is 0.231 -- a 6.2 percentage
point deficit, confirmed to persist unchanged after the Cycle 13 drift fix
(that fix was Brier-neutral by its own bootstrap result, Sec7.5, so this
is a genuinely separate problem, not something drift happened to fix).

Design: a single multiplicative inflation parameter theta applied to the
independent-Poisson joint's own diagonal cells (each game's own diagonal
shape, which is already lambda-dependent, is preserved -- just uniformly
scaled), with the off-diagonal cells rescaled to keep the joint a valid
probability distribution. Fit via matching the AVERAGE diagonal mass across
the fitting population to the real empirical tie rate (closed-form, not an
iterative MLE, since a single global scalar has a direct algebraic
solution here).

Fitting population: regulation-only scores (Sec4.12's reconstruction) for
ALL dev-set games, ties genuinely included -- sidesteps both prior
structural failures (Sec4.11's OT-population-pooling confound, Sec4.13's
missing-cell MLE divergence), since diagonal cells are exactly the ones
this fix targets and every real game (decided or not) contributes real
information about them.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.validate_dixon_coles_regulation import HOME_REG_RATIO, AWAY_REG_RATIO
from src.models.validate_drift import run_validation as run_current_best
from src.utils.paths import DATA_RAW
from src.models.final_holdout_check import DEV_MAX_SEASON


def _indep_joint(lambda_home: float, lambda_away: float, max_goals: int = 12) -> np.ndarray:
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_pmf, away_pmf)


def diagonal_mass(joint: np.ndarray) -> float:
    return float(np.trace(joint))


def inflate_diagonal(joint: np.ndarray, theta: float) -> np.ndarray:
    """theta=1 is a no-op (identity); theta>1 inflates the diagonal (tie)
    mass, rescaling the off-diagonal to keep the joint valid. Raises if
    theta would push the diagonal mass above 1 for this specific game
    (should never happen for any theta actually fit on real data, but
    checked rather than silently clipped)."""
    n = joint.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    d = diagonal_mass(joint)
    new_d = theta * d
    if new_d >= 1.0:
        raise ValueError(f"theta={theta} would push diagonal mass to {new_d} >= 1 for this game's own lambdas")
    off_diag_scale = (1 - new_d) / (1 - d) if d < 1 else 0.0
    out = joint.copy()
    out[diag_mask] *= theta
    out[~diag_mask] *= off_diag_scale
    return out


def fit_theta(lambda_home: np.ndarray, lambda_away: np.ndarray, reg_home: np.ndarray, reg_away: np.ndarray,
              max_goals: int = 12) -> float:
    """Closed-form: theta = (real average tie rate) / (average model-implied
    diagonal mass under independent Poisson), since inflate_diagonal scales
    every game's own diagonal mass by the same theta -- the AVERAGE diagonal
    mass across the fitting population scales by theta too."""
    real_tie_rate = float((reg_home == reg_away).mean())
    model_diag_masses = [
        diagonal_mass(_indep_joint(lh, la, max_goals)) for lh, la in zip(lambda_home, lambda_away)
    ]
    mean_model_diag = float(np.mean(model_diag_masses))
    return real_tie_rate / mean_model_diag


if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base, _adj = run_current_best()
    base = base[(base["season"] >= 20102011) & (base["season"] < DEV_MAX_SEASON)]
    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    theta = fit_theta(base["lambda_home_reg"].values, base["lambda_away_reg"].values,
                       base["reg_home_score"].values, base["reg_away_score"].values)
    print(f"fitted theta={theta:.4f} on {len(base)} real dev-set games (all games, ties included)")

    real_tie_rate = (base["reg_home_score"] == base["reg_away_score"]).mean()
    mean_model_diag_before = np.mean([diagonal_mass(_indep_joint(lh, la))
                                       for lh, la in zip(base["lambda_home_reg"], base["lambda_away_reg"])])
    print(f"real tie rate: {real_tie_rate:.4f}, model-implied diagonal mass BEFORE inflation: {mean_model_diag_before:.4f}")
    print(f"model-implied diagonal mass AFTER inflation (theta * before): {theta * mean_model_diag_before:.4f}")
