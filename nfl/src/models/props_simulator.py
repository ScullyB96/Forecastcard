"""Compound Monte Carlo props simulator (Phase 3.2, review #2.2).

The review's original spec called for a parametric generative chain (plays ~
NegBinomial, target vector ~ Dirichlet-Multinomial(attempts, concentration *
share_vector), yards|reception ~ Gamma/lognormal, TDs ~ Poisson-binomial).
Before building that, tested its core premise directly against real data:
is a real player's week-to-week target count uniformly overdispersed relative
to pure multinomial sampling (which is what a single Dirichlet-Multinomial
concentration parameter models)? It is NOT. Standardized residuals
((actual - predicted) / multinomial_sd) have mean=0.38, std=1.71 (vs the ~0/1
a well-calibrated multinomial model would show), a 95th percentile of 2.96
and a 99th of 6.0 -- a real, heavy right tail, and this holds *within* every
usage tier (fringe/role-player/primary), not just as an artifact of mixing
them. The data-generating process looks like a mixture (mostly-tight weeks
plus a persistent minority of large-surprise weeks -- almost certainly real
injury-driven role changes, blowout game scripts, coaching decisions the
share engine's EWMA can't anticipate), not a single-parameter Dirichlet-
Multinomial.

Rather than assume a parametric mixture shape (Beta-Binomial mixture,
contaminated-normal, etc. -- more unverified assumptions), this reuses the
exact successful pattern from Phase 2's game_simulator.py: bootstrap-resample
the REAL empirical distribution of standardized residuals (stratified by
usage tier) instead of parametrically modeling it. This reproduces whatever
the real shape is -- skew, heavy tails, everything -- without assuming a
specific family.
"""

import numpy as np
import pandas as pd

USAGE_TIER_EDGES = [0.0, 0.05, 0.15, 1.0]
USAGE_TIER_LABELS = ["fringe", "role_player", "primary"]
MIN_RESID_POOL = 200


def usage_tier(share: float) -> str:
    for lo, hi, label in zip(USAGE_TIER_EDGES[:-1], USAGE_TIER_EDGES[1:], USAGE_TIER_LABELS):
        if lo <= share < hi:
            return label
    return USAGE_TIER_LABELS[-1]


def build_standardized_residuals(
    share_result: pd.DataFrame, share_col: str, team_touch_col: str, actual_touch_col: str
) -> pd.DataFrame:
    """share_result: walk-forward ShareEngine output, already merged with the
    team's real total touches that week (e.g. team_targets). Returns one row
    per involved player-week with a `standardized_resid` and `usage_tier`."""
    df = share_result[share_result[team_touch_col] > 0].copy()
    df["predicted_touches"] = df[f"pregame_{share_col}"] * df[team_touch_col]
    df["multinomial_sd"] = np.sqrt(
        df[team_touch_col] * df[f"pregame_{share_col}"] * (1 - df[f"pregame_{share_col}"])
    ).clip(lower=0.5)
    df["standardized_resid"] = (df[actual_touch_col] - df["predicted_touches"]) / df["multinomial_sd"]
    df["usage_tier"] = df[f"pregame_{share_col}"].apply(usage_tier)
    return df


class TouchCountSimulator:
    """Bootstrap-resample a real historical standardized residual (matching
    the player's own usage tier), apply it to this week's prediction to draw
    a simulated touch count. Falls back to the pooled (all-tier) residual
    distribution if a tier's pool is too small."""

    def __init__(self, residual_table: pd.DataFrame, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._pools = {tier: g["standardized_resid"].to_numpy() for tier, g in residual_table.groupby("usage_tier")}
        self._all = residual_table["standardized_resid"].to_numpy()

    def simulate(self, predicted_share: float, team_touches: float, n_trials: int = 1000) -> np.ndarray:
        tier = usage_tier(predicted_share)
        pool = self._pools.get(tier)
        if pool is None or len(pool) < MIN_RESID_POOL:
            pool = self._all
        sampled_resid = pool[self.rng.integers(0, len(pool), size=n_trials)]
        predicted_touches = predicted_share * team_touches
        multinomial_sd = max(np.sqrt(team_touches * predicted_share * (1 - predicted_share)), 0.5)
        simulated = predicted_touches + sampled_resid * multinomial_sd
        return np.clip(np.round(simulated), 0, team_touches)
