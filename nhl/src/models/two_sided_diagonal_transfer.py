"""Cycle 25 (Sec21/Sec24): two-sided diagonal (tie-mass) transfer, correcting
Cycle 20's structural flaw. §21.1 proves (via the coherence identity
`final_mean = regulation_mean + P(reach OT) * 1`) that a ONE-SIDED transfer
(moving mass only from the "above-total" neighbors (x+1,x)/(x,x+1) into the
diagonal (x,x)) is total-conserving by construction: it lowers the
regulation mean by exactly the amount the OT bonus-goal re-add raises it
back by, netting exactly zero on the final mean -- which is why Sec14.2
found total-MAE identical at every tested delta.

Fix: pull mass into (x,x) from BOTH the above-total neighbors (total 2x+1)
AND the below-total neighbors (x-1,x)/(x,x-1) (total 2x-1), in a GLOBALLY
balanced proportion (not per-cell, since (0,0) has no below-neighbor and
per-cell symmetry is impossible there). Moving mass from above lowers the
regulation mean; moving mass from below raises it by the same per-unit
amount -- balancing the AGGREGATE amounts moved from each side across the
whole population keeps the regulation mean fixed on average while still
raising diagonal (tie) mass, which the OT bonus re-add then correctly
converts into a real, positively-signed increase in the FINAL mean.
"""

import numpy as np
from scipy.stats import poisson


def _indep_joint(lambda_home: float, lambda_away: float, max_goals: int = 12) -> np.ndarray:
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    return np.outer(home_pmf, away_pmf)


def diagonal_mass(joint: np.ndarray) -> float:
    return float(np.trace(joint))


def above_mass(joint: np.ndarray) -> float:
    """Total mass at (x+1,x)/(x,x+1) for all x -- the "above-total"
    neighbors of every diagonal cell (total 2x+1 vs. the diagonal's 2x)."""
    n = joint.shape[0]
    total = 0.0
    for x in range(n - 1):
        total += joint[x + 1, x] + joint[x, x + 1]
    return float(total)


def below_mass(joint: np.ndarray) -> float:
    """Total mass at (x-1,x)/(x,x-1) for all x>=1 -- the "below-total"
    neighbors (total 2x-1 vs. the diagonal's 2x). x=0 has none."""
    n = joint.shape[0]
    total = 0.0
    for x in range(1, n):
        total += joint[x - 1, x] + joint[x, x - 1]
    return float(total)


def two_sided_transfer(joint: np.ndarray, delta_above: float, delta_below: float) -> np.ndarray:
    """delta_above=delta_below=0 is a no-op. Moves `delta_above` fraction of
    each diagonal level's own above-pair mass, and `delta_below` fraction of
    each level's own below-pair mass (where it exists -- x=0 has none),
    into that level's diagonal cell. Mass is moved directly (not
    rescaled-and-renormalized), so the joint stays automatically valid for
    delta_above, delta_below in [0, 1)."""
    if not (0 <= delta_above < 1) or not (0 <= delta_below < 1):
        raise ValueError(f"delta_above={delta_above}, delta_below={delta_below} outside valid range [0, 1)")
    n = joint.shape[0]
    out = joint.copy()
    for x in range(n):
        moved = 0.0
        if x < n - 1:
            above = delta_above * (joint[x + 1, x] + joint[x, x + 1])
            out[x + 1, x] *= (1 - delta_above)
            out[x, x + 1] *= (1 - delta_above)
            moved += above
        if x >= 1:
            below = delta_below * (joint[x - 1, x] + joint[x, x - 1])
            out[x - 1, x] *= (1 - delta_below)
            out[x, x - 1] *= (1 - delta_below)
            moved += below
        out[x, x] += moved
    return out


def fit_deltas(lambda_home: np.ndarray, lambda_away: np.ndarray, reg_home: np.ndarray, reg_away: np.ndarray,
               max_goals: int = 12) -> tuple[float, float]:
    """Closed-form joint fit of (delta_above, delta_below), enforcing the
    GLOBAL mean-preservation constraint across the fitting population
    (Sec21.2/Sec24): choosing `delta_below = delta_above * mean(above_mass)
    / mean(below_mass)` makes the population-average regulation-mean shift
    exactly zero (moving mass from above reduces the mean by 1 unit per
    unit probability moved; moving mass from below raises it by the same;
    balancing the AGGREGATE amounts, not per-game amounts, is what handles
    the x=0 boundary correctly without forcing an impossible per-cell
    symmetry there). With that constraint substituted in, the average
    diagonal-mass increase becomes `2 * delta_above * mean(above_mass)`, so
    matching the real average tie rate has a direct closed-form solution
    for `delta_above` alone."""
    real_tie_rate = float((reg_home == reg_away).mean())
    diag_masses, above_masses, below_masses = [], [], []
    for lh, la in zip(lambda_home, lambda_away):
        joint = _indep_joint(lh, la, max_goals)
        diag_masses.append(diagonal_mass(joint))
        above_masses.append(above_mass(joint))
        below_masses.append(below_mass(joint))
    mean_diag = float(np.mean(diag_masses))
    mean_above = float(np.mean(above_masses))
    mean_below = float(np.mean(below_masses))

    delta_above = (real_tie_rate - mean_diag) / (2 * mean_above)
    delta_below = delta_above * mean_above / mean_below
    return delta_above, delta_below


def fit_deltas_per_game(diag_mass: np.ndarray, above_mass: np.ndarray, below_mass: np.ndarray,
                         target_diag: np.ndarray, max_delta: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Cycle 26 (Sec25.6/Sec38): per-game generalization of `fit_deltas`,
    solving each game's own (delta_above, delta_below) so ITS OWN diagonal
    mass moves toward ITS OWN target -- rather than fitting one shared
    constant against the population average (Sec25's design). Sec25.6
    flagged that a NAIVE per-game redesign -- solving each game's delta so
    its diagonal mass equals the trailing empirical OT rate directly (a
    LEVEL target) -- would repeat Cycle 17's mistake in new clothes: forcing
    every game toward the same absolute tie probability regardless of how
    lopsided or close that specific matchup is. `target_diag` should
    instead come from a walk-forward calibration RATIO applied to each
    game's own model-implied diagonal mass (walk_forward_tie_ratio.py), so
    cross-game matchup variation is preserved rather than erased.

    The SAME global-balance constraint from `fit_deltas`
    (`delta_below = delta_above * above_mass / below_mass`) is applied PER
    GAME here instead of once on population averages -- which makes the
    mean-preservation guarantee EXACT for every individual game (not just on
    average across the population, like Sec25's own design), since the
    per-cell mean-shift argument (moving mass from an above-neighbor lowers
    the regulation mean by exactly 1 unit per unit probability moved; from a
    below-neighbor raises it by exactly 1) holds identically whether applied
    once in aggregate or once per game. `max_delta` bounds both deltas to a
    valid, well-conditioned range (matching `two_sided_transfer`'s own
    `[0, 1)` validity check with margin to spare) -- clipping `delta_above`
    only ever affects how closely the target is hit, never the exact
    per-game neutrality guarantee, since `delta_below` is always DERIVED
    from whatever `delta_above` value is actually used, clipped or not."""
    diag_mass = np.asarray(diag_mass, dtype=float)
    above_mass = np.asarray(above_mass, dtype=float)
    below_mass = np.asarray(below_mass, dtype=float)
    target_diag = np.asarray(target_diag, dtype=float)

    ratio_below_above = np.divide(below_mass, above_mass, out=np.zeros_like(above_mass), where=above_mass > 0)
    delta_above_cap = max_delta * np.minimum(1.0, ratio_below_above)
    raw_delta_above = np.divide(target_diag - diag_mass, 2 * above_mass,
                                 out=np.zeros_like(above_mass), where=above_mass > 0)
    delta_above = np.clip(raw_delta_above, 0.0, delta_above_cap)
    delta_below = np.divide(delta_above * above_mass, below_mass,
                             out=np.zeros_like(below_mass), where=below_mass > 0)
    return delta_above, delta_below
