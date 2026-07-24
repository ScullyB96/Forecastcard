"""Empirically calibrate PRIOR_GAMES_GOALIE (team_strength_goalie.py), which
has been a placeholder ("~10 games worth," never checked) since Cycle 5.
Uses a standard split-half reliability method (the same family of approach
behind Russell Carleton's published sabermetric stabilization points, which
the sibling MLB project's true_talent.py cites directly for its own
per-outcome constants):

1. For each (goalie, season) with enough appearances, split chronologically
   into a first half and second half; compute each half's raw (unshrunk)
   mean GSAx-per-appearance.
2. Correlate first-half means against second-half means across all
   qualifying goalie-seasons -- this is the split-half reliability at
   "half a typical goalie-season" of appearances.
3. Step up to full-season reliability via the Spearman-Brown prophecy
   formula (standard psychometric correction for combining two half-length
   measures back into a full-length one): r_full = 2r / (1 + r).
4. Solve reliability = N / (N + K) for K (the shrinkage prior, in games),
   using N = the average number of TOTAL appearances per qualifying
   goalie-season.

This is a one-time, whole-dataset descriptive calibration of a MODELING
CONSTANT (not itself a walk-forward prediction), the same category of
one-off step Carleton's own published constants were -- legitimate to
compute from the full historical dataset rather than needing its own
walk-forward split.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.models.team_strength_goalie import build_primary_goalie_per_team_game
from src.utils.paths import DATA_RAW

MIN_APPEARANCES_PER_SEASON = 20  # need enough games in each half to make the half-mean itself not pure noise


def compute_split_half_reliability() -> dict:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    log = build_primary_goalie_per_team_game(schedule)
    log = log.sort_values(["goalieIdForShot", "season", "gameDate"]).reset_index(drop=True)

    first_half_means, second_half_means, n_appearances = [], [], []
    for (_goalie, _season), grp in log.groupby(["goalieIdForShot", "season"]):
        n = len(grp)
        if n < MIN_APPEARANCES_PER_SEASON:
            continue
        half = n // 2
        first_half_means.append(grp["gsax"].iloc[:half].mean())
        second_half_means.append(grp["gsax"].iloc[half:2 * half].mean())  # equal-sized halves, drop any odd leftover
        n_appearances.append(2 * half)

    r_half, _p = pearsonr(first_half_means, second_half_means)
    r_full = 2 * r_half / (1 + r_half)  # Spearman-Brown step-up
    avg_n = float(np.mean(n_appearances))
    k = avg_n * (1 - r_full) / r_full  # solve reliability = N/(N+K) for K

    return {
        "n_qualifying_goalie_seasons": len(first_half_means),
        "avg_appearances_per_qualifying_season": avg_n,
        "split_half_correlation": r_half,
        "spearman_brown_full_reliability": r_full,
        "implied_prior_games_k": k,
    }


if __name__ == "__main__":
    result = compute_split_half_reliability()
    for k, v in result.items():
        print(f"{k}: {v}")
