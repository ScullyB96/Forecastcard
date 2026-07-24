"""Cycle 18 (§11/§12): cross-season team-strength prior, on top of the full
current-best pipeline (situational strength + goalie overlay + rest + real
OT/SO resolution + the Cycle 13 drift-decay fix). Motivated by a real,
measured effect (Sec12.1): a team's own last-season xG-differential
correlates with its early-season (games 1-15) performance at r=0.15
(p<1e-40) -- signal the current season-reset prior discards entirely at
game 1 of every season, and the likely explanation for Sec11's front-loaded
model-vs-market Brier gap (2.8x larger in the first 15 games of a season
than after 41+, 8.7x in the two disrupted-offseason seasons).

Grid-searches `cross_season_weight` (0 = off, exact reproduction of
drift_adjusted_poisson) against the current best model, holding
HALFLIFE_GAMES and PRIOR_MINUTES_* fixed first -- per Sec7.5's own lesson
that priors and any new carryover/decay term are entangled, a joint re-grid
with PRIOR_MINUTES is the natural follow-up if a real effect is found here,
not assumed away.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_rest import run_validation as run_full_chain
from src.utils.paths import DATA_RAW

CROSS_SEASON_WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _per_game_metrics(df: pd.DataFrame) -> dict:
    actual_home_win = (df["actual_home"] > df["actual_away"]).astype(int).values
    p = df["home_win_prob_full"].clip(1e-6, 1 - 1e-6).values
    su_correct = ((p > 0.5).astype(int) == actual_home_win).astype(float)
    brier = (p - actual_home_win) ** 2
    total_ae = ((df["lambda_home"] + df["lambda_away"]) - (df["actual_home"] + df["actual_away"])).abs().values
    margin_ae = ((df["lambda_home"] - df["lambda_away"]) - (df["actual_home"] - df["actual_away"])).abs().values
    return {"su": su_correct, "brier": brier, "total_ae": total_ae, "margin_ae": margin_ae}


def paired_bootstrap(base: pd.DataFrame, treated: pd.DataFrame, n_resamples: int = 5000,
                      seed: int = 20260723) -> dict:
    base_m = _per_game_metrics(base)
    treated_m = _per_game_metrics(treated)
    n = len(base)
    rng = np.random.default_rng(seed)
    idx_draws = rng.integers(0, n, size=(n_resamples, n))
    out = {}
    for key in ("su", "brier", "total_ae", "margin_ae"):
        diff = treated_m[key] - base_m[key]
        boot_means = diff[idx_draws].mean(axis=1)
        out[key] = {"mean_diff": float(diff.mean()), "ci_low": float(np.percentile(boot_means, 2.5)),
                    "ci_high": float(np.percentile(boot_means, 97.5))}
    return out


if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]

    print(f"grid-searching cross_season_weight over {CROSS_SEASON_WEIGHT_GRID}, "
          f"HALFLIFE_GAMES={HALFLIFE_GAMES} held fixed\n")

    runs = {}
    for w in CROSS_SEASON_WEIGHT_GRID:
        r, _adj = run_full_chain(league_avg_halflife_games=HALFLIFE_GAMES,
                                  cross_season_weight=(None if w == 0.0 else w))
        r = r[r["season"] < DEV_MAX_SEASON].reset_index(drop=True)
        runs[w] = r
        m = _per_game_metrics(r)
        print(f"weight={w:.2f}: n={len(r)} SU={m['su'].mean():.4f} Brier={m['brier'].mean():.5f} "
              f"total_mae={m['total_ae'].mean():.5f} margin_mae={m['margin_ae'].mean():.5f}")

    baseline = runs[0.0]
    print("\npaired bootstrap (5000 resamples) vs weight=0.0 (== drift_adjusted_poisson):")
    for w in CROSS_SEASON_WEIGHT_GRID[1:]:
        boot = paired_bootstrap(baseline, runs[w])
        print(f"\n  weight={w:.2f}:")
        for key in ("su", "brier", "total_ae", "margin_ae"):
            b = boot[key]
            print(f"    {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")
