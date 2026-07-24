"""Cycle 19 (§13): joint grid search of `cross_season_weight` against
`prior_minutes_multiplier`, per §7.5's own lesson that priors and any new
carryover/decay term are entangled -- Cycle 18 (§12) adopted
cross_season_weight=0.5 with PRIOR_MINUTES_* held fixed at their Cycle 13
values, which may leave real gains unclaimed if the right prior-strength
scale changed now that the prior itself carries real cross-season signal
instead of being pure league average.

HALFLIFE_GAMES=600 held fixed throughout (not re-entangled here -- Cycle 13
already jointly resolved halflife vs. prior scale; this cycle only tests
whether ADDING cross-season carryover shifts the right prior scale further).
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_rest import run_validation as run_full_chain

CROSS_SEASON_WEIGHT_GRID = [0.25, 0.5, 0.75]
PRIOR_MINUTES_MULTIPLIER_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]


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
    print(f"joint grid: cross_season_weight x {CROSS_SEASON_WEIGHT_GRID}, "
          f"prior_minutes_multiplier x {PRIOR_MINUTES_MULTIPLIER_GRID}, HALFLIFE_GAMES={HALFLIFE_GAMES} fixed\n")

    current_best, _adj = run_full_chain(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=0.5,
                                         prior_minutes_multiplier=1.0)
    current_best = current_best[current_best["season"] < DEV_MAX_SEASON].reset_index(drop=True)
    m0 = _per_game_metrics(current_best)
    print(f"CURRENT BEST (weight=0.5, multiplier=1.0): SU={m0['su'].mean():.4f} Brier={m0['brier'].mean():.5f} "
          f"total_mae={m0['total_ae'].mean():.5f} margin_mae={m0['margin_ae'].mean():.5f}\n")

    runs = {}
    for w in CROSS_SEASON_WEIGHT_GRID:
        for mult in PRIOR_MINUTES_MULTIPLIER_GRID:
            r, _adj = run_full_chain(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=w,
                                      prior_minutes_multiplier=mult)
            r = r[r["season"] < DEV_MAX_SEASON].reset_index(drop=True)
            runs[(w, mult)] = r
            m = _per_game_metrics(r)
            print(f"weight={w:.2f} mult={mult:.2f}: SU={m['su'].mean():.4f} Brier={m['brier'].mean():.5f} "
                  f"total_mae={m['total_ae'].mean():.5f} margin_mae={m['margin_ae'].mean():.5f}")

    # Rank candidates by Brier (primary tie-break metric this project has used throughout).
    best_key = min(runs, key=lambda k: _per_game_metrics(runs[k])["brier"].mean())
    print(f"\nbest by dev-set Brier: weight={best_key[0]}, multiplier={best_key[1]}")

    print(f"\npaired bootstrap (5000 resamples) vs CURRENT BEST (weight=0.5, multiplier=1.0):")
    for key in sorted(runs, key=lambda k: _per_game_metrics(runs[k])["brier"].mean())[:5]:
        boot = paired_bootstrap(current_best, runs[key])
        print(f"\n  weight={key[0]}, multiplier={key[1]}:")
        for metric in ("su", "brier", "total_ae", "margin_ae"):
            b = boot[metric]
            print(f"    {metric}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")
