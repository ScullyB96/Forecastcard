"""Re-run of Sec6.8's market benchmark against the CURRENT best model
(`walk_forward_tie_ratio_poisson`, Sec26/Sec35, as of 2026-07-24 -- this
script always targets whichever model is current best, updating the import
here each time it changes, same pattern as Sec6.8's own history:
`rest_adjusted_poisson` -> `drift_adjusted_poisson` -> `cross_season_
prior_poisson` -> `walk_forward_tie_ratio_poisson` (Sec32's `ev_toi_
halflife_poisson` was adopted then REVERTED under a corrected holdout
check, Sec35 -- never became this script's target) -> briefly the GBM
stack (Sec34/Sec36 -- ADOPTED then REVERTED once its own adoption bootstrap
turned out to be scored in-sample too, Sec36) -> back to the base model.

Targets `validate_tie_mass_ratio.run_treated` DIRECTLY, no stacking layer
of any kind -- matching `validate_market_totals_benchmark.py`'s own
long-standing pattern. The GBM's brief tenure as this script's target
(Sec35.9-35.10) is kept there as the historical record of an in-sample
contamination mistake and its OOF-scored correction; that correction
(gap=0.00257) turned out to be statistically indistinguishable from this
number anyway, since the GBM itself was later found to add no real signal
(Sec36) and was reverted -- there is no stacking-layer risk left to guard
against here.

Pre-registered expectation: essentially identical to Sec31.1's original
pre-GBM figure (0.00247), since the base model itself (Sec26/Sec35.4) is
unchanged from that measurement.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_tie_mass_ratio import run_treated as run_current_best
from src.utils.paths import DATA_RAW


def paired_bootstrap_1d(diff: np.ndarray, n_resamples: int = 5000, seed: int = 20260723) -> dict:
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx_draws = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx_draws].mean(axis=1)
    return {
        "mean": float(diff.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
    }


def run_validation() -> pd.DataFrame:
    odds = pd.read_parquet(DATA_RAW / "sbro_odds_games.parquet")
    odds = odds.dropna(subset=["gameId", "close_home_prob_novig"])
    odds = odds[["gameId", "close_home_prob_novig", "open_home_prob_novig"]]

    base = run_current_best(min_season=20102011, max_season=DEV_MAX_SEASON)
    merged = base.merge(odds, on="gameId", how="inner")
    return merged


if __name__ == "__main__":
    r = run_validation()
    print(f"matched {len(r)} real games with both a walk_forward_tie_ratio_poisson prediction and a real "
          f"closing-line market probability")

    actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int).values
    model_p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6).values
    market_close_p = r["close_home_prob_novig"].clip(1e-6, 1 - 1e-6).values
    market_open_p = r["open_home_prob_novig"].clip(1e-6, 1 - 1e-6).values

    model_brier = (model_p - actual_home_win) ** 2
    market_close_brier = (market_close_p - actual_home_win) ** 2
    market_open_brier = (market_open_p - actual_home_win) ** 2

    print(f"model Brier: {model_brier.mean():.5f}")
    print(f"market CLOSE Brier: {market_close_brier.mean():.5f}")
    print(f"market OPEN Brier: {market_open_brier.mean():.5f}")

    gap = model_brier - market_close_brier
    boot = paired_bootstrap_1d(gap)
    print(f"\nmodel-minus-market-close Brier gap: mean={boot['mean']:.5f}, "
          f"95% CI [{boot['ci_low']:.5f}, {boot['ci_high']:.5f}]")

    print("\nper-season breakdown (model Brier, market-close Brier, gap, n games):")
    r = r.copy()
    r["_model_brier"] = model_brier
    r["_market_brier"] = market_close_brier
    r["_gap"] = gap
    by_season = r.groupby("season").agg(
        n=("gameId", "count"),
        model_brier=("_model_brier", "mean"),
        market_brier=("_market_brier", "mean"),
        gap=("_gap", "mean"),
    )
    for season, row in by_season.iterrows():
        print(f"  {season}: n={int(row['n']):>5}  model={row['model_brier']:.5f}  "
              f"market={row['market_brier']:.5f}  gap={row['gap']:+.5f}")
