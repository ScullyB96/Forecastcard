"""Companion to validate_market_benchmark.py (moneyline): re-runs the
totals-line P(over) benchmark (Sec16.2/Sec26.6's construction) on the SAME
real-odds-matched game set against `walk_forward_tie_ratio_poisson`
(Sec26/Sec35) -- correctly still the target here even though current best
overall is `gbm_stack_poisson` (Sec34/Sec35.7), because the GBM stack only
replaces win probability; totals/margin/`cdf_total` are untouched and still
come directly from the Poisson base, which the GBM has no equivalent for.
Pre-registered expectation (Sec30): mid-40s%, not ~49.5% -- the odds match
has no 2022-23+ coverage (Sec6.1), so this window (13 seasons through
2021-22) contains both known real scoring-level jumps and cannot reflect
the stationary-era calibration Sec18-26 actually achieved.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_tie_mass_ratio import run_treated
from src.utils.paths import DATA_RAW


def run_validation() -> pd.DataFrame:
    odds = pd.read_parquet(DATA_RAW / "sbro_odds_games.parquet")
    odds = odds.dropna(subset=["gameId", "close_total"])[["gameId", "close_total"]]
    base = run_treated(min_season=20102011, max_season=DEV_MAX_SEASON)
    merged = base.merge(odds, on="gameId", how="inner")
    return merged


if __name__ == "__main__":
    r = run_validation()
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["is_push"] = r["actual_total"] == r["close_total"]
    r["actual_over"] = (r["actual_total"] > r["close_total"]).astype(int)

    def _p_over(row) -> float:
        return float(1 - row["cdf_total"][int(np.floor(row["close_total"]))])

    r["model_p_over"] = r.apply(_p_over, axis=1)
    no_push = r[~r["is_push"]].copy()
    brier = ((no_push["model_p_over"].clip(1e-6, 1 - 1e-6) - no_push["actual_over"]) ** 2).mean()
    market_baseline_brier = ((0.5 - no_push["actual_over"]) ** 2).mean()

    print(f"matched {len(r)} real games with a closing total line ({r['is_push'].sum()} pushes excluded, "
          f"{len(no_push)} decided)")
    print(f"mean P(over): {no_push['model_p_over'].mean():.4f}  (pre-registered expectation, Sec30: mid-40s%)")
    print(f"model Brier (P(over) vs. actual over/under): {brier:.5f}")
    print(f"market-implied baseline Brier (constant 0.5): {market_baseline_brier:.5f}")
    print(f"actual over rate at the market's own line: {no_push['actual_over'].mean():.4f}")

    print("\nper-season breakdown:")
    by_season = no_push.groupby("season").agg(
        n=("gameId", "count"), mean_p_over=("model_p_over", "mean"), actual_over_rate=("actual_over", "mean"))
    for season, row in by_season.iterrows():
        print(f"  {season}: n={int(row['n']):>5}  mean_p_over={row['mean_p_over']:.4f}  "
              f"actual_over_rate={row['actual_over_rate']:.4f}")
