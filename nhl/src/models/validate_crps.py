"""Cycle 21 (Sec16): CRPS on totals and margin, plus a totals-line market
benchmark -- both added per external review, motivated by Sec10.6.1's
finding that total-goals/margin MAE cannot reward a correct distributional-
width fix (they score the mean, not the shape), so a future dependence/
dispersion cycle closing the real Var(T)/E(T) gap (real 0.8913 vs. model
~1.02, Sec10.6.1) could show up as a complete wash under the current metric
set even if it's a genuine improvement.

Builds the CURRENT BEST model's full final recorded-score joint (regulation-
ratio-scaled independent Poisson -> OT/SO redistribution, IDENTICAL
construction to Sec14's local-transfer pipeline with delta=0 -- a pure
no-op, since neither Cycle 17's theta nor Cycle 20's delta is in
production) per game, marginalizes to P(total) and P(margin), and computes
the discrete CRPS (src/models/crps.py) for each.

Totals-line benchmark: `data/raw/sbro_odds_games.parquet` has real
`close_total` lines (14,259/14,260 games, confirmed non-null) but NOT real
over/under PRICING (checked directly against the raw SBRO source --
OpenOU/CloseOU columns are just the total number, not accompanying odds;
the two rows that show -110 in that column are a confirmed rare
data-entry artifact, not systematic pricing, 2/33,358 raw rows). Since a
real no-vig market probability can't be computed without real O/U prices,
the honest benchmark instead uses the fact that sportsbooks set the total
LINE (not just the price) specifically to sit near a genuine coin flip --
so 0.5 is a reasonable market-implied P(over) baseline, and the model's own
predicted P(actual_total > close_total) is compared against it via Brier
score on the real over/under outcome."""

import numpy as np
import pandas as pd

from src.models.crps import discrete_crps, margin_pmf_from_joint, total_pmf_from_joint
from src.models.dixon_coles import dc_adjusted_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.validate_cross_season import run_validation as run_current_best
from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
from src.utils.paths import DATA_RAW

MAX_GOALS = 12


def _build_final_joint(lambda_home_reg: float, lambda_away_reg: float, p_home_wins_ot: float,
                        max_goals: int = MAX_GOALS) -> np.ndarray:
    reg_joint = dc_adjusted_joint(lambda_home_reg, lambda_away_reg, rho=0.0, max_goals=max_goals)
    final_joint = np.zeros((max_goals + 2, max_goals + 2))
    n = reg_joint.shape[0]
    for x in range(n):
        for y in range(n):
            p = reg_joint[x, y]
            if x == y:
                final_joint[x + 1, y] += p * p_home_wins_ot
                final_joint[x, y + 1] += p * (1 - p_home_wins_ot)
            else:
                final_joint[x, y] += p
    return final_joint


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=max_season)

    base, _adj = run_current_best()
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()
    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    results = []
    for row in base.itertuples():
        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, p_home_wins_ot)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        total_pmf = total_pmf_from_joint(final_joint)
        margin_pmf, margin_offset = margin_pmf_from_joint(final_joint)

        actual_total = int(row.actual_home + row.actual_away)
        actual_margin = int(row.actual_home - row.actual_away)
        crps_total = discrete_crps(total_pmf, actual_total)
        crps_margin = discrete_crps(margin_pmf, actual_margin + margin_offset)

        cdf_total = np.cumsum(total_pmf)
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": (final_joint * xs).sum(), "lambda_away": (final_joint * ys).sum(),
            "home_win_prob": final_joint[xs > ys].sum(), "away_win_prob": final_joint[xs < ys].sum(),
            "tie_prob": 0.0,
            "crps_total": crps_total, "crps_margin": crps_margin,
            "cdf_total": cdf_total, "total_pmf": total_pmf,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    r = run_validation()
    print(f"validated {len(r)} dev-set games")
    print(f"mean CRPS (total): {r['crps_total'].mean():.5f}")
    print(f"mean CRPS (margin): {r['crps_margin'].mean():.5f}")

    row = append_run(
        r[["actual_home", "actual_away", "lambda_home", "lambda_away", "home_win_prob", "away_win_prob",
           "tie_prob"]],
        model_name="cross_season_prior_poisson_WITH_CRPS",
        config_flags={"split": "development", "cross_season_weight": 0.75, "prior_minutes_multiplier": 2.0},
        notes="Sec16: same current-best model, re-derived through the full final-joint construction "
              "(regulation-ratio + OT/SO redistribution, no tie-mass adjustment) so CRPS on totals/margin "
              "can be computed and logged as first-class metrics alongside the usual SU/Brier/MAE row.",
        extra_metrics={"crps_total": float(r["crps_total"].mean()), "crps_margin": float(r["crps_margin"].mean())},
    )
    print("\nledger row:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    # --- Totals-line market benchmark ---
    print("\n--- Totals-line benchmark ---")
    odds = pd.read_parquet(DATA_RAW / "sbro_odds_games.parquet")
    odds = odds.dropna(subset=["gameId", "close_total"])[["gameId", "close_total"]]
    merged = r.merge(odds, on="gameId", how="inner")
    print(f"matched {len(merged)} games with a real closing total line")

    def _p_over(row) -> float:
        line = row["close_total"]
        cdf = row["cdf_total"]
        # P(total > line) = 1 - P(total <= floor(line)): correct for both X.5 lines
        # (no push possible) and whole-number lines (push at total==line handled
        # separately, excluded below rather than folded into this probability).
        return float(1 - cdf[int(np.floor(line))])

    merged["model_p_over"] = merged.apply(_p_over, axis=1)
    merged["actual_total"] = merged["actual_home"] + merged["actual_away"]
    merged["is_push"] = merged["actual_total"] == merged["close_total"]
    merged["actual_over"] = (merged["actual_total"] > merged["close_total"]).astype(int)
    no_push = merged[~merged["is_push"]].copy()
    print(f"{merged['is_push'].sum()} pushes (actual total == line exactly) excluded from the Brier check, "
          f"{len(no_push)} decided games remain")

    model_brier = ((no_push["model_p_over"].clip(1e-6, 1 - 1e-6) - no_push["actual_over"]) ** 2).mean()
    market_baseline_brier = ((0.5 - no_push["actual_over"]) ** 2).mean()
    print(f"model Brier (P(over) vs. actual over/under): {model_brier:.5f}")
    print(f"market-implied baseline Brier (constant 0.5, since sportsbooks set the LINE to sit near a "
          f"coin flip rather than adjusting price): {market_baseline_brier:.5f}")
    print(f"actual over rate at the market's own line: {no_push['actual_over'].mean():.4f} "
          f"(close to 0.5 confirms the line-setting assumption)")
