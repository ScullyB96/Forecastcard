"""Cycle 10: revisits Sec4.11's score-correlation question properly, now
that regulation-only scores and a real OT/SO sub-model exist (Sec4.12).

Key fix vs Sec4.11's aborted attempt: fitting rho on ALL games' regulation-
only scores does NOT avoid the earlier confound -- checked directly, not
assumed: games that go to OT are, BY DEFINITION, exactly tied after
regulation (reg_home_score == reg_away_score always), so restricting to
"games that went to OT" and correlating their residuals is conditioning on
an equality constraint, a structural (selection) artifact that produces a
spuriously huge correlation (measured r=0.90) regardless of which target
variable (final score or regulation-only score) is used. Fitting rho must
exclude went-to-OT games entirely, not just switch to the "correct" target.
On the regulation-DECIDED subset only (games NOT going to OT, n=12,718):
real correlation r=-0.170 (p<1e-6) -- a similar magnitude to Sec4.11's
naive full-final-score estimate (-0.152), now on a cleaner target and
population.

Approach:
1. Approximate regulation-only lambda from the current model's final-score
   lambda via a simple, non-circular rescaling: the real ratio of mean
   regulation-only goals to mean final recorded goals (0.9613 home,
   0.9593 away, measured on the dev set) -- NOT the model's own tie_prob,
   which was checked and found NOT to match the real OT rate (0.168 model
   vs. 0.231 real), so using it as a stand-in would be circular.
2. Fit rho by MLE on regulation-decided games only.
3. Build the FULL predictive joint for any game (decided or not): the
   Dixon-Coles-adjusted regulation-time joint, then convolve the tied
   (diagonal) probability mass with the real OT/SO sub-model (win rate
   0.509) to get the actual RECORDED final-score joint -- which correctly
   has ~zero mass on the diagonal, matching reality (recorded scores are
   never tied).
"""

import numpy as np
import pandas as pd

from src.models.dixon_coles import dc_adjusted_joint, negative_log_likelihood
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.validate_goalie import run_validation as run_current_best
from src.utils.paths import DATA_RAW
from scipy.optimize import minimize_scalar

HOME_REG_RATIO = 0.961326847938761  # measured, dev set: mean(reg_home_score)/mean(homeScore)
AWAY_REG_RATIO = 0.9592636967650583


def _build_final_joint(lambda_home_reg: float, lambda_away_reg: float, rho: float,
                        p_home_wins_ot: float, max_goals: int = 12) -> np.ndarray:
    """Full RECORDED final-score joint: Dixon-Coles regulation joint with
    all diagonal (tied) mass reallocated to (x+1,x) or (x,x+1) via the
    real OT/SO win rate -- matches reality (no ties in recorded scores)."""
    reg_joint = dc_adjusted_joint(lambda_home_reg, lambda_away_reg, rho, max_goals)
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


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON):
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=max_season)

    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score", "went_to_ot"]]
    base = run_current_best()
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].merge(reg_cols, on="gameId", how="left")

    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    fit_set = base[~base["went_to_ot"]]
    result = minimize_scalar(
        negative_log_likelihood, bounds=(-0.3, 0.3), method="bounded",
        args=(fit_set["lambda_home_reg"].values, fit_set["lambda_away_reg"].values,
              fit_set["reg_home_score"].values, fit_set["reg_away_score"].values),
    )
    rho = float(result.x)

    results = []
    for row in base.itertuples():
        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, rho, p_home_wins_ot)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        home_win_p = final_joint[xs > ys].sum()
        exp_total = (final_joint * (xs + ys)).sum()
        exp_margin = (final_joint * (xs - ys)).sum()
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "home_win_prob_full": home_win_p, "exp_total": exp_total, "exp_margin": exp_margin,
        })
    return pd.DataFrame(results), rho, p_home_wins_ot


if __name__ == "__main__":
    r, rho, p_home_wins_ot = run_validation()
    print(f"validated {len(r)} dev-set games, fitted rho={rho:.5f}, p_home_wins_ot={p_home_wins_ot:.4f}")

    actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int)
    p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
    brier = ((p - actual_home_win) ** 2).mean()
    actual_total = r["actual_home"] + r["actual_away"]
    actual_margin = r["actual_home"] - r["actual_away"]
    total_mae = (r["exp_total"] - actual_total).abs().mean()
    margin_mae = (r["exp_margin"] - actual_margin).abs().mean()
    print(f"Brier={brier:.5f}, total_mae={total_mae:.5f}, margin_mae={margin_mae:.5f}")
