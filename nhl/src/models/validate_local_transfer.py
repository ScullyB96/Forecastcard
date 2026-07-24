"""Validates the local-transfer tie-mass fix (local_transfer_inflation.py,
Cycle 20 / Sec14) on top of the CURRENT best model (`cross_season_prior_poisson`,
Sec12/Sec13) -- not the superseded `drift_adjusted_poisson` Cycle 17 was
built against. Builds the final RECORDED-score joint (regulation, local
transfer applied, then convolved with the real OT/SO layer -- same
construction as Cycle 17's own pipeline, with the diagonal fix swapped for
the residuals-confirmed local-transfer shape) and validates SU/Brier/
total-MAE/margin-MAE against the current best, plus the Var(T)/E(T)
consistency check, computed as a proper population-level statistic (law of
total variance) from the start this time (Sec10.6.1 caught this being done
wrong for Cycle 17 only after the fact).
"""

import numpy as np
import pandas as pd

from src.models.local_transfer_inflation import fit_delta, local_transfer
from src.models.dixon_coles import dc_adjusted_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.validate_dixon_coles_regulation import HOME_REG_RATIO, AWAY_REG_RATIO
from src.models.validate_cross_season import run_validation as run_current_best
from src.utils.paths import DATA_RAW


def _build_final_joint(lambda_home_reg: float, lambda_away_reg: float, delta: float,
                        p_home_wins_ot: float, max_goals: int = 12) -> np.ndarray:
    reg_joint = dc_adjusted_joint(lambda_home_reg, lambda_away_reg, rho=0.0, max_goals=max_goals)
    reg_joint = local_transfer(reg_joint, delta)
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


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON,
                    delta_override: float | None = None):
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

    if delta_override is not None:
        delta = delta_override
    else:
        delta = fit_delta(base["lambda_home_reg"].values, base["lambda_away_reg"].values,
                           base["reg_home_score"].values, base["reg_away_score"].values)

    results = []
    for row in base.itertuples():
        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, delta, p_home_wins_ot)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        home_win_p = final_joint[xs > ys].sum()
        exp_home = (final_joint * xs).sum()
        exp_away = (final_joint * ys).sum()
        exp_total = (final_joint * (xs + ys)).sum()
        var_total = (final_joint * (xs + ys - exp_total) ** 2).sum()
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": exp_home, "lambda_away": exp_away,
            "home_win_prob_full": home_win_p,
            "var_total": var_total, "exp_total": exp_total,
        })
    treated = pd.DataFrame(results)
    return base.reset_index(drop=True), treated, delta, p_home_wins_ot


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
    base, r, delta, p_home_wins_ot = run_validation()
    print(f"validated {len(r)} dev-set games, delta={delta:.4f}, p_home_wins_ot={p_home_wins_ot:.4f}")

    for label, df in [("BASELINE (cross_season_prior_poisson, production)", base),
                       ("LOCAL-TRANSFER (candidate)", r)]:
        actual_home_win = (df["actual_home"] > df["actual_away"]).astype(int)
        p = df["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
        su = ((p > 0.5).astype(int) == actual_home_win).mean()
        brier = ((p - actual_home_win) ** 2).mean()
        total_mae = ((df["lambda_home"] + df["lambda_away"]) - (df["actual_home"] + df["actual_away"])).abs().mean()
        margin_mae = ((df["lambda_home"] - df["lambda_away"]) - (df["actual_home"] - df["actual_away"])).abs().mean()
        print(f"[{label}] SU={su:.4f} Brier={brier:.5f} total_mae={total_mae:.5f} margin_mae={margin_mae:.5f}")

    print("\npaired bootstrap (5000 resamples), local-transfer MINUS baseline:")
    boot = paired_bootstrap(base, r)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b = boot[key]
        print(f"  {key}: mean_diff={b['mean_diff']:.5f}, 95% CI [{b['ci_low']:.5f}, {b['ci_high']:.5f}]")

    actual_total = r["actual_home"] + r["actual_away"]
    real_var_over_mean = actual_total.var(ddof=0) / actual_total.mean()
    pop_var_total = r["var_total"].mean() + r["exp_total"].var(ddof=0)
    pop_exp_total = r["exp_total"].mean()
    print(f"\nVar(T)/E(T) (population-level, law of total variance): "
          f"model={pop_var_total/pop_exp_total:.4f} vs. real={real_var_over_mean:.4f}")
