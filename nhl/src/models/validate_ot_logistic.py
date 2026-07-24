"""Validates the team-specific OT win-probability model (Sec17/ot_logistic.py)
on top of the CURRENT best model (`cross_season_prior_poisson`). Replaces
the flat, single-constant tie resolution (`p_home_wins_ot` applied to every
game regardless of team strength) with a per-game resolution:

    P(home wins the tie) = p_decided_in_ot * sigmoid(a + b*lambda_diff_reg)
                            + p_needs_so * p_home_wins_so_flat

where `p_decided_in_ot`/`p_needs_so`/`p_home_wins_so_flat` are fixed
empirical constants (Sec17's residuals-first check found no real
team-strength signal in shootouts) and the logistic term is fit only on
real OT-decided games.
"""

import numpy as np
import pandas as pd

from src.models.dixon_coles import dc_adjusted_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.validate_cross_season import run_validation as run_current_best
from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
from src.utils.paths import DATA_RAW

MAX_GOALS = 12


def _build_final_joint(lambda_home_reg: float, lambda_away_reg: float, p_home_wins_tie: float,
                        max_goals: int = MAX_GOALS) -> np.ndarray:
    reg_joint = dc_adjusted_joint(lambda_home_reg, lambda_away_reg, rho=0.0, max_goals=max_goals)
    final_joint = np.zeros((max_goals + 2, max_goals + 2))
    n = reg_joint.shape[0]
    for x in range(n):
        for y in range(n):
            p = reg_joint[x, y]
            if x == y:
                final_joint[x + 1, y] += p * p_home_wins_tie
                final_joint[x, y + 1] += p * (1 - p_home_wins_tie)
            else:
                final_joint[x, y] += p
    return final_joint


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON,
                    use_team_specific: bool = True):
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base, _adj = run_current_best()
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()
    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    # Fit constants on the SAME dev-set range used elsewhere (season < max_season).
    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=max_season)
    dev_merged = base.merge(schedule_ot[["gameId", "went_to_ot", "lastPeriodType", "homeScore", "awayScore"]],
                             on="gameId", how="left")
    ot_only = dev_merged[dev_merged["lastPeriodType"] == "OT"].copy()
    ot_only["lambda_diff_reg"] = ot_only["lambda_home_reg"] - ot_only["lambda_away_reg"]
    ot_only["home_won"] = (ot_only["homeScore"] > ot_only["awayScore"]).astype(int)
    a, b = fit_ot_logistic(ot_only["lambda_diff_reg"].values, ot_only["home_won"].values)

    flat_p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=max_season)

    results = []
    for row in base.itertuples():
        lambda_diff_reg = row.lambda_home_reg - row.lambda_away_reg
        if use_team_specific:
            p_ot = predict_ot_logistic_prob(a, b, lambda_diff_reg)
            p_home_wins_tie = ot_split["p_decided_in_ot"] * p_ot + \
                ot_split["p_needs_so"] * ot_split["p_home_wins_so_flat"]
        else:
            p_home_wins_tie = flat_p_home_wins_ot

        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, p_home_wins_tie)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        home_win_p = final_joint[xs > ys].sum()
        exp_home = (final_joint * xs).sum()
        exp_away = (final_joint * ys).sum()
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": exp_home, "lambda_away": exp_away,
            "home_win_prob_full": home_win_p,
        })
    return pd.DataFrame(results), (a, b), ot_split, flat_p_home_wins_ot


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
    baseline, _, _, flat_rate = run_validation(use_team_specific=False)
    treated, (a, b), ot_split, _ = run_validation(use_team_specific=True)
    print(f"fitted OT logistic: a={a:.4f}, b={b:.4f} (n={len(treated)} dev games)")
    print(f"OT/SO split: p_decided_in_ot={ot_split['p_decided_in_ot']:.4f}, "
          f"p_needs_so={ot_split['p_needs_so']:.4f}, p_home_wins_so_flat={ot_split['p_home_wins_so_flat']:.4f}")
    print(f"flat p_home_wins_ot (production, all OT/SO pooled): {flat_rate:.4f}")

    for label, df in [("BASELINE (flat OT/SO rate)", baseline), ("TEAM-SPECIFIC OT logistic", treated)]:
        m = _per_game_metrics(df)
        print(f"[{label}] SU={m['su'].mean():.4f} Brier={m['brier'].mean():.5f} "
              f"total_mae={m['total_ae'].mean():.5f} margin_mae={m['margin_ae'].mean():.5f}")

    print("\npaired bootstrap (5000 resamples), team-specific MINUS baseline:")
    boot = paired_bootstrap(baseline, treated)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b_ = boot[key]
        print(f"  {key}: mean_diff={b_['mean_diff']:.5f}, 95% CI [{b_['ci_low']:.5f}, {b_['ci_high']:.5f}]")
