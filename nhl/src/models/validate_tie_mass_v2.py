"""Cycle 25 (Sec21/Sec24/Sec25): the re-attempted tie-mass fix, on top of
the full current-best pipeline (shorthanded_poisson + Sec22's rest fix).
Combines the two corrections Sec21.1-21.2 specified: (1) a two-sided
diagonal transfer (two_sided_diagonal_transfer.py) instead of Cycle 20's
one-sided version, which Sec21.1 proved is total-conserving by
construction and can never touch the ladder's reg-ratio/OT bias; (2)
walk-forward regulation ratios (overtime_shootout.add_walk_forward_reg_ratio)
instead of the fixed dev-fit HOME_REG_RATIO/AWAY_REG_RATIO constants, to
kill the trend component of the growing bias. Resolved through Cycle 22's
skill-informed OT logistic, not a flat rate.
"""

import numpy as np
import pandas as pd

from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, add_walk_forward_reg_ratio
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.two_sided_diagonal_transfer import _indep_joint, fit_deltas, two_sided_transfer
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_team_goalie_layer
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.utils.paths import DATA_RAW

MAX_GOALS = 12


def _build_final_joint(lambda_home_reg: float, lambda_away_reg: float, delta_above: float, delta_below: float,
                        p_home_wins_tie: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    reg_joint = _indep_joint(lambda_home_reg, lambda_away_reg, max_goals)
    reg_joint = two_sided_transfer(reg_joint, delta_above, delta_below)
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


def run_validation(min_season: int = 20102011, max_season: int | None = None,
                    use_two_sided_transfer: bool = True, use_walk_forward_reg_ratio: bool = True):
    """Both flags default True (the full Cycle 25 design); set either False
    to ablate that specific piece for isolation testing."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base = run_team_goalie_layer(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()

    away_b2b_adj = fit_away_b2b_adjustment(schedule, base["actual_away"], base["lambda_away"], base["gameId"])
    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(away_rest, on="gameId", how="left")
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    base = base.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(base["p_b2b_walk_forward"], away_b2b_adj)
    base["lambda_home"] = base["lambda_home"] + credit
    base["lambda_away"] = base["lambda_away"] + credit + (base["away_rest"] == 0) * away_b2b_adj

    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")

    if use_walk_forward_reg_ratio:
        reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
        base = base.merge(reg_ratios, on="gameId", how="left")
        base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
        base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]
    else:
        from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
        base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
        base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    if use_two_sided_transfer:
        delta_above, delta_below = fit_deltas(base["lambda_home_reg"].values, base["lambda_away_reg"].values,
                                               base["reg_home_score"].values, base["reg_away_score"].values)
    else:
        delta_above, delta_below = 0.0, 0.0

    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=max_season)
    merged = base.merge(schedule_ot[["gameId", "lastPeriodType", "homeScore", "awayScore"]], on="gameId", how="left")
    ot_only = merged[merged["lastPeriodType"] == "OT"].copy()
    ot_only["lambda_diff_reg"] = ot_only["lambda_home_reg"] - ot_only["lambda_away_reg"]
    ot_only["home_won"] = (ot_only["homeScore"] > ot_only["awayScore"]).astype(int)
    a, b = fit_ot_logistic(ot_only["lambda_diff_reg"].values, ot_only["home_won"].values)

    results = []
    for row in base.itertuples():
        lambda_diff_reg = row.lambda_home_reg - row.lambda_away_reg
        p_ot = predict_ot_logistic_prob(a, b, lambda_diff_reg)
        p_home_wins_tie = ot_split["p_decided_in_ot"] * p_ot + ot_split["p_needs_so"] * ot_split["p_home_wins_so_flat"]

        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, delta_above, delta_below,
                                          p_home_wins_tie)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": (final_joint * xs).sum(), "lambda_away": (final_joint * ys).sum(),
            "home_win_prob_full": final_joint[xs > ys].sum(),
        })
    return pd.DataFrame(results), (delta_above, delta_below), (a, b)


if __name__ == "__main__":
    r, (delta_above, delta_below), (a, b) = run_validation()
    print(f"validated {len(r)} dev-set games")
    print(f"fitted delta_above={delta_above:.4f}, delta_below={delta_below:.4f}, "
          f"OT logistic a={a:.4f}, b={b:.4f}")
    actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int)
    p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
    su = ((p > 0.5).astype(int) == actual_home_win).mean()
    brier = ((p - actual_home_win) ** 2).mean()
    total_mae = ((r["lambda_home"] + r["lambda_away"]) - (r["actual_home"] + r["actual_away"])).abs().mean()
    margin_mae = ((r["lambda_home"] - r["lambda_away"]) - (r["actual_home"] - r["actual_away"])).abs().mean()
    print(f"SU={su:.4f} Brier={brier:.5f} total_mae={total_mae:.5f} margin_mae={margin_mae:.5f}")
