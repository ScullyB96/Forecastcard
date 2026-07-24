"""Stage-by-stage mean-total bias ladder (Sec19): localizes the gap between
the per-bucket combine-stage bias (Sec18.2: EV -0.091 + PP/PK -0.113 + other
+0.080 = -0.124/game, PRE-SH-fix) and the full-pipeline bias (-0.284/game,
pre-SH-fix logged mean_pred_total 5.5285 vs actual 5.8125) -- roughly
-0.16/game unaccounted for, invisible to the bucket table since that table
only sees the situational-combine stage.

Computes mean predicted total vs. actual, per season, at each real stage of
the production pipeline:
  A. situational combine only (team_strength_situational, incl. cross-season
     prior + prior_minutes_multiplier + SH term -- validate_situational_toi)
  B. + goalie overlay (validate_goalie)
  C. + rest adjustment (away-B2B, dev-fit)
  D. full final-joint pathway (regulation-ratio scaling + OT logistic
     redistribution -- what validate_shorthanded.py actually outputs, and
     what the P(over)/CRPS benchmarks are computed against)
"""

import pandas as pd

from src.models.dixon_coles import dc_adjusted_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_goalie_layer
from src.models.validate_situational_toi import run_validation as run_situational_only
from src.utils.paths import DATA_RAW
import numpy as np


def _bias_by_season(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["pred_total"] = df["lambda_home"] + df["lambda_away"]
    df["actual_total"] = df["actual_home"] + df["actual_away"]
    by_season = df.groupby("season").agg(n=("gameId", "count"), pred=("pred_total", "mean"),
                                          actual=("actual_total", "mean"))
    by_season[f"bias_{label}"] = by_season["pred"] - by_season["actual"]
    return by_season[[f"bias_{label}"]]


if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    # Stage A: situational combine only.
    stage_a = run_situational_only(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                    prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    stage_a = stage_a[stage_a["season"] < DEV_MAX_SEASON]
    bias_a = _bias_by_season(stage_a, "A_combine")

    # Stage B: + goalie overlay.
    stage_b = run_goalie_layer(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    stage_b = stage_b[stage_b["season"] < DEV_MAX_SEASON]
    bias_b = _bias_by_season(stage_b, "B_goalie")

    # Stage C: + rest adjustment (away-B2B, dev-fit, symmetric embedded-bias credit per Sec22).
    stage_c = stage_b.copy()
    away_b2b_adj = fit_away_b2b_adjustment(schedule, stage_c["actual_away"], stage_c["lambda_away"],
                                            stage_c["gameId"])
    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    stage_c = stage_c.merge(away_rest, on="gameId", how="left")
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    stage_c = stage_c.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(stage_c["p_b2b_walk_forward"], away_b2b_adj)
    stage_c["lambda_home"] = stage_c["lambda_home"] + credit
    stage_c["lambda_away"] = stage_c["lambda_away"] + credit + (stage_c["away_rest"] == 0) * away_b2b_adj
    bias_c = _bias_by_season(stage_c, "C_rest")
    print(f"fitted away_b2b_adj={away_b2b_adj:.5f}")

    # Stage D: full final-joint pathway (regulation-ratio + OT logistic redistribution).
    stage_d_base = stage_c.copy()
    stage_d_base["lambda_home_reg"] = stage_d_base["lambda_home"] * HOME_REG_RATIO
    stage_d_base["lambda_away_reg"] = stage_d_base["lambda_away"] * AWAY_REG_RATIO
    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)
    merged = stage_d_base.merge(schedule_ot[["gameId", "lastPeriodType", "homeScore", "awayScore"]],
                                 on="gameId", how="left")
    ot_only = merged[merged["lastPeriodType"] == "OT"].copy()
    ot_only["lambda_diff_reg"] = ot_only["lambda_home_reg"] - ot_only["lambda_away_reg"]
    ot_only["home_won"] = (ot_only["homeScore"] > ot_only["awayScore"]).astype(int)
    a, b = fit_ot_logistic(ot_only["lambda_diff_reg"].values, ot_only["home_won"].values)

    results = []
    for row in stage_d_base.itertuples():
        joint = dc_adjusted_joint(row.lambda_home_reg, row.lambda_away_reg, rho=0.0, max_goals=12)
        n = joint.shape[0]
        final = np.zeros((n + 1, n + 1))
        ld = row.lambda_home_reg - row.lambda_away_reg
        p_ot = predict_ot_logistic_prob(a, b, ld)
        p_tie = ot_split["p_decided_in_ot"] * p_ot + ot_split["p_needs_so"] * ot_split["p_home_wins_so_flat"]
        for x in range(n):
            for y in range(n):
                p = joint[x, y]
                if x == y:
                    final[x + 1, y] += p * p_tie
                    final[x, y + 1] += p * (1 - p_tie)
                else:
                    final[x, y] += p
        xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        results.append({"gameId": row.gameId, "season": row.season, "actual_home": row.actual_home,
                         "actual_away": row.actual_away, "lambda_home": (final * xs).sum(),
                         "lambda_away": (final * ys).sum()})
    stage_d = pd.DataFrame(results)
    bias_d = _bias_by_season(stage_d, "D_full_joint")

    ladder = bias_a.join(bias_b).join(bias_c).join(bias_d)
    pd.set_option("display.width", 160)
    print(ladder)
    print("\noverall means:")
    print(ladder.mean())
