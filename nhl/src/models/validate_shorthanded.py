"""Cycle 23's final, adopted pipeline (supersedes validate_ot_team_specific.py):
identical full chain, with the shorthanded-goals-for term (Sec18) turned on
in the situational layer -- `use_sh_term=True`, threaded through
validate_situational_toi -> validate_goalie -> validate_rest.

Sec18 diagnosed the PP+PK bucket's own 16-season, zero-exception undershoot
as fully attributable to this deliberately-unmodeled term (Cycle 3's own
docstring), confirmed the term's xG basis is already well-calibrated to
real SH goals (ratio 1.008, no fitted conversion factor needed), and
confirmed (via the free EV-bucket check) that the model's REMAINING
EV-bucket undershoot is NOT a data-level issue (EV goals ratio 0.9966 to
EV xG) -- i.e. this term is not expected to close the whole mean-total gap,
only the well-diagnosed PP+PK piece of it.
"""

import numpy as np
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
from src.models.validate_goalie import run_validation as run_team_goalie_layer
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


def run_validation(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
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
    # Sec22: symmetric embedded-bias credit (corrected from Sec19.5/Sec20's wrong,
    # away-only-conditional version) on BOTH lambdas, plus the original, unchanged,
    # tonight-specific conditional term on the away side only.
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    base = base.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(base["p_b2b_walk_forward"], away_b2b_adj)
    base["lambda_home"] = base["lambda_home"] + credit
    base["lambda_away"] = base["lambda_away"] + credit + (base["away_rest"] == 0) * away_b2b_adj

    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=max_season)
    merged = base.merge(schedule_ot[["gameId", "lastPeriodType", "homeScore", "awayScore"]],
                         on="gameId", how="left")
    ot_only = merged[merged["lastPeriodType"] == "OT"].copy()
    ot_only["lambda_diff_reg"] = ot_only["lambda_home_reg"] - ot_only["lambda_away_reg"]
    ot_only["home_won"] = (ot_only["homeScore"] > ot_only["awayScore"]).astype(int)
    a, b = fit_ot_logistic(ot_only["lambda_diff_reg"].values, ot_only["home_won"].values)

    results = []
    for row in base.itertuples():
        lambda_diff_reg = row.lambda_home_reg - row.lambda_away_reg
        p_ot = predict_ot_logistic_prob(a, b, lambda_diff_reg)
        p_home_wins_tie = ot_split["p_decided_in_ot"] * p_ot + ot_split["p_needs_so"] * ot_split["p_home_wins_so_flat"]

        final_joint = _build_final_joint(row.lambda_home_reg, row.lambda_away_reg, p_home_wins_tie)
        n = final_joint.shape[0]
        xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": (final_joint * xs).sum(), "lambda_away": (final_joint * ys).sum(),
            "home_win_prob": final_joint[xs > ys].sum(), "away_win_prob": final_joint[xs < ys].sum(),
            "tie_prob": 0.0,
        })
    return pd.DataFrame(results), (a, b)


if __name__ == "__main__":
    from src.models.metrics_ledger import append_run

    r, (a, b) = run_validation()
    print(f"validated {len(r)} dev-set games, OT logistic a={a:.4f}, b={b:.4f}")
    row = append_run(
        r, model_name="shorthanded_poisson",
        config_flags={"league_avg_halflife_games": HALFLIFE_GAMES, "cross_season_weight": CROSS_SEASON_WEIGHT,
                       "prior_minutes_multiplier": PRIOR_MINUTES_MULTIPLIER, "use_sh_term": True,
                       "ot_logistic_a": round(a, 5), "ot_logistic_b": round(b, 5), "split": "development"},
        notes="Cycle 23 final: shorthanded-goals-for term (league-level xG rate x each team's own "
              "walk-forward PK ice time) added to the situational combine, diagnosed in Sec18 as the "
              "dominant, fully-explained driver of the PP+PK bucket's 16-season undershoot. "
              "See MODEL_DOCUMENTATION.md Sec18.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
