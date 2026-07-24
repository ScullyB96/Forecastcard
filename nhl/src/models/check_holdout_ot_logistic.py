"""Holdout check for the team-specific OT logistic (Sec17), using the
CORRECTED protocol from Sec15: paired bootstrap on holdout-only games,
confirmatory-veto-only. All constants (the OT logistic a/b, the OT/SO
split, the flat SO rate, the rest adjustment) fit on DEV-ONLY data, applied
to both splits without refitting -- same discipline as every other holdout
check in this project (Sec7.6, Sec12.4, Sec15.2). Uses
`validate_goalie.run_validation` (not `validate_cross_season`) as the base,
same reason as `check_holdout_cross_season.py`: the higher-level wrapper
filters its OWN output to season < max_season internally, which would
silently drop every holdout row before this script ever saw them."""

import numpy as np
import pandas as pd

from src.models.dixon_coles import dc_adjusted_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_team_goalie_layer
from src.models.validate_ot_logistic import _build_final_joint, _per_game_metrics, paired_bootstrap
from src.utils.paths import DATA_RAW


def run_full_range_with_dev_fit_constants(use_team_specific: bool, min_season: int = 20102011) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base = run_team_goalie_layer(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER)
    base = base[base["season"] >= min_season].copy()

    # Rest adjustment: fit dev-only, applied to both splits (same as every other holdout check here).
    dev_only = base[base["season"] < DEV_MAX_SEASON]
    away_b2b_adj = fit_away_b2b_adjustment(schedule, dev_only["actual_away"], dev_only["lambda_away"],
                                            dev_only["gameId"])
    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(away_rest, on="gameId", how="left")
    # Sec22: symmetric embedded-bias credit (corrected from Sec19.5/Sec20's wrong version).
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    base = base.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(base["p_b2b_walk_forward"], away_b2b_adj)
    base["lambda_home"] = base["lambda_home"] + credit
    base["lambda_away"] = base["lambda_away"] + credit + (base["away_rest"] == 0) * away_b2b_adj

    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

    # Fit ALL constants dev-only.
    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)
    dev_merged = base[base["season"] < DEV_MAX_SEASON].merge(
        schedule_ot[["gameId", "lastPeriodType", "homeScore", "awayScore"]], on="gameId", how="left")
    ot_only = dev_merged[dev_merged["lastPeriodType"] == "OT"].copy()
    ot_only["lambda_diff_reg"] = ot_only["lambda_home_reg"] - ot_only["lambda_away_reg"]
    ot_only["home_won"] = (ot_only["homeScore"] > ot_only["awayScore"]).astype(int)
    a, b = fit_ot_logistic(ot_only["lambda_diff_reg"].values, ot_only["home_won"].values)
    flat_p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)

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
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": (final_joint * xs).sum(), "lambda_away": (final_joint * ys).sum(),
            "home_win_prob_full": final_joint[xs > ys].sum(),
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    baseline = run_full_range_with_dev_fit_constants(use_team_specific=False)
    treated = run_full_range_with_dev_fit_constants(use_team_specific=True)

    baseline_holdout = baseline[baseline["season"] >= DEV_MAX_SEASON].reset_index(drop=True)
    treated_holdout = treated[treated["season"] >= DEV_MAX_SEASON].reset_index(drop=True)
    print(f"holdout: {len(baseline_holdout)} games")

    for label, df in [("BASELINE, holdout", baseline_holdout), ("TEAM-SPECIFIC, holdout", treated_holdout)]:
        m = _per_game_metrics(df)
        print(f"[{label}] SU={m['su'].mean():.4f} Brier={m['brier'].mean():.5f} "
              f"total_mae={m['total_ae'].mean():.5f} margin_mae={m['margin_ae'].mean():.5f}")

    print("\nHOLDOUT paired bootstrap (5000 resamples), team-specific MINUS baseline:")
    boot = paired_bootstrap(baseline_holdout, treated_holdout)
    for key in ("su", "brier", "total_ae", "margin_ae"):
        b_ = boot[key]
        print(f"  {key}: mean_diff={b_['mean_diff']:.5f}, 95% CI [{b_['ci_low']:.5f}, {b_['ci_high']:.5f}]")
