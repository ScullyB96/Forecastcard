"""Cycle 27 (Sec27): joint HALFLIFE_GAMES x prior_minutes_multiplier grid,
pre-registered spec in MODEL_DOCUMENTATION.md Sec27 -- replaces the frozen
HALFLIFE_GAMES=600 (validate_drift.py) re-tune that Sec24.4 deferred until
the mean-total structural chain (SH term, rest fix, tie-mass) was complete.
cross_season_weight is held fixed (Sec27.1: entangles more weakly, through
the prior rather than the baseline).

Full current-best pipeline (shorthanded_poisson + Sec22 rest fix + Sec26
walk-forward tie-mass ratio) rebuilt per grid cell, since HALFLIFE_GAMES
touches every downstream stage (situational combine, goalie overlay,
walk-forward reg ratio, walk-forward OT-calibration ratio) -- no caching
opportunity across cells. Sec27.2's selection rule: discard any cell with a
real dev-set regression on Brier/SU/margin-MAE vs. current best; among
survivors, minimize RMS of per-season mean-bias; touch holdout exactly once,
on the winning cell only.
"""

import numpy as np
import pandas as pd

from src.models.crps import discrete_crps, margin_pmf_from_joint, total_pmf_from_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, add_walk_forward_reg_ratio
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.two_sided_diagonal_transfer import _indep_joint, fit_deltas_per_game, two_sided_transfer
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT
from src.models.validate_goalie import run_validation as run_team_goalie_layer
from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio
from src.models.validate_tie_mass_ratio import _bootstrap, _metrics, _resolve_joint, _score_row
from src.utils.paths import DATA_RAW

MAX_GOALS = 12
CURRENT_BEST_HALFLIFE = 600
CURRENT_BEST_MULTIPLIER = 2.0

HALFLIFE_GRID = [200, 400, 600, 800]
MULTIPLIER_GRID = [1.0, 2.0, 4.0]


def _build_dev_base(halflife: float, multiplier: float, min_season: int, max_season: int):
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base = run_team_goalie_layer(league_avg_halflife_games=halflife, cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=multiplier, use_sh_term=True)
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
    return base, schedule_ot


def run_cell(halflife: float, multiplier: float, min_season: int = 20102011,
             max_season: int | None = None) -> pd.DataFrame:
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    base, schedule_ot = _build_dev_base(halflife, multiplier, min_season, max_season)

    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, halflife)
    base = base.merge(reg_ratios, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
    base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]

    ratio_df = add_walk_forward_ot_calibration_ratio(base, schedule_ot, halflife, MAX_GOALS)
    base = base.merge(ratio_df, on="gameId", how="left")
    base["target_diag"] = base["ot_calibration_ratio"] * base["model_diag_mass"]
    delta_above, delta_below = fit_deltas_per_game(
        base["model_diag_mass"].values, base["above_mass"].values, base["below_mass"].values,
        base["target_diag"].values)
    base["delta_above"], base["delta_below"] = delta_above, delta_below

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
        reg_joint = _indep_joint(row.lambda_home_reg, row.lambda_away_reg, MAX_GOALS)
        reg_joint = two_sided_transfer(reg_joint, row.delta_above, row.delta_below)
        final_joint = _resolve_joint(reg_joint, p_home_wins_tie, MAX_GOALS)
        results.append(_score_row(row, final_joint))
    return pd.DataFrame(results)


def bias_rms(r: pd.DataFrame) -> float:
    r = r.copy()
    r["pred_total"] = r["lambda_home"] + r["lambda_away"]
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    by_season = r.groupby("season").apply(lambda d: (d["pred_total"] - d["actual_total"]).mean())
    return float(np.sqrt((by_season ** 2).mean()))


if __name__ == "__main__":
    print(f"Building current-best reference cell (halflife={CURRENT_BEST_HALFLIFE}, "
          f"multiplier={CURRENT_BEST_MULTIPLIER})...")
    current_best = run_cell(CURRENT_BEST_HALFLIFE, CURRENT_BEST_MULTIPLIER)
    m_current = _metrics(current_best)
    rms_current = bias_rms(current_best)
    print(f"current best: bias RMS={rms_current:.5f}\n")

    cells = []
    for hl in HALFLIFE_GRID:
        for mult in MULTIPLIER_GRID:
            if hl == CURRENT_BEST_HALFLIFE and mult == CURRENT_BEST_MULTIPLIER:
                r = current_best
            else:
                print(f"Building cell halflife={hl}, multiplier={mult}...")
                r = run_cell(hl, mult)
            m = _metrics(r)
            d_brier, lo_brier, hi_brier = _bootstrap(m_current["brier"], m["brier"])
            d_su, lo_su, hi_su = _bootstrap(m_current["su"], m["su"])
            d_margin, lo_margin, hi_margin = _bootstrap(m_current["margin_mae"], m["margin_mae"])
            real_regression = (lo_brier > 0) or (lo_su < 0 and hi_su < 0) or (lo_margin > 0)
            rms = bias_rms(r)
            cells.append({
                "halflife": hl, "multiplier": mult,
                "brier_diff": d_brier, "brier_lo": lo_brier, "brier_hi": hi_brier,
                "su_diff": d_su, "su_lo": lo_su, "su_hi": hi_su,
                "margin_diff": d_margin, "margin_lo": lo_margin, "margin_hi": hi_margin,
                "real_regression": real_regression,
                "bias_rms": rms,
            })
            print(f"  brier_diff={d_brier:+.5f} CI[{lo_brier:+.5f},{hi_brier:+.5f}]  "
                  f"su_diff={d_su:+.5f} CI[{lo_su:+.5f},{hi_su:+.5f}]  "
                  f"margin_diff={d_margin:+.5f} CI[{lo_margin:+.5f},{hi_margin:+.5f}]  "
                  f"real_regression={real_regression}  bias_rms={rms:.5f}")

    grid_df = pd.DataFrame(cells)
    pd.set_option("display.width", 200)
    print("\n=== Full grid ===")
    print(grid_df.to_string(index=False))

    survivors = grid_df[~grid_df["real_regression"]]
    print(f"\n{len(survivors)}/{len(grid_df)} cells survive the no-real-regression bar")
    if len(survivors):
        winner = survivors.loc[survivors["bias_rms"].idxmin()]
        print(f"\nWINNER: halflife={winner['halflife']}, multiplier={winner['multiplier']}, "
              f"bias_rms={winner['bias_rms']:.5f} (current best bias_rms={rms_current:.5f})")
    else:
        print("\nNO SURVIVORS -- every cell shows a real regression vs current best")
