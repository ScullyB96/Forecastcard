"""Cycle 28 (Sec18.3/Sec19.2/Sec28.5): validates the walk-forward EV-bucket
intercept (ev_residual_intercept.py) on top of the full current-best
pipeline (shorthanded_poisson + Sec22 rest fix + Sec26 walk-forward
tie-mass ratio). Grids the intercept's OWN halflife (the one constant this
term introduces); everything else about the term is measured, not fit.

Also reports the per-season fitted-intercept trajectory -- the instrument
Sec28's build request designated to settle the Sec28.4 dispute (real,
turning-point-lag overshoot vs. coincidental netting among independent
structural pieces) -- and checks the inherited prediction that Sec22's and
Sec26's holdout total-MAE regressions should substantially reverse once this
term ships.
"""

import numpy as np
import pandas as pd

from src.models.ev_residual_intercept import add_walk_forward_ev_residual, compute_raw_ev_residual_log, \
    symmetric_ev_intercept_credit
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_walk_forward_reg_ratio
from src.models.two_sided_diagonal_transfer import _indep_joint, fit_deltas_per_game, two_sided_transfer
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_tie_mass_ratio import MAX_GOALS, _bootstrap, _build_dev_base, _metrics, _resolve_joint, \
    _score_row, run_treated
from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio

INTERCEPT_HALFLIFE_GRID = [200, 400, 600, 900, 1200]


def run_with_ev_intercept(intercept_halflife: float, min_season: int = 20102011,
                           max_season: int | None = None, raw_ev_log: pd.DataFrame | None = None
                           ) -> tuple[pd.DataFrame, pd.Series]:
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    base, schedule_ot = _build_dev_base(min_season, max_season)

    ev_residual = add_walk_forward_ev_residual(min_season, intercept_halflife, raw_log=raw_ev_log)
    base = base.merge(ev_residual, on=["gameId", "season"], how="left")
    ev_credit = symmetric_ev_intercept_credit(base["ev_residual_wf"])
    base["lambda_home"] = base["lambda_home"] + ev_credit
    base["lambda_away"] = base["lambda_away"] + ev_credit

    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
    base = base.merge(reg_ratios, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
    base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]

    ratio_df = add_walk_forward_ot_calibration_ratio(base, schedule_ot, HALFLIFE_GAMES, MAX_GOALS)
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
    return pd.DataFrame(results), base["ev_residual_wf"]


if __name__ == "__main__":
    print("Building current-best reference (Cycle 26, no EV intercept)...")
    current_best = run_treated()
    m_current = _metrics(current_best)

    print("Computing the raw (pre-EWMA) EV residual log once, full history "
          "(2010-11 onward, dev+holdout both), reused across every grid cell "
          "and the holdout check -- never rebuilt per season window, so the "
          "walk-forward trailing mean is never reset at a season boundary...")
    raw_ev_log = compute_raw_ev_residual_log(20102011)

    print("\n=== Grid over the EV intercept's own halflife ===")
    cells = []
    for hl in INTERCEPT_HALFLIFE_GRID:
        print(f"Building intercept halflife={hl}...")
        r, credit = run_with_ev_intercept(hl, raw_ev_log=raw_ev_log)
        m = _metrics(r)
        d_brier, lo_brier, hi_brier = _bootstrap(m_current["brier"], m["brier"])
        d_su, lo_su, hi_su = _bootstrap(m_current["su"], m["su"])
        d_margin, lo_margin, hi_margin = _bootstrap(m_current["margin_mae"], m["margin_mae"])
        d_total, lo_total, hi_total = _bootstrap(m_current["total_mae"], m["total_mae"])
        real_regression = (lo_brier > 0) or (hi_su < 0) or (abs(d_margin) > 1e-9)
        cells.append({"halflife": hl, "brier_diff": d_brier, "brier_lo": lo_brier, "brier_hi": hi_brier,
                      "su_diff": d_su, "su_lo": lo_su, "su_hi": hi_su,
                      "margin_diff": d_margin, "margin_lo": lo_margin, "margin_hi": hi_margin,
                      "total_diff": d_total, "total_lo": lo_total, "total_hi": hi_total,
                      "real_regression": real_regression})
        print(f"  brier_diff={d_brier:+.8f} CI[{lo_brier:+.5f},{hi_brier:+.5f}]  "
              f"su_diff={d_su:+.5f} CI[{lo_su:+.5f},{hi_su:+.5f}]  "
              f"margin_diff={d_margin:+.10f} (machine-precision check)  "
              f"total_diff={d_total:+.5f} CI[{lo_total:+.5f},{hi_total:+.5f}]")

    grid_df = pd.DataFrame(cells)
    pd.set_option("display.width", 200)
    print("\n=== Full grid ===")
    print(grid_df.to_string(index=False))

    survivors = grid_df[~grid_df["real_regression"]]
    print(f"\n{len(survivors)}/{len(grid_df)} cells survive the no-real-regression bar")
    winner_hl = survivors.loc[survivors["total_diff"].idxmin(), "halflife"] if len(survivors) else None
    print(f"WINNER (most negative/real total-MAE improvement among survivors): halflife={winner_hl}")

    if winner_hl is not None:
        print(f"\n=== Per-season fitted EV-intercept trajectory (winning halflife={winner_hl}), "
              f"FULL history dev+holdout -- the instrument settling Sec28.4 ===")
        r_win, credit_win = run_with_ev_intercept(winner_hl, min_season=20102011, max_season=20999999,
                                                   raw_ev_log=raw_ev_log)
        traj = pd.DataFrame({"season": r_win["season"], "credit": credit_win.values})
        # credit is HALF the whole-game residual added to each side; report the
        # full whole-game intercept value (both halves) for readability against
        # the original -0.091/game EV-residual figure.
        traj["full_intercept"] = traj["credit"] * 2
        by_season = traj.groupby("season")["full_intercept"].mean()
        print(by_season.to_string())

        print(f"\n=== Holdout confirmatory check (winning halflife={winner_hl}) ===")
        h_current = run_treated(min_season=DEV_MAX_SEASON, max_season=20999999)
        h_treated, h_credit = run_with_ev_intercept(winner_hl, min_season=DEV_MAX_SEASON, max_season=20999999,
                                                     raw_ev_log=raw_ev_log)
        mh_current, mh_treated = _metrics(h_current), _metrics(h_treated)
        for key in ("su", "brier", "margin_mae", "total_mae"):
            d, lo, hi = _bootstrap(mh_current[key], mh_treated[key])
            real = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
            print(f"{key:12s} mean_diff={d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]  ({real})")
