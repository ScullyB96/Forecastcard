"""Cycle 29 (Sec29's follow-up): root-cause fix for the stationary EV-bucket
gap_ev (~-0.09/game, Sec18.3/Sec29.5.1) -- threads a real halflife through
`league_avg_ev_toi_min`/`league_avg_other_toi_min` (validate_situational_toi.
py's `toi_halflife_games`, a NEW, separate parameter from the already-live
`league_avg_halflife_games`) instead of the original infinite-memory
expanding mean. Confirmed mechanism directly against real data before
building this: mean EV-TOI gap -0.99 min/game (expanding mean vs. real),
implied whole-game goal impact -0.077/game, closely matching the flat-era
gap_ev of -0.085/game (Sec29.5.1) -- a genuine root-cause fix, of exactly
the class Sec29.3.1 says is the only kind that can reach a residual this
size.

Full current-best pipeline (shorthanded_poisson + Sec22 rest fix + Sec26
walk-forward tie-mass ratio), rebuilt per grid cell on `toi_halflife_games`
since it touches the situational combine from the ground up.

ADOPTED (Sec32.6, 2026-07-24) at TOI_HALFLIFE_GAMES=1800: no real regression
on any ledger metric at any tested halflife; a real, bootstrap-confirmed
margin-MAE improvement at this specific halflife (CI entirely negative);
total-MAE itself does not clear bootstrap significance despite the
confirmed, root-caused mechanism -- adopted on the real margin-MAE gain and
the mechanism's own evidentiary strength, per the user's explicit call
(current best model name in the ledger: `ev_toi_halflife_poisson`).
`run_current_best()` is the new production entry point.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, add_walk_forward_reg_ratio
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.two_sided_diagonal_transfer import _indep_joint, fit_deltas_per_game, two_sided_transfer
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_team_goalie_layer
from src.models.validate_tie_mass_ratio import MAX_GOALS, MIN_DEV_SEASON, _bootstrap, _fit_dev_only_ot_logistic, \
    _metrics, _resolve_joint, _score_row, run_treated
from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio
from src.utils.paths import DATA_RAW

TOI_HALFLIFE_GRID = [300, 600, 900, 1200, 1800]
TOI_HALFLIFE_GAMES = 1800  # ADOPTED, Sec32.6


def run_current_best(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
    """The current best model (`ev_toi_halflife_poisson`, Sec32): Cycle 26's
    walk-forward tie-mass ratio pipeline plus this cycle's EV-TOI halflife
    fix at its adopted value."""
    return run_with_toi_halflife(TOI_HALFLIFE_GAMES, min_season=min_season, max_season=max_season)


def run_with_toi_halflife(toi_halflife_games: float, min_season: int = 20102011,
                           max_season: int | None = None) -> pd.DataFrame:
    """FIXED (post-review, 2026-07-24): previously filtered `base` to the
    caller's requested season range BEFORE fitting `away_b2b_adj`/
    `fit_ot_so_split`/`fit_ot_logistic` and before computing the walk-forward
    tie-mass calibration ratio -- meaning a holdout-only call
    (min_season=DEV_MAX_SEASON) fit every constant on the holdout games'
    own outcomes and truncated the calibration ratio's trailing memory to
    the holdout window alone. Same fix as `validate_tie_mass_ratio.
    _build_dev_base`/`_fit_dev_only_ot_logistic`: all constants and trailing
    state are built on the FULL history first; the caller's season range is
    applied only as a final output filter."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base = run_team_goalie_layer(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True,
                                  toi_halflife_games=toi_halflife_games)

    dev_only = base[(base["season"] >= MIN_DEV_SEASON) & (base["season"] < DEV_MAX_SEASON)]
    away_b2b_adj = fit_away_b2b_adjustment(schedule, dev_only["actual_away"], dev_only["lambda_away"],
                                            dev_only["gameId"])
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

    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
    base = base.merge(reg_ratios, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
    base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]

    # Full (unfiltered) base -- correct trailing memory for the calibration ratio's EWMA.
    ratio_df = add_walk_forward_ot_calibration_ratio(base, schedule_ot, HALFLIFE_GAMES, MAX_GOALS)
    base = base.merge(ratio_df, on="gameId", how="left")
    base["target_diag"] = base["ot_calibration_ratio"] * base["model_diag_mass"]
    delta_above, delta_below = fit_deltas_per_game(
        base["model_diag_mass"].values, base["above_mass"].values, base["below_mass"].values,
        base["target_diag"].values)
    base["delta_above"], base["delta_below"] = delta_above, delta_below

    # Constants fit DEV-ONLY, always -- never on the caller's own scored range.
    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)
    a, b = _fit_dev_only_ot_logistic(base, schedule_ot)

    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()

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


if __name__ == "__main__":
    print("Building current-best reference (Cycle 26, toi_halflife_games=None, original expanding mean)...")
    current_best = run_treated()
    m_current = _metrics(current_best)

    print("\n=== Grid over toi_halflife_games ===")
    cells = []
    for hl in TOI_HALFLIFE_GRID:
        print(f"Building toi_halflife_games={hl}...")
        r = run_with_toi_halflife(hl)
        m = _metrics(r)
        d_brier, lo_brier, hi_brier = _bootstrap(m_current["brier"], m["brier"])
        d_su, lo_su, hi_su = _bootstrap(m_current["su"], m["su"])
        d_margin, lo_margin, hi_margin = _bootstrap(m_current["margin_mae"], m["margin_mae"])
        d_total, lo_total, hi_total = _bootstrap(m_current["total_mae"], m["total_mae"])
        real_regression = (lo_brier > 0) or (hi_su < 0) or (lo_margin > 0)
        cells.append({"toi_halflife": hl, "brier_diff": d_brier, "brier_lo": lo_brier, "brier_hi": hi_brier,
                      "su_diff": d_su, "su_lo": lo_su, "su_hi": hi_su,
                      "margin_diff": d_margin, "margin_lo": lo_margin, "margin_hi": hi_margin,
                      "total_diff": d_total, "total_lo": lo_total, "total_hi": hi_total,
                      "real_regression": real_regression})
        print(f"  brier_diff={d_brier:+.5f} CI[{lo_brier:+.5f},{hi_brier:+.5f}]  "
              f"su_diff={d_su:+.5f} CI[{lo_su:+.5f},{hi_su:+.5f}]  "
              f"margin_diff={d_margin:+.5f} CI[{lo_margin:+.5f},{hi_margin:+.5f}]  "
              f"total_diff={d_total:+.5f} CI[{lo_total:+.5f},{hi_total:+.5f}]  real_regression={real_regression}")

    grid_df = pd.DataFrame(cells)
    pd.set_option("display.width", 200)
    print("\n=== Full grid ===")
    print(grid_df.to_string(index=False))

    survivors = grid_df[~grid_df["real_regression"]]
    print(f"\n{len(survivors)}/{len(grid_df)} cells survive the no-real-regression bar")
    if len(survivors):
        winner = survivors.loc[survivors["total_diff"].idxmin()]
        print(f"WINNER (most real total-MAE improvement among survivors): toi_halflife={winner['toi_halflife']}")

        print(f"\n=== Per-season bias, winning cell vs current best ===")
        r_win = run_with_toi_halflife(int(winner["toi_halflife"]), min_season=20102011, max_season=20999999)
        cb_full = run_treated(min_season=20102011, max_season=20999999)
        by_season = pd.DataFrame({
            "season": cb_full["season"],
            "current_total": cb_full["lambda_home"] + cb_full["lambda_away"],
            "winner_total": r_win["lambda_home"] + r_win["lambda_away"],
            "actual_total": cb_full["actual_home"] + cb_full["actual_away"],
        }).groupby("season").mean()
        by_season["current_bias"] = by_season["current_total"] - by_season["actual_total"]
        by_season["winner_bias"] = by_season["winner_total"] - by_season["actual_total"]
        print(by_season[["current_bias", "winner_bias"]])

        print(f"\n=== Holdout confirmatory check (winning cell, toi_halflife={winner['toi_halflife']}) ===")
        h_current = run_treated(min_season=DEV_MAX_SEASON, max_season=20999999)
        h_treated = run_with_toi_halflife(int(winner["toi_halflife"]), min_season=DEV_MAX_SEASON, max_season=20999999)
        mh_current, mh_treated = _metrics(h_current), _metrics(h_treated)
        for key in ("su", "brier", "margin_mae", "total_mae"):
            d, lo, hi = _bootstrap(mh_current[key], mh_treated[key])
            real = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
            print(f"{key:12s} mean_diff={d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]  ({real})")

    print(f"\n=== Logging ADOPTED model (TOI_HALFLIFE_GAMES={TOI_HALFLIFE_GAMES}) to the ledger ===")
    from src.models.metrics_ledger import append_run
    adopted = run_current_best()
    m_adopted = _metrics(adopted)
    row = append_run(
        adopted[["actual_home", "actual_away", "lambda_home", "lambda_away", "home_win_prob_full"]]
        .rename(columns={"home_win_prob_full": "home_win_prob"}).assign(away_win_prob=lambda d: 1 - d["home_win_prob"],
                                                                          tie_prob=0.0),
        model_name="ev_toi_halflife_poisson",
        config_flags={"split": "development", "toi_halflife_games": TOI_HALFLIFE_GAMES,
                      "halflife_games": HALFLIFE_GAMES, "cross_season_weight": CROSS_SEASON_WEIGHT,
                      "prior_minutes_multiplier": PRIOR_MINUTES_MULTIPLIER},
        notes="Cycle 29 (Sec32): EV-TOI expanding-mean root-cause fix, ADOPTED at "
              "toi_halflife_games=1800 per Sec32.6 -- real margin-MAE improvement, no real "
              "regression on any metric, total-MAE itself does not clear bootstrap "
              "significance despite the confirmed mechanism.",
    )
    print("ledger row:")
    for k, v in row.items():
        print(f"  {k}: {v}")
