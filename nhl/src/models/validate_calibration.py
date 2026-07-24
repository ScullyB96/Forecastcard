"""Cycle 12 (final piece): fits the isotonic calibration layer on a
TEMPORAL split of the development set (first 70% of dev-set games by date
to fit, last 30% -- never touched during fitting -- to evaluate), so the
reported improvement isn't just the calibration curve memorizing its own
fitting data. Compares calibrated vs. uncalibrated Brier/log-loss on the
held-out 30% only.
"""

import pandas as pd

from src.models.calibration import brier, fit_isotonic_calibration, log_loss
from src.models.validate_rest import run_validation as run_current_best

CALIBRATION_FIT_FRACTION = 0.7


def run_validation():
    r, adj = run_current_best()
    r = r.sort_values("gameId").reset_index(drop=True)
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)

    # Temporal split by season+date order (games are already date-ordered upstream;
    # gameId itself is date-ordered within season for the regular-season schedule).
    r = r.sort_values(["season", "gameId"]).reset_index(drop=True)
    split_idx = int(len(r) * CALIBRATION_FIT_FRACTION)
    fit_fold, eval_fold = r.iloc[:split_idx], r.iloc[split_idx:]

    iso = fit_isotonic_calibration(fit_fold["home_win_prob_full"], fit_fold["actual_home_win"])
    eval_fold = eval_fold.copy()
    eval_fold["calibrated_prob"] = iso.predict(eval_fold["home_win_prob_full"])
    return fit_fold, eval_fold


if __name__ == "__main__":
    fit_fold, eval_fold = run_validation()
    print(f"fit fold: {len(fit_fold)} games, eval fold (held out from fitting): {len(eval_fold)} games")

    raw_brier = brier(eval_fold["home_win_prob_full"], eval_fold["actual_home_win"])
    cal_brier = brier(eval_fold["calibrated_prob"], eval_fold["actual_home_win"])
    raw_ll = log_loss(eval_fold["home_win_prob_full"], eval_fold["actual_home_win"])
    cal_ll = log_loss(eval_fold["calibrated_prob"], eval_fold["actual_home_win"])

    print(f"UNCALIBRATED (eval fold): Brier={raw_brier:.5f}, log-loss={raw_ll:.5f}")
    print(f"CALIBRATED   (eval fold): Brier={cal_brier:.5f}, log-loss={cal_ll:.5f}")
    print(f"mean raw prob: {eval_fold['home_win_prob_full'].mean():.4f}, "
          f"mean calibrated prob: {eval_fold['calibrated_prob'].mean():.4f}, "
          f"actual home win rate: {eval_fold['actual_home_win'].mean():.4f}")
