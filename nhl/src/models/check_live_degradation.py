"""The degradation-budget check RUNBOOK.md describes but, until now, only
described -- turns §37.4's rolling 4-week check from a paragraph of
instructions into a runnable script. Joins the immutable prediction log
(`prediction_log.py`) against real completed-game outcomes, and compares
live performance to the frozen model's own backtest-holdout performance
via an independent-samples bootstrap (NOT paired -- live games and
holdout games are different games entirely; this bootstraps each set's
own sampling uncertainty separately, then compares the two resulting
distributions for the gap -- the same design this project's holdout-check
protocol already uses for dev-vs-holdout comparisons elsewhere).

Run as `python -m src.models.check_live_degradation` any time there are
completed live games with logged predictions to check -- correct to run
early with a small sample (the bootstrap CI will just be wide) or on the
intended rolling 4-week cadence once the season is underway.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.prediction_log import read_log
from src.models.validate_tie_mass_ratio import run_treated
from src.utils.paths import DATA_RAW

N_BOOTSTRAP = 5000
SEED = 20260725

# Sec37.3's pre-registered budget -- routine (realistic-heuristic-comparator) and escalation
# (worst-case zero-information) thresholds. margin_mae's escalation ceiling is a REAL, confirmed
# number (Sec37.2), not a noise bound like the other five.
ROUTINE_BUDGET = {"margin_mae": 0.00382, "brier": 0.00088, "total_mae": 0.00610}
ESCALATION_CEILING = {"margin_mae": 0.01371, "brier": 0.00110, "total_mae": 0.00491}


def load_live_predictions_with_outcomes(schedule: pd.DataFrame) -> pd.DataFrame:
    """Joins the immutable log against real final scores -- games with no
    real result yet are excluded (inner join), never scored against an
    outcome that doesn't exist. Regular-season predictions only
    (`game_type == 2`) -- preseason predictions (Sec39.3/Sec41) were never
    meant to be evaluated as real ones."""
    entries = read_log()
    if not entries:
        return pd.DataFrame()
    log_df = pd.DataFrame(entries)
    log_df = log_df[log_df["game_type"] == 2]
    if log_df.empty:
        return pd.DataFrame()

    finals = schedule[["gameId", "homeScore", "awayScore"]].rename(columns={"gameId": "game_id"})
    return log_df.merge(finals, on="game_id", how="inner")


def _metrics(pred_home_win_prob, lambda_home, lambda_away, actual_home, actual_away) -> dict:
    p = np.clip(pred_home_win_prob.astype(float), 1e-6, 1 - 1e-6)
    actual_home_win = (actual_home > actual_away).astype(int)
    return {
        "brier": ((p - actual_home_win) ** 2).values,
        "margin_mae": ((lambda_home - lambda_away) - (actual_home - actual_away)).abs().values,
        "total_mae": ((lambda_home + lambda_away) - (actual_home + actual_away)).abs().values,
    }


def _bootstrap_gap(live_vals: np.ndarray, holdout_vals: np.ndarray,
                    n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """Independent-samples bootstrap (live and holdout are different
    games, not paired): bootstraps each set's own mean separately, then
    returns (gap_mean, ci_low, ci_high) for live - holdout."""
    rng = np.random.default_rng(seed)
    live_boot = live_vals[rng.integers(0, len(live_vals), size=(n_bootstrap, len(live_vals)))].mean(axis=1)
    holdout_boot = holdout_vals[rng.integers(0, len(holdout_vals), size=(n_bootstrap, len(holdout_vals)))].mean(axis=1)
    gap = live_boot - holdout_boot
    return float((live_vals.mean() - holdout_vals.mean())), float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))


def _verdict(metric_name: str, lo: float, hi: float) -> str:
    escalation, routine = ESCALATION_CEILING[metric_name], ROUTINE_BUDGET[metric_name]
    if lo > escalation:
        return "ESCALATION -- real, unbudgeted anomaly; investigate a specific cause before assuming the model needs refitting"
    if lo > routine:
        if metric_name == "margin_mae":
            return "ROUTINE BREACH -- investigate the goalie-data pipeline first (Sec37.4's named pattern)"
        return "ROUTINE BREACH -- investigate a specific cause before refitting anything"
    if hi < 0:
        return "live outperforming backtest -- no action, just note it"
    return "within budget -- no action"


def check_degradation_budget(holdout_reference: pd.DataFrame | None = None) -> dict:
    """`holdout_reference`: pass a pre-built `run_treated()` holdout
    DataFrame to avoid recomputing it (expensive); None rebuilds it fresh."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"].isin(["OFF", "FINAL"]))]

    live = load_live_predictions_with_outcomes(schedule)
    if live.empty:
        print("no completed live games with logged predictions yet -- nothing to check", flush=True)
        return {}

    if holdout_reference is None:
        holdout_reference = run_treated(min_season=DEV_MAX_SEASON, max_season=20999999)

    live_m = _metrics(live["home_win_prob_full"], live["lambda_home"], live["lambda_away"],
                       live["homeScore"], live["awayScore"])
    holdout_m = _metrics(holdout_reference["home_win_prob_full"], holdout_reference["lambda_home"],
                          holdout_reference["lambda_away"], holdout_reference["actual_home"],
                          holdout_reference["actual_away"])

    print(f"=== live degradation-budget check: {len(live)} completed live game(s) vs "
          f"{len(holdout_reference)} backtest-holdout games ===", flush=True)
    if len(live) < 30:
        print(f"NOTE: only {len(live)} live game(s) so far -- the bootstrap CI below will be wide "
              f"and any verdict is provisional until closer to a real 4-week window (~220-240 games)", flush=True)

    results = {}
    for name in ("margin_mae", "brier", "total_mae"):
        gap_mean, lo, hi = _bootstrap_gap(live_m[name], holdout_m[name])
        verdict = _verdict(name, lo, hi)
        results[name] = {
            "live_n": len(live_m[name]), "holdout_n": len(holdout_m[name]),
            "live_mean": float(live_m[name].mean()), "holdout_mean": float(holdout_m[name].mean()),
            "gap_mean": gap_mean, "gap_ci95": (lo, hi),
            "routine_budget": ROUTINE_BUDGET[name], "escalation_ceiling": ESCALATION_CEILING[name],
            "verdict": verdict,
        }
        print(f"{name:12s} live={live_m[name].mean():.5f} (n={len(live_m[name])})  "
              f"holdout={holdout_m[name].mean():.5f} (n={len(holdout_m[name])})  "
              f"gap={gap_mean:+.5f}  CI=[{lo:+.5f},{hi:+.5f}]  "
              f"routine<={ROUTINE_BUDGET[name]}  escalation<={ESCALATION_CEILING[name]}  -> {verdict}", flush=True)
    return results


if __name__ == "__main__":
    check_degradation_budget()
