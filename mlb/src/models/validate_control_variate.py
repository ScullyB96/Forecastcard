"""Monte Carlo variance-reduction control variate: measurement + go/no-go
(2026-08-04, external research report recommendation #1 -- see
compass_artifact_wf-43cc19e7...md and the approved plan this followed).

This measures, on real data, whether (runs_this_pa -
TransitionTable.conditional_mean_runs(state, outcome)) -- a per-PA quantity
with EXACT expectation zero (a martingale difference, see
base_out_transitions.py's own docstring) -- correlates enough with a game's
simulated run totals to materially shrink the Monte Carlo standard error of
total-runs/margin estimators at fixed trial count. The point of this tool:
several signals this project has repeatedly found "real but CI includes
zero" (bat speed, pulled-air rate, all 3 park-geometry factors, platoon x
TTO) may partly reflect an unresolvable noise floor rather than a true
null -- this is the honest test of whether that noise floor can actually be
shrunk before spending more compute re-testing that backlog.

Pre-registered acceptance bar (this project's own established discipline of
stating the bar BEFORE reading results, matching every other keep/revert
decision this session): >= 2x variance reduction on total runs. Below that,
this is reported as a negative finding and NOT wired into the canonical
backtest -- report but don't roll out a weak result.
"""
import numpy as np
import pandas as pd

from src.models.validate_game_simulator import build_shared_tables, run_validation
from src.models.metrics_ledger import append_run
from src.utils.paths import DATA_PROCESSED

TEST_SEASONS = {2023, 2024, 2025}
N_GAMES = 600
N_TRIALS = 200
SEED = 42
ACCEPTANCE_BAR = 2.0  # minimum variance-reduction factor on total runs to proceed


def _fit_beta_and_reduction(sim_arr: np.ndarray, cv_arr: np.ndarray) -> dict:
    """sim_arr/cv_arr: (n_games, n_trials) raw per-trial arrays, one row per
    game. Demeans sim WITHIN each game (a fixed-effects treatment) before
    pooling -- cv's own true mean is exactly 0 in EVERY game individually
    (the martingale property doesn't depend on which game it is), but sim's
    true mean varies meaningfully game to game (different matchups score
    differently); demeaning isolates the within-game relationship a control
    variate actually exploits, rather than letting between-game variation
    in sim's mean spuriously inflate the fitted correlation."""
    sim_dev = sim_arr - sim_arr.mean(axis=1, keepdims=True)
    cv_flat = cv_arr.ravel()
    sim_dev_flat = sim_dev.ravel()
    cv_mean_check = float(cv_arr.mean())  # unbiasedness sanity check -- should be ~0
    beta = float(np.cov(sim_dev_flat, cv_flat)[0, 1] / np.var(cv_flat))
    sim_adj = sim_arr - beta * cv_arr
    var_before = float(sim_arr.var(axis=1).mean())  # avg within-game trial variance
    var_after = float(sim_adj.var(axis=1).mean())
    reduction = var_before / var_after if var_after > 0 else float("inf")
    return {"beta": beta, "cv_mean": cv_mean_check, "cv_std": float(cv_arr.std()),
            "var_before": var_before, "var_after": var_after, "reduction_factor": reduction}


def main():
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building shared tables...", flush=True)
    shared = build_shared_tables(pa, TEST_SEASONS)

    trial_capture, cv_capture = {}, {}
    print(f"running {N_GAMES} games x {N_TRIALS} trials with control-variate tracking...", flush=True)
    r = run_validation(shared, TEST_SEASONS, N_GAMES, N_TRIALS, seed=SEED, write_ledger=False,
                        trial_capture=trial_capture, cv_capture=cv_capture,
                        notes="control-variate measurement run (not the canonical backtest)")

    game_pks = sorted(trial_capture.keys())
    sim_home = np.array([trial_capture[g][0] for g in game_pks], dtype=float)
    sim_away = np.array([trial_capture[g][1] for g in game_pks], dtype=float)
    cv_home = np.array([cv_capture[g][0] for g in game_pks], dtype=float)
    cv_away = np.array([cv_capture[g][1] for g in game_pks], dtype=float)
    cv_home_full = np.array([cv_capture[g][2] for g in game_pks], dtype=float)
    cv_away_full = np.array([cv_capture[g][3] for g in game_pks], dtype=float)
    sim_total, sim_margin = sim_home + sim_away, sim_home - sim_away

    def _measure(tag, cv_home_arr, cv_away_arr):
        cv_total_arr, cv_margin_arr = cv_home_arr + cv_away_arr, cv_home_arr - cv_away_arr
        out = {}
        for label, sim_arr, cv_arr in [
            ("home", sim_home, cv_home_arr), ("away", sim_away, cv_away_arr),
            ("total", sim_total, cv_total_arr), ("margin", sim_margin, cv_margin_arr),
        ]:
            res = _fit_beta_and_reduction(sim_arr, cv_arr)
            out[label] = res
            print(f"  [{tag}] {label:>8}: beta={res['beta']:+.4f}  cv_mean={res['cv_mean']:+.5f} (sanity, want ~0)  "
                  f"cv_std={res['cv_std']:.4f}  var {res['var_before']:.4f}->{res['var_after']:.4f}  "
                  f"reduction={res['reduction_factor']:.3f}x")
        return out

    print(f"\n=== control-variate measurement: n={len(game_pks)} games, {N_TRIALS} trials/game ===")
    print("-- narrow version (conditioned on realized outcome) --")
    results_narrow = _measure("narrow", cv_home, cv_away)
    print("-- full version (marginalized over the whole known per-PA distribution) --")
    results_full = _measure("full", cv_home_full, cv_away_full)

    reduction_narrow = results_narrow["total"]["reduction_factor"]
    reduction_full = results_full["total"]["reduction_factor"]
    best_tag, best_reduction = ("full", reduction_full) if reduction_full >= reduction_narrow \
        else ("narrow", reduction_narrow)
    passes = best_reduction >= ACCEPTANCE_BAR
    verdict = (f"PASSES bar via the {best_tag} version -- worth wiring into the canonical backtest" if passes
               else f"FAILS pre-registered bar ({ACCEPTANCE_BAR}x) on both versions -- do not wire in, report as a negative finding")
    print(f"\nPre-registered acceptance bar: >= {ACCEPTANCE_BAR}x reduction on total runs.")
    print(f"Measured: narrow={reduction_narrow:.3f}x, full={reduction_full:.3f}x -- {verdict}")

    row = append_run(
        r,
        config_flags={
            "task": "control_variate_measurement", "n_games_requested": N_GAMES,
            "test_seasons": sorted(TEST_SEASONS),
            "reduction_total_narrow": reduction_narrow, "reduction_total_full": reduction_full,
            "reduction_margin_narrow": results_narrow["margin"]["reduction_factor"],
            "reduction_margin_full": results_full["margin"]["reduction_factor"],
            "cv_mean_home_narrow": results_narrow["home"]["cv_mean"],
            "cv_mean_home_full": results_full["home"]["cv_mean"],
            "cv_mean_away_narrow": results_narrow["away"]["cv_mean"],
            "cv_mean_away_full": results_full["away"]["cv_mean"],
            "acceptance_bar": ACCEPTANCE_BAR, "passes_bar": passes, "best_version": best_tag,
        },
        n_trials=N_TRIALS,
        notes=(
            f"Control-variate variance-reduction measurement (2026-08-04, external report rec #1). "
            f"n={len(game_pks)} games x {N_TRIALS} trials, TEST_SEASONS={sorted(TEST_SEASONS)}. Two versions "
            f"compared: 'narrow' = per-PA (runs_this_pa - conditional_mean_runs(state, realized_outcome)) "
            f"(captures only the transition-table resample noise); 'full' = per-PA (runs_this_pa - "
            f"sum_o dist[o]*conditional_mean_runs(state,o)) marginalized over the whole known pre-PA "
            f"distribution (captures both the outcome-category draw AND the transition-table resample). "
            f"Both are exact martingale differences (expectation zero by construction). Measured total-runs "
            f"reduction: narrow={reduction_narrow:.2f}x, full={reduction_full:.2f}x. Margin reduction: "
            f"narrow={results_narrow['margin']['reduction_factor']:.2f}x, full={results_full['margin']['reduction_factor']:.2f}x. "
            f"Unbiasedness sanity check (E[cv] should be ~0 in every game) -- narrow home={results_narrow['home']['cv_mean']:+.5f}, "
            f"full home={results_full['home']['cv_mean']:+.5f}. Pre-registered bar >= {ACCEPTANCE_BAR}x on total runs: {verdict}"
        ),
    )
    print("\nLogged to metrics_ledger.parquet:", row["notes"][:80], "...")


if __name__ == "__main__":
    main()
