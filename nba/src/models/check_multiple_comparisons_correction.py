"""Multiple-comparisons robustness check (2026-08-01, addressing the full-
model audit's own validation-methodology concern): this session ran roughly
9 independent one-time confirmatory holdout reads as adoption gates, each at
the standard 95% CI (per-test alpha=0.05). With that many tests, a family-
wise Bonferroni correction would use a much stricter per-test alpha =
0.05/9 = 0.00556 (a 99.444% CI) to hold the FAMILY-WISE false-positive rate
at 5%. This script re-examines the ADOPTED holdout comparisons (not the
already-rejected ones -- those don't need a stricter bar, they already
failed the loose one) at that stricter interval, reusing each holdout-
check script's own prediction-building functions directly (no logic
duplicated) so the exact same holdout data and predictions are being
re-tested, just with a tighter percentile cut.

This is NOT a new model change and spends no new holdout read (the holdout
data was already read once per adoption decision; recomputing a stricter
percentile from the SAME bootstrap resamples is not a new "look" at the
data by this project's own confirmatory-veto standard -- it's a stricter
lens on an already-observed result, not a new decision point).

Run as `python -m src.models.check_multiple_comparisons_correction`.
"""

import numpy as np
import pandas as pd

from src.models.bootstrap_significance import _MAX_IDX_MATRIX_BYTES, N_BOOTSTRAP
from src.models.final_holdout_check import split_dev_holdout

N_ADOPTION_TESTS = 9
BONFERRONI_ALPHA = 0.05 / N_ADOPTION_TESTS


def bootstrap_delta_ci(per_game_a: pd.DataFrame, per_game_b: pd.DataFrame, game_id_col: str,
                        metrics: list[dict], alpha: float, n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Same resampling as `bootstrap_significance.bootstrap_compare`, parameterized on `alpha`
    instead of a fixed 95%, so the exact same procedure can be re-examined at a stricter
    (Bonferroni-corrected) percentile without touching the shared function's interface."""
    merged = per_game_a.merge(per_game_b, on=game_id_col, suffixes=("_a", "_b"), how="inner")
    n_games = len(merged)
    rng = np.random.default_rng(seed)
    batch_size = max(1, min(n_bootstrap, _MAX_IDX_MATRIX_BYTES // (8 * n_games)))
    arrays_a = {spec["name"]: merged[f"{spec['col']}_a"].to_numpy() for spec in metrics}
    arrays_b = {spec["name"]: merged[f"{spec['col']}_b"].to_numpy() for spec in metrics}
    deltas = {spec["name"]: np.empty(n_bootstrap) for spec in metrics}
    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        idx = rng.integers(0, n_games, size=(end - start, n_games))
        for spec in metrics:
            name = spec["name"]
            deltas[name][start:end] = arrays_a[name][idx].mean(axis=1) - arrays_b[name][idx].mean(axis=1)
        del idx
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    results = {}
    for spec in metrics:
        name = spec["name"]
        point = float(arrays_a[name].mean() - arrays_b[name].mean())
        ci = np.percentile(deltas[name], [lo_pct, hi_pct])
        results[name] = {"delta": point, "ci": (float(ci[0]), float(ci[1])), "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return results


def _check_phase1_joint_margin_fix() -> None:
    """IMPORTANT: `run_joint_margin_fix_holdout_check.py`'s own
    `_full_dev_and_holdout_fit` calls `add_team_ratings` WITHOUT overriding
    `cross_season_weight`/`own_halflife_games` -- meaning both now silently
    default to the LATER-adopted CROSS_SEASON_WEIGHT_RATING/
    OWN_HALFLIFE_GAMES_RATING (Sec33/36), which didn't exist when this
    lever was originally tested (Sec29). Left as-is, this would confound
    the re-check: both the "candidate" and "current" arms would get those
    two later levers applied EQUALLY, silently narrowing the gap between
    them (since the later levers independently also improve margin_mae,
    reducing how much residual credit is left for prior_games/halflife
    specifically once they're already present). Explicitly pin
    cross_season_weight=0.0, own_halflife_games=None on BOTH arms here to
    faithfully reproduce Sec29's ORIGINAL test conditions instead."""
    from src.models.team_strength import add_team_ratings, build_team_game_log
    from src.models.run_joint_margin_fix_holdout_check import CANDIDATE_HALFLIFE, CANDIDATE_PRIOR_GAMES_RATING
    from src.models.run_joint_margin_fix_holdout_check import _arms
    from src.models.validate_joint_margin_fix import _to_wide_games, build_predictions
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season

    current_season = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current_season)

    def _fit(prior_games_rating, halflife):
        l = add_team_ratings(log.copy(), prior_games_rating=prior_games_rating, league_avg_halflife_games=halflife,
                              cross_season_weight=0.0, own_halflife_games=None)
        games = _to_wide_games(l)
        return build_predictions(games).dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)

    cand = _arms(_fit(CANDIDATE_PRIOR_GAMES_RATING, CANDIDATE_HALFLIFE))
    current = _arms(_fit(15.0, None))
    _, cand_h = split_dev_holdout(cand)
    _, cur_h = split_dev_holdout(current)
    r = bootstrap_delta_ci(cand_h, cur_h, "gameId", [
        {"name": "total_mae", "col": "abs_total_err"}, {"name": "margin_mae", "col": "abs_margin_err"}], BONFERRONI_ALPHA)
    _report("Phase1 joint margin fix (prior_games_rating=12, halflife=2000), faithfully isolated", r)


def _check_phase1_cross_season_weight() -> None:
    from src.models.run_cross_season_weight_holdout_check import CANDIDATE_CROSS_SEASON_WEIGHT
    from src.models.run_cross_season_weight_holdout_check import _arms, _full_dev_and_holdout_fit

    cand = _arms(_full_dev_and_holdout_fit(CANDIDATE_CROSS_SEASON_WEIGHT))
    current = _arms(_full_dev_and_holdout_fit(0.0))
    _, cand_h = split_dev_holdout(cand)
    _, cur_h = split_dev_holdout(current)
    r = bootstrap_delta_ci(cand_h, cur_h, "gameId", [
        {"name": "total_mae", "col": "abs_total_err"}, {"name": "margin_mae", "col": "abs_margin_err"}], BONFERRONI_ALPHA)
    _report("Phase1 cross_season_weight=0.3", r)


def _check_phase1_own_halflife() -> None:
    from src.models.run_own_halflife_holdout_check import CANDIDATE_OWN_HALFLIFE_GAMES
    from src.models.run_own_halflife_holdout_check import _arms, _full_dev_and_holdout_fit

    cand = _arms(_full_dev_and_holdout_fit(CANDIDATE_OWN_HALFLIFE_GAMES))
    current = _arms(_full_dev_and_holdout_fit(None))
    _, cand_h = split_dev_holdout(cand)
    _, cur_h = split_dev_holdout(current)
    r = bootstrap_delta_ci(cand_h, cur_h, "gameId", [
        {"name": "total_mae", "col": "abs_total_err"}, {"name": "margin_mae", "col": "abs_margin_err"}], BONFERRONI_ALPHA)
    _report("Phase1 own_halflife_games=40", r)


def _check_team_stat_cross_season(categories) -> None:
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.run_team_stat_cross_season_holdout_check import CANDIDATE_CROSS_SEASON_WEIGHT
    from src.models.run_team_stat_cross_season_holdout_check import _fit_and_predict, _per_side
    from src.models.team_stat_rates import build_team_stat_game_log

    current_season = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current_season)
    for label in categories:
        cand = _per_side(_fit_and_predict(log, label, CANDIDATE_CROSS_SEASON_WEIGHT))
        current = _per_side(_fit_and_predict(log, label, 0.0))
        _, cand_h = split_dev_holdout(cand)
        _, cur_h = split_dev_holdout(current)
        r = bootstrap_delta_ci(cand_h, cur_h, "gameId", [{"name": "mae", "col": "abs_err"}], BONFERRONI_ALPHA)
        _report(f"team_stat_rates cross_season_weight=0.25 [{label}]", r)


def _check_team_stat_own_halflife(categories) -> None:
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.run_team_stat_own_halflife_holdout_check import CANDIDATE_OWN_HALFLIFE_GAMES
    from src.models.run_team_stat_own_halflife_holdout_check import _fit_and_predict, _per_side
    from src.models.team_stat_rates import build_team_stat_game_log

    current_season = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current_season)
    for label in categories:
        cand = _per_side(_fit_and_predict(log, label, CANDIDATE_OWN_HALFLIFE_GAMES))
        current = _per_side(_fit_and_predict(log, label, None))
        _, cand_h = split_dev_holdout(cand)
        _, cur_h = split_dev_holdout(current)
        r = bootstrap_delta_ci(cand_h, cur_h, "gameId", [{"name": "mae", "col": "abs_err"}], BONFERRONI_ALPHA)
        _report(f"team_stat_rates own_halflife_games=20 [{label}]", r)


def _report(label: str, results: dict) -> None:
    print(f"\n--- {label} ---", flush=True)
    for name, r in results.items():
        verdict = "SURVIVES Bonferroni-corrected CI" if r["excludes_zero"] else "DOES NOT survive -- now indistinguishable from zero"
        print(f"  {name}: delta={r['delta']:+.4f}  {100*(1-BONFERRONI_ALPHA):.3f}% CI=({r['ci'][0]:+.4f}, {r['ci'][1]:+.4f})  -> {verdict}", flush=True)


if __name__ == "__main__":
    print(f"=== Multiple-comparisons robustness check: {N_ADOPTION_TESTS} adoption tests this session, "
          f"Bonferroni per-test alpha={BONFERRONI_ALPHA:.5f} ({100*(1-BONFERRONI_ALPHA):.3f}% CI) ===", flush=True)
    print("Re-examining only the ADOPTED wins (rejected candidates already failed the looser 95% bar).\n", flush=True)

    _check_phase1_joint_margin_fix()
    _check_phase1_cross_season_weight()
    _check_phase1_own_halflife()
    _check_team_stat_cross_season(["dreb", "tov", "stl", "blk"])
    _check_team_stat_own_halflife(["dreb", "tov", "stl", "blk"])

    print("\n=== Done ===", flush=True)
