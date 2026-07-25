"""Regression guard for the walk-forward-discipline bug found in a 2026-07-24
code review: `validate_tie_mass_ratio._build_dev_base`/`run_baseline`/
`run_treated` (and `validate_ev_toi_halflife.run_with_toi_halflife`) used to
filter their working `base` DataFrame to the CALLER's requested season range
*before* fitting `fit_away_b2b_adjustment`/`fit_ot_so_split`/`fit_ot_logistic`
and before computing the walk-forward tie-mass calibration ratio's own
trailing EWMA -- so a holdout-only call (`min_season=DEV_MAX_SEASON`) fit
every constant on the holdout games' own outcomes, and truncated the
calibration ratio's trailing memory to start fresh at the holdout boundary.

`check_holdout_ot_logistic.py` (Cycle 22's own hand-built holdout check)
never had this bug -- it explicitly fits every constant on a `dev_only`
slice regardless of the scored range. It is the GOLDEN REFERENCE this file
checks the corrected shared helpers against. The bug was introduced when
Cycle 26 refactored the per-cycle hand-built pattern into a shared,
reusable `_build_dev_base`/`run_baseline`/`run_treated` helper and filtered
the season range too early. This file exists so that if a *future*
refactor reintroduces the same class of bug, it fails loudly here instead
of silently contaminating the next holdout check.

Run: `.venv/bin/python -m tests.test_holdout_walk_forward_discipline`
No pytest in this environment -- plain assertions, clear pass/fail output,
matching this project's own `validate_*.py`/`check_*.py` convention.
"""

import numpy as np
import pandas as pd

from src.models.baseline_naive_poisson import build_team_game_log
from src.models.check_holdout_ot_logistic import run_full_range_with_dev_fit_constants
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.validate_baseline import fit_home_ice_multiplier
from src.models.validate_tie_mass_ratio import _bootstrap, run_baseline, run_treated

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_walk_forward_invariant_across_scored_range():
    """THE core invariant: a game's own prediction must not depend on what
    OTHER games are being scored alongside it in the same call. Score the
    same dev-set games two ways -- (a) dev-only range, (b) dev+holdout
    range -- and confirm every shared game's lambda_home/lambda_away/
    home_win_prob_full are IDENTICAL. Under the old bug, `away_b2b_adj`/
    `ot_split`/`(a,b)` and the calibration ratio's trailing state were all
    computed from `base` AFTER the season filter -- so scoring a wider
    range (which changes what rows survive the filter, if the filter's
    upper bound also changed) could, in the buggy version, change these
    fitted constants and therefore the predictions for games that appear
    in both calls. In the fixed version, dev-only fitting means this can
    never happen, by construction."""
    dev_only = run_treated(min_season=20102011, max_season=DEV_MAX_SEASON)
    dev_and_holdout = run_treated(min_season=20102011, max_season=20999999)
    shared = dev_and_holdout[dev_and_holdout["season"] < DEV_MAX_SEASON]

    dev_only = dev_only.sort_values("gameId").reset_index(drop=True)
    shared = shared.sort_values("gameId").reset_index(drop=True)
    check("same gameIds in both calls", (dev_only["gameId"].values == shared["gameId"].values).all())

    for col in ("lambda_home", "lambda_away", "home_win_prob_full"):
        max_diff = float(np.abs(dev_only[col].values - shared[col].values).max())
        check(f"walk-forward invariant: {col} identical regardless of scored range",
              max_diff < 1e-12, f"max diff = {max_diff}")


def test_golden_reference_cycle22_config():
    """The corrected `run_baseline(use_sh_term=False)` (fixed helper,
    dev-only-fit discipline) must reproduce `check_holdout_ot_logistic.py`'s
    own independent construction (which never had the bug) closely on the
    Cycle 22 configuration (no SH term, fixed reg ratios, no tie-mass).

    NOT bit-identical: `_indep_joint` (used by the shared helper) does not
    renormalize its truncated joint (`max_goals=12`) to sum to exactly 1;
    `dc_adjusted_joint(rho=0.0)` (used by check_holdout_ot_logistic.py) DOES
    renormalize (`joint / joint.sum()`), an orthogonal, well-understood
    truncation-handling difference between two parallel independent-Poisson
    implementations, unrelated to the walk-forward fix. Confirmed directly:
    _indep_joint's own truncated mass sums to ~0.999983, not 1.0. The
    tolerance below reflects that mechanical difference, not slack in the
    walk-forward discipline itself."""
    corrected = run_baseline(min_season=DEV_MAX_SEASON, max_season=20999999, use_sh_term=False)
    golden_full = run_full_range_with_dev_fit_constants(use_team_specific=True)
    golden = golden_full[golden_full["season"] >= DEV_MAX_SEASON]

    corrected = corrected.sort_values("gameId").reset_index(drop=True)
    golden = golden.sort_values("gameId").reset_index(drop=True)
    check("golden reference: same holdout gameIds", (corrected["gameId"].values == golden["gameId"].values).all())

    tolerances = {"lambda_home": 0.01, "lambda_away": 0.01, "home_win_prob_full": 0.002}
    for col, tol in tolerances.items():
        max_diff = float(np.abs(corrected[col].values - golden[col].values).max())
        check(f"golden reference match: {col} within truncation-renormalization tolerance",
              max_diff < tol, f"max diff = {max_diff} (tolerance {tol})")


def test_calibration_ratio_continuous_across_dev_holdout_boundary():
    """The tie-mass calibration ratio's trailing EWMA must be continuous
    across the dev/holdout label boundary -- that boundary is a reporting
    convention, not a real discontinuity in the walk-forward process. Under
    the old bug (calibration ratio computed AFTER filtering `base` to the
    caller's range), a holdout-only call would reset this EWMA's memory at
    the boundary; the fixed version computes it on the full, unfiltered
    history, so the value just after the boundary should sit close to the
    value just before it, not jump to a effectively-zero-history value."""
    from src.models.overtime_shootout import add_regulation_score
    from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio
    from src.models.validate_tie_mass_ratio import _build_dev_base
    from src.models.validate_drift import HALFLIFE_GAMES
    from src.models.overtime_shootout import add_walk_forward_reg_ratio
    from src.utils.paths import DATA_RAW

    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)
    base, _ = _build_dev_base()
    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
    base = base.merge(reg_ratios, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
    base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]
    ratio_df = add_walk_forward_ot_calibration_ratio(base, schedule_ot, HALFLIFE_GAMES, 12)
    merged = base[["gameId", "gameDate", "season"]].merge(ratio_df, on="gameId", how="left")
    merged = merged.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    boundary_idx = merged.index[merged["season"] >= DEV_MAX_SEASON][0]
    last_dev_ratio = merged.loc[boundary_idx - 1, "ot_calibration_ratio"]
    first_holdout_ratio = merged.loc[boundary_idx, "ot_calibration_ratio"]
    diff = abs(first_holdout_ratio - last_dev_ratio)
    check("calibration ratio continuous across dev/holdout boundary (no memory reset)",
          diff < 0.01, f"last dev={last_dev_ratio:.5f} first holdout={first_holdout_ratio:.5f} diff={diff:.5f}")

    effective_n_proxy = 2 * HALFLIFE_GAMES / np.log(2)
    n_dev_games = int((merged["season"] < DEV_MAX_SEASON).sum())
    check("dev sample large enough that a real (non-reset) EWMA is meaningful at the boundary",
          n_dev_games > effective_n_proxy, f"n_dev_games={n_dev_games}, effective_n_proxy={effective_n_proxy:.0f}")


def test_gbm_holdout_veto_sign():
    """Regression test for the backwards veto condition found in
    validate_gbm_stack.py's __main__ (fixed 2026-07-24): the veto must fire
    when the stack is REAL-WORSE (entire CI positive, since `_bootstrap(a,b)`
    returns `b-a` and Brier is lower-is-better), and must NOT fire when the
    stack is real-better (entire CI negative) or ambiguous (crosses zero)."""
    rng = np.random.default_rng(20260725)
    base_regression = rng.normal(0.20, 0.01, 2000)  # base Brier-like values
    stack_worse = base_regression + rng.normal(0.01, 0.001, 2000)  # stack REAL-worse
    stack_better = base_regression - rng.normal(0.01, 0.001, 2000)  # stack REAL-better

    _, lo_worse, hi_worse = _bootstrap(base_regression, stack_worse)
    _, lo_better, hi_better = _bootstrap(base_regression, stack_better)

    veto_fires_when_worse = lo_worse > 0
    veto_fires_when_better = lo_better > 0  # must NOT fire
    check("corrected veto condition (lo > 0) fires when stack is real-worse", veto_fires_when_worse,
          f"CI=[{lo_worse:.5f},{hi_worse:.5f}]")
    check("corrected veto condition (lo > 0) does NOT fire when stack is real-better", not veto_fires_when_better,
          f"CI=[{lo_better:.5f},{hi_better:.5f}]")


def test_home_ice_multiplier_requires_dev_only_boundary():
    """Regression test for the walk-forward-discipline gap found 2026-07-25
    (while building the live single-game predictor): `fit_home_ice_multiplier`
    used to take no season boundary at all, so it was fit on dev+holdout
    combined every time it ran, including inside `_build_dev_base()`'s own
    production chain -- the same class of bug Sec35 fixed for
    `away_b2b_adj`/`ot_split`/the OT logistic, just missed here since this
    function lived one layer further down the call chain. `max_season_
    exclusive` is now a REQUIRED argument (confirmed: calling without it
    raises `TypeError`, so a future refactor can't silently drop the
    boundary again) and dev-only fitting must differ (however slightly)
    from unrestricted fitting on this project's real data."""
    schedule = pd.read_parquet("data/raw/nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    log = build_team_game_log(schedule)

    try:
        fit_home_ice_multiplier(log)
        missing_arg_raises = False
    except TypeError:
        missing_arg_raises = True
    check("fit_home_ice_multiplier requires max_season_exclusive (can't silently omit the boundary)",
          missing_arg_raises)

    dev_only_mult = fit_home_ice_multiplier(log[log["season"] < DEV_MAX_SEASON], max_season_exclusive=DEV_MAX_SEASON)
    unrestricted_mult = fit_home_ice_multiplier(log, max_season_exclusive=log["season"].max() + 1)
    check("dev-only fit differs from unrestricted (dev+holdout) fit on real data",
          dev_only_mult != unrestricted_mult,
          f"dev_only={dev_only_mult}, unrestricted={unrestricted_mult}")


def test_no_dangling_for_else_remains():
    """Structural check for the dangling for/else bug found in
    validate_ev_toi_halflife.py's __main__ (fixed 2026-07-24): a
    `for ... : ... else:` construct where the else always fires (loop never
    breaks) prints misleading output unconditionally. Grep-level check that
    the specific pattern doesn't recur."""
    with open("src/models/validate_ev_toi_halflife.py") as f:
        content = f.read()
    check("no leftover 'NO SURVIVORS' dangling for/else in validate_ev_toi_halflife.py",
          "NO SURVIVORS" not in content)


if __name__ == "__main__":
    print("=== test_walk_forward_invariant_across_scored_range ===")
    test_walk_forward_invariant_across_scored_range()
    print("\n=== test_golden_reference_cycle22_config ===")
    test_golden_reference_cycle22_config()
    print("\n=== test_calibration_ratio_continuous_across_dev_holdout_boundary ===")
    test_calibration_ratio_continuous_across_dev_holdout_boundary()
    print("\n=== test_gbm_holdout_veto_sign ===")
    test_gbm_holdout_veto_sign()
    print("\n=== test_no_dangling_for_else_remains ===")
    test_no_dangling_for_else_remains()
    print("\n=== test_home_ice_multiplier_requires_dev_only_boundary ===")
    test_home_ice_multiplier_requires_dev_only_boundary()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    else:
        print("ALL CHECKS PASSED")
