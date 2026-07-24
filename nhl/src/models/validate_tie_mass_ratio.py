"""Cycle 26 (Sec25.6/Sec38): walk-forward per-game tie-mass delta, tested
against the pre-registered net-trade-off criterion from Sec25.6.5 (not the
automatic margin-MAE veto Sec25's candidate was rejected under -- that
candidate stays rejected; this is a genuinely new, untested design).

Same base pipeline as Sec25 (shorthanded_poisson + Sec22's rest fix), but
Sec25's single dev-fit global (delta_above, delta_below) is replaced by
`two_sided_diagonal_transfer.fit_deltas_per_game`, targeting each game's own
diagonal mass scaled by a walk-forward calibration RATIO
(`walk_forward_tie_ratio.py`) rather than a shared level -- preserving
cross-game matchup variation by construction, per Sec25.6's design
correction. Walk-forward regulation ratios (Sec21.2/Sec25) are bundled in as
part of this same tested unit, exactly as Sec25 bundled them.

`_build_baseline_joint`/`_build_treated_joint` reconstruct the SAME
per-game joint both the baseline (`shorthanded_poisson` + Sec22, fixed reg
ratios, zero tie-mass transfer -- bit-for-bit production) and the treated
candidate resolve to, so CRPS(total)/CRPS(margin) are directly comparable
per game, not just at the aggregate metric level.
"""

import numpy as np
import pandas as pd

from src.models.crps import discrete_crps, margin_pmf_from_joint, total_pmf_from_joint
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.ot_logistic import fit_ot_logistic, fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, add_walk_forward_reg_ratio
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.two_sided_diagonal_transfer import _indep_joint, fit_deltas_per_game, two_sided_transfer
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_dixon_coles_regulation import AWAY_REG_RATIO, HOME_REG_RATIO
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_team_goalie_layer
from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio
from src.utils.paths import DATA_RAW

MAX_GOALS = 12
MIN_DEV_SEASON = 20102011  # the standing dev floor everywhere in this project -- ALSO the floor used
# for fitting every global constant below, regardless of what season range a caller ultimately scores.


def _resolve_joint(reg_joint: np.ndarray, p_home_wins_tie: float, max_goals: int = MAX_GOALS) -> np.ndarray:
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


def _build_dev_base(use_sh_term: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-history base (situational+goalie+SH term+symmetric rest-fix
    credit) -- covers ALL available seasons, NEVER pre-filtered by a
    caller's min_season/max_season. `away_b2b_adj` (the one global scalar
    fit here) is fit on `[MIN_DEV_SEASON, DEV_MAX_SEASON)` ALWAYS, regardless
    of what range a caller will eventually score. `use_sh_term=False`
    exists only so tests can reproduce a pre-Cycle-23 configuration (e.g.
    the Cycle 22 golden reference, `check_holdout_ot_logistic.py`) -- every
    real caller keeps the default.

    FIXED (post-review, 2026-07-24): this function used to filter `base` to
    the caller's requested season range BEFORE fitting `away_b2b_adj` --
    meaning a holdout-only call (min_season=DEV_MAX_SEASON) fit the credit
    on the holdout games' own outcomes. `check_holdout_ot_logistic.py`'s own
    hand-built holdout check (Cycle 22) never had this bug -- it explicitly
    fit every constant on a `dev_only` slice regardless of the scored range;
    this shared helper broke that discipline when it was introduced (Cycle
    26) by filtering first. Callers now filter the OUTPUT of run_baseline/
    run_treated to their desired range, never this function's input."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base = run_team_goalie_layer(league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=use_sh_term)

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
    return base, schedule_ot


def _fit_dev_only_ot_logistic(base: pd.DataFrame, schedule_ot: pd.DataFrame) -> tuple[float, float]:
    """The OT-logistic (a, b) fit on OT-decided DEV-ONLY games
    (`[MIN_DEV_SEASON, DEV_MAX_SEASON)`), always -- regardless of what range
    `base` will eventually be filtered to for scoring. `base` here must
    still be full-history (pre-filter) when this is called."""
    merged = base.merge(schedule_ot[["gameId", "lastPeriodType", "homeScore", "awayScore"]], on="gameId", how="left")
    dev_ot_only = merged[(merged["lastPeriodType"] == "OT")
                          & (merged["season"] >= MIN_DEV_SEASON) & (merged["season"] < DEV_MAX_SEASON)].copy()
    dev_ot_only["lambda_diff_reg"] = dev_ot_only["lambda_home_reg"] - dev_ot_only["lambda_away_reg"]
    dev_ot_only["home_won"] = (dev_ot_only["homeScore"] > dev_ot_only["awayScore"]).astype(int)
    return fit_ot_logistic(dev_ot_only["lambda_diff_reg"].values, dev_ot_only["home_won"].values)


def run_baseline(min_season: int = 20102011, max_season: int | None = None,
                  use_sh_term: bool = True) -> pd.DataFrame:
    """Bit-for-bit `shorthanded_poisson` + Sec22 rest fix: fixed reg ratios,
    zero tie-mass transfer. Reconstructed here (rather than imported from
    `validate_shorthanded.py`) only so its per-game joint/PMF is exposed for
    a directly-comparable CRPS bootstrap against the treated candidate.
    `use_sh_term=False` reproduces a pre-Cycle-23 configuration (only used
    by the Cycle 22 golden-reference regression test)."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    base, schedule_ot = _build_dev_base(use_sh_term=use_sh_term)
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO

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
        final_joint = _resolve_joint(reg_joint, p_home_wins_tie, MAX_GOALS)
        results.append(_score_row(row, final_joint))
    return pd.DataFrame(results)


def run_treated(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
    """Cycle 26 candidate: walk-forward reg ratios + per-game, ratio-
    targeted tie-mass delta."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    base, schedule_ot = _build_dev_base()

    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
    base = base.merge(reg_ratios, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * base["home_reg_ratio_wf"]
    base["lambda_away_reg"] = base["lambda_away"] * base["away_reg_ratio_wf"]

    # Computed on the FULL (unfiltered) base -- its own trailing EWMA now
    # correctly carries the real, full walk-forward history regardless of
    # what range the caller will eventually score.
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


def _score_row(row, final_joint: np.ndarray) -> dict:
    n = final_joint.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total_pmf = total_pmf_from_joint(final_joint)
    margin_pmf, margin_offset = margin_pmf_from_joint(final_joint)
    actual_total = int(row.actual_home + row.actual_away)
    actual_margin = int(row.actual_home - row.actual_away)
    return {
        "gameId": row.gameId, "season": row.season,
        "actual_home": row.actual_home, "actual_away": row.actual_away,
        "lambda_home": (final_joint * xs).sum(), "lambda_away": (final_joint * ys).sum(),
        "home_win_prob_full": final_joint[xs > ys].sum(),
        "crps_total": discrete_crps(total_pmf, actual_total),
        "crps_margin": discrete_crps(margin_pmf, actual_margin + margin_offset),
        "cdf_total": np.cumsum(total_pmf),
    }


def _p_over_benchmark(r: pd.DataFrame, label: str) -> None:
    odds = pd.read_parquet(DATA_RAW / "sbro_odds_games.parquet")
    odds = odds.dropna(subset=["gameId", "close_total"])[["gameId", "close_total"]]
    m = r.merge(odds, on="gameId", how="inner")
    m["actual_total"] = m["actual_home"] + m["actual_away"]
    m["is_push"] = m["actual_total"] == m["close_total"]
    m["actual_over"] = (m["actual_total"] > m["close_total"]).astype(int)

    def _p_over(row) -> float:
        return float(1 - row["cdf_total"][int(np.floor(row["close_total"]))])

    m["model_p_over"] = m.apply(_p_over, axis=1)
    no_push = m[~m["is_push"]].copy()
    brier = ((no_push["model_p_over"].clip(1e-6, 1 - 1e-6) - no_push["actual_over"]) ** 2).mean()
    print(f"{label}: mean P(over)={no_push['model_p_over'].mean():.4f}  Brier={brier:.5f}  "
          f"n={len(no_push)} decided, {m['is_push'].sum()} pushes excluded")


def _bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 5000, seed: int = 20260723):
    diff = b - a
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _metrics(r: pd.DataFrame) -> dict:
    actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int)
    p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
    return {
        "su": ((p > 0.5).astype(int) == actual_home_win).values.astype(float),
        "brier": ((p - actual_home_win) ** 2).values,
        "total_mae": ((r["lambda_home"] + r["lambda_away"]) - (r["actual_home"] + r["actual_away"])).abs().values,
        "margin_mae": ((r["lambda_home"] - r["lambda_away"]) - (r["actual_home"] - r["actual_away"])).abs().values,
        "crps_total": r["crps_total"].values,
        "crps_margin": r["crps_margin"].values,
        "mean_pred_total": (r["lambda_home"] + r["lambda_away"]).values,
    }


if __name__ == "__main__":
    print("Building baseline (shorthanded_poisson + Sec22 rest fix, fixed reg ratios, no tie-mass)...")
    baseline = run_baseline()
    print("Building treated (Cycle 26: walk-forward reg ratios + per-game ratio-targeted tie-mass delta)...")
    treated = run_treated()
    assert len(baseline) == len(treated) and (baseline["gameId"].values == treated["gameId"].values).all()
    print(f"{len(baseline)} dev-set games\n")

    mb, mt = _metrics(baseline), _metrics(treated)

    print("=== Dev-set bootstrap: treated - baseline ===")
    for key in ("su", "brier", "total_mae", "margin_mae"):
        d, lo, hi = _bootstrap(mb[key], mt[key])
        real = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
        print(f"{key:12s} mean_diff={d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]  ({real})")

    print("\n=== CRPS bootstrap ===")
    crps_sum_b = mb["crps_total"] + mb["crps_margin"]
    crps_sum_t = mt["crps_total"] + mt["crps_margin"]
    for key, b_arr, t_arr in (("crps_total", mb["crps_total"], mt["crps_total"]),
                              ("crps_margin", mb["crps_margin"], mt["crps_margin"]),
                              ("crps_SUM(total+margin)", crps_sum_b, crps_sum_t)):
        d, lo, hi = _bootstrap(b_arr, t_arr)
        real = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
        print(f"{key:24s} mean_diff={d:+.5f} CI [{lo:+.5f}, {hi:+.5f}]  ({real})")
        print(f"{'':24s} baseline={b_arr.mean():.5f} treated={t_arr.mean():.5f}")

    print("\n=== Point estimates ===")
    print(f"baseline: SU={mb['su'].mean():.4f} Brier={mb['brier'].mean():.5f} "
          f"total_mae={mb['total_mae'].mean():.5f} margin_mae={mb['margin_mae'].mean():.5f} "
          f"mean_pred_total={mb['mean_pred_total'].mean():.4f}")
    print(f"treated:  SU={mt['su'].mean():.4f} Brier={mt['brier'].mean():.5f} "
          f"total_mae={mt['total_mae'].mean():.5f} margin_mae={mt['margin_mae'].mean():.5f} "
          f"mean_pred_total={mt['mean_pred_total'].mean():.4f}")

    actual_mean_total = (baseline["actual_home"] + baseline["actual_away"]).mean()
    print(f"actual mean total: {actual_mean_total:.4f}")

    print("\n=== Per-season bias (treated mean_pred_total - baseline mean_pred_total), era-flattening check ===")
    baseline["mean_pred_total"] = mb["mean_pred_total"]
    treated["mean_pred_total"] = mt["mean_pred_total"]
    by_season = pd.DataFrame({
        "season": baseline["season"],
        "baseline_total": baseline["mean_pred_total"],
        "treated_total": treated["mean_pred_total"],
        "actual_total": baseline["actual_home"] + baseline["actual_away"],
    }).groupby("season").mean()
    by_season["baseline_bias"] = by_season["baseline_total"] - by_season["actual_total"]
    by_season["treated_bias"] = by_season["treated_total"] - by_season["actual_total"]
    by_season["shift"] = by_season["treated_bias"] - by_season["baseline_bias"]
    pd.set_option("display.width", 160)
    print(by_season[["baseline_bias", "treated_bias", "shift"]])

    print("\n=== Totals-line P(over) benchmark (dev set) ===")
    _p_over_benchmark(baseline, "baseline")
    _p_over_benchmark(treated, "treated")

    row = append_run(
        treated[["actual_home", "actual_away", "lambda_home", "lambda_away", "home_win_prob_full"]]
        .rename(columns={"home_win_prob_full": "home_win_prob"}).assign(away_win_prob=lambda d: 1 - d["home_win_prob"],
                                                                          tie_prob=0.0),
        model_name="walk_forward_tie_ratio_poisson",
        config_flags={"split": "development", "cross_season_weight": CROSS_SEASON_WEIGHT,
                      "prior_minutes_multiplier": PRIOR_MINUTES_MULTIPLIER, "halflife_games": HALFLIFE_GAMES},
        notes="Cycle 26 (Sec25.6/Sec38): shorthanded_poisson + Sec22 rest fix + walk-forward reg ratios "
              "+ per-game, walk-forward-ratio-targeted two-sided tie-mass transfer. ADOPTED per the "
              "pre-registered net-trade-off rule (Sec25.6.5): summed CRPS(total)+CRPS(margin) real "
              "improvement, no real Brier/SU regression (dev or holdout).",
        extra_metrics={"crps_total": float(mt["crps_total"].mean()), "crps_margin": float(mt["crps_margin"].mean())},
    )
    print("\nledger row:")
    for k, v in row.items():
        print(f"  {k}: {v}")
