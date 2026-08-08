"""Does the game simulator's recentered win probability (weekly_update.py, review round 4)
need a calibration correction, the same way TD_PROB_CALIB does for props (§9.4)? Prompted by
a reliability-table pattern found in validate_game_simulator.py's full-TEST re-validation
(2026-08): overconfident in the (0.4, 0.5] bucket, underconfident in (0.7, 1.0].

That pattern was measured on the SAME games whose drive outcomes feed the TRAIN-fit
resampling pools' neighborhood (all of TEST=2022-2025, scored by a simulator whose pools are
fixed once from TRAIN=2018-2021 -- no per-game leakage, but also no genuine calibration-fit/
holdout split). Unlike TD_PROB_CALIB's raw_td_prob (which is walk-forward *within* TRAIN, so
TRAIN pairs are safe to fit a calibration map on), this simulator's pools are NOT walk-forward
within TRAIN -- they're a single fixed pool built from the whole TRAIN block at once. Fitting
a calibration map on TRAIN games here would risk leakage (a TRAIN game's own drives are inside
the very pool used to simulate it). So: split TEST itself into CALIB=2022-2023 (fit candidate
calibration maps) and HOLDOUT=2024-2025 (score them, never touched while fitting) -- a genuine
out-of-sample test, following this project's decision rule from §9.4: auto-refit linear ships
as the default; isotonic or Beta only get promoted if one wins THIS strict comparison outright.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.drive_transitions import implied_total_quantile_bin
from src.models.game_simulator import build_simulator_for_season_range
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
CALIB_SEASONS = {2022, 2023}
HOLDOUT_SEASONS = {2024, 2025}
N_TRIALS = 5000  # matches production (raised from 300, 2026-08) -- fitting the calibration
# constants on 300-trial outputs and applying them to production's 5000-trial outputs is a
# real errors-in-variables mismatch (MC noise at fit time attenuates the fitted slope
# relative to what cleaner inputs need), flagged by external review; refit on matching noise.


def brier(pred: pd.Series, actual: pd.Series) -> float:
    return float(((pred - actual) ** 2).mean())


def bucket_table(pred: pd.Series, actual: pd.Series, name: str) -> pd.DataFrame:
    df = pd.DataFrame({name: pred, "actual_home_win": actual})
    df["bucket"] = pd.cut(df[name], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    return df.groupby("bucket", observed=True).agg(
        mean_pred=(name, "mean"), actual_win_rate=("actual_home_win", "mean"), n=("actual_home_win", "size")
    )


if __name__ == "__main__":
    print("loading data...")
    pbp = pd.read_parquet(f"{DATA_RAW}/pbp_2016_2025.parquet")
    schedules = pd.read_parquet(f"{DATA_RAW}/schedules_2016_2025.parquet")
    layer1 = pd.read_parquet(f"{DATA_PROCESSED}/layer1_games_with_ratings.parquet")

    print("building drive-resampling pools from TRAIN=2018-2021 only (no TEST/CALIB/HOLDOUT lookahead)...")
    pbp_train = pbp[pbp["season"].isin(TRAIN_SEASONS)]
    schedules_train = schedules[schedules["season"].isin(TRAIN_SEASONS)]
    sim, bin_edges = build_simulator_for_season_range(pbp_train, schedules_train)

    market = schedules[["game_id", "spread_line", "total_line"]]
    layer1_m = layer1.merge(market, on="game_id", how="left")
    layer1_m["actual_total"] = layer1_m["home_score"] + layer1_m["away_score"]

    eval_seasons = CALIB_SEASONS | HOLDOUT_SEASONS
    games = layer1_m[layer1_m["season"].isin(eval_seasons)].dropna(
        subset=["team_a", "team_b", "spread_line", "total_line"]
    ).copy()
    games["home_implied_total"] = games["total_line"] / 2 + games["spread_line"] / 2
    games["away_implied_total"] = games["total_line"] / 2 - games["spread_line"] / 2
    print(f"n CALIB+HOLDOUT games: {len(games)}")

    print(f"\nsimulating all CALIB+HOLDOUT games ({N_TRIALS} trials each)...")
    recentered_wp = []
    for i, row in enumerate(games.itertuples()):
        home_q = implied_total_quantile_bin(row.home_implied_total, bin_edges)
        away_q = implied_total_quantile_bin(row.away_implied_total, bin_edges)
        result = sim.simulate_game(home_q, away_q, n_trials=N_TRIALS)
        # exact recentering weekly_update.py applies before deriving win probability (review
        # round 4) -- our_margin's real production-time inputs (QB-swap/injury adjustments)
        # aren't available walk-forward here, so spread_line stands in as the location anchor,
        # same substitution validate_game_simulator.py already makes for its "current model"
        # baseline (§4.3: our_margin IS the market spread_line directly on a game with a real
        # line and no active adjustment).
        recentered_margins = result["margins"] - result["margin_mean"] + row.spread_line
        recentered_wp.append(float((recentered_margins > 0).mean()))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(games)} games simulated...")

    games["recentered_wp"] = recentered_wp
    games["actual_home_win"] = (games["actual_margin"] > 0).astype(float)
    valid = games.dropna(subset=["recentered_wp"])

    calib_set = valid[valid["season"].isin(CALIB_SEASONS)]
    holdout_set = valid[valid["season"].isin(HOLDOUT_SEASONS)]
    print(f"\nCALIB (fit) n={len(calib_set)}, HOLDOUT (score, never touched while fitting) n={len(holdout_set)}")

    # --- baseline: no calibration at all ---
    brier_none = brier(holdout_set["recentered_wp"], holdout_set["actual_home_win"])

    # --- candidate 1: linear refit on decile-bucketed means (§9.4's TD_PROB_CALIB pattern) ---
    calib_bucketed = calib_set.copy()
    calib_bucketed["bucket"] = pd.qcut(calib_bucketed["recentered_wp"], 10, labels=False, duplicates="drop")
    bucket_means = calib_bucketed.groupby("bucket").agg(
        mean_predicted=("recentered_wp", "mean"), actual_win_rate=("actual_home_win", "mean")
    )
    a_lin, b_lin = fit_linear(bucket_means["mean_predicted"], bucket_means["actual_win_rate"])
    holdout_linear = np.clip(a_lin + b_lin * holdout_set["recentered_wp"], 0.0, 1.0)
    brier_linear = brier(holdout_linear, holdout_set["actual_home_win"])

    # --- candidate 2: isotonic ---
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(calib_set["recentered_wp"], calib_set["actual_home_win"])
    holdout_iso = iso.predict(holdout_set["recentered_wp"])
    brier_iso = brier(holdout_iso, holdout_set["actual_home_win"])

    # --- candidate 3: Beta calibration (Kull et al., §9.4's 3-parameter middle ground) ---
    eps = 1e-6
    p_calib = calib_set["recentered_wp"].clip(eps, 1 - eps)
    X_calib = np.column_stack([np.log(p_calib), -np.log(1 - p_calib)])
    beta_clf = LogisticRegression(penalty=None)
    beta_clf.fit(X_calib, calib_set["actual_home_win"])
    p_holdout = holdout_set["recentered_wp"].clip(eps, 1 - eps)
    X_holdout = np.column_stack([np.log(p_holdout), -np.log(1 - p_holdout)])
    holdout_beta = beta_clf.predict_proba(X_holdout)[:, 1]
    brier_beta = brier(holdout_beta, holdout_set["actual_home_win"])

    print("\n=== STRICT HOLDOUT COMPARISON (fit on CALIB=2022-2023, scored on HOLDOUT=2024-2025) ===")
    print(f"Brier: no calibration={brier_none:.4f}  linear-refit={brier_linear:.4f}  "
          f"isotonic={brier_iso:.4f}  beta={brier_beta:.4f}")
    best = min(
        [("no calibration", brier_none), ("linear-refit", brier_linear), ("isotonic", brier_iso), ("beta", brier_beta)],
        key=lambda x: x[1],
    )
    print(f"best: {best[0]} ({best[1]:.4f})")

    print("\nreliability, HOLDOUT, no calibration (the pattern this investigation was chasing):")
    print(bucket_table(holdout_set["recentered_wp"], holdout_set["actual_home_win"], "recentered_wp").round(3))

    print(f"\nreliability, HOLDOUT, {best[0]} (the winner above):")
    winner_pred = {"no calibration": holdout_set["recentered_wp"], "linear-refit": holdout_linear,
                   "isotonic": holdout_iso, "beta": holdout_beta}[best[0]]
    print(bucket_table(pd.Series(winner_pred, index=holdout_set.index), holdout_set["actual_home_win"], "pred").round(3))

    # --- Frozen production constant: refit the winning (linear) map on ALL of CALIB+HOLDOUT
    # (2022-2025) -- the strict CALIB->HOLDOUT split above already answered "does this
    # generalize out-of-sample," so the shipped constant can use every available data point,
    # same convention as this project's other frozen coefficients (JOINT_COEFS_FORWARD,
    # SWAP_B_MARKET). Frozen, not auto-refit every run, because doing this safely without
    # leakage in production would require re-simulating years of history walk-forward on
    # every pipeline run (the drive pools aren't walk-forward internally, unlike the share/
    # rate engines) -- not worth the runtime cost for a ~1% Brier improvement.
    all_bucketed = valid.copy()
    all_bucketed["bucket"] = pd.qcut(all_bucketed["recentered_wp"], 10, labels=False, duplicates="drop")
    all_bucket_means = all_bucketed.groupby("bucket").agg(
        mean_predicted=("recentered_wp", "mean"), actual_win_rate=("actual_home_win", "mean")
    )
    a_final, b_final = fit_linear(all_bucket_means["mean_predicted"], all_bucket_means["actual_win_rate"])
    print(f"\nFROZEN production constant (fit on all CALIB+HOLDOUT=2022-2025, n={len(valid)}): "
          f"WIN_PROB_CALIB_A={a_final:.4f}, WIN_PROB_CALIB_B={b_final:.4f}")
    print(f"  calibrated_wp = clip({a_final:.4f} + {b_final:.4f} * recentered_win_prob, 0, 1)")
