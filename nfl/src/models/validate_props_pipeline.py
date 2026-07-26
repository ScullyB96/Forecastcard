"""End-to-end validation of the props pipeline: not "does target share
correlate" (already validated in player_usage.py) but "when you run the WHOLE
pipeline -- team plays x pass rate x player share -> projected targets/
carries, TD rate -> TD probability -- on a real held-out period, how close
are the numbers to what actually happened?" Component validation isn't
pipeline validation, and this closes that gap with the same discipline
(walk-forward, train/test split, calibration curves) applied to margin.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.game_environment import PassRateEngine, build_full_game_pass_rate, build_pass_rate_table
from src.models.player_usage import ShareEngine, TdRateEngine, build_player_week_shares, fit_empirical_bayes_prior_weight
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
# LEAGUE_AVG_PLAYS computed fresh from TRAIN below (review round 2, #4.2 stale-coupling
# sweep) -- was a hardcoded 62.859, duplicated identically in weekly_update.py and
# predict_props_2026.py, never recomputed; real play volume has drifted (2016-2025 mean
# 62.40 vs. 2024-2025-only 61.04). Fit on TRAIN only here, matching walk-forward discipline.


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    pbp = pd.read_parquet(DATA_RAW / "pbp_2016_2025.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")

    train_plays_pbp = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run"]) & pbp["season"].isin(TRAIN_SEASONS)]
    LEAGUE_AVG_PLAYS = train_plays_pbp.groupby(["season", "week", "posteam"]).size().mean()
    print(f"LEAGUE_AVG_PLAYS (TRAIN={min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}): {LEAGUE_AVG_PLAYS:.2f}\n")

    # --- rebuild walk-forward state fresh, consistent with all fixes this session ---
    shares = build_player_week_shares(weekly)
    tgt_engine = ShareEngine()
    tgt_result = tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    carry_engine = ShareEngine()
    carry_result = carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")

    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()
    # EB-fit prior_weight (review round 2, #1.1 / #1.2), fit on TRAIN only -- the live
    # pipeline (weekly_update.py) fits this on all available history since it's a forward
    # production hyperparameter, but this script's whole purpose is an honest TEST-set
    # check, so the hyperparameter itself must not see TEST, same walk-forward discipline
    # as every other fitted quantity here.
    train_shares = shares[shares["season"].isin(TRAIN_SEASONS)]
    train_targets = train_shares[train_shares["targets"] > 0]
    train_carries = train_shares[train_shares["carries"] > 0]
    career_rec_td = train_targets.groupby("player_id").agg(touches=("targets", "sum"), hits=("receiving_tds", "sum"))
    career_rush_td = train_carries.groupby("player_id").agg(touches=("carries", "sum"), hits=("rushing_tds", "sum"))
    rec_td_pw = fit_empirical_bayes_prior_weight(
        career_rec_td["touches"], career_rec_td["hits"], league_rec_td_rate * (1 - league_rec_td_rate)
    )["prior_weight"]
    rush_td_pw = fit_empirical_bayes_prior_weight(
        career_rush_td["touches"], career_rush_td["hits"], league_rush_td_rate * (1 - league_rush_td_rate)
    )["prior_weight"]
    print(f"EB-fit prior_weight (TRAIN only): rec_td={rec_td_pw:.1f}  rush_td={rush_td_pw:.1f}\n")
    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=rec_td_pw)
    rec_td_result = rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")
    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=rush_td_pw)
    rush_td_result = rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    pr_table = build_pass_rate_table(pbp)
    pr_engine = PassRateEngine()
    pr_result = pr_engine.run_walk_forward(pr_table)

    # margin prediction per team-game (for the game-script pass-rate adjustment)
    train_l1 = layer1[layer1["season"].isin(TRAIN_SEASONS)]
    base_a, base_b = fit_linear(train_l1["pregame_rating_diff"], train_l1["actual_margin"])
    layer1 = layer1.copy()
    layer1["base_pred"] = base_a + base_b * layer1["pregame_rating_diff"]
    home_margin = layer1.rename(columns={"team_a": "team"})[["season", "week", "game_id", "team", "base_pred"]]
    away_margin = layer1.rename(columns={"team_b": "team"})[["season", "week", "game_id", "team"]]
    away_margin["base_pred"] = -layer1["base_pred"].to_numpy()
    team_margin = pd.concat([home_margin, away_margin])

    # game-script adjustment coefficients (refit fresh, consistent with game_environment.py methodology)
    full_pr = build_full_game_pass_rate(pbp)
    sched_reg = schedules[schedules["game_type"] == "REG"][["game_id", "home_team", "away_team", "home_score", "away_score"]]
    full_pr = full_pr.merge(sched_reg, on="game_id", how="inner")
    full_pr["team_margin"] = np.where(
        full_pr["team"] == full_pr["home_team"], full_pr["home_score"] - full_pr["away_score"],
        full_pr["away_score"] - full_pr["home_score"],
    )
    full_pr = full_pr.merge(
        pr_result[["game_id", "team", "pregame_pass_rate_pred"]], on=["game_id", "team"], how="inner"
    )
    full_pr["pass_rate_residual"] = full_pr["full_pass_rate"] - full_pr["pregame_pass_rate_pred"]
    train_mask = full_pr["season"].isin(TRAIN_SEASONS)
    # BUG FIX 2026-07: was `script_slope, script_intercept = np.polyfit(...)[::-1]`,
    # which swaps slope/intercept -- see src/utils/stats.py's fit_linear docstring.
    script_intercept, script_slope = fit_linear(
        full_pr.loc[train_mask, "team_margin"], full_pr.loc[train_mask, "pass_rate_residual"]
    )

    # --- assemble player-game projections ---
    tgt_result = tgt_result.rename(columns={"recent_team": "team"})
    carry_result = carry_result.rename(columns={"recent_team": "team"})
    proj = tgt_result[["season", "week", "player_id", "player_display_name", "team", "involved",
                        "pregame_target_share_calc", "targets"]].merge(
        carry_result[["season", "week", "player_id", "team", "pregame_carry_share_calc", "carries"]],
        on=["season", "week", "player_id", "team"], how="outer"
    )
    proj = proj.merge(
        rec_td_result[["season", "week", "player_id", "pregame_receiving_tds_rate", "receiving_tds"]],
        on=["season", "week", "player_id"], how="left"
    )
    proj = proj.merge(
        rush_td_result[["season", "week", "player_id", "pregame_rushing_tds_rate", "rushing_tds"]],
        on=["season", "week", "player_id"], how="left"
    )
    proj = proj.merge(team_margin, on=["season", "week", "team"], how="left")
    proj = proj.merge(
        pr_result[["season", "week", "team", "pregame_pass_rate_pred"]], on=["season", "week", "team"], how="left"
    )
    proj = proj.dropna(subset=["base_pred", "pregame_pass_rate_pred"])

    proj["adj_pass_rate"] = (proj["pregame_pass_rate_pred"] + script_intercept + script_slope * proj["base_pred"]).clip(0.30, 0.80)
    proj["proj_targets"] = LEAGUE_AVG_PLAYS * proj["adj_pass_rate"] * proj["pregame_target_share_calc"].fillna(0)
    proj["proj_carries"] = LEAGUE_AVG_PLAYS * (1 - proj["adj_pass_rate"]) * proj["pregame_carry_share_calc"].fillna(0)
    proj["proj_rec_tds"] = proj["proj_targets"] * proj["pregame_receiving_tds_rate"].fillna(league_rec_td_rate)
    proj["proj_rush_tds"] = proj["proj_carries"] * proj["pregame_rushing_tds_rate"].fillna(league_rush_td_rate)
    proj["td_prob"] = 1 - np.exp(-(proj["proj_rec_tds"] + proj["proj_rush_tds"]))
    proj["actual_td"] = ((proj["receiving_tds"].fillna(0) + proj["rushing_tds"].fillna(0)) >= 1).astype(int)
    proj["targets"] = proj["targets"].fillna(0)
    proj["carries"] = proj["carries"].fillna(0)

    test = proj[proj["season"].isin(TEST_SEASONS)]

    print("=== VOLUME PROJECTIONS: full pipeline vs actual, test seasons 2022-2025 ===\n")
    # only meaningfully-involved player-games (avoid diluting with garbage-time/inactive noise)
    tgt_test = test[test["pregame_target_share_calc"] > 0.03]
    mae_t = (tgt_test["proj_targets"] - tgt_test["targets"]).abs().mean()
    naive_t = (tgt_test["targets"] - tgt_test["targets"].mean()).abs().mean()
    corr_t = np.corrcoef(tgt_test["proj_targets"], tgt_test["targets"])[0, 1]
    print(f"targets:  MAE={mae_t:.2f}  naive(predict mean)={naive_t:.2f}  corr={corr_t:.3f}  n={len(tgt_test)}")

    carry_test = test[test["pregame_carry_share_calc"] > 0.03]
    mae_c = (carry_test["proj_carries"] - carry_test["carries"]).abs().mean()
    naive_c = (carry_test["carries"] - carry_test["carries"].mean()).abs().mean()
    corr_c = np.corrcoef(carry_test["proj_carries"], carry_test["carries"])[0, 1]
    print(f"carries:  MAE={mae_c:.2f}  naive(predict mean)={naive_c:.2f}  corr={corr_c:.3f}  n={len(carry_test)}")

    print("\n=== TD PROBABILITY CALIBRATION (decile buckets) ===\n")
    relevant = test[(test["pregame_target_share_calc"] > 0.03) | (test["pregame_carry_share_calc"] > 0.03)].copy()
    relevant["bucket"] = pd.qcut(relevant["td_prob"], 10, labels=False, duplicates="drop")
    calib = relevant.groupby("bucket").agg(
        mean_predicted=("td_prob", "mean"), actual_td_rate=("actual_td", "mean"), n=("actual_td", "size")
    )
    print(calib.round(3))
    calib_corr = np.corrcoef(calib["mean_predicted"], calib["actual_td_rate"])[0, 1]
    a_cal, b_cal = fit_linear(calib["mean_predicted"], calib["actual_td_rate"])
    print(f"\nbucket-level calibration: actual = {a_cal:.3f} + {b_cal:.3f}*predicted (slope 1.0 = perfect)  corr={calib_corr:.3f}")

    print("\n=== S4.1 (review round 3, #5): isotonic calibration vs. the linear refit ===")
    print("Linear refit (a_cal, b_cal above) can only stretch/shift the whole curve uniformly --")
    print("structurally can't fix a top-decile-specific overshoot. Isotonic can. Fit on TRAIN")
    print("only here (matching this script's walk-forward discipline), applied to TEST.")
    train_relevant = proj[proj["season"].isin(TRAIN_SEASONS)]
    train_relevant = train_relevant[
        (train_relevant["pregame_target_share_calc"] > 0.03) | (train_relevant["pregame_carry_share_calc"] > 0.03)
    ]
    train_calib_bucket = train_relevant.copy()
    train_calib_bucket["bucket"] = pd.qcut(train_calib_bucket["td_prob"], 10, labels=False, duplicates="drop")
    train_bucket_means = train_calib_bucket.groupby("bucket").agg(
        mean_predicted=("td_prob", "mean"), actual_td_rate=("actual_td", "mean")
    )
    a_cal_train, b_cal_train = fit_linear(train_bucket_means["mean_predicted"], train_bucket_means["actual_td_rate"])
    relevant["td_prob_linear"] = np.clip(a_cal_train + b_cal_train * relevant["td_prob"], 0.0, 1.0)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train_relevant["td_prob"], train_relevant["actual_td"])
    relevant["td_prob_isotonic"] = iso.predict(relevant["td_prob"])

    # Beta calibration (review round 4, #7): 3-parameter monotone map, developed for exactly
    # this situation -- small samples, rare events, needs more shape than linear but not
    # isotonic's unconstrained degrees of freedom. logit(calibrated) = a*log(p) + b*log(1-p) + c,
    # fit via logistic regression on the two log-transformed features (a real MLE fit against
    # the binary outcome, not an OLS fit against bucket means).
    eps = 1e-6
    p_train = train_relevant["td_prob"].clip(eps, 1 - eps)
    X_train = np.column_stack([np.log(p_train), -np.log(1 - p_train)])
    beta_clf = LogisticRegression(penalty=None)
    beta_clf.fit(X_train, train_relevant["actual_td"])
    p_all = relevant["td_prob"].clip(eps, 1 - eps)
    X_all = np.column_stack([np.log(p_all), -np.log(1 - p_all)])
    relevant["td_prob_beta"] = beta_clf.predict_proba(X_all)[:, 1]

    iso_calib = relevant.groupby("bucket").agg(
        mean_predicted_linear=("td_prob_linear", "mean"),
        mean_predicted_isotonic=("td_prob_isotonic", "mean"),
        mean_predicted_beta=("td_prob_beta", "mean"),
        actual_td_rate=("actual_td", "mean"),
        n=("actual_td", "size"),
    )
    print(iso_calib.round(3))
    top_decile = iso_calib.iloc[-1]
    print(f"\ntop decile: linear-refit predicted={top_decile['mean_predicted_linear']:.3f}  "
          f"isotonic predicted={top_decile['mean_predicted_isotonic']:.3f}  "
          f"beta predicted={top_decile['mean_predicted_beta']:.3f}  actual={top_decile['actual_td_rate']:.3f}")
    print(f"top-decile overshoot: linear-refit={top_decile['mean_predicted_linear']-top_decile['actual_td_rate']:+.3f}  "
          f"isotonic={top_decile['mean_predicted_isotonic']-top_decile['actual_td_rate']:+.3f}  "
          f"beta={top_decile['mean_predicted_beta']-top_decile['actual_td_rate']:+.3f}")
    print("(strict walk-forward: isotonic UNDERPERFORMED the linear refit here -- a real,")
    print(" surprising result, plausibly isotonic overfitting TRAIN's specific tail noise with")
    print(" only 4 seasons to fit from. Decision rule (review round 4, #7): auto-refit linear")
    print(" ships as the default; isotonic or Beta only get promoted if one wins THIS strict")
    print(" walk-forward comparison outright, not a full-history one.)")

    print("\n=== S4.1 continued: same comparison, BOTH fit on ALL history ===")
    print("weekly_update.py's live isotonic calibrator is fit on ALL available history (not a")
    print("TRAIN/TEST split), matching this project's forward-production convention (e.g. the EB")
    print("prior_weight fit) -- this checks whether the walk-forward result above was really an")
    print("isotonic-vs-linear difference, or a small-sample overfitting artifact of the TRAIN-only fit.")
    all_relevant = proj[(proj["pregame_target_share_calc"] > 0.03) | (proj["pregame_carry_share_calc"] > 0.03)]
    all_bucket = all_relevant.copy()
    all_bucket["bucket"] = pd.qcut(all_bucket["td_prob"], 10, labels=False, duplicates="drop")
    all_bucket_means = all_bucket.groupby("bucket").agg(mean_predicted=("td_prob", "mean"), actual_td_rate=("actual_td", "mean"))
    a_cal_all, b_cal_all = fit_linear(all_bucket_means["mean_predicted"], all_bucket_means["actual_td_rate"])

    iso_all = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_all.fit(all_relevant["td_prob"], all_relevant["actual_td"])
    relevant["td_prob_linear_allhist"] = np.clip(a_cal_all + b_cal_all * relevant["td_prob"], 0.0, 1.0)
    relevant["td_prob_isotonic_allhist"] = iso_all.predict(relevant["td_prob"])
    allhist_calib = relevant.groupby("bucket").agg(
        mean_predicted_linear=("td_prob_linear_allhist", "mean"),
        mean_predicted_isotonic=("td_prob_isotonic_allhist", "mean"),
        actual_td_rate=("actual_td", "mean"),
    )
    print(allhist_calib.round(3))
    top_allhist = allhist_calib.iloc[-1]
    print(f"\ntop decile (both fit on all history): linear-refit overshoot="
          f"{top_allhist['mean_predicted_linear']-top_allhist['actual_td_rate']:+.3f}  "
          f"isotonic overshoot={top_allhist['mean_predicted_isotonic']-top_allhist['actual_td_rate']:+.3f}")
    print("(NOTE: this comparison is NOT a held-out test -- TEST's own player-weeks are part of")
    print(" the 'all history' fit set here, matching weekly_update.py's own live convention and")
    print(" its own caveats, e.g. the EB prior_weight fit. It answers 'does isotonic help when")
    print(" given enough data,' not 'does it generalize honestly out-of-sample' -- that question")
    print(" is what the strict walk-forward comparison above already answered, unfavorably for")
    print(" isotonic at a 4-season sample size.)")
