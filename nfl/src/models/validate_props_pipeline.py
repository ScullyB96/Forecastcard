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

from src.models.game_environment import PassRateEngine, build_full_game_pass_rate, build_pass_rate_table
from src.models.player_usage import ShareEngine, TdRateEngine, build_player_week_shares
from src.utils.paths import DATA_PROCESSED, DATA_RAW

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
LEAGUE_AVG_PLAYS = 62.859


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    pbp = pd.read_parquet(DATA_RAW / "pbp_2016_2025.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")

    # --- rebuild walk-forward state fresh, consistent with all fixes this session ---
    shares = build_player_week_shares(weekly)
    tgt_engine = ShareEngine()
    tgt_result = tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    carry_engine = ShareEngine()
    carry_result = carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")

    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()
    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=30.0)
    rec_td_result = rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")
    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=30.0)
    rush_td_result = rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    pr_table = build_pass_rate_table(pbp)
    pr_engine = PassRateEngine()
    pr_result = pr_engine.run_walk_forward(pr_table)

    # margin prediction per team-game (for the game-script pass-rate adjustment)
    train_l1 = layer1[layer1["season"].isin(TRAIN_SEASONS)]
    base_a, base_b = np.polyfit(train_l1["pregame_rating_diff"], train_l1["actual_margin"], 1)[::-1]
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
    script_slope, script_intercept = np.polyfit(
        full_pr.loc[train_mask, "team_margin"], full_pr.loc[train_mask, "pass_rate_residual"], 1
    )[::-1]

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
    b_cal, a_cal = np.polyfit(calib["mean_predicted"], calib["actual_td_rate"], 1)
    print(f"\nbucket-level calibration: actual = {a_cal:.3f} + {b_cal:.3f}*predicted (slope 1.0 = perfect)  corr={calib_corr:.3f}")
