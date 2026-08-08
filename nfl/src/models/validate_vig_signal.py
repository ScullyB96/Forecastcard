"""Does the real market VIG (not just the spread/total line itself, but the actual prices
`home_spread_odds`/`away_spread_odds`/`over_odds`/`under_odds`, already sitting in the
schedules table, never used anywhere in this project) carry information beyond the line?

Real mechanism: sportsbooks shade prices away from a flat -110/-110 in response to betting
volume/sharp money on one side. If that shading reflects real information (not just balancing
their own book), the vig-implied probability should predict the actual ATS/O-U outcome beyond
a coin flip -- a genuinely different KIND of signal than anything else in this project (market
microstructure, not team/player modeling).

No fitting needed -- vig-implied probability is a probability by construction (remove the
vig's hold by normalizing home/away implied probabilities to sum to 1). This is a direct
calibration/edge check, not a regression: bucket by implied probability, compare to the real
outcome rate, and separately check whether "bet the side the vig favors when it deviates from
50%" would have shown a real, positive ATS%/O-U% edge, walk-forward-consistent (TRAIN used
only to eyeball the calibration table, TEST is the primary held-out check since there's no
coefficient to overfit here).
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED, DATA_RAW

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def american_to_prob(odds: pd.Series) -> pd.Series:
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def calibration_table(df, prob_col, actual_col, label):
    d = df.copy()
    d["bucket"] = pd.cut(d[prob_col], bins=[0, 0.45, 0.48, 0.52, 0.55, 1.0])
    t = d.groupby("bucket", observed=True).agg(mean_pred=(prob_col, "mean"), actual_rate=(actual_col, "mean"), n=(actual_col, "size"))
    print(f"\n{label} calibration:")
    print(t.round(3))


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    reg = schedules[schedules["game_type"] == "REG"][
        ["game_id", "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"]
    ].dropna()
    df = layer1.merge(reg, on="game_id", how="left").dropna(subset=["home_spread_odds", "away_spread_odds"])
    df["actual_total"] = df["home_score"] + df["away_score"]

    # remove the vig's hold: normalize home/away implied probs to sum to 1
    p_home_raw = american_to_prob(df["home_spread_odds"])
    p_away_raw = american_to_prob(df["away_spread_odds"])
    df["p_home_cover"] = p_home_raw / (p_home_raw + p_away_raw)
    df["home_covers"] = (df["actual_margin"] > df["spread_line"]).astype(float)  # push excluded below

    p_over_raw = american_to_prob(df["over_odds"])
    p_under_raw = american_to_prob(df["under_odds"])
    df["p_over"] = p_over_raw / (p_over_raw + p_under_raw)
    df["went_over"] = (df["actual_total"] > df["total_line"]).astype(float)

    df = df[~np.isclose(df["actual_margin"], df["spread_line"])]  # drop pushes for ATS framing

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n TRAIN={len(train)}  n TEST={len(test)}")
    print(f"\nhold (vig) stats: mean p_home_cover+p_away_cover (pre-normalize) = "
          f"{(american_to_prob(df['home_spread_odds']) + american_to_prob(df['away_spread_odds'])).mean():.4f}  "
          f"(1.0 = no hold, ~1.045 matches this project's own ~4.55% vig assumption elsewhere)")
    print(f"p_home_cover range: [{df['p_home_cover'].min():.3f}, {df['p_home_cover'].max():.3f}]  "
          f"mean={df['p_home_cover'].mean():.3f}  std={df['p_home_cover'].std():.3f}")

    calibration_table(train, "p_home_cover", "home_covers", "TRAIN spread-vig")
    calibration_table(test, "p_home_cover", "home_covers", "TEST spread-vig")

    print("\n=== Does 'bet the side the vig favors when it deviates from 50%' show a real edge? ===")
    for min_dev in [0.0, 0.02, 0.04, 0.06]:
        dev = (test["p_home_cover"] - 0.5).abs()
        sub = test[dev >= min_dev]
        picks_home = sub["p_home_cover"] > 0.5
        correct = np.where(picks_home, sub["home_covers"] == 1, sub["home_covers"] == 0)
        win_rate = correct.mean() if len(sub) else float("nan")
        n = len(sub)
        # bootstrap CI
        rng = np.random.default_rng(0)
        boots = [correct[rng.integers(0, n, n)].mean() for _ in range(2000)] if n else []
        ci = (np.quantile(boots, 0.025), np.quantile(boots, 0.975)) if boots else (float("nan"), float("nan"))
        print(f"  min |deviation from 50%|={min_dev:.2f}: n={n}  win_rate={win_rate:.4f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]")

    calibration_table(train, "p_over", "went_over", "TRAIN total-vig")
    calibration_table(test, "p_over", "went_over", "TEST total-vig")

    print("\n=== Does 'bet the side the OVER/UNDER vig favors when it deviates from 50%' show a real edge? ===")
    for min_dev in [0.0, 0.02, 0.04, 0.06]:
        dev = (test["p_over"] - 0.5).abs()
        sub = test[dev >= min_dev]
        picks_over = sub["p_over"] > 0.5
        correct = np.where(picks_over, sub["went_over"] == 1, sub["went_over"] == 0)
        win_rate = correct.mean() if len(sub) else float("nan")
        n = len(sub)
        rng = np.random.default_rng(1)
        boots = [correct[rng.integers(0, n, n)].mean() for _ in range(2000)] if n else []
        ci = (np.quantile(boots, 0.025), np.quantile(boots, 0.975)) if boots else (float("nan"), float("nan"))
        print(f"  min |deviation from 50%|={min_dev:.2f}: n={n}  win_rate={win_rate:.4f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]")

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
