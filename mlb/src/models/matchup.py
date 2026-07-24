"""Combine a batter's and a pitcher's true-talent rates into a matchup-
specific probability via the ODDS-RATIO method (Tango's preferred variant --
NOT log5, which has documented tail bias for extreme pairings, per Morey &
Cohen 2015 and Tango's own writeups on the topic).

odds(p) = p / (1-p). For batter rate b, pitcher-allowed rate p, league rate L:
  matchup_odds = odds(L) * (odds(b)/odds(L)) * (odds(p)/odds(L))
  matchup_prob = matchup_odds / (1 + matchup_odds)

Validated here on two well-understood, contrasting BINARY outcomes first
(strikeout: fast-stabilizing, pitcher-controlled; home_run: much noisier,
less pitcher-controlled per DIPS theory) before extending to the full
multi-category simulator -- deliberately incremental, same discipline as the
rest of this project.
"""

import numpy as np
import pandas as pd

from src.models.true_talent import build_pregame_rates
from src.utils.paths import DATA_PROCESSED


def odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


def prob_from_odds(o):
    return o / (1 + o)


def combine_odds_ratio(batter_rate, pitcher_rate, league_rate):
    matchup_odds = odds(league_rate) * (odds(batter_rate) / odds(league_rate)) * (odds(pitcher_rate) / odds(league_rate))
    return prob_from_odds(matchup_odds)


def build_matchup_probs(pa: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """One row per PA: the odds-ratio-combined matchup probability for
    `outcome`, plus whether it actually happened -- for calibration testing."""
    batter_result = build_pregame_rates(pa, "batter", outcome)[
        ["game_pk", "at_bat_number", "batter", "season", "pregame_rate", "outcome"]
    ].rename(columns={"pregame_rate": "batter_rate"})
    pitcher_result = build_pregame_rates(pa, "pitcher", outcome)[
        ["game_pk", "at_bat_number", "pitcher", "pregame_rate"]
    ].rename(columns={"pregame_rate": "pitcher_rate"})

    merged = batter_result.merge(pitcher_result, on=["game_pk", "at_bat_number"], how="inner")

    # league rate: same walk-forward-safe (prior-seasons-only) logic used
    # inside true_talent.py's own shrinkage, applied here as the 3rd odds-ratio input.
    league_by_season = {}
    for season in sorted(pa["season"].unique()):
        prior = pa[pa["season"] < season]
        league_by_season[season] = (prior["outcome"] == outcome).mean() if len(prior) else (pa[pa["season"] == season]["outcome"] == outcome).mean()
    merged["league_rate"] = merged["season"].map(league_by_season)

    merged["matchup_prob"] = combine_odds_ratio(merged["batter_rate"], merged["pitcher_rate"], merged["league_rate"])
    merged["actual"] = (merged["outcome"] == outcome).astype(int)
    return merged


def calibration_report(merged: pd.DataFrame, test_seasons: set[int], n_buckets: int = 10) -> pd.DataFrame:
    test = merged[merged["season"].isin(test_seasons)].copy()
    test["bucket"] = pd.qcut(test["matchup_prob"], n_buckets, labels=False, duplicates="drop")
    report = test.groupby("bucket").agg(
        mean_predicted=("matchup_prob", "mean"), actual_rate=("actual", "mean"), n=("actual", "size")
    )
    return report


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    TEST_SEASONS = {2024, 2025}

    for outcome in ["strikeout", "home_run"]:
        print(f"\n=== {outcome}: odds-ratio matchup calibration (test seasons {sorted(TEST_SEASONS)}) ===")
        merged = build_matchup_probs(pa, outcome)
        report = calibration_report(merged, TEST_SEASONS)
        print(report.round(4))
        corr = report["mean_predicted"].corr(report["actual_rate"])
        b, a = np.polyfit(report["mean_predicted"].to_numpy(dtype=float), report["actual_rate"].to_numpy(dtype=float), 1)
        print(f"bucket-level calibration: actual = {a:.4f} + {b:.4f}*predicted (slope 1.0 = perfect)  corr={corr:.4f}")

        test = merged[merged["season"].isin(TEST_SEASONS)]
        mae = (test["matchup_prob"] - test["actual"]).abs().mean()
        naive_mae = (test["actual"].mean() - test["actual"]).abs().mean()
        print(f"PA-level MAE: matchup={mae:.4f}  naive(league rate only)={naive_mae:.4f}")
