"""Does a head coach's career experience (games coached as HC, any team, tracked by name --
real data already in schedules via `home_coach`/`away_coach`, never used) predict the market
residual? Distinct from the already-tested-and-rejected "coaching-change effect" (§13.2,
which tested whether a team having JUST CHANGED coaches shifts its margin) -- this tests a
continuous experience/tenure signal instead: does a rookie HC (first season, still learning
in-game management -- clock management, timeout usage, 4th-down/2-point aggressiveness
calibration) underperform relative to what the market already prices in, and does that fade
with real career experience?

Walk-forward by construction: `career_games` for a coach at a given game only counts games
BEFORE it (tracked chronologically across ALL that coach's team stops, not reset on a team
change -- a coach who changes teams keeps his real career experience). TRAIN=2018-2021 fit,
TEST=2022-2025 held out, same discipline as every other check in this project.

**Left-censoring bug, found and fixed (2026-08, external review).** The first version of
this script started the `career_games` counter empty at 2016 (this project's own cached data
horizon) -- which silently censored every coach hired before 2016 (Reid, Belichick, Tomlin,
Carroll, etc.) to just 2-5 apparent years of experience heading into TRAIN=2018-2021,
compressing the true 0-400+ game range down to roughly 0-160 and risking exactly the kind of
mismeasurement that can manufacture a spurious effect or mask a real one. Real nflverse
schedule data (via `nflreadpy`) goes back to 1999 -- fetched 1999-2015 (schedules only, for
`home_coach`/`away_coach`, not full play-by-play) specifically to seed real pre-2016 tenure
before this script's TRAIN window starts, closing the censoring gap rather than just caveating
it.
"""

import numpy as np
import pandas as pd

from src.models.scoring import crps_gaussian, fit_residual_sigma, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
ROOKIE_MAX_GAMES = 16  # first real season as HC
TEAM_ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}  # real relocations, pre-2016 data uses old codes


def fit_ols_tstat(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    a, b = fit_linear(x, y)
    resid = y.to_numpy() - (a + b * x.to_numpy())
    n = len(x)
    if n <= 2 or np.allclose(x, x.iloc[0]):
        return a, b, 0.0
    sigma2 = (resid ** 2).sum() / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se_b = np.sqrt(sigma2 / sxx) if sxx > 0 else np.inf
    return a, b, (b / se_b if se_b > 0 else 0.0)


def build_coach_experience(schedules: pd.DataFrame) -> pd.DataFrame:
    """schedules should span from real history (1999+) through the analysis window so the
    career_games counter isn't left-censored -- only rows with game_id present in the
    caller's actual analysis set get returned with their (correctly seeded) experience."""
    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg["home_team"] = reg["home_team"].replace(TEAM_ALIAS)
    reg["away_team"] = reg["away_team"].replace(TEAM_ALIAS)
    reg = reg.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    career_games: dict[str, int] = {}
    home_exp, away_exp = [], []
    for row in reg.itertuples():
        home_exp.append(career_games.get(row.home_coach, 0))
        away_exp.append(career_games.get(row.away_coach, 0))
        career_games[row.home_coach] = career_games.get(row.home_coach, 0) + 1
        career_games[row.away_coach] = career_games.get(row.away_coach, 0) + 1
    reg["home_coach_exp"] = home_exp
    reg["away_coach_exp"] = away_exp
    return reg


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules_2016 = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    schedules_pre2016 = pd.read_parquet(DATA_RAW / "schedules_1999_2015_coachonly.parquet")
    common_cols = ["season", "week", "game_id", "game_type", "home_team", "away_team", "home_coach", "away_coach"]
    schedules = pd.concat([schedules_pre2016[common_cols], schedules_2016[common_cols]], ignore_index=True)
    print(f"seeded coach-experience counter from {schedules['season'].min()} "
          f"(closes the left-censoring gap -- previous version started counting at 2016)")

    exp = build_coach_experience(schedules)[
        ["game_id", "home_coach_exp", "away_coach_exp"]
    ]
    market = schedules_2016[schedules_2016["game_type"] == "REG"][["game_id", "spread_line", "total_line"]]
    exp = exp.merge(market, on="game_id", how="inner")  # restrict to the 2016+ window this analysis actually scores
    df = layer1.merge(exp, on="game_id", how="left").dropna(subset=["spread_line", "total_line"])
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["market_margin"] = df["spread_line"]
    df["resid_margin"] = df["actual_margin"] - df["market_margin"]
    df["resid_total"] = df["actual_total"] - df["total_line"]
    df["exp_diff"] = df["home_coach_exp"] - df["away_coach_exp"]
    df["home_rookie"] = (df["home_coach_exp"] < ROOKIE_MAX_GAMES).astype(float)
    df["away_rookie"] = (df["away_coach_exp"] < ROOKIE_MAX_GAMES).astype(float)
    df["rookie_diff"] = df["away_rookie"] - df["home_rookie"]  # positive = home has the more experienced coach
    df["either_rookie"] = ((df["home_rookie"] == 1) | (df["away_rookie"] == 1)).astype(float)

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n TRAIN={len(train)}  n TEST={len(test)}")
    print(f"rookie-coach games in TEST: home={int(test['home_rookie'].sum())}  away={int(test['away_rookie'].sum())}  "
          f"either={int(test['either_rookie'].sum())}")
    sigma_m = fit_residual_sigma(train["market_margin"], train["actual_margin"])

    print("\n=== MARGIN: resid_margin ~ continuous experience_diff (career games, home - away) ===")
    a1, b1, t1 = fit_ols_tstat(train["exp_diff"], train["resid_margin"])
    print(f"TRAIN fit: intercept={a1:+.5f}  slope={b1:+.5f}  t-stat={t1:+.2f}")
    test["adj1"] = test["market_margin"] + a1 + b1 * test["exp_diff"]
    mae0 = (test["market_margin"] - test["actual_margin"]).abs().mean()
    mae1 = (test["adj1"] - test["actual_margin"]).abs().mean()
    crps0 = crps_gaussian(test["market_margin"], test["actual_margin"], sigma_m)["crps"]
    crps1 = crps_gaussian(test["adj1"], test["actual_margin"], sigma_m)["crps"]
    print(f"TEST MAE {mae0:.4f} -> {mae1:.4f}   CRPS {crps0:.4f} -> {crps1:.4f}")

    print("\n=== MARGIN: resid_margin ~ rookie_diff (binary: does home have the rookie-year coach?) ===")
    a2, b2, t2 = fit_ols_tstat(train["rookie_diff"], train["resid_margin"])
    print(f"TRAIN fit: intercept={a2:+.5f}  slope={b2:+.5f}  t-stat={t2:+.2f}")
    test["adj2"] = test["market_margin"] + a2 + b2 * test["rookie_diff"]
    mae2 = (test["adj2"] - test["actual_margin"]).abs().mean()
    crps2 = crps_gaussian(test["adj2"], test["actual_margin"], sigma_m)["crps"]
    print(f"TEST MAE {mae0:.4f} -> {mae2:.4f}   CRPS {crps0:.4f} -> {crps2:.4f}")

    print("\n=== MARGIN, either-team-has-a-rookie-coach subset ===")
    sub = test[test["either_rookie"] == 1.0]
    b_sub = signed_bias(sub["market_margin"], sub["actual_margin"])
    print(f"n={len(sub)}: signed_bias (market alone) = {b_sub['bias']:+.3f}  CI=[{b_sub['ci'][0]:+.3f},{b_sub['ci'][1]:+.3f}]")

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
