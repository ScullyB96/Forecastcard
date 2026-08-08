"""Does playoff "stakes" -- a team sitting right on the wild-card bubble, with real playoff
positioning on the line -- predict the market residual? Distinct from the already-rejected
"clinched/eliminated rest effect" (§13.2, which is specifically about teams RESTING starters
once mathematically out of it or already locked in); this tests the opposite population --
teams still fighting for a spot, hypothesized to play with either extra urgency (outperform)
or extra pressure (underperform) relative to what the market already prices in.

Real standings computed walk-forward from real game results already in schedules (no
lookahead: a team's win_pct/rank entering week W uses only games strictly before week W,
same season). Two real simplifications, disclosed rather than hidden:
  1. Tiebreak by real point differential, not the NFL's actual multi-step tiebreaker
     procedure (division record, common games, strength of victory, etc.) -- a defensible
     proxy for "who's ahead," not an attempt at exact seeding.
  2. No division-winner-vs-wild-card distinction -- just conference rank vs. the real
     number of playoff spots for that season's format (6 seeds/conference through 2019,
     7 from 2020 on).
Restricted to weeks 10-17 (standings are too noisy to mean anything before week 10; week 18
is deliberately excluded to avoid re-deriving the already-tested clinched/eliminated result).
"""

import numpy as np
import pandas as pd

from src.models.scoring import crps_gaussian, fit_residual_sigma, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
BUBBLE_WINDOW = 1  # "on the bubble" = within this many spots of the real cutoff line
MIN_WEEK, MAX_WEEK = 10, 17

TEAM_ALIAS = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
CONFERENCE = {
    "BUF": "AFC", "MIA": "AFC", "NE": "AFC", "NYJ": "AFC",
    "BAL": "AFC", "CIN": "AFC", "CLE": "AFC", "PIT": "AFC",
    "HOU": "AFC", "IND": "AFC", "JAX": "AFC", "TEN": "AFC",
    "DEN": "AFC", "KC": "AFC", "LV": "AFC", "LAC": "AFC",
    "DAL": "NFC", "NYG": "NFC", "PHI": "NFC", "WAS": "NFC",
    "CHI": "NFC", "DET": "NFC", "GB": "NFC", "MIN": "NFC",
    "ATL": "NFC", "CAR": "NFC", "NO": "NFC", "TB": "NFC",
    "ARI": "NFC", "LA": "NFC", "SEA": "NFC", "SF": "NFC",
}


def n_playoff_spots(season: int) -> int:
    return 7 if season >= 2020 else 6


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


def build_conference_ranks(schedules: pd.DataFrame) -> pd.DataFrame:
    """Standings entering each WEEK (not progressively within it) -- avoids any same-week
    simultaneous-game leakage: every team's rank for its week-W game reflects results only
    through week W-1, regardless of kickoff order within week W."""
    reg = schedules[schedules["game_type"] == "REG"].copy()
    reg["home_team"] = reg["home_team"].replace(TEAM_ALIAS)
    reg["away_team"] = reg["away_team"].replace(TEAM_ALIAS)
    reg = reg.sort_values(["season", "week"]).reset_index(drop=True)
    conf_teams = list(CONFERENCE.keys())

    record: dict[tuple[int, str], list[float]] = {}  # (season, team) -> [wins, losses, ties, point_diff]

    def get(season, team):
        return record.setdefault((season, team), [0.0, 0.0, 0.0, 0.0])

    rank_by_season_week_team: dict[tuple[int, int, str], int] = {}
    for (season, week), _ in reg.groupby(["season", "week"]):
        rows = []
        for t in conf_teams:
            w, l, t_, pd_ = get(season, t)
            games = w + l + t_
            win_pct = (w + 0.5 * t_) / games if games > 0 else -1.0  # sentinel: winless/gameless sorts last
            rows.append((t, win_pct, pd_))
        standings = pd.DataFrame(rows, columns=["team", "win_pct", "point_diff"])
        standings["conf"] = standings["team"].map(CONFERENCE)
        standings = standings.sort_values(["conf", "win_pct", "point_diff"], ascending=[True, False, False])
        standings["rank"] = standings.groupby("conf").cumcount() + 1
        for t, r in zip(standings["team"], standings["rank"]):
            rank_by_season_week_team[(season, week, t)] = r

        # now fold in this week's real results, for next week's standings
        week_games = reg[(reg["season"] == season) & (reg["week"] == week)]
        for row in week_games.itertuples():
            if pd.notna(row.home_score) and pd.notna(row.away_score):
                hw, hl, ht, hpd = get(season, row.home_team)
                aw, al, at, apd = get(season, row.away_team)
                diff = row.home_score - row.away_score
                if diff > 0:
                    hw += 1; al += 1
                elif diff < 0:
                    hl += 1; aw += 1
                else:
                    ht += 1; at += 1
                hpd += diff; apd -= diff
                record[(season, row.home_team)] = [hw, hl, ht, hpd]
                record[(season, row.away_team)] = [aw, al, at, apd]

    reg["home_conf_rank"] = [rank_by_season_week_team.get((s, w, t), np.nan) for s, w, t in zip(reg["season"], reg["week"], reg["home_team"])]
    reg["away_conf_rank"] = [rank_by_season_week_team.get((s, w, t), np.nan) for s, w, t in zip(reg["season"], reg["week"], reg["away_team"])]
    return reg


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    ranks = build_conference_ranks(schedules)[
        ["game_id", "season", "week", "spread_line", "total_line", "home_conf_rank", "away_conf_rank"]
    ]
    df = layer1.merge(ranks, on="game_id", how="left", suffixes=("", "_r"))
    df = df.dropna(subset=["spread_line", "total_line", "home_conf_rank", "away_conf_rank"])
    df = df[(df["week"] >= MIN_WEEK) & (df["week"] <= MAX_WEEK)]
    df["n_spots"] = df["season"].apply(n_playoff_spots)
    df["home_bubble"] = (df["home_conf_rank"] - df["n_spots"]).abs() <= BUBBLE_WINDOW
    df["away_bubble"] = (df["away_conf_rank"] - df["n_spots"]).abs() <= BUBBLE_WINDOW
    df["either_bubble"] = (df["home_bubble"] | df["away_bubble"]).astype(float)
    df["bubble_diff"] = df["home_bubble"].astype(float) - df["away_bubble"].astype(float)

    df["actual_total"] = df["home_score"] + df["away_score"]
    df["market_margin"] = df["spread_line"]
    df["resid_margin"] = df["actual_margin"] - df["market_margin"]
    df["resid_total"] = df["actual_total"] - df["total_line"]

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"n TRAIN={len(train)}  n TEST={len(test)} (weeks {MIN_WEEK}-{MAX_WEEK} only)")
    print(f"either-team-on-bubble games in TEST: {int(test['either_bubble'].sum())}")
    sigma_m = fit_residual_sigma(train["market_margin"], train["actual_margin"])
    sigma_t = fit_residual_sigma(train["total_line"], train["actual_total"])

    print("\n=== MARGIN: resid_margin ~ bubble_diff (does the bubble team over/underperform?) ===")
    a1, b1, t1 = fit_ols_tstat(train["bubble_diff"], train["resid_margin"])
    print(f"TRAIN fit: intercept={a1:+.4f}  slope={b1:+.4f}  t-stat={t1:+.2f}")
    test["adj_m"] = test["market_margin"] + a1 + b1 * test["bubble_diff"]
    mae0 = (test["market_margin"] - test["actual_margin"]).abs().mean()
    mae1 = (test["adj_m"] - test["actual_margin"]).abs().mean()
    crps0 = crps_gaussian(test["market_margin"], test["actual_margin"], sigma_m)["crps"]
    crps1 = crps_gaussian(test["adj_m"], test["actual_margin"], sigma_m)["crps"]
    print(f"TEST MAE {mae0:.4f} -> {mae1:.4f}   CRPS {crps0:.4f} -> {crps1:.4f}")

    print("\n=== MARGIN, either-team-on-bubble subset (does the game run closer/less close than the line implies?) ===")
    sub = test[test["either_bubble"] == 1.0]
    b_sub = signed_bias(sub["market_margin"], sub["actual_margin"])
    print(f"n={len(sub)}: signed_bias (market alone) = {b_sub['bias']:+.3f}  CI=[{b_sub['ci'][0]:+.3f},{b_sub['ci'][1]:+.3f}]")
    mae_sub = (sub["market_margin"] - sub["actual_margin"]).abs().mean()
    mae_all = (test["market_margin"] - test["actual_margin"]).abs().mean()
    print(f"MAE on bubble subset={mae_sub:.3f}  vs. all TEST games={mae_all:.3f} (tighter would mean more predictable, not less)")

    print("\n=== TOTAL: resid_total ~ either_bubble (do high-stakes games see different scoring?) ===")
    a2, b2, t2 = fit_ols_tstat(train["either_bubble"], train["resid_total"])
    print(f"TRAIN fit: intercept={a2:+.4f}  slope={b2:+.4f}  t-stat={t2:+.2f}")
    test["adj_t"] = test["total_line"] + a2 + b2 * test["either_bubble"]
    mae0_t = (test["total_line"] - test["actual_total"]).abs().mean()
    mae1_t = (test["adj_t"] - test["actual_total"]).abs().mean()
    print(f"TEST MAE {mae0_t:.4f} -> {mae1_t:.4f}")

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
