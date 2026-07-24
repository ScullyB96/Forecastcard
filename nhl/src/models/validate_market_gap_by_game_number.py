"""Residuals-first check (external review, follow-up 2026-07-23) for the
cross-season team-strength-prior hypothesis: the per-season Sec6.8 breakdown
showed its two largest model-vs-market Brier gaps in 2012-13 (the 48-game
lockout-shortened season, no normal training camp) and 2021-22 (the first
full-length season after two COVID-disrupted ones) -- both seasons where
teams' TRUE current strength was most disconnected from a season-reset
league-average prior, while the market opens with real preseason team-
specific information every season regardless. If that's the real
mechanism, the model-vs-market gap should be FRONT-LOADED within every
season (worst early, before enough current-season games accumulate to
overwhelm the reset prior), not flat across the season.

Bins games by the EARLIER team's within-season game count (min of home/away
team's own games-played-so-far), since a game is only as well-informed as
its LESS-experienced-this-season side.
"""

import pandas as pd

from src.models.baseline_naive_poisson import build_team_game_log
from src.models.validate_market_benchmark import run_validation as run_market_benchmark
from src.utils.paths import DATA_RAW

GAME_NUMBER_BINS = [(1, 15), (16, 40), (41, 100)]


def _team_game_numbers(schedule: pd.DataFrame) -> pd.DataFrame:
    log = build_team_game_log(schedule)
    log = log.sort_values(["team", "season", "gameDate", "gameId"]).reset_index(drop=True)
    log["team_game_number"] = log.groupby(["team", "season"]).cumcount() + 1
    return log[["gameId", "team", "is_home", "team_game_number"]]


if __name__ == "__main__":
    r = run_market_benchmark()

    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    game_numbers = _team_game_numbers(schedule)

    home_gn = game_numbers[game_numbers["is_home"]][["gameId", "team_game_number"]].rename(
        columns={"team_game_number": "home_game_number"})
    away_gn = game_numbers[~game_numbers["is_home"]][["gameId", "team_game_number"]].rename(
        columns={"team_game_number": "away_game_number"})
    r = r.merge(home_gn, on="gameId", how="left").merge(away_gn, on="gameId", how="left")
    r["min_game_number"] = r[["home_game_number", "away_game_number"]].min(axis=1)

    actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int)
    model_p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
    market_p = r["close_home_prob_novig"].clip(1e-6, 1 - 1e-6)
    r["_model_brier"] = (model_p - actual_home_win) ** 2
    r["_market_brier"] = (market_p - actual_home_win) ** 2
    r["_gap"] = r["_model_brier"] - r["_market_brier"]

    print(f"{len(r)} games with market data, min_game_number range "
          f"[{r['min_game_number'].min()}, {r['min_game_number'].max()}]")

    print("\noverall breakdown by game-number-in-season bin (both teams pooled, min of the two):")
    print(f"{'bin':>10} {'n':>6} {'model_brier':>12} {'market_brier':>13} {'gap':>9}")
    for lo, hi in GAME_NUMBER_BINS:
        sub = r[(r["min_game_number"] >= lo) & (r["min_game_number"] <= hi)]
        print(f"{f'{lo}-{hi}':>10} {len(sub):>6} {sub['_model_brier'].mean():>12.5f} "
              f"{sub['_market_brier'].mean():>13.5f} {sub['_gap'].mean():>+9.5f}")

    print("\nsame breakdown, split: the two flagged seasons (2012-13 lockout, 2021-22 post-COVID) "
          "vs. all other seasons:")
    flagged = r[r["season"].isin([20122013, 20212022])]
    other = r[~r["season"].isin([20122013, 20212022])]
    for label, sub_all in [("FLAGGED (2012-13, 2021-22)", flagged), ("OTHER seasons", other)]:
        print(f"\n  {label}: {len(sub_all)} games")
        for lo, hi in GAME_NUMBER_BINS:
            sub = sub_all[(sub_all["min_game_number"] >= lo) & (sub_all["min_game_number"] <= hi)]
            if len(sub) == 0:
                continue
            print(f"    {f'{lo}-{hi}':>10} n={len(sub):>5}  model={sub['_model_brier'].mean():.5f}  "
                  f"market={sub['_market_brier'].mean():.5f}  gap={sub['_gap'].mean():+.5f}")
