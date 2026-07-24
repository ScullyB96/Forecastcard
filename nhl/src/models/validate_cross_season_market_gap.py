"""Direct mechanism check for the cross-season prior (Sec12): does it
actually narrow the front-loaded model-vs-market Brier gap (Sec11.1) that
motivated it? Re-runs the Sec11.1 game-number breakdown comparing
cross_season_weight=0 (current best) against a real candidate weight on the
identical 11,803 market-matched games.
"""

import sys

import pandas as pd

from src.models.baseline_naive_poisson import build_team_game_log
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate, full_win_probability
from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_rest import run_validation as run_full_chain
from src.utils.paths import DATA_RAW

GAME_NUMBER_BINS = [(1, 15), (16, 40), (41, 100)]


def _team_game_numbers(schedule: pd.DataFrame) -> pd.DataFrame:
    log = build_team_game_log(schedule)
    log = log.sort_values(["team", "season", "gameDate", "gameId"]).reset_index(drop=True)
    log["team_game_number"] = log.groupby(["team", "season"]).cumcount() + 1
    return log[["gameId", "team", "is_home", "team_game_number"]]


if __name__ == "__main__":
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

    odds = pd.read_parquet(DATA_RAW / "sbro_odds_games.parquet")
    odds = odds.dropna(subset=["gameId", "close_home_prob_novig"])[["gameId", "close_home_prob_novig"]]

    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    game_numbers = _team_game_numbers(schedule)
    home_gn = game_numbers[game_numbers["is_home"]][["gameId", "team_game_number"]].rename(
        columns={"team_game_number": "home_game_number"})
    away_gn = game_numbers[~game_numbers["is_home"]][["gameId", "team_game_number"]].rename(
        columns={"team_game_number": "away_game_number"})

    print(f"comparing weight=0.0 (baseline) vs weight={weight} on the {len(odds)}-game market-matched set\n")

    for w in (0.0, weight):
        r, _adj = run_full_chain(league_avg_halflife_games=HALFLIFE_GAMES,
                                  cross_season_weight=(None if w == 0.0 else w))
        r = r[r["season"] < DEV_MAX_SEASON].copy()
        r = r.merge(odds, on="gameId", how="inner").merge(home_gn, on="gameId", how="left").merge(
            away_gn, on="gameId", how="left")
        r["min_game_number"] = r[["home_game_number", "away_game_number"]].min(axis=1)

        actual_home_win = (r["actual_home"] > r["actual_away"]).astype(int)
        model_p = r["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
        market_p = r["close_home_prob_novig"].clip(1e-6, 1 - 1e-6)
        r["_gap"] = (model_p - actual_home_win) ** 2 - (market_p - actual_home_win) ** 2

        print(f"weight={w}: n={len(r)}")
        for lo, hi in GAME_NUMBER_BINS:
            sub = r[(r["min_game_number"] >= lo) & (r["min_game_number"] <= hi)]
            print(f"  {f'{lo}-{hi}':>10}: n={len(sub):>5}  gap={sub['_gap'].mean():+.5f}")
        print()
