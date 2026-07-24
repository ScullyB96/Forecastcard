"""Cycle 7: tests whether a goalie's own GSAx history should carry over
across the off-season (reset_each_season=False in add_walk_forward_mean)
instead of resetting to a cold start every season (Cycle 5's original,
flagged-as-a-simplification behavior). Isolates exactly this one variable:
same K=12 shrinkage prior, same relative-to-league-average adjustment, same
everything else as goalie_overlay_poisson.

Per the policy established in MODEL_DOCUMENTATION.md Sec4.8/4.9, this and
every future cycle validates against the DEVELOPMENT SET ONLY (seasons
2010-11 through 2023-24) -- the 2024-25/2025-26 holdout is not used for
this comparison.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.team_strength_goalie import (
    GOALIE_ADJUSTMENT_FLOOR, PRIOR_GAMES_GOALIE, add_walk_forward_goalie_strength,
    build_primary_goalie_per_team_game,
)
from src.models.validate_situational_toi import run_validation as run_situational_toi
from src.utils.paths import DATA_RAW


def run_validation(reset_each_season: bool, min_season: int = 20102011, max_season: int = DEV_MAX_SEASON) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    base = run_situational_toi(min_season=min_season)
    base = base[base["season"] < max_season]

    goalie_log = build_primary_goalie_per_team_game(schedule)
    goalie_log = add_walk_forward_goalie_strength(goalie_log, prior_games=PRIOR_GAMES_GOALIE,
                                                   reset_each_season=reset_each_season)
    goalie_log["goalie_relative"] = (goalie_log["goalie_shrunk_mean"] - goalie_log["goalie_league_avg"]).fillna(0.0)
    goalie_ratings = goalie_log[["gameId", "team", "goalie_relative"]]

    sched_teams = schedule[["gameId", "homeTeamAbbrev", "awayTeamAbbrev"]].drop_duplicates()
    merged = base.merge(sched_teams, on="gameId", how="left")
    merged = merged.merge(goalie_ratings.rename(columns={"team": "homeTeamAbbrev", "goalie_relative": "home_goalie"}),
                           on=["gameId", "homeTeamAbbrev"], how="left")
    merged = merged.merge(goalie_ratings.rename(columns={"team": "awayTeamAbbrev", "goalie_relative": "away_goalie"}),
                           on=["gameId", "awayTeamAbbrev"], how="left")
    merged["home_goalie"] = merged["home_goalie"].fillna(0.0)
    merged["away_goalie"] = merged["away_goalie"].fillna(0.0)

    results = []
    for row in merged.itertuples():
        adj_home = max(row.lambda_home - row.away_goalie, GOALIE_ADJUSTMENT_FLOOR)
        adj_away = max(row.lambda_away - row.home_goalie, GOALIE_ADJUSTMENT_FLOOR)
        joint = score_distribution(adj_home, adj_away)
        hp, ap, tp = home_win_prob_regulation(joint)
        results.append({
            "gameId": row.gameId, "gameDate": row.gameDate, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": adj_home, "lambda_away": adj_away,
            "home_win_prob": hp, "away_win_prob": ap, "tie_prob": tp,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    r_cross = run_validation(reset_each_season=False)
    print(f"validated {len(r_cross)} DEV-SET games (cross-season goalie history)")
    row = append_run(
        r_cross, model_name="goalie_overlay_cross_season_poisson",
        config_flags={"prior_games_goalie": PRIOR_GAMES_GOALIE, "reset_each_season": False,
                       "split": "development", "dev_max_season": DEV_MAX_SEASON},
        notes="Cycle 7: goalie GSAx history carries over across the off-season instead of resetting each "
              "season, same K=12 prior as goalie_overlay_poisson. Dev-set-only per Sec4.8/4.9 policy. "
              "See MODEL_DOCUMENTATION.md Sec4.10.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
