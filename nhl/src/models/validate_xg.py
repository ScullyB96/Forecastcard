"""Walk-forward validation of the xG-based team-strength model (Cycle 2),
run on the IDENTICAL real-game set and metric definitions as
validate_baseline.py, so the ledger comparison is apples-to-apples. Only
the team-strength input changes (xG vs. raw goals) -- home-ice methodology
is reused unchanged from the baseline run to keep this a single-variable
test, per team_strength_xg.py's stated scope.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, predict_game_lambdas, score_distribution
from src.models.metrics_ledger import append_run
from src.models.team_strength_xg import add_walk_forward_xg_strength, build_team_game_xg_log
from src.models.validate_baseline import fit_home_ice_multiplier
from src.utils.paths import DATA_RAW


def run_validation(min_season: int = 20102011) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")

    log = build_team_game_xg_log(schedule, moneypuck)
    log = add_walk_forward_xg_strength(log)
    log["xg_league_avg"] = log["xg_league_avg"].bfill()

    # Home ice fit on real goals, identical methodology to the baseline run
    # (see module docstring) -- isolates xG-vs-goals as the only variable.
    home_ice_multiplier = fit_home_ice_multiplier(log)

    games = log[log["is_home"]].copy()
    away_side = log[~log["is_home"]][["gameId", "xg_attack_rate", "xg_defense_rate", "games_played_before"]]
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    games = games[games["season"] >= min_season]

    results = []
    for row in games.itertuples():
        lam_home, lam_away = predict_game_lambdas(
            row.xg_attack_rate_home, row.xg_defense_rate_home,
            row.xg_attack_rate_away, row.xg_defense_rate_away,
            row.xg_league_avg, home_ice_multiplier,
        )
        joint = score_distribution(lam_home, lam_away)
        home_win_p, away_win_p, tie_p = home_win_prob_regulation(joint)
        results.append({
            "gameId": row.gameId, "gameDate": row.gameDate, "season": row.season,
            "actual_home": row.goals_for, "actual_away": row.goals_against,
            "lambda_home": lam_home, "lambda_away": lam_away,
            "home_win_prob": home_win_p, "away_win_prob": away_win_p, "tie_prob": tie_p,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    r = run_validation()
    print(f"validated {len(r)} real games, seasons {sorted(r['season'].unique())}")
    row = append_run(
        r, model_name="xg_based_poisson",
        config_flags={"prior_games": 10, "home_ice": "fit_empirically_on_goals", "min_season": 20102011,
                       "xg_situation": "all"},
        notes="Cycle 2: identical to naive_raw_goals_poisson except team strength is built from MoneyPuck's "
              "whole-game xGoalsFor/xGoalsAgainst instead of raw goals. See MODEL_DOCUMENTATION.md Sec4.3.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
