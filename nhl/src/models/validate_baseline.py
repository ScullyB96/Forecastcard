"""Walk-forward validation of the naive season-to-date raw-goals Poisson
baseline (baseline_naive_poisson.py) against every real regular-season game
in the dataset. This is the FLOOR run -- its numbers go in the metrics
ledger as model_name="naive_raw_goals_poisson", and every later refinement
(xG-based strength, goalie overlay, NegBin, ...) must be checked against
this same real-game set before being kept.

Also fits (rather than assumes) the home-ice multiplier placeholder from
league-wide real data, as a first honest empirical step -- see
`fit_home_ice_multiplier`.
"""

import numpy as np
import pandas as pd

from src.ingest.team_codes import normalize_moneypuck_team_code, normalize_nhl_api_team_code
from src.models.baseline_naive_poisson import (
    add_walk_forward_strength, build_team_game_log, home_win_prob_regulation,
    predict_game_lambdas, score_distribution,
)
from src.models.metrics_ledger import append_run
from src.utils.paths import DATA_RAW


def fit_home_ice_multiplier(log: pd.DataFrame) -> float:
    """Empirical home-ice goal multiplier: ratio of average home goals/game
    to average away goals/game across all real historical games. Replaces
    the HOME_ICE_GOAL_MULTIPLIER placeholder with a number actually measured
    on our own data, per MODEL_DOCUMENTATION.md Sec3.2's stated standard."""
    home_avg = log.loc[log["is_home"], "goals_for"].mean()
    away_avg = log.loc[~log["is_home"], "goals_for"].mean()
    return float(np.sqrt(home_avg / away_avg))  # split the ratio symmetrically across home/away, matching
    # predict_game_lambdas's symmetric multiplier/divisor application


def run_validation(min_season: int = 20102011) -> pd.DataFrame:
    """min_season: skip the earliest seasons in the dataset when scoring
    (2008-09/2009-10 are needed to warm up the trailing league-average, but
    are themselves the noisiest, least-warmed-up predictions -- scoring
    starts from 2010-11 on, so the "floor" number isn't dragged down by an
    unfair cold-start period every later model would suffer equally, but
    which isn't representative of the model's real steady-state behavior)."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    log = build_team_game_log(schedule)
    log = add_walk_forward_strength(log)
    log["goals_league_avg"] = log["goals_league_avg"].bfill()  # only affects the very first row(s)

    home_ice_multiplier = fit_home_ice_multiplier(log)

    games = log[log["is_home"]].copy()  # one row per real game (home side carries both teams' info via merge)
    away_side = log[~log["is_home"]][["gameId", "goals_attack_rate", "goals_defense_rate", "games_played_before"]]
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    games = games[games["season"] >= min_season]

    results = []
    for row in games.itertuples():
        lam_home, lam_away = predict_game_lambdas(
            row.goals_attack_rate_home, row.goals_defense_rate_home,
            row.goals_attack_rate_away, row.goals_defense_rate_away,
            row.goals_league_avg, home_ice_multiplier,
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
        r, model_name="naive_raw_goals_poisson",
        config_flags={"prior_games": 10, "home_ice": "fit_empirically", "min_season": 20102011},
        notes="Deliberately naive floor baseline: season-to-date raw goals, no xG, no situational split, "
              "no goalie overlay, no OT/SO submodel (tie mass renormalized out of win-prob metrics). "
              "See MODEL_DOCUMENTATION.md Sec4.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
