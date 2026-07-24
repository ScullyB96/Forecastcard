"""Walk-forward validation of the situational (even-strength/PP/PK/other)
team-strength model (Cycle 3), run on the same real-game methodology as
Cycles 1-2 -- home-ice fit on real goals, ties reported/renormalized rather
than assumed, results logged to the same metrics ledger for a direct
comparison against xg_based_poisson.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.metrics_ledger import append_run
from src.models.team_strength_situational import (
    add_walk_forward_situational_strength, build_team_game_situational_log, predict_situational_lambda,
)
from src.models.validate_baseline import fit_home_ice_multiplier
from src.utils.paths import DATA_RAW


def run_validation(min_season: int = 20102011) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")

    log = build_team_game_situational_log(schedule, moneypuck)
    log = add_walk_forward_situational_strength(log)
    for col in ("ev_league_avg_for_per60", "ev_league_avg_against_per60",
                "pp_league_avg_for_per60", "pk_league_avg_against_per60",
                "other_league_avg_for_per60", "other_league_avg_against_per60"):
        log[col] = log[col].bfill()

    # League-average TOI/game per situation, walk-forward (trailing expanding
    # mean, no season reset) -- the shared combine-weight described in
    # team_strength_situational.py's module docstring.
    log["league_avg_ev_toi_min"] = log["ev_toi_min"].shift(1).expanding().mean().bfill()
    log["league_avg_pp_toi_min"] = log["pp_toi_min"].shift(1).expanding().mean().bfill()
    log["league_avg_other_toi_min"] = log["other_toi_min"].shift(1).expanding().mean().bfill()

    home_ice_multiplier = fit_home_ice_multiplier(log)

    games = log[log["is_home"]].copy()
    away_cols = ["gameId", "ev_attack_rate_per60", "ev_defense_rate_per60",
                 "ev_league_avg_for_per60", "ev_league_avg_against_per60",
                 "pp_attack_rate_per60", "pk_defense_rate_per60",
                 "pp_league_avg_for_per60", "pk_league_avg_against_per60",
                 "other_attack_rate_per60", "other_defense_rate_per60",
                 "other_league_avg_for_per60", "other_league_avg_against_per60"]
    away_side = log[~log["is_home"]][away_cols]
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    games = games[games["season"] >= min_season]

    results = []
    for row in games.itertuples():
        lam_home_raw, lam_away_raw = predict_situational_lambda(
            home_ev_attack=row.ev_attack_rate_per60_home, home_ev_league_avg_for=row.ev_league_avg_for_per60_home,
            home_ev_defense=row.ev_defense_rate_per60_home, home_ev_league_avg_against=row.ev_league_avg_against_per60_home,
            away_ev_attack=row.ev_attack_rate_per60_away, away_ev_league_avg_for=row.ev_league_avg_for_per60_away,
            away_ev_defense=row.ev_defense_rate_per60_away, away_ev_league_avg_against=row.ev_league_avg_against_per60_away,
            home_pp_attack=row.pp_attack_rate_per60_home, home_pp_league_avg_for=row.pp_league_avg_for_per60_home,
            away_pp_attack=row.pp_attack_rate_per60_away, away_pp_league_avg_for=row.pp_league_avg_for_per60_away,
            home_pk_defense=row.pk_defense_rate_per60_home, home_pk_league_avg_against=row.pk_league_avg_against_per60_home,
            away_pk_defense=row.pk_defense_rate_per60_away, away_pk_league_avg_against=row.pk_league_avg_against_per60_away,
            home_other_attack=row.other_attack_rate_per60_home, home_other_league_avg_for=row.other_league_avg_for_per60_home,
            home_other_defense=row.other_defense_rate_per60_home, home_other_league_avg_against=row.other_league_avg_against_per60_home,
            away_other_attack=row.other_attack_rate_per60_away, away_other_league_avg_for=row.other_league_avg_for_per60_away,
            away_other_defense=row.other_defense_rate_per60_away, away_other_league_avg_against=row.other_league_avg_against_per60_away,
            league_avg_ev_toi_min=row.league_avg_ev_toi_min,
            home_pp_toi_min=row.league_avg_pp_toi_min, away_pp_toi_min=row.league_avg_pp_toi_min,
            league_avg_other_toi_min=row.league_avg_other_toi_min,
        )
        lam_home = lam_home_raw * home_ice_multiplier
        lam_away = lam_away_raw / home_ice_multiplier
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
        r, model_name="situational_ev_pp_pk_poisson",
        config_flags={"prior_minutes_ev": 487, "prior_minutes_pp": 49, "prior_minutes_pk": 49,
                       "prior_minutes_other": 23, "home_ice": "fit_empirically_on_goals", "min_season": 20102011,
                       "toi_weighting": "league_average_shared"},
        notes="Cycle 3: splits team strength into even-strength + PP-for/PK-against + other (4v4/3v3-OT/"
              "empty-net) components using MoneyPuck's situational rows, vs. xg_based_poisson's whole-game xG. "
              "TOI weighting is a shared league-average (not team-specific) by design -- see "
              "team_strength_situational.py docstring. See MODEL_DOCUMENTATION.md Sec4.4.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
