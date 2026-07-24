"""Cycle 4: replace Cycle 3's SHARED league-average PP time-on-ice weight
with each team's OWN walk-forward-shrunk expected PP/PK minutes-per-game --
testing whether a team's own penalty-drawing/taking history (not just its
per-60 scoring/suppression rate) carries real signal. Everything else
(situational rate split, EV/other TOI weighting, home-ice methodology) is
identical to Cycle 3, isolating this one variable per the project's
incremental-validation discipline.

A real game's PP minutes for the home team automatically equal the away
team's PK minutes (same clock) -- so for a specific matchup there are TWO
independent walk-forward estimates of the same real quantity: the home
team's own history of getting power plays, and the away team's own history
of killing them. Used the average of the two rather than picking one,
since both genuinely inform the same real number.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.metrics_ledger import append_run
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength_situational import (
    add_walk_forward_situational_strength, build_team_game_situational_log, predict_situational_lambda,
)
from src.models.validate_baseline import fit_home_ice_multiplier
from src.utils.paths import DATA_RAW

PRIOR_GAMES_TOI = 10  # same placeholder standard as every other shrinkage constant here


def run_validation(min_season: int = 20102011, league_avg_halflife_games: float | None = None,
                    cross_season_weight: float | None = None,
                    prior_minutes_multiplier: float = 1.0,
                    use_sh_term: bool = False,
                    toi_halflife_games: float | None = None) -> pd.DataFrame:
    """`league_avg_halflife_games`: passed through to
    team_strength_situational.add_walk_forward_situational_strength -- None
    (default) preserves this cycle's original behavior exactly; see
    MODEL_DOCUMENTATION.md Sec4.15 for the scoring-era-drift test that uses
    a real value here. `cross_season_weight`: same passthrough, see Sec12.
    `prior_minutes_multiplier`: same passthrough, see Sec13.
    `use_sh_term`: adds the shorthanded-goals-for term (Sec18, Cycle 23) --
    False (default) preserves this cycle's original behavior exactly
    (league_avg_sh_rate_per60/home_pk_toi_min/away_pk_toi_min all default to
    0.0 in predict_situational_lambda, an exact no-op).
    `toi_halflife_games` (Sec29's follow-up, Cycle 29): decays
    league_avg_ev_toi_min/league_avg_other_toi_min with this halflife instead
    of an infinite-memory expanding mean -- a DELIBERATELY SEPARATE parameter
    from `league_avg_halflife_games` (which production already sets to a
    real value, 600) so this fix stays a genuine opt-in, not a silent
    behavior change riding on an already-live parameter. None (default)
    preserves the original expanding-mean behavior exactly."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")

    log = build_team_game_situational_log(schedule, moneypuck)
    log = add_walk_forward_situational_strength(log, league_avg_halflife_games=league_avg_halflife_games,
                                                 cross_season_weight=cross_season_weight,
                                                 prior_minutes_multiplier=prior_minutes_multiplier)
    for col in ("ev_league_avg_for_per60", "ev_league_avg_against_per60",
                "pp_league_avg_for_per60", "pk_league_avg_against_per60", "pk_league_avg_for_per60",
                "other_league_avg_for_per60", "other_league_avg_against_per60"):
        log[col] = log[col].bfill()

    # Sec29's follow-up (fourth instance of the Sec7.1 two-baselines-different-
    # memory bug class): an infinite-memory expanding mean of EV/other TOI
    # permanently lags a real, slowly-rising EV-TOI trend (penalties have
    # declined ~continuously since 2010), producing a real, stationary
    # EV-bucket under-allocation -- confirmed directly against real data
    # (mean gap -0.99 min/game, implied whole-game goal impact -0.077/game,
    # closely matching the flat-era gap_ev of -0.085/game, Sec29.5.1).
    # `toi_halflife_games=None` (default) preserves every existing caller's
    # exact prior behavior (the original expanding mean).
    if toi_halflife_games is None:
        log["league_avg_ev_toi_min"] = log["ev_toi_min"].shift(1).expanding().mean().bfill()
        log["league_avg_other_toi_min"] = log["other_toi_min"].shift(1).expanding().mean().bfill()
    else:
        log["league_avg_ev_toi_min"] = log["ev_toi_min"].shift(1).ewm(
            halflife=toi_halflife_games, min_periods=1).mean().bfill()
        log["league_avg_other_toi_min"] = log["other_toi_min"].shift(1).ewm(
            halflife=toi_halflife_games, min_periods=1).mean().bfill()

    # NEW this cycle: each team's own shrunk expected PP/PK minutes-per-game,
    # replacing Cycle 3's shared league-average PP TOI constant.
    log = add_walk_forward_mean(log, "pp_toi_min", PRIOR_GAMES_TOI, "pp_toi")
    log = add_walk_forward_mean(log, "pk_toi_min", PRIOR_GAMES_TOI, "pk_toi")
    log["pp_toi_shrunk_mean"] = log["pp_toi_shrunk_mean"].bfill()
    log["pk_toi_shrunk_mean"] = log["pk_toi_shrunk_mean"].bfill()

    home_ice_multiplier = fit_home_ice_multiplier(log)

    games = log[log["is_home"]].copy()
    away_cols = ["gameId", "ev_attack_rate_per60", "ev_defense_rate_per60",
                 "ev_league_avg_for_per60", "ev_league_avg_against_per60",
                 "pp_attack_rate_per60", "pk_defense_rate_per60",
                 "pp_league_avg_for_per60", "pk_league_avg_against_per60",
                 "other_attack_rate_per60", "other_defense_rate_per60",
                 "other_league_avg_for_per60", "other_league_avg_against_per60",
                 "pp_toi_shrunk_mean", "pk_toi_shrunk_mean"]
    away_side = log[~log["is_home"]][away_cols]
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    games = games[games["season"] >= min_season]

    results = []
    for row in games.itertuples():
        # Home's expected PP minutes = average of (home's own PP-drawing
        # history, away's own PK-taking history); symmetric for away's PP.
        home_pp_toi = (row.pp_toi_shrunk_mean_home + row.pk_toi_shrunk_mean_away) / 2
        away_pp_toi = (row.pp_toi_shrunk_mean_away + row.pk_toi_shrunk_mean_home) / 2
        # Home's own PK time is literally the SAME real-world clock quantity as
        # away's own PP time (Sec18) -- reuse rather than recompute.
        sh_kwargs = {}
        if use_sh_term:
            sh_kwargs = {
                "league_avg_sh_rate_per60": row.pk_league_avg_for_per60,
                "home_pk_toi_min": away_pp_toi, "away_pk_toi_min": home_pp_toi,
            }

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
            home_pp_toi_min=home_pp_toi, away_pp_toi_min=away_pp_toi,
            league_avg_other_toi_min=row.league_avg_other_toi_min,
            **sh_kwargs,
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
        r, model_name="situational_team_specific_pp_toi_poisson",
        config_flags={"prior_minutes_ev": 487, "prior_minutes_pp": 49, "prior_minutes_pk": 49,
                       "prior_minutes_other": 23, "prior_games_toi": 10,
                       "home_ice": "fit_empirically_on_goals", "min_season": 20102011,
                       "toi_weighting": "team_specific_pp_pk"},
        notes="Cycle 4: replaces Cycle 3's shared league-average PP TOI weight with each team's own "
              "walk-forward-shrunk expected PP/PK minutes-per-game (averaged across the matchup's two "
              "independent estimates of the same real quantity). EV/other TOI still shared. "
              "See MODEL_DOCUMENTATION.md Sec4.5.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
