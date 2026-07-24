"""Walk-forward validation of the starting-goalie overlay (Cycle 5) on top
of the current best team model (situational_team_specific_pp_toi_poisson,
Cycle 4). Applies the goalie adjustment as the LAST step, on top of the
already home-ice-adjusted lambda -- an independent additive correction, not
re-derived through the situational combine machinery.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.metrics_ledger import append_run
from src.models.team_strength_goalie import (
    GOALIE_ADJUSTMENT_FLOOR, add_walk_forward_goalie_strength, build_primary_goalie_per_team_game,
)
from src.models.validate_situational_toi import run_validation as run_situational_toi
from src.utils.paths import DATA_RAW


def run_validation(min_season: int = 20102011, league_avg_halflife_games: float | None = None,
                    cross_season_weight: float | None = None,
                    prior_minutes_multiplier: float = 1.0,
                    use_sh_term: bool = False,
                    goalie_league_avg_halflife_games: float | None = None,
                    toi_halflife_games: float | None = None) -> pd.DataFrame:
    """`league_avg_halflife_games`, `cross_season_weight`, `prior_minutes_multiplier`,
    `use_sh_term`: passed through to the situational layer; defaults
    preserve this cycle's original behavior exactly.
    `goalie_league_avg_halflife_games` (Sec19, Cycle 24): separately decays
    the goalie overlay's OWN `goalie_league_avg` baseline -- None (default)
    preserves the original (buggy, infinite-memory) behavior; see
    team_strength_goalie.add_walk_forward_goalie_strength's docstring.
    `toi_halflife_games` (Sec29's follow-up, Cycle 29): passed through to
    validate_situational_toi -- decays league_avg_ev_toi_min/
    league_avg_other_toi_min instead of an infinite-memory expanding mean.
    None (default) preserves the original behavior exactly."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")

    base = run_situational_toi(
        min_season=min_season, league_avg_halflife_games=league_avg_halflife_games,
        cross_season_weight=cross_season_weight,
        prior_minutes_multiplier=prior_minutes_multiplier,
        use_sh_term=use_sh_term, toi_halflife_games=toi_halflife_games)  # gameId, season, actual_home/away, lambda_home/away

    goalie_log = build_primary_goalie_per_team_game(schedule)
    goalie_log = add_walk_forward_goalie_strength(
        goalie_log, league_avg_halflife_games=goalie_league_avg_halflife_games)
    # BUG FOUND AND FIXED (2026-07-23): using goalie_shrunk_mean directly (an
    # ABSOLUTE GSAx level) as the adjustment introduced a large systematic
    # bias -- confirmed real, not a one-season fluke: averaged -0.51 to -0.56
    # across ALL 46,440 goalie-game rows, because MoneyPuck's xGoal, summed
    # only over the shots-actually-on-goal subset (excluding empty net,
    # which was itself a separate real bug -- see
    # fetch_moneypuck_goalie_games.py), systematically undershoots the real
    # goal rate on THAT subset specifically. Subtracting this systematically
    # negative absolute level from every game's lambda added roughly +1.0
    # goals/game across the board -- exactly matching an observed jump in
    # mean predicted total goals to ~6.8 against a real ~5.86. The fix: use
    # the goalie's rating RELATIVE TO THE TRAILING LEAGUE AVERAGE AT THAT
    # SAME WALK-FORWARD POINT IN TIME (goalie_shrunk_mean - goalie_league_avg)
    # -- any systematic scope-level miscalibration affects the goalie's own
    # level and the league average it's compared to identically, so it
    # cancels in the difference, leaving only the genuine relative signal
    # ("better or worse than an average goalie right now"), which is what an
    # additive overlay on top of an already-calibrated team model should be
    # using in the first place.
    goalie_log["goalie_relative"] = goalie_log["goalie_shrunk_mean"] - goalie_log["goalie_league_avg"]
    goalie_log["goalie_relative"] = goalie_log["goalie_relative"].fillna(0.0)  # bfill would leak; a goalie with
    # zero prior appearances anywhere in the dataset has no informative rating yet -- 0 (== exactly
    # league-average) is the correct walk-forward-safe fallback.
    goalie_ratings = goalie_log[["gameId", "team", "goalie_relative"]]

    sched_teams = schedule[["gameId", "homeTeamAbbrev", "awayTeamAbbrev"]].drop_duplicates()
    merged = base.merge(sched_teams, on="gameId", how="left")
    merged = merged.merge(goalie_ratings.rename(columns={"team": "homeTeamAbbrev", "goalie_relative": "home_goalie"}),
                           on=["gameId", "homeTeamAbbrev"], how="left")
    merged = merged.merge(goalie_ratings.rename(columns={"team": "awayTeamAbbrev", "goalie_relative": "away_goalie"}),
                           on=["gameId", "awayTeamAbbrev"], how="left")
    n_missing = merged["home_goalie"].isna().sum() + merged["away_goalie"].isna().sum()
    if n_missing:
        print(f"validate_goalie: {n_missing} team-game sides had no matched goalie rating, treated as "
              f"league-average (0.0) rather than dropped")
    merged["home_goalie"] = merged["home_goalie"].fillna(0.0)
    merged["away_goalie"] = merged["away_goalie"].fillna(0.0)

    results = []
    for row in merged.itertuples():
        adj_lambda_home = max(row.lambda_home - row.away_goalie, GOALIE_ADJUSTMENT_FLOOR)
        adj_lambda_away = max(row.lambda_away - row.home_goalie, GOALIE_ADJUSTMENT_FLOOR)
        joint = score_distribution(adj_lambda_home, adj_lambda_away)
        home_win_p, away_win_p, tie_p = home_win_prob_regulation(joint)
        results.append({
            "gameId": row.gameId, "gameDate": row.gameDate, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": adj_lambda_home, "lambda_away": adj_lambda_away,
            "home_win_prob": home_win_p, "away_win_prob": away_win_p, "tie_prob": tie_p,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.models.team_strength_goalie import PRIOR_GAMES_GOALIE

    r = run_validation()
    print(f"validated {len(r)} real games, seasons {sorted(r['season'].unique())}")
    row = append_run(
        r, model_name="goalie_overlay_poisson",
        config_flags={"prior_games_goalie": PRIOR_GAMES_GOALIE, "goalie_adjustment": "additive_post_home_ice",
                       "min_season": 20102011},
        notes="Cycle 5: starting-goalie GSAx overlay (own primary goalie per team-game, from "
              "moneypuck_goalie_games.parquet) subtracted from the OPPONENT's lambda, applied on top of "
              "situational_team_specific_pp_toi_poisson's already home-ice-adjusted output. "
              "See MODEL_DOCUMENTATION.md Sec4.6 (initial run) / Sec4.7 (calibrated prior).",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
