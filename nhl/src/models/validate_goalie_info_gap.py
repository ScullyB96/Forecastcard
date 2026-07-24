"""Measures the live-vs-backtest information gap created by the starting-
goalie overlay (Sec4.6/Sec4.7), for task #43's degradation budget.

Backtests score every historical game knowing the REAL starting goalie
(already public in the box-score data); a live prediction made before
lineups are confirmed does not have this yet. Two comparisons, both
against the same real-starter baseline (`validate_goalie.run_validation`,
the actual committed production goalie overlay -- HALFLIFE_GAMES=600,
CROSS_SEASON_WEIGHT=0.75, PRIOR_MINUTES_MULTIPLIER=2.0, use_sh_term=True,
matching `_build_dev_base`'s own call exactly):

1. **Worst case -- real starter vs. ZERO information**
   (`validate_situational_toi.run_validation`, identical everything else,
   simply no goalie term at all). An honest upper bound, since a live
   system would never actually have zero information about who is likely
   to start.

2. **Realistic case -- real starter vs. a RECENT-WORKHORSE HEURISTIC**
   (predict each team's starter as whichever goalie had the most starts
   in that team's trailing `HEURISTIC_WINDOW_GAMES` games, strictly prior
   -- a simple proxy for what a depth-chart/recent-usage-informed live
   guess would produce without needing real scraped lineup data). The
   predicted goalie's OWN trailing `goalie_relative` (his own walk-
   forward-shrunk GSAx entering that date, via `pd.merge_asof`,
   `allow_exact_matches=False` so a goalie can never see his own outcome
   from the very game being predicted) stands in for the real starter's
   rating whenever the heuristic's guess differs from who actually played.

Both comparisons run all three metrics this project tracks (Brier,
margin-MAE, total-MAE) -- Sec4.6's own original adoption evidence for this
overlay was a real MARGIN-MAE improvement specifically (CI [0.00691,
0.01233]), not Brier or total-MAE, so a degradation budget that omits
margin-MAE is checking the two metrics goalie information matters least
for and could under-trigger exactly where live degradation would actually
show up.
"""

import numpy as np
import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.team_strength_goalie import (
    GOALIE_ADJUSTMENT_FLOOR, add_walk_forward_goalie_strength, build_primary_goalie_per_team_game,
)
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_goalie import run_validation as run_real_goalie
from src.models.validate_situational_toi import run_validation as run_zero_info
from src.models.validate_tie_mass_ratio import _bootstrap
from src.utils.paths import DATA_RAW

HEURISTIC_WINDOW_GAMES = 10  # trailing team-games used to predict the likely starter


def _predict_starter_per_team_game(goalie_log: pd.DataFrame, window_games: int = HEURISTIC_WINDOW_GAMES
                                    ) -> pd.DataFrame:
    """For each real team-game row, predict the starter as the goalie with
    the most starts among that TEAM's own trailing `window_games` games
    (strictly prior games only -- the current game's own actual starter is
    never in its own window)."""
    goalie_log = goalie_log.sort_values(["team", "gameDate", "gameId"]).reset_index(drop=True)
    predicted = np.empty(len(goalie_log), dtype=object)
    for team, idx in goalie_log.groupby("team").groups.items():
        idx = list(idx)
        goalies = goalie_log.loc[idx, "goalieIdForShot"].values
        for pos, row_idx in enumerate(idx):
            window = goalies[max(0, pos - window_games):pos]
            if len(window) == 0:
                predicted[row_idx] = None
            else:
                vals, counts = np.unique(window, return_counts=True)
                predicted[row_idx] = vals[np.argmax(counts)]
    goalie_log = goalie_log.copy()
    goalie_log["predicted_goalie"] = predicted
    return goalie_log[["gameId", "team", "gameDate", "predicted_goalie"]]


def _heuristic_goalie_ratings(schedule: pd.DataFrame) -> pd.DataFrame:
    """Returns gameId/team/goalie_relative using the recent-workhorse
    heuristic's predicted starter's OWN trailing rating, instead of the
    real actual starter's."""
    goalie_log = build_primary_goalie_per_team_game(schedule)
    goalie_log_wf = add_walk_forward_goalie_strength(goalie_log)  # league_avg_halflife_games=None, matching
    # _build_dev_base's own (un-passed, default) call -- the current committed behavior.
    goalie_log_wf["goalie_relative"] = (
        goalie_log_wf["goalie_shrunk_mean"] - goalie_log_wf["goalie_league_avg"]).fillna(0.0)

    predicted = _predict_starter_per_team_game(goalie_log)

    # Per-goalie own trailing-rating time series (one row per real appearance,
    # sorted by date) -- the as-of lookup source for a PREDICTED goalie who may
    # not be the one who actually played this specific game.
    own_series = goalie_log_wf[["goalieIdForShot", "gameDate", "goalie_relative"]].copy()
    own_series["gameDate"] = pd.to_datetime(own_series["gameDate"])
    own_series = own_series.sort_values("gameDate").reset_index(drop=True)  # merge_asof requires
    # both sides globally sorted by the "on" key in this pandas version, even with "by" grouping

    predicted = predicted.copy()
    predicted["gameDate"] = pd.to_datetime(predicted["gameDate"])
    has_pred = predicted[predicted["predicted_goalie"].notna()].copy()
    has_pred["predicted_goalie"] = has_pred["predicted_goalie"].astype(own_series["goalieIdForShot"].dtype)
    has_pred = has_pred.sort_values("gameDate").reset_index(drop=True)  # merge_asof requires the LEFT
    # frame globally sorted by the "on" key (grouping is handled entirely by "by")
    no_pred = predicted[predicted["predicted_goalie"].isna()].copy()
    no_pred["goalie_relative"] = 0.0  # no trailing team history yet -- matches the walk-forward-safe
    # fallback used everywhere else in this project.

    merged = pd.merge_asof(
        has_pred, own_series.rename(columns={"goalieIdForShot": "predicted_goalie"}),
        on="gameDate", by="predicted_goalie", direction="backward", allow_exact_matches=False)
    merged["goalie_relative"] = merged["goalie_relative"].fillna(0.0)  # predicted goalie has no prior
    # appearance of his own anywhere yet (e.g. a rookie) -- league-average fallback.

    out = pd.concat([merged[["gameId", "team", "goalie_relative"]],
                      no_pred[["gameId", "team", "goalie_relative"]]], ignore_index=True)
    return out


def run_heuristic_validation(min_season: int = 20102011) -> pd.DataFrame:
    """Same construction as `validate_goalie.run_validation` (situational
    base + additive goalie overlay), but using the recent-workhorse
    heuristic's predicted starter's rating instead of the real starter's."""
    from src.models.validate_situational_toi import run_validation as run_situational_toi
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]

    base = run_situational_toi(min_season=min_season, league_avg_halflife_games=HALFLIFE_GAMES,
                                cross_season_weight=CROSS_SEASON_WEIGHT,
                                prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    ratings = _heuristic_goalie_ratings(schedule)

    sched_teams = schedule[["gameId", "homeTeamAbbrev", "awayTeamAbbrev"]].drop_duplicates()
    merged = base.merge(sched_teams, on="gameId", how="left")
    merged = merged.merge(ratings.rename(columns={"team": "homeTeamAbbrev", "goalie_relative": "home_goalie"}),
                           on=["gameId", "homeTeamAbbrev"], how="left")
    merged = merged.merge(ratings.rename(columns={"team": "awayTeamAbbrev", "goalie_relative": "away_goalie"}),
                           on=["gameId", "awayTeamAbbrev"], how="left")
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


def _metrics(df: pd.DataFrame) -> dict:
    y = (df["actual_home"] > df["actual_away"]).astype(int).values
    p = df["home_win_prob"].clip(1e-6, 1 - 1e-6).values
    brier = (p - y) ** 2
    margin_mae = np.abs((df["lambda_home"] - df["lambda_away"]) - (df["actual_home"] - df["actual_away"])).values
    total_mae = np.abs((df["lambda_home"] + df["lambda_away"]) - (df["actual_home"] + df["actual_away"])).values
    return {"brier": brier, "margin_mae": margin_mae, "total_mae": total_mae}


def run_validation(min_season: int = 20102011) -> dict:
    real = run_real_goalie(min_season=min_season, league_avg_halflife_games=HALFLIFE_GAMES,
                            cross_season_weight=CROSS_SEASON_WEIGHT,
                            prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    zero = run_zero_info(min_season=min_season, league_avg_halflife_games=HALFLIFE_GAMES,
                          cross_season_weight=CROSS_SEASON_WEIGHT,
                          prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER, use_sh_term=True)
    heuristic = run_heuristic_validation(min_season=min_season)

    real = real.sort_values("gameId").reset_index(drop=True)
    zero = zero.sort_values("gameId").reset_index(drop=True)
    heuristic = heuristic.sort_values("gameId").reset_index(drop=True)
    assert (real["gameId"].values == zero["gameId"].values).all()
    assert (real["gameId"].values == heuristic["gameId"].values).all()

    m_real, m_zero, m_heur = _metrics(real), _metrics(zero), _metrics(heuristic)

    out = {}
    for metric in ("brier", "margin_mae", "total_mae"):
        d_zero, lo_zero, hi_zero = _bootstrap(m_zero[metric], m_real[metric])
        d_heur, lo_heur, hi_heur = _bootstrap(m_heur[metric], m_real[metric])
        out[metric] = {
            "worst_case_vs_zero": {"diff": d_zero, "ci": (lo_zero, hi_zero)},
            "realistic_vs_heuristic": {"diff": d_heur, "ci": (lo_heur, hi_heur)},
        }
    return out


if __name__ == "__main__":
    results = run_validation()
    for metric, arms in results.items():
        print(f"\n=== {metric} ===")
        for arm, r in arms.items():
            lo, hi = r["ci"]
            real_flag = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
            print(f"  {arm}: diff(real-comparator)={r['diff']:+.5f} CI[{lo:+.5f},{hi:+.5f}] ({real_flag})")
