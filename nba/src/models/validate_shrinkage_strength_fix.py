"""Second candidate fix for Sec9.5's still-open scoring-era-drift margin
regression -- a genuinely different mechanism from Sec24's (already
decisively ruled out) recency-weighted league average.

MOTIVATION: confirmed directly (not assumed) that cross-team quality
SPREAD -- std dev across teams of each team's own average point
differential -- rose materially in the holdout era: mostly 4.0-5.15
across 2015-2022 (one exception: 2018-2021 sits at 4.85-4.94), then
5.56/6.00/6.24 in 2023/2024/2025. A FIXED `prior_games_rating=15.0`
shrinkage weight, implicitly calibrated against the dev era's typical
spread, would OVER-shrink relative to a genuinely wider TRUE spread --
pulling teams' ratings too close to average precisely when real
differentiation has grown, understating true margins. This is a
different lever from Sec24: that test changed what the league-average
BLENDING TARGET tracks (ruled out, made margin worse at every halflife
tried); this one changes how STRONGLY a team's own rating is pulled
toward that target in the first place.

Same two-stage dev-only discipline as Sec24's script: fit on the FULL dev
range (so a stronger shrinkage's effect actually manifests, not just
noise from a short window), evaluate the recent-dev slice first (cheap
screen), then the full dev range if that passes, before ever considering
a holdout read.

Run as `python -m src.models.validate_shrinkage_strength_fix`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.team_strength import PRIOR_GAMES_RATING, add_team_ratings, build_team_game_log, project_game

RECENT_SLICE_START_SEASON = DEV_MAX_SEASON - 3  # last 3 dev seasons: 2021-2023
CANDIDATE_PRIOR_GAMES = (3.0, 5.0, 8.0, 10.0, 12.0)  # all LESS shrinkage than the current 15.0


def _to_wide_games(log: pd.DataFrame) -> pd.DataFrame:
    home = log[log["is_home"]].copy()
    away = log[~log["is_home"]].copy()
    games = home.merge(away, on=["gameId", "gameDate", "season"], suffixes=("_home", "_away"))
    games = games.rename(columns={
        "actualScore_home": "actual_home_score", "actualScore_away": "actual_away_score",
        "pace_league_avg_home": "league_avg_pace", "rtg_league_avg_home": "league_avg_rtg",
    })
    return games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def build_predictions(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)
    preds = []
    for row in games.itertuples(index=False):
        _, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        preds.append({"gameId": row.gameId, "season": row.season,
                       "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
                       "pred_home": pred_home, "pred_away": pred_away})
    return pd.DataFrame(preds)


def _arms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_total_err"] = ((df["pred_home"] + df["pred_away"]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err"] = ((df["pred_home"] - df["pred_away"]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su"] = ((df["pred_home"] > df["pred_away"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return df[["gameId", "abs_total_err", "abs_margin_err", "su"]]


def _compare(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} (n={len(candidate_df)}) ===", flush=True)
    return bootstrap_compare(
        _arms(candidate_df), _arms(baseline_df), game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="less_shrinkage", label_b="current_prior15",
    )


def _full_range_fit(prior_games_rating: float) -> pd.DataFrame:
    log = build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = add_team_ratings(log, prior_games_rating=prior_games_rating)
    games = _to_wide_games(log)
    df = build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


if __name__ == "__main__":
    print(f"Fitting baseline (prior_games_rating={PRIOR_GAMES_RATING}) on the full dev range once...", flush=True)
    baseline_full = _full_range_fit(PRIOR_GAMES_RATING)
    baseline_recent = baseline_full[baseline_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)

    print(f"\n=== Stage 1: evaluate on the recent-dev slice only (seasons {RECENT_SLICE_START_SEASON}-{DEV_MAX_SEASON - 1}), "
          f"fit on the FULL dev range ===", flush=True)
    candidate_fits = {}
    best_prior, best_margin_delta = None, 0.0
    for pg in CANDIDATE_PRIOR_GAMES:
        candidate_full = _full_range_fit(pg)
        candidate_fits[pg] = candidate_full
        candidate_recent = candidate_full[candidate_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)
        result = _compare(baseline_recent, candidate_recent, f"prior_games_rating={pg}")
        margin = result["margin_mae"]
        if "REAL IMPROVEMENT" in margin["verdict"] and margin["delta"] < best_margin_delta:
            best_prior, best_margin_delta = pg, margin["delta"]

    if best_prior is None:
        print("\nNo candidate prior_games_rating showed a real margin_mae improvement on the recent-dev "
              "slice -- stopping here, nothing to re-validate on full dev range.", flush=True)
    else:
        print(f"\nBest candidate: prior_games_rating={best_prior} (margin_mae delta={best_margin_delta:+.4f} on recent slice)", flush=True)
        print(f"\n=== Stage 2: same fit, evaluated across the FULL dev range (seasons {FIRST_DEV_SEASON}-{DEV_MAX_SEASON - 1}) ===", flush=True)
        full_result = _compare(baseline_full, candidate_fits[best_prior], f"FULL DEV RANGE, prior_games_rating={best_prior}")

        if "REAL IMPROVEMENT" in full_result["margin_mae"]["verdict"]:
            print(f"\nPASSES full-dev-range re-validation -- prior_games_rating={best_prior} is eligible for a "
                  f"genuinely new one-time confirmatory holdout read.", flush=True)
        else:
            print(f"\nDOES NOT hold at full-dev-range scale ({full_result['margin_mae']['verdict']}) -- "
                  f"reverting, not spending a holdout read on this.", flush=True)
