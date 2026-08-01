"""Domain-signal lever (full-model-audit research pass, 2026-08-01): the
back-to-back residual correlation was found REAL early in this project
(`rest_schedule.py`'s own diagnostic, run inside
`validate_team_strength_baseline.run_rest_diagnostic` -- p=8.0e-7 on a
real dev-range check) but was NEVER actually adopted as a live feature:
`fit_b2b_adjustment`/`add_rest_days` are confirmed (via a repo-wide grep)
to be used nowhere in `team_strength.py` or either live pipeline -- a
confirmed-real signal that's sat completely unused. This script properly
validates ADOPTING it (a residual correlating with something is not the
same claim as "adding a correction for it actually improves accuracy
out-of-sample" -- this project's standing discipline is to test the
adopted-feature claim directly, not stop at the diagnostic).

Walk-forward-safe by construction: the B2B adjustment magnitude for any
game is the expanding mean of B2B-game residuals from STRICTLY PRIOR
games only (same "collapse to one row per gameId, shift(1), expanding"
leak guard `shrinkage._trailing_league_stat` uses elsewhere), never the
same magnitude used to fit and then evaluate on the same games.

Run as `python -m src.models.validate_b2b_adjustment`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.rest_schedule import add_rest_days
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game

RECENT_SLICE_START_SEASON = DEV_MAX_SEASON - 3  # last 3 dev seasons: 2021-2023


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
        preds.append({"gameId": row.gameId, "gameDate": row.gameDate, "season": row.season,
                       "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
                       "pred_home": pred_home, "pred_away": pred_away,
                       "home_is_b2b": row.is_b2b_home, "away_is_b2b": row.is_b2b_away})
    return pd.DataFrame(preds)


def _add_walk_forward_b2b_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `home_pred_adj`/`away_pred_adj` -- the baseline prediction plus
    a walk-forward, strictly-prior-games-only B2B correction: the
    expanding mean of (actual - pred) on every B2B team-side-game seen so
    far (home and away pooled, since a rest-fatigue effect isn't
    plausibly home/away-specific), using ONLY residuals from games
    strictly before the CURRENT game (by gameDate).

    Processes games in one explicit chronological pass (not a vectorized
    shift/expanding on a pre-stacked home+away frame) specifically to
    avoid the same-game leak trap this project has hit before
    (`shrinkage.py`'s own docstring): if a game's home and away rows sit
    adjacent in a stacked/sorted frame, a naive `shift(1)` can leak one
    side's own same-game residual into the other side's "trailing"
    value. An explicit loop that reads the running sum/count BEFORE
    appending this game's own two residuals makes that leak structurally
    impossible, at the cost of being O(n) Python rather than vectorized --
    fine at dev-range scale (~10K games)."""
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    home_resid = (df["actual_home"] - df["pred_home"]).to_numpy()
    away_resid = (df["actual_away"] - df["pred_away"]).to_numpy()
    home_is_b2b = df["home_is_b2b"].fillna(False).to_numpy()
    away_is_b2b = df["away_is_b2b"].fillna(False).to_numpy()

    adjustment = [0.0] * len(df)
    running_sum, running_count = 0.0, 0
    for i in range(len(df)):
        adjustment[i] = (running_sum / running_count) if running_count > 0 else 0.0
        if home_is_b2b[i]:
            running_sum += home_resid[i]
            running_count += 1
        if away_is_b2b[i]:
            running_sum += away_resid[i]
            running_count += 1

    df = df.copy()
    adjustment = pd.Series(adjustment, index=df.index)
    df["home_pred_adj"] = df["pred_home"] + df["home_is_b2b"].fillna(False).astype(float) * adjustment
    df["away_pred_adj"] = df["pred_away"] + df["away_is_b2b"].fillna(False).astype(float) * adjustment
    return df


def _arms(df: pd.DataFrame, pred_home_col: str, pred_away_col: str) -> pd.DataFrame:
    df = df.copy()
    df["abs_total_err"] = ((df[pred_home_col] + df[pred_away_col]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err"] = ((df[pred_home_col] - df[pred_away_col]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su"] = ((df[pred_home_col] > df[pred_away_col]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return df[["gameId", "abs_total_err", "abs_margin_err", "su"]]


def _compare(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} (n={len(candidate_df)}) ===", flush=True)
    return bootstrap_compare(
        _arms(candidate_df, "home_pred_adj", "away_pred_adj"), _arms(baseline_df, "pred_home", "pred_away"),
        game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="b2b_adjusted", label_b="current_baseline",
    )


if __name__ == "__main__":
    print("Building dev-range predictions (current, already-adopted Phase 1 config) once...", flush=True)
    log = build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = add_team_ratings(log)
    log = add_rest_days(log)
    games = _to_wide_games(log)
    full_preds = build_predictions(games)
    full_preds = full_preds.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)
    adjusted = _add_walk_forward_b2b_adjustment(full_preds)

    recent = adjusted[adjusted["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)
    print(f"\n=== Stage 1: recent-dev slice (seasons {RECENT_SLICE_START_SEASON}-{DEV_MAX_SEASON - 1}) ===", flush=True)
    stage1 = _compare(recent, recent, "B2B walk-forward adjustment vs no adjustment")

    if "REAL IMPROVEMENT" in stage1["total_mae"]["verdict"] or "REAL IMPROVEMENT" in stage1["margin_mae"]["verdict"]:
        if "REAL REGRESSION" not in stage1["total_mae"]["verdict"] and "REAL REGRESSION" not in stage1["margin_mae"]["verdict"]:
            print("\n=== Stage 2: full dev range ===", flush=True)
            stage2 = _compare(adjusted, adjusted, "B2B walk-forward adjustment vs no adjustment, FULL DEV RANGE")
            if ("REAL IMPROVEMENT" in stage2["total_mae"]["verdict"] or "REAL IMPROVEMENT" in stage2["margin_mae"]["verdict"]) and \
               "REAL REGRESSION" not in stage2["total_mae"]["verdict"] and "REAL REGRESSION" not in stage2["margin_mae"]["verdict"]:
                print("\nPASSES full-dev-range re-validation -- eligible for a genuinely new one-time "
                      "confirmatory holdout read.", flush=True)
            else:
                print("\nDOES NOT hold at full-dev-range scale -- reverting, not spending a holdout read.", flush=True)
        else:
            print("\nReal regression on at least one metric at Stage 1 -- stopping here.", flush=True)
    else:
        print("\nNo real improvement on the recent-dev slice -- stopping here.", flush=True)
