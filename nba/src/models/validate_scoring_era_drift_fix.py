"""Candidate fix for Sec9.5's still-open scoring-era-drift finding: does a
recency-weighted (EWMA) league average for PACE/OFF_RTG/DEF_RTG's
shrinkage (`shrinkage.py`'s new `league_avg_halflife_games` option) beat
the CURRENT flat-cumulative Phase 1 baseline on margin_mae -- the exact
metric with the confirmed real dev/holdout gap?

Two-stage dev-only discipline (never touches holdout for the decision, per
the confirmatory-veto protocol): both stages FIT on the FULL dev range
(2015-2023) -- fitting on a shorter recent-only slice would never let the
flat-cumulative baseline's staleness actually accumulate, defeating the
whole point of the test. (1) EVALUATE each candidate halflife's fit on
just the last 3 dev seasons (2021-2023, where staleness is worst) against
the current baseline, cheap sweep; (2) re-validate the best candidate's
same full-range fit EVALUATED across the entire dev range before it's
eligible for a genuinely new one-time holdout read. Mirrors the same
recent-first/full-range-second/holdout-only-if-both-pass discipline used
for the props subsystem's Sec16 fast-follow attempt.

Run as `python -m src.models.validate_scoring_era_drift_fix`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game

RECENT_SLICE_START_SEASON = DEV_MAX_SEASON - 3  # last 3 dev seasons: 2021-2023
CANDIDATE_HALFLIVES = (100.0, 300.0, 600.0, 1000.0)


def _to_wide_games(log: pd.DataFrame) -> pd.DataFrame:
    """Mirrors `validate_team_strength_baseline._to_wide_games` exactly,
    including its rename step -- `home_court.fit_home_court_walk_forward`
    (via `_baseline_log_ratios`) specifically needs `actual_home_score`/
    `actual_away_score`/`league_avg_pace`/`league_avg_rtg`, not the raw
    `actualScore_home`/`pace_league_avg_home`/`rtg_league_avg_home` names
    the plain wide-merge produces."""
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
    baseline_arm = _arms(baseline_df)
    candidate_arm = _arms(candidate_df)
    print(f"\n=== {label} (n={len(candidate_df)}) ===", flush=True)
    return bootstrap_compare(
        candidate_arm, baseline_arm, game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="ewma_adaptive", label_b="flat_cumulative",
    )


def _full_range_fit(halflife: float | None) -> pd.DataFrame:
    """Always fits on the FULL dev range (2015-2023) regardless of which
    slice will later be used for evaluation -- the flat-cumulative
    baseline's staleness only shows up after years of accumulated history,
    so fitting on a shorter window would never reproduce the actual
    problem being tested."""
    log = build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = add_team_ratings(log, league_avg_halflife_games=halflife)
    games = _to_wide_games(log)
    df = build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


if __name__ == "__main__":
    print("Fitting baseline (flat-cumulative) on the full dev range once...", flush=True)
    baseline_full = _full_range_fit(None)
    baseline_recent = baseline_full[baseline_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)

    print(f"\n=== Stage 1: evaluate on the recent-dev slice only (seasons {RECENT_SLICE_START_SEASON}-{DEV_MAX_SEASON - 1}), "
          f"fit on the FULL dev range ===", flush=True)
    candidate_fits = {}
    best_halflife, best_margin_delta = None, 0.0
    for hl in CANDIDATE_HALFLIVES:
        candidate_full = _full_range_fit(hl)
        candidate_fits[hl] = candidate_full
        candidate_recent = candidate_full[candidate_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)
        result = _compare(baseline_recent, candidate_recent, f"halflife={hl}")
        margin = result["margin_mae"]
        if "REAL IMPROVEMENT" in margin["verdict"] and margin["delta"] < best_margin_delta:
            best_halflife, best_margin_delta = hl, margin["delta"]

    if best_halflife is None:
        print("\nNo candidate halflife showed a real margin_mae improvement on the recent-dev slice "
              "-- stopping here, nothing to re-validate on full dev range.", flush=True)
    else:
        print(f"\nBest candidate: halflife={best_halflife} (margin_mae delta={best_margin_delta:+.4f} on recent slice)", flush=True)
        print(f"\n=== Stage 2: same fit, evaluated across the FULL dev range (seasons {FIRST_DEV_SEASON}-{DEV_MAX_SEASON - 1}) ===", flush=True)
        full_result = _compare(baseline_full, candidate_fits[best_halflife], f"FULL DEV RANGE, halflife={best_halflife}")

        if "REAL IMPROVEMENT" in full_result["margin_mae"]["verdict"]:
            print(f"\nPASSES full-dev-range re-validation -- halflife={best_halflife} is eligible for a "
                  f"genuinely new one-time confirmatory holdout read.", flush=True)
        else:
            print(f"\nDOES NOT hold at full-dev-range scale ({full_result['margin_mae']['verdict']}) -- "
                  f"reverting, not spending a holdout read on this.", flush=True)
