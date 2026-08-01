"""Untried-synthesis lever (full-model-audit research pass, 2026-08-01):
tests the recency-weighted league-average target (Sec24, `league_avg_halflife_games`)
and reduced shrinkage strength (Sec26, `prior_games_rating`) TOGETHER, for
the first time -- both were previously swept independently on
`team_strength.add_team_ratings`, holding the other at its stock default,
and neither alone was a clean win for Phase 1's still-open margin/scoring-
era-drift regression (Sec9.5):
- Sec24 (recency-weighted target alone): real REGRESSION on margin_mae at
  every halflife tested (100-10000), monotonically shrinking toward but
  never crossing zero.
- Sec26 (reduced shrinkage alone): real IMPROVEMENT on margin_mae
  (prior_games_rating=10.0), but a REAL REGRESSION on total_mae of LARGER
  absolute magnitude -- a net tradeoff, not adopted.

Since each lever's own cost sits on the metric the OTHER lever is good at
fixing (Sec24 fixes total_mae's scale bias but hurts margin via added
target noise; Sec26 fixes margin but destabilizes total_mae), a joint
configuration has a plausible mechanism to net out both regressions at
once -- untested until now. Same two-stage dev-only discipline as Sec24/26:
fit on the FULL dev range, screen on the recent-dev slice first, full-dev-
range second, before ever considering a holdout read.

Run as `python -m src.models.validate_joint_margin_fix`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.team_strength import PRIOR_GAMES_RATING, add_team_ratings, build_team_game_log, project_game

RECENT_SLICE_START_SEASON = DEV_MAX_SEASON - 3  # last 3 dev seasons: 2021-2023
CANDIDATE_PRIOR_GAMES = (8.0, 10.0, 12.0)
CANDIDATE_HALFLIVES = (1000.0, 2000.0, 5000.0, 10000.0)


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
        label_a="joint_candidate", label_b="current_baseline",
    )


def _full_range_fit(prior_games_rating: float, halflife: float | None) -> pd.DataFrame:
    log = build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = add_team_ratings(log, prior_games_rating=prior_games_rating, league_avg_halflife_games=halflife)
    games = _to_wide_games(log)
    df = build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


if __name__ == "__main__":
    print(f"Fitting baseline (prior_games_rating={PRIOR_GAMES_RATING}, flat league avg) on the full dev range once...", flush=True)
    baseline_full = _full_range_fit(PRIOR_GAMES_RATING, None)
    baseline_recent = baseline_full[baseline_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)

    print(f"\n=== Stage 1: joint grid sweep, evaluated on the recent-dev slice "
          f"(seasons {RECENT_SLICE_START_SEASON}-{DEV_MAX_SEASON - 1}) ===", flush=True)
    candidate_fits = {}
    results_grid = {}
    for pg in CANDIDATE_PRIOR_GAMES:
        for hl in CANDIDATE_HALFLIVES:
            candidate_full = _full_range_fit(pg, hl)
            candidate_fits[(pg, hl)] = candidate_full
            candidate_recent = candidate_full[candidate_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)
            result = _compare(baseline_recent, candidate_recent, f"prior_games_rating={pg}, halflife={hl}")
            results_grid[(pg, hl)] = result

    # Winning criterion: BOTH total_mae and margin_mae must be REAL IMPROVEMENT (or at worst NOISE
    # for one of them) -- a genuine net win, not another one-metric-for-another tradeoff.
    def is_net_win(result):
        total_ok = "REAL IMPROVEMENT" in result["total_mae"]["verdict"] or "NOISE" in result["total_mae"]["verdict"]
        margin_ok = "REAL IMPROVEMENT" in result["margin_mae"]["verdict"] or "NOISE" in result["margin_mae"]["verdict"]
        no_regression = "REAL REGRESSION" not in result["total_mae"]["verdict"] and "REAL REGRESSION" not in result["margin_mae"]["verdict"]
        real_gain = "REAL IMPROVEMENT" in result["total_mae"]["verdict"] or "REAL IMPROVEMENT" in result["margin_mae"]["verdict"]
        return no_regression and real_gain

    winners = {cfg: r for cfg, r in results_grid.items() if is_net_win(r)}
    print("\n=== Stage 1 summary ===", flush=True)
    for cfg, r in results_grid.items():
        tag = "NET WIN CANDIDATE" if cfg in winners else ""
        print(f"  prior_games_rating={cfg[0]}, halflife={cfg[1]}: "
              f"total={r['total_mae']['verdict']}, margin={r['margin_mae']['verdict']} {tag}", flush=True)

    if not winners:
        print("\nNo joint configuration cleared the Stage 1 bar (no regression on either metric, "
              "real improvement on at least one) -- stopping here.", flush=True)
    else:
        # pick the winner with the best combined (total_mae + margin_mae) delta
        best_cfg = min(winners, key=lambda cfg: winners[cfg]["total_mae"]["delta"] + winners[cfg]["margin_mae"]["delta"])
        print(f"\nBest candidate: prior_games_rating={best_cfg[0]}, halflife={best_cfg[1]}", flush=True)
        print(f"\n=== Stage 2: same fit, evaluated across the FULL dev range "
              f"(seasons {FIRST_DEV_SEASON}-{DEV_MAX_SEASON - 1}) ===", flush=True)
        full_result = _compare(baseline_full, candidate_fits[best_cfg], f"FULL DEV RANGE, {best_cfg}")

        if is_net_win(full_result):
            print(f"\nPASSES full-dev-range re-validation -- {best_cfg} is eligible for a genuinely "
                  f"new one-time confirmatory holdout read.", flush=True)
        else:
            print(f"\nDOES NOT hold at full-dev-range scale -- reverting, not spending a holdout read.", flush=True)
