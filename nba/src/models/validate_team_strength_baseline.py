"""Phase 1's acceptance gate: does the pace x rating closed-form combine
(`team_strength.project_game`, walk-forward shrunk PACE/OFF_RATING/
DEF_RATING, walk-forward-fit home-court multiplier) beat the literal naive
floor the user specified -- a team's own trailing scoring average, with NO
opponent adjustment at all?

Dev-only (season < final_holdout_check.DEV_MAX_SEASON) -- holdout is never
read here, per the confirmatory-veto protocol. Every game's prediction uses
only that team's own STRICTLY PRIOR games (the shrinkage/home-court fits are
walk-forward by construction; see shrinkage.py and home_court.py), so this
is a genuine backtest, not an in-sample fit check.

Run as `python -m src.models.validate_team_strength_baseline`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_by_season, fit_home_court_walk_forward
from src.models.rest_schedule import add_rest_days, correlate_residual_with_rest
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game

NAIVE_PRIOR_GAMES = 5.0  # small prior -- deliberately a weak, simple floor, not itself tuned


def _rated_dev_log() -> pd.DataFrame:
    log = build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    if log.empty:
        raise RuntimeError("no dev-range data cached yet -- run the Phase 0 ingest backfill first")
    log = add_team_ratings(log)
    log = add_walk_forward_mean(log, "actualScore", NAIVE_PRIOR_GAMES, prefix="naive")
    log = add_rest_days(log)
    return log


def _to_wide_games(log: pd.DataFrame) -> pd.DataFrame:
    """One row per real game, home_/away_ prefixed columns -- the shape both
    `home_court.fit_home_court_walk_forward` and this script's per-game
    projection loop need."""
    home = log[log["is_home"]].copy()
    away = log[~log["is_home"]].copy()
    games = home.merge(away, on=["gameId", "gameDate", "season"], suffixes=("_home", "_away"))
    games = games.rename(columns={
        "actualScore_home": "actual_home_score", "actualScore_away": "actual_away_score",
        "pace_home": "home_pace", "pace_away": "away_pace",
        "offRtg_home": "home_oRtg", "defRtg_home": "home_dRtg",
        "offRtg_away": "away_oRtg", "defRtg_away": "away_dRtg",
        "pace_league_avg_home": "league_avg_pace", "rtg_league_avg_home": "league_avg_rtg",
    })
    return games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def build_dev_predictions() -> pd.DataFrame:
    log = _rated_dev_log()
    games = _to_wide_games(log)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)

    preds = []
    for row in games.itertuples(index=False):
        proj_pace, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg,
            home_court_mult=row.home_court_mult,
        )
        preds.append({
            "gameId": row.gameId, "season": row.season, "gameDate": row.gameDate, "projected_pace": proj_pace,
            "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
            "pred_home": pred_home, "pred_away": pred_away,
            "naive_home": row.naive_shrunk_mean_home, "naive_away": row.naive_shrunk_mean_away,
        })
    return pd.DataFrame(preds)


def run_rest_diagnostic(log: pd.DataFrame, df: pd.DataFrame) -> None:
    """Informational only -- NOT folded into the Phase 1 predictions above.
    Correlates each team-side's residual (actual - model prediction) against
    that side's own rest_days/is_b2b, per rest_schedule.py's discipline."""
    pred_long = pd.concat([
        df[["gameId"]].assign(is_home=True, pred=df["pred_home"], actual=df["actual_home"]),
        df[["gameId"]].assign(is_home=False, pred=df["pred_away"], actual=df["actual_away"]),
    ], ignore_index=True)
    merged = log.merge(pred_long, on=["gameId", "is_home"], how="inner")
    merged["residual"] = merged["actual"] - merged["pred"]
    result = correlate_residual_with_rest(merged, residual_col="residual", rest_col="rest_days")
    print(f"rest_days vs residual: r={result['r']:+.4f} (p={result['p']:.4g}, n={result['n']})", flush=True)
    print(f"B2B effect: mean_residual_b2b={result['mean_residual_b2b']:+.3f} vs "
          f"mean_residual_rested={result['mean_residual_rested']:+.3f} "
          f"(delta={result['b2b_effect']:+.3f}, t={result['b2b_ttest_t']:+.3f}, "
          f"p={result['b2b_ttest_p']:.4g}, n_b2b={result['n_b2b']})", flush=True)
    if result["b2b_ttest_p"] < 0.01:
        print("  -> looks real: candidate for a tested Phase 1b increment (not yet adopted)", flush=True)
    else:
        print("  -> not distinguishable from noise at this sample size; leave unadopted", flush=True)


if __name__ == "__main__":
    print("=== Phase 1 diagnostic: home-court multiplier by season (drift check) ===", flush=True)
    drift_log = add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1))
    print(fit_home_court_by_season(_to_wide_games(drift_log)), flush=True)

    dev_log = _rated_dev_log()
    df = build_dev_predictions()
    n_before = len(df)
    df = df.dropna(subset=["pred_home", "pred_away", "naive_home", "naive_away"]).reset_index(drop=True)
    if len(df) != n_before:
        # Expected: the very first game(s) of the whole dev range have no trailing history yet
        # (no prior games to compute a league average or naive floor from) -- not a data bug.
        print(f"dropped {n_before - len(df)} game(s) with no prior history yet (start of the dev range)", flush=True)
    print(f"\nbuilt {len(df)} dev-set game predictions "
          f"(seasons {sorted(df['season'].unique())})", flush=True)

    df["abs_total_err_model"] = ((df["pred_home"] + df["pred_away"]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err_model"] = ((df["pred_home"] - df["pred_away"]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su_model"] = ((df["pred_home"] > df["pred_away"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)

    df["abs_total_err_naive"] = ((df["naive_home"] + df["naive_away"]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err_naive"] = ((df["naive_home"] - df["naive_away"]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su_naive"] = ((df["naive_home"] > df["naive_away"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)

    arm_model = df[["gameId", "abs_total_err_model", "abs_margin_err_model", "su_model"]].rename(
        columns={"abs_total_err_model": "abs_total_err", "abs_margin_err_model": "abs_margin_err", "su_model": "su"})
    arm_naive = df[["gameId", "abs_total_err_naive", "abs_margin_err_naive", "su_naive"]].rename(
        columns={"abs_total_err_naive": "abs_total_err", "abs_margin_err_naive": "abs_margin_err", "su_naive": "su"})

    print("\n=== Phase 1 vs naive floor: paired bootstrap ===", flush=True)
    bootstrap_compare(
        arm_model, arm_naive, game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a="pace_x_rating", label_b="naive_own_avg",
    )

    print("\n=== rest/back-to-back diagnostic (NOT yet adopted, informational only) ===", flush=True)
    run_rest_diagnostic(dev_log, df)

    metrics_ledger.append_run(
        df[["gameId", "season", "actual_home", "actual_away", "pred_home", "pred_away"]],
        model_name="phase1_pace_x_rating",
        config_flags={"prior_games_rating": 15.0, "prior_games_pace": 15.0, "dev_max_season": DEV_MAX_SEASON},
        notes="Phase 1 team-strength baseline vs naive own-average floor",
    )
    naive_arm = df[["gameId", "season", "actual_home", "actual_away", "naive_home", "naive_away"]].rename(
        columns={"naive_home": "pred_home", "naive_away": "pred_away"})
    # NOTE: renaming naive_home/naive_away to pred_home/pred_away on a SUBSET selected first (not
    # on `df` directly) -- df itself already has its own pred_home/pred_away columns from the
    # model arm, so renaming in place on the full frame creates two columns sharing the same name
    # (confirmed real: this crashed metrics_ledger.compute_run_metrics with "Cannot set a
    # DataFrame with multiple columns to the single column pred_total" the first time this ran on
    # the real, full dev range).
    metrics_ledger.append_run(
        naive_arm,
        model_name="naive_own_avg",
        config_flags={"prior_games_naive": NAIVE_PRIOR_GAMES, "dev_max_season": DEV_MAX_SEASON},
        notes="Phase 1 naive floor (no opponent adjustment)",
    )
    print("\nboth arms appended to metrics_ledger.py", flush=True)
