"""Formalizes MODEL_DOCUMENTATION.md Sec44.2's ad hoc margin-MAE
normalization diagnostic into a real, reproducible script (item 5a of
TECHNICAL_REFERENCE.md's P1 checklist -- "worth doing if this line of
investigation continues").

NOT an adoption gate -- there is no lever to adopt here. This is a pure
DATA-DESCRIPTION diagnostic characterizing whether Phase 1's rising raw
margin_mae over time reflects genuine model decay or rising game-to-game
unpredictability (the games themselves getting harder to call), same as
Sec9.5's own "pure data description, not a repeated holdout performance
read -- doesn't touch the one-time-read budget" precedent. It does not
compare candidate configurations against each other or against holdout to
make an adoption decision, so it does not spend or re-spend any
confirmatory-veto read.

Uses the CURRENT fully-adopted live configuration (mirrors
`run_final_holdout_check.py`'s own `INCLUDE_LINEUP_ADJUSTMENT` flag, so
this always reflects whatever's actually deployed -- currently Phase 1 +
Phase 2 probabilistic predictive-minutes mode, Sec65 -- not a frozen
snapshot of whatever config happened to be live when Sec44.2's original ad
hoc version ran).

Run as `python -m src.models.validate_margin_mae_normalization`.
"""

import pandas as pd
from scipy import stats

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.run_final_holdout_check import (
    INCLUDE_LINEUP_ADJUSTMENT, build_phase1_plus_lineup_predictions, build_phase1_predictions,
)
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength import add_team_ratings, build_team_game_log
from src.models.validate_team_strength_baseline import NAIVE_PRIOR_GAMES, _to_wide_games


def _full_range_naive_predictions() -> pd.DataFrame:
    """Same construction as `validate_team_strength_baseline._rated_dev_log`'s
    naive-floor column, just spanning the full dev+holdout range instead of
    stopping at `DEV_MAX_SEASON - 1` -- mirrors `run_final_holdout_check.
    _full_range_long_log`'s own dev-vs-full-range extension pattern."""
    last_season = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, last_season)
    log = add_team_ratings(log)
    log = add_walk_forward_mean(log, "actualScore", NAIVE_PRIOR_GAMES, prefix="naive")
    games = _to_wide_games(log)
    return games[["gameId", "naive_shrunk_mean_home", "naive_shrunk_mean_away"]].rename(
        columns={"naive_shrunk_mean_home": "naive_home", "naive_shrunk_mean_away": "naive_away"})


def build_full_range_comparison() -> pd.DataFrame:
    """One row per dev+holdout game: `pred_home`/`pred_away` from the
    CURRENT fully-adopted config, `naive_home`/`naive_away` from the naive
    floor, `actual_home`/`actual_away`, `season`."""
    last_season = current_nba_season()
    full_log = add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, last_season))
    model_preds = (build_phase1_plus_lineup_predictions(full_log) if INCLUDE_LINEUP_ADJUSTMENT
                   else build_phase1_predictions(full_log))
    naive_preds = _full_range_naive_predictions()
    return model_preds.merge(naive_preds, on="gameId", how="inner")


def per_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per season: model_margin_mae, naive_margin_mae,
    realized_margin_std (true game-to-game unpredictability), and the two
    normalized ratios Sec44.2 used to reframe the raw-MAE trend."""
    df = df.copy()
    df["actual_margin"] = df["actual_home"] - df["actual_away"]
    df["model_margin_err"] = ((df["pred_home"] - df["pred_away"]) - df["actual_margin"]).abs()
    df["naive_margin_err"] = ((df["naive_home"] - df["naive_away"]) - df["actual_margin"]).abs()

    rows = []
    for season, g in df.groupby("season"):
        model_mae = g["model_margin_err"].mean()
        naive_mae = g["naive_margin_err"].mean()
        realized_std = g["actual_margin"].std()
        rows.append({
            "season": season, "n_games": len(g),
            "model_margin_mae": model_mae, "naive_margin_mae": naive_mae, "realized_margin_std": realized_std,
            "model_vs_naive_ratio": model_mae / naive_mae,
            "model_vs_realized_std_ratio": model_mae / realized_std,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def _correlate_vs_season(summary: pd.DataFrame, col: str) -> tuple:
    r, p = stats.pearsonr(summary["season"], summary[col])
    return float(r), float(p)


if __name__ == "__main__":
    print(f"=== Margin-MAE normalization diagnostic (formalized, Sec44.2 -> item 5a) ===", flush=True)
    print(f"INCLUDE_LINEUP_ADJUSTMENT = {INCLUDE_LINEUP_ADJUSTMENT} (mirrors the current live config)\n", flush=True)

    print("Building full dev+holdout comparison (model vs naive floor)...", flush=True)
    full = build_full_range_comparison()
    print(f"n={len(full)} games, seasons {sorted(full['season'].unique())}\n", flush=True)

    summary = per_season_summary(full)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    print("\n=== Per-season Pearson correlations vs season (does the metric trend with time?) ===", flush=True)
    for col, label in (
        ("model_margin_mae", "raw model margin_mae"),
        ("realized_margin_std", "realized margin std (true task difficulty)"),
        ("model_vs_naive_ratio", "model_margin_mae / naive_margin_mae (relative skill vs floor)"),
        ("model_vs_realized_std_ratio", "model_margin_mae / realized_margin_std (error vs true difficulty)"),
    ):
        r, p = _correlate_vs_season(summary, col)
        print(f"  {label}: r={r:+.4f}  p={p:.4g}", flush=True)

    dev_summary, holdout_summary = split_dev_holdout(summary)
    print(f"\n=== Dev (seasons < {DEV_MAX_SEASON}) vs holdout mean, normalized ratios ===", flush=True)
    for col, label in (
        ("model_vs_naive_ratio", "model_margin_mae / naive_margin_mae"),
        ("model_vs_realized_std_ratio", "model_margin_mae / realized_margin_std"),
    ):
        dev_mean = dev_summary[col].mean()
        holdout_mean = holdout_summary[col].mean()
        direction = "BETTER (lower)" if holdout_mean < dev_mean else "worse (higher)"
        print(f"  {label}: dev={dev_mean:.4f}  holdout={holdout_mean:.4f}  -> holdout is {direction} than dev", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    r_raw, p_raw = _correlate_vs_season(summary, "model_margin_mae")
    r_naive_ratio, p_naive_ratio = _correlate_vs_season(summary, "model_vs_naive_ratio")
    r_std_ratio, p_std_ratio = _correlate_vs_season(summary, "model_vs_realized_std_ratio")
    raw_trend_real = p_raw < 0.05 and r_raw > 0
    normalized_trend_real = (p_naive_ratio < 0.05 and r_naive_ratio > 0) or (p_std_ratio < 0.05 and r_std_ratio > 0)
    if raw_trend_real and not normalized_trend_real:
        print("Raw margin_mae rises significantly with season, but NEITHER normalized ratio does -- "
              "consistent with Sec44.2's original finding: the raw trend is very likely predominantly a "
              "scale artifact of rising real game-to-game unpredictability, not genuine model decay.", flush=True)
    elif raw_trend_real and normalized_trend_real:
        print("Raw margin_mae rises significantly AND at least one normalized ratio also rises significantly "
              "-- this would REVERSE Sec44.2's finding and warrant re-opening margin drift as a real, "
              "unresolved problem. Do not treat this as settled without re-reading the printed correlations above.", flush=True)
    else:
        print("Raw margin_mae does not show a significant trend with season at the current sample size.", flush=True)
