"""Dev-range-only market benchmark: how does the model's margin/total accuracy
compare to real historical closing lines, on the seasons where
SportsbookReviewsOnline data actually exists (2015-16 through 2022-23,
2022-23 itself only partial)? This data was fetched and cached early in this
project (`fetch_odds_sbro.py`) as a "nice-to-have calibration check" and then
never actually used by any validation script -- this closes that gap.

**Sign-convention correction (2026-08-02)**: `fetch_odds_sbro.py`'s own
docstring describes `homeSpreadClose` as "positive = that team is the
underdog getting points" -- empirically this is backwards for how the value
ended up stored. Checked directly: `homeSpreadClose` correlates POSITIVELY
with the real `homeScore - awayScore` margin (r=0.445, p<1e-300, n=9367,
2015-16 through 2022-23) and its mean (+2.45) almost exactly matches the
real mean home-court margin (+2.44) -- i.e. `homeSpreadClose` IS the
market's own predicted home margin directly (positive = home favored by
that many points), not a "points given" convention requiring a sign flip.
Used directly as `market_pred_margin = homeSpreadClose` here.

**Cannot benchmark holdout (2024-2025) at all** -- SBRO's own site text says
this archive "will not be updated," confirmed to stop at 2022-23 (itself
only partial, ~664 games through mid-January). This script is dev-range-only
by construction; it does not touch the one-time confirmatory holdout
protocol in any way (no holdout read is possible here even if desired).

Run as `python -m src.models.validate_vs_market`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.ingest.fetch_odds_sbro import FIRST_AVAILABLE_SEASON, LAST_AVAILABLE_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.home_court import fit_home_court_walk_forward
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
from src.utils.paths import DATA_EXTERNAL


def _load_odds(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for y in range(start_year, end_year + 1):
        season_str = f"{y}-{str((y + 1) % 100).zfill(2)}"
        path = DATA_EXTERNAL / f"odds_sbro_{season_str}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise RuntimeError("no cached odds_sbro_*.parquet files found -- run fetch_odds_sbro.py first")
    odds = pd.concat(frames, ignore_index=True)
    odds["gameDate"] = pd.to_datetime(odds["gameDate"])
    return odds


def _build_model_predictions(start_year: int, end_year: int) -> pd.DataFrame:
    log = build_team_game_log(FIRST_DEV_SEASON, end_year)
    l = add_team_ratings(log.copy())
    l = add_walk_forward_mean(l, "actualScore", 5.0, prefix="naive")

    home = l[l["is_home"]].copy()
    away = l[~l["is_home"]].copy()
    games = home.merge(away, on=["gameId", "gameDate", "season"], suffixes=("_home", "_away"))
    games = games.rename(columns={
        "actualScore_home": "actual_home_score", "actualScore_away": "actual_away_score",
        "pace_league_avg_home": "league_avg_pace", "rtg_league_avg_home": "league_avg_rtg",
    })
    games = games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)

    rows = []
    for row in games.itertuples(index=False):
        _, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        rows.append({
            "gameDate": row.gameDate, "homeTeamId": row.team_home, "awayTeamId": row.team_away,
            "season": row.season, "actual_home": row.actual_home_score, "actual_away": row.actual_away_score,
            "pred_home": pred_home, "pred_away": pred_away,
            "naive_home": row.naive_shrunk_mean_home, "naive_away": row.naive_shrunk_mean_away,
        })
    return pd.DataFrame(rows).dropna(subset=["pred_home", "pred_away"])


if __name__ == "__main__":
    print("=== Dev-range market benchmark: model vs. real closing lines vs. naive floor ===", flush=True)
    print(f"SBRO coverage: {FIRST_AVAILABLE_SEASON}-{LAST_AVAILABLE_SEASON} only "
          f"(cannot benchmark holdout, {DEV_MAX_SEASON}+ -- no data exists past 2022-23)\n", flush=True)

    odds = _load_odds(FIRST_AVAILABLE_SEASON, LAST_AVAILABLE_SEASON)
    preds = _build_model_predictions(FIRST_DEV_SEASON, LAST_AVAILABLE_SEASON)

    df = preds.merge(odds, on=["gameDate", "homeTeamId", "awayTeamId"], how="inner",
                      suffixes=("", "_odds"))
    n_before_odds_dropna = len(df)
    df = df.dropna(subset=["homeSpreadClose", "totalClose"]).reset_index(drop=True)
    if len(df) != n_before_odds_dropna:
        # A handful of real SBRO rows have a missing Close line (postponed/rescheduled games with
        # only an Open line captured, or a parsing edge case) -- NaN, not zero, so must be dropped
        # explicitly rather than left in: bootstrap_compare's vectorized numpy .mean() does NOT
        # skip NaN the way pandas' .mean() does, so even one NaN row silently poisons every
        # resample's mean to NaN (confirmed real: this happened on the first run of this script).
        print(f"dropped {n_before_odds_dropna - len(df)} game(s) with a missing closing line", flush=True)
    print(f"matched {len(df)} games with both a model prediction and a real closing line "
          f"(of {len(preds)} model predictions and {len(odds)} odds rows in range)", flush=True)

    df["actual_margin"] = df["actual_home"] - df["actual_away"]
    df["actual_total"] = df["actual_home"] + df["actual_away"]
    df["pred_margin"] = df["pred_home"] - df["pred_away"]
    df["naive_margin"] = df["naive_home"] - df["naive_away"]
    df["market_margin"] = df["homeSpreadClose"]
    df["market_total"] = df["totalClose"]

    df["abs_margin_err_model"] = (df["pred_margin"] - df["actual_margin"]).abs()
    df["abs_margin_err_naive"] = (df["naive_margin"] - df["actual_margin"]).abs()
    df["abs_margin_err_market"] = (df["market_margin"] - df["actual_margin"]).abs()
    df["abs_total_err_model"] = ((df["pred_home"] + df["pred_away"]) - df["actual_total"]).abs()
    df["abs_total_err_market"] = (df["market_total"] - df["actual_total"]).abs()
    df["su_model"] = ((df["pred_margin"] > 0).astype(int) == (df["actual_margin"] > 0).astype(int)).astype(float)
    df["su_market"] = ((df["market_margin"] > 0).astype(int) == (df["actual_margin"] > 0).astype(int)).astype(float)

    df["game_key"] = df["gameDate"].astype(str) + "_" + df["homeTeamId"].astype(str) + "_" + df["awayTeamId"].astype(str)

    print("\n--- margin_mae: model vs naive ---", flush=True)
    bootstrap_compare(
        df[["game_key", "abs_margin_err_model"]].rename(columns={"abs_margin_err_model": "abs_err"}),
        df[["game_key", "abs_margin_err_naive"]].rename(columns={"abs_margin_err_naive": "abs_err"}),
        game_id_col="game_key", metrics=[{"name": "margin_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="model", label_b="naive",
    )

    print("\n--- margin_mae: model vs MARKET (the decision-relevant comparison for this script) ---", flush=True)
    bootstrap_compare(
        df[["game_key", "abs_margin_err_model"]].rename(columns={"abs_margin_err_model": "abs_err"}),
        df[["game_key", "abs_margin_err_market"]].rename(columns={"abs_margin_err_market": "abs_err"}),
        game_id_col="game_key", metrics=[{"name": "margin_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="model", label_b="market_closing_line",
    )

    print("\n--- total_mae: model vs MARKET ---", flush=True)
    bootstrap_compare(
        df[["game_key", "abs_total_err_model"]].rename(columns={"abs_total_err_model": "abs_err"}),
        df[["game_key", "abs_total_err_market"]].rename(columns={"abs_total_err_market": "abs_err"}),
        game_id_col="game_key", metrics=[{"name": "total_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="model", label_b="market_closing_line",
    )

    print("\n--- straight-up accuracy: model vs MARKET (picking the spread favorite) ---", flush=True)
    bootstrap_compare(
        df[["game_key", "su_model"]].rename(columns={"su_model": "su"}),
        df[["game_key", "su_market"]].rename(columns={"su_market": "su"}),
        game_id_col="game_key", metrics=[{"name": "su", "col": "su", "higher_is_better": True}],
        label_a="model", label_b="market_closing_line",
    )

    print("\n=== Raw point estimates (for reference) ===", flush=True)
    print(f"  model margin_mae:  {df['abs_margin_err_model'].mean():.4f}", flush=True)
    print(f"  market margin_mae: {df['abs_margin_err_market'].mean():.4f}", flush=True)
    print(f"  naive margin_mae:  {df['abs_margin_err_naive'].mean():.4f}", flush=True)
    print(f"  model total_mae:   {df['abs_total_err_model'].mean():.4f}", flush=True)
    print(f"  market total_mae:  {df['abs_total_err_market'].mean():.4f}", flush=True)
    print(f"  model SU accuracy:  {df['su_model'].mean():.4f}", flush=True)
    print(f"  market SU accuracy: {df['su_market'].mean():.4f}", flush=True)
