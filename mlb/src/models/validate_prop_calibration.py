"""The one validation this project never actually did until now: are the
PROP-LEVEL probabilities themselves calibrated against real outcomes, not
just the final score? Every other validator in this project (validate_game_
simulator.py, validate_predictive_bullpen.py) checks game-level accuracy
(total/margin MAE, straight-up win accuracy) -- but props.py's actual stated
deliverable is per-batter/per-pitcher prop probabilities ("this batter has a
34% chance of 1+ hits tonight"), and nothing has ever checked whether THOSE
specific numbers are trustworthy.

Uses props.py's real, deployed generate_game_props path (the predictive/
live-usable one, with sampled roster bullpen -- NOT the oracle real-sequence
backtest) on real completed games, since that's what's actually used for
real predictions. Real lineups/starters are used (this is a backtest of
already-played games, so they're known), matching validate_game_simulator.py's
own convention -- only the BULLPEN remains genuinely predictive.
"""

import numpy as np
import pandas as pd

from src.models.props import HIT_OUTCOMES, RBI_EXCLUDED_OUTCOMES, build_pregame_context, generate_game_props
from src.utils.paths import DATA_PROCESSED, DATA_RAW

N_GAMES = 150
N_TRIALS = 150  # reduced from props.py's live default (1000) -- this validator
                 # aggregates across many games, so per-game trial precision
                 # matters less than it would for a single live prediction.
TEST_SEASONS = [2024, 2025]

BATTER_PROP_COLUMNS = ["p_1plus_hit", "p_2plus_hits", "p_1plus_hr", "p_1plus_bb", "p_1plus_rbi"]
PITCHER_PROP_COLUMNS = ["p_6plus_k"]


def _real_batter_outcomes(game_pa: pd.DataFrame) -> pd.DataFrame:
    """Real per-batter outcome counts for one specific already-played game --
    the ground truth to check predicted batter props against."""
    tmp = game_pa.copy()
    tmp["is_hit"] = tmp["outcome"].isin(HIT_OUTCOMES)
    tmp["is_hr"] = tmp["outcome"] == "home_run"
    tmp["is_bb"] = tmp["outcome"].isin({"walk", "intent_walk"})
    tmp["is_k"] = tmp["outcome"] == "strikeout"
    # Task #154: matches props.py's RBI_EXCLUDED_OUTCOMES exactly, so the
    # "real" ground-truth RBI label uses the SAME definition of RBI as the
    # simulated prop being checked against it -- otherwise a recalibration
    # fit would be comparing an improved simulated metric against a stale,
    # still-approximate "ground truth".
    tmp["rbi_eligible_runs"] = tmp["runs_scored"].where(~tmp["outcome"].isin(RBI_EXCLUDED_OUTCOMES), 0)
    real = tmp.groupby("batter").agg(hits=("is_hit", "sum"), hr=("is_hr", "sum"), bb=("is_bb", "sum"),
                                       rbi=("rbi_eligible_runs", "sum"), k=("is_k", "sum")).reset_index()
    real["actual_1plus_hit"] = (real["hits"] >= 1).astype(int)
    real["actual_2plus_hits"] = (real["hits"] >= 2).astype(int)
    real["actual_1plus_hr"] = (real["hr"] >= 1).astype(int)
    real["actual_1plus_bb"] = (real["bb"] >= 1).astype(int)
    real["actual_1plus_rbi"] = (real["rbi"] >= 1).astype(int)
    return real.rename(columns={"batter": "batter_id"})


def _real_pitcher_outcomes(game_pa: pd.DataFrame) -> pd.DataFrame:
    """Real per-pitcher strikeout count for one specific already-played game."""
    tmp = game_pa.copy()
    tmp["is_k"] = tmp["outcome"] == "strikeout"
    real = tmp.groupby("pitcher").agg(k=("is_k", "sum")).reset_index()
    real["actual_6plus_k"] = (real["k"] >= 6).astype(int)
    return real.rename(columns={"pitcher": "pitcher_id"})


def collect_prop_predictions(pa: pd.DataFrame, seasons=TEST_SEASONS, n_games: int = N_GAMES,
                              n_trials: int = N_TRIALS, random_state: int = 1,
                              geometry_hr_enabled: bool = False,
                              geometry_xbh_enabled: bool = False) -> dict[str, pd.DataFrame]:
    """Runs generate_game_props on n_games real completed games (split evenly
    across `seasons`) and returns {batter_props, pitcher_props}, each one row
    per (game_pk, player_id) with both the model's predicted prop
    probabilities and the real outcome from that specific game.

    geometry_hr_enabled/geometry_xbh_enabled: forwarded to build_pregame_
    context (2026-08-05, see its own docstring) -- both False (default) is
    a byte-for-byte no-op."""
    print("building pregame context (all walk-forward tables)...", flush=True)
    ctx = build_pregame_context(pa, geometry_hr_enabled=geometry_hr_enabled,
                                 geometry_xbh_enabled=geometry_xbh_enabled)

    schedules, lineups = {}, {}
    for season in seasons:
        schedules[season] = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
        lineups[season] = pd.read_parquet(DATA_RAW / f"lineups_{season}.parquet")

    candidates = []
    for season in seasons:
        sched = schedules[season][(schedules[season]["game_type"] == "R") & (schedules[season]["status"] == "Final")]
        lu = lineups[season]
        lineup_counts = lu.groupby(["game_pk", "team_side"]).size()
        complete_games = lineup_counts[lineup_counts == 9].reset_index()["game_pk"].unique()
        sched = sched[sched["game_pk"].isin(complete_games)]
        for row in sched.sample(min(len(sched), n_games // len(seasons)), random_state=random_state).itertuples():
            candidates.append((season, row.game_pk, row.home_team, row.away_team, row.date))

    print(f"generating props for {len(candidates)} real games, {n_trials} trials each...", flush=True)
    rng = np.random.default_rng(random_state)
    batter_rows, pitcher_rows = [], []
    for i, (season, game_pk, home_team, away_team, game_date) in enumerate(candidates):
        lu = lineups[season]
        game_lu = lu[lu["game_pk"] == game_pk]
        home_ids = game_lu[game_lu["team_side"] == "home"].sort_values("batting_order")["player_id"].tolist()
        away_ids = game_lu[game_lu["team_side"] == "away"].sort_values("batting_order")["player_id"].tolist()
        if len(home_ids) != 9 or len(away_ids) != 9:
            continue

        game_pa = pa[pa["game_pk"] == game_pk]
        top_pa = game_pa[game_pa["inning_topbot"] == "Top"].sort_values(["inning", "at_bat_number"])
        bot_pa = game_pa[game_pa["inning_topbot"] == "Bot"].sort_values(["inning", "at_bat_number"])
        if top_pa.empty or bot_pa.empty:
            continue
        home_pitcher_id, away_pitcher_id = int(top_pa["pitcher"].iloc[0]), int(bot_pa["pitcher"].iloc[0])

        try:
            props = generate_game_props(ctx, season, game_pk, home_team, away_team, game_date,
                                         home_ids, away_ids, home_pitcher_id, away_pitcher_id,
                                         n_trials=n_trials, rng=rng)
        except ValueError:
            continue  # missing pregame snapshot (e.g. an MLB debut) -- same skip as generate_daily_props.py

        real_batters = _real_batter_outcomes(game_pa)
        bp = props["batter_props"].reset_index().rename(columns={"batter_id": "batter_id"})
        bp = bp.merge(real_batters, on="batter_id", how="inner")
        bp["game_pk"] = game_pk
        batter_rows.append(bp)

        real_pitchers = _real_pitcher_outcomes(game_pa)
        pp = props["pitcher_props"].reset_index().rename(columns={"pitcher_id": "pitcher_id"})
        pp = pp.merge(real_pitchers, on="pitcher_id", how="inner")
        pp["game_pk"] = game_pk
        pitcher_rows.append(pp)

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(candidates)} games done", flush=True)

    return {
        "batter_props": pd.concat(batter_rows, ignore_index=True) if batter_rows else pd.DataFrame(),
        "pitcher_props": pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else pd.DataFrame(),
    }


def calibration_report(df: pd.DataFrame, pred_col: str, actual_col: str, n_buckets: int = 8) -> pd.DataFrame:
    d = df.dropna(subset=[pred_col, actual_col]).copy()
    d["bucket"] = pd.qcut(d[pred_col], min(n_buckets, d[pred_col].nunique()), labels=False, duplicates="drop")
    return d.groupby("bucket").agg(mean_predicted=(pred_col, "mean"), actual_rate=(actual_col, "mean"),
                                     n=(actual_col, "size"))


def report_calibration(df: pd.DataFrame, pred_col: str, actual_col: str, label: str) -> None:
    report = calibration_report(df, pred_col, actual_col)
    if len(report) < 3:
        print(f"\n=== {label}: too few distinct buckets to report (n={len(df)}) ===")
        return
    b, a = np.polyfit(report["mean_predicted"].to_numpy(dtype=float), report["actual_rate"].to_numpy(dtype=float), 1)
    corr = report["mean_predicted"].corr(report["actual_rate"])
    d = df.dropna(subset=[pred_col, actual_col])
    brier = ((d[pred_col] - d[actual_col]) ** 2).mean()
    naive_p = d[actual_col].mean()
    naive_brier = ((naive_p - d[actual_col]) ** 2).mean()
    print(f"\n=== {label} (n={len(d)} player-games) ===")
    print(report.round(4))
    print(f"calibration: actual = {a:.4f} + {b:.4f}*predicted (slope 1.0 = perfect)  corr={corr:.4f}")
    print(f"Brier: model={brier:.4f}  naive(always {naive_p:.3f})={naive_brier:.4f}")


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    result = collect_prop_predictions(pa)
    result["batter_props"].to_parquet(DATA_PROCESSED / "prop_calibration_batter.parquet", index=False)
    result["pitcher_props"].to_parquet(DATA_PROCESSED / "prop_calibration_pitcher.parquet", index=False)

    bp = result["batter_props"]
    print(f"\n\n{'='*30} BATTER PROPS {'='*30}")
    for prop_col, actual_col in [("p_1plus_hit", "actual_1plus_hit"), ("p_2plus_hits", "actual_2plus_hits"),
                                   ("p_1plus_hr", "actual_1plus_hr"), ("p_1plus_bb", "actual_1plus_bb"),
                                   ("p_1plus_rbi", "actual_1plus_rbi")]:
        report_calibration(bp, prop_col, actual_col, prop_col)
        # RAW (pre-calibration) report (M19 bypass, see props.py's
        # _apply_batter_prop_calibration): THIS is the column any
        # BATTER_PROP_CALIBRATION refit must be fit against -- fitting on
        # the calibrated column composes the old slopes into the new ones.
        raw_col = f"{prop_col}_raw"
        if raw_col in bp.columns:
            report_calibration(bp, raw_col, actual_col, f"{prop_col} (RAW -- refit against this)")

    pp = result["pitcher_props"]
    print(f"\n\n{'='*30} PITCHER PROPS {'='*30}")
    report_calibration(pp, "p_6plus_k", "actual_6plus_k", "p_6plus_k")

    print(f"\n\n{'='*30} CONTINUOUS PROP MAE (mean_X vs. real count) {'='*30}")
    for mean_col, real_col in [("mean_hits", "hits"), ("mean_k", "k")]:
        if mean_col in bp.columns and real_col in bp.columns:
            mae = (bp[mean_col] - bp[real_col]).abs().mean()
            naive_mae = (bp[real_col].mean() - bp[real_col]).abs().mean()
            print(f"batter {mean_col}: MAE={mae:.4f}  naive={naive_mae:.4f}  corr={bp[mean_col].corr(bp[real_col]):.4f}")
    if "mean_k" in pp.columns:
        mae = (pp["mean_k"] - pp["k"]).abs().mean()
        naive_mae = (pp["k"].mean() - pp["k"]).abs().mean()
        print(f"pitcher mean_k: MAE={mae:.4f}  naive={naive_mae:.4f}  corr={pp['mean_k'].corr(pp['k']):.4f}")
