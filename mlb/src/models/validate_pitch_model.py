"""Phase 1 go/no-go gate for the pitch-by-pitch simulator rebuild: does
COMPOSING pitch_talent.py's per-bucket batter rates through the deterministic
count-state machinery (pitch_talent.resolve_terminal_probs) reproduce a
batter's real, full-season K/BB/HBP rate at least as well as the CURRENT
true_talent.py direct-rate approach -- leakage-free (preseason-only
prediction vs. real target-season actual), same methodology used to validate
every other factor in this project? Also home to the Phase 2a gate:
build_matchup_pitch_probs, which closes the gap Phase 1's batter-only check
left open by exercising the real batter x pitcher x league odds-ratio
combine end-to-end (see its own docstring).

resolve_terminal_probs itself lives in pitch_talent.py, not here -- it's
load-bearing production code (Phase 2 wires it into game_simulator.py), not
validator-only logic.
"""

import numpy as np
import pandas as pd

from src.models.matchup import combine_odds_ratio
from src.models.pitch_talent import DECISIONS, build_preseason_bucket_rates, resolve_terminal_probs
from src.models.true_talent import build_preseason_priors
from src.utils.paths import DATA_PROCESSED


def build_composed_predictions(pitches: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """One row per batter: composed preseason-only strikeout/walk/hbp
    probability for target_season, via resolve_terminal_probs."""
    decision_data = {}
    for decision in DECISIONS:
        priors_df, league_rates = build_preseason_bucket_rates(pitches, "batter", decision, target_season)
        decision_data[decision] = (priors_df, league_rates)

    batters = set()
    for decision in DECISIONS:
        batters |= set(decision_data[decision][0]["batter"].unique())

    rows = []
    for batter in batters:
        bucket_rates = {}
        for decision in DECISIONS:
            priors_df, league_rates = decision_data[decision]
            player_rows = priors_df[priors_df["batter"] == batter]
            rates = dict(zip(player_rows["bucket"], player_rows["preseason_rate"]))
            for bucket, lr in league_rates.items():
                rates.setdefault(bucket, lr)
            bucket_rates[decision] = rates
        terminal = resolve_terminal_probs(bucket_rates)
        rows.append({"batter": batter, "composed_k": terminal["strikeout"],
                      "composed_bb": terminal["walk"], "composed_hbp": terminal["hit_by_pitch"]})
    return pd.DataFrame(rows)


def _true_talent_predictions(pa: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Current model's own preseason-only prediction (build_preseason_priors,
    not the in-season-blended build_pregame_rates) for the same 3 outcomes,
    for a head-to-head comparison on the exact same basis."""
    frames = []
    for outcome, col in [("strikeout", "tt_k"), ("walk", "tt_bb"), ("hit_by_pitch", "tt_hbp")]:
        priors_df, league_rate = build_preseason_priors(pa, "batter", outcome, target_season)
        if priors_df.empty:
            continue
        frames.append(priors_df.rename(columns={"preseason_rate": col})[["batter", col]])
    if not frames:
        return pd.DataFrame(columns=["batter", "tt_k", "tt_bb", "tt_hbp"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="batter", how="outer")
    return out


def _real_rates(pa: pd.DataFrame, target_season: int) -> pd.DataFrame:
    season_pa = pa[pa["season"] == target_season]
    g = season_pa.groupby("batter")
    out = g.agg(n_pa=("outcome", "size")).reset_index()
    out["real_k"] = g["outcome"].apply(lambda s: (s == "strikeout").mean()).values
    out["real_bb"] = g["outcome"].apply(lambda s: (s == "walk").mean()).values
    out["real_hbp"] = g["outcome"].apply(lambda s: (s == "hit_by_pitch").mean()).values
    return out


def build_matchup_pitch_probs(pitches: pd.DataFrame, pa: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Phase 2a go/no-go gate: does the actual BATTER x PITCHER x LEAGUE
    matchup combine (via matchup.py's existing combine_odds_ratio, one call
    per decision per bucket) feeding resolve_terminal_probs produce
    well-calibrated, PA-level strikeout/walk/hit_by_pitch predictions?

    Phase 1's own validation used a batter's own rate vs. an EXACTLY-average
    opponent, which collapses combine_odds_ratio down to just the batter's
    own rate (odds(lg)*odds(b)/odds(lg)*odds(lg)/odds(lg) = odds(b)) -- the
    3-way combine itself was never actually exercised until this check.
    Preseason-only bucket rates (not the in-season-blended
    build_pregame_bucket_rates) are used here deliberately: this is a test of
    whether the COMBINE MECHANISM itself is sound, not a test of in-season
    blending, which build_pregame_bucket_rates already covers separately.

    One row per real PA in target_season, with the composed matchup
    probability plus what actually happened -- feed into a bucketed
    calibration_report the same way matchup.py's own __main__ does for
    strikeout/home_run."""
    league_rates = {}
    batter_priors, pitcher_priors = {}, {}
    for decision in DECISIONS:
        b_df, b_league = build_preseason_bucket_rates(pitches, "batter", decision, target_season)
        p_df, _ = build_preseason_bucket_rates(pitches, "pitcher", decision, target_season)
        league_rates[decision] = b_league  # league_rate doesn't depend on player_col -- same population either way
        batter_priors[decision] = {(r.batter, r.bucket): r.preseason_rate for r in b_df.itertuples()}
        pitcher_priors[decision] = {(r.pitcher, r.bucket): r.preseason_rate for r in p_df.itertuples()}

    season_pa = pa[pa["season"] == target_season].copy()
    pairs = season_pa[["batter", "pitcher"]].drop_duplicates()

    pair_results = []
    for pair in pairs.itertuples(index=False):
        batter, pitcher = pair.batter, pair.pitcher
        bucket_rates = {}
        for decision in DECISIONS:
            lr = league_rates[decision]
            rates = {}
            for bucket, lg in lr.items():
                b = batter_priors[decision].get((batter, bucket), lg)
                p = pitcher_priors[decision].get((pitcher, bucket), lg)
                rates[bucket] = combine_odds_ratio(b, p, lg)
            bucket_rates[decision] = rates
        terminal = resolve_terminal_probs(bucket_rates)
        pair_results.append({"batter": batter, "pitcher": pitcher, "pred_k": terminal["strikeout"],
                              "pred_bb": terminal["walk"], "pred_hbp": terminal["hit_by_pitch"]})
    pair_df = pd.DataFrame(pair_results)

    season_pa = season_pa.merge(pair_df, on=["batter", "pitcher"], how="inner")
    season_pa["actual_k"] = (season_pa["outcome"] == "strikeout").astype(int)
    season_pa["actual_bb"] = (season_pa["outcome"] == "walk").astype(int)
    season_pa["actual_hbp"] = (season_pa["outcome"] == "hit_by_pitch").astype(int)
    return season_pa


def calibration_report(merged: pd.DataFrame, pred_col: str, actual_col: str, n_buckets: int = 10) -> pd.DataFrame:
    test = merged.copy()
    test["bucket"] = pd.qcut(test[pred_col], n_buckets, labels=False, duplicates="drop")
    return test.groupby("bucket").agg(
        mean_predicted=(pred_col, "mean"), actual_rate=(actual_col, "mean"), n=(actual_col, "size")
    )


if __name__ == "__main__":
    pitches = pd.read_parquet(DATA_PROCESSED / "pitch_table_2023_2026.parquet")
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")

    seasons_available = sorted(pitches["season"].unique())
    target_seasons = [s for s in seasons_available if s - 1 in seasons_available]

    print("=== Phase 1 gate: composed pitch-level model vs. current true_talent.py, "
          "leakage-free preseason-only prediction vs. real target-season actual ===\n")

    for target_season in target_seasons:
        composed = build_composed_predictions(pitches, target_season)
        tt = _true_talent_predictions(pa, target_season)
        real = _real_rates(pa, target_season)

        merged = composed.merge(tt, on="batter", how="inner").merge(real, on="batter", how="inner")
        merged = merged[merged["n_pa"] >= 100]

        print(f"--- target season {target_season} (n_batters={len(merged)}, min 100 PA) ---")
        for label, composed_col, tt_col, real_col in [
            ("strikeout", "composed_k", "tt_k", "real_k"),
            ("walk", "composed_bb", "tt_bb", "real_bb"),
            ("hit_by_pitch", "composed_hbp", "tt_hbp", "real_hbp"),
        ]:
            sub = merged.dropna(subset=[composed_col, tt_col, real_col])
            composed_corr = sub[composed_col].corr(sub[real_col])
            tt_corr = sub[tt_col].corr(sub[real_col])
            composed_mae = (sub[composed_col] - sub[real_col]).abs().mean()
            tt_mae = (sub[tt_col] - sub[real_col]).abs().mean()
            print(f"  {label:14s} composed: corr={composed_corr:.3f} MAE={composed_mae:.4f}   "
                  f"true_talent: corr={tt_corr:.3f} MAE={tt_mae:.4f}   (n={len(sub)})")
        print()

    print("\n=== Phase 2a gate: matchup-level (batter x pitcher x league) PA-level calibration ===\n")
    for target_season in target_seasons:
        matchup_pa = build_matchup_pitch_probs(pitches, pa, target_season)
        print(f"--- target season {target_season} (n_pa={len(matchup_pa)}) ---")
        for label, pred_col, actual_col in [
            ("strikeout", "pred_k", "actual_k"), ("walk", "pred_bb", "actual_bb"), ("hit_by_pitch", "pred_hbp", "actual_hbp"),
        ]:
            report = calibration_report(matchup_pa, pred_col, actual_col)
            corr = report["mean_predicted"].corr(report["actual_rate"])
            b, a = np.polyfit(report["mean_predicted"].to_numpy(dtype=float), report["actual_rate"].to_numpy(dtype=float), 1)
            mae = (matchup_pa[pred_col] - matchup_pa[actual_col]).abs().mean()
            naive_mae = (matchup_pa[actual_col].mean() - matchup_pa[actual_col]).abs().mean()
            print(f"  {label:14s} bucket-calib: actual = {a:.4f} + {b:.4f}*predicted (slope 1.0 = perfect)  "
                  f"corr={corr:.4f}  PA-level MAE={mae:.4f}  naive_MAE={naive_mae:.4f}")
        print()
