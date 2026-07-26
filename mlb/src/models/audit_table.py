"""The systematic audit table (2026-07-25), replacing ad-hoc factor-hunting
with a ranked checklist: for every outcome category x side (batter/pitcher),
how much real variance does the CURRENT PRODUCTION estimate leave
unexplained, and how much does that unexplained variance actually matter in
run-value terms? This is exactly how the GB/FB pitcher HR-allowed signal
(the single largest win found this session) was actually discovered --
generalizing that discovery process into a standing tool instead of relying
on which ideas happen to come up in conversation.

Method, matching this project's own established leakage-free convention
used throughout (barrel rate, pulled-air, sprint speed, GB/FB, etc. in
expected_stats.py): for target_season in {2024, 2025} (both have real prior
seasons to build a genuine preseason prior from, unlike 2023), compute each
qualifying player's CURRENT PRODUCTION preseason estimate
(true_talent.build_preseason_priors -- the exact Marcel/Carleton-K/age-
adjusted prior actually used in production, not a re-derived one) and
correlate it against that player's REAL realized rate over the whole target
season. R^2 = corr^2. Multiply (1 - R^2) by each category's run-value
leverage (LINEAR_WEIGHTS x average real per-PA frequency) to rank where a
BETTER ESTIMATOR (not just a better shrinkage constant -- task #64 already
individually K-swept every category) would move the needle most.

This does NOT re-litigate whether K is well-tuned (already done, see
STABILIZATION_PA_BATTER/_PITCHER's own per-category sweep comments) -- it
asks a different, complementary question: even at the best possible K for
this estimator SHAPE (outcome-history-only, Marcel-shrunk), how much real
signal is still on the table, and is it worth chasing for THIS category
specifically."""
import numpy as np
import pandas as pd

from src.models.park_factors import build_outcome_park_factors
from src.models.run_value_screen import LINEAR_WEIGHTS
from src.models.true_talent import (
    STABILIZATION_PA_BATTER,
    STABILIZATION_PA_PITCHER,
    build_preseason_priors,
)
from src.utils.paths import DATA_PROCESSED

TARGET_SEASONS = [2024, 2025]
MIN_REAL_PA = 100  # require a real, reasonably stable target-season rate to correlate against


def _real_season_rate(pa: pd.DataFrame, player_col: str, outcome: str, target_season: int) -> pd.DataFrame:
    """Real (not park-neutral, not shrunk) rate for `outcome` over the whole
    target_season -- the ground truth the preseason estimate is checked
    against. No leakage concern here (this IS the season being predicted),
    only build_preseason_priors' own inputs are restricted to prior seasons."""
    sub = pa[pa["season"] == target_season]
    is_outcome = (sub["outcome"] == outcome).astype(int)
    out = sub.groupby(player_col).agg(pa_count=("outcome", "size"))
    out["events"] = is_outcome.groupby(sub[player_col]).sum()
    out["real_rate"] = out["events"] / out["pa_count"]
    return out.reset_index()[[player_col, "pa_count", "real_rate"]]


def audit_category(pa: pd.DataFrame, park_factors_df: pd.DataFrame, player_col: str, outcome: str) -> dict:
    corrs, ns, freqs = [], [], []
    for target_season in TARGET_SEASONS:
        priors_df, league_rate = build_preseason_priors(pa, player_col, outcome, target_season, park_factors_df)
        if priors_df.empty:
            continue
        real_df = _real_season_rate(pa, player_col, outcome, target_season)
        real_df = real_df[real_df["pa_count"] >= MIN_REAL_PA]
        merged = priors_df.merge(real_df, on=player_col, how="inner")
        if len(merged) < 20:
            continue
        corr = merged["preseason_rate"].corr(merged["real_rate"])
        if pd.notna(corr):
            corrs.append(corr)
            ns.append(len(merged))
            freqs.append(merged["real_rate"].mean())
    if not corrs:
        return {"player_col": player_col, "outcome": outcome, "r2": np.nan, "n_seasons": 0,
                "mean_n": 0, "mean_freq": np.nan, "leverage": np.nan, "priority_score": np.nan}
    mean_corr = np.mean(corrs)
    r2 = mean_corr ** 2
    mean_freq = np.mean(freqs)
    leverage = abs(LINEAR_WEIGHTS.get(outcome, 0.0)) * mean_freq
    return {
        "player_col": player_col, "outcome": outcome, "r2": r2, "n_seasons": len(corrs),
        "mean_n": int(np.mean(ns)), "mean_freq": mean_freq, "leverage": leverage,
        "priority_score": (1 - r2) * leverage,
    }


def build_audit_table(pa: pd.DataFrame) -> pd.DataFrame:
    park_factors_df = build_outcome_park_factors(pa)
    rows = []
    for player_col, stab_dict in [("batter", STABILIZATION_PA_BATTER), ("pitcher", STABILIZATION_PA_PITCHER)]:
        for outcome in stab_dict:
            rows.append(audit_category(pa, park_factors_df, player_col, outcome))
    df = pd.DataFrame(rows)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building audit table (target_seasons=2024,2025, leakage-free preseason-vs-real)...", flush=True)
    table = build_audit_table(pa)
    table.to_parquet(DATA_PROCESSED / "audit_table.parquet", index=False)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print(table.round(4).to_string())
