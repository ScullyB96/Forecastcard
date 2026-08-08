"""Slice-level calibration diagnostics for the game-level win-probability
output (`validate_game_simulator.py`'s `sim_home_win_prob`) -- the check this
project has never actually done. Every existing validator reports AGGREGATE
SU/Brier/log-loss across the whole test set, but an aggregate number can look
healthy while a specific real-world slice (a big favorite, a low-scoring
pitcher's duel, one specific park) is badly miscalibrated underneath it --
invisible until you actually bin by something and look.

Four independent slices, each answering a different "is this real" question:

1. RELIABILITY CURVE (predicted-probability decile -> actual win rate): the
   textbook calibration check. If the model says "70%" a hundred times, real
   outcomes should land at ~70% too, not 60% or 80%. Uses raw home-win-prob
   deciles.
2. CONFIDENCE TERCILE (|p-0.5| -- how confident the model is, regardless of
   home/away): reframed onto "the favored side" so a big home favorite and a
   big road favorite are pooled onto one comparable scale. Answers "are our
   most-confident picks actually right more often," not just "is 50/50 called
   50/50."
3. SCORING-ENVIRONMENT TERCILE (predicted total runs): pitcher's duels vs.
   slugfests are a real different variance regime -- checks whether SU/Brier
   holds up across both, not just in aggregate.
4. PER-PARK (home team): the most likely place a real, undetected bias would
   hide (a park-factor error, a home-field-advantage misfit specific to one
   venue) -- gated on a minimum sample size since 30 parks over ~7000 games is
   already a thin ~240 games/park before slicing further.

All three tercile/decile splits use Wilson score confidence intervals on the
observed rate (robust near 0/1, unlike a naive normal approximation) and flag
a bin "OUT OF CI" only when the model's own predicted mean falls outside that
bin's real-outcome CI -- the same "don't over-read a noisy bin" discipline
this project already applies everywhere else (see ab_significance.py).
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED, DATA_RAW


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion -- correct at the
    extremes (near 0 or 1) where a naive p +/- z*se(p) interval can go
    negative or above 1, which real predicted-win-prob deciles routinely are."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def _load_validation(path=None) -> pd.DataFrame:
    path = path or (DATA_PROCESSED / "game_simulator_validation.parquet")
    r = pd.read_parquet(path)
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["sim_total_mean"] = r["sim_home_mean"] + r["sim_away_mean"]
    r["brier"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
    r["su_correct"] = ((r["sim_home_win_prob"] > 0.5).astype(int) == r["actual_home_win"]).astype(int)
    return r


def _join_schedule(r: pd.DataFrame) -> pd.DataFrame:
    """Attach home_team/venue_name for the per-park slice -- the validation
    parquet itself only carries game_pk, so this is a real join, not a
    pre-existing column."""
    schedules = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA_RAW.glob("schedule_*.parquet"))],
        ignore_index=True,
    )[["game_pk", "home_team", "venue_name"]].drop_duplicates("game_pk")
    return r.merge(schedules, on="game_pk", how="left")


def reliability_curve(r: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Decile bins of sim_home_win_prob (quantile-based, so each bin has a
    comparable real sample size rather than equal-width bins that would be
    nearly empty near 0/1) -- mean predicted vs. actual win rate per bin."""
    bins = pd.qcut(r["sim_home_win_prob"], n_bins, duplicates="drop")
    rows = []
    for interval, grp in r.groupby(bins, observed=True):
        n = len(grp)
        wins = int(grp["actual_home_win"].sum())
        lo, hi = wilson_ci(wins, n)
        pred_mean = grp["sim_home_win_prob"].mean()
        rows.append({
            "bin": str(interval), "n": n, "pred_mean": pred_mean,
            "actual_rate": wins / n, "ci_lo": lo, "ci_hi": hi,
            "out_of_ci": not (lo <= pred_mean <= hi),
        })
    return pd.DataFrame(rows)


def confidence_tercile(r: pd.DataFrame) -> pd.DataFrame:
    """Reframes every game onto 'the favored side' (whichever of home/away
    sim_home_win_prob favors) so home and road favorites of similar
    confidence pool onto one scale, then terciles by |p-0.5|. A game with
    sim_home_win_prob=0.5 exactly (favored_prob=0.5) is real but uninformative
    for this check -- kept in, just contributes ~0 signal either way."""
    tmp = r.copy()
    tmp["favored_prob"] = np.maximum(tmp["sim_home_win_prob"], 1 - tmp["sim_home_win_prob"])
    tmp["favored_won"] = np.where(
        tmp["sim_home_win_prob"] >= 0.5, tmp["actual_home_win"], 1 - tmp["actual_home_win"],
    )
    tercile = pd.qcut(tmp["favored_prob"], 3, labels=["close", "medium", "lopsided"])
    rows = []
    for label, grp in tmp.groupby(tercile, observed=True):
        n = len(grp)
        wins = int(grp["favored_won"].sum())
        lo, hi = wilson_ci(wins, n)
        pred_mean = grp["favored_prob"].mean()
        rows.append({
            "tercile": str(label), "n": n, "pred_favored_win_rate": pred_mean,
            "actual_favored_win_rate": wins / n, "ci_lo": lo, "ci_hi": hi,
            "out_of_ci": not (lo <= pred_mean <= hi),
            "brier": grp["brier"].mean(), "su": grp["su_correct"].mean(),
        })
    return pd.DataFrame(rows)


def scoring_environment_tercile(r: pd.DataFrame) -> pd.DataFrame:
    """Terciles by the model's OWN predicted total runs (sim_total_mean, not
    actual -- a real pregame-available split, not one that peeks at the
    outcome) -- checks whether SU/Brier holds up in pitcher's duels vs.
    slugfests, not just in aggregate."""
    tercile = pd.qcut(r["sim_total_mean"], 3, labels=["low_scoring", "medium_scoring", "high_scoring"])
    rows = []
    for label, grp in r.groupby(tercile, observed=True):
        n = len(grp)
        wins = int(grp["actual_home_win"].sum())
        lo, hi = wilson_ci(wins, n)
        pred_mean = grp["sim_home_win_prob"].mean()
        rows.append({
            "tercile": str(label), "n": n, "mean_pred_total": grp["sim_total_mean"].mean(),
            "mean_actual_total": grp["actual_total"].mean(),
            "pred_home_win_mean": pred_mean, "actual_home_win_rate": wins / n,
            "ci_lo": lo, "ci_hi": hi, "out_of_ci": not (lo <= pred_mean <= hi),
            "brier": grp["brier"].mean(), "su": grp["su_correct"].mean(),
        })
    return pd.DataFrame(rows)


def per_park(r: pd.DataFrame, min_n: int = 150) -> pd.DataFrame:
    """Per-home-team calibration, gated on min_n -- ~30 teams over ~7237
    games is already a thin ~241 games/team average before any further
    slicing, so a park showing up here at all is already at the edge of
    what this sample size can resolve; treat every row as a lead to
    re-check with more data, not a standalone verdict."""
    rows = []
    for team, grp in r.groupby("home_team", observed=True):
        n = len(grp)
        if n < min_n:
            continue
        wins = int(grp["actual_home_win"].sum())
        lo, hi = wilson_ci(wins, n)
        pred_mean = grp["sim_home_win_prob"].mean()
        rows.append({
            "home_team": team, "n": n, "pred_home_win_mean": pred_mean,
            "actual_home_win_rate": wins / n, "ci_lo": lo, "ci_hi": hi,
            "out_of_ci": not (lo <= pred_mean <= hi),
            "brier": grp["brier"].mean(), "su": grp["su_correct"].mean(),
        })
    return pd.DataFrame(rows).sort_values("home_team")


if __name__ == "__main__":
    r = _load_validation()
    r = _join_schedule(r)
    print(f"loaded {len(r)} games from game_simulator_validation.parquet\n")

    print("=== 1. RELIABILITY CURVE (predicted win-prob decile vs. actual rate) ===")
    rc = reliability_curve(r)
    print(rc.to_string(index=False))
    n_out = int(rc["out_of_ci"].sum())
    print(f"{n_out}/{len(rc)} bins OUT OF CI\n")

    print("=== 2. CONFIDENCE TERCILE (favored-side win rate, home/road pooled) ===")
    ct = confidence_tercile(r)
    print(ct.to_string(index=False))
    print(f"{int(ct['out_of_ci'].sum())}/{len(ct)} terciles OUT OF CI\n")

    print("=== 3. SCORING-ENVIRONMENT TERCILE (by predicted total runs) ===")
    st = scoring_environment_tercile(r)
    print(st.to_string(index=False))
    print(f"{int(st['out_of_ci'].sum())}/{len(st)} terciles OUT OF CI\n")

    print("=== 4. PER-PARK (home team, min n=150) ===")
    pp = per_park(r)
    print(pp.to_string(index=False))
    print(f"{int(pp['out_of_ci'].sum())}/{len(pp)} parks OUT OF CI (of {r['home_team'].nunique()} total teams, "
          f"{len(pp)} met the min-n threshold)")
