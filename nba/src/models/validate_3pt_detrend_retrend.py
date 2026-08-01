"""Untried-synthesis lever (full-model-audit research pass, 2026-08-01):
tests `player_rate_shrinkage.add_era_adjusted_player_rate` (the detrend-
then-retrend architecture built and validated in Sec17) on 3PT-MAKE-RATE,
a category it was never actually applied to.

Sec17 built and rigorously validated this architecture EXCLUSIVELY for
steals, where it failed decisively on holdout -- but steals' problem was
diagnosed as a genuine REGIME CHANGE with zero within-dev precedent (the
rate dipped 2017-2023, then jumped only at the exact dev/holdout boundary
in 2024), exactly the pattern no historical-extrapolation technique can
predict. 3PT volume is structurally different: confirmed directly from
real per-season data (own diagnostic below) -- attempt/make volume rises
close to continuously across the ENTIRE dev range, much closer to Phase 1
margin's scoring-drift profile (Sec24/29) than to steals' regime change.
This is exactly the kind of category the tool was built for but was never
pointed at.

Same two-stage dev-only discipline as every other candidate fix in this
project: fit on the full dev range, screen the recent-dev slice first,
full-dev-range second, before ever considering a holdout read.

Run as `python -m src.models.validate_3pt_detrend_retrend`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_minutes import build_player_game_log
from src.models.player_rate_shrinkage import add_era_adjusted_player_rate, add_walk_forward_player_rate
from src.models.player_scoring_rates import PRIOR_ATTEMPTS_3PM

RECENT_SLICE_START_SEASON = DEV_MAX_SEASON - 3  # last 3 dev seasons: 2021-2023
CANDIDATE_HALFLIVES = (25.0, 50.0, 100.0, 200.0, 400.0)


def _per_season_diagnostic() -> None:
    """Real-data description (not a repeated holdout read): does 3PT
    volume actually show a continuous trend throughout dev, or a
    regime-change pattern like steals?"""
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["minutes"] > 0]
    by_season_att = log.groupby("season")["fg3Att"].mean()
    by_season_made = log.groupby("season")["fg3Made"].mean()
    print("=== 3PT attempt/make volume by dev season (real data, not a holdout read) ===", flush=True)
    for s in sorted(by_season_att.index):
        print(f"  {s}: mean_3pt_att/player-game={by_season_att[s]:.3f}  mean_3pt_made={by_season_made[s]:.3f}", flush=True)


def _fit_and_predict(log: pd.DataFrame, halflife: float | None) -> pd.DataFrame:
    log = log.dropna(subset=["fg3Att", "fg3Made"]).copy()
    if halflife is None:
        rated = add_walk_forward_player_rate(log, "fg3Made", "fg3Att", PRIOR_ATTEMPTS_3PM, prefix="3pt_make")
    else:
        rated = add_era_adjusted_player_rate(log, "fg3Made", "fg3Att", PRIOR_ATTEMPTS_3PM, prefix="3pt_make",
                                              current_rate_halflife_games=halflife)
    rated["proj_made"] = rated["3pt_make_shrunk_rate"] * rated["fg3Att"]
    return rated.dropna(subset=["proj_made"])


def _arms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["abs_err"] = (d["proj_made"] - d["fg3Made"]).abs()
    d["game_player_key"] = d["gameId"].astype(str) + "_" + d["playerId"].astype(str)
    return d[["game_player_key", "abs_err"]]


def _compare(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, label: str) -> dict:
    print(f"\n=== {label} (n={len(candidate_df)}) ===", flush=True)
    return bootstrap_compare(
        _arms(candidate_df), _arms(baseline_df), game_id_col="game_player_key",
        metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
        label_a="detrend_retrend", label_b="current_flat",
    )


if __name__ == "__main__":
    _per_season_diagnostic()

    print("\nBuilding base player-game log once...", flush=True)
    base_log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    base_log = base_log[base_log["minutes"] > 0].copy()
    cols = ["gameId", "playerId", "gameDate", "season", "fg3Att", "fg3Made"]

    print("Fitting baseline (current flat expanding-shrinkage) on the full dev range once...", flush=True)
    baseline_full = _fit_and_predict(base_log[cols], None)
    baseline_recent = baseline_full[baseline_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)

    print(f"\n=== Stage 1: evaluate on the recent-dev slice (seasons {RECENT_SLICE_START_SEASON}-{DEV_MAX_SEASON - 1}), "
          f"fit on the FULL dev range ===", flush=True)
    candidate_fits = {}
    best_halflife, best_delta = None, 0.0
    for hl in CANDIDATE_HALFLIVES:
        candidate_full = _fit_and_predict(base_log[cols], hl)
        candidate_fits[hl] = candidate_full
        candidate_recent = candidate_full[candidate_full["season"] >= RECENT_SLICE_START_SEASON].reset_index(drop=True)
        result = _compare(baseline_recent, candidate_recent, f"halflife={hl}")
        mae = result["mae"]
        if "REAL IMPROVEMENT" in mae["verdict"] and mae["delta"] < best_delta:
            best_halflife, best_delta = hl, mae["delta"]

    if best_halflife is None:
        print("\nNo candidate halflife showed a real improvement on the recent-dev slice -- stopping here.", flush=True)
    else:
        print(f"\nBest candidate: halflife={best_halflife} (delta={best_delta:+.4f} on recent slice)", flush=True)
        print(f"\n=== Stage 2: same fit, evaluated across the FULL dev range ===", flush=True)
        full_result = _compare(baseline_full, candidate_fits[best_halflife], f"FULL DEV RANGE, halflife={best_halflife}")
        if "REAL IMPROVEMENT" in full_result["mae"]["verdict"]:
            print(f"\nPASSES full-dev-range re-validation -- halflife={best_halflife} is eligible for a "
                  f"genuinely new one-time confirmatory holdout read.", flush=True)
        else:
            print(f"\nDOES NOT hold at full-dev-range scale ({full_result['mae']['verdict']}) -- "
                  f"reverting, not spending a holdout read on this.", flush=True)
