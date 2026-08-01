"""Phase 4 for the props subsystem: the ONE-TIME confirmatory holdout
check for every dev-adopted per-player rate model and both matchup-
difficulty levels, mirroring `run_final_holdout_check.py`'s exact protocol
for the game-score model (Sec9) -- none of these categories has ever been
checked against holdout (season >= `DEV_MAX_SEASON`) before this script.

For each category: builds the FULL dev+holdout range predictions from
literally the same, already dev-validated model code (no re-tuning, no
threshold-picking), computes each category's own per-player-game absolute
error, splits by `DEV_MAX_SEASON`, and calls
`validate_holdout_bootstrap.generic_holdout_confirmatory_check` -- the
same dev-vs-holdout-GAP bootstrap (not holdout performance in isolation)
used for the game-score model, so a genuinely expected amount of dev/
holdout drift doesn't misfire as a false veto.

Run as `python -m src.models.run_props_holdout_check` -- ONLY ONCE per
model configuration. Re-running after seeing a result and adjusting
anything defeats the entire point of a confirmatory-veto holdout.
"""

import numpy as np
import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.matchup_difficulty import (
    MATCHUP_DATA_START_SEASON,
    add_defender_difficulty_rate,
    add_position_group_difficulty_rate,
    build_defender_game_log,
    build_position_group_matchup_log,
)
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_minutes import add_minutes_rating, attach_player_track, build_player_game_log
from src.models.player_playmaking_rates import add_playmaking_rates
from src.models.player_rebounding_rates import add_rebounding_rates
from src.models.player_scoring_rates import add_scoring_rates
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check


def _check_category(df: pd.DataFrame, actual_col: str, proj_col: str, name: str, ledger_model_name: str) -> dict:
    df = df.dropna(subset=[actual_col, proj_col])
    df["abs_err"] = (df[proj_col] - df[actual_col]).abs()
    dev, holdout = split_dev_holdout(df)
    if holdout.empty:
        print(f"  SKIPPED {name}: no holdout-range rows present", flush=True)
        return {}
    print(f"\n=== {name}: dev n={len(dev)}, holdout n={len(holdout)} ===", flush=True)
    result = generic_holdout_confirmatory_check(dev["abs_err"].to_numpy(), holdout["abs_err"].to_numpy(),
                                                 metric_name=f"{name}_mae", higher_is_better=False)
    metrics_ledger.append_generic_run(
        metrics={"n_dev": len(dev), "n_holdout": len(holdout), f"{name}_mae_dev": result["dev"],
                 f"{name}_mae_holdout": result["holdout"], f"{name}_mae_gap_ci_lo": result["gap_ci95"][0],
                 f"{name}_mae_gap_ci_hi": result["gap_ci95"][1], f"{name}_mae_verdict": result["verdict"]},
        model_name=ledger_model_name, config_flags={"dev_max_season": DEV_MAX_SEASON},
        notes=f"Phase 4 (props): {name} dev-vs-holdout confirmatory gap check",
    )
    return result


def run() -> dict:
    current = current_nba_season()
    print(f"=== Props Phase 4 confirmatory holdout check (dev < {DEV_MAX_SEASON}, "
          f"holdout >= {DEV_MAX_SEASON}) ===", flush=True)
    print("THIS IS A ONE-TIME READ. If this has already been run once for this model configuration, "
          "STOP -- re-running after seeing the result and tweaking anything is exactly what the "
          "confirmatory-veto protocol exists to prevent.\n", flush=True)

    all_results = {}

    base_log = build_player_game_log(FIRST_DEV_SEASON, current)
    minutes_log = add_minutes_rating(base_log.copy())
    all_results["minutes"] = _check_category(minutes_log, "minutes", "minutes_ewm_rate", "minutes", "player_minutes")

    scoring_log = add_scoring_rates(base_log.copy())
    scoring_log["points_proj"] = (2 * scoring_log["2pt_proj_made"] + 3 * scoring_log["3pt_proj_made"]
                                   + scoring_log["ft_proj_made"])
    all_results["2pt_made"] = _check_category(scoring_log, "fgMade2", "2pt_proj_made", "2pt_made", "player_scoring_2pt")
    all_results["3pt_made"] = _check_category(scoring_log, "fg3Made", "3pt_proj_made", "3pt_made", "player_scoring_3pt")
    all_results["ft_made"] = _check_category(scoring_log, "ftMade", "ft_proj_made", "ft_made", "player_scoring_ft")

    defensive_log = add_defensive_event_rates(base_log.copy())
    all_results["stl"] = _check_category(defensive_log, "steals", "stl_proj", "stl", "player_defensive_event_stl")
    all_results["blk"] = _check_category(defensive_log, "blocks", "blk_proj", "blk", "player_defensive_event_blk")

    tracked = attach_player_track(base_log.copy(), FIRST_DEV_SEASON, current)
    rebounding_log = add_rebounding_rates(tracked.dropna(subset=["reboundChancesOffensive", "reboundChancesDefensive"]).copy())
    all_results["oreb"] = _check_category(rebounding_log, "oreb", "oreb_proj", "oreb", "player_rebounding_oreb")
    all_results["dreb"] = _check_category(rebounding_log, "dreb", "dreb_proj", "dreb", "player_rebounding_dreb")

    playmaking_log = add_playmaking_rates(tracked.dropna(subset=["touches"]).copy())
    all_results["ast"] = _check_category(playmaking_log, "assists", "ast_proj", "ast", "player_playmaking_ast")
    all_results["tov"] = _check_category(playmaking_log, "turnovers", "tov_proj", "tov", "player_playmaking_tov")

    if current >= MATCHUP_DATA_START_SEASON:
        defender_log = add_defender_difficulty_rate(build_defender_game_log(MATCHUP_DATA_START_SEASON, current))
        all_results["defender_difficulty"] = _check_category(
            defender_log, "points_allowed", "difficulty_proj", "defender_difficulty", "matchup_difficulty_defender_level")

        posgroup_log = add_position_group_difficulty_rate(build_position_group_matchup_log(MATCHUP_DATA_START_SEASON, current))
        posgroup_log["posgroup_proj"] = posgroup_log["posgroup_difficulty_rate"] * posgroup_log["matchup_minutes"]
        all_results["posgroup_difficulty"] = _check_category(
            posgroup_log, "points_allowed", "posgroup_proj", "posgroup_difficulty", "matchup_difficulty_posgroup_level")

    print("\n=== SUMMARY ===", flush=True)
    for name, result in all_results.items():
        if result:
            print(f"  {name:20s} -> {result['verdict']}", flush=True)
    vetoes = [name for name, r in all_results.items() if r and "VETO" in r["verdict"]]
    if vetoes:
        print(f"\n{len(vetoes)} categor{'y' if len(vetoes)==1 else 'ies'} REAL REGRESSION on holdout -- "
              f"VETO triggered for: {', '.join(vetoes)}. See MODEL_DOCUMENTATION.md for the required writeup.", flush=True)
    else:
        print("\nNo vetoes -- every category's dev/holdout gap is NOISE or an (unusual, non-veto) improvement.", flush=True)

    return all_results


if __name__ == "__main__":
    run()
