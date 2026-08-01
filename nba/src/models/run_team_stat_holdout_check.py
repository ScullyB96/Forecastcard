"""Task #24's one-time confirmatory holdout check for the 6 team-level
stat rates (OREB/DREB/AST/TOV/STL/BLK), mirroring
`run_final_holdout_check.py`/`run_props_holdout_check.py`'s exact protocol:
build the FULL dev+holdout range predictions from literally the same,
already dev-validated model code (no re-tuning), compute each category's
own per-team-side absolute error, split by `DEV_MAX_SEASON`, and call
`validate_holdout_bootstrap.generic_holdout_confirmatory_check` -- the
dev-vs-holdout GAP bootstrap, not holdout performance in isolation, so
expected dev/holdout drift doesn't misfire as a false veto.

Run as `python -m src.models.run_team_stat_holdout_check` -- ONLY ONCE per
model configuration. Re-running after seeing a result and adjusting
anything defeats the entire point of a confirmatory-veto holdout.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.team_stat_rates import STAT_COLUMNS, add_team_stat_ratings, build_team_stat_game_log, project_team_stat, to_wide_games
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check


def _build_full_range_predictions(label: str, games: pd.DataFrame) -> pd.DataFrame:
    league_avg_col = f"{label}_league_avg_home"
    rows = []
    for row in games.itertuples(index=False):
        home_pred, away_pred = project_team_stat(
            home_attack=getattr(row, f"{label}_attack_rate_home"), home_defense=getattr(row, f"{label}_defense_rate_home"),
            away_attack=getattr(row, f"{label}_attack_rate_away"), away_defense=getattr(row, f"{label}_defense_rate_away"),
            league_avg=getattr(row, league_avg_col),
        )
        rows.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": getattr(row, f"{label}_for_home"), "actual_away": getattr(row, f"{label}_for_away"),
            "pred_home": home_pred, "pred_away": away_pred,
        })
    return pd.DataFrame(rows)


def _per_side_rows(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["gameId", "season"]].assign(abs_err=(df["pred_home"] - df["actual_home"]).abs())
    away = df[["gameId", "season"]].assign(abs_err=(df["pred_away"] - df["actual_away"]).abs())
    return pd.concat([home, away], ignore_index=True)


def run() -> dict:
    current = current_nba_season()
    print(f"=== Task #24 confirmatory holdout check (dev < {DEV_MAX_SEASON}, "
          f"holdout >= {DEV_MAX_SEASON}) ===", flush=True)
    print("THIS IS A ONE-TIME READ. If this has already been run once for this model configuration, "
          "STOP -- re-running after seeing the result and tweaking anything is exactly what the "
          "confirmatory-veto protocol exists to prevent.\n", flush=True)

    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)
    log = add_team_stat_ratings(log)
    games = to_wide_games(log)

    all_results = {}
    for label in STAT_COLUMNS:
        df = _build_full_range_predictions(label, games)
        df = df.dropna(subset=["pred_home", "pred_away"])
        rows = _per_side_rows(df)
        dev, holdout = split_dev_holdout(rows)
        if holdout.empty:
            print(f"  SKIPPED {label}: no holdout-range rows present", flush=True)
            continue
        print(f"\n=== {label.upper()}: dev n={len(dev)}, holdout n={len(holdout)} ===", flush=True)
        result = generic_holdout_confirmatory_check(
            dev["abs_err"].to_numpy(), holdout["abs_err"].to_numpy(), metric_name=f"{label}_mae", higher_is_better=False)
        all_results[label] = result
        metrics_ledger.append_generic_run(
            metrics={"n_dev": len(dev), "n_holdout": len(holdout), f"{label}_mae_dev": result["dev"],
                     f"{label}_mae_holdout": result["holdout"], f"{label}_mae_gap_ci_lo": result["gap_ci95"][0],
                     f"{label}_mae_gap_ci_hi": result["gap_ci95"][1], f"{label}_mae_verdict": result["verdict"]},
            model_name=f"team_stat_{label}_holdout", config_flags={"dev_max_season": DEV_MAX_SEASON},
            notes=f"Task #24: team-level {label} dev-vs-holdout confirmatory gap check",
        )

    print("\n=== SUMMARY ===", flush=True)
    for name, result in all_results.items():
        print(f"  {name:20s} -> {result['verdict']}", flush=True)
    vetoes = [name for name, r in all_results.items() if "VETO" in r["verdict"]]
    if vetoes:
        print(f"\n{len(vetoes)} categor{'y' if len(vetoes)==1 else 'ies'} REAL REGRESSION on holdout -- "
              f"VETO triggered for: {', '.join(vetoes)}. See MODEL_DOCUMENTATION.md for the required writeup.", flush=True)
    else:
        print("\nNo vetoes -- every category's dev/holdout gap is NOISE or an (unusual, non-veto) improvement.", flush=True)

    return all_results


if __name__ == "__main__":
    run()
