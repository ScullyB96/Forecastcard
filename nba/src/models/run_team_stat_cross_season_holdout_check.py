"""The one-time confirmatory holdout read for `cross_season_weight` applied to
all 5 ADOPTED_CATEGORIES team_stat_rates categories (dreb/ast/tov/stl/blk) --
the same lever that gave `team_strength.add_team_ratings` two consecutive real
wins (Sec29, Sec33) via the identical `shrinkage.add_walk_forward_rate`
primitive, tested here for the first time on the team_stat_rates categories
(OREB, the 6th category, was tested separately in
`run_oreb_cross_season_holdout_check.py` and did NOT clear holdout -- OREB
stays unanchored and is deliberately excluded here).

Chosen via dev-only Stage 1 (recent-dev-slice sweep at 0.25, all 5 categories
showed real improvement vs current csw=0.0) + Stage 2 (full dev range, all 5
beat BOTH the current config AND the naive floor) -- a genuinely new
configuration never holdout-tested before, eligible for its own one-time read.

Run as `python -m src.models.run_team_stat_cross_season_holdout_check` --
ONLY ONCE.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.shrinkage import add_walk_forward_rate
from src.models.team_stat_rates import ADOPTED_CATEGORIES, PRIOR_GAMES, build_team_stat_game_log, project_team_stat, to_wide_games

CANDIDATE_CROSS_SEASON_WEIGHT = 0.25


def _fit_and_predict(log: pd.DataFrame, label: str, cross_season_weight: float) -> pd.DataFrame:
    l = add_walk_forward_rate(log.copy(), f"{label}_for", f"{label}_against", PRIOR_GAMES[label],
                               prefix=label, cross_season_weight=cross_season_weight)
    games = to_wide_games(l)
    rows = []
    for row in games.itertuples(index=False):
        home_pred, away_pred = project_team_stat(
            home_attack=getattr(row, f"{label}_attack_rate_home"), home_defense=getattr(row, f"{label}_defense_rate_home"),
            away_attack=getattr(row, f"{label}_attack_rate_away"), away_defense=getattr(row, f"{label}_defense_rate_away"),
            league_avg=getattr(row, f"{label}_league_avg_home"),
        )
        rows.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": getattr(row, f"{label}_for_home"), "actual_away": getattr(row, f"{label}_for_away"),
            "pred_home": home_pred, "pred_away": away_pred,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def _per_side(df: pd.DataFrame) -> pd.DataFrame:
    home = df[["gameId", "season"]].assign(abs_err=(df["pred_home"] - df["actual_home"]).abs())
    away = df[["gameId", "season"]].assign(abs_err=(df["pred_away"] - df["actual_away"]).abs())
    home["gameId"] = home["gameId"].astype(str) + "_home"
    away["gameId"] = away["gameId"].astype(str) + "_away"
    return pd.concat([home, away], ignore_index=True)


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: team_stat_rates cross_season_weight="
          f"{CANDIDATE_CROSS_SEASON_WEIGHT} vs current (cross_season_weight=0.0) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    current = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)

    verdicts = {}
    for label in ADOPTED_CATEGORIES:
        cand_full = _fit_and_predict(log, label, CANDIDATE_CROSS_SEASON_WEIGHT)
        current_full = _fit_and_predict(log, label, 0.0)

        cand_rows = _per_side(cand_full)
        cand_dev, cand_holdout = split_dev_holdout(cand_rows)
        current_rows = _per_side(current_full)
        _, current_holdout = split_dev_holdout(current_rows)

        print(f"--- {label}: candidate vs current config, holdout games only ---", flush=True)
        result = bootstrap_compare(
            cand_holdout, current_holdout, game_id_col="gameId",
            metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
            label_a=f"{label}_csw{CANDIDATE_CROSS_SEASON_WEIGHT}", label_b=f"{label}_current",
        )
        verdicts[label] = result["mae"]["verdict"]

        metrics_ledger.append_generic_run(
            metrics={"n_holdout": len(cand_holdout), "mae_delta": result["mae"]["delta"], "mae_verdict": result["mae"]["verdict"]},
            model_name=f"team_stat_{label}_cross_season_holdout",
            config_flags={"cross_season_weight": CANDIDATE_CROSS_SEASON_WEIGHT, "dev_max_season": DEV_MAX_SEASON},
            notes=f"One-time confirmatory holdout check: {label} cross_season_weight=0.25 vs current config, holdout only",
        )

    print("\n=== SUMMARY (holdout-only, new config vs current config, THE decision-relevant comparison) ===", flush=True)
    for label, verdict in verdicts.items():
        print(f"  {label}: {verdict}", flush=True)

    any_regression = any("REAL REGRESSION" in v for v in verdicts.values())
    any_improvement = any("REAL IMPROVEMENT" in v for v in verdicts.values())
    if not any_regression and any_improvement:
        print("\n-> NET POSITIVE on real holdout: no regression on any category, real improvement on "
              "at least one. ADOPT for all 5.", flush=True)
    elif not any_regression:
        print("\n-> NEUTRAL on real holdout: no regression, but no clear real improvement either.", flush=True)
    else:
        print("\n-> REAL REGRESSION on at least one category on real holdout -- do not adopt uniformly; "
              "consider per-category adoption instead.", flush=True)
