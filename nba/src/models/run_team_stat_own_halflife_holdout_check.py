"""The one-time confirmatory holdout read for `own_halflife_games` applied
to all 6 team_stat_rates categories, including OREB -- the same lever that
just gave `team_strength.add_team_ratings` a fifth real win (Sec36) via the
identical `shrinkage.add_walk_forward_rate` primitive. Unlike
`cross_season_weight` (Sec34/35, where OREB specifically converged to a
rejected ceiling while the other 5 categories won), `own_halflife_games`
showed REAL IMPROVEMENT for OREB too on both dev stages -- a structurally
different lever (recency-weights the team's own history, not the
early-season prior), so OREB is deliberately tested here rather than assumed
to fail the same way.

Chosen via dev-only Stage 1 (recent-dev slice sweep of 20/40 games, on top of
each category's already-adopted cross_season_weight default): all 6
categories showed REAL IMPROVEMENT or NOISE (never regression) at both
values; 20 the stronger candidate for most. Stage 2 (full dev range) then
confirmed all 6 beat BOTH the current config AND the naive floor at
own_halflife_games=20.

Run as `python -m src.models.run_team_stat_own_halflife_holdout_check` --
ONLY ONCE.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.shrinkage import add_walk_forward_rate
from src.models.team_stat_rates import (DEFAULT_CROSS_SEASON_WEIGHTS, PRIOR_GAMES, STAT_COLUMNS,
                                         build_team_stat_game_log, project_team_stat, to_wide_games)
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check

CANDIDATE_OWN_HALFLIFE_GAMES = 20.0


def _fit_and_predict(log: pd.DataFrame, label: str, own_halflife_games: float | None) -> pd.DataFrame:
    l = add_walk_forward_rate(log.copy(), f"{label}_for", f"{label}_against", PRIOR_GAMES[label],
                               prefix=label, cross_season_weight=DEFAULT_CROSS_SEASON_WEIGHTS[label],
                               own_halflife_games=own_halflife_games)
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
    print(f"=== One-time confirmatory holdout read: team_stat_rates own_halflife_games="
          f"{CANDIDATE_OWN_HALFLIFE_GAMES} vs current (own_halflife_games=None) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    current = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)

    verdicts = {}
    for label in STAT_COLUMNS:
        cand_full = _fit_and_predict(log, label, CANDIDATE_OWN_HALFLIFE_GAMES)
        current_full = _fit_and_predict(log, label, None)

        cand_rows = _per_side(cand_full)
        cand_dev, cand_holdout = split_dev_holdout(cand_rows)
        current_rows = _per_side(current_full)
        _, current_holdout = split_dev_holdout(current_rows)

        print(f"--- {label}: candidate's own dev-vs-holdout gap ---", flush=True)
        gap_result = generic_holdout_confirmatory_check(
            cand_dev["abs_err"].to_numpy(), cand_holdout["abs_err"].to_numpy(), metric_name=f"{label}_mae", higher_is_better=False)

        print(f"--- {label}: candidate vs current config, holdout games only ---", flush=True)
        result = bootstrap_compare(
            cand_holdout, current_holdout, game_id_col="gameId",
            metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
            label_a=f"{label}_ownhl{CANDIDATE_OWN_HALFLIFE_GAMES}", label_b=f"{label}_current",
        )
        verdicts[label] = result["mae"]["verdict"]

        metrics_ledger.append_generic_run(
            metrics={"n_holdout": len(cand_holdout), "gap_verdict": gap_result["verdict"],
                     "mae_delta": result["mae"]["delta"], "mae_verdict": result["mae"]["verdict"]},
            model_name=f"team_stat_{label}_own_halflife_holdout",
            config_flags={"own_halflife_games": CANDIDATE_OWN_HALFLIFE_GAMES, "dev_max_season": DEV_MAX_SEASON},
            notes=f"One-time confirmatory holdout check: {label} own_halflife_games=20 vs current config, holdout only",
        )

    print("\n=== SUMMARY (holdout-only, new config vs current config, THE decision-relevant comparison) ===", flush=True)
    for label, verdict in verdicts.items():
        print(f"  {label}: {verdict}", flush=True)

    any_regression = any("REAL REGRESSION" in v for v in verdicts.values())
    any_improvement = any("REAL IMPROVEMENT" in v for v in verdicts.values())
    if not any_regression and any_improvement:
        print("\n-> NET POSITIVE on real holdout: no regression on any category, real improvement on "
              "at least one. ADOPT for all clearing categories.", flush=True)
    elif not any_regression:
        print("\n-> NEUTRAL on real holdout: no regression, but no clear real improvement either.", flush=True)
    else:
        print("\n-> REAL REGRESSION on at least one category on real holdout -- do not adopt uniformly; "
              "consider per-category adoption instead.", flush=True)
