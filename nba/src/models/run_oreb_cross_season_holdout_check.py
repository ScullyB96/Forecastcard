"""The one-time confirmatory holdout read for team-level OREB's
`cross_season_weight` candidate (full-model-audit research pass,
2026-08-01) -- a genuinely new, third mechanism for OREB never tried
before: Sec26 tried reduced shrinkage strength (failed, made things
worse) and an adaptive league-average target (real progress, moved OREB
from a clear holdout loss to naive to a statistical tie, but insufficient
to adopt). `cross_season_weight` blends a team's own immediately-
preceding season OREB rate into its early-season prior instead of
resetting fully to the league average -- the same lever that just gave
`team_strength.add_team_ratings` two consecutive real wins (Sec29, Sec33)
via the identical `shrinkage.add_walk_forward_rate` primitive.

Chosen via dev-only Stage 1 (recent-dev-slice sweep of 0.1-1.0, best at
0.25) + Stage 2 (full dev range, beats BOTH the current csw=0.0 config AND
naive). THE decision-relevant question, per Sec23's own precedent (OREB
was excluded from ADOPTED_CATEGORIES because the model genuinely LOST to
naive on holdout, in absolute terms): does this candidate actually beat
naive on real holdout data.

Run as `python -m src.models.run_oreb_cross_season_holdout_check` -- ONLY
ONCE.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.shrinkage import add_walk_forward_mean, add_walk_forward_rate
from src.models.team_stat_rates import PRIOR_GAMES, build_team_stat_game_log, project_team_stat, to_wide_games
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check

LABEL = "oreb"
CANDIDATE_CROSS_SEASON_WEIGHT = 0.25
NAIVE_PRIOR_GAMES = 5.0


def _fit_and_predict(log: pd.DataFrame, cross_season_weight: float) -> pd.DataFrame:
    l = add_walk_forward_rate(log.copy(), f"{LABEL}_for", f"{LABEL}_against", PRIOR_GAMES[LABEL],
                               prefix=LABEL, cross_season_weight=cross_season_weight)
    l = add_walk_forward_mean(l, f"{LABEL}_for", NAIVE_PRIOR_GAMES, prefix=f"{LABEL}_naive")
    games = to_wide_games(l)
    rows = []
    for row in games.itertuples(index=False):
        home_pred, away_pred = project_team_stat(
            home_attack=getattr(row, f"{LABEL}_attack_rate_home"), home_defense=getattr(row, f"{LABEL}_defense_rate_home"),
            away_attack=getattr(row, f"{LABEL}_attack_rate_away"), away_defense=getattr(row, f"{LABEL}_defense_rate_away"),
            league_avg=getattr(row, f"{LABEL}_league_avg_home"),
        )
        rows.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": getattr(row, f"{LABEL}_for_home"), "actual_away": getattr(row, f"{LABEL}_for_away"),
            "pred_home": home_pred, "pred_away": away_pred,
            "naive_home": getattr(row, f"{LABEL}_naive_shrunk_mean_home"), "naive_away": getattr(row, f"{LABEL}_naive_shrunk_mean_away"),
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def _per_side(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    home = df[["gameId", "season"]].assign(abs_err=(df[f"{pred_col}_home"] - df["actual_home"]).abs())
    away = df[["gameId", "season"]].assign(abs_err=(df[f"{pred_col}_away"] - df["actual_away"]).abs())
    home["gameId"] = home["gameId"].astype(str) + "_home"
    away["gameId"] = away["gameId"].astype(str) + "_away"
    return pd.concat([home, away], ignore_index=True)


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: OREB cross_season_weight={CANDIDATE_CROSS_SEASON_WEIGHT} ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    current = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)
    full = _fit_and_predict(log, CANDIDATE_CROSS_SEASON_WEIGHT)
    dev, holdout = split_dev_holdout(full)

    print("--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    cand_rows = _per_side(full, "pred")
    cand_dev, cand_holdout = split_dev_holdout(cand_rows)
    gap_result = generic_holdout_confirmatory_check(
        cand_dev["abs_err"].to_numpy(), cand_holdout["abs_err"].to_numpy(), metric_name="oreb_mae", higher_is_better=False)

    print("\n--- Check 2: candidate vs naive floor, holdout games only (THE decision-relevant comparison) ---", flush=True)
    naive_rows = _per_side(full, "naive")
    naive_holdout = split_dev_holdout(naive_rows)[1]
    result = bootstrap_compare(
        cand_holdout, naive_holdout, game_id_col="gameId",
        metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
        label_a=f"oreb_cross_season_{CANDIDATE_CROSS_SEASON_WEIGHT}", label_b="naive_own_avg",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(cand_holdout), "oreb_mae_dev": gap_result["dev"], "oreb_mae_holdout": gap_result["holdout"],
                 "oreb_mae_gap_verdict": gap_result["verdict"], "vs_naive_delta": result["mae"]["delta"],
                 "vs_naive_verdict": result["mae"]["verdict"]},
        model_name="team_stat_oreb_cross_season_holdout", config_flags={"cross_season_weight": CANDIDATE_CROSS_SEASON_WEIGHT, "dev_max_season": DEV_MAX_SEASON},
        notes="One-time confirmatory holdout check: OREB cross_season_weight candidate vs naive floor, holdout only",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"  own dev-vs-holdout gap: {gap_result['verdict']}", flush=True)
    print(f"  vs naive on holdout: {result['mae']['verdict']}", flush=True)
    if "REAL IMPROVEMENT" in result["mae"]["verdict"]:
        print("\n-> The candidate BEATS naive on real holdout data -- ADOPT for OREB team-level anchoring.", flush=True)
    elif "NOISE" in result["mae"]["verdict"]:
        print("\n-> Statistically indistinguishable from naive on holdout -- NOT a clean win, do not adopt "
              "(OREB stays unanchored).", flush=True)
    else:
        print("\n-> STILL loses to naive on real holdout data -- NOT adopted, OREB stays unanchored.", flush=True)
