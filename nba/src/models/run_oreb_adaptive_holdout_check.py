"""The one-time confirmatory holdout read for team-level OREB's adaptive
(EWMA) league-average candidate -- a genuinely NEW configuration, never
holdout-tested before, chosen via `validate_shrinkage_strength_fix.py`-style
dev-only Stage 1 (recent-dev slice) + Stage 2 (full dev range), both REAL
IMPROVEMENT vs the current flat-cumulative OREB model (halflife=300 best
on the recent slice, -0.0077 delta; still real at full-dev-range, -0.0023).

Two other levers were tried and ruled out first, for context (not repeated
here): reducing OREB's shrinkage strength (`PRIOR_GAMES['oreb']`) made
things monotonically WORSE at every value tested (opposite of what helped
Phase 1's margin) -- decisively rules out the "over-shrinking" hypothesis
for OREB specifically. And OREB's holdout error was found to be roughly
uniform across early- vs late-season games, ruling out a stale-early-
season-prior mechanism. The adaptive league average is the one candidate
that actually showed real, consistent improvement across a wide dev-only
sweep -- worth its own one-time holdout read.

THE decision-relevant question (per Sec23's own precedent: OREB was
excluded from ADOPTED_CATEGORIES because the ORIGINAL model genuinely
LOST to naive on holdout, in absolute terms, not just a widening gap) is
whether THIS candidate actually beats naive on real holdout data -- not
just whether it beats the current (already-failing) OREB model.

Run as `python -m src.models.run_oreb_adaptive_holdout_check` -- ONLY ONCE.
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
CANDIDATE_HALFLIFE = 300.0
NAIVE_PRIOR_GAMES = 5.0  # matches validate_team_stat_rates.py's own naive-floor convention


def _fit_and_predict(log: pd.DataFrame, halflife: float | None) -> pd.DataFrame:
    l = add_walk_forward_rate(log.copy(), f"{LABEL}_for", f"{LABEL}_against", PRIOR_GAMES[LABEL],
                               prefix=LABEL, league_avg_halflife_games=halflife)
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


def _per_side(df: pd.DataFrame, pred_col: str, actual_col_home: str = "actual_home", actual_col_away: str = "actual_away") -> pd.DataFrame:
    home = df[["gameId", "season"]].assign(abs_err=(df[f"{pred_col}_home"] - df[actual_col_home]).abs())
    away = df[["gameId", "season"]].assign(abs_err=(df[f"{pred_col}_away"] - df[actual_col_away]).abs())
    home["gameId"] = home["gameId"].astype(str) + "_home"
    away["gameId"] = away["gameId"].astype(str) + "_away"
    return pd.concat([home, away], ignore_index=True)


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: OREB adaptive league avg (halflife={CANDIDATE_HALFLIFE}) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    current = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)
    full = _fit_and_predict(log, CANDIDATE_HALFLIFE)
    dev, holdout = split_dev_holdout(full)

    # --- Check 1: candidate's own dev-vs-holdout gap ---
    print("--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    cand_rows = _per_side(full, "pred")
    cand_dev, cand_holdout = split_dev_holdout(cand_rows)
    gap_result = generic_holdout_confirmatory_check(
        cand_dev["abs_err"].to_numpy(), cand_holdout["abs_err"].to_numpy(), metric_name="oreb_mae", higher_is_better=False)

    # --- Check 2: candidate vs naive, HOLDOUT ONLY (the actual bar OREB failed before) ---
    print("\n--- Check 2: candidate vs naive floor, holdout games only ---", flush=True)
    naive_rows = _per_side(full, "naive")
    cand_holdout_arm = cand_holdout
    naive_holdout_arm = split_dev_holdout(naive_rows)[1]
    result = bootstrap_compare(
        cand_holdout_arm, naive_holdout_arm, game_id_col="gameId",
        metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
        label_a=f"oreb_adaptive_hl{CANDIDATE_HALFLIFE}", label_b="naive_own_avg",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(cand_holdout_arm), "oreb_mae_dev": gap_result["dev"], "oreb_mae_holdout": gap_result["holdout"],
                 "oreb_mae_gap_verdict": gap_result["verdict"], "vs_naive_delta": result["mae"]["delta"],
                 "vs_naive_verdict": result["mae"]["verdict"]},
        model_name="team_stat_oreb_adaptive_holdout", config_flags={"halflife": CANDIDATE_HALFLIFE, "dev_max_season": DEV_MAX_SEASON},
        notes="One-time confirmatory holdout check: OREB adaptive league-avg candidate vs naive floor, holdout only",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"  own dev-vs-holdout gap: {gap_result['verdict']}", flush=True)
    print(f"  vs naive on holdout: {result['mae']['verdict']}", flush=True)
    if "REAL IMPROVEMENT" in result["mae"]["verdict"]:
        print("\n-> The adaptive candidate BEATS naive on real holdout data -- ADOPT for OREB team-level anchoring.", flush=True)
    elif "NOISE" in result["mae"]["verdict"]:
        print("\n-> Candidate is statistically indistinguishable from naive on holdout -- NOT a clean win, "
              "do not adopt (OREB stays unanchored).", flush=True)
    else:
        print("\n-> Candidate STILL loses to naive on real holdout data -- NOT adopted, OREB stays unanchored. "
              "Consistent with a genuine regime change no historical-extrapolation technique can fix in advance.", flush=True)
