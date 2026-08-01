"""The one-time confirmatory holdout read for `validate_shrinkage_strength_fix.py`'s
winning candidate (prior_games_rating=10.0, chosen via dev-only Stage 1 +
Stage 2, both REAL IMPROVEMENT on margin_mae). This is a genuinely NEW
configuration never holdout-tested before -- eligible for its own one-time
read per the confirmatory-veto protocol.

Two things checked, since dev showed a real tradeoff (margin_mae improves,
total_mae regresses) that must be resolved on real holdout data, not
assumed to transfer:
1. The dev-vs-holdout GAP for the new config (does IT show the same kind
   of real regression the current config does?).
2. A direct ARM-vs-ARM comparison against the CURRENT prior_games_rating=15
   config, evaluated on HOLDOUT ONLY -- the actually decision-relevant
   question ("does the new config predict holdout games better, in
   absolute terms, than what's already deployed?"), mirroring how Task
   #24's OREB adoption decision was made (dev/holdout gap alone isn't the
   full picture; net absolute holdout performance is what matters for a
   real adoption decision).

Run as `python -m src.models.run_shrinkage_strength_holdout_check` -- ONLY
ONCE. Re-running after seeing a result and adjusting anything defeats the
point.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.team_strength import PRIOR_GAMES_RATING
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check
from src.models.validate_shrinkage_strength_fix import _arms, _full_range_fit as _dev_only_fit

CANDIDATE_PRIOR_GAMES_RATING = 10.0


def _full_dev_and_holdout_fit(prior_games_rating: float) -> pd.DataFrame:
    """Same as validate_shrinkage_strength_fix._full_range_fit but spanning
    the FULL range including holdout seasons -- only ever called here, in
    the one-time confirmatory script, never in the dev-only sweep."""
    from src.models.home_court import fit_home_court_walk_forward
    from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
    from src.models.validate_shrinkage_strength_fix import _to_wide_games, build_predictions

    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    log = add_team_ratings(log, prior_games_rating=prior_games_rating)
    games = _to_wide_games(log)
    df = build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: prior_games_rating={CANDIDATE_PRIOR_GAMES_RATING} "
          f"vs current={PRIOR_GAMES_RATING} ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    print("Building full dev+holdout predictions for BOTH configs...", flush=True)
    candidate_full = _full_dev_and_holdout_fit(CANDIDATE_PRIOR_GAMES_RATING)
    current_full = _full_dev_and_holdout_fit(PRIOR_GAMES_RATING)

    # --- Check 1: does the NEW config show its own real dev-vs-holdout gap? ---
    print("\n--- Check 1: new config's own dev-vs-holdout gap ---", flush=True)
    cand_arm = _arms(candidate_full).merge(candidate_full[["gameId", "season"]], on="gameId")
    dev_arm, holdout_arm = split_dev_holdout(cand_arm)
    gap_results = {}
    for metric, higher_is_better in [("abs_total_err", False), ("abs_margin_err", False), ("su", True)]:
        gap_results[metric] = generic_holdout_confirmatory_check(
            dev_arm[metric].to_numpy(), holdout_arm[metric].to_numpy(),
            metric_name=metric, higher_is_better=higher_is_better)

    # --- Check 2: direct arm-vs-arm comparison on HOLDOUT ONLY ---
    print("\n--- Check 2: new config vs CURRENT config, holdout games only ---", flush=True)
    cand_holdout = candidate_full[candidate_full["season"] >= DEV_MAX_SEASON].reset_index(drop=True)
    current_holdout = current_full[current_full["season"] >= DEV_MAX_SEASON].reset_index(drop=True)
    holdout_result = bootstrap_compare(
        _arms(cand_holdout), _arms(current_holdout), game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a=f"prior_games_rating={CANDIDATE_PRIOR_GAMES_RATING}", label_b=f"current_prior{PRIOR_GAMES_RATING}",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(cand_holdout),
                 "total_mae_delta": holdout_result["total_mae"]["delta"], "total_mae_verdict": holdout_result["total_mae"]["verdict"],
                 "margin_mae_delta": holdout_result["margin_mae"]["delta"], "margin_mae_verdict": holdout_result["margin_mae"]["verdict"],
                 "su_delta": holdout_result["su"]["delta"], "su_verdict": holdout_result["su"]["verdict"]},
        model_name="phase1_prior_games_rating_10_vs_15",
        config_flags={"candidate_prior_games_rating": CANDIDATE_PRIOR_GAMES_RATING, "current_prior_games_rating": PRIOR_GAMES_RATING},
        notes="One-time confirmatory holdout-only comparison: reduced shrinkage strength vs current Phase 1 config",
    )

    print("\n=== SUMMARY ===", flush=True)
    print("Holdout-only, new config vs current config (this is THE decision-relevant comparison):", flush=True)
    for name in ("total_mae", "margin_mae", "su"):
        print(f"  {name}: {holdout_result[name]['verdict']}", flush=True)

    margin_better = "REAL IMPROVEMENT" in holdout_result["margin_mae"]["verdict"]
    total_worse = "REAL REGRESSION" in holdout_result["total_mae"]["verdict"]
    if margin_better and not total_worse:
        print("\n-> NET POSITIVE on real holdout: margin_mae genuinely improves, total_mae doesn't regress. ADOPT.", flush=True)
    elif margin_better and total_worse:
        print("\n-> REAL TRADEOFF on real holdout: margin_mae improves but total_mae regresses. "
              "NOT a clean win -- do not adopt without an explicit decision on which metric matters more.", flush=True)
    else:
        print("\n-> Does not clearly improve margin_mae on real holdout -- NOT adopted.", flush=True)
