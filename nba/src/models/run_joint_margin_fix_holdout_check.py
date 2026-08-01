"""The one-time confirmatory holdout read for `validate_joint_margin_fix.py`'s
winning candidate (prior_games_rating=12.0, league_avg_halflife_games=2000.0),
chosen via dev-only Stage 1 (recent-slice grid sweep) + Stage 2 (full dev
range), both showing a genuine NET WIN on total_mae AND margin_mae
together -- the first configuration in this whole investigation (Sec24,
Sec26) to pass both metrics simultaneously with no tradeoff. This is a
genuinely new configuration never holdout-tested before, eligible for its
own one-time read per the confirmatory-veto protocol.

Run as `python -m src.models.run_joint_margin_fix_holdout_check` -- ONLY
ONCE.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.home_court import fit_home_court_walk_forward
from src.models.team_strength import PRIOR_GAMES_RATING, add_team_ratings, build_team_game_log, project_game
from src.models.validate_joint_margin_fix import _to_wide_games, build_predictions

CANDIDATE_PRIOR_GAMES_RATING = 12.0
CANDIDATE_HALFLIFE = 2000.0


def _full_dev_and_holdout_fit(prior_games_rating: float, halflife: float | None) -> pd.DataFrame:
    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    log = add_team_ratings(log, prior_games_rating=prior_games_rating, league_avg_halflife_games=halflife)
    games = _to_wide_games(log)
    df = build_predictions(games)
    return df.dropna(subset=["pred_home", "pred_away"]).reset_index(drop=True)


def _arms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_total_err"] = ((df["pred_home"] + df["pred_away"]) - (df["actual_home"] + df["actual_away"])).abs()
    df["abs_margin_err"] = ((df["pred_home"] - df["pred_away"]) - (df["actual_home"] - df["actual_away"])).abs()
    df["su"] = ((df["pred_home"] > df["pred_away"]).astype(int) == (df["actual_home"] > df["actual_away"]).astype(int)).astype(float)
    return df[["gameId", "season", "abs_total_err", "abs_margin_err", "su"]]


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: joint fix "
          f"(prior_games_rating={CANDIDATE_PRIOR_GAMES_RATING}, halflife={CANDIDATE_HALFLIFE}) "
          f"vs current (prior_games_rating={PRIOR_GAMES_RATING}, flat) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    print("Building full dev+holdout predictions for BOTH configs...", flush=True)
    candidate_full = _full_dev_and_holdout_fit(CANDIDATE_PRIOR_GAMES_RATING, CANDIDATE_HALFLIFE)
    current_full = _full_dev_and_holdout_fit(PRIOR_GAMES_RATING, None)

    cand_arm = _arms(candidate_full)
    cand_dev, cand_holdout = split_dev_holdout(cand_arm)
    current_arm = _arms(current_full)
    current_dev, current_holdout = split_dev_holdout(current_arm)

    print("\n--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check
    gap_results = {}
    for metric, higher_is_better in [("abs_total_err", False), ("abs_margin_err", False), ("su", True)]:
        gap_results[metric] = generic_holdout_confirmatory_check(
            cand_dev[metric].to_numpy(), cand_holdout[metric].to_numpy(), metric_name=metric, higher_is_better=higher_is_better)

    print("\n--- Check 2: candidate vs CURRENT config, holdout games only (THE decision-relevant comparison) ---", flush=True)
    holdout_result = bootstrap_compare(
        cand_holdout, current_holdout, game_id_col="gameId",
        metrics=[
            {"name": "total_mae", "col": "abs_total_err", "higher_is_better": False},
            {"name": "margin_mae", "col": "abs_margin_err", "higher_is_better": False},
            {"name": "su", "col": "su", "higher_is_better": True},
        ],
        label_a=f"joint_fix_{CANDIDATE_PRIOR_GAMES_RATING}_{CANDIDATE_HALFLIFE}", label_b="current_config",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(cand_holdout),
                 "total_mae_delta": holdout_result["total_mae"]["delta"], "total_mae_verdict": holdout_result["total_mae"]["verdict"],
                 "margin_mae_delta": holdout_result["margin_mae"]["delta"], "margin_mae_verdict": holdout_result["margin_mae"]["verdict"],
                 "su_delta": holdout_result["su"]["delta"], "su_verdict": holdout_result["su"]["verdict"]},
        model_name="phase1_joint_margin_fix_vs_current",
        config_flags={"prior_games_rating": CANDIDATE_PRIOR_GAMES_RATING, "league_avg_halflife_games": CANDIDATE_HALFLIFE},
        notes="One-time confirmatory holdout-only comparison: joint recency+shrinkage fix vs current Phase 1 config",
    )

    print("\n=== SUMMARY ===", flush=True)
    print("Holdout-only, new config vs current config (THE decision-relevant comparison):", flush=True)
    for name in ("total_mae", "margin_mae", "su"):
        print(f"  {name}: {holdout_result[name]['verdict']}", flush=True)

    total_ok = "REAL REGRESSION" not in holdout_result["total_mae"]["verdict"]
    margin_ok = "REAL REGRESSION" not in holdout_result["margin_mae"]["verdict"]
    any_real_gain = "REAL IMPROVEMENT" in holdout_result["total_mae"]["verdict"] or "REAL IMPROVEMENT" in holdout_result["margin_mae"]["verdict"]
    if total_ok and margin_ok and any_real_gain:
        print("\n-> NET POSITIVE on real holdout: no regression on either metric, real improvement on "
              "at least one. ADOPT.", flush=True)
    elif total_ok and margin_ok:
        print("\n-> NEUTRAL on real holdout: no regression, but no clear real improvement either -- "
              "NOISE on both. Marginal case, lean toward NOT adopting extra complexity for no proven gain.", flush=True)
    else:
        print("\n-> REAL REGRESSION on at least one metric on real holdout -- NOT adopted.", flush=True)
