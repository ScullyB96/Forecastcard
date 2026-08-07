"""The one-time confirmatory holdout read for the referee-crew tendency
correction (task #64, MODEL_DOCUMENTATION.md Sec59): does officiating
crew tendency (trailing average of total FTA in games each official has
worked, `referee_rates.add_referee_tendency`) carry real signal for the
game TOTAL beyond team_strength's baseline?

Chosen via dev-only Stage 1 (recent-dev slice sweep of K=0.5/1.0/1.5/2.0/3.0):
only K=0.5 cleared (REAL IMPROVEMENT, total_mae -0.0087); higher K values
overcorrect into NOISE. Stage 2 (full dev range, n=10,727) confirmed K=0.5
beats the current config (total_mae -0.0056, CI entirely negative).
Sanity-swept `prior_games` (25/50/100/200/400) at K=0.5 on the recent-dev
slice: 100/200/400 all show REAL IMPROVEMENT with a smooth, monotonically
shrinking effect size as shrinkage strength increases -- the pattern of a
genuine signal, not a lucky pick. prior_games=100 (the first value tested)
chosen for this one-time read.

The correction is applied ONLY to the game TOTAL (split evenly, preserving
the margin/home-away split entirely unchanged) -- so margin_mae/su are
unaffected by construction; only total_mae is a meaningful comparison
here.

Run as `python -m src.models.run_referee_tendency_holdout_check` -- ONLY ONCE.
"""

import glob
import re

import pandas as pd

from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON, split_dev_holdout
from src.models.referee_rates import add_referee_tendency, build_official_game_log
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check
from src.models.validate_team_strength_baseline import build_dev_predictions
from src.utils.paths import DATA_RAW

CANDIDATE_K = 0.5
PRIOR_GAMES = 100.0


def _load_referee_crew_deviation() -> pd.DataFrame:
    officials_paths = sorted(glob.glob(str(DATA_RAW / "officials_*.parquet")))
    officials = pd.concat([pd.read_parquet(p) for p in officials_paths], ignore_index=True)
    seasons_cached = [re.search(r"officials_(\d{4}-\d{2})", p).group(1) for p in officials_paths]

    box = pd.concat([pd.read_parquet(DATA_RAW / f"boxscore_trad_player_{s}.parquet") for s in seasons_cached
                      if (DATA_RAW / f"boxscore_trad_player_{s}.parquet").exists()], ignore_index=True)
    schedule = pd.concat([pd.read_parquet(DATA_RAW / f"schedule_{s}.parquet") for s in seasons_cached
                           if (DATA_RAW / f"schedule_{s}.parquet").exists()], ignore_index=True)
    schedule = schedule[schedule["seasonType"] == "Regular Season"]

    log = build_official_game_log(officials, box, schedule)
    log = add_referee_tendency(log, "total_fta", prior_games=PRIOR_GAMES)
    crew = log.groupby("gameId", as_index=False).agg(
        crew_tendency=("total_fta_official_tendency", "mean"),
        crew_league_avg=("total_fta_league_avg", "mean"))
    crew["deviation"] = crew["crew_tendency"] - crew["crew_league_avg"]
    return crew[["gameId", "deviation"]]


def _build_predictions() -> pd.DataFrame:
    """Note: `build_dev_predictions` is dev-range-named but this call site
    needs the FULL dev+holdout range for a genuine one-time holdout read --
    `_rated_dev_log`'s own underlying `build_team_game_log` call is
    parameterized by `final_holdout_check.DEV_MAX_SEASON - 1` internally,
    so this DOES stop short of holdout by construction. Mirrors every
    other one-time holdout check in this project's own pattern of building
    a SEPARATE full-range fetch rather than reusing the dev-only helper --
    see `run_own_halflife_holdout_check.py`'s own `_full_dev_and_holdout_fit`."""
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.home_court import fit_home_court_walk_forward
    from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
    from src.models.validate_team_strength_baseline import _to_wide_games

    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    log = add_team_ratings(log)
    games = _to_wide_games(log)
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)

    preds = []
    for row in games.itertuples(index=False):
        _, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        preds.append({"gameId": row.gameId, "season": row.season,
                       "actual_total": row.actual_home_score + row.actual_away_score,
                       "pred_total": pred_home + pred_away})
    return pd.DataFrame(preds).dropna(subset=["pred_total"]).reset_index(drop=True)


def _arms(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    d = df[["gameId", "season"]].copy()
    d["abs_total_err"] = (df[pred_col] - df["actual_total"]).abs()
    return d


if __name__ == "__main__":
    print(f"=== One-time confirmatory holdout read: referee-crew tendency correction "
          f"(K={CANDIDATE_K}, prior_games={PRIOR_GAMES}) vs current (no correction) ===", flush=True)
    print("THIS IS A ONE-TIME READ.\n", flush=True)

    print("Building full dev+holdout predictions + referee crew deviation...", flush=True)
    preds = _build_predictions()
    crew_dev = _load_referee_crew_deviation()
    full = preds.merge(crew_dev, on="gameId", how="inner").dropna(subset=["deviation"]).reset_index(drop=True)
    full["pred_total_adj"] = full["pred_total"] + CANDIDATE_K * full["deviation"]
    print(f"n={len(full)} games with both a prediction and a real crew tendency", flush=True)

    dev, holdout = split_dev_holdout(full)

    print("\n--- Check 1: candidate's own dev-vs-holdout gap ---", flush=True)
    dev_err = (dev["pred_total_adj"] - dev["actual_total"]).abs().to_numpy()
    holdout_err = (holdout["pred_total_adj"] - holdout["actual_total"]).abs().to_numpy()
    generic_holdout_confirmatory_check(dev_err, holdout_err, metric_name="abs_total_err", higher_is_better=False)

    print("\n--- Check 2: candidate vs CURRENT config, holdout games only (THE decision-relevant comparison) ---", flush=True)
    holdout_result = bootstrap_compare(
        _arms(holdout, "pred_total_adj"), _arms(holdout, "pred_total"), game_id_col="gameId",
        metrics=[{"name": "total_mae", "col": "abs_total_err", "higher_is_better": False}],
        label_a="referee_adjusted", label_b="current_config",
    )

    metrics_ledger.append_generic_run(
        metrics={"n_holdout": len(holdout),
                 "total_mae_delta": holdout_result["total_mae"]["delta"], "total_mae_verdict": holdout_result["total_mae"]["verdict"]},
        model_name="phase1_referee_tendency_vs_current",
        config_flags={"K": CANDIDATE_K, "prior_games": PRIOR_GAMES},
        notes="One-time confirmatory holdout-only comparison: referee-crew FTA-tendency total correction vs current config",
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"Holdout-only, referee-adjusted vs current config: {holdout_result['total_mae']['verdict']}", flush=True)
    if "REAL IMPROVEMENT" in holdout_result["total_mae"]["verdict"]:
        print("\n-> NET POSITIVE on real holdout. ADOPT.", flush=True)
    elif "REAL REGRESSION" in holdout_result["total_mae"]["verdict"]:
        print("\n-> REAL REGRESSION on real holdout -- NOT adopted.", flush=True)
    else:
        print("\n-> NEUTRAL on real holdout (noise) -- NOT adopted (no real out-of-sample gain confirmed).", flush=True)
