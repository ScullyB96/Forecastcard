"""Task #24's acceptance gate: does the team_stat_rates.py walk-forward
for/against combine (mirroring Phase 1's pace x rating architecture) beat
the literal naive floor -- a team's own trailing average of that stat, with
no opponent adjustment -- for each of OREB/DREB/AST/TOV/STL/BLK?

Dev-only (season < final_holdout_check.DEV_MAX_SEASON); holdout is never
read here, per the confirmatory-veto protocol every phase in this project
follows. Family choice (EWMA vs expanding-shrinkage) is NOT assumed to
transfer across categories or from OFF/DEF_RATING -- `add_walk_forward_rate`
is the one family tested here since it's the direct team-level analog of
what Phase 1 already validated; a category that fails this gate gets its
own follow-up investigation before being adopted in a different form,
exactly like OREB/STL/BLK's matchup-difficulty gap was handled.

Run as `python -m src.models.validate_team_stat_rates`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_stat_rates import PRIOR_GAMES, STAT_COLUMNS, add_team_stat_ratings, build_team_stat_game_log, project_team_stat, to_wide_games

NAIVE_PRIOR_GAMES = 5.0  # same weak, untuned floor convention as Phase 1's own naive arm


def _rated_dev_log() -> pd.DataFrame:
    log = build_team_stat_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    if log.empty:
        raise RuntimeError("no dev-range data cached yet -- run the Phase 0 ingest backfill first")
    log = add_team_stat_ratings(log)
    for label in STAT_COLUMNS:
        log = add_walk_forward_mean(log, f"{label}_for", NAIVE_PRIOR_GAMES, prefix=f"{label}_naive")
    return log


def build_dev_predictions(games: pd.DataFrame, label: str) -> pd.DataFrame:
    league_avg_col = f"{label}_league_avg_home"  # add_walk_forward_rate names its league-avg column this way
    rows = []
    for row in games.itertuples(index=False):
        league_avg = getattr(row, league_avg_col)
        home_pred, away_pred = project_team_stat(
            home_attack=getattr(row, f"{label}_attack_rate_home"), home_defense=getattr(row, f"{label}_defense_rate_home"),
            away_attack=getattr(row, f"{label}_attack_rate_away"), away_defense=getattr(row, f"{label}_defense_rate_away"),
            league_avg=league_avg,
        )
        rows.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": getattr(row, f"{label}_for_home"), "actual_away": getattr(row, f"{label}_for_away"),
            "pred_home": home_pred, "pred_away": away_pred,
            "naive_home": getattr(row, f"{label}_naive_shrunk_mean_home"),
            "naive_away": getattr(row, f"{label}_naive_shrunk_mean_away"),
        })
    return pd.DataFrame(rows)


def _per_side_arms(df: pd.DataFrame, pred_col_home: str, pred_col_away: str) -> pd.DataFrame:
    """Long format, one row per team-side (gameId_team key), matching this
    project's `f"{gameId}_{playerId}"` composite-key convention for
    non-game-shaped validation (see props' validate_matchup_difficulty.py)."""
    home = pd.DataFrame({
        "key": df["gameId"].astype(str) + "_home",
        "abs_err": (df[pred_col_home] - df["actual_home"]).abs(),
    })
    away = pd.DataFrame({
        "key": df["gameId"].astype(str) + "_away",
        "abs_err": (df[pred_col_away] - df["actual_away"]).abs(),
    })
    return pd.concat([home, away], ignore_index=True)


if __name__ == "__main__":
    dev_log = _rated_dev_log()
    games = to_wide_games(dev_log)

    for label in STAT_COLUMNS:
        df = build_dev_predictions(games, label)
        n_before = len(df)
        df = df.dropna(subset=["pred_home", "pred_away", "naive_home", "naive_away"]).reset_index(drop=True)
        dropped = n_before - len(df)

        model_arm = _per_side_arms(df, "pred_home", "pred_away")
        naive_arm = _per_side_arms(df, "naive_home", "naive_away")

        print(f"\n=== {label.upper()} ({len(df)} games, dropped {dropped} with no prior history) ===", flush=True)
        result = bootstrap_compare(
            model_arm, naive_arm, game_id_col="key",
            metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}],
            label_a=f"{label}_team_rate", label_b="naive_own_avg",
        )

        metrics_ledger.append_generic_run(
            {"n_games": len(df), f"{label}_mae": result["mae"]["a"]},
            model_name=f"team_stat_{label}",
            config_flags={"prior_games": PRIOR_GAMES[label], "dev_max_season": DEV_MAX_SEASON},
            notes=f"Task #24 team-level {label} vs naive own-average floor",
        )
        metrics_ledger.append_generic_run(
            {"n_games": len(df), f"{label}_mae": result["mae"]["b"]},
            model_name=f"team_stat_{label}_naive",
            config_flags={"prior_games": NAIVE_PRIOR_GAMES, "dev_max_season": DEV_MAX_SEASON},
            notes=f"Task #24 naive floor for {label}",
        )

    print("\nall 6 categories' arms appended to metrics_ledger.py", flush=True)
