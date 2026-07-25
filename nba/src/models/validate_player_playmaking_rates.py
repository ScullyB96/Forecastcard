"""Acceptance gate for `player_playmaking_rates.py`: does the shrunk
per-touch assist/turnover rate beat a naive floor (that player's own
almost-unshrunk trailing per-touch rate)? Same discipline as the other
`validate_player_*.py` scripts. Only runs on rows where `BoxScorePlayerTrackV3`
data is available (touches) -- rows from before the backfill reaches them,
or genuine API gaps, are dropped rather than silently zero-filled.

Run as `python -m src.models.validate_player_playmaking_rates`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_minutes import attach_player_track, build_player_game_log
from src.models.player_playmaking_rates import add_playmaking_rates
from src.models.player_rate_shrinkage import add_walk_forward_player_rate

NAIVE_PRIOR_TOUCHES = 1.0  # deliberately weak floor -- almost-unshrunk trailing own rate

_CATEGORIES = (
    ("ast", "assists"),
    ("tov", "turnovers"),
)


def run_category(log: pd.DataFrame, label: str, event_col: str) -> None:
    rated = add_playmaking_rates(log.copy())
    naive = add_walk_forward_player_rate(log.copy(), event_col, "touches", NAIVE_PRIOR_TOUCHES, prefix=f"naive_{label}")
    naive[f"naive_{label}_proj"] = naive[f"naive_{label}_shrunk_rate"] * naive["touches"]

    df = rated[["gameId", "playerId", event_col, f"{label}_proj"]].merge(
        naive[["gameId", "playerId", f"naive_{label}_proj"]], on=["gameId", "playerId"], how="inner")
    df = df.dropna(subset=[f"{label}_proj", f"naive_{label}_proj"])
    df["game_player_key"] = df["gameId"].astype(str) + "_" + df["playerId"].astype(str)
    df["abs_err_model"] = (df[f"{label}_proj"] - df[event_col]).abs()
    df["abs_err_naive"] = (df[f"naive_{label}_proj"] - df[event_col]).abs()

    arm_model = df[["game_player_key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"})
    arm_naive = df[["game_player_key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"})
    print(f"\n=== {label}: n={len(df)} ===", flush=True)
    result = bootstrap_compare(
        arm_model, arm_naive, game_id_col="game_player_key",
        metrics=[{"name": f"{label}_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result[f"{label}_mae"]
    metrics_ledger.append_generic_run(
        metrics={"n_player_games": len(df), f"{label}_mae_model": mae["a"],
                 f"{label}_mae_naive": mae["b"], f"{label}_mae_delta": mae["delta"], f"{label}_mae_verdict": mae["verdict"]},
        model_name=f"player_playmaking_{label}",
        config_flags={"dev_max_season": DEV_MAX_SEASON},
        notes=f"Player {label} per-touch rate vs weak naive floor",
    )


if __name__ == "__main__":
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = attach_player_track(log, FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    have_track = log["touches"].notna().sum()
    print(f"built {len(log)} dev-range player-game rows, {have_track} with player-track data", flush=True)
    for label, event_col in _CATEGORIES:
        run_category(log, label, event_col)
    print("\nall categories appended to metrics_ledger.py", flush=True)
