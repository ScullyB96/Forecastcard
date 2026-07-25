"""Acceptance gate for `player_scoring_rates.py`: does the shrunk
volume x efficiency projection beat a naive floor (that player's own
unshrunk trailing per-minute attempt rate x unshrunk trailing make rate)
for each of 2PT/3PT/FT makes? Same discipline as
`validate_team_strength_baseline.py`/`validate_player_minutes.py`.

Run as `python -m src.models.validate_player_scoring_rates`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_minutes import build_player_game_log
from src.models.player_rate_shrinkage import add_walk_forward_player_rate
from src.models.player_scoring_rates import add_scoring_rates

_CATEGORIES = (
    ("2pt", "fgAtt2", "fgMade2"),
    ("3pt", "fg3Att", "fg3Made"),
    ("ft", "ftAtt", "ftMade"),
)

NAIVE_PRIOR = 1.0  # deliberately weak floor -- almost-unshrunk trailing own rate


def run_category(log: pd.DataFrame, label: str, att_col: str, make_col: str) -> None:
    rated = add_scoring_rates(log.copy())

    naive = log.copy()
    naive = add_walk_forward_player_rate(naive, att_col, "minutes", NAIVE_PRIOR, prefix=f"naive_{label}_att")
    naive = add_walk_forward_player_rate(naive, make_col, att_col, NAIVE_PRIOR, prefix=f"naive_{label}_make")
    naive[f"naive_{label}_proj_made"] = (
        naive[f"naive_{label}_att_shrunk_rate"] * naive["minutes"] * naive[f"naive_{label}_make_shrunk_rate"])

    df = rated[["gameId", "playerId", make_col, f"{label}_proj_made"]].merge(
        naive[["gameId", "playerId", f"naive_{label}_proj_made"]], on=["gameId", "playerId"], how="inner")
    df = df.dropna(subset=[f"{label}_proj_made", f"naive_{label}_proj_made"])
    df["game_player_key"] = df["gameId"].astype(str) + "_" + df["playerId"].astype(str)
    df["abs_err_model"] = (df[f"{label}_proj_made"] - df[make_col]).abs()
    df["abs_err_naive"] = (df[f"naive_{label}_proj_made"] - df[make_col]).abs()

    arm_model = df[["game_player_key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"})
    arm_naive = df[["game_player_key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"})
    print(f"\n=== {label} makes: n={len(df)} ===", flush=True)
    result = bootstrap_compare(
        arm_model, arm_naive, game_id_col="game_player_key",
        metrics=[{"name": f"{label}_made_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result[f"{label}_made_mae"]
    metrics_ledger.append_generic_run(
        metrics={"n_player_games": len(df), f"{label}_made_mae_model": mae["a"],
                 f"{label}_made_mae_naive": mae["b"], f"{label}_made_mae_delta": mae["delta"],
                 f"{label}_made_mae_verdict": mae["verdict"]},
        model_name=f"player_scoring_{label}",
        config_flags={"dev_max_season": DEV_MAX_SEASON},
        notes=f"Player {label} makes: shrunk volume x efficiency vs weak naive floor",
    )


if __name__ == "__main__":
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    print(f"built {len(log)} dev-range player-game rows", flush=True)
    for label, att_col, make_col in _CATEGORIES:
        run_category(log, label, att_col, make_col)
    print("\nall categories appended to metrics_ledger.py", flush=True)
