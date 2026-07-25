"""Acceptance gate for `player_minutes.py`: does the recency-weighted
(EWMA) walk-forward minutes projection beat the literal naive floor -- a
player's own unshrunk trailing SEASON average (infinite memory, no
recency-weighting at all)? Same discipline as
`validate_team_strength_baseline.py`.

Note this project's own history on this specific model: the FIRST version
used an infinite-memory expanding-shrinkage approach (mirroring
`shrinkage.add_walk_forward_rate`'s team-level pattern) and was found, via
this exact validation script, to be a REAL REGRESSION against a naive
floor -- confirmed via bootstrap, not assumed. That result is why
`player_minutes.add_minutes_rating` now uses
`player_rate_shrinkage.add_walk_forward_player_mean_ewm` instead. See that
function's docstring for the full comparison (expanding-shrinkage variants,
rolling windows, EWMA halflives) that led to the halflife=2 choice.

Run as `python -m src.models.validate_player_minutes`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_minutes import MINUTES_HALFLIFE_GAMES, add_minutes_rating, build_player_game_log

NAIVE_PRIOR_GAMES = 1000.0  # effectively infinite memory -- the literal "trailing season average,
# no recency weighting at all" floor, built via the (deliberately unshrunk, huge-prior) rate primitive.


def build_dev_predictions() -> pd.DataFrame:
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    if log.empty:
        raise RuntimeError("no dev-range player-game data cached yet")
    rated = add_minutes_rating(log)

    from src.models.player_rate_shrinkage import add_walk_forward_player_rate
    naive_log = log.copy()
    naive_log["_games_exposure"] = 1.0
    naive = add_walk_forward_player_rate(naive_log, "minutes", "_games_exposure", NAIVE_PRIOR_GAMES, prefix="naive_min")

    out = rated[["gameId", "season", "playerId", "minutes", "minutes_ewm_rate"]].merge(
        naive[["gameId", "playerId", "naive_min_shrunk_rate"]], on=["gameId", "playerId"], how="inner")
    return out.dropna(subset=["minutes_ewm_rate", "naive_min_shrunk_rate"]).reset_index(drop=True)


if __name__ == "__main__":
    df = build_dev_predictions()
    print(f"built {len(df)} dev-range (player, game) minutes predictions", flush=True)

    df["abs_err_model"] = (df["minutes_ewm_rate"] - df["minutes"]).abs()
    df["abs_err_naive"] = (df["naive_min_shrunk_rate"] - df["minutes"]).abs()
    df["game_player_key"] = df["gameId"].astype(str) + "_" + df["playerId"].astype(str)

    arm_model = df[["game_player_key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"})
    arm_naive = df[["game_player_key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"})

    result = bootstrap_compare(
        arm_model, arm_naive, game_id_col="game_player_key",
        metrics=[{"name": "minutes_mae", "col": "abs_err", "higher_is_better": False}],
        label_a=f"ewm_halflife_{MINUTES_HALFLIFE_GAMES}", label_b="naive_season_avg",
    )

    mae = result["minutes_mae"]
    metrics_ledger.append_generic_run(
        metrics={
            "n_player_games": len(df), "minutes_mae_model": mae["a"], "minutes_mae_naive": mae["b"],
            "minutes_mae_delta": mae["delta"], "minutes_mae_ci_lo": mae["delta_ci95"][0],
            "minutes_mae_ci_hi": mae["delta_ci95"][1], "minutes_mae_verdict": mae["verdict"],
        },
        model_name="player_minutes_ewm",
        config_flags={"halflife_games": MINUTES_HALFLIFE_GAMES, "dev_max_season": DEV_MAX_SEASON},
        notes="Player minutes EWMA (halflife=2) vs naive season-average floor",
    )
    print("\nappended to metrics_ledger.py", flush=True)
