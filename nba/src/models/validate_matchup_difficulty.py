"""Acceptance gate for `matchup_difficulty.py`: does each level of the
shrinkage hierarchy (defender-specific, position-group-vs-team) beat a
naive floor (that defender's/team-posgroup's own almost-unshrunk trailing
rate) at predicting points allowed per matchup-minute? Same discipline as
the other `validate_*.py` scripts. Runs on the 2017-2023 dev subset only
(`MATCHUP_DATA_START_SEASON` through `DEV_MAX_SEASON - 1`) -- the matchup
layer's own dev range, narrower than the six per-player rate models' full
2015-2023 range (see `matchup_difficulty.py`'s module docstring).

Run as `python -m src.models.validate_matchup_difficulty`.
"""

from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.matchup_difficulty import (
    MATCHUP_DATA_START_SEASON,
    add_defender_difficulty_rate,
    add_position_group_difficulty_rate,
    build_defender_game_log,
    build_position_group_matchup_log,
)
from src.models.player_rate_shrinkage import add_walk_forward_player_rate

NAIVE_PRIOR_MINUTES = 1.0  # deliberately weak floor -- almost-unshrunk trailing own rate


def run_defender_level() -> None:
    log = build_defender_game_log(MATCHUP_DATA_START_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["matchup_minutes"] > 0].copy()

    rated = add_defender_difficulty_rate(log.copy())
    naive = add_walk_forward_player_rate(log.copy(), "points_allowed", "matchup_minutes", NAIVE_PRIOR_MINUTES, prefix="naive")
    naive["naive_proj"] = naive["naive_shrunk_rate"] * naive["matchup_minutes"]

    df = rated[["gameId", "playerId", "points_allowed", "difficulty_proj"]].merge(
        naive[["gameId", "playerId", "naive_proj"]], on=["gameId", "playerId"], how="inner")
    df = df.dropna(subset=["difficulty_proj", "naive_proj"])
    df["key"] = df["gameId"].astype(str) + "_" + df["playerId"].astype(str)
    df["abs_err_model"] = (df["difficulty_proj"] - df["points_allowed"]).abs()
    df["abs_err_naive"] = (df["naive_proj"] - df["points_allowed"]).abs()

    print(f"\n=== defender-specific difficulty: n={len(df)} ===", flush=True)
    result = bootstrap_compare(
        df[["key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"}),
        df[["key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"}),
        game_id_col="key", metrics=[{"name": "defender_difficulty_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result["defender_difficulty_mae"]
    metrics_ledger.append_generic_run(
        metrics={"n_defender_games": len(df), "defender_difficulty_mae_model": mae["a"],
                 "defender_difficulty_mae_naive": mae["b"], "defender_difficulty_mae_delta": mae["delta"],
                 "defender_difficulty_mae_verdict": mae["verdict"]},
        model_name="matchup_difficulty_defender_level",
        config_flags={"dev_max_season": DEV_MAX_SEASON, "matchup_data_start_season": MATCHUP_DATA_START_SEASON},
        notes="Defender-specific points-allowed-per-matchup-minute vs weak naive floor",
    )


def run_position_group_level() -> None:
    log = build_position_group_matchup_log(MATCHUP_DATA_START_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["matchup_minutes"] > 0].copy()

    rated = add_position_group_difficulty_rate(log.copy())
    rated["model_proj"] = rated["posgroup_difficulty_rate"] * rated["matchup_minutes"]

    # naive floor = expanding-shrinkage with a huge prior ("effectively infinite memory", the
    # literal trailing season average) -- matches validate_player_minutes.py's exact convention
    # for contrasting an EWMA-family model against the simplest possible baseline.
    naive = log.copy()
    naive["team_posgroup"] = naive["team"].astype(str) + "_" + naive["position_group"]
    naive = add_walk_forward_player_rate(naive, "points_allowed", "matchup_minutes", 1000.0, prefix="naive", group_col="team_posgroup")
    naive["naive_proj"] = naive["naive_shrunk_rate"] * naive["matchup_minutes"]

    df = rated[["gameId", "team", "position_group", "points_allowed", "model_proj"]].merge(
        naive[["gameId", "team", "position_group", "naive_proj"]], on=["gameId", "team", "position_group"], how="inner")
    df = df.dropna(subset=["model_proj", "naive_proj"])
    df["key"] = df["gameId"].astype(str) + "_" + df["team"].astype(str) + "_" + df["position_group"]
    df["abs_err_model"] = (df["model_proj"] - df["points_allowed"]).abs()
    df["abs_err_naive"] = (df["naive_proj"] - df["points_allowed"]).abs()

    print(f"\n=== position-group-vs-team difficulty: n={len(df)} ===", flush=True)
    result = bootstrap_compare(
        df[["key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"}),
        df[["key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"}),
        game_id_col="key", metrics=[{"name": "posgroup_difficulty_mae", "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result["posgroup_difficulty_mae"]
    metrics_ledger.append_generic_run(
        metrics={"n_team_posgroup_games": len(df), "posgroup_difficulty_mae_model": mae["a"],
                 "posgroup_difficulty_mae_naive": mae["b"], "posgroup_difficulty_mae_delta": mae["delta"],
                 "posgroup_difficulty_mae_verdict": mae["verdict"]},
        model_name="matchup_difficulty_posgroup_level",
        config_flags={"dev_max_season": DEV_MAX_SEASON, "matchup_data_start_season": MATCHUP_DATA_START_SEASON},
        notes="Team-vs-position-group points-allowed-per-matchup-minute vs weak naive floor",
    )


if __name__ == "__main__":
    run_defender_level()
    run_position_group_level()
    print("\nboth levels appended to metrics_ledger.py", flush=True)
