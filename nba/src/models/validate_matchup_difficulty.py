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
    POSGROUP_HALFLIFE_GAMES,
    POSGROUP_HALFLIFE_GAMES_AST,
    POSGROUP_PRIOR_MATCHUP_MINUTES_TOV,
    PRIOR_MATCHUP_MINUTES_AST,
    PRIOR_MATCHUP_MINUTES_DIFFICULTY,
    PRIOR_MATCHUP_MINUTES_TOV,
    add_defender_stat_difficulty_rate,
    add_position_group_stat_difficulty_rate,
    add_position_group_stat_difficulty_rate_expanding,
    build_defender_game_log,
    build_position_group_matchup_log,
)
from src.models.player_rate_shrinkage import add_walk_forward_player_rate

NAIVE_PRIOR_MINUTES = 1.0  # deliberately weak floor -- almost-unshrunk trailing own rate
NAIVE_PRIOR_EWMA_FAMILY = 1000.0  # "effectively infinite memory" naive floor for an EWMA-family model

_DEFENDER_CATEGORIES = (
    # (stat_col, prior_matchup_minutes, prefix, model_name, label)
    ("points_allowed", PRIOR_MATCHUP_MINUTES_DIFFICULTY, "difficulty", "matchup_difficulty_defender_level",
     "defender-specific points-allowed"),
    ("ast_allowed", PRIOR_MATCHUP_MINUTES_AST, "ast_difficulty", "matchup_difficulty_defender_ast",
     "defender-specific AST-allowed"),
    ("tov_forced", PRIOR_MATCHUP_MINUTES_TOV, "tov_difficulty", "matchup_difficulty_defender_tov",
     "defender-specific TOV-forced"),
)

_POSGROUP_CATEGORIES = (
    # (stat_col, family, param, prefix, model_name, label)
    ("points_allowed", "ewma", POSGROUP_HALFLIFE_GAMES, "posgroup", "matchup_difficulty_posgroup_level",
     "position-group-vs-team points-allowed"),
    ("ast_allowed", "ewma", POSGROUP_HALFLIFE_GAMES_AST, "ast_posgroup", "matchup_difficulty_posgroup_ast",
     "position-group-vs-team AST-allowed"),
    ("tov_forced", "expanding", POSGROUP_PRIOR_MATCHUP_MINUTES_TOV, "tov_posgroup", "matchup_difficulty_posgroup_tov",
     "position-group-vs-team TOV-forced"),
)


def run_defender_level(stat_col: str, prior_matchup_minutes: float, prefix: str, model_name: str, label: str) -> None:
    log = build_defender_game_log(MATCHUP_DATA_START_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["matchup_minutes"] > 0].copy()

    rated = add_defender_stat_difficulty_rate(log.copy(), stat_col, prior_matchup_minutes, prefix=prefix)
    naive = add_walk_forward_player_rate(log.copy(), stat_col, "matchup_minutes", NAIVE_PRIOR_MINUTES, prefix="naive")
    naive["naive_proj"] = naive["naive_shrunk_rate"] * naive["matchup_minutes"]

    df = rated[["gameId", "playerId", stat_col, f"{prefix}_proj"]].merge(
        naive[["gameId", "playerId", "naive_proj"]], on=["gameId", "playerId"], how="inner")
    df = df.dropna(subset=[f"{prefix}_proj", "naive_proj"])
    df["key"] = df["gameId"].astype(str) + "_" + df["playerId"].astype(str)
    df["abs_err_model"] = (df[f"{prefix}_proj"] - df[stat_col]).abs()
    df["abs_err_naive"] = (df["naive_proj"] - df[stat_col]).abs()

    print(f"\n=== {label}: n={len(df)} ===", flush=True)
    metric_name = f"{prefix}_mae"
    result = bootstrap_compare(
        df[["key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"}),
        df[["key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"}),
        game_id_col="key", metrics=[{"name": metric_name, "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result[metric_name]
    metrics_ledger.append_generic_run(
        metrics={"n_defender_games": len(df), f"{metric_name}_model": mae["a"],
                 f"{metric_name}_naive": mae["b"], f"{metric_name}_delta": mae["delta"],
                 f"{metric_name}_verdict": mae["verdict"]},
        model_name=model_name,
        config_flags={"dev_max_season": DEV_MAX_SEASON, "matchup_data_start_season": MATCHUP_DATA_START_SEASON},
        notes=f"{label} per-matchup-minute vs weak naive floor",
    )


def run_position_group_level(stat_col: str, family: str, param: float, prefix: str,
                              model_name: str, label: str) -> None:
    log = build_position_group_matchup_log(MATCHUP_DATA_START_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["matchup_minutes"] > 0].copy()

    if family == "ewma":
        rated = add_position_group_stat_difficulty_rate(log.copy(), stat_col, param, prefix=prefix)
        # naive floor = expanding-shrinkage with a huge prior ("effectively infinite memory", the
        # literal trailing season average) -- matches validate_player_minutes.py's convention for
        # contrasting an EWMA-family model against the simplest possible baseline.
        naive_prior = NAIVE_PRIOR_EWMA_FAMILY
    else:
        rated = add_position_group_stat_difficulty_rate_expanding(log.copy(), stat_col, param, prefix=prefix)
        naive_prior = NAIVE_PRIOR_MINUTES
    rated["model_proj"] = rated[f"{prefix}_difficulty_rate"] * rated["matchup_minutes"]

    naive = log.copy()
    naive["team_posgroup"] = naive["team"].astype(str) + "_" + naive["position_group"]
    naive = add_walk_forward_player_rate(naive, stat_col, "matchup_minutes", naive_prior, prefix="naive", group_col="team_posgroup")
    naive["naive_proj"] = naive["naive_shrunk_rate"] * naive["matchup_minutes"]

    df = rated[["gameId", "team", "position_group", stat_col, "model_proj"]].merge(
        naive[["gameId", "team", "position_group", "naive_proj"]], on=["gameId", "team", "position_group"], how="inner")
    df = df.dropna(subset=["model_proj", "naive_proj"])
    df["key"] = df["gameId"].astype(str) + "_" + df["team"].astype(str) + "_" + df["position_group"]
    df["abs_err_model"] = (df["model_proj"] - df[stat_col]).abs()
    df["abs_err_naive"] = (df["naive_proj"] - df[stat_col]).abs()

    print(f"\n=== {label}: n={len(df)} ===", flush=True)
    metric_name = f"{prefix}_difficulty_mae"
    result = bootstrap_compare(
        df[["key", "abs_err_model"]].rename(columns={"abs_err_model": "abs_err"}),
        df[["key", "abs_err_naive"]].rename(columns={"abs_err_naive": "abs_err"}),
        game_id_col="key", metrics=[{"name": metric_name, "col": "abs_err", "higher_is_better": False}],
        label_a="shrunk", label_b="naive",
    )
    mae = result[metric_name]
    metrics_ledger.append_generic_run(
        metrics={"n_team_posgroup_games": len(df), f"{metric_name}_model": mae["a"],
                 f"{metric_name}_naive": mae["b"], f"{metric_name}_delta": mae["delta"],
                 f"{metric_name}_verdict": mae["verdict"]},
        model_name=model_name,
        config_flags={"dev_max_season": DEV_MAX_SEASON, "matchup_data_start_season": MATCHUP_DATA_START_SEASON, "family": family},
        notes=f"{label} per-matchup-minute vs weak naive floor",
    )


if __name__ == "__main__":
    for stat_col, prior, prefix, model_name, label in _DEFENDER_CATEGORIES:
        run_defender_level(stat_col, prior, prefix, model_name, label)
    for stat_col, family, param, prefix, model_name, label in _POSGROUP_CATEGORIES:
        run_position_group_level(stat_col, family, param, prefix, model_name, label)
    print("\nall categories appended to metrics_ledger.py", flush=True)
