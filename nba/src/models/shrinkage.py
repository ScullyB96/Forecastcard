"""Shared walk-forward shrinkage-rate estimator for NBA team-level ratings
(PACE/OFF_RATING/DEF_RATING). Reimplemented independently for this project
(no import from the sibling nhl/ project) but replicating its exact
walk-forward-safety discipline, including a same-game-leak trap that
project's own history found and fixed: if a trailing league-average
statistic is computed as a flat `.shift(1)` over a log with TWO rows per
game (home + away), and the concat/sort ever puts one game's home row
immediately before that SAME game's away row, the away row's "trailing"
average would leak the home team's own result from the game currently being
scored. `_trailing_league_stat` below avoids this structurally: it collapses
to ONE row per real `gameId` first, computes the trailing statistic on that
game-level series (so `shift(1)` always skips a whole game, never one side
of it), then broadcasts back to both team-rows via a merge on `gameId` --
leaking within a game is not possible by construction, regardless of row
order.

Exposure unit is GAMES, not minutes/possessions: unlike NHL's PP/PK
time-on-ice (which varies a lot game-to-game and would let a low-TOI game's
noisy rate count as much as a high-TOI one), OFF_RATING/DEF_RATING/PACE are
already possession-normalized per game by stats.nba.com, so one game is a
consistent unit of exposure regardless of how many possessions it contained.
"""

import pandas as pd


def _trailing_league_stat(log: pd.DataFrame, value_col: str) -> pd.Series:
    """Aligned-to-`log`'s-index Series: for each row, the trailing (strictly
    prior real games only) league-wide MEAN of `value_col`, computed by
    first collapsing to one row per gameId (mean of the two teams' own
    values) so a game's own two rows can never see each other."""
    per_game = log.groupby("gameId", as_index=False).agg(gameDate=("gameDate", "first"), _val=(value_col, "mean"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    per_game["_trailing"] = per_game["_val"].shift(1).expanding().mean()
    return log.merge(per_game[["gameId", "_trailing"]], on="gameId", how="left")["_trailing"]


def _trailing_league_stat_ewma(log: pd.DataFrame, value_col: str, halflife_games: float) -> pd.Series:
    """The recency-weighted analog of `_trailing_league_stat`: an EWMA
    (half-life in GAMES, not calendar time) over the same per-gameId-
    collapsed, `shift(1)`-guarded series, instead of an infinite-memory
    flat cumulative mean. Same same-game leak guard, same interface.

    MOTIVATION (2026-08-01, Sec9.5's still-open scoring-era-drift finding):
    mean actual points/team-game rose ~13 points from 2015-16 (102.7) to
    2025-26 (115.6), ALMOST MONOTONICALLY THROUGHOUT THE ENTIRE DEV RANGE
    (102.7 -> 105.6 -> 106.3 -> 111.2 -> ... -> 114.2 by 2023-24, not a
    sudden jump only at the dev/holdout boundary) -- confirmed directly
    from `team_strength.build_team_game_log`'s own per-season means. This
    is a structurally DIFFERENT situation from the props subsystem's
    steals investigation (Sec16-19), where the SAME family of fix
    (recency-weighted league average, then a full detrend-then-retrend)
    was tried THREE times and failed on real holdout every time -- but
    steals' problem was diagnosed as a genuine REGIME CHANGE with NO
    precedent anywhere in the pre-2024 dev data (the shift only appeared
    at the exact dev/holdout boundary), which by definition no
    within-dev-range technique could have learned to extrapolate. Scoring
    inflation is the opposite case: the trend is continuously present and
    visible THROUGHOUT the dev range itself, so a technique that tracks it
    within dev has real signal to learn from, not just noise to overfit --
    a genuinely different empirical situation, not the same bet retried a
    fourth time. Still not assumed to work -- tested via the exact same
    dev-only-first, holdout-only-once discipline as everything else."""
    per_game = log.groupby("gameId", as_index=False).agg(gameDate=("gameDate", "first"), _val=(value_col, "mean"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    per_game["_trailing_ewma"] = per_game["_val"].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
    return log.merge(per_game[["gameId", "_trailing_ewma"]], on="gameId", how="left")["_trailing_ewma"]


def add_walk_forward_rate(log: pd.DataFrame, for_col: str, against_col: str,
                           prior_games: float, prefix: str,
                           cross_season_weight: float = 0.0,
                           league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """log must have columns: gameId, gameDate, season, team, <for_col>,
    <against_col>, one row per team per game. Adds f"{prefix}_league_avg",
    f"{prefix}_attack_rate", f"{prefix}_defense_rate", and
    "games_played_before" (shared across prefixes).

    `league_avg_halflife_games` (default None, i.e. the original flat
    infinite-memory cumulative average): when set, uses
    `_trailing_league_stat_ewma` instead of `_trailing_league_stat` for the
    league-average blending TARGET, so it tracks a genuine multi-year trend
    (see that function's docstring -- Sec9.5's scoring-era-drift finding)
    instead of averaging it away across the full dev history. Mirrors
    `player_rate_shrinkage.py`'s identical optional param exactly, same
    default-preserving discipline: None reproduces the original column
    byte-for-byte (regression-tested).

    `cross_season_weight` (default 0.0, i.e. a full within-season reset,
    matching the NHL sibling's own default): at 0, a team's rating shrinks
    toward the trailing LEAGUE average at game 1 of every season, discarding
    that team's own prior-season rate entirely. A nonzero weight instead
    shrinks early-season games toward a blend of the team's own immediately-
    preceding season full-season rate and the trailing league average --
    `prior_mean = w * last_season_team_rate + (1-w) * trailing_league_avg`
    -- carrying forward roster/scheme continuity signal instead of resetting
    to a flat league-average prior. This is a genuinely untested hypothesis
    for NBA specifically (not yet validated the way NHL validated its own
    r=0.15 early-season correlation, see that project's shrinkage.py) --
    treat as a hyperparameter to grid-search in validate_team_strength_baseline.py,
    not an assumed-correct default; the 0.0 default preserves the
    conservative, already-standard behavior until a nonzero weight is
    actually shown to help via paired bootstrap."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg"
    log[league_avg_col] = (_trailing_league_stat(log, for_col) if league_avg_halflife_games is None
                            else _trailing_league_stat_ewma(log, for_col, league_avg_halflife_games))

    grp_team_season = log.groupby(["team", "season"])
    games_before = grp_team_season.cumcount()
    for_cumsum_before = grp_team_season[for_col].cumsum() - log[for_col]
    against_cumsum_before = grp_team_season[against_col].cumsum() - log[against_col]

    prior_mean_for = log[league_avg_col].copy()
    prior_mean_against = log[league_avg_col].copy()
    if cross_season_weight:
        season_totals = log.groupby(["team", "season"], as_index=False).agg(
            _season_for=(for_col, "mean"), _season_against=(against_col, "mean"))
        season_totals = season_totals.sort_values(["team", "season"]).reset_index(drop=True)
        season_totals["_prior_season_for"] = season_totals.groupby("team")["_season_for"].shift(1)
        season_totals["_prior_season_against"] = season_totals.groupby("team")["_season_against"].shift(1)
        lookup = season_totals[["team", "season", "_prior_season_for", "_prior_season_against"]]
        log = log.merge(lookup, on=["team", "season"], how="left")
        has_prior = log["_prior_season_for"].notna()
        prior_mean_for[has_prior] = (cross_season_weight * log.loc[has_prior, "_prior_season_for"]
                                      + (1 - cross_season_weight) * log.loc[has_prior, league_avg_col])
        prior_mean_against[has_prior] = (cross_season_weight * log.loc[has_prior, "_prior_season_against"]
                                          + (1 - cross_season_weight) * log.loc[has_prior, league_avg_col])
        log = log.drop(columns=["_prior_season_for", "_prior_season_against"])

    log[f"{prefix}_attack_rate"] = (for_cumsum_before + prior_games * prior_mean_for) / (games_before + prior_games)
    log[f"{prefix}_defense_rate"] = (against_cumsum_before + prior_games * prior_mean_against) / (games_before + prior_games)
    log["games_played_before"] = games_before
    return log


def add_walk_forward_mean(log: pd.DataFrame, value_col: str, prior_games: float, prefix: str,
                           cross_season_weight: float = 0.0,
                           league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """Games-based shrinkage of a SINGLE per-game value (not a for/against
    pair) toward its trailing league average -- used for PACE, which isn't a
    for/against pair (both teams in a game share one realized pace). Resets
    per season by default (`cross_season_weight=0.0`); see
    `add_walk_forward_rate`'s docstring for the blend semantics when nonzero,
    and for `league_avg_halflife_games`'s semantics (default None preserves
    the original flat cumulative average exactly)."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg"
    log[league_avg_col] = (_trailing_league_stat(log, value_col) if league_avg_halflife_games is None
                            else _trailing_league_stat_ewma(log, value_col, league_avg_halflife_games))

    grp_team_season = log.groupby(["team", "season"])
    games_before = grp_team_season.cumcount()
    cumsum_before = grp_team_season[value_col].cumsum() - log[value_col]

    prior_mean = log[league_avg_col].copy()
    if cross_season_weight:
        season_totals = log.groupby(["team", "season"], as_index=False).agg(_season_val=(value_col, "mean"))
        season_totals = season_totals.sort_values(["team", "season"]).reset_index(drop=True)
        season_totals["_prior_season_val"] = season_totals.groupby("team")["_season_val"].shift(1)
        lookup = season_totals[["team", "season", "_prior_season_val"]]
        log = log.merge(lookup, on=["team", "season"], how="left")
        has_prior = log["_prior_season_val"].notna()
        prior_mean[has_prior] = (cross_season_weight * log.loc[has_prior, "_prior_season_val"]
                                  + (1 - cross_season_weight) * log.loc[has_prior, league_avg_col])
        log = log.drop(columns=["_prior_season_val"])

    log[f"{prefix}_shrunk_mean"] = (cumsum_before + prior_games * prior_mean) / (games_before + prior_games)
    return log
