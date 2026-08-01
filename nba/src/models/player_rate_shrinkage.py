"""Shared walk-forward shrinkage-rate estimator for PLAYER-level prop
stats (shooting, rebounding, playmaking, defensive events) -- generalizes
`shrinkage.py`'s team-level pattern to (a) an arbitrary EXPOSURE unit
(attempts, rebound chances, touches, minutes -- not just "1 game = 1 unit")
and (b) player-level grouping instead of team-level.

Team-level `shrinkage._trailing_league_stat` collapses a game's two
already-normalized team-rate rows via MEAN, because there are always
exactly 2 rows per game and each is already a per-team rate. Player rows
are NOT pre-normalized per row (a player's raw numerator/exposure for one
game is just their own counting stats, not a rate), and there are ~20-26
player-rows per game, not 2 -- so the correct trailing league rate here is
a POOLED rate: sum(numerator)/sum(exposure) across every player-row sharing
a gameId, from strictly prior games. This mirrors the NHL sibling's
`add_walk_forward_toi_rate` (pools cumulative numerator/denominator
separately, then divides), not `shrinkage.add_walk_forward_rate`'s
per-row-mean approach.

Same structural leak guard as `shrinkage.py`: collapse to one row per real
`gameId` FIRST (so a game's own rows -- however many players are in it --
can never see each other), then `shift(1)` on that game-level series, then
broadcast back via a merge on `gameId`. Season-reset by default (no
cross-season blending in this primitive for v1 -- see MODEL_DOCUMENTATION.md
for why this is a deliberate simplification, not an oversight; team-level
`shrinkage.py`'s cross_season_weight option could be added here later if a
player-level version of that hypothesis is worth testing).
"""

import pandas as pd


def _trailing_league_rate(log: pd.DataFrame, numerator_col: str, exposure_col: str) -> pd.Series:
    """Aligned-to-`log`'s-index Series: for each row, the trailing (strictly
    prior real games only) POOLED league-wide rate = cumulative
    sum(numerator) / cumulative sum(exposure) across ALL players in every
    strictly-prior game. Collapses to one row per gameId first (summing
    both columns across every player-row sharing that gameId) so a game's
    own rows can never leak into their own prediction, regardless of how
    many players appear in that game."""
    per_game = log.groupby("gameId", as_index=False).agg(
        gameDate=("gameDate", "first"), _num=(numerator_col, "sum"), _exp=(exposure_col, "sum"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    cum_num = per_game["_num"].shift(1).expanding().sum()
    cum_exp = per_game["_exp"].shift(1).expanding().sum()
    per_game["_trailing_rate"] = cum_num / cum_exp
    return log.merge(per_game[["gameId", "_trailing_rate"]], on="gameId", how="left")["_trailing_rate"]


def _trailing_league_rate_ewma(log: pd.DataFrame, numerator_col: str, exposure_col: str,
                                halflife_games: float) -> pd.Series:
    """The recency-weighted analog of `_trailing_league_rate`: instead of a
    FLAT CUMULATIVE average pooled across the ENTIRE history since the
    start of the range (every season blended together, no reset -- unlike
    the player-level cumulative sums in `add_walk_forward_player_rate`,
    which ARE season-reset), this EWMA-weights the per-game league totals
    so a genuine league-wide RATE SHIFT (not just noise) is tracked within
    a few dozen games instead of being diluted by years of stale history.

    MOTIVATION (2026-08-01, Sec14/15 props-Phase-4 fast-follow): the
    props subsystem's first holdout check found several categories (3PT
    makes, OREB, assists, steals) with a REAL dev-vs-holdout MAE gap,
    diagnosed as genuine era-driven league-wide trend shifts (rising 3PT
    volume, rising assist rates, non-monotonic OREB/steal trends). A
    same-day dev-only retune of `prior_exposure` alone (Sec15) fixed 2 of
    5 in absolute terms but did NOT close the relative gap for either --
    evidence the problem isn't the SHRINKAGE STRENGTH, it's that the
    blending TARGET itself (the league average) is stale. This function
    is the more targeted fix: still pooled across every player (low
    variance, hundreds of players per game), just recency-weighted instead
    of infinite-memory.

    Same per-game collapse + `shift(1)` leak guard as `_trailing_league_rate`
    -- EWMA is applied to the per-game POOLED numerator/exposure sums
    separately, then divided (NOT an EWMA of the per-game RATIO itself,
    which would over-weight low-exposure games)."""
    per_game = log.groupby("gameId", as_index=False).agg(
        gameDate=("gameDate", "first"), _num=(numerator_col, "sum"), _exp=(exposure_col, "sum"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    ewm_num = per_game["_num"].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
    ewm_exp = per_game["_exp"].shift(1).ewm(halflife=halflife_games, min_periods=1).mean()
    per_game["_trailing_rate_ewma"] = ewm_num / ewm_exp
    return log.merge(per_game[["gameId", "_trailing_rate_ewma"]], on="gameId", how="left")["_trailing_rate_ewma"]


def add_era_adjusted_player_rate(log: pd.DataFrame, numerator_col: str, exposure_col: str,
                                  prior_exposure: float, prefix: str, group_col: str = "playerId",
                                  current_rate_halflife_games: float = 100.0) -> pd.DataFrame:
    """DETREND-THEN-RETREND: a structurally different attempt at the same
    problem `league_avg_halflife_games` tried and failed to fix reliably
    (see Sec16 -- recency-weighting the shrinkage BLENDING TARGET directly
    passed every dev-only check for steals but still made real holdout
    performance worse, most likely because it injects the noisy/responsive
    league estimate into every single historical row's blending computation
    at once).

    This function never touches `add_walk_forward_player_rate`'s own
    formula. Instead it feeds that UNCHANGED, already-validated machinery a
    DEFLATED numerator -- each historical game's raw count divided by the
    league rate AS OF THAT GAME (walk-forward safe: uses only info strictly
    prior to that historical game, same leak guard as everywhere else) --
    so a player's cumulative history is expressed in "performance relative
    to their own era," not raw counts blended across eras with different
    true rates. The existing shrinkage math then does exactly what it
    always did, just on this era-normalized scale. Only at the very END is
    a RESPONSIVE (EWMA) estimate of the CURRENT league rate multiplied back
    in -- a single number applied once per prediction, not smeared across
    every row's own blending the way the Sec16 attempt was.

    Adds f"{prefix}_shrunk_rate" (the final era-adjusted rate, same
    interface as `add_walk_forward_player_rate`) and
    f"{prefix}_relative_shrunk_rate" (the intermediate "relative-to-era-
    average" value before re-inflation, useful for diagnostics)."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_rate_at_time = _trailing_league_rate(log, numerator_col, exposure_col)
    deflated_col = f"_{prefix}_deflated_numerator"
    log[deflated_col] = log[numerator_col] / league_rate_at_time

    log = add_walk_forward_player_rate(log, deflated_col, exposure_col, prior_exposure,
                                        prefix=f"{prefix}_relative", group_col=group_col)
    log = log.rename(columns={f"{prefix}_relative_shrunk_rate": f"{prefix}_relative_shrunk_rate"})

    current_league_rate = _trailing_league_rate_ewma(log, numerator_col, exposure_col, current_rate_halflife_games)
    log[f"{prefix}_current_league_rate"] = current_league_rate
    log[f"{prefix}_shrunk_rate"] = log[f"{prefix}_relative_shrunk_rate"] * current_league_rate
    return log.drop(columns=[deflated_col])


def add_walk_forward_player_mean_ewm(log: pd.DataFrame, value_col: str, halflife_games: float,
                                      prefix: str, group_col: str = "playerId") -> pd.DataFrame:
    """Recency-weighted walk-forward projection for a single per-game VALUE
    (not a numerator/exposure rate pair) -- e.g. minutes. Adds
    f"{prefix}_ewm_rate" = an exponentially-weighted moving average of this
    player's own STRICTLY PRIOR games (within season), with the most recent
    games weighted much more heavily than early-season ones.

    FOUND EMPIRICALLY (not assumed) while validating `player_minutes.py`:
    the infinite-memory expanding-average shrinkage in
    `add_walk_forward_player_rate` (even at its least-shrunk setting tested,
    prior_exposure=1 game) was a REAL REGRESSION vs. a naive floor using a
    SHORTER window -- confirmed via a real bootstrap run on the full dev
    range (275,138 player-games): MAE got monotonically WORSE as more prior
    games were blended in (6.43 at prior_games=1 up to 7.99 at
    prior_games=20). A direct comparison of rolling-window and EWMA
    alternatives confirmed why: minutes are NOT a slowly-drifting team-level
    quantity like PACE (where infinite-memory shrinkage toward a stable
    prior is exactly right) -- a player's ROLE can change within a season
    (rotation change, returning from injury with a minutes restriction, a
    trade), so recent games are genuinely more informative than the
    season-to-date average, unlike anything else this project has modeled
    so far. EWMA halflife=2 games gave the best real MAE (5.41) of every
    option tested (expanding-shrinkage variants, rolling windows of
    3/5/10/20 games, EWMA halflives of 2/5/10/20) -- confirmed by direct
    comparison, not assumed by analogy to the NHL sibling's own halflife
    fix (which was for a LEAGUE-wide scoring-era drift, a different
    phenomenon, at a MUCH longer halflife).

    No additional league-average blending for new players: `min_periods=1`
    means a player's 2nd career game projects from their 1st game alone --
    tested and confirmed this already outperforms the alternatives above;
    adding a league-average floor on top was not tested and is not assumed
    to help without evidence, per this project's standing discipline."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    grp = log.groupby([group_col, "season"])[value_col]
    log[f"{prefix}_ewm_rate"] = grp.transform(lambda s: s.shift(1).ewm(halflife=halflife_games, min_periods=1).mean())
    return log


def add_walk_forward_player_rate(log: pd.DataFrame, numerator_col: str, exposure_col: str,
                                  prior_exposure: float, prefix: str,
                                  group_col: str = "playerId",
                                  league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """log must have columns: gameId, gameDate, season, <group_col>,
    <numerator_col>, <exposure_col>, one row per player per game. Adds
    f"{prefix}_league_avg_rate", f"{prefix}_shrunk_rate", and
    "games_played_before" (player-season games played before this row,
    season-reset).

    `shrunk_rate = (numerator_cumsum_before + prior_exposure * league_avg_rate)
                   / (exposure_cumsum_before + prior_exposure)`

    -- same algebra as `shrinkage.add_walk_forward_rate`, just with
    `prior_exposure` denominated in the category's natural unit (attempts,
    chances, touches, minutes) instead of games, which is what lets a
    high-volume stat (e.g. FTA) stabilize faster in real terms than a
    low-volume one (e.g. 3PA) even at the same `prior_exposure` game-count-
    equivalent.

    `league_avg_halflife_games`: `None` (default) uses `_trailing_league_rate`,
    the ORIGINAL flat-cumulative league average -- exact prior behavior,
    unchanged, so every already-validated model configuration is completely
    unaffected by this parameter's mere existence. Pass a halflife to use
    `_trailing_league_rate_ewma` instead -- a recency-weighted league
    average that can track a genuine era-wide rate shift instead of being
    diluted by years of stale history (see that function's docstring for
    the Sec14/15 finding that motivated this option)."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg_rate"
    if league_avg_halflife_games is None:
        log[league_avg_col] = _trailing_league_rate(log, numerator_col, exposure_col)
    else:
        log[league_avg_col] = _trailing_league_rate_ewma(log, numerator_col, exposure_col, league_avg_halflife_games)

    grp = log.groupby([group_col, "season"])
    games_before = grp.cumcount()
    numerator_before = grp[numerator_col].cumsum() - log[numerator_col]
    exposure_before = grp[exposure_col].cumsum() - log[exposure_col]

    league_avg = log[league_avg_col]
    log[f"{prefix}_shrunk_rate"] = (numerator_before + prior_exposure * league_avg) / (exposure_before + prior_exposure)
    log["games_played_before"] = games_before
    return log
