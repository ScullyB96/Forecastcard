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
                                  group_col: str = "playerId") -> pd.DataFrame:
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
    equivalent."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg_rate"
    log[league_avg_col] = _trailing_league_rate(log, numerator_col, exposure_col)

    grp = log.groupby([group_col, "season"])
    games_before = grp.cumcount()
    numerator_before = grp[numerator_col].cumsum() - log[numerator_col]
    exposure_before = grp[exposure_col].cumsum() - log[exposure_col]

    league_avg = log[league_avg_col]
    log[f"{prefix}_shrunk_rate"] = (numerator_before + prior_exposure * league_avg) / (exposure_before + prior_exposure)
    log["games_played_before"] = games_before
    return log
