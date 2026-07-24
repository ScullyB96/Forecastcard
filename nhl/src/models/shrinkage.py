"""Shared walk-forward shrinkage-rate estimator, factored out of
baseline_naive_poisson.py once a second consumer (the xG-based strength
model, team_strength_xg.py) needed the identical logic on a different pair
of for/against columns -- extracted now rather than duplicated, not
speculatively built ahead of a second real use.

Same guarantee regardless of which columns it's given: a team's rate for
game G is computed ONLY from that team's own rows strictly before G's date,
within the same season; the league-average used for shrinkage is a trailing
expanding statistic over ALL rows strictly before G's date, with no season
reset.

BUG FOUND AND FIXED (2026-07-23), confirmed real before fixing rather than
assumed: the league-average trailing statistic was originally computed with
a flat `.shift(1).expanding()` over the raw log, which has TWO rows per real
game (home + away). Checked directly: `build_team_game_log`'s concat always
puts the home row before the away row, and pandas' sort is stable, so after
sorting by (gameDate, gameId) the home row of game G *always* sorts
immediately before the away row of that same game G -- meaning the away
row's "trailing" league-average actually included the home team's own
result FROM THE SAME GAME being predicted (confirmed empirically: home
sorts first in 21,612/21,612 real games, zero exceptions). Home rows didn't
have the reverse problem (nothing from game G sorts before them), so this
was also an asymmetric home/away bug, not just a leak. `_trailing_league_stat`
below fixes this by aggregating to ONE row per real gameId first, computing
the trailing statistic on that game-level series (so shift(1) always skips
a whole game, never just one side of it), then broadcasting back to both
team-rows via a merge on gameId -- structurally impossible to leak within a
game. All three validated Cycle 1-2 models were re-run after this fix (see
MODEL_DOCUMENTATION.md Sec4.5) to check whether it moved any real number.
"""

import pandas as pd


def _trailing_league_stat(log: pd.DataFrame, value_col: str, agg: str,
                           halflife_games: float | None = None) -> pd.Series:
    """Aligned-to-`log`'s-index Series: for each row, the trailing (strictly
    prior real games only) league-wide statistic of `value_col`, computed by
    first collapsing to one row per gameId (agg="mean" -- average of the two
    teams' own values, used for a per-team-per-game league average; agg="sum"
    -- combined total across both teams, used for a per-60-minute pooled
    rate's numerator/denominator) so a game's own two rows can never see
    each other.

    `halflife_games` (default None, preserving every existing caller's exact
    behavior): None uses a plain expanding mean/sum -- infinite memory, every
    prior game weighted equally regardless of age. A real number switches to
    an exponentially-weighted mean with that half-life (in games), for the
    Sec4.15 scoring-era-drift fix -- confirmed real and substantial (2026-07-
    23): the plain expanding mean lags a genuine league-wide scoring-level
    shift (the real 2017-18 jump from ~2.78 to ~2.97+ goals/team/game) by
    0.10-0.32 goals/team/game for every one of the 9 seasons since, because
    an infinite-memory average can never fully "forget" 9+ years of
    now-stale, lower-scoring history. When halflife_games is given, BOTH
    agg="mean" and agg="sum" callers get an EWMA of the per-game value
    series (not a literal decayed sum) -- mathematically equivalent for
    every downstream use here: agg="sum" is only ever used as one of two
    matched numerator/denominator terms whose RATIO is what actually
    matters (see add_walk_forward_toi_rate), and that ratio is invariant to
    substituting "EWMA mean of per-game sums" for "raw cumulative sum" as
    long as the same weighting is applied to both terms."""
    per_game = log.groupby("gameId", as_index=False).agg(gameDate=("gameDate", "first"), _val=(value_col, agg))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    shifted = per_game["_val"].shift(1)
    if halflife_games is None:
        per_game["_trailing"] = shifted.expanding().mean() if agg == "mean" else shifted.expanding().sum()
    else:
        per_game["_trailing"] = shifted.ewm(halflife=halflife_games, min_periods=1).mean()
    return log.merge(per_game[["gameId", "_trailing"]], on="gameId", how="left")["_trailing"]


def add_walk_forward_rate(log: pd.DataFrame, for_col: str, against_col: str,
                           prior_games: float, prefix: str,
                           league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """log must have columns: gameId, gameDate, season, team, is_home,
    <for_col>, <against_col>, one row per team per game. Adds
    f"{prefix}_league_avg", f"{prefix}_attack_rate", f"{prefix}_defense_rate",
    and "games_played_before" (shared across prefixes -- games-played count
    doesn't depend on which stat is being rated).

    `league_avg_halflife_games`: see _trailing_league_stat -- None (default)
    preserves every existing caller's exact behavior."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg"
    log[league_avg_col] = _trailing_league_stat(log, for_col, agg="mean", halflife_games=league_avg_halflife_games)

    grp_team_season = log.groupby(["team", "season"])
    games_before = grp_team_season.cumcount()
    for_cumsum_before = grp_team_season[for_col].cumsum() - log[for_col]
    against_cumsum_before = grp_team_season[against_col].cumsum() - log[against_col]

    league_avg = log[league_avg_col]
    log[f"{prefix}_attack_rate"] = (for_cumsum_before + prior_games * league_avg) / (games_before + prior_games)
    log[f"{prefix}_defense_rate"] = (against_cumsum_before + prior_games * league_avg) / (games_before + prior_games)
    log["games_played_before"] = games_before
    return log


def add_walk_forward_mean(log: pd.DataFrame, value_col: str, prior_games: float, prefix: str,
                           reset_each_season: bool = True,
                           league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """Games-based shrinkage of a SINGLE per-game value (not a for/against
    pair) toward its trailing league average -- e.g. a team's own average
    PP time-on-ice per game, used by team_strength_situational.py to test
    whether a team's own penalty-drawing/taking history should weight its
    expected PP/PK minutes, instead of a shared league-average constant.
    Adds f"{prefix}_league_avg" and f"{prefix}_shrunk_mean".

    `reset_each_season` (default True, preserving every existing caller's
    behavior unchanged): when True, a team/goalie's own cumulative history
    resets at each season boundary (groups by "team" AND "season"). When
    False, history accumulates across the entire dataset regardless of
    season boundary (groups by "team" only) -- added for
    team_strength_goalie.py's cross-season-goalie-history test (see
    MODEL_DOCUMENTATION.md Sec4.10): an established goalie's skill doesn't
    plausibly reset over the summer the way a team's own roster composition
    can meaningfully change, so testing whether carrying it over helps is a
    genuinely different, real question from every other use of this
    function so far, all of which keep the season reset."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    league_avg_col = f"{prefix}_league_avg"
    log[league_avg_col] = _trailing_league_stat(log, value_col, agg="mean", halflife_games=league_avg_halflife_games)

    group_cols = ["team", "season"] if reset_each_season else ["team"]
    grp = log.groupby(group_cols)
    games_before = grp.cumcount()
    cumsum_before = grp[value_col].cumsum() - log[value_col]

    league_avg = log[league_avg_col]
    log[f"{prefix}_shrunk_mean"] = (cumsum_before + prior_games * league_avg) / (games_before + prior_games)
    return log


def add_walk_forward_toi_rate(log: pd.DataFrame, for_col: str, against_col: str, toi_col_seconds: str,
                               prior_minutes: float, prefix: str,
                               league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """Same walk-forward/shrinkage guarantee as add_walk_forward_rate, but
    for situations (5v5/PP/PK) where the natural exposure unit is ice time,
    not games -- a team can play a full game with only 2 minutes of power
    play, so averaging GAME-level rates would let that low-TOI game's noisy
    rate count as much as a 10-minute-PP game. Instead pools cumulative xG
    and cumulative TOI (minutes) separately across all prior GAMES (not
    prior rows -- see _trailing_league_stat), THEN divides, both for the
    league-average and the team-level shrunk rate.

    `prior_minutes` is the shrinkage pseudo-exposure, in minutes (the TOI
    analogue of `prior_games` in add_walk_forward_rate) -- a PLACEHOLDER,
    not yet empirically calibrated, same standard as every other
    stabilization constant in this project (see MODEL_DOCUMENTATION.md
    Sec3.2). Adds f"{prefix}_league_avg_for_per60", f"{prefix}_league_avg_against_per60",
    f"{prefix}_attack_rate_per60", f"{prefix}_defense_rate_per60", and
    "{prefix}_toi_min" (this row's own TOI in minutes, useful for
    diagnostics).

    BUG FOUND AND FIXED (2026-07-23): originally computed ONE league average
    (from for_col only) and used it as the shrinkage/ratio baseline for BOTH
    attack_rate and defense_rate. That's harmless when for_col/against_col
    are symmetric league-wide (true for whole-game and 5v5 goals: total
    goals-for across the league trivially equals total goals-against) but
    WRONG for PP/PK: a team's own "pk_xgf" (rare shorthanded goals scored
    while killing a penalty, confirmed ~0.77/60 empirically) and "pk_xga"
    (goals CONCEDED while killing -- essentially the opponent's power-play
    production, confirmed ~6/60) are entirely different-magnitude events,
    not symmetric with each other. Comparing a team's pk_defense_rate
    against the wrong (tiny, for_col-derived) league average inflated the
    ratio ~8x, confirmed directly by printing a real sample game's rates
    before this fix. Now computes for_col's and against_col's league
    averages SEPARATELY -- the true symmetric partner of PP's "for" league
    average is PK's "against" league average (a power-play goal for one
    team is simultaneously a shorthanded-goal-against for the other, same
    clock window), which callers must pair up correctly themselves (see
    team_strength_situational.predict_situational_lambda)."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    toi_min_col = f"{prefix}_toi_min"
    log[toi_min_col] = log[toi_col_seconds] / 60.0

    cum_toi_league = _trailing_league_stat(log, toi_min_col, agg="sum", halflife_games=league_avg_halflife_games)
    league_avg_for_col = f"{prefix}_league_avg_for_per60"
    league_avg_against_col = f"{prefix}_league_avg_against_per60"
    log[league_avg_for_col] = 60 * _trailing_league_stat(
        log, for_col, agg="sum", halflife_games=league_avg_halflife_games) / cum_toi_league
    log[league_avg_against_col] = 60 * _trailing_league_stat(
        log, against_col, agg="sum", halflife_games=league_avg_halflife_games) / cum_toi_league

    grp_team_season = log.groupby(["team", "season"])
    toi_cumsum_before = grp_team_season[toi_min_col].cumsum() - log[toi_min_col]
    for_cumsum_before = grp_team_season[for_col].cumsum() - log[for_col]
    against_cumsum_before = grp_team_season[against_col].cumsum() - log[against_col]

    denom = toi_cumsum_before + prior_minutes
    prior_for_equiv = (prior_minutes / 60.0) * log[league_avg_for_col]
    prior_against_equiv = (prior_minutes / 60.0) * log[league_avg_against_col]
    log[f"{prefix}_attack_rate_per60"] = 60 * (for_cumsum_before + prior_for_equiv) / denom
    log[f"{prefix}_defense_rate_per60"] = 60 * (against_cumsum_before + prior_against_equiv) / denom
    return log


def add_walk_forward_toi_rate_cross_season(log: pd.DataFrame, for_col: str, against_col: str,
                                            toi_col_seconds: str, prior_minutes: float, prefix: str,
                                            cross_season_weight: float,
                                            league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """Same as add_walk_forward_toi_rate, except the shrinkage PRIOR (the
    "prior_minutes worth of league average" term) is replaced by a blend of
    the trailing league average and the SAME team's own immediately-
    preceding season's full-season rate:

        prior_mean = cross_season_weight * last_season_team_rate
                     + (1 - cross_season_weight) * trailing_league_avg

    cross_season_weight=0 is an exact no-op (identical output to
    add_walk_forward_toi_rate). A team with no matched immediately-preceding
    season in the data (first season on record, e.g. an expansion team, or
    the dataset's own first season) falls back to the pure trailing league
    average for that team-season -- there is no real prior-season signal to
    blend in, and NaN must not be allowed to propagate into the rate.

    Motivated by a real, measured effect (MODEL_DOCUMENTATION.md Sec12.1):
    a team's own last-season xG-differential residual correlates with its
    EARLY-season (games 1-15) performance at r=0.15 (p<1e-40, n=7,800) --
    signal the season-reset prior currently discards completely at game 1
    of every season."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    toi_min_col = f"{prefix}_toi_min"
    log[toi_min_col] = log[toi_col_seconds] / 60.0

    cum_toi_league = _trailing_league_stat(log, toi_min_col, agg="sum", halflife_games=league_avg_halflife_games)
    league_avg_for_col = f"{prefix}_league_avg_for_per60"
    league_avg_against_col = f"{prefix}_league_avg_against_per60"
    log[league_avg_for_col] = 60 * _trailing_league_stat(
        log, for_col, agg="sum", halflife_games=league_avg_halflife_games) / cum_toi_league
    log[league_avg_against_col] = 60 * _trailing_league_stat(
        log, against_col, agg="sum", halflife_games=league_avg_halflife_games) / cum_toi_league

    # Each team-season's FULL-season rate (uses every game of that season --
    # legitimate here because it is only ever attached to the NEXT season's
    # rows, which are strictly later in time; never used within its own season).
    season_totals = log.groupby(["team", "season"], as_index=False).agg(
        _season_for=(for_col, "sum"), _season_against=(against_col, "sum"), _season_toi=(toi_min_col, "sum"))
    season_totals["_season_rate_for"] = 60 * season_totals["_season_for"] / season_totals["_season_toi"]
    season_totals["_season_rate_against"] = 60 * season_totals["_season_against"] / season_totals["_season_toi"]
    season_totals = season_totals.sort_values(["team", "season"]).reset_index(drop=True)
    season_totals["_prior_season_rate_for"] = season_totals.groupby("team")["_season_rate_for"].shift(1)
    season_totals["_prior_season_rate_against"] = season_totals.groupby("team")["_season_rate_against"].shift(1)
    prior_lookup = season_totals[["team", "season", "_prior_season_rate_for", "_prior_season_rate_against"]]

    log = log.merge(prior_lookup, on=["team", "season"], how="left")

    has_prior = log["_prior_season_rate_for"].notna()
    blended_for = log[league_avg_for_col].copy()
    blended_for[has_prior] = (cross_season_weight * log.loc[has_prior, "_prior_season_rate_for"]
                               + (1 - cross_season_weight) * log.loc[has_prior, league_avg_for_col])
    blended_against = log[league_avg_against_col].copy()
    blended_against[has_prior] = (cross_season_weight * log.loc[has_prior, "_prior_season_rate_against"]
                                   + (1 - cross_season_weight) * log.loc[has_prior, league_avg_against_col])

    grp_team_season = log.groupby(["team", "season"])
    toi_cumsum_before = grp_team_season[toi_min_col].cumsum() - log[toi_min_col]
    for_cumsum_before = grp_team_season[for_col].cumsum() - log[for_col]
    against_cumsum_before = grp_team_season[against_col].cumsum() - log[against_col]

    denom = toi_cumsum_before + prior_minutes
    prior_for_equiv = (prior_minutes / 60.0) * blended_for
    prior_against_equiv = (prior_minutes / 60.0) * blended_against
    log[f"{prefix}_attack_rate_per60"] = 60 * (for_cumsum_before + prior_for_equiv) / denom
    log[f"{prefix}_defense_rate_per60"] = 60 * (against_cumsum_before + prior_against_equiv) / denom
    return log.drop(columns=["_prior_season_rate_for", "_prior_season_rate_against"])
