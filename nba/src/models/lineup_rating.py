"""Translates a specific game's active roster into a team-rating
ADJUSTMENT on top of `team_strength.py`'s box-score-derived baseline --
the mechanism that makes a prediction actually respond to "who's out
tonight" rather than just a team's recent average performance.

adjusted_team_oRtg = team_baseline_oRtg
    + sum_over_active_players( minutes_share_p * (rapm_off_p - team_recent_avg_rapm_off) )

`team_recent_avg_rapm_off` is what the team's ACTUAL roster composite has
been recently (a minutes-weighted average of each rotation player's RAPM-lite
rating over their last `lookback_games`) -- this is the roster-quality level
already implicitly baked into `team_strength`'s recent-performance-based
baseline, so the adjustment above measures the DELTA caused by tonight's
active roster differing from that recent norm (e.g. a star being out), not
double-counting the team's whole recent performance level.

Two minutes-share modes, mirroring the MLB sibling's oracle-vs-predictive
backtest distinction (`validate_game_simulator.py` = real historical bullpen
usage, `validate_predictive_bullpen.py` = predicted usage):
- ORACLE (this module's `oracle_minutes_shares`): uses the REAL, already-
  known minutes each player actually played in a given historical game --
  for backtesting "how much value is there in lineup-awareness at all" (an
  upper bound / ceiling test), not a deployable live prediction.
- PREDICTIVE (deferred to Phase 5's live pipeline): would project each
  active player's own trailing-average minutes, renormalized across
  tonight's active roster -- what a live, pre-game system actually has to
  work with. Not yet built; flagged here so the distinction isn't lost.
"""

import pandas as pd

DEFAULT_LOOKBACK_GAMES = 10


def player_minutes_from_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (gameId, playerId, team_side in {'home','away'}),
    with `minutes` = total on-court time across that player's stints in that
    game (derived from stint duration, not box-score MIN -- avoids a second
    data dependency and stays consistent with the possession/points already
    computed from the same stints)."""
    duration_min = (stints["endTenths"] - stints["startTenths"]) / 600.0  # tenths-of-second -> minutes
    rows = []
    for side, col in (("home", "homePlayers"), ("away", "awayPlayers")):
        exploded = stints[["gameId", col]].copy()
        exploded["minutes"] = duration_min
        exploded = exploded.explode(col).rename(columns={col: "playerId"})
        exploded["team_side"] = side
        rows.append(exploded.groupby(["gameId", "playerId", "team_side"], as_index=False)["minutes"].sum())
    return pd.concat(rows, ignore_index=True)


def oracle_minutes_shares(player_minutes: pd.DataFrame, game_id: str, team_side: str) -> pd.Series:
    """Real, post-hoc minutes shares (sums to 1.0) for one team-side of one
    game -- the ORACLE input described in this module's docstring."""
    g = player_minutes[(player_minutes["gameId"] == game_id) & (player_minutes["team_side"] == team_side)]
    total = g["minutes"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return (g.set_index("playerId")["minutes"] / total)


def predictive_minutes_shares(player_minutes: pd.DataFrame, game_id: str, team_side: str,
                               team_game_ids_before: list[str], team_side_by_game: dict[str, str],
                               lookback_games: int = DEFAULT_LOOKBACK_GAMES) -> pd.Series:
    """Backtest counterpart to `oracle_minutes_shares`: the PREDICTIVE input
    described in this module's docstring. Uses the SAME active-player set
    as the real historical game (who actually appeared -- a fair backtest
    proxy for "who was on the active/available roster", since an NBA
    team's roster minus same-day injury scratches is knowable pre-game in
    real life too, just as the live pipeline's own RotoWire-based exclusion
    is), but each player's MINUTES value is their own trailing average over
    the team's last `lookback_games` games (strictly prior to this game,
    via `team_game_ids_before`) -- NOT the real minutes they happened to
    play that specific night. That substitution is what makes this
    PREDICTIVE rather than ORACLE: minutes distribution (who gets extra run
    in a blowout, who's in foul trouble) is exactly the in-game information
    a real pre-game system doesn't have. Renormalized to sum to 1.0 over
    the active set; a player with no recent minutes history at all (e.g.
    just activated) gets 0 share rather than crashing renormalization."""
    active_players = list(oracle_minutes_shares(player_minutes, game_id, team_side).index)
    if not active_players:
        return pd.Series(dtype=float)

    recent_game_ids = team_game_ids_before[-lookback_games:]
    if not recent_game_ids:
        return pd.Series(dtype=float)

    rows = []
    for gid in recent_game_ids:
        side = team_side_by_game.get(gid)
        if side is None:
            continue
        rows.append(player_minutes[(player_minutes["gameId"] == gid) & (player_minutes["team_side"] == side)])
    if not rows:
        return pd.Series(dtype=float)
    combined = pd.concat(rows, ignore_index=True)
    trailing_avg = combined.groupby("playerId")["minutes"].mean()

    shares = trailing_avg.reindex(active_players).fillna(0.0)
    total = shares.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return shares / total


def team_recent_roster_rapm(player_minutes: pd.DataFrame, player_ratings_as_of: pd.DataFrame,
                             team_game_ids_before: list[str], team_side_by_game: dict[str, str],
                             lookback_games: int = DEFAULT_LOOKBACK_GAMES) -> tuple[float, float]:
    """Minutes-weighted average off/def RAPM-lite rating across a team's
    last `lookback_games` (real, already-played games strictly before the
    game being projected) -- what this team's actual roster composite has
    been recently. `player_ratings_as_of` must already be the correct
    walk-forward snapshot (the checkpoint valid for the date being
    projected) -- see `rapm_lite.compute_walkforward_player_ratings`.
    Falls back to (0.0, 0.0) -- i.e. no adjustment at all -- if there's no
    lookback history yet (very start of the dataset)."""
    recent_game_ids = team_game_ids_before[-lookback_games:]
    if not recent_game_ids or player_ratings_as_of.empty:
        return 0.0, 0.0

    ratings = player_ratings_as_of.set_index("playerId")[["off_rapm", "def_rapm"]]
    off_vals, def_vals, weights = [], [], []
    for gid in recent_game_ids:
        side = team_side_by_game[gid]
        shares = oracle_minutes_shares(player_minutes, gid, side)
        for player_id, share in shares.items():
            if player_id in ratings.index:
                off_vals.append(ratings.loc[player_id, "off_rapm"])
                def_vals.append(ratings.loc[player_id, "def_rapm"])
                weights.append(share)
    if not weights:
        return 0.0, 0.0
    weights = pd.Series(weights)
    return (float((pd.Series(off_vals) * weights).sum() / weights.sum()),
            float((pd.Series(def_vals) * weights).sum() / weights.sum()))


def project_lineup_adjustment(active_minutes_shares: pd.Series, player_ratings_as_of: pd.DataFrame,
                               team_recent_avg_off: float, team_recent_avg_def: float) -> tuple[float, float]:
    """Returns (off_adjustment, def_adjustment) to ADD to
    `team_strength`'s baseline oRtg/dRtg for this specific game's active
    roster. A player in `active_minutes_shares` with no RAPM-lite rating yet
    (too new / not enough prior stints) contributes 0 deviation -- treated
    as exactly average, the safest default for an unrated player."""
    ratings = player_ratings_as_of.set_index("playerId")[["off_rapm", "def_rapm"]] if not player_ratings_as_of.empty else pd.DataFrame(columns=["off_rapm", "def_rapm"])

    off_adj, def_adj = 0.0, 0.0
    for player_id, share in active_minutes_shares.items():
        off_r = ratings.loc[player_id, "off_rapm"] if player_id in ratings.index else team_recent_avg_off
        def_r = ratings.loc[player_id, "def_rapm"] if player_id in ratings.index else team_recent_avg_def
        off_adj += share * (off_r - team_recent_avg_off)
        def_adj += share * (def_r - team_recent_avg_def)
    return off_adj, def_adj
