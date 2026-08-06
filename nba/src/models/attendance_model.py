"""Stage-1 attendance-probability model (task #58, Sec47's decomposition
finding): the ENTIRE Phase 2 predictive-mode gap is attendance-prediction
error, not share-redistribution (semi-oracle -- real attendance + trailing-
average shares -- is statistically indistinguishable from full oracle;
see MODEL_DOCUMENTATION.md Sec47). `lineup_rating.predictive_minutes_shares`'s
current active-player SET is a crude binary rule (trailing-`lookback_games`-
window UNION: appeared at least once in the last 10 games = fully included
with implicit probability 1.0, otherwise fully excluded with implicit
probability 0.0) -- this module builds a genuinely PROBABILISTIC
alternative using only already-available, walk-forward-safe features (no
new ingest): games-played fraction and current DNP-streak length, per
Sec47's own "narrowed recommendation."

Deliberately does NOT touch the (already-confirmed-adequate, Sec47)
share-redistribution step -- this is Stage 1 ONLY, validated directly
against real oracle attendance (who actually played), not folded into the
full lineup-adjustment MAE, exactly as Sec47 prescribes: "before ever
combining it with the share-redistribution step."
"""

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK_GAMES = 10  # matches lineup_rating.DEFAULT_LOOKBACK_GAMES's own window -- the
# same candidate pool (anyone who appeared at least once in the team's last 10 games), so this
# model is a genuine drop-in refinement of the SAME population, not a differently-scoped one.


def attendance_features_for_game(player_minutes: pd.DataFrame, team_game_ids_before: list,
                                  team_side_by_game: dict, lookback_games: int = DEFAULT_LOOKBACK_GAMES) -> pd.DataFrame:
    """For every player who appeared in AT LEAST ONE of the team's last
    `lookback_games` games (the exact candidate pool
    `lineup_rating.predictive_minutes_shares` already uses), computes two
    walk-forward features from STRICTLY PRIOR games only (`team_game_ids_before`
    is the caller's already-`< today`-filtered list, same convention as
    every other lookback in this codebase):

    - `games_played_fraction`: fraction of the window the player appeared in.
    - `dnp_streak`: consecutive MOST-RECENT games (working backward from the
      game right before tonight) the player did NOT appear in -- 0 if they
      played last game, `lookback_games` if they haven't played at all in
      the window (captures RECENCY of an ongoing absence, which
      `games_played_fraction` alone can't distinguish from an equal number
      of scattered misses).

    Returns one row per candidate playerId, empty if the window itself is
    empty (not enough history yet)."""
    window = team_game_ids_before[-lookback_games:]
    if not window:
        return pd.DataFrame(columns=["playerId", "games_played_fraction", "dnp_streak"])

    per_game_players = {}
    for gid in window:
        side = team_side_by_game.get(gid)
        if side is None:
            per_game_players[gid] = set()
            continue
        g = player_minutes[(player_minutes["gameId"] == gid) & (player_minutes["team_side"] == side)]
        per_game_players[gid] = set(g["playerId"])

    candidates = set()
    for players in per_game_players.values():
        candidates |= players

    rows = []
    for pid in candidates:
        attended = [pid in per_game_players[gid] for gid in window]  # chronological order
        games_played_fraction = float(np.mean(attended))
        streak = 0
        for a in reversed(attended):  # most recent game first
            if a:
                break
            streak += 1
        rows.append({"playerId": pid, "games_played_fraction": games_played_fraction, "dnp_streak": streak})
    return pd.DataFrame(rows)


def predict_attendance_probability(features: pd.DataFrame, streak_decay: float = 0.7) -> pd.Series:
    """P(attend tonight) = `games_played_fraction`, decayed multiplicatively
    by `streak_decay` per consecutive recent DNP -- a currently-ongoing
    absence is stickier than the raw trailing fraction alone suggests (the
    same "recent status carries more information than a flat rolling
    average" idea `shrinkage.own_halflife_games` uses for team ratings,
    applied here to a binary presence signal instead of a continuous rate).
    At `dnp_streak=0` this reduces to the raw fraction unchanged; each
    additional consecutive miss multiplies the estimate down further.
    Clipped to `[0, 1]` (a probability, not a raw ratio --
    `games_played_fraction` is already in that range, but the floating-
    point product is clipped defensively).

    `streak_decay=0.7` (default, ADOPTED 2026-08-06): swept 0.3-1.0 against
    real oracle attendance on the full dev range (n=292,954 candidate
    player-games) -- EVERY value tested is a real, decisive Brier-score
    improvement over the current union rule's implicit P=1.0 for any
    trailing-window member (0.310 pooled), and 0.7 is the single best value
    found (0.1396, roughly HALVING the current rule's error). See
    MODEL_DOCUMENTATION.md Sec53 for the full sweep and calibration check
    (systematically conservative -- real attendance runs somewhat higher
    than predicted at every decile -- but still a large, real improvement
    over the union rule's far more miscalibrated implicit certainty)."""
    prob = features["games_played_fraction"] * (streak_decay ** features["dnp_streak"])
    return prob.clip(0.0, 1.0)
