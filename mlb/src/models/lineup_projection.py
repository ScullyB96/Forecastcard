"""Lineup projection for a game whose real batting order isn't posted yet.
Confirmed directly against the live MLB Stats API: even fetched the night
before, a future game's schedule entry has a real probable pitcher but EMPTY
lineups and weather -- both are only posted a few hours before first pitch.
This fills the lineup half of that gap (weather has its own forecast/
climatological fallback -- see weather_forecast.py).

Two levels, both walk-forward-safe (only games strictly before target_date):

1. project_lineup: each team's MOST RECENT actual complete lineup. Validated
   directly on real 2025 data: consecutive games for the same team share a
   mean of 6.85/9 players (76% roster overlap) and match the exact same
   player in the exact same batting slot 47.9% of the time -- both far
   better than nothing, though NOT a claim that tomorrow's lineup will be
   identical.

2. project_lineup_platoon_aware: the same baseline, but swaps in a
   platoon-appropriate alternative when a real signal exists in this team's
   recent history -- a DIFFERENT player who has started significantly more
   often at that exact defensive position specifically against today's
   opposing pitcher's hand. Confirmed directly on real 2025 data before
   building this: e.g. a clean Athletics DH platoon pair where one player
   started 85% of games vs RHP (113/133) and the other started 81% vs LHP
   (13/16) -- exactly the pattern this is meant to catch, which the plain
   "most recent lineup" baseline would miss whenever the most recent game
   happened to be against the OTHER hand.
"""

import pandas as pd

PLATOON_WINDOW_GAMES = 40  # trailing games to look back for a platoon pattern
MIN_PLATOON_STARTS = 6     # minimum starts vs a given hand before trusting the signal enough to swap


def _most_recent_complete_lineup(schedule: pd.DataFrame, lineups: pd.DataFrame,
                                  team: str, target_date: str) -> pd.DataFrame | None:
    """The full lineup ROWS (batting_order, player_id, position_code) for the
    most recent COMPLETE (9-batter) game this team played before target_date."""
    reg = schedule[
        (schedule["game_type"] == "R") & (schedule["status"] == "Final") & (schedule["date"] < target_date)
        & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
    ].sort_values("date")
    if reg.empty:
        return None
    for row in reg.iloc[::-1].itertuples():
        side = "home" if row.home_team == team else "away"
        game_lu = lineups[(lineups["game_pk"] == row.game_pk) & (lineups["team_side"] == side)]
        game_lu = game_lu.sort_values("batting_order")
        if len(game_lu) == 9:
            return game_lu
    return None


def project_lineup(schedule: pd.DataFrame, lineups: pd.DataFrame, team: str, target_date: str) -> list[int] | None:
    """The most recent COMPLETE (9-batter) lineup this team actually used
    before target_date, in batting order. Returns None if no such game
    exists yet (e.g. the team's very first game in our cached history)."""
    game_lu = _most_recent_complete_lineup(schedule, lineups, team, target_date)
    return game_lu["player_id"].tolist() if game_lu is not None else None


def _team_position_hand_history(schedule: pd.DataFrame, lineups: pd.DataFrame, pitcher_hand: pd.Series,
                                 team: str, target_date: str, window_games: int = PLATOON_WINDOW_GAMES) -> pd.DataFrame:
    """Trailing window_games worth of (position_code, player_id, opposing
    pitcher hand) rows for this team, strictly before target_date."""
    reg = schedule[
        (schedule["game_type"] == "R") & (schedule["status"] == "Final") & (schedule["date"] < target_date)
        & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
    ].sort_values("date").tail(window_games)
    if reg.empty:
        return pd.DataFrame(columns=["position_code", "player_id", "opp_hand"])

    rows = []
    for row in reg.itertuples():
        side = "home" if row.home_team == team else "away"
        opp_pitcher_id = row.away_probable_pitcher_id if side == "home" else row.home_probable_pitcher_id
        opp_hand = pitcher_hand.get(opp_pitcher_id)
        if opp_hand is None or pd.isna(opp_pitcher_id):
            continue
        game_lu = lineups[(lineups["game_pk"] == row.game_pk) & (lineups["team_side"] == side)]
        for r in game_lu.itertuples():
            rows.append({"position_code": r.position_code, "player_id": r.player_id, "opp_hand": opp_hand})
    return pd.DataFrame(rows)


def project_lineup_platoon_aware(schedule: pd.DataFrame, lineups: pd.DataFrame, pitcher_hand: pd.Series,
                                  team: str, target_date: str, opposing_pitcher_id,
                                  window_games: int = PLATOON_WINDOW_GAMES,
                                  min_starts: int = MIN_PLATOON_STARTS) -> list[int] | None:
    """The baseline most-recent-lineup projection, with each slot swapped to
    a platoon-appropriate alternative ONLY when a genuine platoon split is
    evident, in this team's recent history: the CURRENT slot's occupant must
    themselves show a clear minority share of starts at that position
    against today's opposing hand (<35%, i.e. real evidence they're the
    "weak side" of a platoon there), and the alternative must show a clear
    majority share (>=65%) with at least min_starts starts against that same
    hand. A first version that swapped on any "someone else had more total
    starts" signal (regardless of whether the current player showed any
    platoon weakness at all) was found to trigger 3+ swaps per lineup --
    real MLB platoons are usually 0-2 players per team, so that was
    overcorrecting on noise, not real platoon signal; this stricter
    share-based, two-sided test fixes that. Also tracks which players have
    already been placed to guarantee no player appears in two slots (a real
    risk once you're allowed to substitute across positions independently).
    Falls back to the plain baseline if the opposing pitcher's hand is
    unknown, or if this team has no usable recent position/hand history."""
    game_lu = _most_recent_complete_lineup(schedule, lineups, team, target_date)
    if game_lu is None:
        return None
    baseline = game_lu[["batting_order", "player_id", "position_code"]].sort_values("batting_order")

    opp_hand = pitcher_hand.get(opposing_pitcher_id)
    if opp_hand is None or pd.isna(opposing_pitcher_id):
        return baseline["player_id"].tolist()

    history = _team_position_hand_history(schedule, lineups, pitcher_hand, team, target_date, window_games)
    if history.empty:
        return baseline["player_id"].tolist()

    counts = history.groupby(["position_code", "player_id", "opp_hand"]).size().reset_index(name="n")
    pivot = counts.pivot_table(index=["position_code", "player_id"], columns="opp_hand", values="n", fill_value=0)
    for h in ("L", "R"):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot["total"] = pivot["L"] + pivot["R"]
    pivot = pivot.reset_index()

    used_players = set()
    result = []

    def _take(pid, pos, pos_pool):
        """Append pid unless he was already drafted into an earlier slot
        (2026-08-03 audit, finding M18: the baseline-retention branches
        appended without checking used_players, so a player platoon-swapped
        into an earlier slot could ALSO keep his own later baseline slot --
        2.56% of real projected lineups contained a duplicate batter, an
        8-man lineup with one player double-drafted). If taken, fall back
        through widening tiers: most-started unused player at this position
        (min_starts pool) -> ANY unused player with history at this position
        -> ANY unused player on the roster (a real 9th distinct batter
        always beats a duplicate) -> only then let the duplicate stand."""
        if pid not in used_players:
            result.append(pid)
            used_players.add(pid)
            return
        for pool in (
            pos_pool[~pos_pool["player_id"].isin(used_players)],
            pivot[(pivot["position_code"] == pos) & (~pivot["player_id"].isin(used_players))],
            pivot[~pivot["player_id"].isin(used_players)],
        ):
            if len(pool):
                alt = int(pool.sort_values("total", ascending=False).iloc[0]["player_id"])
                result.append(alt)
                used_players.add(alt)
                return
        result.append(pid)

    for row in baseline.itertuples():
        pos, current_pid = row.position_code, row.player_id
        pos_pool = pivot[(pivot["position_code"] == pos) & (pivot["total"] >= min_starts)]
        current_row = pos_pool[pos_pool["player_id"] == current_pid]
        current_share = (
            current_row[opp_hand].iloc[0] / current_row["total"].iloc[0] if len(current_row) else None
        )

        if current_share is None or current_share >= 0.35:
            # no evidence the current player is a platoon weak-side against
            # this specific hand at this position -- trust the baseline
            _take(current_pid, pos, pos_pool)
            continue

        alt_pool = pos_pool[(pos_pool["player_id"] != current_pid) & (~pos_pool["player_id"].isin(used_players))].copy()
        alt_pool["share"] = alt_pool[opp_hand] / alt_pool["total"]
        alt_pool = alt_pool[(alt_pool[opp_hand] >= min_starts) & (alt_pool["share"] >= 0.65)]
        if alt_pool.empty:
            _take(current_pid, pos, pos_pool)
            continue

        best_pid = int(alt_pool.sort_values(opp_hand, ascending=False).iloc[0]["player_id"])
        result.append(best_pid)
        used_players.add(best_pid)
    return result
