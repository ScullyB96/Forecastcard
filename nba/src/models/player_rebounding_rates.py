"""Walk-forward offensive/defensive rebounding-CONVERSION rate models --
rebounds grabbed per rebound CHANCE (opportunity), not per minute or per
game. `reboundChancesOffensive`/`reboundChancesDefensive` (from
`BoxScorePlayerTrackV3`, confirmed full-range coverage, no boundary issue)
is a much better exposure denominator than minutes: two players who play
the same minutes can have wildly different rebound OPPORTUNITY depending
on their team's shot profile and the game's pace, and a rebounding-rate
model should isolate the player's own conversion skill from that.

Uses expanding-shrinkage (not EWMA), following the same reasoning already
confirmed for shooting rates and steal/block rates in this codebase: a
rebound-CHANCE-conversion rate is a persistent physical/positional skill,
not a volatile coaching decision the way raw minutes are -- this is a
testable assumption, not asserted as fact, and `validate_player_rebounding_rates.py`
checks it against a naive floor the same way every other category here
does, once `BoxScorePlayerTrackV3`'s backfill is complete enough to validate on.
"""

import pandas as pd

from src.models.player_rate_shrinkage import add_walk_forward_player_rate

PRIOR_CHANCES_OREB = 100.0  # placeholder, un-calibrated -- see MODEL_DOCUMENTATION.md
PRIOR_CHANCES_DREB = 150.0


def add_rebounding_rates(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `player_minutes.attach_player_track`'s columns
    (oreb/dreb from the base box score, reboundChancesOffensive/Defensive
    from player-track). Rows missing player-track data (not yet backfilled,
    or a genuine API gap) get NaN rates, not a silent zero or a crash --
    callers should dropna on what they need. Adds `oreb_rate`/`dreb_rate`
    (shrunk conversion rate per chance) and `oreb_proj`/`dreb_proj` (this
    row's own realized chances x rate -- a backtest convenience; a live
    caller needs a projected-chances model for a future game, not built
    here)."""
    log = log.copy()
    log = add_walk_forward_player_rate(log, "oreb", "reboundChancesOffensive", PRIOR_CHANCES_OREB, prefix="oreb")
    log = log.rename(columns={"oreb_shrunk_rate": "oreb_rate"})
    log = add_walk_forward_player_rate(log, "dreb", "reboundChancesDefensive", PRIOR_CHANCES_DREB, prefix="dreb")
    log = log.rename(columns={"dreb_shrunk_rate": "dreb_rate"})

    log["oreb_proj"] = log["oreb_rate"] * log["reboundChancesOffensive"]
    log["dreb_proj"] = log["dreb_rate"] * log["reboundChancesDefensive"]
    return log


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.player_minutes import attach_player_track, build_player_game_log

    current = current_nba_season()
    log = build_player_game_log(FIRST_DEV_SEASON, current)
    log = attach_player_track(log, FIRST_DEV_SEASON, current)
    have_track = log["reboundChancesOffensive"].notna().sum()
    print(f"{have_track}/{len(log)} rows have player-track data so far (backfill in progress)", flush=True)

    rated = add_rebounding_rates(log)
    sample = rated.dropna(subset=["dreb_rate"]).tail(10)
    print(sample[["gameId", "playerId", "oreb", "oreb_rate", "oreb_proj",
                  "dreb", "dreb_rate", "dreb_proj"]].to_string(), flush=True)
