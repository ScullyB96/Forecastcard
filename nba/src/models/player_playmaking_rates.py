"""Walk-forward assist/turnover rate models, exposed per TOUCH (from
`BoxScorePlayerTrackV3`) rather than per minute -- a materially better
usage/playmaking-opportunity proxy: two players logging the same minutes
can have very different ball-handling touches depending on role (a
stationary corner shooter vs. a primary ball-handler), and assists/
turnovers are fundamentally events that happen ON a touch, not per minute
of standing around.

Expanding-shrinkage (not EWMA) by default, same reasoning as
`player_rebounding_rates.py` -- a per-touch playmaking rate is treated as
closer to a persistent skill/role than a volatile night-to-night decision;
`validate_player_playmaking_rates.py` checks this assumption against a
naive floor rather than asserting it.

PRELIMINARY FINDING (2026-07-25, single-season smoke test on 2015-16 only --
`BoxScorePlayerTrackV3`'s full backfill isn't done yet, re-validate on the
full dev range once it is): the family (expanding-shrinkage) held up for
both AST and TOV, but the initial `PRIOR_TOUCHES_AST=300` (copy-pasted from
TOV) was itself a REAL REGRESSION (MAE 0.8965 vs naive 0.8933) even though
TOV's identical prior=300 was a real improvement -- a direct sweep on AST
specifically found prior_touches=50 (MAE 0.8880) clearly beats naive, so
assists apparently stabilize over a smaller touch sample than turnovers do.
Lowered `PRIOR_TOUCHES_AST` to 50; `PRIOR_TOUCHES_TOV` unchanged at 300
(already confirmed a real improvement there).
"""

import pandas as pd

from src.models.player_rate_shrinkage import add_walk_forward_player_rate

PRIOR_TOUCHES_AST = 50.0  # see PRELIMINARY FINDING above -- re-check on the full dev range
PRIOR_TOUCHES_TOV = 300.0


def add_playmaking_rates(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `player_minutes.attach_player_track`'s columns
    (assists/turnovers from the base box score, touches from player-track).
    Adds `ast_rate`/`tov_rate` (shrunk per-touch rates) and `ast_proj`/
    `tov_proj` (this row's own realized touches x rate -- a backtest
    convenience; a live caller needs a projected-touches model for a
    future game, not built here)."""
    log = log.copy()
    log = add_walk_forward_player_rate(log, "assists", "touches", PRIOR_TOUCHES_AST, prefix="ast")
    log = log.rename(columns={"ast_shrunk_rate": "ast_rate"})
    log = add_walk_forward_player_rate(log, "turnovers", "touches", PRIOR_TOUCHES_TOV, prefix="tov")
    log = log.rename(columns={"tov_shrunk_rate": "tov_rate"})

    log["ast_proj"] = log["ast_rate"] * log["touches"]
    log["tov_proj"] = log["tov_rate"] * log["touches"]
    return log


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.player_minutes import attach_player_track, build_player_game_log

    current = current_nba_season()
    log = build_player_game_log(FIRST_DEV_SEASON, current)
    log = attach_player_track(log, FIRST_DEV_SEASON, current)
    have_track = log["touches"].notna().sum()
    print(f"{have_track}/{len(log)} rows have player-track data so far (backfill in progress)", flush=True)

    rated = add_playmaking_rates(log)
    sample = rated.dropna(subset=["ast_rate"]).tail(10)
    print(sample[["gameId", "playerId", "assists", "ast_rate", "ast_proj",
                  "turnovers", "tov_rate", "tov_proj"]].to_string(), flush=True)
