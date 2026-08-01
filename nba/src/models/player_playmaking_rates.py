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

FINDING (2026-07-25, confirmed at full dev-range scale): the family
(expanding-shrinkage) held up for both AST and TOV, but the initial
`PRIOR_TOUCHES_AST=300` (copy-pasted from TOV) was itself a REAL REGRESSION
(MAE 0.8965 vs naive 0.8933) even though TOV's identical prior=300 was a
real improvement -- a direct sweep on AST specifically found
prior_touches=50 clearly beats naive. Re-confirmed on the full dev range
(275,138 rows): real improvement, MAE 0.8880 vs naive 0.8933.

**Real fix (2026-08-01, Sec14/15 props-Phase-4 fast-follow)**: the props
subsystem's first-ever holdout check found AST carrying a REAL dev-vs-
holdout MAE gap. Re-swept `PRIOR_TOUCHES_AST` using a chronological 80/20
split WITHIN dev only (never touching real holdout for this decision, per
the confirmatory-veto protocol) -- prior=100 (MAE 0.9401 on the recent-dev
eval slice) beat the original prior=50 (MAE 0.9413), confirmed via
bootstrap on that slice (CI excludes zero). NOTE this is the OPPOSITE
direction from the original hypothesis (a lower prior for a "continuing
rise") written in Sec14's initial diagnosis -- checked empirically rather
than forced to match the earlier guess, and the data said larger, not
smaller. Raised `PRIOR_TOUCHES_AST` to 100; re-validated on the full dev
range and re-checked against real holdout (a genuinely new configuration,
its own one-time confirmatory read) -- see MODEL_DOCUMENTATION.md Sec15.
"""

import pandas as pd

from src.models.player_rate_shrinkage import add_walk_forward_player_rate

PRIOR_TOUCHES_AST = 100.0  # raised from 50 -- see Sec14/15 fast-follow finding above
PRIOR_TOUCHES_TOV = 300.0  # unchanged -- TOV showed no real dev/holdout gap (Sec14)


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
