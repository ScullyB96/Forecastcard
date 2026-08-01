"""Walk-forward rate models for steals and blocks -- both rare, discrete,
per-minute events, but confirmed EMPIRICALLY to need DIFFERENT smoothing
families (not assumed to share one just because they're both "defensive
events" -- this project's standing discipline, see `player_scoring_rates.py`
for the same lesson learned twice on 2PT/3PT vs FT).

STEALS: uses the expanding-shrinkage primitive (`add_walk_forward_player_rate`
with minutes exposure), the Bayes-optimal posterior mean for a Poisson rate
under a Gamma prior (`posterior_rate = (prior_pseudo_minutes*prior_mean +
cumsum_events_before) / (prior_pseudo_minutes + cumsum_minutes_before)` --
exactly the shared primitive's formula). Confirmed empirically: steal rate
per minute is a persistent individual SKILL (a lurking/anticipation habit
that doesn't depend much on matchup) -- infinite-memory expanding-shrinkage
BEATS every EWMA halflife tested (MAE ~0.657-0.660 across prior_minutes
50-1000, vs. ~0.668-0.675 for EWMA halflives 5-40).

BLOCKS: uses EWMA instead (`add_walk_forward_player_mean_ewm`), the OPPOSITE
conclusion from steals. A first version reusing steals' expanding-shrinkage
(prior_minutes=200) was a REAL REGRESSION on the full dev range (shrunk MAE
0.4224 vs naive MAE 0.4102, CI (+0.0117,+0.0128)). Direct empirical re-sweep
on blocks specifically found every expanding-shrinkage prior tested (1-1500)
was flat-to-worse than the naive floor (best was prior->0, degenerating to
naive itself, MAE 0.4952), while EWMA halflife=10-12 games clearly won (MAE
0.4938). This makes basketball sense even though it's the opposite of
steals: block volume is heavily driven by matchup and rim-protection role
(which players you're guarding, recent minutes at center) -- a volume/role
metric like shot attempts or minutes, not a stable gambling-style skill like
steals. No separate Gamma-Poisson implementation needed for the steals
point-estimate; a genuine Poisson/NegBin distribution is only needed
downstream in `prop_distribution.py` for the final predictive spread.

**TWO successive fix attempts NOT ADOPTED, both reverted (2026-08-01, Sec16/17)**:
steals carried a REAL dev-vs-holdout MAE gap (Sec14).
(1) `league_avg_halflife_games=300` (Sec16) passed two dev-only checks but
made holdout WORSE when checked (0.5554 -> 0.5568).
(2) `add_era_adjusted_player_rate`'s detrend-then-retrend architecture
(Sec17) was validated even more rigorously (consistent gains across FOUR
independent dev-internal cutoffs, full-dev-range, and vs-naive) -- and
still made holdout WORSE, and by MORE (0.5554 -> 0.5630, a larger
regression than attempt (1)).

**Both reverted. This MONOTONIC pattern -- each progressively more
sophisticated trend-extrapolation attempt made real holdout performance
WORSE, not better -- is the decisive finding, not a modeling failure to
keep chasing.** It strongly indicates the 2024-2025 steal-rate shift
(Sec14: 0.697 in 2023 -> 0.767 in 2024 -> 0.777 in 2025) is a genuine
REGIME CHANGE, not a smooth continuing trend -- and NO historical
extrapolation technique, however adaptive, can predict a level shift from
data that predates it entirely. Stopping further attempts at this specific
gap via trend-extrapolation. Steals uses the ORIGINAL, simplest, most-
validated configuration (flat-cumulative expanding-shrinkage, prior=200)."""

import numpy as np
import pandas as pd

from src.models.player_rate_shrinkage import add_walk_forward_player_mean_ewm, add_walk_forward_player_rate

PRIOR_MINUTES_STL = 200.0  # placeholder within the empirically-flat 50-1000 range tested
BLOCK_HALFLIFE_GAMES = 10.0  # empirically-swept optimum (10-12 games flat-best), see module docstring


def add_defensive_event_rates(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `player_minutes.build_player_game_log`'s columns
    (minutes, steals, blocks). Adds `stl_rate_per_min` (expanding-shrunk),
    `blk_rate_per_min` (EWMA -- see module docstring), and `stl_proj`/
    `blk_proj` (this row's own realized minutes x rate -- a backtest
    convenience; a live caller projecting a FUTURE game should multiply by
    that game's PROJECTED minutes instead)."""
    log = log.copy()
    log = add_walk_forward_player_rate(log, "steals", "minutes", PRIOR_MINUTES_STL, prefix="stl")
    log = log.rename(columns={"stl_shrunk_rate": "stl_rate_per_min"})

    log["_blk_per_min_raw"] = log["blocks"] / log["minutes"].replace(0, np.nan)
    log = add_walk_forward_player_mean_ewm(log, "_blk_per_min_raw", BLOCK_HALFLIFE_GAMES, prefix="blk")
    log = log.rename(columns={"blk_ewm_rate": "blk_rate_per_min"}).drop(columns=["_blk_per_min_raw"])

    log["stl_proj"] = log["stl_rate_per_min"] * log["minutes"]
    log["blk_proj"] = log["blk_rate_per_min"] * log["minutes"]
    return log


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.player_minutes import build_player_game_log

    current = current_nba_season()
    log = build_player_game_log(FIRST_DEV_SEASON, current)
    rated = add_defensive_event_rates(log)
    sample = rated.dropna(subset=["stl_rate_per_min"]).tail(10)
    print(sample[["gameId", "playerId", "steals", "stl_rate_per_min", "stl_proj",
                  "blocks", "blk_rate_per_min", "blk_proj"]].to_string(), flush=True)
