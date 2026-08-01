"""Walk-forward shooting-volume and shooting-efficiency models: 2PT/3PT/FT
attempts (volume, minutes-exposure) and 2PT/3PT/FT makes (efficiency,
own-attempts-exposure), each modeled independently then recombined --
`projected_made = projected_attempts x shrunk_make_rate`. Volume and
efficiency need separate exposure units (minutes vs. attempts) because a
made-rate computed over 3 attempts is far less confident than one computed
over 300, which "games played" alone doesn't distinguish, exactly the
reasoning in `player_rate_shrinkage.py`'s own docstring.

CONFIRMED EMPIRICALLY (found via a real bootstrap REGRESSION on the full
dev range, not assumed by analogy): attempts and makes need DIFFERENT
smoothing approaches, extending the same volume-vs-skill pattern already
found for minutes (`player_minutes.py`) and steals/blocks
(`player_defensive_event_rates.py`):
- 2PT/3PT ATTEMPTS are a volume/role metric (how much a player shoots
  depends on role, hot streaks, coaching trust) -- like minutes, EWMA
  (recency-weighted) beats expanding-shrinkage. Direct comparison on real
  3PA data: EWMA halflife=10 games (MAE 1.173) beat every expanding-
  shrinkage prior tested (best was 1.198 at prior_minutes=50, and
  expanding-shrinkage got WORSE as its prior increased -- the same
  monotonic-regression pattern `player_minutes.py` found).
- 2PT/3PT MAKE RATE is a stable shooting-skill metric -- like steal/block
  rate, expanding-shrinkage beats EWMA, decisively: on real 3PT-make-rate
  data, expanding-shrinkage (MAE 0.553 at prior_attempts=150, still
  improving with an even larger prior) massively beat every EWMA halflife
  tested (worst-case MAE 0.77, ~40% worse) -- EWMA is actively harmful
  here, not just unhelpful.
- FT IS DIFFERENT FROM 2PT/3PT ON BOTH HALVES, confirmed by a second real
  regression: applying the 2PT/3PT treatment (EWMA attempts halflife=10,
  expanding-shrinkage makes prior=100) to FT produced shrunk MAE=1.0726 vs
  naive MAE=1.0538 (CI (+0.0121,+0.0279), a REAL REGRESSION). Direct
  empirical re-sweep on FT specifically found BOTH halves want expanding-
  shrinkage, not the EWMA/expanding split that works for 2PT/3PT:
  - FTA per-minute: expanding-shrinkage beats every EWMA halflife tested
    (best expanding prior_minutes=20 -> MAE 1.5303; best EWMA halflife=20
    -> MAE 1.5615). Getting to the line is a stable per-minute foul-
    drawing tendency (how a player attacks the rim), not a volume/role
    decision the way shot selection is -- so it behaves like a skill
    metric, not like 2PT/3PT attempts.
  - FT make rate: expanding-shrinkage prior_attempts=15 (MAE 0.3724) beats
    the previous prior=100 (MAE 0.3778) and the naive floor (MAE 0.3773) --
    FT-shooting skill stabilizes over a much SMALLER attempt sample than
    2PT/3PT accuracy does, so the old shared prior=100 was over-shrinking.
  This is exactly the kind of per-category empirical check this project's
  discipline requires -- never assume a smoothing family transfers by
  analogy across categories, even superficially similar ones (FT "attempts"
  sounds like 2PT/3PT "attempts" but is driven by a different mechanism:
  literal fouls drawn, not shot selection).

Uses `player_minutes.build_player_game_log`'s shared columns (`fgMade2`/
`fgAtt2`, `fg3Made`/`fg3Att`, `ftMade`/`ftAtt`) -- already attached from the
cached traditional box score, no new ingest needed for this file.
"""

import numpy as np
import pandas as pd

from src.models.player_rate_shrinkage import add_walk_forward_player_mean_ewm, add_walk_forward_player_rate

# 2PT/3PT attempts: EWMA halflife (games), confirmed empirically (see module docstring) -- NOT
# the same tuning as player_minutes.py's halflife=2 (that comparison was on raw minutes; this one
# is specifically shot-attempt volume, which happened to prefer a slightly longer halflife=10).
ATTEMPT_HALFLIFE_GAMES = 10.0

# 2PT/3PT makes: expanding-shrinkage prior, in OWN ATTEMPTS -- placeholders in the ballpark
# confirmed empirically to help (larger priors kept improving in the tested range; not
# exhaustively optimized beyond confirming the family choice itself). See MODEL_DOCUMENTATION.md.
PRIOR_ATTEMPTS_2PM = 150.0
PRIOR_ATTEMPTS_3PM = 150.0

# FT is smoothed with a DIFFERENT family on BOTH halves than 2PT/3PT (see module docstring) --
# expanding-shrinkage for attempts too, keyed on minutes exposure like a rate stat, plus its own
# (much smaller) make-rate prior. Both values are the empirically-swept optimum, not placeholders.
PRIOR_MINUTES_FTA = 20.0
PRIOR_ATTEMPTS_FTM = 15.0

# NOT ADOPTED: `add_walk_forward_player_rate`'s new `league_avg_halflife_games` option (Sec16
# props-Phase-4 fast-follow) looked like a real improvement for FT make-rate on a recent-dev-only
# chronological slice (MAE 0.2939 vs 0.2943, CI excluded zero) -- but did NOT hold up at FULL
# dev-range scale: re-validating there showed FT makes REGRESS from a real improvement (shrunk
# 1.0425 vs naive 1.0434) to NOISE (shrunk 1.0430 vs naive 1.0434, CI includes zero). A caught,
# reverted mistake, not a silent one -- exactly why every candidate fix gets a full-dev-range
# re-check before adoption, not just the narrower recent-slice signal that motivated trying it.

_EWMA_CATEGORIES = (
    # (label, attempt_col, make_col, prior_attempts_for_make_rate)
    ("2pt", "fgAtt2", "fgMade2", PRIOR_ATTEMPTS_2PM),
    ("3pt", "fg3Att", "fg3Made", PRIOR_ATTEMPTS_3PM),
)


def add_scoring_rates(log: pd.DataFrame) -> pd.DataFrame:
    """log must have `player_minutes.build_player_game_log`'s columns
    (minutes, fgMade2/fgAtt2, fg3Made/fg3Att, ftMade/ftAtt). Adds, per
    category ("2pt"/"3pt"/"ft"): f"{cat}_att_rate_per_min" (2pt/3pt: EWMA-
    recency-weighted attempts per minute; ft: expanding-shrunk attempts per
    minute -- see module docstring), f"{cat}_make_rate" (expanding-shrunk
    make rate per own attempt, all three categories), and
    f"{cat}_proj_att"/f"{cat}_proj_made" using THIS row's own realized
    minutes (a backtest convenience -- a live caller projecting a FUTURE
    game should multiply `{cat}_att_rate_per_min` by that game's PROJECTED
    minutes from `player_minutes.py` instead, not this row's own played
    minutes)."""
    log = log.copy()
    for label, att_col, make_col, prior_attempts in _EWMA_CATEGORIES:
        log[f"_{label}_att_per_min_raw"] = log[att_col] / log["minutes"].replace(0, np.nan)
        log = add_walk_forward_player_mean_ewm(log, f"_{label}_att_per_min_raw", ATTEMPT_HALFLIFE_GAMES, prefix=f"{label}_att")
        log = log.rename(columns={f"{label}_att_ewm_rate": f"{label}_att_rate_per_min"})

        log = add_walk_forward_player_rate(log, make_col, att_col, prior_attempts, prefix=f"{label}_make")
        log = log.rename(columns={f"{label}_make_shrunk_rate": f"{label}_make_rate"})

        log[f"{label}_proj_att"] = log[f"{label}_att_rate_per_min"] * log["minutes"]
        log[f"{label}_proj_made"] = log[f"{label}_proj_att"] * log[f"{label}_make_rate"]
        log = log.drop(columns=[f"_{label}_att_per_min_raw"])

    log = add_walk_forward_player_rate(log, "ftAtt", "minutes", PRIOR_MINUTES_FTA, prefix="ft_att")
    log = log.rename(columns={"ft_att_shrunk_rate": "ft_att_rate_per_min"})

    log = add_walk_forward_player_rate(log, "ftMade", "ftAtt", PRIOR_ATTEMPTS_FTM, prefix="ft_make")
    log = log.rename(columns={"ft_make_shrunk_rate": "ft_make_rate"})

    log["ft_proj_att"] = log["ft_att_rate_per_min"] * log["minutes"]
    log["ft_proj_made"] = log["ft_proj_att"] * log["ft_make_rate"]
    return log


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season
    from src.models.player_minutes import build_player_game_log

    current = current_nba_season()
    log = build_player_game_log(FIRST_DEV_SEASON, current)
    rated = add_scoring_rates(log)
    sample = rated.dropna(subset=["3pt_make_rate"]).tail(10)
    print(sample[["gameId", "playerId", "fg3Att", "fg3Made", "3pt_att_rate_per_min", "3pt_make_rate", "3pt_proj_made"]].to_string(), flush=True)
