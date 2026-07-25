"""Empirically-fit home-court advantage, expressed as a multiplicative
factor on projected rating (see `team_strength.project_game`) rather than
assumed as a fixed constant plucked from folklore ("home team +3").

Fit as: for every game, the log-ratio of actual-to-baseline rating on each
side should equal +log(mult) for the home side and -log(mult) for the away
side (baseline = the pace x rating combine with mult forced to 1). Averaging
the home-side log-ratio and the negative of the away-side log-ratio (both
independent estimates of the same log(mult)) uses the whole game's
information, not just one side of it.

Checked for temporal drift (not assumed stable across 2015-2026 -- NBA home
court advantage is publicly documented as having compressed over the last
decade) via `fit_home_court_by_season`, printed as a diagnostic in
`validate_team_strength_baseline.py`. If real drift is confirmed, use
`fit_home_court_walk_forward` (a trailing EWMA over strictly-prior games) in
production instead of one static constant fit over the whole dev range.
"""

import numpy as np
import pandas as pd


def _baseline_log_ratios(games: pd.DataFrame) -> pd.DataFrame:
    """games: one row per real game, using the SAME walk-forward PRE-GAME
    columns `team_strength.project_game` itself consumes: pace_shrunk_mean_home,
    pace_shrunk_mean_away, league_avg_pace, rtg_attack_rate_home,
    rtg_defense_rate_home, rtg_attack_rate_away, rtg_defense_rate_away,
    league_avg_rtg, actual_home_score, actual_away_score. Returns the same
    rows with home_log_ratio/away_log_ratio added (mult-free baseline vs.
    actual, in rating-space log units).

    BUG FOUND AND FIXED (2026-07-24), confirmed real on live data before
    fixing: the original version read `home_pace`/`away_pace`/`home_oRtg`/
    `home_dRtg`/`away_oRtg`/`away_dRtg` -- column names that DO exist on the
    wide per-game frame built by `validate_team_strength_baseline._to_wide_games`,
    but are that SPECIFIC GAME's own RAW REALIZED box-score rating (renamed
    straight from `offRtg_home`/`defRtg_home`), not a walk-forward PRE-game
    prediction. That made "actual_home_rtg" and "base_home_rtg" both
    circularly derived from the same game's own real outcome, rather than
    comparing a real outcome against an honest pre-game baseline. Confirmed
    on real 2015-16..2017-18 data: this produced a home-court multiplier
    BELOW 1.0 in every one of 3 seasons (0.987/0.985/0.990) despite that
    same data showing a completely normal, textbook real home-court edge
    (home teams averaged 106.2 vs. away teams' 103.5 points, 58.4% home win
    rate) -- an unambiguous sign that the fitting function itself was
    broken, not that home-court advantage was somehow negative. Fixed by
    reading the correct walk-forward columns."""
    g = games.copy()
    proj_pace = (g["league_avg_pace"] * (g["pace_shrunk_mean_home"] / g["league_avg_pace"])
                 * (g["pace_shrunk_mean_away"] / g["league_avg_pace"]))
    base_home_rtg = (g["league_avg_rtg"] * (g["rtg_attack_rate_home"] / g["league_avg_rtg"])
                     * (g["rtg_defense_rate_away"] / g["league_avg_rtg"]))
    base_away_rtg = (g["league_avg_rtg"] * (g["rtg_attack_rate_away"] / g["league_avg_rtg"])
                     * (g["rtg_defense_rate_home"] / g["league_avg_rtg"]))

    actual_home_rtg = 100.0 * g["actual_home_score"] / proj_pace
    actual_away_rtg = 100.0 * g["actual_away_score"] / proj_pace

    g["home_log_ratio"] = np.log(actual_home_rtg / base_home_rtg)
    g["away_log_ratio"] = np.log(actual_away_rtg / base_away_rtg)
    return g


def fit_home_court_multiplier(games: pd.DataFrame) -> float:
    """Single static multiplier fit over every row of `games` (dev-set-wide
    fit -- NOT walk-forward-safe on its own; only use this for the drift
    diagnostic / as a sanity-check floor, not as a live per-game feature)."""
    g = _baseline_log_ratios(games)
    log_mult = (g["home_log_ratio"].mean() - g["away_log_ratio"].mean()) / 2.0
    return float(np.exp(log_mult))


def fit_home_court_by_season(games: pd.DataFrame) -> pd.Series:
    """One static multiplier per season -- the temporal-drift diagnostic.
    Each season's fit is independent, so this is safe to run over the full
    dev range purely for diagnosis (it is not itself walk-forward)."""
    g = _baseline_log_ratios(games)
    per_season = g.groupby("season").apply(
        lambda s: float(np.exp((s["home_log_ratio"].mean() - s["away_log_ratio"].mean()) / 2.0)),
        include_groups=False,
    )
    return per_season


def fit_home_court_walk_forward(games: pd.DataFrame, halflife_games: float = 400.0) -> pd.Series:
    """Aligned-to-`games`-index Series: for each game (sorted by date), the
    trailing EWMA-fit home-court multiplier using ONLY strictly-prior games
    league-wide -- walk-forward-safe, for use as an actual per-game feature
    if `fit_home_court_by_season` confirms real drift. `halflife_games`
    (games, not calendar days -- roughly 5 seasons at the default) is an
    un-calibrated placeholder, same standard as every other stabilization
    constant in this project."""
    g = games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    g = _baseline_log_ratios(g)
    combined = (g["home_log_ratio"] - g["away_log_ratio"]) / 2.0
    trailing_log_mult = combined.shift(1).ewm(halflife=halflife_games, min_periods=30).mean()
    return np.exp(trailing_log_mult)
