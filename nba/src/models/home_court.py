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


def fit_team_home_court_walk_forward(games: pd.DataFrame, prior_games: float = 200.0,
                                      halflife_games: float = 400.0) -> pd.Series:
    """Aligned-to-`games`-index Series: for each game (sorted by date), a
    PER-TEAM (not league-wide) trailing home-court multiplier for that
    game's home team -- shrunk toward the trailing LEAGUE-WIDE walk-forward
    multiplier (`fit_home_court_walk_forward`'s own log-mult computation) at
    `prior_games` strength, same games-count-weighted blend shape as
    `shrinkage.add_walk_forward_mean`.

    Deliberately NOT season-reset (unlike every team-rating lever in
    `shrinkage.py`): home-court advantage is hypothesized to be a property
    of the VENUE (crowd noise, altitude -- e.g. Denver's documented
    altitude edge, travel fatigue imposed on visitors) rather than of a
    given season's roster, so a team's own `home_log_ratio` history
    accumulates across ALL its games in `games` regardless of season -- a
    stadium's altitude doesn't reset every October the way a roster does.
    Only that team's own HOME-side rows inform its own estimate: an away
    game's `away_log_ratio` is evidence about the extent of the HOST's home
    boost that night, not this (visiting) team's own home-court property
    (see `_baseline_log_ratios` -- home_log_ratio/away_log_ratio are two
    sides of the SAME single per-game home-boost estimate).

    MOTIVATION (2026-08-06, Fable 5 critique item 1f): tests whether
    home-court advantage varies meaningfully by team -- never previously
    tested, unlike the home-court EWMA `halflife_games` itself (tested
    earlier this session and found not to matter). See
    MODEL_DOCUMENTATION.md for the holdout result once read."""
    g = games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    g = _baseline_log_ratios(g)

    league_combined = (g["home_log_ratio"] - g["away_log_ratio"]) / 2.0
    league_trailing_log_mult = league_combined.shift(1).ewm(halflife=halflife_games, min_periods=30).mean()

    team_games_before = g.groupby("team_home").cumcount()
    team_home_ewma = g.groupby("team_home")["home_log_ratio"].transform(
        lambda s: s.shift(1).ewm(halflife=halflife_games, min_periods=1).mean()).fillna(0.0)

    shrunk_log_mult = ((team_games_before * team_home_ewma + prior_games * league_trailing_log_mult)
                        / (team_games_before + prior_games))
    return np.exp(shrunk_log_mult)


DENVER_TEAM_ID = 1610612743  # nba_api's static team_codes.ABBREV_TO_TEAM_ID["DEN"]


def fit_denver_specific_home_court_walk_forward(games: pd.DataFrame, prior_games: float = 100.0,
                                                 halflife_games: float = 400.0) -> pd.Series:
    """Task #67 (external-review follow-up, MODEL_DOCUMENTATION.md Sec49):
    Sec49's GENERAL per-team home-court sweep (`fit_team_home_court_walk_forward`
    applied uniformly to all 30 teams) showed no real effect at any
    shrinkage strength -- but real, peer-reviewed research (Lopez, Matthews
    & Baumer, *Annals of Applied Statistics*, 11 seasons / 13,290 games of
    betting-line data with hierarchical partial pooling) confirms Denver's
    altitude edge IS a real outlier, larger than any other team's. That
    study needed a decade+ of lower-noise market data and pooling across
    all 30 teams just to isolate it -- Sec49's own sweep, diluted 1-in-30
    across mostly-null teams on a 3-season evaluation slice, plausibly
    lacked the power to detect a REAL single-team effect, not evidence
    the effect doesn't exist.

    This tests the better-powered version directly: apply
    `fit_team_home_court_walk_forward`'s own per-team estimate ONLY to
    Denver's home games (concentrating all statistical power on the one
    team with real prior evidence), while every other team keeps using
    the existing plain league-wide `fit_home_court_walk_forward` value
    completely unchanged -- not a diluted 30-team sweep."""
    league_wide = fit_home_court_walk_forward(games, halflife_games=halflife_games)
    denver_specific = fit_team_home_court_walk_forward(games, prior_games=prior_games, halflife_games=halflife_games)
    is_denver_home = games["team_home"] == DENVER_TEAM_ID
    return league_wide.where(~is_denver_home, denver_specific)


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
