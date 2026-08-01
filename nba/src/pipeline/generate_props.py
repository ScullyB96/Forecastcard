"""Live daily player-props entry point: reuses `generate_predictions.py`'s
already-RAPM-adjusted team point totals as the macro anchor (a single
source of truth -- props and game predictions can never silently disagree
on a team's total), resolves tonight's active roster via the SHARED
`active_roster.py` helpers, projects each active player's full stat line
from every rate-category model built this session, composes points via
`usage_allocation.py`'s macro-anchor + micro-reallocation rule, attaches a
full predictive distribution per `prop_distribution.py` (Task #26 -- see
that module's `CATEGORY_FAMILY` docstring: every category is Poisson or
Negative-Binomial, refit fresh from historical data every call), and
writes one row per (game, player, stat) with `proj_mean`/`proj_var`/
`family`/`family_param` -- enough for any downstream caller to compute
`prop_distribution.over_under_prob(line, proj_mean, family,
json.loads(family_param))` for an arbitrary betting line.

Run as `python -m src.pipeline.generate_props [YYYY-MM-DD]` (defaults to
today).

**Known v1 simplifications, documented rather than silently baked in:**
- `TEAM_MINUTES_PER_GAME = 240.0` (5 players x 48 minutes) ignores
  overtime -- a live projection can't know in advance a game will go to
  OT, so this is the correct expectation for a PRE-game projection, not an
  approximation error.
- REB/AST/TOV/STL/BLK need PROJECTED rebound-chances/touches for tonight,
  which no dedicated model exists for. Approximated here via each player's
  own trailing chances-per-minute / touches-per-minute (expanding-
  shrinkage, exposure = minutes) x their projected minutes -- a
  low-leverage UNIT CONVERSION step, not a second validated rate model;
  the category's own already-validated conversion-rate model
  (`player_rebounding_rates.py`/`player_playmaking_rates.py`) still does
  the actual prediction.
- DREB/AST/TOV/STL/BLK are ANCHORED (Task #24) to `team_stat_rates.py`'s
  team-level walk-forward total via the same macro-anchor +
  micro-reallocation rule as points (see `usage_allocation.py`'s module
  docstring and MODEL_DOCUMENTATION.md Sec23). OREB is the one category
  that failed the confirmatory holdout check in absolute terms (not just
  a widening gap) and stays UNANCHORED -- each player's own `oreb_proj`
  is reported directly, no team-total rescaling.
- Matchup difficulty is applied to POINTS and AST/TOV, all three via the
  FULL 3-level hierarchy (defender-specific -> position-group-vs-team ->
  league floor, Sec13/18/19) -- gated on `season >= MATCHUP_DATA_START_SEASON`
  and `MIN_MATCHUP_MINUTES_TO_TRUST`; an offensive player whose position
  group can't be resolved (no roster data) gets no adjustment
  (`matchup_adjusted: False`), not a crash or a silent zero passed off as
  "checked and found neutral". The matchup delta is applied as a direct
  additive nudge to the player's own raw projection BEFORE that category's
  own anchoring step (clipped at 0 for AST/TOV) -- points' and AST/TOV's
  matchup deltas are both pre-anchor reshaping, not a second, independent
  adjustment layered on top of anchoring; AST/TOV are themselves ANCHORED
  (Task #24, see `usage_allocation.py`'s module docstring and
  MODEL_DOCUMENTATION.md Sec23) via the same team-total mechanism as
  points, just against `team_stat_rates.py`'s total instead of RAPM's.
"""

import json
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report, warn_if_stale_for_backtest
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_for_date
from src.models.matchup_difficulty import (
    MATCHUP_DATA_START_SEASON,
    MIN_MATCHUP_MINUTES_TO_TRUST,
    POSGROUP_HALFLIFE_GAMES_AST,
    POSGROUP_PRIOR_MATCHUP_MINUTES_TOV,
    PRIOR_MATCHUP_MINUTES_AST,
    PRIOR_MATCHUP_MINUTES_TOV,
    add_defender_difficulty_rate,
    add_defender_stat_difficulty_rate,
    add_position_group_difficulty_rate,
    add_position_group_stat_difficulty_rate,
    add_position_group_stat_difficulty_rate_expanding,
    build_defender_game_log,
    build_defender_position_group_minutes,
    build_position_group_matchup_log,
    defender_position_group_minutes_asof,
    defender_total_minutes_asof,
    load_roster_position_lookup,
    opponent_defense_adjustment,
)
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_minutes import attach_player_track, build_player_game_log
from src.models.player_playmaking_rates import add_playmaking_rates
from src.models.player_rate_shrinkage import add_walk_forward_player_rate
from src.models.player_rebounding_rates import add_rebounding_rates
from src.models.player_scoring_rates import add_scoring_rates
from src.models.prop_distribution import CATEGORY_FAMILY, fit_continuous_family, fit_count_family
from src.models.team_stat_rates import ADOPTED_CATEGORIES, add_team_stat_ratings, build_team_stat_game_log, project_team_stat
from src.models.team_strength import add_team_ratings, build_team_game_log
from src.models.usage_allocation import allocate_team_total, compute_usage_shares, matchup_point_delta, raw_projected_points
from src.pipeline.active_roster import build_team_history, games_on_date, load_current_roster_player_ids, resolve_active_lineup
from src.pipeline.generate_predictions import _fit_latest_player_ratings, run as generate_game_predictions
from src.pipeline.refresh_data import refresh_all_data
from src.utils.paths import DATA_PROCESSED

TEAM_MINUTES_PER_GAME = 240.0  # 5 players x 48 minutes; OT ignored for a PRE-game projection (see module docstring)
PRIOR_MINUTES_EXPOSURE_UNIT = 100.0  # expanding-shrinkage prior for the chances/touches-per-minute unit-conversion rates
POSITION_GROUPS = ("G", "F", "C")

STAT_COLUMNS = ("2pt_proj_made", "3pt_proj_made", "ft_proj_made",
                "oreb_proj", "dreb_proj", "ast_proj", "tov_proj", "stl_proj", "blk_proj")


def _latest_snapshot(log: pd.DataFrame, cols: list, group_col: str = "playerId") -> pd.DataFrame:
    """Each player's most recent NON-NULL value per column -- same "as of
    before that row's own game" convention as
    `generate_predictions._latest_team_ratings`, but taking the last REAL
    value per column independently rather than literally whichever row is
    chronologically last (see fix below for why that distinction matters).

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): EWMA-based
    columns (minutes, 2PT/3PT attempt-rate, block-rate --
    `add_walk_forward_player_mean_ewm`) reset to NaN on every player's
    FIRST game of every season BY DESIGN (a season-boundary reset, not
    just a rookie's literal debut) -- confirmed directly: the sibling
    expanding-shrinkage primitive (`add_walk_forward_player_rate`, used
    for makes/OREB/DREB/AST/TOV) instead falls back to a real league-
    average number at zero prior exposure, never NaN, so this asymmetry is
    specific to the EWMA family. A plain `.last()` here picked up that
    NaN row as the "latest" snapshot for a player's SECOND game of a new
    season -- even though a real, non-NaN value from late LAST season was
    sitting in an earlier row of this exact log. Because
    `generate_props.py`'s downstream `fillna(0.0)` is explicitly meant for
    a genuine call-up with NO history at all, this silently zeroed real,
    established veterans' points/2PT/3PT/block props (reallocating their
    share to teammates) every single season, leaguewide, during the week
    after opening night -- a gap none of this project's own spot-check
    dates happened to land in. Forward-filling before taking the last row
    carries forward last season's real estimate across the reset instead,
    while a genuine rookie with no history at all still correctly comes
    out NaN (there's nothing to forward-fill from)."""
    log = log.sort_values("gameDate")
    return log.groupby(group_col)[cols].apply(lambda g: g.ffill().iloc[-1])


def _before(log: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """Filters a log to `gameDate < game_date` -- applied to every log
    BEFORE `_latest_snapshot` extracts "the most recent row per group".
    Without this, "most recent" silently means "most recent as of the real
    wall-clock today", not "as of `game_date`" -- a real bug found via a
    2025-01-15 spot-check (a star player's own trailing-minutes projection
    reached into a much later, unrelated season and came out ~11 minutes
    instead of his true ~38-minute trailing average). See
    MODEL_DOCUMENTATION.md for the full writeup. Harmless for a genuine
    live "tonight" call -- there are no cached rows on or after today
    anyway, so this is a no-op in that case."""
    return log[pd.to_datetime(log["gameDate"]) < pd.Timestamp(game_date)]


def _project_active_roster_stats(active_player_ids: list, projected_minutes: pd.Series,
                                  scoring_log: pd.DataFrame, rebounding_log: pd.DataFrame,
                                  playmaking_log: pd.DataFrame, defensive_log: pd.DataFrame) -> pd.DataFrame:
    """Builds each active player's raw stat-line projection (pre-team-
    total-anchoring for points) from each rate model's latest walk-forward
    snapshot x tonight's projected exposure. A player missing from one
    category's log (e.g. a brand-new call-up with no rebound-chance
    history yet) simply gets NaN for that category's columns -- NOT
    silently zero -- so a downstream caller can distinguish "genuinely
    projects near zero" from "no data available"."""
    scoring_latest = _latest_snapshot(scoring_log, [
        "2pt_att_rate_per_min", "2pt_make_rate", "3pt_att_rate_per_min", "3pt_make_rate",
        "ft_att_rate_per_min", "ft_make_rate"])
    rebound_latest = _latest_snapshot(rebounding_log, ["oreb_rate", "dreb_rate"])
    playmaking_latest = _latest_snapshot(playmaking_log, ["ast_rate", "tov_rate"])
    defensive_latest = _latest_snapshot(defensive_log, ["stl_rate_per_min", "blk_rate_per_min"])

    # Unit-conversion rates: projected minutes -> projected rebound chances / touches, from each
    # player's own trailing chances-per-minute / touches-per-minute (see module docstring).
    reb_exposure = add_walk_forward_player_rate(
        rebounding_log.dropna(subset=["reboundChancesOffensive", "reboundChancesDefensive"]).copy(),
        "reboundChancesOffensive", "minutes", PRIOR_MINUTES_EXPOSURE_UNIT, prefix="oreb_chances_per_min")
    reb_exposure = add_walk_forward_player_rate(
        reb_exposure, "reboundChancesDefensive", "minutes", PRIOR_MINUTES_EXPOSURE_UNIT, prefix="dreb_chances_per_min")
    exposure_latest = _latest_snapshot(reb_exposure, [
        "oreb_chances_per_min_shrunk_rate", "dreb_chances_per_min_shrunk_rate"])

    touch_exposure = add_walk_forward_player_rate(
        playmaking_log.dropna(subset=["touches"]).copy(),
        "touches", "minutes", PRIOR_MINUTES_EXPOSURE_UNIT, prefix="touches_per_min")
    touch_latest = _latest_snapshot(touch_exposure, ["touches_per_min_shrunk_rate"])

    rows = []
    for pid in active_player_ids:
        minutes = float(projected_minutes.get(pid, 0.0))
        if minutes <= 0:
            continue
        row = {"playerId": pid, "projected_minutes": minutes}

        if pid in scoring_latest.index:
            s = scoring_latest.loc[pid]
            row["2pt_proj_att"] = s["2pt_att_rate_per_min"] * minutes
            row["2pt_proj_made"] = row["2pt_proj_att"] * s["2pt_make_rate"]
            row["3pt_proj_att"] = s["3pt_att_rate_per_min"] * minutes
            row["3pt_proj_made"] = row["3pt_proj_att"] * s["3pt_make_rate"]
            row["ft_proj_att"] = s["ft_att_rate_per_min"] * minutes
            row["ft_proj_made"] = row["ft_proj_att"] * s["ft_make_rate"]

        if pid in exposure_latest.index and pid in rebound_latest.index:
            e, r = exposure_latest.loc[pid], rebound_latest.loc[pid]
            row["oreb_proj"] = e["oreb_chances_per_min_shrunk_rate"] * minutes * r["oreb_rate"]
            row["dreb_proj"] = e["dreb_chances_per_min_shrunk_rate"] * minutes * r["dreb_rate"]

        if pid in touch_latest.index and pid in playmaking_latest.index:
            t, p = touch_latest.loc[pid], playmaking_latest.loc[pid]
            row["ast_proj"] = t["touches_per_min_shrunk_rate"] * minutes * p["ast_rate"]
            row["tov_proj"] = t["touches_per_min_shrunk_rate"] * minutes * p["tov_rate"]

        if pid in defensive_latest.index:
            d = defensive_latest.loc[pid]
            row["stl_proj"] = d["stl_rate_per_min"] * minutes
            row["blk_proj"] = d["blk_rate_per_min"] * minutes

        rows.append(row)
    return pd.DataFrame(rows).set_index("playerId") if rows else pd.DataFrame()


def _latest_team_stat_ratings(team_stat_log: pd.DataFrame) -> pd.DataFrame:
    """Each team's most recent walk-forward DREB/AST/TOV/STL/BLK/OREB
    attack/defense/league-avg row -- same "as of before this row's own
    game" convention as `generate_predictions._latest_team_ratings`."""
    return team_stat_log.sort_values("gameDate").groupby("team").tail(1).set_index("team")


def _team_stat_totals(latest_team_stat: pd.DataFrame, home_id: int, away_id: int) -> dict:
    """Projects this specific game's team total for every ADOPTED_CATEGORIES
    label, mirroring how `team_strength.project_game` combines two teams'
    ratings for points. Returns {} if either team has no team-stat history
    yet (e.g. the very first game(s) of a season) -- callers fall back to
    the unanchored bottom-up sum in that case, same honest-fallback
    convention as everywhere else in this pipeline."""
    if home_id not in latest_team_stat.index or away_id not in latest_team_stat.index:
        return {}
    h, a = latest_team_stat.loc[home_id], latest_team_stat.loc[away_id]
    totals = {}
    for label in ADOPTED_CATEGORIES:
        home_total, away_total = project_team_stat(
            home_attack=h[f"{label}_attack_rate"], home_defense=h[f"{label}_defense_rate"],
            away_attack=a[f"{label}_attack_rate"], away_defense=a[f"{label}_defense_rate"],
            league_avg=h[f"{label}_league_avg"],
        )
        totals[label] = {"home": home_total, "away": away_total}
    return totals


def _anchor_preserving_missing(raw_with_nan: pd.Series, side_total: float | None) -> pd.Series:
    """Anchors `raw_with_nan` (a per-player raw category projection, which
    may contain NaN for a player missing that category's rate-model
    history) to `side_total` via `compute_usage_shares` + `allocate_team_total`
    if a total is available, else passes the raw values through unchanged
    -- but either way, restores NaN for exactly the players who were NaN
    on input.

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): this used to
    be an unconditional `.fillna(0.0)` with no way to recover which
    players were genuinely missing history (e.g. a two-way/call-up with
    no cached rebounding/playmaking/defensive-event data) vs. genuinely
    projected near zero -- contradicting `_project_active_roster_stats`'s
    own documented NaN-not-zero contract, which OREB (never anchored, so
    never hits this path) still correctly honors. A missing player would
    get a confident-looking `proj_mean=0.0` in the final output instead of
    being omitted, indistinguishable downstream from a real, data-backed
    near-certainty. The 0.0 fill is still needed internally so
    `compute_usage_shares`/`allocate_team_total` have a real number to sum
    over, but the missing-ness must survive to the caller."""
    missing_mask = raw_with_nan.isna()
    raw = raw_with_nan.fillna(0.0)
    final = raw.copy() if side_total is None else allocate_team_total(compute_usage_shares(raw), side_total)
    final[missing_mask] = np.nan
    return final


def _build_matchup_context(current_season: int, game_date: str) -> dict | None:
    """Everything `_matchup_point_delta` needs, computed ONCE per `run()`
    call (not per player) -- `None` if `current_season` predates matchup
    data entirely (impossible for a live TONIGHT call, but this function
    is also reachable for historical backtesting dates)."""
    if current_season < MATCHUP_DATA_START_SEASON:
        return None

    raw_defender_log = _before(build_defender_game_log(MATCHUP_DATA_START_SEASON, current_season), game_date)

    defender_log = add_defender_difficulty_rate(raw_defender_log.copy())
    defender_ratings = _latest_snapshot(defender_log, ["difficulty_rate"])["difficulty_rate"]

    posgroup_rated = add_position_group_difficulty_rate(
        _before(build_position_group_matchup_log(MATCHUP_DATA_START_SEASON, current_season), game_date))
    posgroup_latest = (posgroup_rated.sort_values("gameDate")
                        .groupby(["team", "position_group"])["posgroup_difficulty_rate"].last())
    league_avg_by_posgroup = posgroup_latest.groupby(level="position_group").mean().to_dict()

    defender_posgroup_log = _before(build_defender_position_group_minutes(MATCHUP_DATA_START_SEASON, current_season), game_date)
    minutes_by_posgroup = {pg: defender_position_group_minutes_asof(defender_posgroup_log, pg, game_date)
                            for pg in POSITION_GROUPS}

    roster_lookup = load_roster_position_lookup(FIRST_DEV_SEASON, current_season)
    roster_pg_by_season_player = roster_lookup.set_index(["season", "playerId"])["position_group"].to_dict()

    # AST/TOV (Sec18/19): full 3-level hierarchy, mirroring points. Level-1 (defender-specific).
    ast_log = add_defender_stat_difficulty_rate(raw_defender_log.copy(), "ast_allowed",
                                                 PRIOR_MATCHUP_MINUTES_AST, prefix="ast_difficulty")
    ast_ratings = _latest_snapshot(ast_log, ["ast_difficulty_rate"])["ast_difficulty_rate"]

    tov_log = add_defender_stat_difficulty_rate(raw_defender_log.copy(), "tov_forced",
                                                 PRIOR_MATCHUP_MINUTES_TOV, prefix="tov_difficulty")
    tov_ratings = _latest_snapshot(tov_log, ["tov_difficulty_rate"])["tov_difficulty_rate"]

    # Level-2 (position-group-vs-team). AST uses EWMA (matches points); TOV uses expanding-shrinkage
    # (does NOT match points -- confirmed empirically, see matchup_difficulty.py's module docstring).
    posgroup_matchup_log = _before(build_position_group_matchup_log(MATCHUP_DATA_START_SEASON, current_season), game_date)
    ast_posgroup_rated = add_position_group_stat_difficulty_rate(
        posgroup_matchup_log.copy(), "ast_allowed", POSGROUP_HALFLIFE_GAMES_AST, prefix="ast_posgroup")
    ast_posgroup_latest = (ast_posgroup_rated.sort_values("gameDate")
                           .groupby(["team", "position_group"])["ast_posgroup_difficulty_rate"].last())
    ast_league_avg_by_posgroup = ast_posgroup_latest.groupby(level="position_group").mean().to_dict()

    tov_posgroup_rated = add_position_group_stat_difficulty_rate_expanding(
        posgroup_matchup_log.copy(), "tov_forced", POSGROUP_PRIOR_MATCHUP_MINUTES_TOV, prefix="tov_posgroup")
    tov_posgroup_latest = (tov_posgroup_rated.sort_values("gameDate")
                           .groupby(["team", "position_group"])["tov_posgroup_difficulty_rate"].last())
    tov_league_avg_by_posgroup = tov_posgroup_latest.groupby(level="position_group").mean().to_dict()

    # Exposure weighting (w(p,d)) is stat-agnostic -- how much a defender has guarded a position
    # group recently doesn't depend on which stat we're predicting -- so `minutes_by_posgroup`
    # (already built above for points) is reused directly for AST/TOV too.
    total_minutes = defender_total_minutes_asof(raw_defender_log, game_date)

    return {
        "defender_ratings": defender_ratings, "posgroup_latest": posgroup_latest,
        "league_avg_by_posgroup": league_avg_by_posgroup, "minutes_by_posgroup": minutes_by_posgroup,
        "roster_pg_by_season_player": roster_pg_by_season_player,
        "ast_ratings": ast_ratings, "ast_posgroup_latest": ast_posgroup_latest,
        "ast_league_avg_by_posgroup": ast_league_avg_by_posgroup,
        "tov_ratings": tov_ratings, "tov_posgroup_latest": tov_posgroup_latest,
        "tov_league_avg_by_posgroup": tov_league_avg_by_posgroup,
        "total_minutes": total_minutes,
    }


def _matchup_point_delta(pid: int, offense_season: int, opposing_team_id: int, opposing_roster_ids: list,
                          projected_minutes_for_pid: float, matchup_ctx: dict | None) -> tuple[float, bool]:
    """Returns (point_delta, was_adjusted). `was_adjusted=False` (delta=0.0)
    when the offensive player's position group can't be resolved (no
    roster data for them this season) -- an honest "no adjustment applied"
    signal, not a silent zero passed off as "checked and found neutral"."""
    if matchup_ctx is None:
        return 0.0, False
    position_group_value = matchup_ctx["roster_pg_by_season_player"].get((offense_season, pid), "")
    if position_group_value == "":
        return 0.0, False

    minutes_at_pg = matchup_ctx["minutes_by_posgroup"].get(position_group_value)
    posgroup_rating = matchup_ctx["posgroup_latest"].get((opposing_team_id, position_group_value))
    league_avg = matchup_ctx["league_avg_by_posgroup"].get(position_group_value, 0.0)

    rate_delta = opponent_defense_adjustment(
        opposing_roster_player_ids=opposing_roster_ids, defender_ratings=matchup_ctx["defender_ratings"],
        defender_minutes_at_posgroup=minutes_at_pg, posgroup_rating=posgroup_rating,
        league_avg_posgroup_rating=league_avg, min_matchup_minutes_to_trust=MIN_MATCHUP_MINUTES_TO_TRUST,
    )
    return matchup_point_delta(rate_delta, projected_minutes_for_pid), True


def _ast_tov_matchup_delta(stat: str, pid: int, offense_season: int, opposing_team_id: int,
                           opposing_roster_ids: list, projected_minutes_for_pid: float,
                           matchup_ctx: dict | None) -> tuple[float, bool]:
    """AST/TOV analog of `_matchup_point_delta` (Sec18/19) -- now the FULL
    3-level hierarchy (defender-specific -> position-group-vs-team ->
    league floor), same as points. `stat` is "ast" or "tov". Returns
    (delta, was_adjusted) -- `was_adjusted=False` when there's no matchup
    context at all, OR the offensive player's position group can't be
    resolved (same honest-fallback convention as points)."""
    if matchup_ctx is None:
        return 0.0, False
    position_group_value = matchup_ctx["roster_pg_by_season_player"].get((offense_season, pid), "")
    if position_group_value == "":
        return 0.0, False

    ratings = matchup_ctx[f"{stat}_ratings"]
    posgroup_rating = matchup_ctx[f"{stat}_posgroup_latest"].get((opposing_team_id, position_group_value))
    league_avg = matchup_ctx[f"{stat}_league_avg_by_posgroup"].get(position_group_value, 0.0)
    minutes_at_pg = matchup_ctx["minutes_by_posgroup"].get(position_group_value)

    rate_delta = opponent_defense_adjustment(
        opposing_roster_player_ids=opposing_roster_ids, defender_ratings=ratings,
        defender_minutes_at_posgroup=minutes_at_pg, posgroup_rating=posgroup_rating,
        league_avg_posgroup_rating=league_avg, min_matchup_minutes_to_trust=MIN_MATCHUP_MINUTES_TO_TRUST,
    )
    return matchup_point_delta(rate_delta, projected_minutes_for_pid), True


def _build_rate_logs(current_season: int, game_date: str) -> tuple:
    """One shared `build_player_game_log` call, reused for every category
    that doesn't need player-tracking data (scoring, defensive events),
    plus one `attach_player_track` pass for the two that do (rebounding,
    playmaking) -- avoids rebuilding the same expensive merge three times.
    Each returned log is filtered to `gameDate < game_date` (see `_before`)
    so every downstream `_latest_snapshot` call reflects "as of game_date",
    not "as of the real wall-clock today"."""
    base_log = build_player_game_log(FIRST_DEV_SEASON, current_season)
    scoring_log = _before(add_scoring_rates(base_log.copy()), game_date)
    defensive_log = _before(add_defensive_event_rates(base_log.copy()), game_date)

    tracked = attach_player_track(base_log.copy(), FIRST_DEV_SEASON, current_season)
    rebounding_log = _before(add_rebounding_rates(tracked.dropna(subset=["reboundChancesOffensive", "reboundChancesDefensive"]).copy()), game_date)
    playmaking_log = _before(add_playmaking_rates(tracked.dropna(subset=["touches"]).copy()), game_date)
    return scoring_log, rebounding_log, playmaking_log, defensive_log


# name -> (actual_col, proj_col) on the log that category naturally lives on. "points" is
# special-cased in _fit_prop_distributions (proj = sum of the 3 scoring sub-categories, not a
# single rate model's own column).
_CATEGORY_ACTUAL_PROJ_COLS = {
    "points": ("points", None),
    "2pt_made": ("fgMade2", "2pt_proj_made"), "3pt_made": ("fg3Made", "3pt_proj_made"),
    "ft_made": ("ftMade", "ft_proj_made"),
    "oreb": ("oreb", "oreb_proj"), "dreb": ("dreb", "dreb_proj"),
    "ast": ("assists", "ast_proj"), "tov": ("turnovers", "tov_proj"),
    "stl": ("steals", "stl_proj"), "blk": ("blocks", "blk_proj"),
}


def _fit_prop_distributions(scoring_log: pd.DataFrame, rebounding_log: pd.DataFrame,
                             playmaking_log: pd.DataFrame, defensive_log: pd.DataFrame) -> dict:
    """Refits each category's own family parameters fresh from historical
    residuals every call (mirrors `generate_predictions.run`'s identical
    live-refit-over-caching tradeoff for `score_distribution.py` --
    correctness over efficiency, caching deferred). Driven entirely by
    `prop_distribution.CATEGORY_FAMILY` so a category's fitting logic
    automatically follows its own validated config, not a hardcoded
    per-category branch here:
    - "count" (all 10 categories currently, see CATEGORY_FAMILY's
      docstring -- points/2PT/3PT/FT-made included, NOT the plan's
      original continuous assumption): refits via `fit_count_family`;
      Poisson needs nothing further, NegBin needs the fitted dispersion
      `r`. `fitted[name]` is `{}` for Poisson (present so the category is
      known-fitted, distinguishing it from "no data available yet").
    - "continuous" (none currently adopted, kept generic in case a future
      category needs it): refits a variance-vs-exposure model + Student-t df.
    Every log passed in is already `_before`-filtered by the caller, so
    this never sees game_date's own (or later) games."""
    logs_by_name = {
        "points": scoring_log, "2pt_made": scoring_log, "3pt_made": scoring_log, "ft_made": scoring_log,
        "oreb": rebounding_log, "dreb": rebounding_log, "ast": playmaking_log, "tov": playmaking_log,
        "stl": defensive_log, "blk": defensive_log,
    }
    points_proj = (2 * scoring_log["2pt_proj_made"] + 3 * scoring_log["3pt_proj_made"] + scoring_log["ft_proj_made"])

    fitted = {}
    for name, (actual_col, proj_col) in _CATEGORY_ACTUAL_PROJ_COLS.items():
        spec = CATEGORY_FAMILY[name]
        log = logs_by_name[name]
        proj = points_proj if name == "points" else log[proj_col]
        rows = pd.DataFrame({"_actual": log[actual_col], "_proj": proj})
        if spec["family_type"] == "continuous":
            rows["_exposure"] = log[spec["exposure_col"]]
        rows = rows.dropna()
        if rows.empty:
            continue

        if spec["family_type"] == "count":
            count_fit = fit_count_family(rows["_actual"].to_numpy(), rows["_proj"].to_numpy())
            fitted[name] = {"r": count_fit["r"]} if count_fit["family"] == "negbin" else {}
        else:
            resid = (rows["_actual"] - rows["_proj"]).to_numpy()
            fitted[name] = fit_continuous_family(resid, rows["_exposure"].to_numpy())
    return fitted


def _distribution_fields(stat_name: str, mean: float, distribution_fits: dict) -> tuple[float | None, str, dict]:
    """(proj_var, family, family_param) for one output row. `proj_var` is
    the IMPLIED variance under the fitted family (mean+mean^2/r for
    NegBin, mean itself for Poisson, informational only -- neither
    `over_under_prob` nor `log_score` needs it passed back in for a count
    family, they derive it from mean/r themselves) -- NaN only if this
    category has no fit yet (e.g. no historical data at all for a
    brand-new call-up situation, extremely unlikely at the category
    level but handled honestly rather than assumed impossible)."""
    spec = CATEGORY_FAMILY[stat_name]
    if stat_name not in distribution_fits:
        return None, spec["family"], {}
    fit = distribution_fits[stat_name]
    if spec["family"] == "poisson":
        return mean, "poisson", {}
    if spec["family"] == "negbin":
        r = fit["r"]
        return mean + mean ** 2 / r, "negbin", {"r": r}
    # continuous (kept generic for a future category) -- needs its own projected exposure, which
    # this generic per-row call site doesn't have; not reachable for any currently-adopted category.
    return None, spec["family"], {}


def run(game_date: str) -> pd.DataFrame:
    game_preds = generate_game_predictions(game_date)
    if game_preds.empty:
        return pd.DataFrame()

    refresh_all_data()  # ensure real-time data is backfilled; season logic below uses game_date, not "today"
    # `current_season` derived PURELY from `game_date` (see `season_for_date`) -- see `_before`'s
    # docstring and MODEL_DOCUMENTATION.md for the real bug this fixes (a star player's trailing-
    # minutes projection silently reaching into a much later, unrelated season).
    current_season = season_for_date(game_date)
    todays_games = games_on_date(game_date)
    team_id_by_abbrev = {}
    for r in todays_games.itertuples(index=False):
        team_id_by_abbrev[r.homeAbbrev] = r.homeTeamId
        team_id_by_abbrev[r.awayAbbrev] = r.awayTeamId

    warn_if_stale_for_backtest(game_date)
    injury_report = fetch_current_injury_report()
    _, player_minutes_stints = _fit_latest_player_ratings(current_season, before_date=game_date)

    team_log = add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, current_season))
    team_log = _before(team_log, game_date)
    latest_team_stat = _latest_team_stat_ratings(
        add_team_stat_ratings(_before(build_team_stat_game_log(FIRST_DEV_SEASON, current_season), game_date)))
    # `team_history`/`team_side` must never blend across a season boundary -- see
    # generate_predictions.py's own fix for the full writeup (a real bug found via a 2023-11-08
    # early-season spot-check: the "last 10 games" lookback silently pulled in games, and
    # therefore players, from the PRIOR season, inflating the resolved active roster with players
    # no longer on the team and diluting every real player's projected share).
    team_history, team_side = build_team_history(team_log[team_log["season"] == current_season])
    current_roster = load_current_roster_player_ids(current_season)

    scoring_log, rebounding_log, playmaking_log, defensive_log = _build_rate_logs(current_season, game_date)
    distribution_fits = _fit_prop_distributions(scoring_log, rebounding_log, playmaking_log, defensive_log)
    matchup_ctx = _build_matchup_context(current_season, game_date)
    if matchup_ctx is None:
        print(f"  NOTE: season {current_season} predates matchup data (starts {MATCHUP_DATA_START_SEASON}) "
              f"-- points will not be matchup-adjusted", flush=True)

    all_rows = []
    for row in game_preds.itertuples(index=False):
        home_id = team_id_by_abbrev.get(row.homeAbbrev)
        away_id = team_id_by_abbrev.get(row.awayAbbrev)
        if home_id is None or away_id is None:
            print(f"  WARNING: could not resolve team IDs for {row.awayAbbrev} @ {row.homeAbbrev}, skipping props", flush=True)
            continue

        home_shares, home_tag = resolve_active_lineup(
            row.homeAbbrev, team_history.get(home_id, []), player_minutes_stints, team_side.get(home_id, {}), injury_report,
            current_roster_ids=current_roster.get(home_id))
        away_shares, away_tag = resolve_active_lineup(
            row.awayAbbrev, team_history.get(away_id, []), player_minutes_stints, team_side.get(away_id, {}), injury_report,
            current_roster_ids=current_roster.get(away_id))
        # {} (not a per-category crash) when either team has no team-stat history yet -- see
        # `_team_stat_totals`'s own fallback docstring.
        game_stat_totals = _team_stat_totals(latest_team_stat, home_id, away_id)

        for side, shares, tag, team_abbrev, opp_abbrev, opposing_team_id, opposing_shares, team_total in (
            ("home", home_shares, home_tag, row.homeAbbrev, row.awayAbbrev, away_id, away_shares, row.pred_home),
            ("away", away_shares, away_tag, row.awayAbbrev, row.homeAbbrev, home_id, home_shares, row.pred_away),
        ):
            if shares.empty:
                print(f"    {team_abbrev}: no active-roster data this call ({tag}) -- skipping props for this team", flush=True)
                continue
            projected_minutes = shares * TEAM_MINUTES_PER_GAME

            proj = _project_active_roster_stats(list(projected_minutes.index), projected_minutes,
                                                  scoring_log, rebounding_log, playmaking_log, defensive_log)
            if proj.empty:
                continue

            # A player missing scoring history (e.g. a brand-new call-up) gets NaN raw points --
            # fillna(0) here specifically, so ONE missing player can't poison compute_usage_shares'
            # sum for the WHOLE team (the same NaN-poisons-aggregate bug pattern already found
            # twice elsewhere in this project -- guarded against explicitly here, not by luck).
            raw_points = raw_projected_points(proj).fillna(0.0)

            opposing_roster_ids = list(opposing_shares.index)
            matchup_adjusted_flags = {}
            ast_adjusted_flags = {}
            tov_adjusted_flags = {}
            adjusted_points = raw_points.copy()
            for pid in raw_points.index:
                delta, was_adjusted = _matchup_point_delta(
                    pid, current_season, opposing_team_id, opposing_roster_ids,
                    float(projected_minutes.get(pid, 0.0)), matchup_ctx)
                # REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): `delta` (from
                # `opponent_defense_adjustment`, via `matchup_point_delta`'s pure unit conversion)
                # is POSITIVE for a TOUGHER-than-average matchup, and its own docstring is explicit
                # that a positive value means "suppress the offensive player's projection" -- so it
                # must be SUBTRACTED, not added. Adding it (the original code) INCREASED points
                # against tougher defenders and DECREASED them against weaker ones -- exactly
                # backwards. Subtracting is also the correct fix for AST/TOV below despite TOV's
                # opposite rate polarity (higher tov_forced = tougher defense, not lower): the
                # underlying delta = league_avg_rate - defender_rate already encodes each stat's
                # own convention correctly, so `raw - delta` is equivalent to `raw + (defender_rate
                # - league_avg_rate)` in every case -- a higher defender_rate always pushes the
                # offensive stat up, whether that means "weak defender allows more points" or
                # "tough defender forces more turnovers". See MODEL_DOCUMENTATION.md for the
                # worked-through derivation.
                adjusted_points[pid] = raw_points[pid] - delta
                matchup_adjusted_flags[pid] = was_adjusted

                # AST/TOV (Sec18/19): the matchup delta is a direct additive nudge to the player's
                # own RAW projection, applied BEFORE the ADOPTED_CATEGORIES anchoring loop below --
                # ast/tov ARE anchored to team_stat_rates' team total (Task #24), same as points.
                pid_minutes = float(projected_minutes.get(pid, 0.0))
                ast_delta, ast_adjusted = _ast_tov_matchup_delta(
                    "ast", pid, current_season, opposing_team_id, opposing_roster_ids, pid_minutes, matchup_ctx)
                tov_delta, tov_adjusted = _ast_tov_matchup_delta(
                    "tov", pid, current_season, opposing_team_id, opposing_roster_ids, pid_minutes, matchup_ctx)
                if pid in proj.index:
                    if ast_adjusted and pd.notna(proj.loc[pid].get("ast_proj")):
                        proj.loc[pid, "ast_proj"] = max(0.0, proj.loc[pid, "ast_proj"] - ast_delta)
                    if tov_adjusted and pd.notna(proj.loc[pid].get("tov_proj")):
                        proj.loc[pid, "tov_proj"] = max(0.0, proj.loc[pid, "tov_proj"] - tov_delta)
                ast_adjusted_flags[pid] = ast_adjusted
                tov_adjusted_flags[pid] = tov_adjusted

            usage_shares = compute_usage_shares(adjusted_points)
            final_points = allocate_team_total(usage_shares, team_total)

            # DREB/AST/TOV/STL/BLK (Task #24): same macro-anchor + micro-reallocation rule as
            # points, using team_stat_rates' team-level total in place of RAPM's. Each category's
            # raw (matchup-adjusted, for ast/tov) per-player projection sets a usage share; falls
            # back to the raw bottom-up projection (unanchored) if this game has no team-stat
            # history yet, same honest-fallback convention as everywhere else in this pipeline.
            final_by_category = {}
            anchored_flags = {}
            for label in ADOPTED_CATEGORIES:
                col = f"{label}_proj"
                if col not in proj.columns:
                    continue
                side_total = game_stat_totals.get(label, {}).get(side)
                final_by_category[label] = _anchor_preserving_missing(proj[col], side_total)
                anchored_flags[label] = side_total is not None

            for pid in proj.index:
                stat_values = {
                    "points": final_points.get(pid), "2pt_made": proj.loc[pid].get("2pt_proj_made"),
                    "3pt_made": proj.loc[pid].get("3pt_proj_made"), "ft_made": proj.loc[pid].get("ft_proj_made"),
                    "oreb": proj.loc[pid].get("oreb_proj"),
                    "dreb": final_by_category["dreb"].get(pid) if "dreb" in final_by_category else proj.loc[pid].get("dreb_proj"),
                    "ast": final_by_category["ast"].get(pid) if "ast" in final_by_category else proj.loc[pid].get("ast_proj"),
                    "tov": final_by_category["tov"].get(pid) if "tov" in final_by_category else proj.loc[pid].get("tov_proj"),
                    "stl": final_by_category["stl"].get(pid) if "stl" in final_by_category else proj.loc[pid].get("stl_proj"),
                    "blk": final_by_category["blk"].get(pid) if "blk" in final_by_category else proj.loc[pid].get("blk_proj"),
                }
                for stat_name, mean in stat_values.items():
                    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                        continue
                    proj_var, family, family_param = _distribution_fields(stat_name, float(mean), distribution_fits)
                    all_rows.append({
                        "gameId": row.gameId, "playerId": pid, "team": team_abbrev, "opponent": opp_abbrev,
                        "stat_name": stat_name, "proj_mean": float(mean),
                        "proj_var": proj_var, "family": family, "family_param": json.dumps(family_param),
                        "minutes_tier_tag": tag,
                        "anchored_to_team_total": stat_name == "points" or anchored_flags.get(stat_name, False),
                        # matchup difficulty reshapes points (position-group-aware) and ast/tov
                        # (defender-level only, Sec18); every other category is never adjusted.
                        "matchup_adjusted": (
                            (stat_name == "points" and matchup_adjusted_flags.get(pid, False))
                            or (stat_name == "ast" and ast_adjusted_flags.get(pid, False))
                            or (stat_name == "tov" and tov_adjusted_flags.get(pid, False))
                        ),
                    })

    result_df = pd.DataFrame(all_rows)
    if not result_df.empty:
        out_path = DATA_PROCESSED / f"daily_props_{game_date}.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"saved {out_path} ({len(result_df)} rows, {result_df['playerId'].nunique()} players)", flush=True)
    return result_df


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    run(target_date)
