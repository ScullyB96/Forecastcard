"""Live daily player-props entry point: reuses `generate_predictions.py`'s
already-RAPM-adjusted team point totals as the macro anchor (a single
source of truth -- props and game predictions can never silently disagree
on a team's total), resolves tonight's active roster via the SHARED
`active_roster.py` helpers, projects each active player's full stat line
from every rate-category model built this session, composes points via
`usage_allocation.py`'s macro-anchor + micro-reallocation rule, and writes
one row per (game, player, stat).

Run as `python -m src.pipeline.generate_props [YYYY-MM-DD]` (defaults to
today).

**Known v1 simplifications, documented rather than silently baked in:**
- `TEAM_MINUTES_PER_GAME = 240.0` (5 players x 48 minutes) ignores
  overtime -- a live projection can't know in advance a game will go to
  OT, so this is the correct expectation for a PRE-game projection, not an
  approximation error.
- REB/AST/TOV/STL/BLK need PROJECTED rebound-chances/touches for tonight,
  which no dedicated model exists for (out of scope vs. the user's
  verbatim ask -- see `usage_allocation.py`'s module docstring on the
  unanchored categories). Approximated here via each player's own trailing
  chances-per-minute / touches-per-minute (expanding-shrinkage, exposure =
  minutes) x their projected minutes -- a low-leverage UNIT CONVERSION
  step, not a second validated rate model; the category's own already-
  validated conversion-rate model (`player_rebounding_rates.py`/
  `player_playmaking_rates.py`) still does the actual prediction.
- Matchup difficulty is applied to POINTS and AST/TOV, all three via the
  FULL 3-level hierarchy (defender-specific -> position-group-vs-team ->
  league floor, Sec13/18/19) -- gated on `season >= MATCHUP_DATA_START_SEASON`
  and `MIN_MATCHUP_MINUTES_TO_TRUST`; an offensive player whose position
  group can't be resolved (no roster data) gets no adjustment
  (`matchup_adjusted: False`), not a crash or a silent zero passed off as
  "checked and found neutral". AST/TOV are UNANCHORED (per
  `usage_allocation.py`'s v1 scope), so their matchup adjustment is a
  direct additive delta to the player's own projection (clipped at 0), not
  a share-reallocation the way points' adjustment is.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report
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
from src.models.team_strength import add_team_ratings, build_team_game_log
from src.models.usage_allocation import allocate_team_points, compute_usage_shares, matchup_point_delta, raw_projected_points
from src.pipeline.active_roster import build_team_history, games_on_date, resolve_active_lineup
from src.pipeline.generate_predictions import _fit_latest_player_ratings, run as generate_game_predictions
from src.pipeline.refresh_data import refresh_all_data
from src.utils.paths import DATA_PROCESSED

TEAM_MINUTES_PER_GAME = 240.0  # 5 players x 48 minutes; OT ignored for a PRE-game projection (see module docstring)
PRIOR_MINUTES_EXPOSURE_UNIT = 100.0  # expanding-shrinkage prior for the chances/touches-per-minute unit-conversion rates
POSITION_GROUPS = ("G", "F", "C")

STAT_COLUMNS = ("2pt_proj_made", "3pt_proj_made", "ft_proj_made",
                "oreb_proj", "dreb_proj", "ast_proj", "tov_proj", "stl_proj", "blk_proj")


def _latest_snapshot(log: pd.DataFrame, cols: list, group_col: str = "playerId") -> pd.DataFrame:
    """Each player's most recent walk-forward row for the given columns --
    same convention as `generate_predictions._latest_team_ratings`: the
    stored value already reflects "as of before that row's own game", the
    accepted live-pipeline approximation for "current rate" used
    throughout this project's live entry points."""
    return log.sort_values("gameDate").groupby(group_col)[cols].last()


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

    injury_report = fetch_current_injury_report()
    _, player_minutes_stints = _fit_latest_player_ratings(current_season, before_date=game_date)

    team_log = add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, current_season))
    team_log = _before(team_log, game_date)
    # `team_history`/`team_side` must never blend across a season boundary -- see
    # generate_predictions.py's own fix for the full writeup (a real bug found via a 2023-11-08
    # early-season spot-check: the "last 10 games" lookback silently pulled in games, and
    # therefore players, from the PRIOR season, inflating the resolved active roster with players
    # no longer on the team and diluting every real player's projected share).
    team_history, team_side = build_team_history(team_log[team_log["season"] == current_season])

    scoring_log, rebounding_log, playmaking_log, defensive_log = _build_rate_logs(current_season, game_date)
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
            row.homeAbbrev, team_history.get(home_id, []), player_minutes_stints, team_side.get(home_id, {}), injury_report)
        away_shares, away_tag = resolve_active_lineup(
            row.awayAbbrev, team_history.get(away_id, []), player_minutes_stints, team_side.get(away_id, {}), injury_report)

        for shares, tag, team_abbrev, opp_abbrev, opposing_team_id, opposing_shares, team_total in (
            (home_shares, home_tag, row.homeAbbrev, row.awayAbbrev, away_id, away_shares, row.pred_home),
            (away_shares, away_tag, row.awayAbbrev, row.homeAbbrev, home_id, home_shares, row.pred_away),
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
                adjusted_points[pid] = raw_points[pid] + delta
                matchup_adjusted_flags[pid] = was_adjusted

                # AST/TOV (Sec18/19): unanchored, so a direct additive delta to the player's own
                # projection -- no team-total redistribution needed the way points has.
                pid_minutes = float(projected_minutes.get(pid, 0.0))
                ast_delta, ast_adjusted = _ast_tov_matchup_delta(
                    "ast", pid, current_season, opposing_team_id, opposing_roster_ids, pid_minutes, matchup_ctx)
                tov_delta, tov_adjusted = _ast_tov_matchup_delta(
                    "tov", pid, current_season, opposing_team_id, opposing_roster_ids, pid_minutes, matchup_ctx)
                if pid in proj.index:
                    if ast_adjusted and pd.notna(proj.loc[pid].get("ast_proj")):
                        proj.loc[pid, "ast_proj"] = max(0.0, proj.loc[pid, "ast_proj"] + ast_delta)
                    if tov_adjusted and pd.notna(proj.loc[pid].get("tov_proj")):
                        proj.loc[pid, "tov_proj"] = max(0.0, proj.loc[pid, "tov_proj"] + tov_delta)
                ast_adjusted_flags[pid] = ast_adjusted
                tov_adjusted_flags[pid] = tov_adjusted

            usage_shares = compute_usage_shares(adjusted_points)
            final_points = allocate_team_points(usage_shares, team_total)

            for pid in proj.index:
                stat_values = {
                    "points": final_points.get(pid), "2pt_made": proj.loc[pid].get("2pt_proj_made"),
                    "3pt_made": proj.loc[pid].get("3pt_proj_made"), "ft_made": proj.loc[pid].get("ft_proj_made"),
                    "oreb": proj.loc[pid].get("oreb_proj"), "dreb": proj.loc[pid].get("dreb_proj"),
                    "ast": proj.loc[pid].get("ast_proj"), "tov": proj.loc[pid].get("tov_proj"),
                    "stl": proj.loc[pid].get("stl_proj"), "blk": proj.loc[pid].get("blk_proj"),
                }
                for stat_name, mean in stat_values.items():
                    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
                        continue
                    all_rows.append({
                        "gameId": row.gameId, "playerId": pid, "team": team_abbrev, "opponent": opp_abbrev,
                        "stat_name": stat_name, "proj_mean": float(mean),
                        "minutes_tier_tag": tag, "anchored_to_team_total": stat_name == "points",
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
