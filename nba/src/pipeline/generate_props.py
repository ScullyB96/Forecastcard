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
- **Matchup difficulty (`matchup_difficulty.py`) is NOT wired into this
  live call yet**, even though it's fully built and validated (see
  MODEL_DOCUMENTATION.md Sec10) -- wiring it live requires resolving each
  offensive player's own position group (from `CommonTeamRoster`) and the
  full opposing active roster's defender ratings for tonight specifically,
  which is real additional live-data-assembly work, not a trivial call.
  Points are anchored to the team total via `usage_allocation.py` WITHOUT
  a matchup-difficulty reshape for now -- an honest, flagged fast-follow,
  not a silently-dropped feature.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report
from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_minutes import attach_player_track, build_player_game_log
from src.models.player_playmaking_rates import add_playmaking_rates
from src.models.player_rate_shrinkage import add_walk_forward_player_rate
from src.models.player_rebounding_rates import add_rebounding_rates
from src.models.player_scoring_rates import add_scoring_rates
from src.models.team_strength import add_team_ratings, build_team_game_log
from src.models.usage_allocation import allocate_team_points, compute_usage_shares, raw_projected_points
from src.pipeline.active_roster import build_team_history, games_on_date, resolve_active_lineup
from src.pipeline.generate_predictions import _fit_latest_player_ratings, run as generate_game_predictions
from src.pipeline.refresh_data import refresh_all_data
from src.utils.paths import DATA_PROCESSED

TEAM_MINUTES_PER_GAME = 240.0  # 5 players x 48 minutes; OT ignored for a PRE-game projection (see module docstring)
PRIOR_MINUTES_EXPOSURE_UNIT = 100.0  # expanding-shrinkage prior for the chances/touches-per-minute unit-conversion rates

STAT_COLUMNS = ("2pt_proj_made", "3pt_proj_made", "ft_proj_made",
                "oreb_proj", "dreb_proj", "ast_proj", "tov_proj", "stl_proj", "blk_proj")


def _latest_snapshot(log: pd.DataFrame, cols: list, group_col: str = "playerId") -> pd.DataFrame:
    """Each player's most recent walk-forward row for the given columns --
    same convention as `generate_predictions._latest_team_ratings`: the
    stored value already reflects "as of before that row's own game", the
    accepted live-pipeline approximation for "current rate" used
    throughout this project's live entry points."""
    return log.sort_values("gameDate").groupby(group_col)[cols].last()


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


def _build_rate_logs(current_season: int) -> tuple:
    """One shared `build_player_game_log` call, reused for every category
    that doesn't need player-tracking data (scoring, defensive events),
    plus one `attach_player_track` pass for the two that do (rebounding,
    playmaking) -- avoids rebuilding the same expensive merge three times."""
    base_log = build_player_game_log(FIRST_DEV_SEASON, current_season)
    scoring_log = add_scoring_rates(base_log.copy())
    defensive_log = add_defensive_event_rates(base_log.copy())

    tracked = attach_player_track(base_log.copy(), FIRST_DEV_SEASON, current_season)
    rebounding_log = add_rebounding_rates(tracked.dropna(subset=["reboundChancesOffensive", "reboundChancesDefensive"]).copy())
    playmaking_log = add_playmaking_rates(tracked.dropna(subset=["touches"]).copy())
    return scoring_log, rebounding_log, playmaking_log, defensive_log


def run(game_date: str) -> pd.DataFrame:
    game_preds = generate_game_predictions(game_date)
    if game_preds.empty:
        return pd.DataFrame()

    current_season = refresh_all_data()
    todays_games = games_on_date(game_date)
    team_id_by_abbrev = {}
    for r in todays_games.itertuples(index=False):
        team_id_by_abbrev[r.homeAbbrev] = r.homeTeamId
        team_id_by_abbrev[r.awayAbbrev] = r.awayTeamId

    injury_report = fetch_current_injury_report()
    _, player_minutes_stints = _fit_latest_player_ratings(current_season)

    team_log = add_team_ratings(build_team_game_log(FIRST_DEV_SEASON, current_season))
    team_history, team_side = build_team_history(team_log)

    scoring_log, rebounding_log, playmaking_log, defensive_log = _build_rate_logs(current_season)

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

        for shares, tag, team_abbrev, opp_abbrev, team_total in (
            (home_shares, home_tag, row.homeAbbrev, row.awayAbbrev, row.pred_home),
            (away_shares, away_tag, row.awayAbbrev, row.homeAbbrev, row.pred_away),
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
            usage_shares = compute_usage_shares(raw_points)
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
                        "matchup_adjusted": False,  # see module docstring -- not wired into this live call yet
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
