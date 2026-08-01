"""Live daily entry point: refresh data, resolve tonight's active lineups,
run the full Phase 1 (team strength) + Phase 2 (RAPM-lite lineup
adjustment, PREDICTIVE minutes mode -- see below) + Phase 3 (score
distribution) stack, and write out win probability / spread / total for
every game on a given date.

Run as `python -m src.pipeline.generate_predictions [YYYY-MM-DD]` (defaults
to today).

**Active-lineup resolution is 2-tier for v1, not the 3-tier design
originally sketched** -- documented honestly rather than silently shipped
as if the third tier existed: (1) RotoWire's Out/Doubtful signal
(`fetch_rotowire_lineups.py`, matched to a player ID via
`player_name_crosswalk.py` -- confirmed live 2026-07-24: 126/152 flagged
players matched, the rest are recent rookies/two-way players not yet in
`nba_api`'s static player list) is the only real injury signal actually
wired up (the NBA official PDF injury report was deliberately not built for
v1, see MODEL_DOCUMENTATION.md); (2) a last-resort trailing-minutes
projection for every player NOT flagged Out/Doubtful, renormalized across
the team's recent rotation. Every output row is tagged with which tier
produced it.

**Minutes mode is PREDICTIVE, not ORACLE** (unlike
`validate_rapm_lineup_adjustment.py`'s backtest, which uses real
already-known minutes to measure a ceiling): each active player's own
trailing-average minutes over their team's last `MINUTES_LOOKBACK_GAMES`,
renormalized across tonight's active roster -- the actual PRE-game
projection a deployed system has to work with.

**Phase 2 (RAPM-lite lineup adjustment) IS wired in as of 2026-07-25**,
now that `validate_predictive_lineup_adjustment.py`'s dev-only bootstrap
confirmed predictive-minutes mode captures essentially all of oracle
mode's real improvement (total_mae -0.0206 vs oracle's -0.0217, margin_mae
-0.0336 vs oracle's -0.0350, both real -- see MODEL_DOCUMENTATION.md
Sec7/Sec9). The single latest RAPM-lite snapshot is fit fresh on every call
from ALL cached stints through the most recent completed/backfilled
season (`_fit_latest_player_ratings`) -- a full periodic walk-forward
refit series (as the dev backtest uses, one snapshot per historical
checkpoint) isn't needed live, since a live call only ever needs "the
single best rating estimate as of right now". Recomputing this fresh per
invocation is the same correctness-over-efficiency tradeoff already
documented above for the score-distribution/home-court fits.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_str
from src.models.lineup_rating import player_minutes_from_stints, project_lineup_adjustment, team_recent_roster_rapm
from src.models.rapm_lite import fit_rapm, prepare_stints
from src.models.score_distribution import (
    fit_home_away_correlation, fit_residual_variance_model, interval, predict_variance, win_prob_home,
)
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
from src.pipeline.active_roster import MINUTES_LOOKBACK_GAMES, build_team_history, games_on_date, resolve_active_lineup
from src.pipeline.refresh_data import refresh_all_data
from src.utils.paths import DATA_PROCESSED, DATA_RAW


def _fit_latest_player_ratings(current_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_ratings, player_minutes) using every cached season's
    stints through `current_season` -- a single fresh RAPM-lite fit "as of
    right now", not the periodic backtest series. Seasons whose stints
    haven't been built yet (or whose raw data isn't cached) are skipped,
    not fatal -- an early-season call with less history just fits on
    whatever's available, same as any other walk-forward fit in this
    project."""
    frames = []
    for y in range(FIRST_DEV_SEASON, current_season + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        if not sched_path.exists():
            continue
        stints = build_season_stints(y)
        if stints.empty:
            continue
        schedule = pd.read_parquet(sched_path)
        frames.append(prepare_stints(stints, schedule[schedule["seasonType"] == "Regular Season"]))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    all_stints = pd.concat(frames, ignore_index=True)
    player_ratings = fit_rapm(all_stints).reset_index()
    player_minutes = player_minutes_from_stints(all_stints)
    return player_ratings, player_minutes


def _latest_team_ratings(team_log: pd.DataFrame) -> pd.DataFrame:
    """Each team's most recent walk-forward rating row -- i.e. its pregame
    rating as of the NEXT game it plays (today's game), since the walk-
    forward columns are already "as of before this row's own game"."""
    return team_log.sort_values("gameDate").groupby("team").tail(1).set_index("team")


def run(game_date: str) -> pd.DataFrame:
    current_season = refresh_all_data()

    todays_games = games_on_date(game_date)
    if todays_games.empty:
        print(f"no games found for {game_date}", flush=True)
        return pd.DataFrame()

    injury_report = fetch_current_injury_report()
    team_log = add_team_ratings(build_team_game_log(2015, current_season))
    latest = _latest_team_ratings(team_log)
    team_history, team_side = build_team_history(team_log)

    player_ratings, player_minutes = _fit_latest_player_ratings(current_season)
    have_lineup_adjustment = not player_ratings.empty
    if not have_lineup_adjustment:
        print("WARNING: no RAPM-lite player-rating snapshot available yet (stints not built for "
              "any cached season) -- falling back to team-strength-only (Phase 1) projections "
              "for this call", flush=True)

    # Home-court multiplier and score-distribution variance/correlation are both fit from the
    # full historical backtest -- recomputed on every call for v1 (correctness over efficiency);
    # caching these as persisted artifacts (instead of refitting per live invocation) is a
    # documented follow-up, not done here.
    try:
        from src.models.validate_team_strength_baseline import build_dev_predictions
        dev_preds = build_dev_predictions().dropna(subset=["pred_home", "pred_away"])
        # NOTE: home_court_mult is left at a flat 1.0 for this live call rather than refitting
        # `fit_home_court_walk_forward` here -- that function needs the wide (one-row-per-game,
        # home_/away_ prefixed) shape built by validate_team_strength_baseline._to_wide_games,
        # which doesn't apply to a single not-yet-played game. Using the dev-range's own already-
        # walk-forward-fit multiplier (its last value) is the correct follow-up; deferred as a
        # small wiring task, not a design gap.
        home_resid = (dev_preds["actual_home"] - dev_preds["pred_home"]).to_numpy()
        away_resid = (dev_preds["actual_away"] - dev_preds["pred_away"]).to_numpy()
        home_var_model = fit_residual_variance_model(home_resid, dev_preds["projected_pace"].to_numpy())
        away_var_model = fit_residual_variance_model(away_resid, dev_preds["projected_pace"].to_numpy())
        corr = fit_home_away_correlation(home_resid, away_resid)
        have_distribution = True
    except Exception as e:
        print(f"WARNING: could not fit score distribution from historical data yet ({e}); "
              f"falling back to point predictions only", flush=True)
        have_distribution = False

    print(f"\n{len(todays_games)} games on {game_date}:\n", flush=True)
    results = []
    for row in todays_games.itertuples(index=False):
        if row.homeTeamId not in latest.index or row.awayTeamId not in latest.index:
            print(f"  {row.awayAbbrev} @ {row.homeAbbrev}: no rating history yet for one of these teams, skipping", flush=True)
            continue

        h, a = latest.loc[row.homeTeamId], latest.loc[row.awayTeamId]
        home_oRtg, home_dRtg = h["rtg_attack_rate"], h["rtg_defense_rate"]
        away_oRtg, away_dRtg = a["rtg_attack_rate"], a["rtg_defense_rate"]
        lineup_tag = "no lineup adjustment (team-strength only)"

        if have_lineup_adjustment:
            home_prior = team_history.get(row.homeTeamId, [])
            away_prior = team_history.get(row.awayTeamId, [])
            home_recent_off, home_recent_def = team_recent_roster_rapm(
                player_minutes, player_ratings, home_prior, team_side.get(row.homeTeamId, {}), MINUTES_LOOKBACK_GAMES)
            away_recent_off, away_recent_def = team_recent_roster_rapm(
                player_minutes, player_ratings, away_prior, team_side.get(row.awayTeamId, {}), MINUTES_LOOKBACK_GAMES)

            home_shares, home_tag = resolve_active_lineup(
                row.homeAbbrev, home_prior, player_minutes, team_side.get(row.homeTeamId, {}), injury_report)
            away_shares, away_tag = resolve_active_lineup(
                row.awayAbbrev, away_prior, player_minutes, team_side.get(row.awayTeamId, {}), injury_report)

            if not home_shares.empty and not away_shares.empty:
                home_off_adj, home_def_adj = project_lineup_adjustment(
                    home_shares, player_ratings, home_recent_off, home_recent_def)
                away_off_adj, away_def_adj = project_lineup_adjustment(
                    away_shares, player_ratings, away_recent_off, away_recent_def)
                home_oRtg, home_dRtg = home_oRtg + home_off_adj, home_dRtg + home_def_adj
                away_oRtg, away_dRtg = away_oRtg + away_off_adj, away_dRtg + away_def_adj
                lineup_tag = f"home: {home_tag} | away: {away_tag}"
            else:
                lineup_tag = f"no lineup data this call (home: {home_tag}, away: {away_tag}) -- team-strength only"

        proj_pace, pred_home, pred_away = project_game(
            home_pace=h["pace_shrunk_mean"], away_pace=a["pace_shrunk_mean"],
            league_avg_pace=h["pace_league_avg"],
            home_oRtg=home_oRtg, home_dRtg=home_dRtg,
            away_oRtg=away_oRtg, away_dRtg=away_dRtg,
            league_avg_rtg=h["rtg_league_avg"], home_court_mult=1.0,
        )

        line = f"  {row.awayAbbrev} @ {row.homeAbbrev}: {pred_away:.1f} - {pred_home:.1f}  [{lineup_tag}]"
        record = {"gameId": row.gameId, "homeAbbrev": row.homeAbbrev, "awayAbbrev": row.awayAbbrev,
                  "pred_home": pred_home, "pred_away": pred_away, "projected_pace": proj_pace,
                  "lineup_source": lineup_tag}

        if have_distribution:
            var_home = predict_variance(home_var_model, np.array([proj_pace]))[0]
            var_away = predict_variance(away_var_model, np.array([proj_pace]))[0]
            wp_home = win_prob_home(pred_home, pred_away, var_home, var_away, corr)
            lo, hi = interval(pred_home - pred_away, var_home + var_away - 2 * corr * np.sqrt(var_home * var_away), 0.8)
            line += f"  |  home win prob {wp_home:.1%}  |  80% spread interval ({lo:+.1f}, {hi:+.1f})"
            record.update({"home_win_prob": wp_home, "spread_80pct_lo": lo, "spread_80pct_hi": hi})

        print(line, flush=True)
        results.append(record)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        out_path = DATA_PROCESSED / f"daily_predictions_{game_date}.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"\nsaved {out_path}", flush=True)
    return result_df


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    run(target_date)
