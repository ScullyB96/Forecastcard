"""Live daily entry point: refresh data, resolve tonight's active lineups,
run the Phase 1 (team strength) + Phase 3 (score distribution) stack, and
write out win probability / spread / total for every game on a given date.

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
renormalized across tonight's active roster.

**Phase 2 (RAPM-lite lineup adjustment) is DISABLED as of 2026-08-01**
(`INCLUDE_LINEUP_ADJUSTMENT = False`), reverting a 2026-07-25 decision.
The dev-only bootstrap that originally justified wiring it in
(`validate_predictive_lineup_adjustment.py`) had a real hindsight leak in
its own active-player-set selection (it used the real historical game's
actual attendance -- perfect foreknowledge of who plays -- instead of the
same trailing-rotation-union `resolve_active_lineup` genuinely has to work
with, live). Fixed and re-ran (Sec28.2 of MODEL_DOCUMENTATION.md): under
the honest methodology, Phase 2 shows a REAL REGRESSION on margin_mae on
BOTH dev (+0.0146) and holdout (+0.0181), reversing the original "real
improvement" conclusion. Oracle mode (unaffected by the leak) still
confirms lineup-awareness carries real signal in principle -- the
deployable minutes-share estimation mechanism just doesn't capture it net
of its own noise. All the RAPM-lite/lineup-adjustment machinery
(`rapm_lite.py`, `lineup_rating.py`, `active_roster.py`) is left in place,
tested, and available -- flip `INCLUDE_LINEUP_ADJUSTMENT` back on only
after a genuinely improved minutes-projection mechanism is built and
re-validated, not by reverting this decision on its own.

**Non-regular-season games are flagged, not silently predicted as if equivalent (2026-08-02)**:
every training signal in this codebase (`team_strength.build_team_game_log`,
`team_stat_rates.build_team_stat_game_log`) is regular-season-only -- playoff rotations, pace, and
home-court advantage are all documented to shift in the postseason, and this model has never been
validated against that regime. `active_roster.games_on_date` now classifies each game's type from
its `gameId` prefix; a non-regular-season game gets `regime_warning: "non_regular_season_untested_regime"`
on its output row instead of being predicted as if it were an ordinary night. This does not change
any regular-season prediction -- the ratings/combine/distribution math is completely unaffected --
it only surfaces an honest caveat on the (currently rare, playoffs-only) games where the model's
own training assumption doesn't hold.
"""

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.ingest.build_stints import build_season_stints
from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report, warn_if_stale_for_backtest
from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_for_date, season_str
from src.models.lineup_rating import player_minutes_from_stints, project_lineup_adjustment, team_recent_roster_rapm
from src.models.rapm_lite import fit_rapm, prepare_stints
from src.models.score_distribution import (
    compute_walkforward_variance_model, fit_home_away_correlation, interval, predict_variance_walk_forward,
    win_prob_home,
)
from src.models.home_court import fit_home_court_walk_forward
from src.models.team_strength import add_team_ratings, build_team_game_log, project_game
from src.models.validate_team_strength_baseline import _to_wide_games
from src.pipeline.active_roster import (
    MINUTES_LOOKBACK_GAMES, build_team_history, games_on_date, load_current_roster_player_ids, resolve_active_lineup,
)
from src.pipeline.refresh_data import refresh_all_data
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.tz import default_slate_date

INCLUDE_LINEUP_ADJUSTMENT = False  # disabled 2026-08-01 -- see module docstring and
# MODEL_DOCUMENTATION.md Sec28.2: the dev/holdout result that justified adopting this had a real
# hindsight leak in its own validation; fixed and re-run, Phase 2 now shows a real regression.


def _fit_latest_player_ratings(current_season: int, before_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_ratings, player_minutes) using every cached season's
    stints through `current_season` -- a single fresh RAPM-lite fit "as of
    right now" (or as of `before_date`, if given -- see module docstring's
    note on historical backtesting), not the periodic backtest series.
    Seasons whose stints haven't been built yet (or whose raw data isn't
    cached) are skipped, not fatal -- an early-season call with less
    history just fits on whatever's available, same as any other walk-
    forward fit in this project.

    `before_date`: when set, the schedule (and therefore every stint merged
    against it) is filtered to `gameDate < before_date` BEFORE fitting --
    critical for a trustworthy historical backtest, since without this the
    "current" RAPM/minutes snapshot silently reaches past the target date
    into whatever's most recent as of the real wall-clock today (see
    MODEL_DOCUMENTATION.md's writeup of this exact bug, found via a real
    2025-01-15 spot-check)."""
    frames = []
    for y in range(FIRST_DEV_SEASON, current_season + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        if not sched_path.exists():
            continue
        stints = build_season_stints(y)
        if stints.empty:
            continue
        schedule = pd.read_parquet(sched_path)
        schedule = schedule[schedule["seasonType"] == "Regular Season"]
        if before_date is not None:
            schedule = schedule[pd.to_datetime(schedule["gameDate"]) < pd.Timestamp(before_date)]
            if schedule.empty:
                continue
        frames.append(prepare_stints(stints, schedule))
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


def _latest_home_court_mult(team_log: pd.DataFrame) -> float:
    """The most recent walk-forward-fit home-court multiplier, computed
    from the SAME `team_log` this call already built (already filtered to
    `gameDate < game_date`, so this is walk-forward safe) -- NOT the
    hardcoded 1.0 this function replaces.

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): `run()`
    previously called `project_game(..., home_court_mult=1.0)` literally,
    discarding the empirically-validated home-court effect (Sec4: home
    teams average 106.2 vs away 103.5 points, 58.4% home win rate) that
    every validation/backtest script in this project actually fits and
    applies. Two equally-rated teams got an IDENTICAL projected score
    regardless of who was home -- the live pipeline could not express a
    home-court edge at all, and `win_prob_home`/`interval` (fit on
    residuals that DO assume the correction) were applied to a
    systematically biased mean. `team_log` already has every column
    `home_court.fit_home_court_walk_forward` needs (same construction
    path as `validate_team_strength_baseline._rated_dev_log`, just spanning
    through `target_season` instead of stopping at `DEV_MAX_SEASON - 1`)
    once reshaped wide via that module's own `_to_wide_games` helper.
    Falls back to 1.0 (the old behavior) only if there's no game history
    yet to fit from -- an honest "no correction available", not a silent
    wrong number."""
    games = _to_wide_games(team_log)
    if games.empty:
        return 1.0
    mult = fit_home_court_walk_forward(games).iloc[-1]
    return float(mult) if pd.notna(mult) else 1.0


def _full_range_variance_checkpoints(team_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Task #56 (MODEL_DOCUMENTATION.md Sec51): walk-forward periodic-refit
    variance checkpoints (home, away) + a single home/away residual
    correlation, built from the SAME through-`game_date` `team_log`
    `_latest_home_court_mult` already uses -- NOT the dev-range-only static
    fit this replaces.

    REAL STALENESS BUG FOUND AND FIXED here: the static fit this replaces
    (`validate_team_strength_baseline.build_dev_predictions()`) stops at
    `DEV_MAX_SEASON - 1` (currently 2023) regardless of `game_date` -- so
    every live call in 2024-2026 was quietly excluding 2+ REAL,
    ALREADY-CACHED seasons of residual history from its own variance/
    interval calibration, the exact same shape of bug `_latest_home_court_mult`
    was already built to fix for the home-court multiplier. This ALSO
    resolves Sec45.3's declining-margin-coverage-trend finding (evaluated
    walk-forward-honestly instead of via a single full-range fit with
    look-ahead information, the trend weakens from real/significant to
    NOISE -- see Sec51) -- a strict improvement on both axes, not a
    tradeoff. Falls back to an empty checkpoint frame if there's no game
    history yet to fit from; callers must check `.empty` the same way
    `have_distribution` already gates on a fit failure."""
    games = _to_wide_games(team_log)
    if games.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0
    games = games.copy()
    games["home_court_mult"] = fit_home_court_walk_forward(games).fillna(1.0)

    rows = []
    for row in games.itertuples(index=False):
        proj_pace, pred_home, pred_away = project_game(
            home_pace=row.pace_shrunk_mean_home, away_pace=row.pace_shrunk_mean_away,
            league_avg_pace=row.league_avg_pace,
            home_oRtg=row.rtg_attack_rate_home, home_dRtg=row.rtg_defense_rate_home,
            away_oRtg=row.rtg_attack_rate_away, away_dRtg=row.rtg_defense_rate_away,
            league_avg_rtg=row.league_avg_rtg, home_court_mult=row.home_court_mult,
        )
        rows.append({"gameDate": row.gameDate, "projected_pace": proj_pace,
                      "home_resid": row.actual_home_score - pred_home, "away_resid": row.actual_away_score - pred_away})
    preds = pd.DataFrame(rows).dropna(subset=["home_resid", "away_resid"])
    if preds.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    home_checkpoints = compute_walkforward_variance_model(
        preds.rename(columns={"home_resid": "residual", "projected_pace": "pace"})[["gameDate", "residual", "pace"]])
    away_checkpoints = compute_walkforward_variance_model(
        preds.rename(columns={"away_resid": "residual", "projected_pace": "pace"})[["gameDate", "residual", "pace"]])
    corr = fit_home_away_correlation(preds["home_resid"].to_numpy(), preds["away_resid"].to_numpy())
    return home_checkpoints, away_checkpoints, corr


def run(game_date: str) -> pd.DataFrame:
    refresh_all_data()  # ensure real-time data is backfilled; season logic below uses game_date, not "today"
    # `target_season` is derived PURELY from `game_date` (see `season_for_date`'s docstring), never
    # from wall-clock "now" -- critical for a trustworthy historical backtest. A real bug was found
    # here (2026-08-01, via a real 2025-01-15 spot-check): using the LIVE "current season" for a
    # historical `game_date` silently pulled trailing minutes/ratings from whatever's most recent
    # as of the real today, not as of `game_date` -- e.g. projecting a star player for ~11 minutes
    # instead of their true ~38-minute trailing average, because the "trailing 10 games" reached
    # into a much later, unrelated season. See MODEL_DOCUMENTATION.md for the full writeup.
    target_season = season_for_date(game_date)

    todays_games = games_on_date(game_date)
    if todays_games.empty:
        print(f"no games found for {game_date}", flush=True)
        return pd.DataFrame()

    warn_if_stale_for_backtest(game_date)
    injury_report = fetch_current_injury_report()
    team_log = add_team_ratings(build_team_game_log(2015, target_season))
    team_log = team_log[pd.to_datetime(team_log["gameDate"]) < pd.Timestamp(game_date)]
    latest = _latest_team_ratings(team_log)
    home_court_mult = _latest_home_court_mult(team_log)
    # `team_history`/`team_side` feed active-roster/minutes resolution and the "recent roster
    # composite" RAPM lookback -- BOTH represent "who is on this team's roster right now", which
    # must never blend across a season boundary (unlike the team-level PACE/RATING above, which
    # has its own intentional cross-season option elsewhere). Real bug found via a 2023-11-08
    # early-season spot-check: without this restriction, the "last 10 games" lookback silently
    # pulled in games (and therefore players) from the PRIOR season, diluting the active roster
    # with players no longer even on the team -- confirmed directly (DAL's trailing 10 games as of
    # 2023-11-08 included 3 games from 2022-23). See MODEL_DOCUMENTATION.md for the full writeup.
    team_history, team_side = build_team_history(team_log[team_log["season"] == target_season])
    current_roster = load_current_roster_player_ids(target_season)

    if INCLUDE_LINEUP_ADJUSTMENT:
        player_ratings, player_minutes = _fit_latest_player_ratings(target_season, before_date=game_date)
        have_lineup_adjustment = not player_ratings.empty
        if not have_lineup_adjustment:
            print("WARNING: no RAPM-lite player-rating snapshot available yet (stints not built for "
                  "any cached season) -- falling back to team-strength-only (Phase 1) projections "
                  "for this call", flush=True)
    else:
        # Phase 2 disabled (see module docstring/Sec28.2) -- skip the RAPM-lite fit entirely
        # rather than computing it and then discarding it.
        player_ratings, player_minutes = pd.DataFrame(), pd.DataFrame()
        have_lineup_adjustment = False

    # Score-distribution variance/correlation is now a genuine walk-forward periodic refit
    # (`_full_range_variance_checkpoints`, task #56/Sec51) -- spans the SAME through-`game_date`
    # `team_log` used for `home_court_mult`, not the dev-range-only static fit this replaces (see
    # that function's own docstring for the staleness bug this fixes).
    try:
        home_checkpoints, away_checkpoints, corr = _full_range_variance_checkpoints(team_log)
        have_distribution = not home_checkpoints.empty and not away_checkpoints.empty
        if not have_distribution:
            print("WARNING: not enough historical residual data yet to fit any variance checkpoint; "
                  "falling back to point predictions only", flush=True)
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
                row.homeAbbrev, home_prior, player_minutes, team_side.get(row.homeTeamId, {}), injury_report,
                current_roster_ids=current_roster.get(row.homeTeamId))
            away_shares, away_tag = resolve_active_lineup(
                row.awayAbbrev, away_prior, player_minutes, team_side.get(row.awayTeamId, {}), injury_report,
                current_roster_ids=current_roster.get(row.awayTeamId))

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
            league_avg_rtg=h["rtg_league_avg"], home_court_mult=home_court_mult,
        )

        game_type = getattr(row, "gameType", "regular_season")
        is_regular_season = game_type == "regular_season"
        regime_note = "" if is_regular_season else "  [WARNING: non-regular-season game -- model trained on regular season only]"

        line = f"  {row.awayAbbrev} @ {row.homeAbbrev}: {pred_away:.1f} - {pred_home:.1f}  [{lineup_tag}]{regime_note}"
        record = {"gameId": row.gameId, "homeAbbrev": row.homeAbbrev, "awayAbbrev": row.awayAbbrev,
                  "pred_home": pred_home, "pred_away": pred_away, "projected_pace": proj_pace,
                  "lineup_source": lineup_tag, "gameType": game_type,
                  "regime_warning": None if is_regular_season else "non_regular_season_untested_regime"}

        if have_distribution:
            game_date_series = pd.Series([game_date])
            var_home = predict_variance_walk_forward(home_checkpoints, game_date_series, np.array([proj_pace]))[0]
            var_away = predict_variance_walk_forward(away_checkpoints, game_date_series, np.array([proj_pace]))[0]
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
    from src.utils.nba_proxy import configure_proxy
    configure_proxy()
    target_date = sys.argv[1] if len(sys.argv) > 1 else str(default_slate_date())
    run(target_date)
