"""Live daily entry point: refresh data, resolve the real games scheduled
on a given date, price each one with `predict_game.py`, and log the
output. The first live orchestration layer this project has had (Sec41-42
of MODEL_DOCUMENTATION.md).

Run as `python -m src.pipeline.generate_predictions [YYYY-MM-DD]` (defaults
to today).

**Preseason games are priced too, deliberately** -- not because their
predictions are meaningful (rosters/effort are out-of-population, per
Sec39.3/Sec41), but because the Sep 19-20 dress rehearsal's whole purpose
is exercising this exact machinery before real regular-season games matter
(see MODEL_DOCUMENTATION.md Sec39.3's own framing). Every output row is
tagged with `gameType` so a caller can filter preseason predictions out of
anything that matters, without this script needing to know the calendar.

**Known limitation, not silently hidden**: `predict_game._global_constants`
rebuilds the ENTIRE historical fit (situational log, goalie log, dev-only
constants) from scratch on every call to this script -- correctness over
efficiency for v1, matching this project's own established pattern for
the score-distribution/home-court fits elsewhere. Worth caching to a
persisted artifact if daily latency becomes a real problem once this runs
every day of the season (not yet measured as one).
"""

import sys
from datetime import date

import pandas as pd

from src.ingest.fetch_nhl_api import fetch_schedule_range
from src.ingest.fetch_rotowire_injuries import fetch_current_injury_report
from src.models.predict_game import _global_constants, _load_schedule, predict_game
from src.models.prediction_log import append_prediction
from src.pipeline.refresh_data import refresh_all
from src.utils.paths import DATA_PROCESSED


def games_on_date(game_date: str) -> pd.DataFrame:
    """Live fetch (not the cache -- works for today/future dates the
    historical cache was never meant to contain) of every real game
    scheduled on `game_date`, any `gameType` (1=preseason, 2=regular,
    3=playoff)."""
    week = fetch_schedule_range(game_date, game_date)
    return week[week["gameDate"] == game_date]


def run(game_date: str) -> pd.DataFrame:
    refresh_all()

    todays_games = games_on_date(game_date)
    if todays_games.empty:
        print(f"no games found for {game_date}", flush=True)
        return pd.DataFrame()

    schedule = _load_schedule()
    state = _global_constants(schedule)

    # Fetched ONCE per run (not once per game) -- same real-time injury snapshot applies to
    # every game priced in this call, and gets logged alongside each one (Sec42.2) so the exact
    # input a live prediction was made from is reconstructable later.
    injury_report = fetch_current_injury_report()

    print(f"\n{len(todays_games)} games on {game_date}:\n", flush=True)
    results = []
    for row in todays_games.itertuples(index=False):
        try:
            pred = predict_game(row.homeTeamAbbrev, row.awayTeamAbbrev, game_date, state=state,
                                 injury_report=injury_report)
        except ValueError as e:
            print(f"  {row.awayTeamAbbrev} @ {row.homeTeamAbbrev}: skipped ({e})", flush=True)
            continue

        gametype_tag = {1: "PRESEASON -- not a real prediction", 2: "regular season", 3: "playoff"}.get(
            row.gameType, f"gameType={row.gameType}")
        line = (f"  {pred['away_team']} @ {pred['home_team']}: "
                f"{pred['lambda_away']:.2f} - {pred['lambda_home']:.2f}  "
                f"[home win prob {pred['home_win_prob_full']:.1%}]  ({gametype_tag})")
        print(line, flush=True)
        results.append({**pred, "gameId": row.gameId, "gameType": row.gameType})

        # Logged immediately per game, before puck drop -- the immutable record Sec37's
        # live-vs-market series and degradation budget both assume exists (Sec42.2).
        append_prediction(
            game_id=row.gameId, game_date=game_date, home_team=pred["home_team"], away_team=pred["away_team"],
            lambda_home=pred["lambda_home"], lambda_away=pred["lambda_away"],
            home_win_prob_full=pred["home_win_prob_full"],
            home_starter_goalie_id=pred["home_starter_goalie_id"], away_starter_goalie_id=pred["away_starter_goalie_id"],
            injury_report_snapshot=injury_report, game_type=row.gameType,
        )

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        out_path = DATA_PROCESSED / f"daily_predictions_{game_date}.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"\nsaved {out_path} (mutable convenience snapshot -- the immutable record is "
              f"data/processed/live_prediction_log.jsonl)", flush=True)
    return result_df


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    run(target_date)
