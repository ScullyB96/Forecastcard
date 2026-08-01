"""Pull the season-by-season game log (schedule + final scores) from
stats.nba.com via `nba_api` and cache it locally as parquet.

Confirmed live (2026-07-24): plain `requests` gets a 403 from stats.nba.com
(needs browser-like headers) but `nba_api`'s `LeagueGameLog` endpoint (which
sets those headers internally) works with no auth. `LeagueGameLog` returns
one row per TEAM per game (two rows per game, matching MATCHUP's "@"/"vs."
side), which is exactly what's needed to build a schedule with both teams'
final scores without a separate box-score call.

Season-boundary detection lesson carried from the MLB/NHL siblings (both
learned this the hard way): don't hardcode a calendar-to-season mapping.
`current_nba_season()` instead asks the schedule itself which season has
games, and picks the latest one with any row -- correct across shifted
openers, in-progress seasons, and the summer off-season gap alike.
"""

import datetime as _dt
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog

from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 0.6  # unofficial stats.nba.com convention (no published rate limit)
MAX_FETCH_RETRIES = 3
FIRST_DEV_SEASON = 2015  # 2015-16, start of the modern 3PT/pace-and-space era (user-confirmed)
SEASON_TYPES = ("Regular Season", "Playoffs")


def season_str(start_year: int) -> str:
    """2015 -> '2015-16'."""
    return f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"


def season_for_date(date_str: str) -> int:
    """Pure calendar-based season lookup for an ARBITRARY date (past,
    present, or a genuine live "tonight") -- unlike `current_nba_season`,
    this makes no live API call and doesn't depend on wall-clock "now", so
    it's safe to use for historical backtesting. August is used as the
    cutoff month (not October, the actual season start) as a deliberately
    safe buffer -- the real season never starts before October or extends
    past July, so any date in August or later of year Y belongs to the
    season labeled Y-(Y+1); any date before August belongs to the season
    that started the PRIOR calendar year.

    KNOWN EDGE CASE, not handled: the COVID-disrupted 2019-20 season
    actually ran into October 2020 (Finals concluded October 11, 2020), so
    a date in the ~10-day window between this function's August 1 cutoff
    and that season's real end would be misclassified as 2020-21. A
    narrow, known, single-season historical anomaly -- not worth a
    schedule-aware lookup for one 10-day window; a caller who genuinely
    needs to backtest a date in that specific window should hardcode
    `start_year=2019` for that call instead of using this function."""
    d = pd.Timestamp(date_str)
    return d.year if d.month >= 8 else d.year - 1


def _fetch_with_retry(start_year: int, season_type: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            lgl = leaguegamelog.LeagueGameLog(
                season=season_str(start_year),
                season_type_all_star=season_type,
                timeout=30,
            )
            return lgl.get_data_frames()[0]
        except (requests.exceptions.RequestException, KeyError) as e:
            last_err = e
            print(f"  fetch {season_str(start_year)} {season_type} failed "
                  f"(attempt {attempt + 1}/{MAX_FETCH_RETRIES}): {e}, retrying...", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LeagueGameLog fetch failed after {MAX_FETCH_RETRIES} attempts: {last_err}")


def _team_rows_to_games(df: pd.DataFrame, season_start_year: int, season_type: str) -> pd.DataFrame:
    """Collapse LeagueGameLog's 2-rows-per-game (one per team) into one row
    per game with home/away resolved from MATCHUP's '@' (away) vs 'vs.' (home)."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["is_home"] = df["MATCHUP"].str.contains("vs.", regex=False)
    games = {}
    for _, row in df.iterrows():
        gid = row["GAME_ID"]
        g = games.setdefault(gid, {
            "gameId": gid,
            "gameDate": row["GAME_DATE"],
            "season": season_start_year,
            "seasonType": season_type,
        })
        side = "home" if row["is_home"] else "away"
        g[f"{side}TeamId"] = row["TEAM_ID"]
        g[f"{side}TeamAbbrev"] = row["TEAM_ABBREVIATION"]
        g[f"{side}Score"] = row["PTS"]
        g[f"{side}WL"] = row["WL"]
    out = pd.DataFrame(games.values())
    # a handful of in-progress/edge games may be missing one side; drop those defensively
    before = len(out)
    out = out.dropna(subset=["homeTeamId", "awayTeamId", "homeScore", "awayScore"])
    if len(out) != before:
        print(f"  dropped {before - len(out)} game(s) missing a home or away side", flush=True)
    return out.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def fetch_season(start_year: int, force: bool = False) -> pd.DataFrame:
    path = DATA_RAW / f"schedule_{season_str(start_year)}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    frames = []
    for season_type in SEASON_TYPES:
        raw = _fetch_with_retry(start_year, season_type)
        time.sleep(REQUEST_DELAY_SECONDS)
        games = _team_rows_to_games(raw, start_year, season_type)
        print(f"  {season_str(start_year)} {season_type}: {len(games)} games", flush=True)
        frames.append(games)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_parquet(path, index=False)
    print(f"saved {path.name}: {len(df)} games total", flush=True)
    return df


def current_nba_season() -> int:
    """Resolve the current season's start year by asking the live schedule
    which season currently has any games, rather than assuming from today's
    calendar date (correct through the summer off-season gap and through any
    shifted opener). An NBA season labeled "2015-16" is queried as start_year
    2015 regardless of which calendar year today falls in, so the two
    plausible candidates (this calendar year and last) are checked newest
    first and the first one with any regular-season row wins."""
    guess = _dt.date.today().year
    for candidate in (guess, guess - 1):
        df = _fetch_with_retry(candidate, "Regular Season")
        time.sleep(REQUEST_DELAY_SECONDS)
        if len(df) > 0:
            return candidate
    raise RuntimeError("could not resolve current NBA season from live schedule")


def fetch_all_seasons(end_year: int, force: bool = False) -> dict[int, pd.DataFrame]:
    return {y: fetch_season(y, force=force) for y in range(FIRST_DEV_SEASON, end_year + 1)}


if __name__ == "__main__":
    current = current_nba_season()
    print(f"current_nba_season() -> {current} ({season_str(current)})", flush=True)
    all_seasons = fetch_all_seasons(current)
    total = sum(len(df) for df in all_seasons.values())
    print(f"fetched {len(all_seasons)} seasons, {total} games total", flush=True)
