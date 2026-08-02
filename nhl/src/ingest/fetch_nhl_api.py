"""Pull the full NHL schedule + final scores from the public api-web.nhle.com
API and cache it locally as parquet.

Confirmed live (2026-07-23, see MODEL_DOCUMENTATION.md Sec1.1): no auth/key
needed. `GET /v1/schedule/{date}` returns a full 7-day window (`gameWeek`,
verified 7 distinct dates for one request) with `nextStartDate` pointing
exactly one week later -- so a single continuous week-by-week walk from
before the 2008-09 season to today collects the entire schedule in ~1000
requests, without needing to know each season's exact start/end date (which
vary year to year -- lockouts, COVID-shortened seasons, play-in formats).
gameType is kept (1=preseason, 2=regular, 3=playoff) rather than filtered
here, so downstream code decides what to include.

Added `lastPeriodType` (2026-07-23, for the OT/SO sub-model -- MODEL_DOCUMENTATION.md
Sec4.12): confirmed live that `gameOutcome.lastPeriodType` (REG/OT/SO) is already present in
this same schedule endpoint -- no separate play-by-play ingestion needed. Combined with the
final score and which team won, this is enough to reconstruct the REGULATION-only score
directly: exactly one bonus goal (the OT winner or the shootout's awarded goal) is added to
the winning team's tally for any OT/SO-decided game, confirmed against real games (e.g.
2024-01-15 ANA 5 @ FLA 4, OT -- ANA scored the winner, so regulation was 4-4; CBJ 4 @ VAN 3,
SO -- CBJ's shootout goal is the +1, so regulation was 3-3)."""

import time

import pandas as pd
import requests

from src.utils.paths import DATA_RAW

BASE_URL = "https://api-web.nhle.com/v1/schedule"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
REQUEST_DELAY_SECONDS = 0.2
SCHEDULE_COLUMNS = [
    "gameId", "season", "gameType", "gameDate", "startTimeUTC", "gameState",
    "venue", "lastPeriodType", "awayTeamId", "awayTeamAbbrev", "awayScore",
    "homeTeamId", "homeTeamAbbrev", "homeScore",
]


def _fetch_week(date: str) -> dict:
    resp = requests.get(f"{BASE_URL}/{date}", headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _flatten_games(game_week: list[dict]) -> list[dict]:
    rows = []
    for day in game_week:
        for g in day.get("games", []):
            away, home = g.get("awayTeam", {}), g.get("homeTeam", {})
            rows.append({
                "gameId": g["id"],
                "season": g["season"],
                "gameType": g["gameType"],
                "gameDate": day["date"],
                "startTimeUTC": g.get("startTimeUTC"),
                "gameState": g.get("gameState"),
                "venue": (g.get("venue") or {}).get("default"),
                "lastPeriodType": (g.get("gameOutcome") or {}).get("lastPeriodType"),
                "awayTeamId": away.get("id"),
                "awayTeamAbbrev": away.get("abbrev"),
                "awayScore": away.get("score"),
                "homeTeamId": home.get("id"),
                "homeTeamAbbrev": home.get("abbrev"),
                "homeScore": home.get("score"),
            })
    return rows


def fetch_schedule_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Walks week-by-week from start_date until nextStartDate would exceed
    end_date (or stops advancing, e.g. an API error), deduping by gameId."""
    all_rows: dict[int, dict] = {}
    current = start_date
    seen_dates = set()
    while current <= end_date and current not in seen_dates:
        seen_dates.add(current)
        data = _fetch_week(current)
        for row in _flatten_games(data.get("gameWeek", [])):
            all_rows[row["gameId"]] = row
        next_date = data.get("nextStartDate")
        time.sleep(REQUEST_DELAY_SECONDS)
        if not next_date:
            break
        current = next_date
    # task #164 (2026-08-02): a date window with zero scheduled games (e.g. a
    # deep off-season day) leaves all_rows empty -- pd.DataFrame({}.values())
    # produces a columnless frame, which crashes sort_values(["gameDate", ...])
    # with a real KeyError. Only hit in production via games_on_date's
    # single-day query during the off-season; the multi-week historical
    # backfill window is never empty. Same schema either way so callers
    # (e.g. games_on_date's `week["gameDate"] == game_date` filter) don't
    # need to special-case an empty result.
    if not all_rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    return pd.DataFrame(all_rows.values()).sort_values(["gameDate", "gameId"]).reset_index(drop=True)


if __name__ == "__main__":
    # 2008-08-01 predates the 2008-09 season opener; walks through today.
    df = fetch_schedule_range("2008-08-01", "2026-07-23")
    print(f"fetched {len(df)} games, seasons {sorted(df['season'].unique())[:3]}...{sorted(df['season'].unique())[-3:]}")
    print(df["gameType"].value_counts())
    out_path = DATA_RAW / "nhl_schedule_2008_2026.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
