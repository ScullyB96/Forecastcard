"""Live NHL closing-line odds capture -- the piece Sec37.7 of
MODEL_DOCUMENTATION.md calls for: log the real market line for every game
this project prices, starting opening night, so the live model-vs-market
gap (Sec37.5/37.6) can finally be measured against odds this project
captured itself in real time, rather than the frozen 2008-2022 SBRO
archive (`fetch_sbro_odds.py`) that every market number in this project to
date has relied on.

CONFIRMED LIVE 2026-07-25 (via browser network inspection): the NHL odds
page (`hockey/nhl-odds.php`) makes a client-side fetch to --

    GET https://www.rotowire.com/betting/nhl/tables/nhl-games.php?date=YYYY-MM-DD

-- no auth, no API key, and the URL genuinely accepts an arbitrary `date`
query param (confirmed: navigating directly to it with today's real date
and with a real past date both returned valid, distinctly-different-URL
responses, not a generic redirect).

**IMPORTANT, HONEST LIMITATION -- NOT YET VERIFIED against real, POPULATED
data**: every date tried during this investigation (2026-07-25, the real
live "today"; and 2026-04-06, a real past game date) returned an empty
JSON array `[]`. This is consistent with two different explanations that
could NOT be distinguished from outside the off-season: (a) RotoWire's own
odds board is empty right now because no sportsbook has posted lines this
far before a game (plausible for a same-day, real-time board), or (b) this
endpoint doesn't serve past dates at all and only ever has data for
literally the current, imminent slate. Because of this, **the exact JSON
field names/shape below are inferred from this project's own established
convention for a betting-odds table (moneyline/spread/total, open and
close), NOT confirmed against a real non-empty response** -- unlike every
other `fetch_*.py` in this project, which only ever documents fields
actually observed. Treat `parse_games_response` as a best-effort placeholder
until it can be run against a real populated response and corrected.

**Action required before this is trustworthy**: re-run
`fetch_current_odds()` against a real date once the season is close enough
that RotoWire's board is actually populated (expect this works some time
after preseason odds post, plausibly early-to-mid September onward) and
fix `parse_games_response` to match whatever real fields actually come
back -- do not assume the placeholder below is correct without that check.
"""

import json
from datetime import date, datetime, timezone

import requests

from src.utils.paths import DATA_RAW

ODDS_URL = "https://www.rotowire.com/betting/nhl/tables/nhl-games.php"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_current_odds(game_date: str | None = None) -> list[dict]:
    """`game_date`: "YYYY-MM-DD", defaults to today. Returns the RAW JSON
    response (list of dicts, shape UNVERIFIED -- see module docstring)
    and archives a timestamped snapshot to DATA_RAW regardless of shape,
    so a real populated response captured once the season starts is never
    lost even before `parse_games_response` is corrected to match it."""
    game_date = game_date or date.today().isoformat()
    resp = requests.get(ODDS_URL, params={"date": game_date},
                        headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    rows = resp.json()

    snapshot_dir = DATA_RAW / "rotowire_odds_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(snapshot_dir / f"odds_{game_date}_{ts}.json", "w") as f:
        json.dump(rows, f)

    return rows


def parse_games_response(rows: list[dict]) -> list[dict]:
    """PLACEHOLDER, UNVERIFIED -- see module docstring. Best-effort parse
    assuming a shape consistent with this project's own SBRO convention
    (moneyline + total, open + close) until a real populated response is
    available to correct this against. Passes through whatever keys exist
    rather than raising, so a real (differently-shaped) response doesn't
    crash -- but callers should NOT trust the specific field names below
    (`home_ml`/`away_ml`/`total`/etc.) until this function is re-verified."""
    parsed = []
    for row in rows:
        parsed.append({
            "gameId": row.get("gameID") or row.get("gameId") or row.get("ID"),
            "homeTeam": row.get("homeTeam") or row.get("home"),
            "awayTeam": row.get("awayTeam") or row.get("away"),
            "home_ml": row.get("homeML") or row.get("home_ml"),
            "away_ml": row.get("awayML") or row.get("away_ml"),
            "total": row.get("total") or row.get("overUnder"),
            "raw": row,  # keep the untouched original alongside the best-effort parse
        })
    return parsed


if __name__ == "__main__":
    today_rows = fetch_current_odds()
    print(f"fetched {len(today_rows)} raw odds rows for {date.today().isoformat()}", flush=True)
    if today_rows:
        print("sample raw row:", today_rows[0], flush=True)
        print("best-effort parse:", parse_games_response(today_rows[:1]), flush=True)
    else:
        print("EMPTY response -- expected during the off-season (see module docstring); "
              "re-run this closer to the season and confirm parse_games_response against "
              "whatever real fields come back", flush=True)
