"""Fetch real current-week NFL player prop odds from The Odds API (review round 2 #4.4 /
round 3 #9's reframing: a small, decisive experiment -- does the already-validated RB
carry-reallocation result (injury_reallocation.py, MAE 4.49->3.71, p=0.0003, this project's
strongest single result) beat a REAL posted line, not just our own prior estimate?

Confirmed 2026-07 (WebFetch against the-odds-api.com's own docs): the cheapest tier with NFL
player props is the $99/mo "Business" plan (200k requests/mo). Per-request cost is
irrelevant at this data volume (a handful of players/week) -- what matters is the monthly
subscription, a real purchase decision left to the user, not made here.

Player props are NOT on The Odds API's bulk sport-level odds endpoint -- that only carries
h2h/spreads/totals. Player props require the per-EVENT odds endpoint
(`/v4/sports/{sport}/events/{event_id}/odds?markets=...`), so fetching them is a two-step
flow: list this week's events, then pull markets for each event.

MOCK MODE (no ODDS_API_KEY set): every fetch_* function returns hand-built sample data
matching the real, documented response shape (confirmed via the outcome-shape example in
The Odds API's own docs: `{"name": "Over", "description": "<player>", "price": <American
odds>, "point": <line>}`), loudly labeled, so the rest of the pipeline (parsing, name
matching, scoring) can be built and tested end-to-end before any subscription exists.
"""

import os

import pandas as pd
import requests

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
RB_MARKETS = ["player_rush_yds", "player_rush_attempts", "player_anytime_td"]

# --- MOCK DATA, matching The Odds API's real documented v4 response shape -------------------
# Two fictional backup RBs, standing in for "a lead back was just ruled out this week."
MOCK_EVENTS = [
    {
        "id": "mock_evt_0001", "sport_key": SPORT_KEY,
        "commence_time": "2026-09-08T17:00:00Z",
        "home_team": "Cleveland Browns", "away_team": "Jacksonville Jaguars",
    },
    {
        "id": "mock_evt_0002", "sport_key": SPORT_KEY,
        "commence_time": "2026-09-08T17:00:00Z",
        "home_team": "Minnesota Vikings", "away_team": "Green Bay Packers",
    },
]

MOCK_EVENT_ODDS = {
    "mock_evt_0001": {
        "id": "mock_evt_0001", "home_team": "Cleveland Browns", "away_team": "Jacksonville Jaguars",
        "bookmakers": [{
            "key": "draftkings", "title": "DraftKings", "last_update": "2026-09-07T12:00:00Z",
            "markets": [{
                "key": "player_rush_yds", "last_update": "2026-09-07T12:00:00Z",
                "outcomes": [
                    {"name": "Over", "description": "Jerome Ford", "price": -115, "point": 44.5},
                    {"name": "Under", "description": "Jerome Ford", "price": -105, "point": 44.5},
                ],
            }],
        }],
    },
    "mock_evt_0002": {
        "id": "mock_evt_0002", "home_team": "Minnesota Vikings", "away_team": "Green Bay Packers",
        "bookmakers": [{
            "key": "fanduel", "title": "FanDuel", "last_update": "2026-09-07T12:00:00Z",
            "markets": [{
                "key": "player_rush_yds", "last_update": "2026-09-07T12:00:00Z",
                "outcomes": [
                    {"name": "Over", "description": "Ty Chandler", "price": -120, "point": 38.5},
                    {"name": "Under", "description": "Ty Chandler", "price": +100, "point": 38.5},
                ],
            }],
        }],
    },
}


def _mock_warning(fn_name: str) -> None:
    print(f"  [MOCK DATA] {fn_name}: no ODDS_API_KEY set -- returning hand-built sample data, "
          f"NOT a real odds fetch. Set ODDS_API_KEY to fetch real current-week lines.")


def fetch_nfl_events(api_key: str | None = None) -> list[dict]:
    """This week's NFL events (needed first -- player props are fetched per-event, not from
    the bulk sport-level odds endpoint)."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        _mock_warning("fetch_nfl_events")
        return MOCK_EVENTS
    resp = requests.get(f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events", params={"apiKey": api_key})
    resp.raise_for_status()
    return resp.json()


def fetch_event_player_props(event_id: str, markets: list[str] = RB_MARKETS, api_key: str | None = None) -> dict:
    """Player prop odds for ONE event. `markets` defaults to the RB volume markets this
    experiment needs (§6.2's reallocation result is about carries/rushing yards)."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        _mock_warning(f"fetch_event_player_props({event_id})")
        if event_id not in MOCK_EVENT_ODDS:
            raise KeyError(f"no mock odds for event_id={event_id!r} -- use one of {list(MOCK_EVENT_ODDS)}")
        return MOCK_EVENT_ODDS[event_id]
    resp = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={"apiKey": api_key, "regions": "us", "markets": ",".join(markets), "oddsFormat": "american"},
    )
    resp.raise_for_status()
    return resp.json()


def parse_player_prop_odds(event_odds: dict, market_key: str = "player_rush_yds") -> pd.DataFrame:
    """One row per (player, bookmaker) with the posted line and both sides' prices --
    real documented outcome shape: {"name": "Over"/"Under", "description": <player>,
    "price": <American odds>, "point": <line>}. Multiple bookmakers for the same player get
    one row each; callers wanting a single consensus line should aggregate (e.g. median
    `point`) themselves -- kept ungrouped here since 'what the sharpest book posts' vs 'what
    the market median is' is a real, non-obvious choice that shouldn't be hidden."""
    rows = []
    for bk in event_odds.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market["key"] != market_key:
                continue
            by_player = {}
            for outcome in market["outcomes"]:
                player = outcome["description"]
                by_player.setdefault(player, {})[outcome["name"]] = (outcome["price"], outcome["point"])
            for player, sides in by_player.items():
                over_price, point = sides.get("Over", (None, None))
                under_price, point_u = sides.get("Under", (None, None))
                rows.append({
                    "event_id": event_odds["id"], "home_team": event_odds["home_team"],
                    "away_team": event_odds["away_team"], "bookmaker": bk["key"],
                    "player_name_raw": player, "market": market_key,
                    "point": point if point is not None else point_u,
                    "over_price": over_price, "under_price": under_price,
                })
    return pd.DataFrame(rows)


def fetch_all_rb_prop_odds(market_key: str = "player_rush_yds", api_key: str | None = None) -> pd.DataFrame:
    """Convenience: every event this week x the given market, parsed into one table."""
    events = fetch_nfl_events(api_key=api_key)
    frames = []
    for event in events:
        odds = fetch_event_player_props(event["id"], markets=[market_key], api_key=api_key)
        parsed = parse_player_prop_odds(odds, market_key=market_key)
        if not parsed.empty:
            frames.append(parsed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
