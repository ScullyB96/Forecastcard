"""Slack Incoming Webhook notifications: a compact daily-picks digest,
posted once per sport as the tail end of that sport's own daily pipeline,
with a link back to the full slate/props breakdown on the site.

Deliberately NOT a full Slack bot (no OAuth install, no bot token, no
event subscriptions) -- a one-way "post today's picks" digest is exactly
what an Incoming Webhook is for: one secret URL, no app-install flow,
posts as a normal message into one fixed channel. Upgrade to a real Slack
App with a bot token only if two-way interactivity (buttons, slash
commands) is ever wanted -- not needed for this.
"""

import requests

from app import db

SPORT_LABELS = {"mlb": "MLB", "nfl": "NFL", "nba": "NBA", "nhl": "NHL"}
MAX_GAMES_SHOWN = 8  # keep the digest scannable -- the site link is where full depth lives


def _confidence_rank(game: dict) -> float:
    """Sort key: how far this game's win probability sits from a
    coinflip -- plain model CONVICTION (distinct from _confidence_for_game
    in main.py, which scores DATA QUALITY, not the prediction itself).
    Used only to choose which games lead the digest when a slate has more
    than MAX_GAMES_SHOWN games; the site itself still shows every game."""
    wp = game.get("home_win_prob")
    return abs((wp if wp is not None else 0.5) - 0.5)


def _format_game_line(game: dict) -> str:
    home_wp, away_wp = game.get("home_win_prob"), game.get("away_win_prob")
    home, away = game.get("home_team", "?"), game.get("away_team", "?")
    if home_wp is None or away_wp is None:
        return f"*{away} @ {home}* -- win probability not yet available"
    favorite, prob = (home, home_wp) if home_wp >= away_wp else (away, away_wp)
    total = game.get("projected_total")
    total_txt = f" - Proj total {total:.1f}" if total is not None else ""
    return f"*{away} @ {home}* -- {favorite} {prob:.0%}{total_txt}"


def build_slack_message(sport: str, slate_key: str, games: list[dict], site_url: str) -> dict:
    """A Slack Block Kit payload: a header, one line per game (most
    lopsided picks first, capped at MAX_GAMES_SHOWN), and a link to the
    site for the full slate/props breakdown. `text` is set alongside
    `blocks` for notification previews/fallback clients that don't render
    Block Kit."""
    label = SPORT_LABELS.get(sport, sport.upper())
    if not games:
        text = f"{label} -- no games in today's slate ({slate_key})"
        return {"text": text, "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]}

    ranked = sorted(games, key=_confidence_rank, reverse=True)
    shown = ranked[:MAX_GAMES_SHOWN]
    lines = [_format_game_line(g) for g in shown]
    omitted = len(games) - len(shown)
    if omitted > 0:
        lines.append(f"_+{omitted} more game{'s' if omitted != 1 else ''} on the site_")

    header = f"{label} picks -- {slate_key} ({len(games)} game{'s' if len(games) != 1 else ''})"
    link = f"{site_url.rstrip('/')}/{sport}/{slate_key}"
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"<{link}|Full breakdown & player props>"}},
    ]
    return {"text": header, "blocks": blocks}


def post_daily_picks(sport: str, slate_key: str | None, webhook_url: str, site_url: str) -> dict:
    """Fetches today's slate for `sport` (or the latest one, if slate_key
    is None) and posts the digest to `webhook_url` -- but ONLY the first
    time this (sport, slate_key) pair is ever seen (see db.mark_notified).
    A sport's own cron can call the /internal/notify trigger several times
    a day as it refreshes the SAME slate intraday (MLB 4x, NBA/NHL 2x) --
    without this dedup, "daily picks" would spam the channel once per
    firing instead of once per day. Returns a status dict rather than
    raising on a non-2xx Slack response -- this runs as the tail end of an
    already-completed daily pipeline, so a Slack hiccup should be logged,
    not treated as the whole run failing (same "external network hiccup
    must not break the run" convention every sport's own weather-forecast/
    RotoWire fallback already follows)."""
    resolved_slate = slate_key or db.latest_slate_key(sport)
    if resolved_slate is None:
        return {"status": "skipped", "reason": "no slate available yet"}

    if not db.mark_notified(sport, resolved_slate):
        return {"status": "skipped", "reason": "already notified for this slate", "slate_key": resolved_slate}

    games = db.games_for_slate(sport, resolved_slate)
    payload = build_slack_message(sport, resolved_slate, games, site_url)

    resp = requests.post(webhook_url, json=payload, timeout=10)
    return {
        "status": "ok" if resp.ok else "error",
        "slate_key": resolved_slate,
        "games_included": min(len(games), MAX_GAMES_SHOWN),
        "http_status": resp.status_code,
        "slack_response": resp.text[:200],
    }
