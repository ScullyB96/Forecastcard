"""Discord webhook notifications: the Discord-side counterpart to
slack_notify.py -- same daily-picks digest, same site link, different
payload shape. Discord's webhook API takes an `embeds` array (title,
url, description), not Slack's Block Kit `blocks`, and uses **bold**
markdown instead of Slack's *bold* mrkdwn -- otherwise identical in
spirit: one secret webhook URL, no bot install, no OAuth flow.

Posting to Discord and Slack are fully independent (see main.py's notify
route) -- either, both, or neither can be configured; the caller resolves
the slate/dedup ONCE and hands the same `games` list to whichever of
these two modules has a webhook URL configured.
"""

import requests

SPORT_LABELS = {"mlb": "MLB", "nfl": "NFL", "nba": "NBA", "nhl": "NHL"}
MAX_GAMES_SHOWN = 8  # keep the digest scannable -- the site link is where full depth lives


def _confidence_rank(game: dict) -> float:
    """Sort key: how far this game's win probability sits from a
    coinflip -- see slack_notify.py's identical helper for the full
    rationale (kept as its own small copy here rather than a shared
    import -- each notifier module is meant to be fully self-contained)."""
    wp = game.get("home_win_prob")
    return abs((wp if wp is not None else 0.5) - 0.5)


def _format_game_line(game: dict) -> str:
    home_wp, away_wp = game.get("home_win_prob"), game.get("away_win_prob")
    home, away = game.get("home_team", "?"), game.get("away_team", "?")
    if home_wp is None or away_wp is None:
        return f"**{away} @ {home}** -- win probability not yet available"
    favorite, prob = (home, home_wp) if home_wp >= away_wp else (away, away_wp)
    total = game.get("projected_total")
    total_txt = f" - Proj total {total:.1f}" if total is not None else ""
    return f"**{away} @ {home}** -- {favorite} {prob:.0%}{total_txt}"


def build_discord_payload(sport: str, slate_key: str, games: list[dict], site_url: str) -> dict:
    """A Discord webhook payload: one embed whose TITLE is itself the
    clickable link to the site (Discord embed titles support `url`
    directly, unlike Slack's blocks -- no separate "full breakdown" line
    needed), with the ranked game lines as the embed description
    (Discord embed descriptions support markdown links/bold, unlike
    plain message `content`)."""
    label = SPORT_LABELS.get(sport, sport.upper())
    link = f"{site_url.rstrip('/')}/{sport}/{slate_key}"
    if not games:
        title = f"{label} -- no games in today's slate ({slate_key})"
        return {"embeds": [{"title": title, "url": link}]}

    ranked = sorted(games, key=_confidence_rank, reverse=True)
    shown = ranked[:MAX_GAMES_SHOWN]
    lines = [_format_game_line(g) for g in shown]
    omitted = len(games) - len(shown)
    if omitted > 0:
        lines.append(f"*+{omitted} more game{'s' if omitted != 1 else ''} on the site*")

    title = f"{label} picks -- {slate_key} ({len(games)} game{'s' if len(games) != 1 else ''})"
    return {"embeds": [{"title": title, "url": link, "description": "\n".join(lines)}]}


def post_to_discord(webhook_url: str, payload: dict) -> dict:
    """POSTs `payload` to a Discord Incoming Webhook. Returns a status
    dict rather than raising on a non-2xx response -- same "external
    hiccup must not break the run" convention as slack_notify.py."""
    resp = requests.post(webhook_url, json=payload, timeout=10)
    return {"status": "ok" if resp.ok else "error", "http_status": resp.status_code, "response": resp.text[:200]}
