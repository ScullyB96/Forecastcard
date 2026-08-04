"""Slack Incoming Webhook notifications: a compact daily-picks digest,
posted once per sport as the tail end of that sport's own daily pipeline,
with a link back to the full slate/props breakdown on the site.

Deliberately NOT a full Slack bot (no OAuth install, no bot token, no
event subscriptions) -- a one-way "post today's picks" digest is exactly
what an Incoming Webhook is for: one secret URL, no app-install flow,
posts as a normal message into one fixed channel. Upgrade to a real Slack
App with a bot token only if two-way interactivity (buttons, slash
commands) is ever wanted -- not needed for this.

Fully self-contained: builds its own message shape and posts it, with no
knowledge of Discord (see discord_notify.py's own, independent sibling)
or of slate resolution/dedup (see main.py's notify route, which owns
that ONCE for whichever platforms are configured)."""

import requests

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


def build_slack_hr_message(sport: str, slate_key: str, hr_props: list[dict], site_url: str) -> dict:
    """A Slack Block Kit payload for the top-N home-run picks (see
    db.top_hr_props) -- a separate digest from build_slack_message's
    per-game picks, sent alongside it. `hr_props` rows are already
    ranked by projected mean HR count (a real projected number, not a
    probability -- see db.top_hr_props's own docstring for why) and
    joined to their game's matchup string for context. Always returns a
    real payload (never None) for the same reason build_slack_message
    does -- the caller (main.py) decides whether hr_props is non-empty
    enough to be worth sending at all."""
    label = SPORT_LABELS.get(sport, sport.upper())
    if not hr_props:
        text = f"{label} -- no home run picks available yet ({slate_key})"
        return {"text": text, "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]}
    lines = [f"*{p['player_name']}* -- {p['proj_mean']:.2f} projected HR ({p['matchup']})" for p in hr_props]
    header = f"{label} top {len(hr_props)} home run picks -- {slate_key}"
    link = f"{site_url.rstrip('/')}/{sport}/{slate_key}"
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": header}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"<{link}|Full player props>"}},
    ]
    return {"text": header, "blocks": blocks}


def post_to_slack(webhook_url: str, payload: dict) -> dict:
    """POSTs `payload` to a Slack Incoming Webhook. Returns a status dict
    rather than raising on a non-2xx response -- this runs as the tail
    end of an already-completed daily pipeline, so a Slack hiccup should
    be logged, not treated as the whole run failing (same "external
    network hiccup must not break the run" convention every sport's own
    weather-forecast/RotoWire fallback already follows)."""
    resp = requests.post(webhook_url, json=payload, timeout=10)
    return {"status": "ok" if resp.ok else "error", "http_status": resp.status_code, "response": resp.text[:200]}
