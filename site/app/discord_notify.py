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
# No cap here (unlike slack_notify.MAX_GAMES_SHOWN): 2026-08-04 feedback --
# show every game, not just the top-N most lopsided. A full scoreline per
# game is still well inside Discord's 4096-char embed description limit
# even for a 15-game MLB slate.


def _confidence_rank(game: dict) -> float:
    """Sort key: how far this game's win probability sits from a
    coinflip -- used only to ORDER the full list (most-confident picks
    first), not to select a subset (every game is shown)."""
    wp = game.get("home_win_prob")
    return abs((wp if wp is not None else 0.5) - 0.5)


def _format_game_line(game: dict) -> str:
    """2026-08-04: shows each team's own projected SCORE (not a win
    probability percentage) -- a real final-score prediction, e.g.
    "SF 4.1 @ TEX 5.4", reads the favorite off the higher number without
    a separate percentage."""
    home, away = game.get("home_team", "?"), game.get("away_team", "?")
    home_score, away_score = game.get("projected_home_score"), game.get("projected_away_score")
    if home_score is None or away_score is None:
        return f"**{away} @ {home}** -- score prediction not yet available"
    return f"**{away} {away_score:.1f} @ {home} {home_score:.1f}**"


def build_discord_payload(sport: str, slate_key: str, games: list[dict], site_url: str) -> dict:
    """A Discord webhook payload: one embed, every game's projected score
    (most-lopsided win probability first), no site link -- 2026-08-04:
    the site isn't ready to share yet, so the embed says "coming soon"
    instead of linking out (site_url is still accepted, unused for now,
    so re-enabling the real link later is a one-line change here, not a
    signature change)."""
    label = SPORT_LABELS.get(sport, sport.upper())
    title = f"{label} picks -- {slate_key} ({len(games)} game{'s' if len(games) != 1 else ''})"
    if not games:
        return {"embeds": [{"title": f"{label} -- no games in today's slate ({slate_key})"}]}

    ranked = sorted(games, key=_confidence_rank, reverse=True)
    lines = [_format_game_line(g) for g in ranked]
    lines.append("*Full breakdown on the website -- coming soon*")
    return {"embeds": [{"title": title, "description": "\n".join(lines)}]}


def post_to_discord(webhook_url: str, payload: dict) -> dict:
    """POSTs `payload` to a Discord Incoming Webhook. `wait=true` makes
    Discord return the created message's id (a plain POST returns an
    empty 204 with no way to reference the message again) -- captured
    here so a caller CAN delete-and-repost later if a digest needs
    correcting (see main.py's notify route) instead of asking a human to
    delete it by hand, which is otherwise the only option once a plain
    POST has already fired. Returns a status dict rather than raising on
    a non-2xx response -- same "external hiccup must not break the run"
    convention as slack_notify.py."""
    resp = requests.post(f"{webhook_url}?wait=true", json=payload, timeout=10)
    message_id = None
    if resp.ok:
        try:
            message_id = resp.json().get("id")
        except ValueError:
            pass
    return {
        "status": "ok" if resp.ok else "error",
        "http_status": resp.status_code,
        "response": resp.text[:200],
        "message_id": message_id,
    }


def delete_discord_message(webhook_url: str, message_id: str) -> dict:
    """Deletes a previously-posted webhook message by id (see
    post_to_discord's captured `message_id`) -- lets a caller correct a
    digest (delete the wrong one, post the right one) without needing a
    human to delete it manually in the Discord client."""
    resp = requests.delete(f"{webhook_url}/messages/{message_id}", timeout=10)
    return {"status": "ok" if resp.ok else "error", "http_status": resp.status_code}
