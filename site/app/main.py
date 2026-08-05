"""Combined multi-sport prediction site. Reads the shared Postgres tables
(db/schema.sql) that each sport's own export_to_site_db.py populates --
never imports model code from mlb/nfl/nba/nhl directly (see site/README.md
for the full contract)."""

import hmac
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db, discord_notify, fantasy, slack_notify
from app.access import AccessGateMiddleware, COOKIE_NAME, make_session_cookie_value, site_password
from app.season_openers import OPENERS
from app.team_meta import team_info

app = FastAPI()
app.add_middleware(AccessGateMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["team_info"] = team_info
templates.env.globals["site_password"] = site_password


def _relative_time(dt: datetime) -> str:
    """'Updated 38m ago' style label for a run_at timestamp. Server-rendered
    once per request rather than a JS ticker -- run freshness only needs to
    be roughly right, not live-updating like the season-opener countdown."""
    delta = datetime.now(timezone.utc) - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


templates.env.filters["relative_time"] = _relative_time

STALE_HOURS = 12


def _is_stale(dt: datetime) -> bool:
    """True once a run's data is old enough to warrant a visible warning
    rather than a neutral timestamp. A single flat threshold, not tied to
    first-pitch/lock time (that would need a game_time column this schema
    doesn't have yet) -- reuses the run_at we already track."""
    return (datetime.now(timezone.utc) - dt).total_seconds() > STALE_HOURS * 3600


templates.env.filters["is_stale"] = _is_stale


# Preferred on-page order for known markets within one player's stat line
# (mirrors each sport's own BATTER_MARKETS/PITCHER_MARKETS-style constants)
# -- anything not listed falls back to alphabetical after these, so a
# market this list doesn't know about yet (a new sport, a new stat) still
# renders instead of erroring, just without a curated position.
MARKET_ORDER = [
    "HR", "Hits", "Total Bases", "RBI", "BB", "Batter Strikeouts",
    "Strikeouts", "Walks Allowed", "Hits Allowed", "Runs Allowed", "Batters Faced",
]
_MARKET_RANK = {m: i for i, m in enumerate(MARKET_ORDER)}

# Short label for a stat line (e.g. "K 7.2" instead of "Strikeouts 7.2") --
# falls back to the full market name for anything not listed (a new sport's
# market this project hasn't curated an abbreviation for yet), so nothing
# ever fails to render, just without a shortened label.
MARKET_ABBREV = {
    "HR": "HR", "Hits": "H", "Total Bases": "TB", "RBI": "RBI", "BB": "BB",
    "Batter Strikeouts": "K", "Strikeouts": "K", "Walks Allowed": "BB",
    "Hits Allowed": "H", "Runs Allowed": "R", "Batters Faced": "BF",
}

# Column order for the slate-wide leaderboard, keyed by abbreviation
# instead of full market name (several full names collapse to the same
# abbreviation, e.g. "Batter Strikeouts" and "Strikeouts" both -> "K" --
# that's fine, a batters table and a pitchers table never share one row,
# so there's no real collision, just a shared rank for whichever full
# name gets there first).
_ABBREV_RANK: dict[str, int] = {}
for _i, _m in enumerate(MARKET_ORDER):
    _ABBREV_RANK.setdefault(MARKET_ABBREV.get(_m, _m), _i)

# extra.role is only set by MLB's export today (NFL/NBA/NHL props have no
# role key) -- ROLE_LABELS.get(None) below falls back to a generic "Players"
# section so those sports still render sensibly, just without a batter/
# pitcher split they don't have data for.
ROLE_LABELS = {"batter": "Batters", "pitcher": "Pitchers"}


def _stat_value(p: dict) -> float:
    """A single comparable number for a stat, whichever kind it is --
    projected mean if this market has one, else the raw probability
    (0-1) for the handful of genuinely binary/rare markets (e.g. NFL's
    Anytime TD) that don't have a natural mean equivalent."""
    if p.get("proj_mean") is not None:
        return p["proj_mean"]
    return p.get("over_prob") or 0.0


def _group_props(rows: list[dict]) -> dict:
    """Turn one game's flat prop list into role -> player stat-line
    sections -- one row per player merging every market they have into a
    single line (e.g. "Zack Wheeler  K 7.2 · BB 2.1 · H 5.4 · R 2.8 · BF
    23.6"), sorted by total projected production within each role.

    Replaces the old one-pill-per-market-per-player layout: showing a
    fixed "6+ K"-style probability forced the same arbitrary threshold
    onto every pitcher regardless of their own skill level. A plain
    projected number lets the reader pick their own threshold instead."""
    by_player: dict[tuple, dict] = {}
    for p in rows:
        role = (p.get("extra") or {}).get("role")
        key = (p["player_id"], role)
        entry = by_player.setdefault(key, {
            "player_id": p["player_id"], "player_name": p["player_name"],
            "team": p.get("team"), "role": role, "stats": [],
        })
        entry["stats"].append({
            **p, "abbrev": MARKET_ABBREV.get(p["market"], p["market"]),
        })

    for entry in by_player.values():
        entry["stats"].sort(key=lambda s: (_MARKET_RANK.get(s["market"], len(MARKET_ORDER)), s["market"]))

    by_role: dict[str | None, list[dict]] = {}
    for entry in by_player.values():
        by_role.setdefault(entry["role"], []).append(entry)

    sections = []
    for role in sorted(by_role, key=lambda r: (r is None, r != "batter")):
        players = sorted(by_role[role], key=lambda e: sum(_stat_value(s) for s in e["stats"]), reverse=True)
        sections.append({"role": role, "label": ROLE_LABELS.get(role, "Players"), "players": players})

    return {"total": len(rows), "player_count": len(by_player), "sections": sections}


def _props_by_game_id(rows: list[dict]) -> dict[str, dict]:
    """Bucket one slate's flat prop rows by game_id, then group each
    game's rows into per-player stat lines. Takes the already-fetched
    slate-wide rows (one query, see props_for_slate) rather than querying
    per game_id -- this used to be N queries for an N-game slate."""
    by_game: dict[str, list[dict]] = {}
    for p in rows:
        by_game.setdefault(p["game_id"], []).append(p)
    return {game_id: _group_props(game_rows) for game_id, game_rows in by_game.items()}


def _leaderboard(rows: list[dict], game_lookup: dict[str, str]) -> dict:
    """Slate-wide view: every player's full stat line as one flat table
    row per role (not nested per-market like _group_props), for a
    sortable/filterable leaderboard across the whole day's games at once
    -- "who projects best at X, across every game" rather than "what does
    this one game's roster look like."

    Columns are whatever markets are actually present for that role (not
    hardcoded), so this works for any sport's data shape. is_prob is kept
    per-column (not just per-cell) since a whole column is either a
    probability or a mean -- never mixed within one column."""
    by_role: dict[str | None, dict[int, dict]] = {}
    prob_by_role: dict[str | None, dict[str, bool]] = {}
    for p in rows:
        role = (p.get("extra") or {}).get("role")
        abbrev = MARKET_ABBREV.get(p["market"], p["market"])
        is_prob = p.get("proj_mean") is None
        value = p.get("over_prob") if is_prob else p.get("proj_mean")
        prob_by_role.setdefault(role, {})[abbrev] = is_prob
        players = by_role.setdefault(role, {})
        entry = players.setdefault(p["player_id"], {
            "player_id": p["player_id"], "player_name": p["player_name"],
            "game_id": p["game_id"], "matchup": game_lookup.get(p["game_id"], ""),
            "stats": {},
        })
        entry["stats"][abbrev] = value

    sections = []
    for role in sorted(by_role, key=lambda r: (r is None, r != "batter")):
        prob_flags = prob_by_role[role]
        columns = sorted(prob_flags, key=lambda m: (_ABBREV_RANK.get(m, len(MARKET_ORDER)), m))
        players = sorted(by_role[role].values(), key=lambda e: sum(v or 0 for v in e["stats"].values()), reverse=True)
        sections.append({
            "role": role, "label": ROLE_LABELS.get(role, "Players"),
            "columns": [{"key": c, "is_prob": prob_flags[c]} for c in columns],
            "players": players,
        })
    return {"sections": sections}


def _fantasy_leaderboard(rows: list[dict], game_lookup: dict[str, str]) -> dict:
    """Same slate-wide grouping as _leaderboard (one row per player per
    role, merging every market into a stats dict keyed by MARKET_ABBREV),
    but ranked by real Yahoo-default fantasy points (see fantasy.py)
    instead of a naive unweighted sum. Roles fantasy.fantasy_points()
    doesn't have a scoring table for (any non-MLB role today) are skipped
    entirely -- this stays empty for NFL/NBA/NHL until each has its own
    scoring table, the same "no data for this shape yet" gating
    leaderboard.sections already uses to hide the Players tab there."""
    by_role: dict[str, dict[int, dict]] = {}
    for p in rows:
        role = (p.get("extra") or {}).get("role")
        if role not in fantasy.POINTS_BY_ROLE:
            continue
        abbrev = MARKET_ABBREV.get(p["market"], p["market"])
        players = by_role.setdefault(role, {})
        entry = players.setdefault(p["player_id"], {
            "player_id": p["player_id"], "player_name": p["player_name"],
            "game_id": p["game_id"], "matchup": game_lookup.get(p["game_id"], ""),
            "stats": {},
        })
        entry["stats"][abbrev] = p.get("proj_mean")

    sections = []
    for role in sorted(by_role, key=lambda r: r != "batter"):
        columns = list(fantasy.POINTS_BY_ROLE[role].keys())
        players = list(by_role[role].values())
        for e in players:
            e["points"] = fantasy.fantasy_points(role, e["stats"])
        players.sort(key=lambda e: e["points"] or 0.0, reverse=True)
        for i, e in enumerate(players, start=1):
            e["rank"] = i
        sections.append({
            "role": role, "label": ROLE_LABELS.get(role, "Players"),
            "columns": columns, "players": players,
            "omitted": fantasy.OMITTED_BY_ROLE[role],
        })
    return {"sections": sections}


@app.get("/healthz")
def healthz():
    # deliberately DB-free and always public (see access.py's PUBLIC_PATHS)
    # -- Railway's healthcheck must get a 200 even when SITE_PASSWORD is
    # set, and shouldn't fail the whole service just because Postgres is
    # briefly unreachable.
    return {"status": "ok"}


@app.get("/login")
def login_form(request: Request, next: str = "/"):
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": False})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...), next: str = Form("/")):
    expected = site_password()
    if expected and hmac.compare_digest(password, expected):
        response = RedirectResponse(url=next or "/", status_code=303)
        response.set_cookie("site_session", make_session_cookie_value(), httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": True}, status_code=401)


@app.post("/internal/notify/{sport}")
def notify(sport: str, request: Request, slate_key: str | None = None, force: bool = False):
    """Machine-to-machine trigger: a sport's own daily pipeline calls this
    as its last step (see each railway.toml's startCommand) to post that
    day's picks digest to whichever chat platform(s) are configured.
    Bypasses the user-facing password gate (see access.py's PUBLIC_PATHS)
    -- protected instead by a constant-time comparison against
    NOTIFY_TOKEN, sent as a header (never a query param, so it never
    lands in access logs).

    Slack and Discord are fully independent -- either, both, or neither
    can be configured (SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL), each
    reported separately in the response. The slate is resolved and
    dedup-checked (db.mark_notified) exactly ONCE here, up front, shared
    by both platforms -- a sport's own cron can call this several times a
    day as it refreshes the SAME slate intraday (MLB 4x, NBA/NHL 2x), and
    only the FIRST firing to touch a given (sport, slate_key) actually
    posts anywhere; every later same-day firing for that slate is a
    silent no-op on both platforms at once.

    force=true (2026-08-04, a real correction that came up in practice):
    bypasses the dedup skip and re-sends even for an already-notified
    slate -- for fixing a digest that was already posted (wrong format,
    typo, etc.). On Discord specifically, if a prior post's message id was
    captured (see discord_notify.post_to_discord's `wait=true`), the OLD
    message is deleted before the corrected one is sent, so a correction
    doesn't leave both the wrong and the right message sitting in the
    channel. Slack has no equivalent (a plain Incoming Webhook can't
    delete a past message) -- a force-repost there just adds a new one."""
    token = os.environ.get("NOTIFY_TOKEN")
    if not token or not hmac.compare_digest(request.headers.get("x-notify-token", ""), token):
        raise HTTPException(status_code=401, detail="invalid or missing notify token")
    if sport not in db.SPORTS:
        raise HTTPException(status_code=404, detail="unknown sport")

    slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not slack_webhook and not discord_webhook:
        return {"status": "skipped", "reason": "no webhook configured (SLACK_WEBHOOK_URL / DISCORD_WEBHOOK_URL)"}

    resolved_slate = slate_key or db.latest_slate_key(sport)
    if resolved_slate is None:
        return {"status": "skipped", "reason": "no slate available yet"}
    already_notified = not db.mark_notified(sport, resolved_slate)
    if already_notified and not force:
        return {"status": "skipped", "reason": "already notified for this slate", "slate_key": resolved_slate}

    games = db.games_for_slate(sport, resolved_slate)
    # RAILWAY_PUBLIC_DOMAIN is auto-injected by Railway for any service with
    # a public domain -- the link in the message must point at the PUBLIC
    # url regardless of which (private-network) host this request itself
    # arrived on.
    public_domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
    site_url = f"https://{public_domain}" if public_domain else str(request.base_url)

    result = {"slate_key": resolved_slate}
    if slack_webhook:
        payload = slack_notify.build_slack_message(sport, resolved_slate, games, site_url)
        result["slack"] = slack_notify.post_to_slack(slack_webhook, payload)
    if discord_webhook:
        if force:
            prior_id = db.get_discord_message_id(sport, resolved_slate)
            if prior_id:
                discord_notify.delete_discord_message(discord_webhook, prior_id)
        payload = discord_notify.build_discord_payload(sport, resolved_slate, games, site_url)
        post_result = discord_notify.post_to_discord(discord_webhook, payload)
        result["discord"] = post_result
        if post_result.get("message_id"):
            db.set_discord_message_id(sport, resolved_slate, post_result["message_id"])

    # Top-N home run picks (2026-08-04): a SEPARATE digest from the
    # per-game picks above, sent alongside it -- naturally empty (and so
    # silently skipped) for any sport/slate with no "HR" market rows
    # (every non-MLB sport; see db.top_hr_props's own docstring), no
    # sport-name check needed. Shares the SAME dedup gate as the game
    # digest above (both are part of one daily notification event) rather
    # than tracking its own -- if force=true bypassed the skip for the
    # game digest, it does the same here.
    hr_props = db.top_hr_props(sport, resolved_slate, limit=10)
    if hr_props:
        if slack_webhook:
            payload = slack_notify.build_slack_hr_message(sport, resolved_slate, hr_props, site_url)
            result["slack_hr"] = slack_notify.post_to_slack(slack_webhook, payload)
        if discord_webhook:
            payload = discord_notify.build_discord_hr_payload(sport, resolved_slate, hr_props)
            result["discord_hr"] = discord_notify.post_to_discord(discord_webhook, payload)
    return result


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response



# Labels for the lineup_source values MLB's export already tags per side
# (src/pipeline/generate_daily_props.py's lineup_for_game(): "real" > "rotowire"
# > "projected", in that priority order). Other sports don't set this key
# yet, so .get() below just skips the penalty rather than erroring.
LINEUP_SOURCE_LABEL = {
    "real": "confirmed lineup",
    "rotowire": "Rotowire-projected lineup",
    "projected": "model-projected lineup (no real lineup posted yet)",
}
LINEUP_SOURCE_PENALTY = {"real": 0, "rotowire": 1, "projected": 2}


def _confidence_for_game(game: dict) -> dict | None:
    """Data-quality confidence for one game, built entirely from flags the
    pipeline already exports in `extra` -- no new pipeline work, no schema
    change. This is deliberately NOT model conviction (the win probability
    already reflects that): it answers "how solid is the input data this
    prediction was built on," which today is implicit (you'd have to know
    to distrust a projected lineup) and this makes explicit. Returns None
    when a sport hasn't tagged any of these fields (nothing to score)."""
    extra = game.get("extra") or {}
    if not any(k in extra for k in ("home_lineup_source", "away_lineup_source", "real_weather", "used_debut_fallback")):
        return None

    score = 10
    reasons = []
    for side in ("home", "away"):
        source = extra.get(f"{side}_lineup_source")
        penalty = LINEUP_SOURCE_PENALTY.get(source, 0)
        if penalty:
            score -= penalty
            reasons.append(f"{side.capitalize()} lineup: {LINEUP_SOURCE_LABEL[source]}")
    if extra.get("real_weather") is False:
        score -= 1
        reasons.append("Weather is forecast/climatological, not a real observation yet")
    if extra.get("used_debut_fallback"):
        score -= 1
        reasons.append("A player in this game has no MLB history yet (debut fallback used)")
    score = max(score, 0)

    if score >= 10:
        tier = "confirmed"
    elif score >= 8:
        tier = "high"
    elif score >= 6:
        tier = "medium"
    else:
        tier = "low"
    return {"score": score, "tier": tier, "reasons": reasons}


def _with_confidence(games: list[dict]) -> list[dict]:
    for g in games:
        g["confidence"] = _confidence_for_game(g)
    return games


@app.get("/")
def home(request: Request):
    slate_keys = {s: db.latest_slate_key(s) for s in db.SPORTS}
    games_by_sport = {}
    props_by_sport = {}
    for sport, slate_key in slate_keys.items():
        if not slate_key:
            continue
        games = _with_confidence(db.games_for_slate(sport, slate_key))
        games_by_sport[sport] = games
        props_by_sport[sport] = _props_by_game_id(db.props_for_slate(sport, slate_key))
    runs_by_sport = {r["sport"]: r for r in db.latest_runs()}
    return templates.TemplateResponse(request, "index.html", {
        "sports": db.SPORTS, "slate_keys": slate_keys,
        "games_by_sport": games_by_sport, "props_by_sport": props_by_sport,
        "openers": OPENERS, "runs_by_sport": runs_by_sport,
    })


def _slate_label(slate_key: str) -> str:
    """Human-readable date-picker label ('Sun, Aug 2') for a date-shaped
    slate_key; falls back to the raw key unchanged for sports whose
    slate_key isn't a date (e.g. NFL's "{season}-wk{week}")."""
    try:
        return datetime.strptime(slate_key, "%Y-%m-%d").strftime("%a, %b %-d")
    except ValueError:
        return slate_key


templates.env.filters["slate_label"] = _slate_label


def _render_sport(request: Request, sport: str, slate_key: str | None):
    if sport not in db.SPORTS:
        return RedirectResponse(url="/")
    games = _with_confidence(db.games_for_slate(sport, slate_key)) if slate_key else []
    slate_props = db.props_for_slate(sport, slate_key) if slate_key else []
    props_by_game_id = _props_by_game_id(slate_props)
    game_lookup = {g["game_id"]: g["matchup"] for g in games}
    leaderboard = _leaderboard(slate_props, game_lookup)
    fantasy_leaderboard = _fantasy_leaderboard(slate_props, game_lookup)
    run = db.run_for_slate(sport, slate_key) if slate_key else None
    latest = db.latest_slate_key(sport)

    # available_slates is newest-first -- prev/next are neighbors in that
    # list (the nearest slate with real data), not calendar-adjacent
    # dates, since off days and not-yet-generated dates shouldn't produce
    # a dead link. Mirrors the "<< prev / next >>" pattern real slate-
    # based sites (e.g. Ballpark Pal) use for browsing by date.
    available_slates = db.slate_keys_for_sport(sport)
    prev_slate = next_slate = None
    if slate_key in available_slates:
        idx = available_slates.index(slate_key)
        if idx + 1 < len(available_slates):
            prev_slate = available_slates[idx + 1]
        if idx > 0:
            next_slate = available_slates[idx - 1]

    return templates.TemplateResponse(request, "sport.html", {
        "sports": db.SPORTS, "active_sport": sport, "sport": sport,
        "slate_key": slate_key, "games": games, "props_by_game_id": props_by_game_id,
        "leaderboard": leaderboard, "fantasy_leaderboard": fantasy_leaderboard,
        "opener": OPENERS.get(sport), "run": run,
        "available_slates": available_slates,
        "slate_options": [{"key": s, "label": _slate_label(s)} for s in available_slates],
        "prev_slate": prev_slate, "next_slate": next_slate,
        "is_latest": slate_key == latest,
    })


@app.get("/{sport}")
def sport_page(request: Request, sport: str):
    return _render_sport(request, sport, db.latest_slate_key(sport) if sport in db.SPORTS else None)


@app.get("/{sport}/{slate_key}")
def sport_slate_page(request: Request, sport: str, slate_key: str):
    return _render_sport(request, sport, slate_key)
