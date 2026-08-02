"""Combined multi-sport prediction site. Reads the shared Postgres tables
(db/schema.sql) that each sport's own export_to_site_db.py populates --
never imports model code from mlb/nfl/nba/nhl directly (see site/README.md
for the full contract)."""

import hmac
from datetime import datetime, timezone

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.access import AccessGateMiddleware, make_session_cookie_value, site_password
from app.season_openers import OPENERS
from app.team_meta import team_info

app = FastAPI()
app.add_middleware(AccessGateMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["team_info"] = team_info


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


def _props_by_game_id(sport: str, slate_key: str, games: list[dict]) -> dict[str, list[dict]]:
    return {g["game_id"]: db.props_for_game(sport, slate_key, g["game_id"]) for g in games}


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


@app.get("/")
def home(request: Request):
    slate_keys = {s: db.latest_slate_key(s) for s in db.SPORTS}
    games_by_sport = {}
    props_by_sport = {}
    for sport, slate_key in slate_keys.items():
        if not slate_key:
            continue
        games = db.games_for_slate(sport, slate_key)
        games_by_sport[sport] = games
        props_by_sport[sport] = _props_by_game_id(sport, slate_key, games)
    runs_by_sport = {r["sport"]: r for r in db.latest_runs()}
    return templates.TemplateResponse(request, "index.html", {
        "sports": db.SPORTS, "slate_keys": slate_keys,
        "games_by_sport": games_by_sport, "props_by_sport": props_by_sport,
        "openers": OPENERS, "runs_by_sport": runs_by_sport,
    })


def _render_sport(request: Request, sport: str, slate_key: str | None):
    if sport not in db.SPORTS:
        return RedirectResponse(url="/")
    games = db.games_for_slate(sport, slate_key) if slate_key else []
    props_by_game_id = _props_by_game_id(sport, slate_key, games) if slate_key else {}
    run = db.run_for_slate(sport, slate_key) if slate_key else None
    latest = db.latest_slate_key(sport)
    return templates.TemplateResponse(request, "sport.html", {
        "sports": db.SPORTS, "active_sport": sport, "sport": sport,
        "slate_key": slate_key, "games": games, "props_by_game_id": props_by_game_id,
        "opener": OPENERS.get(sport), "run": run,
        "available_slates": db.slate_keys_for_sport(sport),
        "is_latest": slate_key == latest,
    })


@app.get("/{sport}")
def sport_page(request: Request, sport: str):
    return _render_sport(request, sport, db.latest_slate_key(sport) if sport in db.SPORTS else None)


@app.get("/{sport}/{slate_key}")
def sport_slate_page(request: Request, sport: str, slate_key: str):
    return _render_sport(request, sport, slate_key)
