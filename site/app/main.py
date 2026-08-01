"""Combined multi-sport prediction site. Reads the shared Postgres tables
(db/schema.sql) that each sport's own export_to_site_db.py populates --
never imports model code from mlb/nfl/nba/nhl directly (see site/README.md
for the full contract)."""

import hmac

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.access import AccessGateMiddleware, make_session_cookie_value, site_password

app = FastAPI()
app.add_middleware(AccessGateMiddleware)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


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
    return templates.TemplateResponse(request, "index.html", {
        "sports": db.SPORTS, "slate_keys": slate_keys,
        "games_by_sport": games_by_sport, "props_by_sport": props_by_sport,
    })


@app.get("/{sport}")
def sport_page(request: Request, sport: str):
    if sport not in db.SPORTS:
        return RedirectResponse(url="/")
    slate_key = db.latest_slate_key(sport)
    games = db.games_for_slate(sport, slate_key) if slate_key else []
    props_by_game_id = _props_by_game_id(sport, slate_key, games) if slate_key else {}
    return templates.TemplateResponse(request, "sport.html", {
        "sports": db.SPORTS, "active_sport": sport, "sport": sport,
        "slate_key": slate_key, "games": games, "props_by_game_id": props_by_game_id,
    })
