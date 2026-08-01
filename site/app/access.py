"""Password gate, togglable purely by the SITE_PASSWORD env var -- unset
means fully public (no gate at all, not even a login route wired up);
set means every route requires a signed session cookie obtained by
POSTing the password to /login. No user accounts, no database-backed
sessions -- a single shared password is the actual requirement (this is
a personal/small-team dashboard, not a multi-tenant product).
"""

import os

from itsdangerous import BadSignature, URLSafeTimedSerializer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse

COOKIE_NAME = "site_session"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
PUBLIC_PATHS = {"/login", "/static"}


def site_password() -> str | None:
    return os.environ.get("SITE_PASSWORD") or None


def _serializer() -> URLSafeTimedSerializer:
    # signing key can just be the password itself -- there's no separate
    # secret to manage, and rotating SITE_PASSWORD naturally invalidates
    # every existing session cookie too (the intended behavior).
    return URLSafeTimedSerializer(site_password(), salt="site-access")


def make_session_cookie_value() -> str:
    return _serializer().dumps({"ok": True})


def _is_valid_session(cookie_value: str | None) -> bool:
    if not cookie_value:
        return False
    try:
        _serializer().loads(cookie_value, max_age=COOKIE_MAX_AGE)
        return True
    except BadSignature:
        return False


class AccessGateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not site_password():
            return await call_next(request)
        if any(request.url.path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)
        if _is_valid_session(request.cookies.get(COOKIE_NAME)):
            return await call_next(request)
        return RedirectResponse(url=f"/login?next={request.url.path}")
