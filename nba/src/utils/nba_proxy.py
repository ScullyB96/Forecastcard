"""Routes nba_api's live stats.nba.com calls through a proxy, if configured.

Confirmed live (2026-08-02, from inside the Railway nba-worker container):
Railway's egress IP range is silently black-holed by stats.nba.com (TCP
connects fine, HTTP response never arrives, even with a full browser-like
header set) -- a datacenter-IP reputation block, not a header/UA issue, so
no amount of retrying fixes it. `nba_api`'s HTTP layer already supports
routing through a proxy (`nba_api.library.http.NBAHTTP.send_api_request`
reads the module-level `PROXY` global as its default when no per-call
`proxy=` is passed), so setting that global once per process is enough to
cover every existing fetch_*.py call site without touching any of them.

No-op wherever NBA_STATS_PROXY_URL is unset (e.g. local dev, where the
machine's real IP isn't blocked).
"""

import os

import nba_api.library.http as _nba_http


def configure_proxy() -> None:
    proxy_url = os.environ.get("NBA_STATS_PROXY_URL")
    if proxy_url:
        _nba_http.PROXY = proxy_url
