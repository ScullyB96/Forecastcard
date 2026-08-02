"""Routes nba_api's live stats.nba.com calls through a proxy, with a
browser-realistic TLS fingerprint, if configured.

Confirmed live (2026-08-02, from inside the Railway nba-worker container),
via a systematic test sweep, not a guess:
1. Direct from Railway's IP: stats.nba.com silently black-holes the request
   (TCP/TLS connects fine, HTTP response never arrives) regardless of
   headers -- a datacenter-IP reputation block.
2. Through a Webshare residential proxy (20 different backend IPs tested),
   plain `requests`/curl still gets the identical silent black-hole --
   ruling out "just a bad IP" and pointing at a second, independent block:
   stats.nba.com (or its WAF/bot-manager) also fingerprints the TLS/HTTP
   handshake itself, and plain `requests`'s handshake doesn't look enough
   like a real browser's.
3. The SAME residential proxy IP, using `curl_cffi` (impersonates a real
   Chrome TLS/HTTP fingerprint) instead of plain `requests`: succeeds, full
   real response. Confirmed BOTH a non-blocked egress IP AND a
   browser-realistic fingerprint are independently required -- neither
   alone was sufficient.

`nba_api`'s HTTP layer already supports both pieces without touching any of
the 13 stats.nba.com-dependent files:
- `NBAStatsHTTP.set_session(session)` swaps the shared `requests.Session`-
  compatible client used by every endpoint call (`curl_cffi.requests.Session`
  is a drop-in that accepts the exact same call shape nba_api already uses:
  `.get(url=, params=, headers=, proxies=, timeout=)`).
- `nba_api.library.http.send_api_request` reads the module-level `PROXY`
  global as its default proxy on every call when no per-call `proxy=` is
  passed, so setting that once per process covers every existing
  fetch_*.py call site.

Safe to call unconditionally (including with NBA_STATS_PROXY_URL unset,
e.g. local dev): curl_cffi's session behaves like a normal direct client
when proxies=None, and Chrome-impersonation is harmless without a proxy --
it's only the COMBINATION that was ever required.
"""

import os

from curl_cffi import requests as _curl_cffi_requests
from nba_api.stats.library.http import NBAStatsHTTP


def configure_proxy() -> None:
    NBAStatsHTTP.set_session(_curl_cffi_requests.Session(impersonate="chrome124"))

    proxy_url = os.environ.get("NBA_STATS_PROXY_URL")
    if proxy_url:
        import nba_api.library.http as _nba_http
        _nba_http.PROXY = proxy_url
