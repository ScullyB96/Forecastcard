"""Scrape RotoWire's NBA injury report for tonight's active-lineup
resolution (`src/pipeline/generate_predictions.py`'s primary/fallback
source, alongside the NBA official injury report).

Confirmed live (2026-07-24) via the rendered page's own network traffic
(the HTML itself contains no server-rendered table -- it's a client-side
fetch): `GET https://www.rotowire.com/basketball/tables/injury-report.php
?team=ALL&pos=ALL` returns a clean, public JSON array (no auth, no
X-Requested-With/Referer actually required in testing, though sent anyway
to look like the real page's own request) with one row per flagged player:
`player`, `team` (abbreviation), `position`, `injury`, `status` (Out /
Doubtful / Questionable / Probable / Day-To-Day), and a subscriber-gated
`rDate` (expected-return date, not needed here -- only `status` matters for
excluding a player from tonight's active lineup). This is a materially
easier source than the MLB sibling's RotoWire scrape (that one has to parse
real HTML tables); no PDF parsing needed either, unlike the NBA official
report (`ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<time>
<AM|PM>.pdf`, confirmed to exist via web search but not yet wired up here --
see MODEL_DOCUMENTATION.md for why RotoWire alone is sufficient for v1).

This endpoint only ever reflects the CURRENT day's report -- there is no
historical archive, so this cannot backfill past seasons' lineups. It is a
live-pipeline-only source (Phase 5), not part of the dev/holdout training
data.
"""

from datetime import date

import requests

ROTOWIRE_URL = "https://www.rotowire.com/basketball/tables/injury-report.php"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
STATUS_MEANS_OUT = {"Out", "Doubtful"}


def fetch_current_injury_report() -> list[dict]:
    resp = requests.get(
        ROTOWIRE_URL,
        params={"team": "ALL", "pos": "ALL"},
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://www.rotowire.com/basketball/injury-report.php",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=15,
    )
    resp.raise_for_status()
    rows = resp.json()
    return [
        {
            "player": r["player"],
            "team": r["team"],
            "position": r["position"],
            "injury": r["injury"],
            "status": r["status"],
            "likely_out": r["status"] in STATUS_MEANS_OUT,
        }
        for r in rows
    ]


def warn_if_stale_for_backtest(game_date: str) -> None:
    """Prints an explicit warning when `game_date` isn't real wall-clock
    "today" -- `fetch_current_injury_report` has NO `game_date` parameter
    and can't have one (RotoWire has no historical archive at all, see
    this module's own docstring), so both live pipelines silently apply
    TODAY's Out/Doubtful list to whatever historical `game_date` they were
    asked to run for.

    REAL GAP FOUND (2026-08-01, full-model audit): every one of this
    project's own historical spot-checks (2018-01-15, 2023-11-08,
    2024-03-05, 2025-01-15) was silently exposed to this exact
    contamination -- a player genuinely out on the target historical date
    but healthy today wouldn't be excluded, and a player flagged Out today
    (for an unrelated, much later injury) would be wrongly excluded from a
    past game he actually played real minutes in. Not fixable at the data
    layer (there is nothing to backfill from), so the honest fix is an
    explicit, loud warning rather than a silent, invisible distortion --
    exactly this project's standing discipline for every other
    unresolvable caveat (e.g. the COVID-season edge case documented in
    `season_for_date`)."""
    if game_date != date.today().isoformat():
        print(f"  WARNING: RotoWire's injury report has NO historical archive -- it always reflects "
              f"REAL WALL-CLOCK TODAY ({date.today().isoformat()}), not {game_date}. Active-lineup "
              f"resolution for this historical/backtest call is using TODAY's Out/Doubtful list, "
              f"which may incorrectly include or exclude players relative to the real situation on "
              f"{game_date}.", flush=True)


if __name__ == "__main__":
    report = fetch_current_injury_report()
    print(f"fetched {len(report)} flagged players", flush=True)
    out_count = sum(r["likely_out"] for r in report)
    print(f"{out_count} listed Out/Doubtful", flush=True)
    for r in report[:10]:
        print(f"  {r['player']:25s} {r['team']:4s} {r['status']:12s} {r['injury']}", flush=True)
