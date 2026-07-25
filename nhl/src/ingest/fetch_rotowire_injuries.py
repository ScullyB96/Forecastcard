"""Live NHL injury/lineup-availability report from RotoWire -- the input
task #43/#44's recalibration rule and degradation budget assume exists
(MODEL_DOCUMENTATION.md Sec37): a goalie flagged Out/IR here should
override the recent-workhorse heuristic (Sec37.2), not just be silently
missed.

CONFIRMED LIVE 2026-07-25 (via browser network inspection, mirroring the
MLB sibling's own RotoWire investigation): the injury-report page
(`hockey/injury-report.php`) makes a client-side fetch to a clean, public
JSON endpoint --

    GET https://www.rotowire.com/hockey/tables/injury-report.php?team=ALL&pos=ALL

-- no auth, no API key. Confirmed real fields from a live response (2026-07-25,
preseason injury report already populated for Sep 19-20 2026 games): `ID`
(RotoWire's own numeric player ID), `URL` (player page slug), `firstname`,
`lastname`, `player` (full name), `team` (3-letter abbrev), `position`,
`injury` (free-text description, e.g. "Lower Body", "Knee"), `status` (seen:
"Out", "IR", "IR-LT" -- NHL's IR/IR-LT distinction, unlike the MLB/NBA
siblings' Out/Doubtful/Questionable/Probable scale; no "Questionable"-type
gradient was observed in this sample, so this project's downstream
consumer should NOT assume that gradient exists here), `rDate` (subscriber-
gated, literal HTML "<i>Subscribers Only</i>" string -- not usable), `date`
(the affected player's next scheduled game, e.g. "Sep 19 7:00 PM").

Team codes are RotoWire's own 3-letter abbreviations -- confirmed matching
this project's existing NHL API abbreviations directly in the sample
checked (DAL, STL, TOR, CHI, EDM, MIN, WPG, VGK, LAK, SEA, VAN, NJD, NYI --
all standard). No crosswalk needed unless a future mismatch is found; if
one is, add it to `team_codes.py` rather than patching it here.

NOT YET VERIFIED against real, in-season data: whether `status` ever
carries a finer-grained value (e.g. "Day-To-Day", "Questionable") once the
regular season is underway -- this sample is preseason-only (the site's
own real "today" was 2026-07-25, off-season; the only populated rows were
Sep 19-20 preseason games). Re-confirm the full `status` value space once
real regular-season data exists, per this project's own "confirmed real,
not assumed" discipline -- do not silently assume this list is exhaustive.
"""

import json
from datetime import datetime, timezone

import requests

from src.utils.paths import DATA_RAW

INJURY_REPORT_URL = "https://www.rotowire.com/hockey/tables/injury-report.php"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Confirmed present in a real live response (2026-07-25); "Questionable"/"Day-To-Day" are
# RotoWire's OWN documented status vocabulary elsewhere on the site but were NOT observed in
# this specific sample -- kept here as the full known-possible set, not a validated-exhaustive one.
KNOWN_STATUS_VALUES = {"Out", "IR", "IR-LT", "Questionable", "Day-To-Day", "Doubtful", "Probable"}


def fetch_current_injury_report() -> list[dict]:
    """Returns the raw list of injury-report rows (see module docstring for
    confirmed real fields). Also archives a timestamped snapshot to
    DATA_RAW -- RotoWire itself keeps no historical injury archive, so this
    project's own accumulated snapshots ARE the only historical record
    that will exist for later analysis (e.g. task #45's live goalie-gap
    re-measurement)."""
    resp = requests.get(INJURY_REPORT_URL, params={"team": "ALL", "pos": "ALL"},
                        headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    rows = resp.json()

    unknown_statuses = {r["status"] for r in rows if r.get("status") not in KNOWN_STATUS_VALUES}
    if unknown_statuses:
        print(f"fetch_rotowire_injuries: new, previously-unseen status value(s) {unknown_statuses} -- "
              f"update KNOWN_STATUS_VALUES and re-check any downstream logic that branches on status", flush=True)

    snapshot_dir = DATA_RAW / "rotowire_injury_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(snapshot_dir / f"injury_report_{ts}.json", "w") as f:
        json.dump(rows, f)

    return rows


def likely_out_player_names(injury_report: list[dict], team_abbrev: str) -> list[str]:
    """Player full names for `team_abbrev` whose status implies they will
    NOT play tonight -- "Out"/"IR"/"IR-LT" only. "Questionable"/"Day-To-Day"/
    "Doubtful" are deliberately NOT included here (these players often DO
    play) -- callers wanting a softer, discount-not-exclude treatment for
    those statuses should filter `injury_report` directly rather than
    relying on this helper's binary in/out cut."""
    definite_out = {"Out", "IR", "IR-LT"}
    return [r["player"] for r in injury_report if r["team"] == team_abbrev and r.get("status") in definite_out]


if __name__ == "__main__":
    report = fetch_current_injury_report()
    print(f"fetched {len(report)} injury-report rows", flush=True)
    if report:
        print("sample row:", report[0], flush=True)
        by_status = {}
        for r in report:
            by_status.setdefault(r.get("status"), 0)
            by_status[r.get("status")] += 1
        print(f"status breakdown: {by_status}", flush=True)
