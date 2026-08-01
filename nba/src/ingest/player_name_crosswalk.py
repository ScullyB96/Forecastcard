"""Maps RotoWire's plain-text player names to the PERSON_ID convention used
everywhere else in this project (stints, box scores, rotation all key by
`nba_api`'s numeric player ID) -- the crosswalk `generate_predictions.py`
needs to actually exclude a RotoWire-flagged Out/Doubtful player from
tonight's active roster by ID, not just by name string.

`nba_api.stats.static.players` ships the full historical player list
(name -> id) with no HTTP call required -- confirmed live (2026-07-24):
5,103 players, `{"id": 2544, "full_name": "LeBron James", ...}` matches the
same PERSON_ID seen in real GameRotation/PlayByPlay data for LeBron. Exact
`full_name` match is tried first; a normalized fallback (lowercased, accents
stripped, punctuation removed) handles the common mismatch cases (RotoWire's
"Jokic" vs the static list's "Jokić", suffix formatting differences).

REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the static list has
38 groups of players sharing an exact `full_name` (confirmed live), and the
original dict-comprehension approach silently kept whichever entry happened
to appear LAST in the list's arbitrary (not id-sorted) order -- no
`is_active` preference, no collision detection at all, unlike this
project's own `build_stints.py._roster_lookup`, which already demonstrates
the right pattern (an explicit `dupes` set, refusing to resolve an
ambiguous name). One real collision is live-relevant today: "Brandon
Williams" resolved correctly to the active player (id 1630314, not the
retired id 1585) purely by accident of list ordering. Fixed: prefer the
`is_active=True` entry on a name collision (a live injury report is
overwhelmingly about a currently-active player, so this is a real
tiebreak, not a guess); if BOTH colliding entries share the same
`is_active` status, the name is genuinely ambiguous and is tracked in a
`dupes` set -- `player_id_for_name` returns `None` for it (same as a
no-match) rather than silently picking one, so the caller's existing
"could not be excluded" warning path covers this case too instead of a
silent wrong exclusion."""

import unicodedata

from nba_api.stats.static import players as _static_players

_ALL_PLAYERS = _static_players.get_players()


def _build_name_index(player_list: list[dict], key_fn) -> tuple[dict, set]:
    """name -> id index preferring `is_active=True` on a collision;
    genuinely ambiguous collisions (both entries share the same active
    status) are tracked in the returned `dupes` set rather than silently
    resolved to whichever was inserted first."""
    by_key: dict[str, dict] = {}
    dupes: set = set()
    for p in player_list:
        key = key_fn(p)
        if key not in by_key:
            by_key[key] = p
            continue
        existing = by_key[key]
        if existing["is_active"] == p["is_active"]:
            dupes.add(key)  # can't disambiguate -- same active-status, keep first seen but flag it
            continue
        if p["is_active"] and not existing["is_active"]:
            by_key[key] = p  # prefer the active player
    return {k: v["id"] for k, v in by_key.items()}, dupes


_EXACT_NAME_TO_ID, _EXACT_DUPES = _build_name_index(_ALL_PLAYERS, lambda p: p["full_name"])


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return "".join(c.lower() for c in stripped if c.isalnum())


_NORMALIZED_NAME_TO_ID, _NORMALIZED_DUPES = _build_name_index(_ALL_PLAYERS, lambda p: _normalize(p["full_name"]))

_SUFFIXES = (" Jr.", " Sr.", " II", " III", " IV")
# RotoWire's report omits generational suffixes (confirmed live 2026-07-24: "Jimmy Butler",
# "Trey Murphy", "Dereck Lively", "Scotty Pippen", "Michael Porter" all appear WITHOUT the
# suffix the static player list carries -- "Jimmy Butler III", "Trey Murphy III", "Dereck
# Lively II", "Scotty Pippen Jr.", "Michael Porter Jr." respectively) -- a real, common
# mismatch, not a hypothetical edge case, so it's handled as a real fallback tier rather than
# left to the normalized-name pass (which doesn't strip suffixes at all).
_SUFFIXED_PLAYERS = []
for _p in _ALL_PLAYERS:
    for _suffix in _SUFFIXES:
        if _p["full_name"].endswith(_suffix):
            _SUFFIXED_PLAYERS.append({**_p, "full_name": _p["full_name"][: -len(_suffix)]})
            break

_NORMALIZED_NO_SUFFIX_TO_ID, _NO_SUFFIX_DUPES = _build_name_index(_SUFFIXED_PLAYERS, lambda p: _normalize(p["full_name"]))


def player_id_for_name(name: str) -> int | None:
    """None both for "no match anywhere" AND "an ambiguous match this
    crosswalk can't confidently resolve" -- deliberately the same signal,
    since a caller (`active_roster.resolve_active_lineup`) can't safely
    act on either differently: both mean "don't trust an ID for this
    name.\""""
    if name in _EXACT_NAME_TO_ID and name not in _EXACT_DUPES:
        return _EXACT_NAME_TO_ID[name]
    normalized = _normalize(name)
    if normalized in _NORMALIZED_NAME_TO_ID and normalized not in _NORMALIZED_DUPES:
        return _NORMALIZED_NAME_TO_ID[normalized]
    if normalized in _NORMALIZED_NO_SUFFIX_TO_ID and normalized not in _NO_SUFFIX_DUPES:
        return _NORMALIZED_NO_SUFFIX_TO_ID[normalized]
    return None


if __name__ == "__main__":
    from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report

    report = fetch_current_injury_report()
    matched = sum(1 for r in report if player_id_for_name(r["player"]) is not None)
    print(f"matched {matched}/{len(report)} RotoWire names to a player ID", flush=True)
    unmatched = [r["player"] for r in report if player_id_for_name(r["player"]) is None]
    if unmatched:
        print(f"unmatched: {unmatched}", flush=True)
    print(f"\n{len(_EXACT_DUPES)} exact-name ambiguous collisions, "
          f"{len(_NORMALIZED_DUPES)} normalized, {len(_NO_SUFFIX_DUPES)} suffix-stripped", flush=True)
