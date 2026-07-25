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
"Jokic" vs the static list's "Jokić", suffix formatting differences)."""

import unicodedata

from nba_api.stats.static import players as _static_players

_ALL_PLAYERS = _static_players.get_players()
_EXACT_NAME_TO_ID = {p["full_name"]: p["id"] for p in _ALL_PLAYERS}


def _normalize(name: str) -> str:
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return "".join(c.lower() for c in stripped if c.isalnum())


_NORMALIZED_NAME_TO_ID = {_normalize(p["full_name"]): p["id"] for p in _ALL_PLAYERS}

_SUFFIXES = (" Jr.", " Sr.", " II", " III", " IV")
# RotoWire's report omits generational suffixes (confirmed live 2026-07-24: "Jimmy Butler",
# "Trey Murphy", "Dereck Lively", "Scotty Pippen", "Michael Porter" all appear WITHOUT the
# suffix the static player list carries -- "Jimmy Butler III", "Trey Murphy III", "Dereck
# Lively II", "Scotty Pippen Jr.", "Michael Porter Jr." respectively) -- a real, common
# mismatch, not a hypothetical edge case, so it's handled as a real fallback tier rather than
# left to the normalized-name pass (which doesn't strip suffixes at all).
_NORMALIZED_NO_SUFFIX_TO_ID: dict[str, int] = {}
for _p in _ALL_PLAYERS:
    _name = _p["full_name"]
    for _suffix in _SUFFIXES:
        if _name.endswith(_suffix):
            _NORMALIZED_NO_SUFFIX_TO_ID.setdefault(_normalize(_name[: -len(_suffix)]), _p["id"])
            break


def player_id_for_name(name: str) -> int | None:
    if name in _EXACT_NAME_TO_ID:
        return _EXACT_NAME_TO_ID[name]
    normalized = _normalize(name)
    if normalized in _NORMALIZED_NAME_TO_ID:
        return _NORMALIZED_NAME_TO_ID[normalized]
    return _NORMALIZED_NO_SUFFIX_TO_ID.get(normalized)


if __name__ == "__main__":
    from src.ingest.fetch_rotowire_lineups import fetch_current_injury_report

    report = fetch_current_injury_report()
    matched = sum(1 for r in report if player_id_for_name(r["player"]) is not None)
    print(f"matched {matched}/{len(report)} RotoWire names to a player ID", flush=True)
    unmatched = [r["player"] for r in report if player_id_for_name(r["player"]) is None]
    if unmatched:
        print(f"unmatched: {unmatched}", flush=True)
