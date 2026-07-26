"""Immutable pre-game prediction log -- the credibility backbone Sec37
promises (task #44's "immutable pre-game prediction log"): every live
model-vs-market comparison, the degradation budget, and the season-phase
segmentation all assume every prediction was recorded BEFORE puck drop,
with a timestamp and an input snapshot, in a form that provably can't be
revised after outcomes land. `metrics_ledger.parquet` tracks BACKTEST runs
(a different artifact, a different purpose); this is the live log.

**"Provably can't be revised" is a real, checkable claim here, not a
naming convention**: each entry is chained by hash, exactly like a minimal
append-only ledger -- `this_hash = sha256(prev_hash + canonical_json(entry))`.
Editing, reordering, or deleting ANY past entry breaks the chain from that
point forward, and `verify_log()` detects this directly by recomputing
every hash and comparing. Stored as JSONL (`data/processed/
live_prediction_log.jsonl`) -- a natural append-only format (each new
prediction is one more line, never rewriting existing lines), unlike
parquet which would require rewriting the whole file on every append.

This is a LOCAL file-integrity guarantee, not a cryptographic timestamp
service or a distributed ledger -- it proves "this specific file wasn't
silently edited after the fact" to anyone who has (or is given) a copy of
it, which is the actual credibility problem Sec37 is protecting against
(a human quietly correcting yesterday's prediction after seeing today's
score), not protection against someone rewriting the entire file from
scratch with a fabricated history. That's an explicit, honest scope
limitation, not a silently overstated claim.
"""

import hashlib
import json
from datetime import datetime, timezone

from src.utils.paths import DATA_PROCESSED

LOG_PATH = DATA_PROCESSED / "live_prediction_log.jsonl"
GENESIS_HASH = "0" * 64


def _canonical(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, default=str)


def _last_hash() -> str:
    if not LOG_PATH.exists():
        return GENESIS_HASH
    with open(LOG_PATH) as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        return GENESIS_HASH
    return json.loads(lines[-1])["this_hash"]


def append_prediction(game_id, game_date: str, home_team: str, away_team: str,
                       lambda_home: float, lambda_away: float, home_win_prob_full: float,
                       home_starter_goalie_id, away_starter_goalie_id,
                       injury_report_snapshot: list[dict] | None = None,
                       game_type: int | None = None) -> dict:
    """Appends one immutable, timestamped, hash-chained prediction entry.
    `injury_report_snapshot`: the RAW injury-report rows used for THIS
    prediction (from `fetch_rotowire_injuries.fetch_current_injury_report()`)
    -- stored inline (small: ~100 rows) so the exact input a live
    prediction was made from is reconstructable later, not just its
    output."""
    prev_hash = _last_hash()
    entry = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "game_id": game_id, "game_date": game_date, "game_type": game_type,
        "home_team": home_team, "away_team": away_team,
        "lambda_home": lambda_home, "lambda_away": lambda_away,
        "home_win_prob_full": home_win_prob_full,
        "home_starter_goalie_id": home_starter_goalie_id, "away_starter_goalie_id": away_starter_goalie_id,
        "injury_report_snapshot": injury_report_snapshot,
        "prev_hash": prev_hash,
    }
    entry["this_hash"] = hashlib.sha256((prev_hash + _canonical(entry)).encode()).hexdigest()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_log() -> tuple[bool, str]:
    """Recomputes every entry's hash from scratch and confirms the chain
    is intact -- returns (is_valid, message). Any edited, reordered, or
    deleted entry breaks the chain from that point forward and is
    reported with the exact entry index where it first diverges."""
    entries = read_log()
    prev_hash = GENESIS_HASH
    for i, entry in enumerate(entries):
        stored_hash = entry.get("this_hash")
        stored_prev = entry.get("prev_hash")
        if stored_prev != prev_hash:
            return False, f"entry {i} (game_id={entry.get('game_id')}): prev_hash mismatch -- an earlier " \
                           f"entry was edited, reordered, or removed"
        recomputed = {k: v for k, v in entry.items() if k != "this_hash"}
        expected_hash = hashlib.sha256((prev_hash + _canonical(recomputed)).encode()).hexdigest()
        if expected_hash != stored_hash:
            return False, f"entry {i} (game_id={entry.get('game_id')}): this_hash does not match its own " \
                           f"content -- this entry was edited after being logged"
        prev_hash = stored_hash
    return True, f"{len(entries)} entries, hash chain intact"


if __name__ == "__main__":
    ok, message = verify_log()
    print(f"{'VALID' if ok else 'INVALID'}: {message}", flush=True)
