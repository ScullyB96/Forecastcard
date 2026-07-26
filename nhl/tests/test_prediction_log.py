"""Tests for the immutable pre-game prediction log (Sec42.2, task #44's
credibility backbone for Sec37): confirms the hash chain actually detects
tampering, not just that it appends cleanly.

Uses a temporary log path (monkey-patches `LOG_PATH`) so this never
touches the real live log.

Run: `.venv/bin/python -m tests.test_prediction_log`
"""

import json
import tempfile
from pathlib import Path

from src.models import prediction_log as pl

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_fresh_log_verifies_clean():
    with tempfile.TemporaryDirectory() as tmp:
        pl.LOG_PATH = Path(tmp) / "log.jsonl"
        pl.append_prediction(1, "2026-10-08", "BOS", "TOR", 3.1, 2.9, 0.55, 111, 222)
        pl.append_prediction(2, "2026-10-08", "NYR", "NJD", 2.8, 3.0, 0.48, 333, 444)
        ok, msg = pl.verify_log()
        check("a freshly-appended, untouched log verifies as valid", ok, msg)


def test_editing_a_past_entry_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        pl.LOG_PATH = Path(tmp) / "log.jsonl"
        pl.append_prediction(1, "2026-10-08", "BOS", "TOR", 3.1, 2.9, 0.55, 111, 222)
        pl.append_prediction(2, "2026-10-08", "NYR", "NJD", 2.8, 3.0, 0.48, 333, 444)

        # Tamper: rewrite the FIRST entry's own lambda_home after the fact (e.g. someone quietly
        # "correcting" a prediction after seeing the real score) without recomputing the chain.
        lines = pl.LOG_PATH.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["lambda_home"] = 99.9  # tampered value, this_hash left as-is (the whole point of the attack)
        lines[0] = json.dumps(entry)
        pl.LOG_PATH.write_text("\n".join(lines) + "\n")

        ok, msg = pl.verify_log()
        check("editing a past entry's own field (hash left stale) is detected", not ok, msg)


def test_deleting_a_past_entry_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        pl.LOG_PATH = Path(tmp) / "log.jsonl"
        pl.append_prediction(1, "2026-10-08", "BOS", "TOR", 3.1, 2.9, 0.55, 111, 222)
        pl.append_prediction(2, "2026-10-08", "NYR", "NJD", 2.8, 3.0, 0.48, 333, 444)
        pl.append_prediction(3, "2026-10-09", "SEA", "VAN", 3.3, 2.7, 0.60, 555, 666)

        lines = pl.LOG_PATH.read_text().splitlines()
        del lines[1]  # remove the middle entry entirely
        pl.LOG_PATH.write_text("\n".join(lines) + "\n")

        ok, msg = pl.verify_log()
        check("deleting a past entry breaks the chain for everything after it", not ok, msg)


def test_reordering_entries_is_detected():
    with tempfile.TemporaryDirectory() as tmp:
        pl.LOG_PATH = Path(tmp) / "log.jsonl"
        pl.append_prediction(1, "2026-10-08", "BOS", "TOR", 3.1, 2.9, 0.55, 111, 222)
        pl.append_prediction(2, "2026-10-08", "NYR", "NJD", 2.8, 3.0, 0.48, 333, 444)

        lines = pl.LOG_PATH.read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        pl.LOG_PATH.write_text("\n".join(lines) + "\n")

        ok, msg = pl.verify_log()
        check("swapping the order of two entries is detected", not ok, msg)


def test_injury_snapshot_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        pl.LOG_PATH = Path(tmp) / "log.jsonl"
        snapshot = [{"player": "Jane Doe", "team": "BOS", "status": "Out"}]
        pl.append_prediction(1, "2026-10-08", "BOS", "TOR", 3.1, 2.9, 0.55, 111, 222,
                              injury_report_snapshot=snapshot)
        entries = pl.read_log()
        check("the injury-report snapshot used for a prediction is stored and recoverable",
              entries[0]["injury_report_snapshot"] == snapshot)


if __name__ == "__main__":
    test_fresh_log_verifies_clean()
    test_editing_a_past_entry_is_detected()
    test_deleting_a_past_entry_is_detected()
    test_reordering_entries_is_detected()
    test_injury_snapshot_round_trips()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    else:
        print("ALL CHECKS PASSED")
