"""Tests for the data-dependency canaries (Sec42.1 of MODEL_DOCUMENTATION.md,
task #44's degradation budget): confirms the guards actually fire on a
genuine schema break, and do NOT false-positive on real, currently-known-
good data or on the schedule's own legitimate extra gameType values.

Run: `.venv/bin/python -m tests.test_schema_guards`
"""

import pandas as pd

from src.ingest.schema_guards import (
    SchemaGuardError, assert_moneypuck_situational_schema, assert_schedule_schema,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _good_schedule_row(**overrides):
    row = {"gameId": 1, "season": 20252026, "gameType": 2, "gameDate": "2026-01-01",
           "gameState": "OFF", "lastPeriodType": "REG",
           "awayTeamAbbrev": "TOR", "awayScore": 3, "homeTeamAbbrev": "BOS", "homeScore": 4}
    row.update(overrides)
    return row


def test_schedule_guard_passes_on_legitimate_extra_gametypes():
    """A real, legitimate gameType outside {1,2,3} (e.g. an all-star/
    skills-competition variant, confirmed present in the real cached
    schedule) must NOT trip the guard -- only gameType==2 rows' own
    gameState/lastPeriodType are checked."""
    df = pd.DataFrame([
        _good_schedule_row(gameId=1, gameType=2),
        {**_good_schedule_row(gameId=2), "gameType": 14, "gameState": "OFF", "lastPeriodType": None},
    ])
    try:
        assert_schedule_schema(df)
        passed = True
    except SchemaGuardError as e:
        passed = False
        print(f"  unexpected SchemaGuardError: {e}")
    check("guard does not fire on a legitimate extra gameType value (only checks gameType==2)", passed)


def test_schedule_guard_fires_on_missing_column():
    df = pd.DataFrame([_good_schedule_row()]).drop(columns=["lastPeriodType"])
    try:
        assert_schedule_schema(df)
        fired = False
    except SchemaGuardError:
        fired = True
    check("guard fires when a required schedule column is missing", fired)


def test_schedule_guard_fires_on_unrecognized_lastperiodtype():
    """The exact failure mode this guard exists for: a rule change adding
    a new tiebreaker format (or any other new lastPeriodType value) must
    be caught immediately, not silently misclassified by the OT/SO
    sub-model."""
    df = pd.DataFrame([_good_schedule_row(lastPeriodType="3OT")])
    try:
        assert_schedule_schema(df)
        fired = False
    except SchemaGuardError:
        fired = True
    check("guard fires on an unrecognized lastPeriodType value for a decided game", fired)


def _good_moneypuck_row(situation):
    return {"gameId": 1, "team": "TOR", "season": 2025, "situation": situation,
            "gameDate": "2026-01-01", "xGoalsFor": 1.0, "xGoalsAgainst": 1.0, "iceTime": 3000}


def test_moneypuck_guard_fires_on_missing_situation():
    """The exact failure mode that produced Cycle 3's original real bug:
    the 'other' situation silently absent (or any known situation
    disappearing) must be caught immediately, not discovered later as an
    unexplained ~12.5% xG gap."""
    df = pd.DataFrame([_good_moneypuck_row(s) for s in ("5on5", "5on4", "4on5", "all")])  # 'other' missing
    try:
        assert_moneypuck_situational_schema(df)
        fired = False
    except SchemaGuardError:
        fired = True
    check("guard fires when a known situation (e.g. 'other') is missing from the data", fired)


def test_moneypuck_guard_fires_on_new_situation():
    df = pd.DataFrame([_good_moneypuck_row(s) for s in ("5on5", "5on4", "4on5", "all", "other", "3on3")])
    try:
        assert_moneypuck_situational_schema(df)
        fired = False
    except SchemaGuardError:
        fired = True
    check("guard fires when an unrecognized new situation value appears", fired)


def test_moneypuck_guard_passes_on_real_known_good_situations():
    df = pd.DataFrame([_good_moneypuck_row(s) for s in ("5on5", "5on4", "4on5", "all", "other")])
    try:
        assert_moneypuck_situational_schema(df)
        passed = True
    except SchemaGuardError as e:
        passed = False
        print(f"  unexpected SchemaGuardError: {e}")
    check("guard does not fire on the real, currently-known-good situation set", passed)


if __name__ == "__main__":
    test_schedule_guard_passes_on_legitimate_extra_gametypes()
    test_schedule_guard_fires_on_missing_column()
    test_schedule_guard_fires_on_unrecognized_lastperiodtype()
    test_moneypuck_guard_fires_on_missing_situation()
    test_moneypuck_guard_fires_on_new_situation()
    test_moneypuck_guard_passes_on_real_known_good_situations()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    else:
        print("ALL CHECKS PASSED")
