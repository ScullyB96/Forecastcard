"""Regression guards for real bugs found while building and validating the
live single-game predictor (Sec41 of MODEL_DOCUMENTATION.md, 2026-07-25).

Each bug was found by predicting a REAL, already-completed game using only
data through the day before, and comparing against `run_treated()`'s own
already-computed output for that exact game -- the cleanest available
correctness check, since a real historical game's walk-forward state "as
of yesterday" should reproduce that game's own row almost exactly.

Run: `.venv/bin/python -m tests.test_predict_game`
"""

import numpy as np
import pandas as pd

from src.models.predict_game import _latest_before, goalie_relative_as_of

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_latest_before_excludes_same_day_games_entirely():
    """Bug found validating against a real 4-game day (2026-04-06): the
    reg-ratio/tie-mass-calibration walk-forward series update PER GAME, not
    per calendar day, so on a date with multiple games each one gets a
    slightly different value depending on its exact (gameDate, gameId)
    sort position. Predicting ANY of that day's games must use the state
    from strictly BEFORE that whole day -- not "whichever game happens to
    sort last that day" -- since none of that day's OTHER games have
    happened yet from a live-prediction standpoint either. Confirms
    `_latest_before` returns the value from the day before, identically,
    regardless of how many games (or which one) share the target date."""
    games_sorted = pd.DataFrame({
        "gameDate": ["2026-04-05", "2026-04-06", "2026-04-06", "2026-04-06"],
        "gameId": [1, 2, 3, 4],
        "some_ratio": [0.900, 0.910, 0.920, 0.930],
    })
    value_for_2025 = _latest_before(games_sorted, pd.Timestamp("2026-04-06"), "some_ratio")
    check("value as of a multi-game day comes from the day before, not any of that day's own games",
          value_for_2025 == 0.900, f"got {value_for_2025}, expected 0.900 (the 2026-04-05 row)")


def test_goalie_relative_as_of_advances_one_full_step_not_the_prior_rows_own_value():
    """Bug found validating against a real single-game day (2026-03-23,
    NYR vs OTT): a goalie's own historical appearance ROW stores the value
    ENTERING that appearance, not the value after incorporating its own
    result. Returning "the last prior appearance's own stored value" is
    off by exactly one walk-forward step. Confirmed real on goalie
    8473503.0: his 2026-03-19 appearance's own stored `goalie_relative`
    was -0.189206, but his ACTUAL 2026-03-23 appearance (the very next
    real game in this same walk-forward series) shows -0.215250 -- the
    correct value `goalie_relative_as_of` must reproduce when asked for
    his state "as of just before 2026-03-23", using only his real
    appearances strictly before that date."""
    # A synthetic goalie with two real prior appearances of known gsax, well above the
    # league-average path so the shrunk mean moves in a clearly checkable direction.
    raw = pd.DataFrame([
        {"gameId": 100, "goalieIdForShot": 999.0, "team": "OTT", "gsax": 5.0,
         "gameDate": "2026-01-01", "season": 20252026},
        {"gameId": 101, "goalieIdForShot": 999.0, "team": "OTT", "gsax": 5.0,
         "gameDate": "2026-01-10", "season": 20252026},
        # A different goalie, present only so the pooled league-average has more than one
        # column and the test isn't trivially degenerate.
        {"gameId": 102, "goalieIdForShot": 1.0, "team": "TOR", "gsax": -1.0,
         "gameDate": "2026-01-05", "season": 20252026},
    ])
    as_of = pd.Timestamp("2026-01-20")
    result = goalie_relative_as_of(raw, 999.0, as_of, "OTT")

    # Ground truth: append the same real appearances PLUS a genuine next-game placeholder
    # directly (mirroring what a correct implementation must produce), using the real
    # production function so this isn't a hand-derived, possibly-wrong reference.
    from src.models.team_strength_goalie import add_walk_forward_goalie_strength
    placeholder = pd.DataFrame([{"gameId": -1, "goalieIdForShot": 999.0, "team": "OTT", "gsax": 0.0,
                                  "gameDate": "2026-01-20", "season": 20252026}])
    augmented = pd.concat([raw, placeholder], ignore_index=True)
    rated = add_walk_forward_goalie_strength(augmented)
    rated["goalie_relative"] = rated["goalie_shrunk_mean"] - rated["goalie_league_avg"]
    expected = float(rated.loc[rated["gameId"] == -1, "goalie_relative"].iloc[0])

    check("goalie_relative_as_of matches the real walk-forward function's own next-step value",
          abs(result - expected) < 1e-9, f"got {result}, expected {expected}")

    # And explicitly NOT equal to the goalie's own second real appearance's stored value
    # (gameId 101's own row) -- that would be the bug this test guards against.
    stale_value = rated.loc[rated["gameId"] == 101, "goalie_relative"].iloc[0] if \
        "goalie_relative" in rated.columns and (rated["gameId"] == 101).any() else None
    prior_rows = add_walk_forward_goalie_strength(raw)
    prior_rows["goalie_relative"] = prior_rows["goalie_shrunk_mean"] - prior_rows["goalie_league_avg"]
    stale_value = float(prior_rows.loc[prior_rows["gameId"] == 101, "goalie_relative"].iloc[0])
    check("goalie_relative_as_of does NOT just return the last prior appearance's own stored value",
          abs(result - stale_value) > 1e-6, f"got {result}, which equals the stale value {stale_value}")


def test_goalie_relative_placeholder_gamedate_stays_string_typed():
    """Bug found while debugging the test above: a `pd.Timestamp` placeholder
    `gameDate` alongside real plain-string rows creates a mixed-type column
    that silently breaks the walk-forward function's own date sort (NaN
    result instead of a raised error). Confirms the placeholder's gameDate
    is always a plain string, matching every real row's dtype."""
    raw = pd.DataFrame([
        {"gameId": 100, "goalieIdForShot": 999.0, "team": "OTT", "gsax": 5.0,
         "gameDate": "2026-01-01", "season": 20252026},
    ])
    result = goalie_relative_as_of(raw, 999.0, pd.Timestamp("2026-01-20"), "OTT")
    check("goalie_relative_as_of returns a real (non-NaN) number, not silently NaN from a "
          "mixed-type gameDate column", not np.isnan(result), f"got {result}")


if __name__ == "__main__":
    print("=== test_latest_before_excludes_same_day_games_entirely ===")
    test_latest_before_excludes_same_day_games_entirely()
    print("\n=== test_goalie_relative_as_of_advances_one_full_step_not_the_prior_rows_own_value ===")
    test_goalie_relative_as_of_advances_one_full_step_not_the_prior_rows_own_value()
    print("\n=== test_goalie_relative_placeholder_gamedate_stays_string_typed ===")
    test_goalie_relative_placeholder_gamedate_stays_string_typed()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s) failed: {FAILURES}")
        raise SystemExit(1)
    else:
        print("ALL CHECKS PASSED")
