"""Carves out the genuine, never-touched-for-tuning final holdout this
project's own stated discipline (MODEL_DOCUMENTATION.md Sec3.8) has called
for since the architecture proposal, but hadn't actually implemented yet --
flagged as an overdue process gap in Sec4.7 after six cycles of tuning
decisions were all made against the SAME full 2010-11..2025-26 game set.

POLICY, from this point forward:
- DEV_MAX_SEASON (exclusive) splits the validated game set into a
  DEVELOPMENT set (2010-11 through 2023-24, 16,528 games) that every
  future modeling decision -- new signals, refinements, hyperparameter
  tuning -- must be validated against, and a HOLDOUT set (2024-25 +
  2025-26, 2,624 games) that should NOT be looked at again except at
  infrequent, deliberate checkpoints (not after every micro-decision).
- Honesty about a real limitation: the six cycles already completed
  (naive baseline through the calibrated goalie prior) were ALL validated
  against the full range, holdout included -- so this holdout is not
  perfectly pristine relative to decisions already made. It cannot be
  un-seen. What this script establishes is the discipline GOING FORWARD:
  no further tuning against these games from here on, and an honest
  baseline reading of where the current best model actually stands on
  them, taken now, once, rather than never checked at all.
"""

import pandas as pd

DEV_MAX_SEASON = 20242025  # exclusive -- dev set is seasons < this; holdout is seasons >= this


def split_dev_holdout(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = results[results["season"] < DEV_MAX_SEASON]
    holdout = results[results["season"] >= DEV_MAX_SEASON]
    return dev, holdout


if __name__ == "__main__":
    # Local import (not module-level): this file is imported by many other modules purely for
    # DEV_MAX_SEASON/split_dev_holdout; validate_goalie.run_validation is only needed for this
    # script's own __main__ run, and a module-level import here creates a circular import with
    # validate_goalie -> validate_situational_toi -> final_holdout_check (2026-07-25, found while
    # fixing the home_ice_multiplier walk-forward-discipline gap).
    from src.models.metrics_ledger import append_run, compute_run_metrics
    from src.models.validate_goalie import run_validation

    r = run_validation()
    dev, holdout = split_dev_holdout(r)
    print(f"development set: {len(dev)} games (seasons < {DEV_MAX_SEASON})")
    print(f"holdout set: {len(holdout)} games (seasons >= {DEV_MAX_SEASON})")

    dev_row = append_run(
        dev, model_name="goalie_overlay_poisson_DEV_SET_ONLY",
        config_flags={"prior_games_goalie": 12, "split": "development", "dev_max_season": DEV_MAX_SEASON},
        notes="Sec4.8: development-set-only metrics (seasons 2010-11 through 2023-24) for the current best "
              "model, established as the baseline all FUTURE tuning decisions should be validated against, "
              "not the full range used for Cycles 1-6.",
    )
    holdout_row = append_run(
        holdout, model_name="goalie_overlay_poisson_HOLDOUT_CHECK",
        config_flags={"prior_games_goalie": 12, "split": "holdout", "dev_max_season": DEV_MAX_SEASON},
        notes="Sec4.8: honest one-time holdout reading (2024-25 + 2025-26 seasons) of the current best model. "
              "NOT pristine -- Cycles 1-6 were all validated against the full range including these games -- "
              "but from this point forward these games should not inform any further tuning decision.",
    )

    print("\n--- development set ---")
    for k, v in dev_row.items():
        print(f"  {k}: {v}")
    print("\n--- holdout set ---")
    for k, v in holdout_row.items():
        print(f"  {k}: {v}")
