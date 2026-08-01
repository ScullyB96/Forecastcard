"""THE HOLDOUT TOOL (Phase 0.3, 2026-07-22 roadmap): identical machinery to
validate_game_simulator.py, but targeting ONLY the current in-progress
season (TEST_SEASONS = {2026} below) instead of the 2023-2025 seasons every
kept/reverted signal in this project was ever fit or selected against.

Why this matters: every keep/revert decision this project has made --
bat speed, pulled-air rate, GB/FB pitcher HR-share, HFA, park-neutralization,
the latent pitcher-appearance shock (task #137), all of it -- was decided
using 2023-2025 backtests. Even with fully honest per-test methodology,
selection bias accumulates silently across dozens of such decisions, and no
amount of FURTHER 2023-2025 testing can detect it (you're always grading
against the same data the choices were made on, just sliced differently).
The only real check is a season no decision has ever touched. As of
2026-07-23, that's 2026: real complete-lineup games played AFTER every
existing signal in this codebase was already fixed. This is genuinely held
out, not merely walk-forward-safe within a single already-used season.

RULES FOR USING THIS TOOL, per the roadmap: (1) run it against the CURRENT
FROZEN production stack and record the result -- do not tune anything in
response to what it says, or it stops being a holdout and just becomes more
training data; (2) treat "2026" as a ROLLING holdout going forward -- as
future seasons complete, this script (or its direct descendant) should keep
serving as the never-touched check, not get quietly absorbed into the
fitting set the way 2023-2025 were; (3) batch reads (e.g. one per month of
newly-completed games), every read logged in the metrics ledger via
run_validation's own write_ledger=True default.

2026-07-23: rewritten as a thin wrapper around validate_game_simulator's
build_shared_tables/run_validation (refactored out of that file's __main__
block for task #134) instead of maintaining a hand-duplicated, drifting copy
of the whole pipeline -- guarantees this holdout always tests the EXACT same
code path as the production script, including the frozen shock_sigma=0.40
(task #137) and widen_w=1.0 (task #134, never deployed) production config.
"""
import pandas as pd

from src.models.validate_game_simulator import (
    build_shared_tables, run_validation, WIDEN_W, SHOCK_SIGMA,
)
from src.utils.paths import DATA_PROCESSED

TEST_SEASONS = {2026}
N_GAMES_TO_VALIDATE = 5000  # effectively "every complete-lineup 2026 game" -- see build_shared_tables' own note
N_TRIALS_PER_GAME = 200  # RAISED 50->200 (task #140, 2026-07-26) to match validate_game_
                         # simulator.py's own canonical-protocol upgrade -- kept in sync so
                         # the rolling holdout always reflects the SAME protocol precision as
                         # every other canonical run, not a stale K=50 baseline. This constant
                         # change is NOT itself a holdout read -- per this file's own rule
                         # (never run this reactively), the next real read stays on its
                         # pre-committed first-Monday-of-month schedule regardless.

if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    shared = build_shared_tables(pa, TEST_SEASONS)
    run_validation(
        shared, TEST_SEASONS, N_GAMES_TO_VALIDATE, N_TRIALS_PER_GAME,
        widen_w=WIDEN_W, shock_sigma=SHOCK_SIGMA, crn_pairing=False, seed=42,
        output_path=DATA_PROCESSED / "game_simulator_validation_HOLDOUT_2026.parquet",
        write_ledger=True, notes="validate_holdout_2026.py -- rolling 2026 holdout, never used for any fitting/selection decision",
    )
