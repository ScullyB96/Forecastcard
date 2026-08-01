"""Task #156 (pre-registered 2026-07-25, per external review item #3): the
one-time END-OF-SEASON read that finally resolves bat speed and pulled-air
rate, this project's two longest-tenured "plausible-but-unconfirmed"
signals.

History (see MODEL_DOCUMENTATION.md sec 11.7/11.9 for full detail): both
were originally "kept" on a noisy n=597/200-trial point estimate. Re-tested
at n=995/500 trials (sec 11.7): both CIs included zero (~4pp wide) --
genuinely unresolved, not disproven. Re-tested again at n=7237 (sec 11.9,
after self-catching a stale-baseline bug where the comparison arm didn't
include the already-wired GB/FB pitcher signal): both CIs still included
zero, but roughly HALF as wide (~2.5pp) -- a real null was starting to
separate from a merely underpowered one, but sec 11.7's own estimate is
that "multiple thousands" more games are needed to fully resolve this.

By the end of the 2026 season, ~2,400 more real games become available on
top of the n=7237 protocol (2023-2025), for a combined n approaching
~9,600-9,700 -- run this script THEN, not before, and not iteratively
(same "run once, don't peek and re-run" discipline as the 2026 H1 holdout,
sec 0.3).

PRE-REGISTERED DECISION RULE (written down now, before the data exists, so
the call can't be shaded after the fact):
- If BOTH signals' Brier CI excludes zero in the beneficial direction:
  upgrade both from "plausible-but-unconfirmed" to CONFIRMED real wins.
- If BOTH CIs still include zero even at this much larger n: this licenses
  REMOVING them on simplicity grounds -- not just re-documenting the same
  ambiguous status a third time. "Remove" means: delete
  CONTACT_QUALITY_BATSPEED_*/bat_speed plumbing from expected_stats.py and
  its callers, delete HR_SHARE_PULLEDAIR_*/pulled_air plumbing, matching
  this project's own stated discipline (dead weight that never proved out
  doesn't get to stay just because removing it is more work than leaving
  it). Log the removal decision and the exact CIs that justified it.
- If MIXED (one resolves, one doesn't): treat each independently on its own
  merits, following whichever branch above applies to it individually.

Reuses `validate_game_simulator.py`'s own `run_validation`, which gained
`disable_bat_speed`/`disable_pulled_air` flags (2026-07-25) specifically
for this test -- both False is a byte-for-byte no-op, i.e. the TRUE current
production baseline, avoiding the exact stale/mismatched-baseline bug class
sec 11.9 caught (all three arms below are generated in the SAME run of
THIS script, so there is no way for one arm to silently reflect an older
code state than another).
"""
import sys

import pandas as pd

from src.models.ab_significance import bootstrap_compare
from src.models.metrics_ledger import append_run
from src.models.validate_game_simulator import (
    SHOCK_SIGMA,
    build_shared_tables,
    run_validation,
)
from src.utils.paths import DATA_PROCESSED, DATA_RAW

N_TRIALS = 200  # matches the canonical protocol -- task #140 (2026-07-26) raised
                 # this from 50 to 200; kept in sync here so this one-shot,
                 # no-redo protocol runs at full intended precision (task #160
                 # correctness audit caught this at 50, stale since task #140).
N_GAMES = 25000  # effectively "every complete-lineup game available" --
                  # matches N_GAMES_TO_VALIDATE's own "not really a cap" role.


def main(test_seasons: set[int] | None = None) -> None:
    if test_seasons is None:
        # Default to every season with cached data at run time, not a
        # hardcoded set -- by October 2026 this should include the full,
        # completed 2026 season on top of 2023-2025.
        test_seasons = {
            int(p.stem.split("_")[1]) for p in DATA_RAW.glob("schedule_*.parquet")
        }
    print(f"resolving bat speed / pulled-air rate on test_seasons={sorted(test_seasons)}...", flush=True)

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    pa = pa[pa["season"].isin(test_seasons)]
    shared = build_shared_tables(pa, test_seasons)

    print("\n--- arm 1/3: baseline (current production, both signals ON) ---", flush=True)
    baseline_path = DATA_PROCESSED / "resolve_bsp_BASELINE.parquet"
    run_validation(shared, test_seasons, N_GAMES, N_TRIALS, shock_sigma=SHOCK_SIGMA,
                   output_path=baseline_path, write_ledger=False,
                   notes="task #156: baseline arm (both bat_speed and pulled_air ON)")

    print("\n--- arm 2/3: bat speed OFF ---", flush=True)
    bat_speed_off_path = DATA_PROCESSED / "resolve_bsp_BATSPEED_OFF.parquet"
    run_validation(shared, test_seasons, N_GAMES, N_TRIALS, shock_sigma=SHOCK_SIGMA,
                   disable_bat_speed=True,
                   output_path=bat_speed_off_path, write_ledger=False,
                   notes="task #156: bat_speed OFF arm")

    print("\n--- arm 3/3: pulled-air rate OFF ---", flush=True)
    pulled_air_off_path = DATA_PROCESSED / "resolve_bsp_PULLEDAIR_OFF.parquet"
    run_validation(shared, test_seasons, N_GAMES, N_TRIALS, shock_sigma=SHOCK_SIGMA,
                   disable_pulled_air=True,
                   output_path=pulled_air_off_path, write_ledger=False,
                   notes="task #156: pulled_air OFF arm")

    print("\n\n=== BAT SPEED: baseline (ON) vs. bat_speed OFF ===", flush=True)
    bat_speed_result = bootstrap_compare(str(baseline_path), str(bat_speed_off_path),
                                         label_a="bat_speed_ON", label_b="bat_speed_OFF")
    for k, v in bat_speed_result.items():
        print(f"  {k}: {v}", flush=True)

    print("\n=== PULLED-AIR RATE: baseline (ON) vs. pulled_air OFF ===", flush=True)
    pulled_air_result = bootstrap_compare(str(baseline_path), str(pulled_air_off_path),
                                          label_a="pulled_air_ON", label_b="pulled_air_OFF")
    for k, v in pulled_air_result.items():
        print(f"  {k}: {v}", flush=True)

    def resolved(result: dict) -> str:
        su_ci = result["su_delta_ci95"]
        brier_ci = result["brier_delta_ci95"]
        # Brier is the decision metric (lower is better -- see sec 11.6's
        # own "Brier over raw SU" precedent) -- ON beating OFF means a
        # NEGATIVE brier_delta (ON's brier minus OFF's brier < 0).
        brier_excludes_zero = brier_ci[0] > 0 or brier_ci[1] < 0
        if not brier_excludes_zero:
            return "STILL NULL at this n -- per pre-registered rule, license to REMOVE on simplicity grounds"
        return "CONFIRMED real (Brier CI excludes zero)" if result["brier_delta"] < 0 else \
               "CONFIRMED real REGRESSION (Brier CI excludes zero, wrong direction -- revert immediately)"

    print(f"\n\n=== PRE-REGISTERED VERDICT ===", flush=True)
    print(f"bat speed: {resolved(bat_speed_result)}", flush=True)
    print(f"pulled-air rate: {resolved(pulled_air_result)}", flush=True)

    # task #160 (2026-07-26 correctness audit) fix: this previously read
    # baseline_path for BOTH labels, so both ledger rows' numeric columns
    # (su_primary/brier/etc.) were always the baseline's own metrics,
    # identical regardless of which arm was being logged -- defeating the
    # whole point of metrics_ledger.py (letting a future reader verify a
    # result from the row's own numbers, not just trust the notes text).
    # Each row now carries its OWN arm's actual OFF-path metrics.
    for label, result, off_path in [("bat_speed", bat_speed_result, bat_speed_off_path),
                                     ("pulled_air", pulled_air_result, pulled_air_off_path)]:
        r = pd.read_parquet(off_path)
        append_run(r, config_flags={"task": "156_resolve_bat_speed_pulled_air", "arm": label,
                                     "test_seasons": sorted(test_seasons), "n_trials": N_TRIALS},
                   n_trials=N_TRIALS,
                   notes=(f"Task #156 pre-registered resolution read. {label}: SU delta="
                          f"{result['su_delta']:.4f} CI={result['su_delta_ci95']}, Brier delta="
                          f"{result['brier_delta']:.4f} CI={result['brier_delta_ci95']}. "
                          f"Verdict: {resolved(result)}"))


if __name__ == "__main__":
    seasons_arg = None
    if len(sys.argv) > 1:
        seasons_arg = {int(s) for s in sys.argv[1].split(",")}
    main(seasons_arg)
