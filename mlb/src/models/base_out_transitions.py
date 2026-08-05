"""Empirical base-out state transition table: given a PRE-PA state and an
outcome category (single, walk, double_play, ...), what actually happens to
the base-out state and how many runs score? This is the league-wide
"baserunning physics" layer -- NOT matchup-specific (a given single's
baserunner advancement doesn't meaningfully depend on which two players were
involved, at least for a first version; that's a real, documented, but
second-order effect worth revisiting later, not blocking).

Design: rather than fit a parametric model, this resamples directly from the
empirical historical (post_state, runs_scored) pairs observed for each
(pre_state, outcome) combination -- a nonparametric bootstrap. This is simple
and captures real variability (e.g. some singles advance a runner from 1st to
3rd, most only to 2nd) without needing to model *why* explicitly.

Caveat worth remembering: 2023's rule changes (bigger bases, pitch clock,
limited pickoffs) measurably increased aggressive baserunning/steal success.
All data cached so far (2023-2026) is post-rule-change and can be pooled
safely, but this table should NOT be built from a mix of pre- and post-2023
data without accounting for that shift.
"""

import numpy as np
import pandas as pd

# Component-level test (2026-07-22, real 2023-2026 PA table, restricted to the
# two single-runner pre-states where post-state attribution is unambiguous --
# pre_state bitmask==1 "runner on 1st ONLY" and bitmask==2 "runner on 2nd
# ONLY"): does the REAL runner's own sprint speed predict how far they
# advance? Runner-on-1st + outcome=single -> advances to 3rd/scores: slow
# 26.6%, mid 35.6%, fast 43.3% (n=16,348, a genuine 16.7pp spread). Runner-
# on-2nd + outcome=single -> scores (vs stops at 3rd): slow 53.6%, mid 62.4%,
# fast 65.2% (n=5,217, an 11.6pp spread). Runner-on-1st + outcome=double ->
# scores (vs stops at 3rd) showed NO effect (98-99% across every tercile,
# already near-ceiling -- a runner scores from 1st on a double almost
# regardless of speed) -- NOT conditioned, since there's nothing to gain.
#
# FULL-STACK A/B RESULT (2026-07-22, n=597, wired into game_simulator.py's
# simulate_half_inning, only meaningful/possible when sb_rates runner-identity
# tracking is active): despite the clean, large, real component-level effect
# above, this REGRESSED straight-up accuracy meaningfully -- SU 60.5%->57.3%
# (-3.2pp), one of the largest SU regressions found this session, despite
# total MAE improving slightly (3.409->3.401) and margin MAE moving only
# marginally (3.445->3.450). REVERTED from game_simulator.py and all 3
# consumer files (validate_game_simulator.py, validate_predictive_bullpen.py,
# props.py) -- the same "real, validated component-level signal doesn't
# survive contact with the full-stack simulator" pattern already documented
# for the pitch-by-pitch DP swap, whiff-rate, and pitcher-stuff. The
# functions/constants below (sprint_speed_tercile, attach_runner_speed_bucket,
# TransitionTable's runner_speed_bucket support) are kept as a documented-but-
# unused artifact, same treatment as every other investigated-and-rejected
# idea this session -- NOT currently wired into anything.
# FIXED (not re-fit per season) sprint-speed tercile cutoffs, same
# look-ahead-avoidance rationale as spray.py's PULL_TERCILE_CUTOFFS -- derived
# once from the full Baseball Savant Sprint Speed leaderboard population
# (2023-2025, n=1728 player-seasons).
SPRINT_SPEED_TERCILE_CUTOFFS = (26.8, 28.0)


def sprint_speed_tercile(sprint_speed: float) -> str:
    lo, hi = SPRINT_SPEED_TERCILE_CUTOFFS
    if sprint_speed < lo:
        return "slow"
    if sprint_speed < hi:
        return "mid"
    return "fast"


def attach_runner_speed_bucket(pa: pd.DataFrame, speed_by_season: pd.DataFrame) -> pd.DataFrame:
    """Adds a `runner_speed_bucket` column (str, NaN everywhere else) for the
    two validated single-runner scenarios above. `speed_by_season`: the full
    Sprint Speed leaderboard (see expected_stats.py's load_sprint_speed_by_season),
    columns batter/season/sprint_speed. Joined directly against the ACTUAL
    on-base runner for that historical PA (on_1b/on_2b) -- this table is built
    from the full historical record (not a walk-forward split), same as every
    other TransitionTable input, since league-wide baserunning physics is a
    stable empirical property being resampled, not a player skill being
    predicted forward."""
    pa = pa.copy()
    pa["runner_speed_bucket"] = pd.Series(np.nan, index=pa.index, dtype=object)
    speed_lookup = speed_by_season.rename(columns={"sprint_speed": "_speed"})

    bitmask = pa["pre_state"] // 10
    r1_only = (bitmask == 1) & (pa["outcome"] == "single")
    r2_only = (bitmask == 2) & (pa["outcome"] == "single")

    for mask, runner_col in [(r1_only, "on_1b"), (r2_only, "on_2b")]:
        idx = pa.index[mask]
        if len(idx) == 0:
            continue
        joined = pa.loc[idx, [runner_col, "season"]].merge(
            speed_lookup.rename(columns={"batter": runner_col}), on=[runner_col, "season"], how="left"
        )
        pa.loc[idx, "runner_speed_bucket"] = joined["_speed"].apply(
            lambda s: sprint_speed_tercile(s) if pd.notna(s) else np.nan
        ).values
    return pa


class TransitionTable:
    def __init__(self, pa_table: pd.DataFrame, min_samples: int = 5):
        """min_samples: (pre_state, outcome) combos with fewer than this many
        historical examples fall back to a coarser table, in two tiers --
        some state/outcome combinations are extremely rare (e.g. a triple
        with the bases already loaded and 2 outs) and a handful of raw
        samples would let one freak outcome dominate the resample.

        The two fallback tiers are NOT equally safe. Tier 2 (by_outs_outcome,
        added 2026-07-21 after an audit found a real bug in the original
        single-tier design) pools by (outs, outcome) -- e.g. a rare
        fielders_choice at 2 outs falls back to ALL fielders_choice plays
        that ALSO started at 2 outs, regardless of base occupancy. Tier 3
        (by_outcome_only, the ORIGINAL sole fallback) pools an outcome across
        EVERY out count -- confirmed on real data to be a genuine bug
        source: for (state, outcome) cells with fewer than min_samples where
        tier 2 also wasn't available, falling straight to tier 3 let outs
        DECREASE across a PA in ~95-98% of sampled draws for two real cells
        checked (fielders_choice and catcher_interf at a 2-out state) --
        physically impossible in real baseball, and it let the simulator
        incorrectly continue a half-inning past what should have been the
        3rd out. Tier 2 is out-count-consistent by construction and closes
        that gap for the overwhelming majority of cases; tier 3 remains only
        as a genuine last resort for a (outs, outcome) pair that ALSO has
        fewer than min_samples pooled across every base-occupancy pattern.

        Optional runner-speed conditioning (2026-07-22, see
        attach_runner_speed_bucket above for the validated rationale): if
        pa_table has a `runner_speed_bucket` column (str, NaN where not
        applicable), an ADDITIONAL, more granular table is built keyed by
        (pre_state, outcome, bucket) for exactly the rows where it's set --
        used only when the caller explicitly passes a runner_speed_bucket to
        sample(); every existing caller that never attaches this column
        (or that omits the argument to sample()) gets byte-for-byte identical
        behavior to before this feature existed.
        """
        self._by_state_outcome: dict[tuple[int, str], np.ndarray] = {}
        self._by_state_outcome_speed: dict[tuple[int, str, str], np.ndarray] = {}
        self._by_outs_outcome: dict[tuple[int, str], np.ndarray] = {}
        self._by_outcome_only: dict[str, np.ndarray] = {}
        # (pre_state, outcome, runner_speed_bucket) -> mean runs_scored, lazily
        # populated by conditional_mean_runs() (2026-08-04, control-variate
        # prep) -- this table is immutable after __init__, so memoizing is safe.
        self._mean_cache: dict[tuple, float] = {}

        cols = ["post_state", "terminal", "runs_scored"]
        has_speed_bucket = "runner_speed_bucket" in pa_table.columns
        for outcome, sub in pa_table.groupby("outcome"):
            arr = sub[cols].copy()
            arr["post_state"] = arr["post_state"].fillna(-1).astype(int)
            self._by_outcome_only[outcome] = arr[["post_state", "runs_scored"]].to_numpy()

            outs_of_state = sub["pre_state"] % 10
            for outs, sub_outs in sub.groupby(outs_of_state):
                if len(sub_outs) >= min_samples:
                    a_outs = sub_outs[cols].copy()
                    a_outs["post_state"] = a_outs["post_state"].fillna(-1).astype(int)
                    self._by_outs_outcome[(int(outs), outcome)] = a_outs[["post_state", "runs_scored"]].to_numpy()

            if has_speed_bucket:
                speed_rows = sub[sub["runner_speed_bucket"].notna()]
                for (state, bucket), sub_speed in speed_rows.groupby(["pre_state", "runner_speed_bucket"]):
                    if len(sub_speed) >= min_samples:
                        a_speed = sub_speed[cols].copy()
                        a_speed["post_state"] = a_speed["post_state"].fillna(-1).astype(int)
                        self._by_state_outcome_speed[(state, outcome, bucket)] = (
                            a_speed[["post_state", "runs_scored"]].to_numpy()
                        )

            for state, sub2 in sub.groupby("pre_state"):
                if len(sub2) >= min_samples:
                    a2 = sub2[cols].copy()
                    a2["post_state"] = a2["post_state"].fillna(-1).astype(int)
                    self._by_state_outcome[(state, outcome)] = a2[["post_state", "runs_scored"]].to_numpy()

    def sample(self, pre_state: int, outcome: str, rng: np.random.Generator,
               runner_speed_bucket: str | None = None,
               crn_key: tuple | None = None) -> tuple[int | None, int]:
        """Returns (post_state, runs_scored). post_state is None if the
        half-inning ended on this play.

        runner_speed_bucket: optional "slow"/"mid"/"fast" (see
        sprint_speed_tercile) -- when given, tries the speed-conditioned
        table FIRST, falling through to the normal 3-tier chain below if
        this specific (pre_state, outcome, bucket) wasn't built (either
        because this outcome/state combo isn't one of the two validated
        scenarios, or because it was too sparse). Omitting this argument
        (every pre-existing caller) is byte-for-byte identical to before
        this feature existed.

        crn_key: optional tuple of ints (see crn.py) -- when given, the
        historical row is picked via a deterministic hash of this key
        instead of consuming `rng`'s next draw, for paired A/B testing
        (see game_simulator.py's crn_game_pk/crn_trial params). `rng` is
        still required (crn_key is strictly additive) but goes unused in
        that case."""
        arr = self._resolve_array(pre_state, outcome, runner_speed_bucket)
        if crn_key is not None:
            from src.models.crn import crn_index
            idx = crn_index(len(arr), *crn_key)
        else:
            idx = rng.integers(0, len(arr))
        row = arr[idx]
        post_state = None if row[0] == -1 else int(row[0])
        return post_state, int(row[1])

    def _resolve_array(self, pre_state: int, outcome: str,
                        runner_speed_bucket: str | None) -> np.ndarray:
        """The same 4-tier fallback chain sample() has always used (speed
        bucket -> granular state -> outs-only -> outcome-only, with M14's
        outs-preserving filter on the last-resort tier), factored out
        (2026-08-04, control-variate prep) so conditional_mean_runs() below
        can NEVER resolve to a different population than sample() actually
        draws from -- both call this exact method, so the two are
        guaranteed to agree by construction. No behavior change from the
        pre-refactor inline version."""
        arr = None
        if runner_speed_bucket is not None:
            arr = self._by_state_outcome_speed.get((pre_state, outcome, runner_speed_bucket))
        if arr is None:
            arr = self._by_state_outcome.get((pre_state, outcome))
        if arr is None:
            arr = self._by_outs_outcome.get((pre_state % 10, outcome))
        if arr is None:
            arr = self._by_outcome_only.get(outcome)
            if arr is not None and len(arr):
                # Outs-preserving guard on the outcome-only last resort
                # (2026-08-03 audit, finding M14): this tier pools rows
                # across every out count, so a 2-out draw could return a
                # 0/1-out context's post_state -- outs went BACKWARDS,
                # un-ending the half-inning. Keep only rows that are
                # terminal (post_state == -1; the half-inning ended, outs
                # can only have gone UP) or whose post outs >= pre outs.
                # If nothing survives (never observed on real data), the
                # unfiltered array is kept rather than crashing -- the
                # state-factor hard zeros upstream make reaching here for
                # an impossible outcome ~impossible to begin with.
                mask = (arr[:, 0] == -1) | (arr[:, 0] % 10 >= pre_state % 10)
                filtered = arr[mask]
                if len(filtered):
                    arr = filtered
        if arr is None or len(arr) == 0:
            raise ValueError(f"no historical data for outcome={outcome!r} (pre_state={pre_state})")
        return arr

    def conditional_mean_runs(self, pre_state: int, outcome: str,
                               runner_speed_bucket: str | None = None) -> float:
        """The exact conditional mean of runs_scored that sample() would
        draw from for this (pre_state, outcome[, bucket]) -- the analytic
        control variate the Monte Carlo variance-reduction work needs (see
        validate_control_variate.py): (runs_scored - conditional_mean_runs(...))
        has expectation exactly zero for every PA, by construction, since it
        shares _resolve_array with sample() (guaranteed same population, no
        risk of the two silently drifting apart). Memoized -- this table is
        immutable after __init__, so the same key always resolves to the
        same array/mean, and this is called on the hot simulation path when
        control-variate tracking is enabled."""
        key = (pre_state, outcome, runner_speed_bucket)
        cached = self._mean_cache.get(key)
        if cached is None:
            arr = self._resolve_array(pre_state, outcome, runner_speed_bucket)
            cached = float(arr[:, 1].mean())
            self._mean_cache[key] = cached
        return cached

    def coverage_report(self, pa_table: pd.DataFrame) -> pd.DataFrame:
        """What fraction of (pre_state, outcome) combos in the data have
        enough samples to use the granular table vs. falling back?"""
        counts = pa_table.groupby(["pre_state", "outcome"]).size().reset_index(name="n")
        counts["has_granular"] = counts.apply(lambda r: (r["pre_state"], r["outcome"]) in self._by_state_outcome, axis=1)
        return counts


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    table = TransitionTable(pa)

    report = table.coverage_report(pa)
    weighted_coverage = report.loc[report["has_granular"], "n"].sum() / report["n"].sum()
    print(f"granular (state,outcome) coverage: {weighted_coverage:.4%} of all PAs "
          f"({report['has_granular'].sum()}/{len(report)} combos have >= 5 samples)")

    rng = np.random.default_rng(0)
    print("\n=== sanity: sample 5 outcomes for pre_state=11 (runner on 1st, 1 out), outcome='single' ===")
    for _ in range(5):
        print(" ", table.sample(11, "single", rng))

    print("\n=== sanity: sample 5 outcomes for pre_state=0 (bases empty, 0 outs), outcome='home_run' ===")
    for _ in range(5):
        print(" ", table.sample(0, "home_run", rng))

    print("\n=== sanity: sample 5 outcomes for pre_state=71 (bases loaded, 1 out), outcome='double_play' ===")
    for _ in range(5):
        print(" ", table.sample(71, "double_play", rng))
