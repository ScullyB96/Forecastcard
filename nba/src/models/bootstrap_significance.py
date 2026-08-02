"""Shared paired-bootstrap significance testing, used by every phase's
validate_*.py as the standard adoption gate (5,000 resamples, keyed by
gameId). Reimplemented independently for this project (no import from the
sibling mlb/nhl projects), generalized to compare an arbitrary list of
per-game metrics between two arms (e.g. "team-strength-only" vs
"team-strength + RAPM-lite") rather than a fixed SU/Brier pair, since this
project's phases compare different metric sets at different stages
(MAE-type metrics in Phase 1-2, calibration/CRPS-type metrics in Phase 3).

Not a Monte Carlo game simulator (that's the MLB sibling's approach, not
this project's -- see MODEL_DOCUMENTATION.md) -- this resamples GAMES (not
simulated trials) with replacement, which is the correct unit for comparing
two deterministic point-prediction models on the same real historical games.
"""

import numpy as np
import pandas as pd

N_BOOTSTRAP = 5000
_MAX_IDX_MATRIX_BYTES = 200_000_000  # ~200MB cap on any single resample-index batch, see below


def bootstrap_compare(per_game_a: pd.DataFrame, per_game_b: pd.DataFrame, game_id_col: str,
                       metrics: list[dict], label_a: str = "A", label_b: str = "B",
                       n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """per_game_a/per_game_b: one row per game, each containing `game_id_col`
    plus every column named in `metrics[i]["col"]`. `metrics`: list of
    {"name": str, "col": str, "higher_is_better": bool} -- both frames must
    have this same column name (values differ between arms, hence two frames).
    Returns a dict keyed by metric name, each with point estimates, a 95% CI
    on the paired delta (a - b), and a plain-English verdict; also prints a
    human-readable report.

    BUG FOUND AND FIXED (2026-07-25): the original version built ONE
    `(n_bootstrap, n_games)` index matrix upfront (`rng.integers(0, n_games,
    size=(n_bootstrap, n_games))`) -- fine at team-level scale (n_games ~
    10,000: a 5000x10000 int64 array is ~400MB), but this project's
    player-level validation scripts pass n_games in the HUNDREDS OF
    THOUSANDS (one row per player-game, not per game) -- confirmed real:
    267,124 rows x 5,000 resamples x 8 bytes = ~10.7GB for the index matrix
    ALONE, before even indexing into it, which caused the whole process to
    thrash/swap and never finish (observed directly: stuck in the same
    spot for minutes, process state 'U' uninterruptible-sleep, climbing
    RSS). Fixed by processing resamples in memory-capped BATCHES (each
    batch's index matrix capped at `_MAX_IDX_MATRIX_BYTES`, computed
    dynamically from `n_games` so it scales down automatically as
    `n_games` grows) rather than materializing every resample at once --
    identical statistical result (still exactly `n_bootstrap` real
    resamples, same percentile CI), just bounded peak memory regardless of
    how large `n_games` is."""
    merged = per_game_a.merge(per_game_b, on=game_id_col, suffixes=("_a", "_b"), how="inner")
    n_games = len(merged)
    if n_games == 0:
        raise ValueError(f"no overlapping {game_id_col} between the two arms")

    rng = np.random.default_rng(seed)
    batch_size = max(1, min(n_bootstrap, _MAX_IDX_MATRIX_BYTES // (8 * n_games)))

    arrays_a = {spec["name"]: merged[f"{spec['col']}_a"].to_numpy() for spec in metrics}
    arrays_b = {spec["name"]: merged[f"{spec['col']}_b"].to_numpy() for spec in metrics}
    deltas = {spec["name"]: np.empty(n_bootstrap) for spec in metrics}

    for start in range(0, n_bootstrap, batch_size):
        end = min(start + batch_size, n_bootstrap)
        idx = rng.integers(0, n_games, size=(end - start, n_games))
        for spec in metrics:
            name = spec["name"]
            deltas[name][start:end] = arrays_a[name][idx].mean(axis=1) - arrays_b[name][idx].mean(axis=1)
        del idx

    results = {}
    print(f"=== paired bootstrap: {label_a} vs {label_b} (n={n_games} games, {n_bootstrap} resamples) ===", flush=True)
    for spec in metrics:
        name, higher_is_better = spec["name"], spec["higher_is_better"]
        a_arr, b_arr = arrays_a[name], arrays_b[name]

        point = float(a_arr.mean() - b_arr.mean())
        ci = np.percentile(deltas[name], [2.5, 97.5])
        excludes_zero = ci[0] > 0 or ci[1] < 0
        if not excludes_zero:
            verdict = "NOISE -- CI includes zero"
        else:
            better = (point > 0) if higher_is_better else (point < 0)
            verdict = f"REAL {'IMPROVEMENT' if better else 'REGRESSION'} -- CI excludes zero"

        results[name] = {
            "a": float(a_arr.mean()), "b": float(b_arr.mean()), "delta": point,
            "delta_ci95": (float(ci[0]), float(ci[1])), "verdict": verdict,
        }
        print(f"{name:16s} {label_a}={a_arr.mean():.4f}  {label_b}={b_arr.mean():.4f}  "
              f"delta={point:+.4f}  95% CI=({ci[0]:+.4f}, {ci[1]:+.4f})  -> {verdict}", flush=True)

    results["n_games"] = n_games
    return results


def block_bootstrap_compare(per_row_a: pd.DataFrame, per_row_b: pd.DataFrame, row_id_col: str,
                             block_col: str, metrics: list[dict], label_a: str = "A", label_b: str = "B",
                             n_bootstrap: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Block/cluster variant of `bootstrap_compare` (2026-08-02, the audit's flagged
    validation-methodology lever): resamples whole BLOCKS (e.g. team-season) with replacement instead
    of individual rows, so within-block serial correlation -- e.g. a team's residuals correlated
    across its own consecutive games, from a systematic rating mispricing that persists for a few
    games before the walk-forward fit catches up -- isn't silently assumed away by treating every
    row as an independent draw the way the plain per-row bootstrap does. This only matters for
    per-TEAM-SIDE data (one row per team per game, e.g. the "arms" format most of this project's own
    validation scripts already build via a home/away split) -- a per-GAME row (both teams combined)
    has no single well-defined "team" to block on, since each game involves two.

    `row_id_col`: uniquely identifies each row (e.g. "{gameId}_{side}"), same role as
    `bootstrap_compare`'s `game_id_col`. `block_col`: which correlated group each row belongs to (e.g.
    "team" or "{team}_{season}") -- must be present and identical across both arms after the merge (a
    property of the row itself, not of which arm is being compared).

    Each resample draws `n_blocks` blocks WITH replacement (n_blocks = number of unique blocks
    present) and pools EVERY row belonging to each drawn block together -- a block drawn twice
    contributes its full row set twice, the standard cluster-bootstrap treatment for panel data with
    unequal block sizes. Correctly WIDENS the CI relative to a naive per-row bootstrap when real
    within-block correlation exists (regression-tested with synthetic block-correlated data); when no
    such correlation exists, gives an equivalent CI to the per-row bootstrap (also tested)."""
    merged = per_row_a.merge(per_row_b, on=row_id_col, suffixes=("_a", "_b"), how="inner")
    n_rows = len(merged)
    if n_rows == 0:
        raise ValueError(f"no overlapping {row_id_col} between the two arms")
    if block_col not in merged.columns:
        raise ValueError(f"block_col {block_col!r} not found after merge -- pass it from exactly one "
                          f"arm (or identically from both) so it survives the merge unsuffixed")

    blocks = merged[block_col].to_numpy()
    unique_blocks = np.unique(blocks)
    n_blocks = len(unique_blocks)
    block_to_rowidx = {b: np.where(blocks == b)[0] for b in unique_blocks}

    rng = np.random.default_rng(seed)
    arrays_a = {spec["name"]: merged[f"{spec['col']}_a"].to_numpy() for spec in metrics}
    arrays_b = {spec["name"]: merged[f"{spec['col']}_b"].to_numpy() for spec in metrics}
    deltas = {spec["name"]: np.empty(n_bootstrap) for spec in metrics}

    for i in range(n_bootstrap):
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([block_to_rowidx[b] for b in sampled_blocks])
        for spec in metrics:
            name = spec["name"]
            deltas[name][i] = arrays_a[name][idx].mean() - arrays_b[name][idx].mean()

    results = {}
    print(f"=== block bootstrap ({block_col}-clustered): {label_a} vs {label_b} "
          f"(n={n_rows} rows, {n_blocks} blocks, {n_bootstrap} resamples) ===", flush=True)
    for spec in metrics:
        name, higher_is_better = spec["name"], spec["higher_is_better"]
        a_arr, b_arr = arrays_a[name], arrays_b[name]

        point = float(a_arr.mean() - b_arr.mean())
        ci = np.percentile(deltas[name], [2.5, 97.5])
        excludes_zero = ci[0] > 0 or ci[1] < 0
        if not excludes_zero:
            verdict = "NOISE -- CI includes zero"
        else:
            better = (point > 0) if higher_is_better else (point < 0)
            verdict = f"REAL {'IMPROVEMENT' if better else 'REGRESSION'} -- CI excludes zero"

        results[name] = {
            "a": float(a_arr.mean()), "b": float(b_arr.mean()), "delta": point,
            "delta_ci95": (float(ci[0]), float(ci[1])), "verdict": verdict,
        }
        print(f"{name:16s} {label_a}={a_arr.mean():.4f}  {label_b}={b_arr.mean():.4f}  "
              f"delta={point:+.4f}  95% CI=({ci[0]:+.4f}, {ci[1]:+.4f})  -> {verdict}", flush=True)

    results["n_rows"] = n_rows
    results["n_blocks"] = n_blocks
    return results
