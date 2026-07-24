"""Persisted, append-only metrics ledger (Phase 0.1, 2026-07-22 roadmap).

This project's own history includes a real reference-point mistake this
session (MODEL_DOCUMENTATION.md §11.9: a bat-speed/pulled-air-rate re-test
was compared against a stale baseline because no durable record existed of
exactly which codebase state a saved *_validation.parquet came from) and a
long-standing habit of citing SU/Brier/MAE figures inline in this
documentation that no file actually stores authoritatively. This module
fixes both: every validation run appends ONE row here (game_pk-level output
stays in its own *_validation.parquet as before -- this ledger is the
per-RUN summary, not per-game detail), so any future comparison can look up
exactly what config/git-state a number came from instead of trusting
whatever's currently written in a markdown file.
"""

import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED

LEDGER_PATH = DATA_PROCESSED / "metrics_ledger.parquet"


def _git_state() -> tuple[str, bool]:
    """(head_hash, is_dirty). Returns ("unknown", True) if git isn't
    available for some reason rather than raising -- a missing git hash
    shouldn't block logging a real validation result."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
                               ).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True
                                 ).stdout
        return head, bool(status.strip())
    except Exception:
        return "unknown", True


def compute_run_metrics(r: pd.DataFrame) -> dict:
    """Given a validate_game_simulator.py-style per-game result DataFrame
    (columns: game_pk, actual_home, actual_away, sim_home_mean, sim_away_mean,
    sim_home_std, sim_away_std, sim_home_win_prob), compute every standard
    summary statistic this project reports -- the single source of truth for
    "what does a run's summary actually contain," so validate_game_simulator.py's
    printed output and the ledger row can never silently drift apart."""
    r = r.copy()
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["sim_total_mean"] = r["sim_home_mean"] + r["sim_away_mean"]
    r["actual_margin"] = r["actual_home"] - r["actual_away"]
    r["sim_margin_mean"] = r["sim_home_mean"] - r["sim_away_mean"]
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    r["su_primary"] = (r["sim_home_win_prob"] > 0.5).astype(int) == r["actual_home_win"]
    r["su_legacy"] = np.sign(r["sim_margin_mean"]) == np.sign(r["actual_margin"])
    r["brier"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
    p_clipped = r["sim_home_win_prob"].clip(1e-6, 1 - 1e-6)
    r["log_loss"] = -(r["actual_home_win"] * np.log(p_clipped) + (1 - r["actual_home_win"]) * np.log(1 - p_clipped))
    combined_std = np.sqrt(r["sim_home_std"] ** 2 + r["sim_away_std"] ** 2)
    z = (r["actual_total"] - r["sim_total_mean"]) / combined_std.replace(0, np.nan)

    return {
        "n_games": len(r),
        "home_score_mae": float((r["sim_home_mean"] - r["actual_home"]).abs().mean()),
        "away_score_mae": float((r["sim_away_mean"] - r["actual_away"]).abs().mean()),
        "total_mae": float((r["sim_total_mean"] - r["actual_total"]).abs().mean()),
        "margin_mae": float((r["sim_margin_mean"] - r["actual_margin"]).abs().mean()),
        "su_primary": float(r["su_primary"].mean()),
        "su_legacy": float(r["su_legacy"].mean()),
        "brier": float(r["brier"].mean()),
        "log_loss": float(r["log_loss"].mean()),
        "mean_sim_home_win_prob": float(r["sim_home_win_prob"].mean()),
        "actual_home_win_rate": float(r["actual_home_win"].mean()),
        "mean_actual_total": float(r["actual_total"].mean()),
        "mean_sim_total": float(r["sim_total_mean"].mean()),
        "std_z": float(z.std()),
        "mean_z": float(z.mean()),
        "frac_z_gt_196": float((z.abs() > 1.96).mean()),
        "frac_z_gt_258": float((z.abs() > 2.58).mean()),
        "frac_13plus_actual": float((r["actual_total"] >= 13).mean()),
    }


def append_run(r: pd.DataFrame, config_flags: dict, n_trials: int, notes: str = "") -> dict:
    """Compute the standard metrics for `r` (see compute_run_metrics),
    append one row to the ledger (creating it if missing), and return the
    computed row. config_flags: e.g. {"hfa": True, "park_neutral": True,
    "gbfb_pitcher": True, "gbfb_fullmix": False, "rookie_prior": False,
    "posterior_sample": False, "crn_enabled": False, "test_seasons": "2023,2024,2025"}
    -- a plain dict of whatever factor toggles were in effect for this run,
    stored as a JSON-ish string column so the ledger stays one flat table."""
    import json

    git_hash, git_dirty = _git_state()
    metrics = compute_run_metrics(r)
    row = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "config_flags": json.dumps(config_flags, sort_keys=True),
        "n_trials": n_trials,
        "notes": notes,
        **metrics,
    }
    existing = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_parquet(LEDGER_PATH, index=False)
    return row


if __name__ == "__main__":
    if not LEDGER_PATH.exists():
        print(f"no ledger yet at {LEDGER_PATH}")
    else:
        df = pd.read_parquet(LEDGER_PATH)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.to_string())
