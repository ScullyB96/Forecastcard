"""Persisted, append-only metrics ledger -- one row per validation run, so
later comparisons ("did the RAPM-lite lineup layer actually beat the
team-strength-only baseline?") can look up exactly what git state and config
produced a number. Reimplemented independently for this project (no import
from the sibling nhl/mlb projects) but mirroring their exact schema/rationale.

Straight-up (SU) accuracy here is based on point predictions
(pred_margin_mean > 0) since Phase 1/2 don't yet produce a calibrated win
probability -- that's Phase 3's job (score_distribution.py). Brier/log-loss
are therefore left out of `compute_run_metrics` and only appear once a real
win-probability column is passed in via `extra_metrics`, so this ledger
never reports an invented, uncalibrated probability as if it were real.
"""

import json
import subprocess
from datetime import datetime

import pandas as pd

from src.utils.paths import DATA_PROCESSED

LEDGER_PATH = DATA_PROCESSED / "metrics_ledger.parquet"


def _git_state() -> tuple[str, bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return head, bool(status.strip())
    except Exception:
        return "unknown", True


def compute_run_metrics(r: pd.DataFrame) -> dict:
    """r: one row per real game, columns: actual_home, actual_away,
    pred_home, pred_away (point predictions from `team_strength.project_game`
    or a later phase's projection)."""
    r = r.copy()
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["actual_margin"] = r["actual_home"] - r["actual_away"]
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    r["pred_total"] = r["pred_home"] + r["pred_away"]
    r["pred_margin"] = r["pred_home"] - r["pred_away"]
    r["pred_home_win"] = (r["pred_margin"] > 0).astype(int)

    return {
        "n_games": len(r),
        "total_mae": float((r["pred_total"] - r["actual_total"]).abs().mean()),
        "margin_mae": float((r["pred_margin"] - r["actual_margin"]).abs().mean()),
        "home_mae": float((r["pred_home"] - r["actual_home"]).abs().mean()),
        "away_mae": float((r["pred_away"] - r["actual_away"]).abs().mean()),
        "su_primary": float((r["pred_home_win"] == r["actual_home_win"]).mean()),
        "actual_home_win_rate": float(r["actual_home_win"].mean()),
        "mean_actual_total": float(r["actual_total"].mean()),
        "mean_pred_total": float(r["pred_total"].mean()),
    }


def append_generic_run(metrics: dict, model_name: str, config_flags: dict, notes: str = "") -> dict:
    """The shared timestamp/git-hash/parquet-append plumbing, extracted
    from `append_run`'s tail so a caller with a metrics dict that ISN'T
    home/away-score-shaped (e.g. a per-player-stat props model, whose
    natural metric is something like `minutes_mae`, not `total_mae`/
    `su_primary`) can share this same ledger file without forcing its
    result through `compute_run_metrics`'s score-specific contract (which
    stays unmodified -- home_mae/away_mae/su_primary are used elsewhere and
    must keep meaning exactly what they already mean). Appending a row with
    a different metrics schema than existing rows is fine: parquet/pandas
    concat unions the columns, filling NaN for whichever columns don't
    apply to a given row -- an honest reflection of "this ledger holds
    heterogeneous validation runs," not a bug."""
    git_hash, git_dirty = _git_state()
    row = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "config_flags": json.dumps(config_flags, sort_keys=True),
        "notes": notes,
        **metrics,
    }
    existing = pd.read_parquet(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_parquet(LEDGER_PATH, index=False)
    return row


def append_run(r: pd.DataFrame, model_name: str, config_flags: dict, notes: str = "",
                extra_metrics: dict | None = None) -> dict:
    metrics = compute_run_metrics(r)
    if extra_metrics:
        metrics = {**metrics, **extra_metrics}
    return append_generic_run(metrics, model_name, config_flags, notes)


if __name__ == "__main__":
    if not LEDGER_PATH.exists():
        print(f"no ledger yet at {LEDGER_PATH}", flush=True)
    else:
        df = pd.read_parquet(LEDGER_PATH)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.to_string(), flush=True)
