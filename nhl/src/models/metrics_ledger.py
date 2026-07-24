"""Persisted, append-only metrics ledger -- one row per validation run, so
later comparisons ("did the xG-based rating actually beat the naive
baseline?") can look up exactly what git state and config produced a number,
rather than trusting whatever's currently written in MODEL_DOCUMENTATION.md.
Mirrors the sibling MLB project's src/models/metrics_ledger.py structure.
"""

import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED

LEDGER_PATH = DATA_PROCESSED / "metrics_ledger.parquet"


def _git_state() -> tuple[str, bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
                               ).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return head, bool(status.strip())
    except Exception:
        return "unknown", True


def compute_run_metrics(r: pd.DataFrame) -> dict:
    """r: one row per real game, columns: actual_home, actual_away,
    lambda_home, lambda_away, home_win_prob, away_win_prob, tie_prob (all
    from the model's regulation-only Poisson joint -- see
    baseline_naive_poisson.home_win_prob_regulation). `home_win_prob` here is
    NOT yet adjusted for the Layer-4 OT/SO sub-model (not built yet), so it's
    compared against actual_home_win using a home_win_prob/(1-tie_prob)
    renormalization -- i.e. "conditional on being able to tell, did we call
    the winner right" -- rather than silently assuming a 50/50 OT split that
    would bias Brier/log-loss in an undocumented way."""
    r = r.copy()
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["actual_margin"] = r["actual_home"] - r["actual_away"]
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    r["pred_total_mean"] = r["lambda_home"] + r["lambda_away"]
    r["pred_margin_mean"] = r["lambda_home"] - r["lambda_away"]

    # Renormalize away the not-yet-modeled tie probability mass so this
    # metric isolates Layers 1-3 (team strength + count process), not a
    # guessed OT/SO coin flip that hasn't been built or validated yet.
    denom = (1 - r["tie_prob"]).replace(0, np.nan)
    home_win_prob_conditional = (r["home_win_prob"] / denom).clip(1e-6, 1 - 1e-6)

    r["su_primary"] = (home_win_prob_conditional > 0.5).astype(int) == r["actual_home_win"]
    r["brier"] = (home_win_prob_conditional - r["actual_home_win"]) ** 2
    r["log_loss"] = -(r["actual_home_win"] * np.log(home_win_prob_conditional)
                       + (1 - r["actual_home_win"]) * np.log(1 - home_win_prob_conditional))

    return {
        "n_games": len(r),
        "total_mae": float((r["pred_total_mean"] - r["actual_total"]).abs().mean()),
        "margin_mae": float((r["pred_margin_mean"] - r["actual_margin"]).abs().mean()),
        "su_primary": float(r["su_primary"].mean()),
        "brier": float(r["brier"].mean()),
        "log_loss": float(r["log_loss"].mean()),
        "mean_pred_home_win_prob_conditional": float(home_win_prob_conditional.mean()),
        "mean_tie_prob": float(r["tie_prob"].mean()),
        "actual_home_win_rate": float(r["actual_home_win"].mean()),
        "mean_actual_total": float(r["actual_total"].mean()),
        "mean_pred_total": float(r["pred_total_mean"].mean()),
    }


def append_run(r: pd.DataFrame, model_name: str, config_flags: dict, notes: str = "",
                extra_metrics: dict | None = None) -> dict:
    """`extra_metrics`: optional dict of additional scalar metrics to merge in
    (e.g. CRPS, Sec16) -- None (default) preserves every existing caller's
    exact behavior/schema."""
    git_hash, git_dirty = _git_state()
    metrics = compute_run_metrics(r)
    if extra_metrics:
        metrics = {**metrics, **extra_metrics}
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


if __name__ == "__main__":
    if not LEDGER_PATH.exists():
        print(f"no ledger yet at {LEDGER_PATH}")
    else:
        df = pd.read_parquet(LEDGER_PATH)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.to_string())
