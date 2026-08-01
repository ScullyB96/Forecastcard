"""Acceptance gate for `prop_distribution.py`: fits each family on a
chronological FIT split of the dev range, evaluates calibration
(log-score) on a held-out chronological EVAL split within dev -- the same
discipline `validate_score_distribution.py` uses for Normal-vs-Student-t,
applied here to two representative categories:

- POINTS (high-count/continuous): Normal vs. Student-t, exposure = the
  player's own realized minutes (a backtest convenience, same convention
  every `player_*_rates.py` module already uses -- a live caller would use
  PROJECTED minutes instead).
- BLOCKS (low-count/discrete): Poisson vs. Negative-Binomial, via the
  overdispersion check.

Run as `python -m src.models.validate_prop_distribution`.
"""

import numpy as np
import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_minutes import build_player_game_log
from src.models.player_scoring_rates import add_scoring_rates
from src.models.prop_distribution import fit_continuous_family, fit_count_family, log_score, over_under_prob

FIT_FRACTION = 0.7  # chronological -- matches validate_score_distribution.py's own convention


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    cut = int(len(df) * FIT_FRACTION)
    return df.iloc[:cut], df.iloc[cut:]


def run_points() -> None:
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["minutes"] > 0].copy()
    rated = add_scoring_rates(log)
    rated["points_proj"] = (2 * rated["2pt_proj_made"] + 3 * rated["3pt_proj_made"] + rated["ft_proj_made"])
    rated["points_actual"] = rated["points"]
    rated = rated.dropna(subset=["points_proj", "points_actual"])
    rated["resid"] = rated["points_actual"] - rated["points_proj"]

    fit_df, eval_df = chronological_split(rated)
    print(f"points: fit={len(fit_df)} eval={len(eval_df)} player-games", flush=True)

    fitted = fit_continuous_family(fit_df["resid"].to_numpy(), fit_df["minutes"].to_numpy())
    variance_model, df_t = fitted["variance_model"], fitted["df"]
    print(f"points variance model: {variance_model}, Student-t df: {df_t:.1f} "
          f"(>=~200 means effectively Normal)", flush=True)

    from src.models.score_distribution import predict_variance
    eval_df = eval_df.copy()
    eval_df["variance"] = predict_variance(variance_model, eval_df["minutes"].to_numpy())

    ll_normal = np.mean([log_score(a, m, "normal", {"variance": v})
                         for a, m, v in zip(eval_df["points_actual"], eval_df["points_proj"], eval_df["variance"])])
    ll_t = np.mean([log_score(a, m, "t", {"variance": v, "df": df_t})
                    for a, m, v in zip(eval_df["points_actual"], eval_df["points_proj"], eval_df["variance"])])
    print(f"points mean log-score (lower better): normal={ll_normal:.4f}  t={ll_t:.4f}", flush=True)

    metrics_ledger.append_generic_run(
        metrics={"n_eval": len(eval_df), "points_logscore_normal": float(ll_normal),
                 "points_logscore_t": float(ll_t), "points_student_t_df": df_t,
                 "points_family_adopted": "t" if ll_t < ll_normal else "normal"},
        model_name="prop_distribution_points", config_flags={"dev_max_season": DEV_MAX_SEASON},
        notes="Points continuous-family fit (Normal vs Student-t), log-score on chronological eval split",
    )


def run_blocks() -> None:
    log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    log = log[log["minutes"] > 0].copy()
    rated = add_defensive_event_rates(log)
    rated = rated.dropna(subset=["blk_proj"])

    fit_df, eval_df = chronological_split(rated)
    print(f"blocks: fit={len(fit_df)} eval={len(eval_df)} player-games", flush=True)

    fitted = fit_count_family(fit_df["blocks"].to_numpy(), fit_df["blk_proj"].to_numpy())
    print(f"blocks family fit (overdispersion check): {fitted}", flush=True)

    # always compute log-score for BOTH families on the eval set, regardless of which one the
    # overdispersion check picked -- an honest side-by-side, not just "whichever got picked".
    ll_poisson = np.mean([log_score(a, m, "poisson", {})
                          for a, m in zip(eval_df["blocks"], eval_df["blk_proj"])])
    if fitted["family"] == "negbin":
        ll_negbin = np.mean([log_score(a, m, "negbin", fitted)
                             for a, m in zip(eval_df["blocks"], eval_df["blk_proj"])])
    else:
        ll_negbin = None
    print(f"blocks mean log-score (lower better): poisson={ll_poisson:.4f}"
          + (f"  negbin={ll_negbin:.4f}" if ll_negbin is not None else "  (NB not triggered -- no overdispersion detected)"), flush=True)

    metrics_ledger.append_generic_run(
        metrics={"n_eval": len(eval_df), "blocks_logscore_poisson": float(ll_poisson),
                 "blocks_logscore_negbin": float(ll_negbin) if ll_negbin is not None else None,
                 "blocks_family_adopted": fitted["family"],
                 "blocks_nb_dispersion_r": fitted.get("r")},
        model_name="prop_distribution_blocks", config_flags={"dev_max_season": DEV_MAX_SEASON},
        notes="Blocks count-family fit (Poisson vs Negative-Binomial via overdispersion check), log-score on chronological eval split",
    )


if __name__ == "__main__":
    run_points()
    print()
    run_blocks()
    print("\nboth categories appended to metrics_ledger.py", flush=True)
