"""Acceptance gate for `prop_distribution.py`: fits each family on a
chronological FIT split of the dev range, evaluates calibration
(log-score) on a held-out chronological EVAL split within dev -- the same
discipline `validate_score_distribution.py` uses for Normal-vs-Student-t.

Extended (2026-08-01) from the original 2-category smoke test (points,
blocks) to all 9 prop categories `generate_props.py` actually outputs
(2pt/3pt/ft made, oreb, dreb, ast, tov, stl, blk) -- a real gap: despite
`prop_distribution.py` being built and dev-validated for 2 categories back
in Sec11, it was NEVER actually wired into the live pipeline at all
(`generate_props.py` never imported it), so no prop category has ever
shipped a variance/family/over-under-probability, only a bare point
projection. This script is now the source of truth for the per-category
family choice `generate_props.py` reads (`prop_distribution.CATEGORY_FAMILY`),
each category earning its own check per this project's standing "never
assume a family transfers, even within the same continuous/count axis"
discipline (e.g. blocks and steals could plausibly have different
overdispersion despite both being count/exposure=minutes categories).

Continuous-vs-count AXIS ASSIGNMENT itself is NOT re-derived here (kept as
the plan's original domain-reasoning split: points/2PT/3PT/FT-makes
aggregate over many shot attempts per game, genuinely continuous-like by
CLT; OREB/DREB/AST/TOV/STL/BLK are low-count discrete events) -- only the
SPECIFIC family WITHIN that already-assigned axis (Normal vs t; Poisson vs
NegBin) is decided empirically per category, exactly mirroring how points
(Normal vs t) and blocks (Poisson vs NegBin) were originally handled.

Run as `python -m src.models.validate_prop_distribution`.
"""

import numpy as np
import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON
from src.models import metrics_ledger
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_minutes import attach_player_track, build_player_game_log
from src.models.player_playmaking_rates import add_playmaking_rates
from src.models.player_rebounding_rates import add_rebounding_rates
from src.models.player_scoring_rates import add_scoring_rates
from src.models.prop_distribution import fit_continuous_family, fit_count_family, log_score, over_under_prob, predict_variance

FIT_FRACTION = 0.7  # chronological -- matches validate_score_distribution.py's own convention
CALIBRATION_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)

# name -> (actual_col, proj_col, exposure_col, family_type). `exposure_col` on a "count" row is
# used only as a data-availability filter (dropna), not by the count-family fit itself -- see
# `prop_distribution.CATEGORY_FAMILY`'s docstring for why points/2PT/3PT/FT are "count" here
# despite exposure_col still being their own rate model's exposure unit.
CATEGORY_SPECS = {
    "points": (None, None, "minutes", "count"),  # built specially: points = sum of 3 makes
    "2pt_made": ("fgMade2", "2pt_proj_made", "fgAtt2", "count"),
    "3pt_made": ("fg3Made", "3pt_proj_made", "fg3Att", "count"),
    "ft_made": ("ftMade", "ft_proj_made", "ftAtt", "count"),
    "oreb": ("oreb", "oreb_proj", "reboundChancesOffensive", "count"),
    "dreb": ("dreb", "dreb_proj", "reboundChancesDefensive", "count"),
    "ast": ("assists", "ast_proj", "touches", "count"),
    "tov": ("turnovers", "tov_proj", "touches", "count"),
    "stl": ("steals", "stl_proj", "minutes", "count"),
    "blk": ("blocks", "blk_proj", "minutes", "count"),
}


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    cut = int(len(df) * FIT_FRACTION)
    return df.iloc[:cut], df.iloc[cut:]


def _calibration_error(actual: np.ndarray, proj: np.ndarray, family: str, family_type: str,
                        params_per_row: list) -> dict:
    """The check this module never actually did until now: does the
    CHOSEN family's `over_under_prob` produce trustworthy probabilities,
    not just win the log-score comparison against its rejected
    alternative? Computes the probability integral transform (PIT) --
    `F(actual)` under the fitted distribution -- for every eval row; a
    well-calibrated model has PIT values distributed ~Uniform(0,1), so
    "what fraction of rows have PIT <= q" should be close to q for every
    q. Reuses `over_under_prob` directly (`F(k) = 1 - over_under_prob(k,
    ...)`) rather than re-deriving each family's CDF from scratch.

    Discrete (count) families use the RANDOMIZED PIT (`F(k-1) + U*(F(k) -
    F(k-1))`, `U~Uniform(0,1)`) -- the standard fix for the fact that a
    discrete CDF's PIT is not exactly uniform even for a perfectly correct
    model (it would otherwise show spurious clumping at each integer
    jump)."""
    rng = np.random.default_rng(42)
    pits = []
    for a, m, params in zip(actual, proj, params_per_row):
        f_k = 1.0 - over_under_prob(a, m, family, params)
        if family_type == "count":
            f_km1 = 1.0 - over_under_prob(a - 1, m, family, params)
            pits.append(f_km1 + rng.uniform(0, 1) * (f_k - f_km1))
        else:
            pits.append(f_k)
    pits = np.array(pits)
    detail, max_dev = {}, 0.0
    for q in CALIBRATION_QUANTILES:
        empirical = float((pits <= q).mean())
        dev = abs(empirical - q)
        detail[q] = empirical
        max_dev = max(max_dev, dev)
    return {"detail": detail, "max_calibration_error": max_dev}


def _fit_and_score_continuous(fit_df: pd.DataFrame, eval_df: pd.DataFrame,
                               actual_col: str, proj_col: str, exposure_col: str) -> dict:
    fit_resid = (fit_df[actual_col] - fit_df[proj_col]).to_numpy()
    fitted = fit_continuous_family(fit_resid, fit_df[exposure_col].to_numpy())
    variance_model, df_t = fitted["variance_model"], fitted["df"]

    eval_var = predict_variance(variance_model, eval_df[exposure_col].to_numpy())
    ll_normal = np.mean([log_score(a, m, "normal", {"variance": v})
                         for a, m, v in zip(eval_df[actual_col], eval_df[proj_col], eval_var)])
    ll_t = np.mean([log_score(a, m, "t", {"variance": v, "df": df_t})
                    for a, m, v in zip(eval_df[actual_col], eval_df[proj_col], eval_var)])
    chosen = "t" if ll_t < ll_normal else "normal"

    chosen_params = [({"variance": v, "df": df_t} if chosen == "t" else {"variance": v}) for v in eval_var]
    calibration = _calibration_error(eval_df[actual_col].to_numpy(), eval_df[proj_col].to_numpy(),
                                      chosen, "continuous", chosen_params)
    return {"family_type": "continuous", "variance_model": variance_model, "df": df_t,
            "ll_normal": float(ll_normal), "ll_t": float(ll_t), "chosen": chosen, "calibration": calibration}


def _fit_and_score_count(fit_df: pd.DataFrame, eval_df: pd.DataFrame, actual_col: str, proj_col: str) -> dict:
    fitted = fit_count_family(fit_df[actual_col].to_numpy(), fit_df[proj_col].to_numpy())
    ll_poisson = np.mean([log_score(a, m, "poisson", {})
                          for a, m in zip(eval_df[actual_col], eval_df[proj_col])])
    if fitted["family"] == "negbin":
        ll_negbin = np.mean([log_score(a, m, "negbin", fitted)
                             for a, m in zip(eval_df[actual_col], eval_df[proj_col])])
        chosen = "negbin" if ll_negbin < ll_poisson else "poisson"
    else:
        ll_negbin = None
        chosen = "poisson"

    chosen_params = {"r": fitted["r"]} if chosen == "negbin" else {}
    calibration = _calibration_error(eval_df[actual_col].to_numpy(), eval_df[proj_col].to_numpy(),
                                      chosen, "count", [chosen_params] * len(eval_df))
    return {"family_type": "count", "r": fitted.get("r"),
            "ll_poisson": float(ll_poisson), "ll_negbin": (float(ll_negbin) if ll_negbin is not None else None),
            "chosen": chosen, "calibration": calibration}


def run_category(name: str, log: pd.DataFrame) -> dict:
    actual_col, proj_col, exposure_col, family_type = CATEGORY_SPECS[name]
    log = log.dropna(subset=[actual_col, proj_col, exposure_col]).copy()
    fit_df, eval_df = chronological_split(log)
    print(f"\n=== {name} (family_type={family_type}, fit={len(fit_df)} eval={len(eval_df)} player-games) ===", flush=True)

    if family_type == "continuous":
        result = _fit_and_score_continuous(fit_df, eval_df, actual_col, proj_col, exposure_col)
        print(f"  Student-t df={result['df']:.1f}  log-score: normal={result['ll_normal']:.4f}  "
              f"t={result['ll_t']:.4f}  -> chosen: {result['chosen']}", flush=True)
        print(f"  calibration ({result['chosen']}): {result['calibration']['detail']}  "
              f"max deviation from nominal: {result['calibration']['max_calibration_error']:.4f}", flush=True)
        metrics_ledger.append_generic_run(
            metrics={"n_eval": len(eval_df), f"{name}_logscore_normal": result["ll_normal"],
                     f"{name}_logscore_t": result["ll_t"], f"{name}_student_t_df": result["df"],
                     f"{name}_family_adopted": result["chosen"],
                     f"{name}_max_calibration_error": result["calibration"]["max_calibration_error"]},
            model_name=f"prop_distribution_{name}", config_flags={"dev_max_season": DEV_MAX_SEASON},
            notes=f"{name} continuous-family fit (Normal vs Student-t), log-score + PIT calibration on chronological eval split",
        )
    else:
        result = _fit_and_score_count(fit_df, eval_df, actual_col, proj_col)
        print(f"  overdispersion r={result['r']}  log-score: poisson={result['ll_poisson']:.4f}"
              + (f"  negbin={result['ll_negbin']:.4f}" if result["ll_negbin"] is not None else "  (NB not triggered)")
              + f"  -> chosen: {result['chosen']}", flush=True)
        print(f"  calibration ({result['chosen']}): {result['calibration']['detail']}  "
              f"max deviation from nominal: {result['calibration']['max_calibration_error']:.4f}", flush=True)
        metrics_ledger.append_generic_run(
            metrics={"n_eval": len(eval_df), f"{name}_logscore_poisson": result["ll_poisson"],
                     f"{name}_logscore_negbin": result["ll_negbin"], f"{name}_family_adopted": result["chosen"],
                     f"{name}_nb_dispersion_r": result["r"],
                     f"{name}_max_calibration_error": result["calibration"]["max_calibration_error"]},
            model_name=f"prop_distribution_{name}", config_flags={"dev_max_season": DEV_MAX_SEASON},
            notes=f"{name} count-family fit (Poisson vs Negative-Binomial via overdispersion check), "
                  f"log-score + randomized-PIT calibration on chronological eval split",
        )
    return result


def _build_logs() -> dict:
    """One shared base log, one merge per data source needed -- mirrors
    `generate_props.py._build_rate_logs`'s own reuse discipline."""
    base_log = build_player_game_log(FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    base_log = base_log[base_log["minutes"] > 0].copy()

    scoring_log = add_scoring_rates(base_log.copy())
    scoring_log["points_proj"] = (2 * scoring_log["2pt_proj_made"] + 3 * scoring_log["3pt_proj_made"]
                                   + scoring_log["ft_proj_made"])

    defensive_log = add_defensive_event_rates(base_log.copy())

    tracked = attach_player_track(base_log.copy(), FIRST_DEV_SEASON, DEV_MAX_SEASON - 1)
    rebounding_log = add_rebounding_rates(tracked.dropna(subset=["reboundChancesOffensive", "reboundChancesDefensive"]).copy())
    playmaking_log = add_playmaking_rates(tracked.dropna(subset=["touches"]).copy())

    return {
        "points": scoring_log, "2pt_made": scoring_log, "3pt_made": scoring_log, "ft_made": scoring_log,
        "oreb": rebounding_log, "dreb": rebounding_log, "ast": playmaking_log, "tov": playmaking_log,
        "stl": defensive_log, "blk": defensive_log,
    }


CATEGORY_SPECS["points"] = ("points", "points_proj", "minutes", "count")

if __name__ == "__main__":
    logs = _build_logs()
    results = {name: run_category(name, logs[name]) for name in CATEGORY_SPECS}

    print("\n=== SUMMARY: chosen family per category ===", flush=True)
    for name, result in results.items():
        print(f"  {name:10s} -> {result['chosen']}", flush=True)
    print("\nall categories appended to metrics_ledger.py", flush=True)
