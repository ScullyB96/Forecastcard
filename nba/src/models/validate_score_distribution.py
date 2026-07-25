"""Phase 3's acceptance gate: fits the residual-variance-vs-pace model and
home/away correlation on a chronological FIT split of the dev range, then
evaluates calibration (interval coverage) and log-score/CRPS on a held-out
chronological EVAL split within dev (a lightweight precaution against
overfitting the variance model to the same games used to judge it --
distinct from the Phase 4 dev/holdout split, which is a separate, larger
concern). Compares Normal vs. Student-t; adopts whichever calibrates
better via `bootstrap_significance`, against a naive flat-variance
baseline.

Run as `python -m src.models.validate_score_distribution`.
"""

import numpy as np
import pandas as pd

from src.models import metrics_ledger
from src.models.bootstrap_significance import bootstrap_compare
from src.models.score_distribution import (
    crps_normal, empirical_coverage, fit_home_away_correlation, fit_residual_variance_model,
    fit_student_t_df, log_score, predict_variance,
)
from src.models.validate_team_strength_baseline import build_dev_predictions

FIT_FRACTION = 0.7  # chronological -- first 70% of dev games fit the variance/correlation params


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    cut = int(len(df) * FIT_FRACTION)
    return df.iloc[:cut], df.iloc[cut:]


def run() -> None:
    df = build_dev_predictions().dropna(subset=["pred_home", "pred_away"])
    df["home_resid"] = df["actual_home"] - df["pred_home"]
    df["away_resid"] = df["actual_away"] - df["pred_away"]

    fit_df, eval_df = chronological_split(df)
    print(f"fit set: {len(fit_df)} games, eval set: {len(eval_df)} games", flush=True)

    home_var_model = fit_residual_variance_model(fit_df["home_resid"].to_numpy(), fit_df["projected_pace"].to_numpy())
    away_var_model = fit_residual_variance_model(fit_df["away_resid"].to_numpy(), fit_df["projected_pace"].to_numpy())
    corr = fit_home_away_correlation(fit_df["home_resid"].to_numpy(), fit_df["away_resid"].to_numpy())
    print(f"home variance model: {home_var_model}", flush=True)
    print(f"away variance model: {away_var_model}", flush=True)
    print(f"home/away residual correlation: {corr:+.4f}", flush=True)

    eval_df = eval_df.copy()
    eval_df["var_home"] = predict_variance(home_var_model, eval_df["projected_pace"].to_numpy())
    eval_df["var_away"] = predict_variance(away_var_model, eval_df["projected_pace"].to_numpy())

    home_z = (fit_df["home_resid"] / np.sqrt(predict_variance(home_var_model, fit_df["projected_pace"].to_numpy()))).to_numpy()
    away_z = (fit_df["away_resid"] / np.sqrt(predict_variance(away_var_model, fit_df["projected_pace"].to_numpy()))).to_numpy()
    df_t_home = fit_student_t_df(home_z)
    df_t_away = fit_student_t_df(away_z)
    print(f"Student-t df (home/away): {df_t_home:.1f} / {df_t_away:.1f} "
          f"(>=~200 means effectively Normal, no real excess kurtosis found)", flush=True)

    naive_var_home = float(fit_df["home_resid"].var())
    naive_var_away = float(fit_df["away_resid"].var())

    print("\n=== calibration: nominal vs. empirical interval coverage (eval set) ===", flush=True)
    for coverage in (0.5, 0.8, 0.95):
        for family, df_h, df_a in (("normal", 200.0, 200.0), ("t", df_t_home, df_t_away)):
            cov_home = empirical_coverage(eval_df["actual_home"].to_numpy(), eval_df["pred_home"].to_numpy(),
                                           eval_df["var_home"].to_numpy(), coverage, family, df_h)
            cov_away = empirical_coverage(eval_df["actual_away"].to_numpy(), eval_df["pred_away"].to_numpy(),
                                           eval_df["var_away"].to_numpy(), coverage, family, df_a)
            print(f"  nominal {coverage:.0%} [{family:6s}]: home={cov_home:.3f}  away={cov_away:.3f}", flush=True)

    def per_game_scores(family: str) -> pd.DataFrame:
        d_h = df_t_home if family == "t" else 200.0
        d_a = df_t_away if family == "t" else 200.0
        rows = []
        for row in eval_df.itertuples(index=False):
            rows.append({
                "gameId": row.gameId,
                "log_score": log_score(row.actual_home, row.pred_home, row.var_home, family, d_h)
                             + log_score(row.actual_away, row.pred_away, row.var_away, family, d_a),
                "crps": crps_normal(row.actual_home, row.pred_home, row.var_home)
                        + crps_normal(row.actual_away, row.pred_away, row.var_away) if family == "normal" else np.nan,
            })
        return pd.DataFrame(rows)

    def naive_scores() -> pd.DataFrame:
        rows = []
        for row in eval_df.itertuples(index=False):
            rows.append({
                "gameId": row.gameId,
                "log_score": log_score(row.actual_home, row.pred_home, naive_var_home, "normal", 200.0)
                             + log_score(row.actual_away, row.pred_away, naive_var_away, "normal", 200.0),
                "crps": crps_normal(row.actual_home, row.pred_home, naive_var_home)
                        + crps_normal(row.actual_away, row.pred_away, naive_var_away),
            })
        return pd.DataFrame(rows)

    normal_scores = per_game_scores("normal")
    t_scores = per_game_scores("t")
    naive = naive_scores()

    print("\n=== Normal (pace-scaled variance) vs naive (flat variance): paired bootstrap ===", flush=True)
    bootstrap_compare(
        normal_scores, naive, game_id_col="gameId",
        metrics=[
            {"name": "log_score", "col": "log_score", "higher_is_better": False},
            {"name": "crps", "col": "crps", "higher_is_better": False},
        ],
        label_a="normal_pace_scaled", label_b="naive_flat_variance",
    )

    print("\n=== Student-t vs Normal: paired bootstrap (log-score only, CRPS has no simple t closed form) ===", flush=True)
    bootstrap_compare(
        t_scores[["gameId", "log_score"]], normal_scores[["gameId", "log_score"]], game_id_col="gameId",
        metrics=[{"name": "log_score", "col": "log_score", "higher_is_better": False}],
        label_a="student_t", label_b="normal",
    )

    metrics_ledger.append_run(
        eval_df[["gameId", "season", "actual_home", "actual_away", "pred_home", "pred_away"]],
        model_name="phase3_score_distribution",
        config_flags={"fit_fraction": FIT_FRACTION, "home_var_model": home_var_model,
                      "away_var_model": away_var_model, "corr": corr,
                      "df_t_home": df_t_home, "df_t_away": df_t_away},
        notes="Phase 3 score-distribution calibration run (point-prediction metrics only; "
              "see printed log-score/CRPS/coverage above for the distributional comparison)",
    )


if __name__ == "__main__":
    run()
