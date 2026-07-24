"""Cycle 13's final, adopted pipeline: the same full chain as validate_rest.py
(situational team strength + team-specific PP/PK TOI + goalie overlay + rest
adjustment + real OT/SO resolution), with the scoring-era-drift fix baked
in -- league_avg_halflife_games=600, paired with team_strength_situational.py's
recalibrated (0.6x) shrinkage priors (see MODEL_DOCUMENTATION.md Sec7.5).

HALFLIFE_GAMES=600 was chosen from a joint grid search over halflife and
prior-scale (Sec7.5), then confirmed via paired bootstrap against the
Cycle 11 baseline (`rest_adjusted_poisson`, no decay): margin-MAE
improvement 95% CI entirely positive, Brier CI crosses zero (neutral,
not harmed), total-MAE CI borderline-positive. No metric regressed.
"""

from src.models.validate_rest import run_validation as _run_validation_chain

HALFLIFE_GAMES = 600


def run_validation(min_season: int = 20102011, max_season: int | None = None):
    from src.models.final_holdout_check import DEV_MAX_SEASON
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    return _run_validation_chain(min_season=min_season, max_season=max_season,
                                  league_avg_halflife_games=HALFLIFE_GAMES)


if __name__ == "__main__":
    from src.models.metrics_ledger import append_run

    r, adj = run_validation()
    print(f"validated {len(r)} dev-set games, halflife_games={HALFLIFE_GAMES}, "
          f"prior_minutes_scale=0.6, away_b2b_adjustment={adj:.5f}")
    ledger_input = r.rename(columns={"home_win_prob_full": "home_win_prob"}).assign(
        away_win_prob=lambda d: 1 - d["home_win_prob"], tie_prob=0.0)
    row = append_run(
        ledger_input, model_name="drift_adjusted_poisson",
        config_flags={"league_avg_halflife_games": HALFLIFE_GAMES, "prior_minutes_scale": 0.6,
                       "split": "development"},
        notes="Cycle 13 final: scoring-era-drift fix (decayed league-average baseline) + "
              "recalibrated (0.6x) situational shrinkage priors, on top of the full Cycle 11 pipeline. "
              "Real margin-MAE improvement, neutral Brier, positive-trending total-MAE -- no metric "
              "regressed vs rest_adjusted_poisson. See MODEL_DOCUMENTATION.md Sec7.5.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
