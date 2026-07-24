"""Cycle 19's final, adopted pipeline (supersedes Cycle 18's weight=0.5,
multiplier=1.0): the same full chain as validate_drift.py, with the
cross-season team-strength prior (Sec12) AND the re-tuned prior-strength
multiplier (Sec13/Sec15) baked in -- CROSS_SEASON_WEIGHT=0.75,
PRIOR_MINUTES_MULTIPLIER=2.0.

Sec13 originally REJECTED this exact configuration based on a raw holdout
point-estimate comparison with no confidence interval. Sec15 (a follow-up
review) caught that the dev-set bar (5,000-resample paired bootstrap) and
the holdout bar (unbootstrapped point estimates) were inconsistent, and
that on 2,624 holdout games several of the "decisive" differences were on
the order of 1-2 games. Re-ran the holdout comparison with the SAME
bootstrap standard: weight=0.75/mult=2.0 shows NO real (bootstrap-confirmed)
regression on holdout on any metric, while its dev-set improvement (Brier,
total-MAE, margin-MAE all real, zero dev regressions) stands. Re-adjudicated
as ADOPTED under the corrected protocol (confirmatory-veto-only holdout
role) -- see MODEL_DOCUMENTATION.md Sec15.
"""

from src.models.validate_rest import run_validation as _run_validation_chain
from src.models.validate_drift import HALFLIFE_GAMES

CROSS_SEASON_WEIGHT = 0.75
PRIOR_MINUTES_MULTIPLIER = 2.0


def run_validation(min_season: int = 20102011, max_season: int | None = None):
    from src.models.final_holdout_check import DEV_MAX_SEASON
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    return _run_validation_chain(min_season=min_season, max_season=max_season,
                                  league_avg_halflife_games=HALFLIFE_GAMES,
                                  cross_season_weight=CROSS_SEASON_WEIGHT,
                                  prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER)


if __name__ == "__main__":
    from src.models.metrics_ledger import append_run

    r, adj = run_validation()
    print(f"validated {len(r)} dev-set games, halflife_games={HALFLIFE_GAMES}, "
          f"cross_season_weight={CROSS_SEASON_WEIGHT}, prior_minutes_multiplier={PRIOR_MINUTES_MULTIPLIER}, "
          f"away_b2b_adjustment={adj:.5f}")
    ledger_input = r.rename(columns={"home_win_prob_full": "home_win_prob"}).assign(
        away_win_prob=lambda d: 1 - d["home_win_prob"], tie_prob=0.0)
    row = append_run(
        ledger_input, model_name="cross_season_prior_poisson",
        config_flags={"league_avg_halflife_games": HALFLIFE_GAMES, "cross_season_weight": CROSS_SEASON_WEIGHT,
                       "prior_minutes_multiplier": PRIOR_MINUTES_MULTIPLIER, "split": "development"},
        notes="Cycle 19 final (re-adjudicated, Sec15): cross-season team-strength prior (weight=0.75) "
              "plus a re-tuned prior-strength multiplier (2.0x on top of Cycle 13's 0.6x) on top of the full "
              "Cycle 13 drift-adjusted pipeline. Real Brier/total-MAE/margin-MAE improvement on the dev set, "
              "no real regression on either dev or the properly-bootstrapped holdout. "
              "See MODEL_DOCUMENTATION.md Sec13/Sec15.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
