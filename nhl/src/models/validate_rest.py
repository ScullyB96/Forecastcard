"""Validates the away-back-to-back rest adjustment (Cycle 11) on top of the
current best pipeline (situational team strength + goalie overlay + the
real OT/SO resolution from Cycle 9), on the development set only.

Real, bootstrapped result (2026-07-23, MODEL_DOCUMENTATION.md Sec4.14):
Brier improvement mean 0.00015, 95% CI [0.00001, 0.00029] (entirely
positive, though barely); margin-MAE improvement mean 0.00257, 95% CI
[0.00171, 0.00344] (clearly significant); total-MAE improvement crosses
zero (not distinguishable from noise). Kept -- the same pattern as the
goalie overlay (Sec4.6): a real margin-specific effect, borderline on win
probability, no effect on totals.
"""

import pandas as pd

from src.models.baseline_naive_poisson import home_win_prob_regulation, score_distribution
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.metrics_ledger import append_run
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate, full_win_probability
from src.models.rest_schedule import add_rest_days, add_walk_forward_b2b_incidence, fit_away_b2b_adjustment, \
    symmetric_b2b_bias_credit
from src.models.validate_goalie import run_validation as run_current_best
from src.utils.paths import DATA_RAW


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON,
                    league_avg_halflife_games: float | None = None,
                    cross_season_weight: float | None = None,
                    prior_minutes_multiplier: float = 1.0,
                    use_sh_term: bool = False):
    """`league_avg_halflife_games`, `cross_season_weight`, `prior_minutes_multiplier`,
    `use_sh_term`: passed through to the situational layer; defaults
    preserve this cycle's original behavior exactly."""
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule_reg_only = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule_reg_only)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=max_season)

    base = run_current_best(league_avg_halflife_games=league_avg_halflife_games,
                             cross_season_weight=cross_season_weight,
                             prior_minutes_multiplier=prior_minutes_multiplier,
                             use_sh_term=use_sh_term)
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()

    away_b2b_adj = fit_away_b2b_adjustment(schedule, base["actual_away"], base["lambda_away"], base["gameId"])

    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(away_rest, on="gameId", how="left")
    # Sec22: symmetric embedded-bias credit on BOTH lambdas (corrected from Sec19.5/Sec20's
    # wrong, away-only-conditional version, which shifted every game's margin toward away and
    # caused a real margin-MAE regression), plus the original conditional term unchanged.
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    base = base.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(base["p_b2b_walk_forward"], away_b2b_adj)
    base["lambda_home"] = base["lambda_home"] + credit
    base["lambda_away_adj"] = base["lambda_away"] + credit + (base["away_rest"] == 0) * away_b2b_adj

    results = []
    for row in base.itertuples():
        joint = score_distribution(row.lambda_home, row.lambda_away_adj)
        home_win_reg, away_win_reg, tie_prob = home_win_prob_regulation(joint)
        home_win_full, _ = full_win_probability(home_win_reg, tie_prob, p_home_wins_ot)
        results.append({
            "gameId": row.gameId, "season": row.season,
            "actual_home": row.actual_home, "actual_away": row.actual_away,
            "lambda_home": row.lambda_home, "lambda_away": row.lambda_away_adj,
            "home_win_prob_full": home_win_full,
        })
    return pd.DataFrame(results), away_b2b_adj


if __name__ == "__main__":
    r, adj = run_validation()
    print(f"validated {len(r)} dev-set games, fitted away_b2b_adjustment={adj:.5f}")
    ledger_input = r.rename(columns={"home_win_prob_full": "home_win_prob"}).assign(
        away_win_prob=lambda d: 1 - d["home_win_prob"], tie_prob=0.0)
    row = append_run(
        ledger_input, model_name="rest_adjusted_poisson",
        config_flags={"away_b2b_adjustment": round(adj, 5), "split": "development", "dev_max_season": DEV_MAX_SEASON},
        notes="Cycle 11: away-team-on-back-to-back adjustment on top of goalie_overlay_poisson + Cycle 9's "
              "full OT/SO resolution. Real margin-MAE improvement (bootstrap 95% CI entirely positive), "
              "borderline Brier improvement, no total-MAE effect. See MODEL_DOCUMENTATION.md Sec4.14.",
    )
    for k, v in row.items():
        print(f"  {k}: {v}")
