"""Rolling-window walk-forward backtest utility -- DEV-ONLY, never touches
holdout. Strengthens Stage 1/Stage 2's existing screening (a recent-dev
slice, then one aggregate full-dev-range comparison) with a THIRD lens: a
per-season breakdown across the whole dev range, exposing whether a
candidate's apparent aggregate win is STABLE season-by-season or driven by
one atypical year the aggregate number quietly averages over.

Simpler than a textbook rolling-window backtest needs to be here, because
this project's rate models (`shrinkage.add_walk_forward_rate` and friends)
are ALREADY walk-forward-safe end to end -- every prediction in the full
dev-range fit already uses only strictly-prior games (shift-based cumulative
sums, never a flat single train/test split). There is no leakage to guard
against by re-fitting per rolling origin the way a non-walk-forward model
would need to. The real gap this fills is REPORTING: `bootstrap_compare`
run once over the full dev range gives one pooled number; this slices that
SAME already-computed, already-safe prediction set by season and runs the
identical bootstrap comparison independently per season, so a candidate that
wins in aggregate only because of one unusually favorable season becomes
visible instead of hidden inside the pooled average.

Never reads real holdout (2024-2025) -- this is a pre-holdout dev-only
screening tool, one more filter before the one-time confirmatory holdout
read a genuinely promising candidate still has to clear, not a replacement
for it (per this project's confirmatory-veto protocol, `final_holdout_check.py`).
"""

import pandas as pd

from src.models.bootstrap_significance import bootstrap_compare


def rolling_window_report(cand_df: pd.DataFrame, baseline_df: pd.DataFrame, game_id_col: str,
                           metrics: list[dict], season_col: str = "season",
                           test_seasons: list[int] | None = None,
                           n_bootstrap: int = 2000) -> dict[int, dict]:
    """cand_df/baseline_df: per-game(-arm) rows, each with `game_id_col`,
    `season_col`, and every column named in `metrics[i]["col"]` -- same
    shape `bootstrap_compare` itself expects, just sliced independently per
    season here instead of pooled once.

    `test_seasons` (default: every season present in `cand_df` EXCEPT the
    very first): the first dev season has the thinnest walk-forward history
    (every team's rating is still mostly prior-anchored, not yet
    self-informed) and isn't a meaningful rolling-origin test window on its
    own -- excluded by default, same reasoning `add_walk_forward_rate`'s own
    season-reset convention already applies at the row level.

    `n_bootstrap` defaults lower than `bootstrap_compare`'s own 5,000: this
    function runs one bootstrap PER SEASON (typically 7-8 for the current
    dev range), so keeping each one cheaper keeps the whole report fast
    without materially widening any single season's CI at this sample size.

    Returns {season: bootstrap_compare(...)-shaped result dict}. Does not
    print a verdict itself -- callers should inspect per-season deltas/CIs
    directly, since "REAL IMPROVEMENT every season" vs. "aggregate real
    improvement, but one season shows real regression" are genuinely
    different findings this function is built to distinguish, not collapse
    into one number."""
    if test_seasons is None:
        seasons = sorted(cand_df[season_col].unique())
        test_seasons = seasons[1:] if len(seasons) > 1 else seasons

    results = {}
    for season in test_seasons:
        c = cand_df[cand_df[season_col] == season]
        b = baseline_df[baseline_df[season_col] == season]
        if len(c) == 0 or len(b) == 0:
            continue
        results[season] = bootstrap_compare(
            c, b, game_id_col=game_id_col, metrics=metrics,
            label_a=f"candidate_s{season}", label_b=f"baseline_s{season}", n_bootstrap=n_bootstrap)
    return results


def summarize_rolling_report(results: dict[int, dict], metric_name: str) -> dict:
    """Rolls a `rolling_window_report` result up into one summary for a
    SINGLE metric: how many seasons showed real improvement / noise / real
    regression, and whether any season regressed while the pooled aggregate
    (computed separately, by the caller, via a normal `bootstrap_compare`
    over the full range) looked like a clean win -- the exact "hidden by the
    average" failure mode this module exists to catch."""
    verdicts = {season: r[metric_name]["verdict"] for season, r in results.items()}
    n_improve = sum("REAL IMPROVEMENT" in v for v in verdicts.values())
    n_regress = sum("REAL REGRESSION" in v for v in verdicts.values())
    n_noise = sum("NOISE" in v for v in verdicts.values())
    return {
        "per_season_verdict": verdicts,
        "n_seasons": len(verdicts),
        "n_real_improvement": n_improve,
        "n_real_regression": n_regress,
        "n_noise": n_noise,
        "any_season_regressed": n_regress > 0,
    }
