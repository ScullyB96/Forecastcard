"""Validates the full win-probability pipeline (Layers 1-3 + the new Layer 4
OT/SO sub-model) against real actual wins/losses -- for the first time
without the renormalization workaround every prior cycle used
(`home_win_prob / (1 - tie_prob)`, which implicitly ignored the tie mass's
own contribution rather than actually resolving it).

Development-set only, per the Sec4.8/4.9 policy.
"""

import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.overtime_shootout import add_regulation_score, fit_ot_home_win_rate, full_win_probability
from src.models.validate_goalie import run_validation as run_current_best
from src.utils.paths import DATA_RAW


def run_validation(min_season: int = 20102011, max_season: int = DEV_MAX_SEASON) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    # Regular season, completed games only -- confirmed real (2026-07-23): a single PRESEASON
    # game (gameId 2015010045, gameType=1, SJS 4-1 ARI marked lastPeriodType=OT) fails the
    # regulation-tie consistency check, a genuine NHL API data anomaly for a game type this
    # project has never used in any validation anyway (not a bug in the reconstruction logic).
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)
    p_home_wins_ot = fit_ot_home_win_rate(schedule_ot, max_season_exclusive=max_season)

    base = run_current_best()  # gameId, season, actual_home/away, lambda_home/away, home_win_prob, tie_prob
    base = base[(base["season"] >= min_season) & (base["season"] < max_season)].copy()

    base["actual_home_win"] = (base["actual_home"] > base["actual_away"]).astype(int)
    home_win_full, away_win_full = zip(*[
        full_win_probability(hp, tp, p_home_wins_ot) for hp, tp in zip(base["home_win_prob"], base["tie_prob"])
    ])
    base["home_win_prob_full"] = home_win_full
    base["home_win_prob_renorm"] = (base["home_win_prob"] / (1 - base["tie_prob"]).clip(lower=1e-9)).clip(1e-6, 1 - 1e-6)
    return base, p_home_wins_ot


if __name__ == "__main__":
    import numpy as np

    r, p_home_wins_ot = run_validation()
    print(f"validated {len(r)} dev-set games")
    print(f"empirical P(home wins | reaches OT/SO), fit on dev set: {p_home_wins_ot:.4f}")

    def brier(p):
        return ((p.clip(1e-6, 1 - 1e-6) - r["actual_home_win"]) ** 2).mean()

    def log_loss(p):
        pc = p.clip(1e-6, 1 - 1e-6)
        return -(r["actual_home_win"] * np.log(pc) + (1 - r["actual_home_win"]) * np.log(1 - pc)).mean()

    print(f"\nOLD renormalization workaround: Brier={brier(r['home_win_prob_renorm']):.5f}, "
          f"log-loss={log_loss(r['home_win_prob_renorm']):.5f}")
    print(f"NEW full OT/SO resolution:      Brier={brier(r['home_win_prob_full']):.5f}, "
          f"log-loss={log_loss(r['home_win_prob_full']):.5f}")
