"""Residuals-first check (Sec17), run BEFORE building anything: on real
OT/SO games, does the regulation lambda differential (lambda_home -
lambda_away, from the CURRENT best model's own walk-forward team-strength)
predict who wins the extra period at all? Checked separately for OT-decided
(lastPeriodType=="OT", real 3v3-plus-earlier-OT hockey) and SO-decided
(lastPeriodType=="SO", a shootout skills competition) games, per the prior
expectation that a team-strength differential should predict OT outcomes
more than shootout outcomes (skater 3v3 play is a real hockey skill
contest; a shootout is a narrower, more luck-driven skills competition).
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.overtime_shootout import add_regulation_score
from src.models.validate_cross_season import run_validation as run_current_best
from src.utils.paths import DATA_RAW

if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base, _adj = run_current_best()
    base = base[base["season"] < DEV_MAX_SEASON].copy()

    merged = base.merge(schedule_ot[["gameId", "went_to_ot", "lastPeriodType", "homeScore", "awayScore"]],
                         on="gameId", how="left")
    ot_games = merged[merged["went_to_ot"]].copy()
    ot_games["lambda_diff"] = ot_games["lambda_home"] - ot_games["lambda_away"]
    ot_games["home_won_ot"] = (ot_games["homeScore"] > ot_games["awayScore"]).astype(int)

    print(f"total OT/SO games in dev set: {len(ot_games)}")
    print(f"  OT-decided (lastPeriodType=='OT'): {(ot_games['lastPeriodType'] == 'OT').sum()}")
    print(f"  SO-decided (lastPeriodType=='SO'): {(ot_games['lastPeriodType'] == 'SO').sum()}")

    for label, subset in [("ALL OT/SO", ot_games),
                           ("OT-decided only", ot_games[ot_games["lastPeriodType"] == "OT"]),
                           ("SO-decided only", ot_games[ot_games["lastPeriodType"] == "SO"])]:
        r, p = pearsonr(subset["lambda_diff"], subset["home_won_ot"])
        print(f"\n{label}: n={len(subset)}")
        print(f"  correlation(lambda_diff, home_won_ot): r={r:.4f}, p={p:.4f}")
        # Simple bucketed check too, since a point-biserial r on a binary
        # outcome can understate a real but nonlinear relationship.
        subset = subset.copy()
        subset["favored_home"] = subset["lambda_diff"] > 0
        home_win_rate_when_favored = subset[subset["favored_home"]]["home_won_ot"].mean()
        home_win_rate_when_not = subset[~subset["favored_home"]]["home_won_ot"].mean()
        print(f"  home win rate when model favors home (lambda_diff>0): {home_win_rate_when_favored:.4f} "
              f"(n={subset['favored_home'].sum()})")
        print(f"  home win rate when model favors away (lambda_diff<=0): {home_win_rate_when_not:.4f} "
              f"(n={(~subset['favored_home']).sum()})")
