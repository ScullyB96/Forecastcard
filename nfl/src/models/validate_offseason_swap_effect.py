"""Does SWAP_B_MARKET (2.970, §5.1/§6.1.1) actually apply to OFFSEASON QB changes, or only
to in-season ones? Real, high-stakes question flagged by external review (2026-08): since
Week 1 2026's CB flags can't fire yet (no 2026 injury data exists) and the weather adjustment
is a no-op this far out, `weekly_update.py`'s §5.2 offseason-swap bootstrap is the model's
ENTIRE source of disagreement with the market for the one week about to be published --
every other adjustment layer is silent. SWAP_B_MARKET was refit (§6.1.1) on the general
in-season swap-delta signal; the doc's only offseason-specific number (47 games, Layer-1-era
MAE 7.55->7.39) predates the market-residual basis entirely and was never re-run against it.

The in-season justification ("Layer 1 updates slowly, so the market may lag a sudden midweek
change") does not obviously apply to a March/April trade or free-agent signing -- that's the
single most public, most-priced-in information in the sport, with months for the opening line
to absorb it before Week 1. If the true effect at a season boundary is smaller than (or
indistinguishable from) the in-season effect, applying the full in-season coefficient to
Week 1 overstates the correction on the one game type this model will actually disagree with
the market about.

`build_starter_sequence()` already tracks QB continuity with no season-boundary reset (§5.1)
-- this adds the boundary tag on top and splits the existing swap-delta signal into
CROSS-SEASON (the qb_changed flag fires between a team's last game of season N and first game
of season N+1) vs IN-SEASON (fires between two games in the same season), then fits/scores
SWAP_B separately for each against the real market residual, walk-forward TRAIN=2018-2021 /
TEST=2022-2025 -- exactly the split the in-season fit was already validated with, just
partitioned by boundary type. Games where a team's ONLY qb_changed weeks are all one type are
used for a clean per-type read; the rare case of a genuine ambiguity (unused here) would be a
game where one side's change is cross-season and the other's is in-season.
"""

import numpy as np
import pandas as pd

from src.models.qb_adjustment import QbRatingEngine, build_qb_week_table, build_starter_sequence
from src.models.scoring import crps_gaussian, fit_residual_sigma, signed_bias
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def fit_ols_tstat(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    a, b = fit_linear(x, y)
    resid = y.to_numpy() - (a + b * x.to_numpy())
    n = len(x)
    if n <= 2 or np.allclose(x, x.iloc[0]):
        return a, b, 0.0
    sigma2 = (resid ** 2).sum() / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se_b = np.sqrt(sigma2 / sxx) if sxx > 0 else np.inf
    return a, b, (b / se_b if se_b > 0 else 0.0)


def build_tagged_swap_delta(schedules: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    qb_weeks = build_qb_week_table(weekly)
    starters = build_starter_sequence(schedules)
    starters = starters.sort_values(["team", "season", "week"]).reset_index(drop=True)
    starters["prev_season"] = starters.groupby("team")["season"].shift(1)
    starters["is_cross_season"] = starters["qb_changed"] & (starters["season"] != starters["prev_season"])
    starters["is_in_season"] = starters["qb_changed"] & (starters["season"] == starters["prev_season"])

    engine = QbRatingEngine()
    starters = engine.run_with_starters(starters, qb_weeks)
    starters["swap_delta"] = np.where(
        starters["qb_changed"], starters["pregame_qb_rating"] - starters["prev_qb_rating"], 0.0
    )
    starters["swap_delta"] = starters["swap_delta"].fillna(0.0)

    home = starters.rename(columns={
        "team": "home_team_x", "swap_delta": "home_swap_delta",
        "is_cross_season": "home_cross", "is_in_season": "home_in_season",
    })[["season", "week", "home_team_x", "home_swap_delta", "home_cross", "home_in_season"]]
    away = starters.rename(columns={
        "team": "away_team_x", "swap_delta": "away_swap_delta",
        "is_cross_season": "away_cross", "is_in_season": "away_in_season",
    })[["season", "week", "away_team_x", "away_swap_delta", "away_cross", "away_in_season"]]

    sched = schedules[schedules["game_type"] == "REG"][["game_id", "season", "week", "home_team", "away_team", "spread_line"]]
    merged = sched.merge(home, left_on=["season", "week", "home_team"], right_on=["season", "week", "home_team_x"], how="left")
    merged = merged.merge(away, left_on=["season", "week", "away_team"], right_on=["season", "week", "away_team_x"], how="left")
    for c in ["home_swap_delta", "away_swap_delta"]:
        merged[c] = merged[c].fillna(0.0)
    for c in ["home_cross", "home_in_season", "away_cross", "away_in_season"]:
        merged[c] = merged[c].fillna(False)
    merged["net_swap_delta"] = merged["home_swap_delta"] - merged["away_swap_delta"]

    any_cross = merged["home_cross"] | merged["away_cross"]
    any_in_season = merged["home_in_season"] | merged["away_in_season"]
    merged["cross_season_only"] = any_cross & ~any_in_season
    merged["in_season_only"] = any_in_season & ~any_cross
    return merged[["game_id", "spread_line", "net_swap_delta", "cross_season_only", "in_season_only"]]


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")

    tagged = build_tagged_swap_delta(schedules, weekly)
    df = layer1.merge(tagged, on="game_id", how="left").dropna(subset=["spread_line"])
    df["market_margin"] = df["spread_line"]
    df["resid"] = df["actual_margin"] - df["market_margin"]

    train = df[df["season"].isin(TRAIN_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    sigma = fit_residual_sigma(train["market_margin"], train["actual_margin"])

    print(f"n TRAIN={len(train)}  n TEST={len(test)}")
    print(f"cross-season-only swap games: TRAIN={train['cross_season_only'].sum()}  TEST={test['cross_season_only'].sum()}")
    print(f"in-season-only swap games:    TRAIN={train['in_season_only'].sum()}  TEST={test['in_season_only'].sum()}")

    for label, mask_col in [("CROSS-SEASON (offseason)", "cross_season_only"), ("IN-SEASON", "in_season_only")]:
        print(f"\n=== {label} swap games ===")
        tr = train[train[mask_col]]
        te = test[test[mask_col]].copy()
        if len(tr) < 10 or len(te) < 5:
            print(f"  too few games (TRAIN={len(tr)}, TEST={len(te)}) for a stable fit -- skipping")
            continue
        a, b, t = fit_ols_tstat(tr["net_swap_delta"], tr["resid"])
        print(f"  TRAIN fit: intercept={a:+.4f}  slope(SWAP_B)={b:+.4f}  t-stat={t:+.2f}  (n={len(tr)})")
        te["adj_margin"] = te["market_margin"] + a + b * te["net_swap_delta"]
        mae0 = (te["market_margin"] - te["actual_margin"]).abs().mean()
        mae1 = (te["adj_margin"] - te["actual_margin"]).abs().mean()
        crps0 = crps_gaussian(te["market_margin"], te["actual_margin"], sigma)["crps"]
        crps1 = crps_gaussian(te["adj_margin"], te["actual_margin"], sigma)["crps"]
        print(f"  TEST (n={len(te)}): MAE {mae0:.4f} -> {mae1:.4f}   CRPS {crps0:.4f} -> {crps1:.4f}")
        bias0 = signed_bias(te["market_margin"], te["actual_margin"])
        bias1 = signed_bias(te["adj_margin"], te["actual_margin"])
        print(f"  signed_bias: market alone={bias0['bias']:+.3f} CI=[{bias0['ci'][0]:+.3f},{bias0['ci'][1]:+.3f}]  "
              f"with SWAP_B={bias1['bias']:+.3f} CI=[{bias1['ci'][0]:+.3f},{bias1['ci'][1]:+.3f}]")
        # what does applying the CURRENT PRODUCTION SWAP_B_MARKET=2.970 do to this specific subset?
        te["adj_prod"] = te["market_margin"] + 2.970 * te["net_swap_delta"]
        mae_prod = (te["adj_prod"] - te["actual_margin"]).abs().mean()
        print(f"  for comparison, current production SWAP_B_MARKET=2.970 applied here: TEST MAE {mae0:.4f} -> {mae_prod:.4f}")

    print("\n(Interpretation belongs in MODEL_DOCUMENTATION.md, not hardcoded here.)")
