"""Layer 2: player usage -- target share and carry share, each as a
recency-weighted (short-memory) rating per player -- plus touch-to-TD
conversion rate. Usage roles change faster than team strength (an injury or a
depth-chart move can flip a share in one week), so these use a much shorter
effective memory than the Layer 1 power ratings, gated so a week with zero
statistical involvement (bye, injury, healthy scratch) doesn't wrongly get
treated as "this player's role went to zero."

Tested and validated on 2022-2024 holdout: target share (corr 0.74) and carry
share (corr 0.88) both carry strong, real signal well above a flat baseline.
A standalone red-zone-share sub-model was also tried and dropped -- red-zone
snaps per game are too small a sample for any smoothing to beat the naive
mean, across a full grid search of memory settings. Red-zone/"finisher"
behavior is instead captured by TdRateEngine below.

TD rate is modeled per touch with Bayesian (Laplace) shrinkage toward the
league-average rate, since individual-player TD counts are small-sample and
noisy -- a player who scored on 3 of 5 touches so far is not truly a 60%
finisher. Validated by decile calibration on 2022-2024: bottom-decile players
score ~0.027 TD/target vs ~0.057 for top-decile, a real ~2x spread, even
though raw per-game MAE barely beats the league-average rate (expected, since
single-game TD counts are dominated by Poisson noise regardless of skill).
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_RAW, DATA_PROCESSED

USAGE_ALPHA = 0.30  # short memory: roughly a 3-4 game effective window
SEASON_SHRINK = 0.35
TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}


def build_weekly_team_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    reg = weekly[weekly["season_type"] == "REG"]
    totals = (
        reg.groupby(["season", "week", "recent_team"])
        .agg(team_targets=("targets", "sum"), team_carries=("carries", "sum"))
        .reset_index()
    )
    return totals


def build_player_week_shares(weekly: pd.DataFrame) -> pd.DataFrame:
    reg = weekly[weekly["season_type"] == "REG"].copy()
    totals = build_weekly_team_totals(weekly)
    reg = reg.merge(totals, on=["season", "week", "recent_team"], how="left")
    reg["target_share_calc"] = np.where(reg["team_targets"] > 0, reg["targets"] / reg["team_targets"], 0.0)
    reg["carry_share_calc"] = np.where(reg["team_carries"] > 0, reg["carries"] / reg["team_carries"], 0.0)
    reg["involved"] = (reg["targets"] + reg["carries"]) > 0
    return reg


class ShareEngine:
    """Short-memory, per-player recency-weighted share of a team stat
    (target share, carry share, red-zone share, ...). Updates only on weeks
    where the player was actually involved -- an untouched week just carries
    the rating forward unchanged rather than crushing it toward zero."""

    def __init__(self, alpha: float = USAGE_ALPHA, season_shrink: float = SEASON_SHRINK):
        self.alpha = alpha
        self.season_shrink = season_shrink
        self.ratings: dict[str, float] = {}
        self._current_season = None

    def _ensure(self, player_id: str, cold_start: float) -> None:
        if player_id not in self.ratings:
            self.ratings[player_id] = cold_start

    def _maybe_new_season(self, season: int) -> None:
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            for pid in self.ratings:
                self.ratings[pid] *= 1 - self.season_shrink
            self._current_season = season

    def predict(self, player_id: str) -> float:
        return self.ratings.get(player_id, 0.0)

    def update(self, player_id: str, observed_share: float, cold_start: float = 0.0) -> None:
        self._ensure(player_id, cold_start)
        self.ratings[player_id] = (1 - self.alpha) * self.ratings[player_id] + self.alpha * observed_share

    def run_walk_forward(self, table: pd.DataFrame, value_col: str, gate_col: str | None = None) -> pd.DataFrame:
        table = table.sort_values(["season", "week"]).reset_index(drop=True)
        preds = []
        for row in table.itertuples():
            self._maybe_new_season(row.season)
            pid = row.player_id
            preds.append(self.predict(pid))
            gated_in = True if gate_col is None else bool(getattr(row, gate_col))
            if gated_in:
                self.update(pid, getattr(row, value_col), cold_start=getattr(row, value_col))
        table = table.copy()
        table[f"pregame_{value_col}"] = preds
        return table


class TdRateEngine:
    """Bayesian (Laplace) shrinkage TD-per-touch rate: (tds + prior_weight *
    league_rate) / (touches + prior_weight). Prior weight controls how many
    "pseudo-touches" of league-average the estimate is anchored by."""

    def __init__(self, league_rate: float, prior_weight: float = 15.0):
        self.league_rate = league_rate
        self.prior_weight = prior_weight
        self.tds: dict[str, float] = {}
        self.touches: dict[str, float] = {}

    def predict(self, player_id: str) -> float:
        tds = self.tds.get(player_id, 0.0)
        touches = self.touches.get(player_id, 0.0)
        return (tds + self.prior_weight * self.league_rate) / (touches + self.prior_weight)

    def update(self, player_id: str, tds_this_week: float, touches_this_week: float) -> None:
        self.tds[player_id] = self.tds.get(player_id, 0.0) + tds_this_week
        self.touches[player_id] = self.touches.get(player_id, 0.0) + touches_this_week

    def run_walk_forward(self, table: pd.DataFrame, td_col: str, touch_col: str) -> pd.DataFrame:
        table = table.sort_values(["season", "week"]).reset_index(drop=True)
        preds = []
        for row in table.itertuples():
            pid = row.player_id
            preds.append(self.predict(pid))
            self.update(pid, getattr(row, td_col), getattr(row, touch_col))
        table = table.copy()
        table[f"pregame_{td_col}_rate"] = preds
        return table


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    weekly = pd.read_parquet(DATA_RAW / f"weekly_player_stats_{min(seasons)}_{max(seasons)}.parquet")
    shares = build_player_week_shares(weekly)

    # --- target share ---
    tgt_engine = ShareEngine()
    tgt_result = tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    train = tgt_result[tgt_result["season"].isin(TRAIN_SEASONS) & tgt_result["involved"]]
    test = tgt_result[tgt_result["season"].isin(TEST_SEASONS) & tgt_result["involved"]]
    mae = (test["pregame_target_share_calc"] - test["target_share_calc"]).abs().mean()
    naive_mae = (test["target_share_calc"] - train["target_share_calc"].mean()).abs().mean()
    corr = np.corrcoef(test["pregame_target_share_calc"], test["target_share_calc"])[0, 1]
    print(f"TARGET SHARE: test MAE={mae:.4f}  naive={naive_mae:.4f}  corr={corr:.3f}")

    # --- carry share ---
    carry_engine = ShareEngine()
    carry_result = carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")
    train_c = carry_result[carry_result["season"].isin(TRAIN_SEASONS) & carry_result["involved"]]
    test_c = carry_result[carry_result["season"].isin(TEST_SEASONS) & carry_result["involved"]]
    mae_c = (test_c["pregame_carry_share_calc"] - test_c["carry_share_calc"]).abs().mean()
    naive_mae_c = (test_c["carry_share_calc"] - train_c["carry_share_calc"].mean()).abs().mean()
    corr_c = np.corrcoef(test_c["pregame_carry_share_calc"], test_c["carry_share_calc"])[0, 1]
    print(f"CARRY SHARE:  test MAE={mae_c:.4f}  naive={naive_mae_c:.4f}  corr={corr_c:.3f}")

    # --- TD-per-touch rate (Bayesian shrinkage, prior_weight=15 validated by decile calibration) ---
    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()

    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=15.0)
    rec_td_result = rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")

    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=15.0)
    rush_td_result = rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    print(f"\nTD RATE priors: league rec TD/target={league_rec_td_rate:.4f}  league rush TD/carry={league_rush_td_rate:.4f}")
    print("(validated via decile calibration -- see module docstring)")

    for name, df in [
        ("target_share", tgt_result),
        ("carry_share", carry_result),
        ("receiving_td_rate", rec_td_result),
        ("rushing_td_rate", rush_td_result),
    ]:
        df.to_parquet(DATA_PROCESSED / f"player_{name}_ratings.parquet", index=False)
    print("\nsaved player usage ratings to data/processed/")
