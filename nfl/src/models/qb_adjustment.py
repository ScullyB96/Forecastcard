"""QB-change adjustment for Layer 1.

Diagnosis (see project notes): the Layer 1 team rating updates slowly
(alpha=0.06 by design, since team strength is meant to be sticky), which
means when a team's starting QB changes -- injury, benching, bye-week
return -- the very next game is predicted using a team rating that still
reflects the OLD starter's performance. Measured on 2022-2025 holdout: MAE on
QB-change games is 10.70 vs 9.79 for no-change games, with a systematic +4.2pt
bias when the away team's QB changes (home teams beat our stale prediction).

Fix: a per-QB EPA/dropback rating (ground-truth starters from
schedules.home_qb_id/away_qb_id, not snap-count heuristics), and a one-game
"swap delta" adjustment = new_starter_rating - previous_starter_rating,
applied ONLY on the first game after a change (zero otherwise -- by the
following week Layer 1's own recency-weighted update has started to catch up
on its own).
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_RAW, DATA_PROCESSED
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
QB_ALPHA = 0.15
QB_SEASON_SHRINK = 0.15  # QB skill is far more persistent than team-level context, so shrink gently
MIN_ATTEMPTS = 5


class QbRatingEngine:
    """Recency-weighted EPA/dropback rating per QB (by player_id), gated on a
    minimum attempts threshold so mop-up/trick-play snaps don't move it."""

    def __init__(self, alpha: float = QB_ALPHA, season_shrink: float = QB_SEASON_SHRINK):
        self.alpha = alpha
        self.season_shrink = season_shrink
        self.ratings: dict[str, float] = {}
        self._current_season = None

    def _maybe_new_season(self, season: int) -> None:
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            for qb in self.ratings:
                self.ratings[qb] *= 1 - self.season_shrink
            self._current_season = season

    def predict(self, qb_id: str) -> float:
        return self.ratings.get(qb_id, 0.0)

    def update(self, qb_id: str, epa_per_dropback: float) -> None:
        if qb_id not in self.ratings:
            self.ratings[qb_id] = epa_per_dropback
        self.ratings[qb_id] = (1 - self.alpha) * self.ratings[qb_id] + self.alpha * epa_per_dropback

    def run_with_starters(self, starters: pd.DataFrame, qb_weeks: pd.DataFrame) -> pd.DataFrame:
        """Walk forward week-by-week: for every team-game row in `starters`,
        look up the CURRENT and PREVIOUS starter's rating using this engine's
        live state (so a starter who is injured/inactive this week still
        returns their last known rating, not a missing value) -- then apply
        this week's actual performances from `qb_weeks` before moving on."""
        starters = starters.sort_values(["season", "week"]).reset_index(drop=True)
        qb_weeks = qb_weeks.sort_values(["season", "week"]).reset_index(drop=True)

        starter_groups = {k: v for k, v in starters.groupby(["season", "week"])}
        update_groups = {k: v for k, v in qb_weeks.groupby(["season", "week"])}

        out_rows = []
        for key in sorted(set(starter_groups) | set(update_groups)):
            season, week = key
            self._maybe_new_season(season)
            if key in starter_groups:
                g = starter_groups[key].copy()
                g["pregame_qb_rating"] = g["qb_id"].map(self.predict)
                g["prev_qb_rating"] = g["prev_qb_id"].map(self.predict)
                out_rows.append(g)
            if key in update_groups:
                for row in update_groups[key].itertuples():
                    self.update(row.player_id, row.epa_per_dropback)
        return pd.concat(out_rows, ignore_index=True)


def build_qb_week_table(weekly: pd.DataFrame) -> pd.DataFrame:
    qb = weekly[(weekly["position"] == "QB") & (weekly["attempts"] >= MIN_ATTEMPTS) & (weekly["season_type"] == "REG")]
    qb = qb.assign(epa_per_dropback=qb["passing_epa"] / qb["attempts"])
    return qb[["season", "week", "player_id", "recent_team", "epa_per_dropback"]]


def build_starter_sequence(schedules: pd.DataFrame) -> pd.DataFrame:
    sched = schedules[schedules["game_type"] == "REG"]
    h = sched.rename(columns={"home_team": "team", "home_qb_id": "qb_id"})[
        ["season", "week", "game_id", "team", "qb_id"]
    ]
    a = sched.rename(columns={"away_team": "team", "away_qb_id": "qb_id"})[
        ["season", "week", "game_id", "team", "qb_id"]
    ]
    starters = pd.concat([h, a]).sort_values(["team", "season", "week"]).reset_index(drop=True)
    starters["prev_qb_id"] = starters.groupby("team")["qb_id"].shift(1)
    starters["qb_changed"] = (starters["qb_id"] != starters["prev_qb_id"]) & starters["prev_qb_id"].notna()
    return starters


def fit_calibration(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    return fit_linear(x, y)


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")

    qb_weeks = build_qb_week_table(weekly)
    starters = build_starter_sequence(schedules)

    engine = QbRatingEngine()
    starters = engine.run_with_starters(starters, qb_weeks)
    starters["swap_delta"] = np.where(
        starters["qb_changed"], starters["pregame_qb_rating"] - starters["prev_qb_rating"], 0.0
    )
    starters["swap_delta"] = starters["swap_delta"].fillna(0.0)
    qb_result = starters  # for the saved-ratings parquet below

    home_swap = starters.rename(columns={"team": "home_team_x2", "swap_delta": "home_swap_delta"})[
        ["game_id", "home_team_x2", "home_swap_delta"]
    ]
    away_swap = starters.rename(columns={"team": "away_team_x2", "swap_delta": "away_swap_delta"})[
        ["game_id", "away_team_x2", "away_swap_delta"]
    ]
    merged = layer1.merge(
        home_swap, left_on=["game_id", "team_a"], right_on=["game_id", "home_team_x2"], how="left"
    ).merge(away_swap, left_on=["game_id", "team_b"], right_on=["game_id", "away_team_x2"], how="left")
    merged["home_swap_delta"] = merged["home_swap_delta"].fillna(0.0)
    merged["away_swap_delta"] = merged["away_swap_delta"].fillna(0.0)
    merged["net_swap_delta"] = merged["home_swap_delta"] - merged["away_swap_delta"]

    train = merged[merged["season"].isin(TRAIN_SEASONS)].copy()
    test = merged[merged["season"].isin(TEST_SEASONS)].copy()

    # base Layer 1 calibration (already fit in tune.py, reproduced here for a clean before/after comparison)
    base_a, base_b = fit_calibration(train["pregame_rating_diff"], train["actual_margin"])
    train["base_pred"] = base_a + base_b * train["pregame_rating_diff"]
    test["base_pred"] = base_a + base_b * test["pregame_rating_diff"]
    train["base_resid"] = train["actual_margin"] - train["base_pred"]
    test["base_resid"] = test["actual_margin"] - test["base_pred"]

    base_mae = test["base_resid"].abs().mean()
    print(f"BEFORE (Layer 1 alone): test MAE = {base_mae:.3f}")

    # fit how much of the residual the swap_delta explains, walk-forward safe (fit on train residuals)
    swap_a, swap_b = fit_calibration(train["net_swap_delta"], train["base_resid"])
    print(f"swap adjustment fit: residual = {swap_a:.3f} + {swap_b:.3f} * net_swap_delta")

    test["adjusted_pred"] = test["base_pred"] + (swap_a + swap_b * test["net_swap_delta"])
    adj_mae = (test["adjusted_pred"] - test["actual_margin"]).abs().mean()
    adj_su = (np.sign(test["adjusted_pred"]) == np.sign(test["actual_margin"])).mean()
    base_su = (np.sign(test["base_pred"]) == np.sign(test["actual_margin"])).mean()
    print(f"AFTER  (with QB-swap adjustment): test MAE = {adj_mae:.3f}")
    print(f"straight-up: before={base_su:.3f}  after={adj_su:.3f}")

    changed_mask = test["net_swap_delta"] != 0.0
    print(f"\non the {changed_mask.sum()} games with a QB swap:")
    print(f"  MAE before: {test.loc[changed_mask, 'base_resid'].abs().mean():.3f}")
    print(
        f"  MAE after:  {(test.loc[changed_mask, 'adjusted_pred'] - test.loc[changed_mask, 'actual_margin']).abs().mean():.3f}"
    )

    with open(DATA_PROCESSED / "qb_adjustment_calibration.txt", "w") as fh:
        fh.write(f"swap_a={swap_a}\nswap_b={swap_b}\n")
    qb_result.to_parquet(DATA_PROCESSED / "qb_ratings.parquet", index=False)
    print("\nsaved qb_adjustment_calibration.txt, qb_ratings.parquet")
