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

Air yards / aDOT / WOPR (added 2026-07, review #2.1): air_yards is already in
PBP, previously unused. aDOT (average depth of target) reuses TdRateEngine
generically (it's just another Bayesian-shrunk per-touch rate, air_yards/
targets instead of receiving_yards/targets); air_yards_share reuses
ShareEngine the same way target_share/carry_share do.

TESTED AND REJECTED as a point-estimate improvement, after a real self-caught
methodology bug: an initial test of both WOPR-as-volume-replacement and aDOT-
as-a-receiving-yards-correction showed strong, significant improvements
(p=0.03, p<0.0001) -- but that test approximated a player's projected targets
using a flat pass-attempts proxy (`target_share * ~35`) instead of the real,
game-specific pregame pass-rate prediction the live pipeline actually uses.
Once corrected to use the real historical pass-rate-adjusted target
projection (matching weekly_update.py exactly), BOTH results reversed to
null: WOPR-based receiving-yards MAE was *worse* than target_share-based
(19.20 vs 19.14, p=0.14, wrong direction), and aDOT added no significant
improvement beyond target_share+ypt_rate (MAE 19.206->19.198, p=0.51,
n=17,323, TEST=2022-2025). Neither is wired into the live pipeline.
Lesson, matching §12.2's np.polyfit pattern: a validation shortcut that
doesn't match the real production formula can manufacture a false-positive
result the real formula won't reproduce -- always test against the actual
mechanism being changed, not a simplified stand-in for it.

`compute_wopr()`/`fit_adot_calibration()`/`apply_adot_calibration()` below are
kept as correctly-implemented, reusable building blocks (the aDOT/WOPR
*computations* themselves are fine; it was the validation harness around them
that was flawed) -- useful groundwork for a future variance/distribution
layer (a decile-conditional check found aDOT does carry real information
about outcome variance/boom-potential within a target-share bucket, just not
something a simple point-estimate correction can exploit), not for a mean
correction today.
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
    agg = {"team_targets": ("targets", "sum"), "team_carries": ("carries", "sum")}
    if "air_yards" in reg.columns:
        agg["team_air_yards"] = ("air_yards", "sum")
    totals = reg.groupby(["season", "week", "recent_team"]).agg(**agg).reset_index()
    return totals


def build_player_week_shares(weekly: pd.DataFrame) -> pd.DataFrame:
    reg = weekly[weekly["season_type"] == "REG"].copy()
    totals = build_weekly_team_totals(weekly)
    reg = reg.merge(totals, on=["season", "week", "recent_team"], how="left")
    reg["target_share_calc"] = np.where(reg["team_targets"] > 0, reg["targets"] / reg["team_targets"], 0.0)
    reg["carry_share_calc"] = np.where(reg["team_carries"] > 0, reg["carries"] / reg["team_carries"], 0.0)
    if "team_air_yards" in reg.columns:
        reg["air_yards_share_calc"] = np.where(reg["team_air_yards"] > 0, reg["air_yards"] / reg["team_air_yards"], 0.0)
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


EB_MIN_TOUCHES = 20  # below this a player's career rate is too noisy to inform the variance decomposition


def fit_empirical_bayes_prior_weight(touches: pd.Series, hits: pd.Series, per_touch_variance: float, min_touches: int = EB_MIN_TOUCHES) -> dict:
    """Empirical-Bayes (review 2026-07 #2.3), replacing swept prior_weight
    constants (TD rate=30, catch_rate=15 -- the latter never actually
    validated, just carried over from an early guess). Method-of-moments
    variance decomposition (a DerSimonian-Laird-style random-effects
    estimator, applied per-player instead of per-study): the observed
    variance of career per-touch rates across players equals real
    between-player skill variance PLUS expected within-player sampling
    noise (touches is small-sample, noisy); subtracting the latter isolates
    the former, and prior_weight = per_touch_variance / between_player_variance
    -- literally "how many touches of league-average pseudo-data is one
    real touch worth," derived from the actual shape of the population
    rather than swept against an aggregate metric this project's own
    testing found flat across a wide range (which is blind to individual-
    outlier overconfidence -- exactly the Tory Horton case that originally
    motivated raising TD-rate prior_weight from 15 to 30).

    Validated (TRAIN=2018-2021, receiving TD rate): EB-fit prior_weight=210
    (vs. swept=30) tames a 5-TD/22-target outlier to 1.35x league average
    instead of 2.58x, with BETTER top-decile calibration (predicted 0.065 vs
    actual 0.059, a 10% overshoot, vs swept=30's 0.083 vs 0.060, a 38%
    overshoot) -- at a small aggregate MAE cost (0.069 vs 0.068 touch-
    weighted) consistent with this project's own established finding that
    the aggregate metric is flat across a wide prior_weight range. Same
    pattern held for rush TD rate (EB=148 vs swept=30) and catch rate
    (EB=45 vs swept=15, never previously validated).

    per_touch_variance: the within-player, single-touch outcome variance --
    mu*(1-mu) for a Bernoulli-shaped rate (TD rate, catch rate); for a
    continuous per-touch stat (yards/target, yards/carry) pass the real
    play-level variance of that outcome instead."""
    df = pd.DataFrame({"touches": np.asarray(touches, dtype=float), "hits": np.asarray(hits, dtype=float)})
    df = df[df["touches"] >= min_touches].copy()
    df["rate"] = df["hits"] / df["touches"]
    mu_hat = df["hits"].sum() / df["touches"].sum()
    v_obs = np.average((df["rate"] - mu_hat) ** 2, weights=df["touches"])
    v_sampling = np.average(per_touch_variance / df["touches"], weights=df["touches"])
    v_signal = max(v_obs - v_sampling, per_touch_variance * 1e-4)
    return {"prior_weight": float(per_touch_variance / v_signal), "n_players": len(df), "league_rate": float(mu_hat)}


def compute_wopr(target_share: pd.Series, air_yards_share: pd.Series) -> pd.Series:
    """WOPR (Weighted Opportunity Rating, review 2026-07 #2.1): standard
    fantasy-analytics composite of target share and air-yards share.
    TESTED AND REJECTED as a target_share replacement in the volume-projection
    formula, once properly re-validated against real historical pregame
    pass-rate predictions (not a flat pass-attempts proxy -- see module
    docstring): WOPR-based receiving-yards MAE was worse than target_share-
    based (19.20 vs 19.14, p=0.14, n=17,144, TEST=2022-2025). Not wired into
    the live pipeline; kept as a correctly-implemented utility only."""
    return 0.7 * target_share + 0.3 * air_yards_share


def fit_adot_calibration(proj_current: pd.Series, adot: pd.Series, actual_yards: pd.Series) -> dict:
    """Regression correction: actual_yards = a + b*proj_current + c*adot.
    TESTED AND REJECTED as a live correction (see module docstring): once
    proj_current is reconstructed using the real historical pregame pass-rate
    prediction (matching weekly_update.py's actual formula, not a flat
    pass-attempts proxy), aDOT adds no significant improvement beyond
    target_share+ypt_rate (MAE 19.206->19.198, p=0.51, n=17,323,
    TEST=2022-2025). Kept as a correctly-implemented utility only -- fit on
    TRAIN, apply on TEST/forward, same walk-forward discipline as
    weather_adjustment.py's fit_wind_adjustment, in case a future distribution/
    variance layer wants it."""
    X = np.column_stack([np.ones(len(proj_current)), proj_current, adot])
    y = np.asarray(actual_yards, dtype=float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(coefs[0]), "proj_coef": float(coefs[1]), "adot_coef": float(coefs[2])}


def apply_adot_calibration(coefs: dict, proj_current, adot):
    return coefs["intercept"] + coefs["proj_coef"] * proj_current + coefs["adot_coef"] * adot


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

    # --- TD-per-touch rate (Bayesian shrinkage). prior_weight=30 -- raised from an
    # earlier 15 that was only ever validated via aggregate decile calibration, never
    # swept. A proper sweep (2026-07) found aggregate MAE/calibration are flat from
    # prior_weight=10 to 130, but 15 under-regularizes extreme small-sample outliers
    # (a rookie with a real 5-TD-on-22-target stretch still projected at 3.3x league
    # average after shrinkage). 30 costs nothing in validated aggregate accuracy. ---
    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()

    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=30.0)
    rec_td_result = rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")

    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=30.0)
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
