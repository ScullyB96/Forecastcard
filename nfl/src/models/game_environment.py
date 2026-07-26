"""Layer 1.5: game environment (total points, plays, pass/run split).

Findings from testing this empirically (see project notes): team-level plays
per game is close to unpredictable game-to-game (correlation ~0.05 with a
recency-weighted own-team average) -- it's dominated by game-flow noise
(overtime, weather, injuries, muffed snaps), not stable team identity. So pace
uses a flat league-average baseline rather than a team-specific model; adding
complexity there didn't earn its keep.

Neutral-script pass rate, by contrast, has real and useful persistence
(correlation ~0.25 game-to-game, ~0.33 year-over-year) reflecting genuine
play-calling/scheme identity, and a simple per-team EWMA captures it about as
well as a more complex opponent-adjusted version -- so that's what's used
here. A separate game-script adjustment (from Layer 1's predicted margin)
shifts the neutral-script baseline toward what teams actually do when
leading/trailing.

Total points calibration reuses Layer 1's pregame_total_signal (EPA-based)
and does show a real, if modest, improvement over predicting the mean. Dome
and closed-roof games score ~4-6 points more (total) than pure outdoor games,
a real and consistent effect, so is_indoor is included as a regressor. Wind
was tested too and rejected -- its estimated effect flipped sign between an
isolated outdoor-only test and a joint fit with the other terms, which is the
signature of a fragile, non-robust signal rather than a real one.
"""

import numpy as np
import pandas as pd

from src.models.ratings import PLAY_TYPES
from src.utils.paths import DATA_RAW, DATA_PROCESSED
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
NEUTRAL_SCRIPT_BAND = 8  # |score_differential| <= this counts as "neutral" for pass-rate tendency
MIN_NEUTRAL_PLAYS = 5
PASS_RATE_ALPHA = 0.15
PASS_RATE_SEASON_SHRINK = 0.5


def build_pass_rate_table(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[
        pbp["play_type"].isin(PLAY_TYPES)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & (pbp["season_type"] == "REG")
        & (pbp["score_differential"].abs() <= NEUTRAL_SCRIPT_BAND)
    ]
    plays = plays.assign(is_pass=(plays["play_type"] == "pass").astype(float))
    grouped = (
        plays.groupby(["season", "week", "game_id", "posteam", "defteam"])
        .agg(pass_rate=("is_pass", "mean"), neutral_plays=("is_pass", "size"))
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opponent"})
    )
    return grouped[grouped["neutral_plays"] >= MIN_NEUTRAL_PLAYS]


def build_full_game_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Actual pass rate over the WHOLE game (all scripts), for fitting the
    game-script adjustment against realized margin."""
    plays = pbp[
        pbp["play_type"].isin(PLAY_TYPES)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & (pbp["season_type"] == "REG")
    ]
    plays = plays.assign(is_pass=(plays["play_type"] == "pass").astype(float))
    return (
        plays.groupby(["season", "week", "game_id", "posteam"])
        .agg(full_pass_rate=("is_pass", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )


class PassRateEngine:
    """Simple per-team recency-weighted neutral-script pass rate (no opponent
    adjustment -- testing showed opponent adjustment didn't improve on this)."""

    LEAGUE_AVG_PASS_RATE = 0.57

    def __init__(self, alpha: float = PASS_RATE_ALPHA, season_shrink: float = PASS_RATE_SEASON_SHRINK):
        self.alpha = alpha
        self.season_shrink = season_shrink
        self.ratings: dict[str, float] = {}
        self._current_season = None

    def _ensure_team(self, team: str) -> None:
        if team not in self.ratings:
            self.ratings[team] = self.LEAGUE_AVG_PASS_RATE

    def _maybe_new_season(self, season: int) -> None:
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            for team in self.ratings:
                self.ratings[team] = (
                    1 - self.season_shrink
                ) * self.ratings[team] + self.season_shrink * self.LEAGUE_AVG_PASS_RATE
            self._current_season = season

    def predict(self, team: str) -> float:
        self._ensure_team(team)
        return self.ratings[team]

    def update(self, team: str, observed_pass_rate: float) -> None:
        self._ensure_team(team)
        self.ratings[team] = (1 - self.alpha) * self.ratings[team] + self.alpha * observed_pass_rate

    def run_walk_forward(self, pr_table: pd.DataFrame) -> pd.DataFrame:
        pr_table = pr_table.sort_values(["season", "week"]).reset_index(drop=True)
        preds = []
        for row in pr_table.itertuples():
            self._maybe_new_season(row.season)
            preds.append(self.predict(row.team))
            self.update(row.team, row.pass_rate)
        pr_table = pr_table.copy()
        pr_table["pregame_pass_rate_pred"] = preds
        return pr_table


def fit_calibration(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    return fit_linear(x, y)


def fit_game_script(team_margin: pd.Series, implied_team_total: pd.Series, pass_rate_residual: pd.Series) -> dict:
    """Game-script pass-rate model, v2 (review 2026-07 #1.5): adds the
    market-implied team total as a second regressor alongside team_margin.
    implied_team_total = total_line/2 +/- spread_line/2 -- a team's own
    market-implied point total, distinct from margin (who's favored) since it
    also captures the overall scoring environment (shootout-type games see
    more passing at a given margin than a low-total grind-it-out game).

    Validated walk-forward (TRAIN=2018-2021 fit / TEST=2022-2025 held out):
    MAE 0.07141 (team_margin alone) -> 0.07080 (+ implied_team_total),
    p=0.0406, with a same-direction improvement in all 4 individual TEST
    seasons (not driven by one anomalous year). The raw univariate
    correlation of implied_team_total with pass_rate_residual is negative
    (-0.068) while its coefficient here is positive (+0.0039) -- not a sign of
    a fragile signal (contrast the rejected total-points wind test, §7.4):
    implied_team_total conflates "how favored this team is" (correlated with
    team_margin at r=0.38) with "the scoring environment", and conditioning on
    team_margin isolates the latter, which has a sensible, interpretable
    positive sign (higher-total games see more passing net of who's ahead)."""
    X = np.column_stack([np.ones(len(team_margin)), team_margin, implied_team_total])
    y = np.asarray(pass_rate_residual, dtype=float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(coefs[0]), "margin_coef": float(coefs[1]), "total_coef": float(coefs[2])}


def apply_game_script(coefs: dict, team_margin, implied_team_total):
    return coefs["intercept"] + coefs["margin_coef"] * team_margin + coefs["total_coef"] * implied_team_total


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(seasons)}_{max(seasons)}.parquet")
    schedules = pd.read_parquet(DATA_RAW / f"schedules_{min(seasons)}_{max(seasons)}.parquet")
    sched = schedules[schedules["game_type"] == "REG"][
        ["game_id", "home_team", "away_team", "home_score", "away_score", "spread_line", "total_line"]
    ]

    # --- total points calibration: Layer 1's pregame_total_signal + dome/indoor.
    # Wind was tested too but rejected -- its effect flipped sign between an isolated
    # subset test and a joint fit, a sign of a fragile, non-robust signal.
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    layer1["actual_total"] = layer1["home_score"] + layer1["away_score"]
    roof = schedules[schedules["game_type"] == "REG"][["game_id", "roof"]]
    layer1 = layer1.merge(roof, on="game_id", how="left")
    layer1["is_indoor"] = layer1["roof"].isin(["dome", "closed"]).astype(float)

    train = layer1[layer1["season"].isin(TRAIN_SEASONS)]
    test = layer1[layer1["season"].isin(TEST_SEASONS)]
    X_train = np.column_stack([np.ones(len(train)), train["pregame_total_signal"], train["is_indoor"]])
    coefs, *_ = np.linalg.lstsq(X_train, train["actual_total"].to_numpy(), rcond=None)
    c, d, e = coefs
    X_test = np.column_stack([np.ones(len(test)), test["pregame_total_signal"], test["is_indoor"]])
    pred_total = X_test @ coefs
    total_mae = (pred_total - test["actual_total"]).abs().mean()
    naive_total_mae = (test["actual_total"] - train["actual_total"].mean()).abs().mean()
    print(f"TOTAL POINTS calibration: total = {c:.2f} + {d:.2f} * total_signal + {e:.2f} * is_indoor")
    print(f"  test MAE: {total_mae:.2f} pts  (naive mean-total baseline: {naive_total_mae:.2f} pts)")

    # --- pace: flat league-average baseline (see module docstring for why) ---
    pbp_plays = pbp[
        pbp["play_type"].isin(PLAY_TYPES)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & (pbp["season_type"] == "REG")
    ]
    plays_per_team_game = (
        pbp_plays.groupby(["season", "week", "game_id", "posteam"]).size().reset_index(name="plays")
    )
    league_avg_plays = plays_per_team_game[plays_per_team_game["season"].isin(TRAIN_SEASONS)]["plays"].mean()
    naive_pace_mae = (
        plays_per_team_game[plays_per_team_game["season"].isin(TEST_SEASONS)]["plays"] - league_avg_plays
    ).abs().mean()
    print(f"\nPACE: flat baseline = {league_avg_plays:.2f} plays/team/game")
    print(f"  test MAE using flat baseline: {naive_pace_mae:.2f} plays (no team-specific model beat this in testing)")

    # --- pass rate: simple per-team EWMA ---
    pr_table = build_pass_rate_table(pbp)
    engine = PassRateEngine()
    pr_result = engine.run_walk_forward(pr_table)
    pr_train = pr_result[pr_result["season"].isin(TRAIN_SEASONS)]
    pr_test = pr_result[pr_result["season"].isin(TEST_SEASONS)]
    pr_mae = (pr_test["pregame_pass_rate_pred"] - pr_test["pass_rate"]).abs().mean()
    naive_pr_mae = (pr_test["pass_rate"] - pr_train["pass_rate"].mean()).abs().mean()
    print(f"\nPASS RATE (neutral script, simple per-team EWMA):")
    print(f"  test MAE: {pr_mae:.4f}  (naive mean-pass-rate baseline: {naive_pr_mae:.4f})")

    # --- game-script adjustment: how much full-game pass rate shifts per point of margin ---
    full_pr = build_full_game_pass_rate(pbp)
    full_pr = full_pr.merge(sched, on="game_id", how="inner")
    full_pr["team_margin"] = np.where(
        full_pr["team"] == full_pr["home_team"],
        full_pr["home_score"] - full_pr["away_score"],
        full_pr["away_score"] - full_pr["home_score"],
    )
    full_pr = full_pr.merge(
        pr_result[["game_id", "team", "pregame_pass_rate_pred"]], on=["game_id", "team"], how="inner"
    )
    full_pr["pass_rate_residual"] = full_pr["full_pass_rate"] - full_pr["pregame_pass_rate_pred"]
    train_mask = full_pr["season"].isin(TRAIN_SEASONS)
    script_intercept, script_slope = fit_calibration(
        full_pr.loc[train_mask, "team_margin"], full_pr.loc[train_mask, "pass_rate_residual"]
    )
    test_mask = full_pr["season"].isin(TEST_SEASONS)
    script_pred = script_intercept + script_slope * full_pr.loc[test_mask, "team_margin"]
    script_mae = (script_pred - full_pr.loc[test_mask, "pass_rate_residual"]).abs().mean()
    naive_script_mae = full_pr.loc[test_mask, "pass_rate_residual"].abs().mean()
    print(f"\nGAME SCRIPT adjustment: pass_rate_residual = {script_intercept:.4f} + {script_slope:.5f} * team_margin")
    print(f"  test MAE: {script_mae:.4f} (naive, no script adjustment: {naive_script_mae:.4f})")

    # v2: + market-implied team total (review 2026-07 #1.5) -- see fit_game_script docstring
    full_pr_market = full_pr.dropna(subset=["spread_line", "total_line"]).copy()
    full_pr_market["implied_team_total"] = np.where(
        full_pr_market["team"] == full_pr_market["home_team"],
        full_pr_market["total_line"] / 2 + full_pr_market["spread_line"] / 2,
        full_pr_market["total_line"] / 2 - full_pr_market["spread_line"] / 2,
    )
    train_mask_m = full_pr_market["season"].isin(TRAIN_SEASONS)
    test_mask_m = full_pr_market["season"].isin(TEST_SEASONS)
    game_script_coefs = fit_game_script(
        full_pr_market.loc[train_mask_m, "team_margin"],
        full_pr_market.loc[train_mask_m, "implied_team_total"],
        full_pr_market.loc[train_mask_m, "pass_rate_residual"],
    )
    script_pred_v2 = apply_game_script(
        game_script_coefs, full_pr_market.loc[test_mask_m, "team_margin"], full_pr_market.loc[test_mask_m, "implied_team_total"]
    )
    script_mae_v2 = (script_pred_v2 - full_pr_market.loc[test_mask_m, "pass_rate_residual"]).abs().mean()
    print(f"\nGAME SCRIPT v2 (+ implied_team_total): {game_script_coefs}")
    print(f"  test MAE: {script_mae_v2:.4f} (v1 margin-only: {script_mae:.4f})")

    with open(DATA_PROCESSED / "environment_calibration.txt", "w") as fh:
        fh.write(f"total_c={c}\ntotal_d={d}\ntotal_e_is_indoor={e}\n")
        fh.write(f"league_avg_plays={league_avg_plays}\n")
        fh.write(f"script_slope={script_slope}\nscript_intercept={script_intercept}\n")
        fh.write(
            f"game_script_intercept={game_script_coefs['intercept']}\n"
            f"game_script_margin_coef={game_script_coefs['margin_coef']}\n"
            f"game_script_total_coef={game_script_coefs['total_coef']}\n"
        )

    pr_result.to_parquet(DATA_PROCESSED / "pass_rate_ratings.parquet", index=False)
    print("\nsaved environment_calibration.txt, pass_rate_ratings.parquet")
