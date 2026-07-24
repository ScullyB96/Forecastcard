"""FROZEN, HISTORICAL FILE -- ADOPTED (Sec34) then REVERTED (Sec36, 2026-07-24).
Kept for reproducibility, same status as `validate_ev_toi_halflife.py`; NOT
part of the current model. The original adoption bootstrap in this file's
own `__main__` (below) fit the committed config on the full dev set and
scored it on that SAME dev set -- in-sample tree memorization, the same
genus of bug the market-benchmark contamination (Sec35.9) had. Re-scored on
`run_out_of_fold_predictions` (added below for that correction), the dev
gain crosses zero, matching the already-neutral holdout check -- see Sec36
for the full re-examination. Current best model is `walk_forward_tie_ratio_
poisson` (Cycle 26) directly, no stacking layer.

Cycle 16 (queued since the project's original roadmap, finally reached
Sec33): GBM stacking layer on top of the current-best model
(`ev_toi_halflife_poisson`, Sec32). Tests whether a shallow, heavily-
regularized tree ensemble finds real nonlinear structure in the base
model's own context features that the independent-Poisson combine can't
see -- NOT a backdoor recalibration layer for the Sec33 deployment question
(deliberately no recent-era-specific feature here).

Features: the base model's own log-odds (home_win_prob_full, monotone-
INCREASING constraint -- the stack may never predict a lower win
probability for a higher base-model win probability, all else equal), plus
the context features the base model already consumes: a rating
differential (lambda_home-lambda_away, the base model's own implied
margin, pre-win-probability-transform), the goalie GSAx differential
(recomputed via team_strength_goalie's own functions, same construction as
validate_goalie.py), home/away rest-days and back-to-back flags, a
schedule-density proxy (games in the trailing 7 real days per side), a
season-progress index (min game-number / 82), and ONE planted iid Gaussian
noise feature (fixed seed) as the pre-registered kill switch: if no real
feature's nested walk-forward importance exceeds the noise feature's own,
the cycle terminates as a null before any candidate reaches the standard
bootstrap/holdout validation.

Model: sklearn's HistGradientBoostingClassifier -- the only tree ensemble
available in this environment (no xgboost/lightgbm) that supports a native
monotonic constraint (`monotonic_cst`). It has no row-subsampling knob the
way XGBoost/LightGBM do; `max_features<1.0` (feature-level subsampling) is
used in its place as the closest available "heavy subsampling" analogue,
noted here rather than silently substituted.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, log_loss

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.rest_schedule import add_rest_days
from src.models.team_strength_goalie import add_walk_forward_goalie_strength, build_primary_goalie_per_team_game
from src.models.validate_tie_mass_ratio import run_treated as run_current_best
from src.models.validate_tie_mass_ratio import _bootstrap
from src.utils.paths import DATA_RAW

NOISE_SEED = 20260724
HYPERPARAM_GRID = [
    {"max_depth": 2, "learning_rate": 0.03, "max_leaf_nodes": 7, "max_features": 0.5},
    {"max_depth": 2, "learning_rate": 0.10, "max_leaf_nodes": 7, "max_features": 0.7},
    {"max_depth": 3, "learning_rate": 0.03, "max_leaf_nodes": 15, "max_features": 0.5},
    {"max_depth": 3, "learning_rate": 0.10, "max_leaf_nodes": 15, "max_features": 0.7},
]
WALK_FORWARD_TEST_SEASONS = [20192020, 20202021, 20212022, 20222023, 20232024]


def _schedule_density(schedule: pd.DataFrame) -> pd.DataFrame:
    """Games in the trailing 7 real days per team-game (own team, prior
    games only -- the current game itself is not counted)."""
    from src.models.baseline_naive_poisson import build_team_game_log
    log = build_team_game_log(schedule)
    log["gameDate_dt"] = pd.to_datetime(log["gameDate"])
    log = log.sort_values(["team", "gameDate_dt"]).reset_index(drop=True)
    densities = []
    for team, grp in log.groupby("team"):
        dates = grp["gameDate_dt"].values
        density = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            window_start = dates[i] - np.timedelta64(7, "D")
            density[i] = int(((dates[:i] > window_start) & (dates[:i] < dates[i])).sum())
        grp = grp.copy()
        grp["density_7d"] = density
        densities.append(grp)
    out = pd.concat(densities, ignore_index=True)
    return out[["gameId", "team", "density_7d"]]


def _season_progress(schedule: pd.DataFrame) -> pd.DataFrame:
    from src.models.baseline_naive_poisson import build_team_game_log
    log = build_team_game_log(schedule)
    log = log.sort_values(["team", "season", "gameDate", "gameId"]).reset_index(drop=True)
    log["team_game_number"] = log.groupby(["team", "season"]).cumcount() + 1
    return log[["gameId", "team", "team_game_number"]]


def build_features(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]

    base = run_current_best(min_season=min_season, max_season=max_season)
    base["base_log_odds"] = np.log(base["home_win_prob_full"].clip(1e-6, 1 - 1e-6)
                                    / (1 - base["home_win_prob_full"].clip(1e-6, 1 - 1e-6)))
    base["rating_diff"] = base["lambda_home"] - base["lambda_away"]

    # Goalie GSAx differential -- recomputed independently (same construction as
    # validate_goalie.py) rather than modifying that file to expose it.
    goalie_log = build_primary_goalie_per_team_game(schedule)
    goalie_log = add_walk_forward_goalie_strength(goalie_log)
    goalie_log["goalie_relative"] = (goalie_log["goalie_shrunk_mean"] - goalie_log["goalie_league_avg"]).fillna(0.0)
    sched_teams = schedule[["gameId", "homeTeamAbbrev", "awayTeamAbbrev"]].drop_duplicates()
    base = base.merge(sched_teams, on="gameId", how="left")
    base = base.merge(
        goalie_log[["gameId", "team", "goalie_relative"]].rename(
            columns={"team": "homeTeamAbbrev", "goalie_relative": "home_goalie"}),
        on=["gameId", "homeTeamAbbrev"], how="left")
    base = base.merge(
        goalie_log[["gameId", "team", "goalie_relative"]].rename(
            columns={"team": "awayTeamAbbrev", "goalie_relative": "away_goalie"}),
        on=["gameId", "awayTeamAbbrev"], how="left")
    base["home_goalie"] = base["home_goalie"].fillna(0.0)
    base["away_goalie"] = base["away_goalie"].fillna(0.0)
    base["goalie_gsax_diff"] = base["home_goalie"] - base["away_goalie"]

    # Rest / back-to-back.
    rest = add_rest_days(schedule)
    home_rest = rest[rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "home_rest"})
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    base = base.merge(home_rest, on="gameId", how="left").merge(away_rest, on="gameId", how="left")
    base["home_b2b"] = (base["home_rest"] == 0).astype(float)
    base["away_b2b"] = (base["away_rest"] == 0).astype(float)
    base["home_rest"] = base["home_rest"].fillna(2.0)
    base["away_rest"] = base["away_rest"].fillna(2.0)

    # Schedule density (trailing 7-day games played, own team).
    density = _schedule_density(schedule)
    home_density = density.rename(columns={"team": "homeTeamAbbrev", "density_7d": "home_density_7d"})
    away_density = density.rename(columns={"team": "awayTeamAbbrev", "density_7d": "away_density_7d"})
    base = base.merge(home_density, on=["gameId", "homeTeamAbbrev"], how="left")
    base = base.merge(away_density, on=["gameId", "awayTeamAbbrev"], how="left")

    # Season progress (min of both teams' own game number this season / 82).
    progress = _season_progress(schedule)
    home_prog = progress.rename(columns={"team": "homeTeamAbbrev", "team_game_number": "home_game_number"})
    away_prog = progress.rename(columns={"team": "awayTeamAbbrev", "team_game_number": "away_game_number"})
    base = base.merge(home_prog, on=["gameId", "homeTeamAbbrev"], how="left")
    base = base.merge(away_prog, on=["gameId", "awayTeamAbbrev"], how="left")
    base["season_progress"] = base[["home_game_number", "away_game_number"]].min(axis=1) / 82.0

    # Planted noise feature -- the pre-registered kill switch.
    rng = np.random.default_rng(NOISE_SEED)
    base["noise_feature"] = rng.standard_normal(len(base))

    base["target"] = (base["actual_home"] > base["actual_away"]).astype(int)
    return base


FEATURE_COLS = ["base_log_odds", "rating_diff", "goalie_gsax_diff", "home_rest", "away_rest",
                "home_b2b", "away_b2b", "home_density_7d", "away_density_7d", "season_progress",
                "noise_feature"]
MONOTONIC = [1 if c == "base_log_odds" else 0 for c in FEATURE_COLS]


def _fit(params: dict, X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        monotonic_cst=MONOTONIC, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=15, max_iter=500, random_state=20260724, **params)
    clf.fit(X_train, y_train)
    return clf


COMMITTED_PARAMS = {"max_depth": 2, "learning_rate": 0.1, "max_leaf_nodes": 7, "max_features": 0.7}


def run_final_production(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
    """The actual current-best model's own output: `walk_forward_tie_ratio_
    poisson` base + the committed GBM stack for win probability. Fit ONCE
    on dev-only data (`[20102011, DEV_MAX_SEASON)`), regardless of what
    range is requested -- the same walk-forward discipline Sec35 fixed
    elsewhere in this project, applied here so this function doesn't
    reintroduce the identical class of bug for the GBM stage. Callers
    (market-benchmark scripts, any future consumer needing the true
    production probability) should use this, not re-fit the GBM themselves."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    dev_df = build_features(min_season=20102011, max_season=DEV_MAX_SEASON)
    clf = _fit(COMMITTED_PARAMS, dev_df[FEATURE_COLS].values, dev_df["target"].values)

    out_df = build_features(min_season=min_season, max_season=max_season)
    out_df["home_win_prob_full"] = clf.predict_proba(out_df[FEATURE_COLS].values)[:, 1]
    return out_df


MIN_OOF_TRAIN_GAMES = 1500  # roughly one dev season -- below this, fall back to the
# base model's own probability (no stack), matching what a real deployed system
# would do before it has enough history to fit a meaningful GBM at all.


def run_out_of_fold_predictions(min_season: int = 20102011, max_season: int | None = None) -> pd.DataFrame:
    """Genuine out-of-fold stack predictions for every dev-set game --
    fixes the market-benchmark contamination `run_final_production` has:
    that function fits the GBM ONCE on the full dev set and then scores
    games the model was partly trained on, which flatters exactly the
    games the market benchmark evaluates (a handful of dev-fit Poisson
    scalars have essentially no capacity to memorize individual games; a
    tree ensemble, even shallow and regularized, partially does).

    Expanding-window walk-forward: for each season, fit the COMMITTED
    configuration on all STRICTLY PRIOR seasons only, predict on that
    season. Seasons with fewer than MIN_OOF_TRAIN_GAMES of prior training
    data fall back to the base model's own `home_win_prob_full` (no stack)
    -- a real deployed system wouldn't have a meaningfully-fit GBM that
    early either, so this isn't a workaround, it's the honest simulation."""
    max_season = DEV_MAX_SEASON if max_season is None else max_season
    df = build_features(min_season=20102011, max_season=max_season)  # full history needed for training folds
    df = df.sort_values(["season", "gameId"]).reset_index(drop=True)

    oof_prob = df["home_win_prob_full"].copy()  # base-model fallback for every row by default
    for season in sorted(df["season"].unique()):
        train = df[df["season"] < season]
        test_mask = df["season"] == season
        if len(train) < MIN_OOF_TRAIN_GAMES:
            continue  # leave the base-model fallback in place for this season
        clf = _fit(COMMITTED_PARAMS, train[FEATURE_COLS].values, train["target"].values)
        oof_prob.loc[test_mask] = clf.predict_proba(df.loc[test_mask, FEATURE_COLS].values)[:, 1]

    df["home_win_prob_full"] = oof_prob
    return df[df["season"] >= min_season].copy()


if __name__ == "__main__":
    print("Building features (dev set)...")
    df = build_features()
    print(f"{len(df)} dev-set games, features: {FEATURE_COLS}")

    print("\n=== Nested walk-forward hyperparameter selection ===")
    grid_scores = {i: [] for i in range(len(HYPERPARAM_GRID))}
    for test_season in WALK_FORWARD_TEST_SEASONS:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if len(test) == 0 or len(train) < 1000:
            continue
        X_train, y_train = train[FEATURE_COLS].values, train["target"].values
        X_test, y_test = test[FEATURE_COLS].values, test["target"].values
        for i, params in enumerate(HYPERPARAM_GRID):
            clf = _fit(params, X_train, y_train)
            p = clf.predict_proba(X_test)[:, 1]
            ll = log_loss(y_test, p, labels=[0, 1])
            grid_scores[i].append(ll)
        print(f"  fold test_season={test_season}: train_n={len(train)} test_n={len(test)} done")

    mean_scores = {i: np.mean(v) for i, v in grid_scores.items() if v}
    best_i = min(mean_scores, key=mean_scores.get)
    best_params = HYPERPARAM_GRID[best_i]
    print(f"\nmean walk-forward log-loss per config: {mean_scores}")
    print(f"COMMITTED CONFIGURATION: {best_params}")

    print("\n=== Kill-switch check: permutation importance vs. planted noise feature ===")
    last_test_season = WALK_FORWARD_TEST_SEASONS[-1]
    train = df[df["season"] < last_test_season]
    test = df[df["season"] == last_test_season]
    X_train, y_train = train[FEATURE_COLS].values, train["target"].values
    X_test, y_test = test[FEATURE_COLS].values, test["target"].values
    clf = _fit(best_params, X_train, y_train)
    perm = permutation_importance(clf, X_test, y_test, scoring="neg_log_loss", n_repeats=20,
                                   random_state=20260724)
    importances = dict(zip(FEATURE_COLS, perm.importances_mean))
    for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {feat:18s} {imp:+.6f}")
    noise_importance = importances["noise_feature"]
    real_features = [f for f in FEATURE_COLS if f != "noise_feature"]
    best_real = max(importances[f] for f in real_features)
    print(f"\nnoise_feature importance: {noise_importance:+.6f}")
    print(f"best real feature importance: {best_real:+.6f}")

    if best_real <= noise_importance:
        print("\n*** KILL SWITCH TRIGGERED: no real feature exceeds the planted noise feature. "
              "Cycle terminates as a NULL result. ***")
    else:
        print("\nKill switch not triggered -- proceeding to full-dev bootstrap and holdout.")

        print("\n=== Full-dev fit, paired bootstrap vs. base model ===")
        X_all, y_all = df[FEATURE_COLS].values, df["target"].values
        clf_full = _fit(best_params, X_all, y_all)
        p_stack = clf_full.predict_proba(X_all)[:, 1]
        p_base = df["home_win_prob_full"].clip(1e-6, 1 - 1e-6).values
        brier_stack = (p_stack - y_all) ** 2
        brier_base = (p_base - y_all) ** 2
        d, lo, hi = _bootstrap(brier_base, brier_stack)
        real = "REAL" if (lo > 0) or (hi < 0) else "crosses zero"
        print(f"Brier: base={brier_base.mean():.5f} stack={brier_stack.mean():.5f} "
              f"diff={d:+.5f} CI[{lo:+.5f},{hi:+.5f}] ({real})")

        su_base = ((p_base > 0.5).astype(int) == y_all).astype(float)
        su_stack = ((p_stack > 0.5).astype(int) == y_all).astype(float)
        d_su, lo_su, hi_su = _bootstrap(su_base, su_stack)
        real_su = "REAL" if (lo_su > 0) or (hi_su < 0) else "crosses zero"
        print(f"SU: base={su_base.mean():.4f} stack={su_stack.mean():.4f} "
              f"diff={d_su:+.5f} CI[{lo_su:+.5f},{hi_su:+.5f}] ({real_su})")

        print("\n=== Holdout confirmatory check (single touch, committed config) ===")
        hdf = build_features(min_season=DEV_MAX_SEASON, max_season=20999999)
        Xh, yh = hdf[FEATURE_COLS].values, hdf["target"].values
        ph_stack = clf_full.predict_proba(Xh)[:, 1]
        ph_base = hdf["home_win_prob_full"].clip(1e-6, 1 - 1e-6).values
        bh_base, bh_stack = (ph_base - yh) ** 2, (ph_stack - yh) ** 2
        dh, loh, hih = _bootstrap(bh_base, bh_stack)
        realh = "REAL" if (loh > 0) or (hih < 0) else "crosses zero"
        print(f"Holdout Brier: base={bh_base.mean():.5f} stack={bh_stack.mean():.5f} "
              f"diff={dh:+.5f} CI[{loh:+.5f},{hih:+.5f}] ({realh})")
        if loh > 0:
            print("HOLDOUT VETO: stack is REAL-worse on holdout -- reject despite dev gain.")
        else:
            print("No holdout veto.")
