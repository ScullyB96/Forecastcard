"""Rest-days / back-to-back adjustment. Built with the sibling-project
discipline of correlating against RESIDUALS first and only shipping a
contrast actually found real, rather than assuming a back-to-back penalty
exists just because it's NBA folklore.

Known bias trap to watch for (an NHL sibling lesson): a team's own
walk-forward rating is an average over that team's ACTUAL past games, which
already includes whatever mix of rested/back-to-back games that team
happened to play. If back-to-back frequency correlates with anything about
team quality (it doesn't obviously, schedule-makers don't correlate B2B
count with team strength) this would be a confound; the real risk here is
narrower and still worth guarding: never fit the rest effect by correlating
raw SCORE against rest days -- that would just recover "good teams tend to
have better rest management" or scheduling artifacts. Always correlate the
RESIDUAL (actual - `team_strength.project_game`'s baseline prediction,
computed with no rest term at all) against rest days, so whatever the
walk-forward rating already explains is netted out first.
"""

import numpy as np
import pandas as pd
from scipy import stats

B2B_RATIO_PRIOR_GAMES = 0.0  # placeholder if a shrunk-toward-league fit is later needed


def add_rest_days(log: pd.DataFrame) -> pd.DataFrame:
    """Adds `rest_days` (days since this team's previous game, within the
    SAME season only -- a season-opener's rest is not meaningfully
    comparable to an in-season gap and is left as NaN) and `is_b2b`
    (rest_days == 0, i.e. played yesterday)."""
    log = log.sort_values(["team", "season", "gameDate"]).reset_index(drop=True)
    prev_date = log.groupby(["team", "season"])["gameDate"].shift(1)
    log["rest_days"] = (log["gameDate"] - prev_date).dt.days - 1
    log["is_b2b"] = log["rest_days"] == 0
    return log


def correlate_residual_with_rest(games_with_residual: pd.DataFrame, residual_col: str,
                                  rest_col: str) -> dict:
    """games_with_residual: one row per TEAM-side of a game (not per game),
    with `residual_col` = actual_score - baseline_projected_score (no rest
    term) and `rest_col` = that team's rest_days/is_b2b for that game.
    Returns {"r": pearson_r, "p": p_value, "n": n, "mean_residual_b2b":
    ..., "mean_residual_rested": ...} for both a continuous rest_days
    correlation and a binary is_b2b mean-difference view -- report both, a
    threshold effect (B2B specifically) can be real even if the continuous
    correlation across all rest values is weak."""
    valid = games_with_residual.dropna(subset=[residual_col, rest_col])
    r, p = stats.pearsonr(valid[rest_col].astype(float), valid[residual_col])

    b2b_mask = valid["is_b2b"] if "is_b2b" in valid.columns else valid[rest_col] == 0
    mean_b2b = valid.loc[b2b_mask, residual_col].mean()
    mean_rested = valid.loc[~b2b_mask, residual_col].mean()
    t_stat, t_p = stats.ttest_ind(valid.loc[b2b_mask, residual_col], valid.loc[~b2b_mask, residual_col], equal_var=False)

    return {
        "r": float(r), "p": float(p), "n": int(len(valid)),
        "mean_residual_b2b": float(mean_b2b), "mean_residual_rested": float(mean_rested),
        "b2b_effect": float(mean_b2b - mean_rested),
        "b2b_ttest_t": float(t_stat), "b2b_ttest_p": float(t_p),
        "n_b2b": int(b2b_mask.sum()),
    }


def add_schedule_density(log: pd.DataFrame, window_days: int = 4) -> pd.DataFrame:
    """Adds `games_in_last_{window_days}d`: how many games (STRICTLY
    PRIOR, same team/season) this team has played in the trailing
    `window_days` calendar days, INCLUDING today's own game date as the
    right edge of the window (so a genuine "3-in-4" stretch -- 3 games
    within a 4-day span ending tonight -- reads as 3, not 2). A distinct
    mechanism from `is_b2b` (which only looks at the SINGLE immediately-
    prior game): two teams can both be `is_b2b=True` tonight while one of
    them is on a brutal 4-games-in-5-nights stretch and the other just
    had a normal back-to-back after several rested days -- `is_b2b` alone
    can't distinguish accumulated fatigue from a single short turnaround.

    Walk-forward safe by construction (only counts games strictly before
    the current row's own gameDate, same season -- a season-opener's
    early games correctly show a low count with no artificial cross-
    season bleed, matching `add_rest_days`'s own within-season-only
    convention)."""
    log = log.sort_values(["team", "season", "gameDate"]).reset_index(drop=True)
    col = f"games_in_last_{window_days}d"
    counts = []
    for _, group in log.groupby(["team", "season"], sort=False):
        dates = group["gameDate"].to_numpy()
        for i, d in enumerate(dates):
            window_start = d - np.timedelta64(window_days, "D")
            counts.append(int(((dates[:i] > window_start) & (dates[:i] < d)).sum()))
    log[col] = counts
    return log


def fit_team_b2b_adjustment_walk_forward(df: pd.DataFrame, prior_games: float = 50.0) -> pd.DataFrame:
    """Task #70 (external-review follow-up, MODEL_DOCUMENTATION.md Sec32/54):
    Sec32's flat, POOLED league-wide B2B correction was a real diagnostic
    that didn't survive adoption. Real research confirms the effect is
    HETEROGENEOUS by roster composition (veteran/star-heavy rosters show
    materially larger B2B drops, concentrated in defense) -- a single
    pooled scalar averages a real large effect for some teams with a
    near-zero effect for others, diluting a genuine signal into noise.

    `df`: one row per game (`gameId`, `gameDate`, `team_home`, `team_away`,
    `pred_home`/`pred_away`, `actual_home`/`actual_away`,
    `home_is_b2b`/`away_is_b2b` -- same shape `validate_b2b_adjustment.py`
    already builds). Adds `home_pred_adj`/`away_pred_adj`: baseline plus a
    TEAM-SPECIFIC walk-forward B2B correction -- each team's own trailing
    mean B2B-game residual, count-weighted-shrunk toward the POOLED
    league-wide B2B residual mean at `prior_games` (B2B games, not all
    games) strength, same shrinkage shape as `home_court.
    fit_team_home_court_walk_forward`. Same explicit-sequential-loop
    same-game leak guard as `validate_b2b_adjustment._add_walk_forward_b2b_adjustment`
    (reads each team's running sum/count BEFORE folding in this game's own
    residuals)."""
    df = df.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    home_resid = (df["actual_home"] - df["pred_home"]).to_numpy()
    away_resid = (df["actual_away"] - df["pred_away"]).to_numpy()
    home_is_b2b = df["home_is_b2b"].fillna(False).to_numpy()
    away_is_b2b = df["away_is_b2b"].fillna(False).to_numpy()
    home_team = df["team_home"].to_numpy()
    away_team = df["team_away"].to_numpy()

    team_sum: dict = {}
    team_count: dict = {}
    pooled_sum, pooled_count = 0.0, 0

    home_adj = np.zeros(len(df))
    away_adj = np.zeros(len(df))
    for i in range(len(df)):
        pooled_mean = pooled_sum / pooled_count if pooled_count > 0 else 0.0

        t_home = home_team[i]
        home_adj[i] = (team_sum.get(t_home, 0.0) + prior_games * pooled_mean) / (team_count.get(t_home, 0) + prior_games)
        t_away = away_team[i]
        away_adj[i] = (team_sum.get(t_away, 0.0) + prior_games * pooled_mean) / (team_count.get(t_away, 0) + prior_games)

        if home_is_b2b[i]:
            team_sum[t_home] = team_sum.get(t_home, 0.0) + home_resid[i]
            team_count[t_home] = team_count.get(t_home, 0) + 1
            pooled_sum += home_resid[i]
            pooled_count += 1
        if away_is_b2b[i]:
            team_sum[t_away] = team_sum.get(t_away, 0.0) + away_resid[i]
            team_count[t_away] = team_count.get(t_away, 0) + 1
            pooled_sum += away_resid[i]
            pooled_count += 1

    df = df.copy()
    df["home_pred_adj"] = df["pred_home"] + df["home_is_b2b"].fillna(False).astype(float) * home_adj
    df["away_pred_adj"] = df["pred_away"] + df["away_is_b2b"].fillna(False).astype(float) * away_adj
    return df


def fit_b2b_adjustment(games_with_residual: pd.DataFrame, residual_col: str) -> float:
    """Only call this once `correlate_residual_with_rest` has confirmed a
    real (small p-value, not just nonzero) effect -- returns the mean
    residual specifically for back-to-back games, to be added as a flat
    per-team-side additive point adjustment in `team_strength.project_game`
    when that side is playing a back-to-back. A placeholder magnitude until
    validated; NOT wired into project_game by default."""
    b2b = games_with_residual[games_with_residual["is_b2b"]]
    return float(b2b[residual_col].mean())
