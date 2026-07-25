"""Walk-forward RAPM-lite: an offense/defense-split, ridge-regularized
adjusted plus-minus per player, fit from lineup stints
(`src/ingest/build_stints.py`). This is the mechanism that turns tonight's
actual active roster into a team-rating ADJUSTMENT on top of
`team_strength.py`'s team-level baseline (see `lineup_rating.py`).

Design matrix: two rows per physical stint (one per team's offensive half).
For team A's offensive row: `off_p = 1` for each of A's 5 on-court players,
`def_p = 1` for each of B's 5 on-court players, everything else 0. Target
`y = 100 * points_scored_by_A_in_stint / possessions_in_stint` (per-100, the
same units as `team_strength`'s ratings), row weight = stint possessions x
garbage-time exposure weight (`garbage_time.py`).

Fit as GENERALIZED ridge with a per-player-column penalty (not a single
global alpha): `player_priors.lambda_for_experience` gives a larger penalty
to low-experience columns, shrinking them harder toward the shared 0 =
league-average prior (see that module's docstring for why this replaces a
nonzero prior MEAN with a variable prior VARIANCE instead -- a simpler,
equally-valid empirical-Bayes formulation). Solved via the normal equations
directly (not `sklearn.Ridge`, which only supports one global alpha):
`beta = solve(X^T W X + diag(lambda), X^T W y)`.

Refit cadence is PERIODIC (every `REFIT_PERIOD_DAYS`), not literally once
per unique game date -- a pragmatic tractability compromise, documented
here rather than silently substituted: refitting a several-hundred-to-a-
few-thousand-column ridge system for each of ~1,500+ individual game dates
across the full dev range would be far more compute than a biweekly refit,
while still being fully walk-forward-safe (each refit uses ONLY stints
strictly before its cutoff date, and is applied only to games on/after that
cutoff and before the next one)."""

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.garbage_time import add_garbage_time_weight
from src.models.player_priors import lambda_for_experience

REFIT_PERIOD_DAYS = 14
MIN_STINTS_TO_FIT = 200  # don't attempt a fit on a tiny, unstable early sample


def prepare_stints(stints: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Attaches gameDate, drops zero-possession degenerate stints, and adds
    `marginBeforeStint` + `exposureWeight` (garbage-time down-weight).

    BUG FOUND AND FIXED while running the real full-dev-range validation:
    `schedule["gameDate"]` comes straight from the cached parquet as a plain
    string ("YYYY-MM-DD"), not a datetime -- `compute_walkforward_player_ratings`
    compares this column against `pd.Timestamp` checkpoints
    (`pd.date_range(...)` always returns real Timestamps even when given
    string bounds), which raised `TypeError: Invalid comparison between
    dtype=str and Timestamp` the first time this ran on the real, full
    dev range (a small smoke test earlier had already-parsed
    `pd.date_range` dates on the synthetic side, so it never hit this)."""
    schedule = schedule.copy()
    schedule["gameDate"] = pd.to_datetime(schedule["gameDate"])
    s = stints.merge(schedule[["gameId", "gameDate"]], on="gameId", how="inner")
    s = s[s["possessions"] > 0].copy()
    s = s.sort_values(["gameDate", "gameId", "stintIdx"]).reset_index(drop=True)

    cum_home = s.groupby("gameId")["homePts"].cumsum() - s["homePts"]
    cum_away = s.groupby("gameId")["awayPts"].cumsum() - s["awayPts"]
    s["marginBeforeStint"] = cum_home - cum_away
    return add_garbage_time_weight(s)


def _player_universe(stints: pd.DataFrame) -> dict:
    """All distinct player IDs appearing in ANY stint in `stints` (the
    caller is responsible for only passing strictly-prior data), mapped to
    a stable column index."""
    players = set()
    for col in ("homePlayers", "awayPlayers"):
        for tup in stints[col]:
            players.update(tup)
    return {p: i for i, p in enumerate(sorted(players))}


def _career_games_played(stints: pd.DataFrame, player_index: dict) -> np.ndarray:
    """Distinct-game appearance count per player (in column order) -- the
    exposure measure `player_priors.lambda_for_experience` shrinks by. A
    player appears in both the homePlayers and awayPlayers columns across
    the season (home/away alternate game to game, not a fixed side), so
    appearances from BOTH columns must be pooled into one set of
    (playerId, gameId) pairs before counting distinct games -- counting each
    column separately and taking the max would silently halve a typical
    player's true games-played count."""
    exploded_home = stints[["gameId", "homePlayers"]].explode("homePlayers").rename(columns={"homePlayers": "playerId"})
    exploded_away = stints[["gameId", "awayPlayers"]].explode("awayPlayers").rename(columns={"awayPlayers": "playerId"})
    pooled = pd.concat([exploded_home, exploded_away], ignore_index=True)
    game_counts = pooled.groupby("playerId")["gameId"].nunique().to_dict()
    return np.array([game_counts.get(p, 0) for p in sorted(player_index, key=player_index.get)])


def build_design_matrix(stints: pd.DataFrame, player_index: dict) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    n_players = len(player_index)
    n_cols = 2 * n_players
    n_rows = 2 * len(stints)

    row_idx, col_idx, data = [], [], []
    y = np.empty(n_rows)
    w = np.empty(n_rows)

    r = 0
    for row in stints.itertuples(index=False):
        weight = row.possessions * row.exposureWeight
        for off_players, def_players, pts in (
            (row.homePlayers, row.awayPlayers, row.homePts),
            (row.awayPlayers, row.homePlayers, row.awayPts),
        ):
            y[r] = 100.0 * pts / row.possessions
            w[r] = weight
            for p in off_players:
                if p in player_index:
                    row_idx.append(r); col_idx.append(2 * player_index[p]); data.append(1.0)
            for p in def_players:
                if p in player_index:
                    row_idx.append(r); col_idx.append(2 * player_index[p] + 1); data.append(1.0)
            r += 1

    X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_cols))
    return X, y, w


def fit_rapm(stints_before_cutoff: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by playerId with off_rapm/def_rapm
    (points above/below the already-centered league average per 100
    possessions) and games_played_before. Empty DataFrame if there isn't
    enough data yet to fit anything meaningful."""
    if len(stints_before_cutoff) < MIN_STINTS_TO_FIT:
        return pd.DataFrame(columns=["playerId", "off_rapm", "def_rapm", "games_played_before"])

    player_index = _player_universe(stints_before_cutoff)
    career_games = _career_games_played(stints_before_cutoff, player_index)
    X, y, w = build_design_matrix(stints_before_cutoff, player_index)

    # Center y on its weighted mean first -- coefficients are then naturally
    # "points above/below average per 100 possessions", matching the
    # shrink-to-league-average convention used everywhere else in this project.
    y_centered = y - np.average(y, weights=w)

    Xw = X.multiply(w[:, None])
    gram = (Xw.T @ X).toarray()  # dense: a few hundred-to-low-thousands of columns, trivial to invert
    moment = np.asarray(X.T @ (w * y_centered)).ravel()

    lam = lambda_for_experience(career_games)
    penalty = np.repeat(lam, 2)  # same penalty for a player's off and def columns
    beta = np.linalg.solve(gram + np.diag(penalty), moment)

    players_sorted = sorted(player_index, key=player_index.get)
    return pd.DataFrame({
        "playerId": players_sorted,
        "off_rapm": beta[0::2],
        "def_rapm": beta[1::2],
        "games_played_before": career_games,
    }).set_index("playerId")


def compute_walkforward_player_ratings(stints_all: pd.DataFrame, refit_period_days: int = REFIT_PERIOD_DAYS) -> pd.DataFrame:
    """The walk-forward driver: refits every `refit_period_days` using ONLY
    stints strictly before that checkpoint, tagging each fit with the
    checkpoint date it's valid FROM (until the next checkpoint). Returns one
    row per (asOfDate, playerId) -- the lookup table `lineup_rating.py` uses
    for a given game's date."""
    stints_all = stints_all.sort_values("gameDate").reset_index(drop=True)
    first_date, last_date = stints_all["gameDate"].min(), stints_all["gameDate"].max()

    checkpoints = pd.date_range(first_date, last_date, freq=f"{refit_period_days}D")
    all_ratings = []
    for checkpoint in checkpoints:
        prior = stints_all[stints_all["gameDate"] < checkpoint]
        ratings = fit_rapm(prior)
        if ratings.empty:
            continue
        ratings = ratings.reset_index()
        ratings["asOfDate"] = checkpoint
        all_ratings.append(ratings)

    if not all_ratings:
        return pd.DataFrame(columns=["asOfDate", "playerId", "off_rapm", "def_rapm", "games_played_before"])
    return pd.concat(all_ratings, ignore_index=True)


if __name__ == "__main__":
    import sys

    from src.ingest.build_stints import build_season_stints
    from src.utils.paths import DATA_RAW

    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2015
    from src.ingest.fetch_schedule import season_str
    schedule = pd.read_parquet(DATA_RAW / f"schedule_{season_str(start_year)}.parquet")
    stints = build_season_stints(start_year)
    if stints.empty:
        print("no stints available yet -- backfill not far enough along", flush=True)
        sys.exit(0)

    prepared = prepare_stints(stints, schedule)
    print(f"prepared {len(prepared)} stints", flush=True)

    ratings = compute_walkforward_player_ratings(prepared)
    print(f"computed {len(ratings)} (asOfDate, player) rating rows across "
          f"{ratings['asOfDate'].nunique() if not ratings.empty else 0} checkpoints", flush=True)
    if not ratings.empty:
        last_checkpoint = ratings["asOfDate"].max()
        latest = ratings[ratings["asOfDate"] == last_checkpoint].sort_values("off_rapm", ascending=False)
        print(f"\ntop 10 offensive RAPM-lite as of {last_checkpoint.date()}:", flush=True)
        print(latest.head(10).to_string(), flush=True)
