"""Layer 1: EPA-based, opponent-adjusted team power ratings.

Each team carries an offensive rating (EPA/play generated) and a defensive
rating (EPA/play allowed). Both are updated game-by-game with a recursive
opponent-adjusted filter, and reset toward league average (0) at the start of
each new season. Ratings used to predict a given game reflect only games
played strictly before it -- no lookahead.
"""

import pandas as pd

from src.utils.paths import DATA_RAW, DATA_PROCESSED

PLAY_TYPES = ("pass", "run")


def build_game_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, opponent) with offensive EPA/play."""
    plays = pbp[
        pbp["play_type"].isin(PLAY_TYPES)
        & pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & (pbp["season_type"] == "REG")
    ]
    grouped = (
        plays.groupby(["season", "week", "game_id", "posteam", "defteam"])
        .agg(off_epa_per_play=("epa", "mean"), plays=("epa", "size"))
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opponent"})
    )
    return grouped


def build_matchup_table(game_table: pd.DataFrame, value_col: str = "off_epa_per_play") -> pd.DataFrame:
    """Pair the two teams' rows for each game_id into one game-per-row table.

    value_col names the per-team-game metric being paired (default: offensive
    EPA/play). Reused as-is for other metrics like plays/game or pass rate.
    """
    a = game_table.rename(
        columns={
            "team": "home_team_tmp",
            "opponent": "away_team_tmp",
            value_col: "off_epa_a",
            "plays": "plays_a",
        }
    )
    b = game_table.rename(
        columns={
            "team": "away_team_tmp",
            "opponent": "home_team_tmp",
            value_col: "off_epa_b",
            "plays": "plays_b",
        }
    )
    merged = a.merge(
        b, on=["season", "week", "game_id", "home_team_tmp", "away_team_tmp"]
    )
    # de-dupe: a/b merge produces one row per game since (team,opponent) pairs are unique per game_id
    merged = merged.drop_duplicates(subset=["game_id"])
    return merged.rename(columns={"home_team_tmp": "team_a", "away_team_tmp": "team_b"})


class PowerRatingEngine:
    """Recursive opponent-adjusted EPA/play ratings, offense and defense separately."""

    def __init__(self, alpha: float = 0.06, off_shrink: float = 0.20, def_shrink: float = 0.50):
        self.alpha = alpha
        self.off_shrink = off_shrink
        self.def_shrink = def_shrink
        self.off_ratings: dict[str, float] = {}
        self.def_ratings: dict[str, float] = {}
        self._current_season = None

    def _ensure_team(self, team: str) -> None:
        if team not in self.off_ratings:
            self.off_ratings[team] = 0.0
            self.def_ratings[team] = 0.0

    def _maybe_new_season(self, season: int) -> None:
        if self._current_season is None:
            self._current_season = season
            return
        if season != self._current_season:
            for team in self.off_ratings:
                self.off_ratings[team] *= 1 - self.off_shrink
                self.def_ratings[team] *= 1 - self.def_shrink
            self._current_season = season

    def nets(self, home: str, away: str) -> tuple[float, float]:
        """(home offensive environment vs away D, away offensive environment vs home D).

        def_ratings stores EPA/play allowed beyond league average (positive =
        bad defense, allows more). The additive model is
        observed_epa = off_rating[offense] + def_rating[defense], so
        prediction must ADD the opponent's defensive rating, not subtract it.
        """
        self._ensure_team(home)
        self._ensure_team(away)
        home_net = self.off_ratings[home] + self.def_ratings[away]
        away_net = self.off_ratings[away] + self.def_ratings[home]
        return home_net, away_net

    def rating_diff(self, home: str, away: str) -> float:
        """(home offense vs away defense) - (away offense vs home defense), EPA/play units."""
        home_net, away_net = self.nets(home, away)
        return home_net - away_net

    def update(self, team_a: str, team_b: str, off_epa_a: float, off_epa_b: float) -> None:
        """team_a's offense played team_b's defense (off_epa_a) and vice versa (off_epa_b)."""
        self._ensure_team(team_a)
        self._ensure_team(team_b)
        target_off_a = off_epa_a - self.def_ratings[team_b]
        target_def_b = off_epa_a - self.off_ratings[team_a]
        target_off_b = off_epa_b - self.def_ratings[team_a]
        target_def_a = off_epa_b - self.off_ratings[team_b]

        new_off_a = (1 - self.alpha) * self.off_ratings[team_a] + self.alpha * target_off_a
        new_def_b = (1 - self.alpha) * self.def_ratings[team_b] + self.alpha * target_def_b
        new_off_b = (1 - self.alpha) * self.off_ratings[team_b] + self.alpha * target_off_b
        new_def_a = (1 - self.alpha) * self.def_ratings[team_a] + self.alpha * target_def_a

        self.off_ratings[team_a] = new_off_a
        self.def_ratings[team_b] = new_def_b
        self.off_ratings[team_b] = new_off_b
        self.def_ratings[team_a] = new_def_a

    def run_walk_forward(self, matchups: pd.DataFrame) -> pd.DataFrame:
        """Iterate games chronologically, recording the PRE-game rating_diff for each,
        then updating state with the game's actual result. Returns matchups with an
        added `pregame_rating_diff` column.
        """
        matchups = matchups.sort_values(["season", "week"]).reset_index(drop=True)
        pregame_diffs = []
        pregame_totals = []
        for row in matchups.itertuples():
            self._maybe_new_season(row.season)
            home_net, away_net = self.nets(row.team_a, row.team_b)
            pregame_diffs.append(home_net - away_net)
            pregame_totals.append(home_net + away_net)
            self.update(row.team_a, row.team_b, row.off_epa_a, row.off_epa_b)
        matchups = matchups.copy()
        matchups["pregame_rating_diff"] = pregame_diffs
        matchups["pregame_total_signal"] = pregame_totals
        return matchups

    def run_walk_forward_team_ratings(self, matchups: pd.DataFrame) -> pd.DataFrame:
        """Same walk-forward loop as run_walk_forward, but records each team's
        own PREGAME off_rating/def_rating snapshot per game (not the combined
        home/away net) -- used by drive_transitions.py to condition drive
        resampling on real team strength. Returns [season, week, team,
        off_rating, def_rating], two rows per game (team_a, team_b)."""
        matchups = matchups.sort_values(["season", "week"]).reset_index(drop=True)
        rows = []
        for row in matchups.itertuples():
            self._maybe_new_season(row.season)
            self._ensure_team(row.team_a)
            self._ensure_team(row.team_b)
            rows.append({"season": row.season, "week": row.week, "team": row.team_a,
                         "off_rating": self.off_ratings[row.team_a], "def_rating": self.def_ratings[row.team_a]})
            rows.append({"season": row.season, "week": row.week, "team": row.team_b,
                         "off_rating": self.off_ratings[row.team_b], "def_rating": self.def_ratings[row.team_b]})
            self.update(row.team_a, row.team_b, row.off_epa_a, row.off_epa_b)
        return pd.DataFrame(rows)


def build_dataset(pbp_seasons: list[int], schedule_seasons: list[int] | None = None) -> pd.DataFrame:
    """schedule_seasons defaults to pbp_seasons -- pass it explicitly when the
    two data sources cover different ranges (e.g. schedules already has the
    upcoming season published but pbp doesn't have any plays for it yet)."""
    if schedule_seasons is None:
        schedule_seasons = pbp_seasons
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{min(pbp_seasons)}_{max(pbp_seasons)}.parquet")
    schedules = pd.read_parquet(DATA_RAW / f"schedules_{min(schedule_seasons)}_{max(schedule_seasons)}.parquet")

    game_table = build_game_table(pbp)
    matchups = build_matchup_table(game_table)

    sched = schedules[schedules["game_type"] == "REG"][
        ["game_id", "home_team", "away_team", "home_score", "away_score"]
    ]
    matchups = matchups.merge(sched, on="game_id", how="inner")

    # team_a/team_b from the pbp-derived table may be either home or away; align to home/away
    matchups["actual_margin"] = matchups["home_score"] - matchups["away_score"]
    is_a_home = matchups["team_a"] == matchups["home_team"]
    matchups.loc[~is_a_home, ["team_a", "team_b", "off_epa_a", "off_epa_b", "plays_a", "plays_b"]] = matchups.loc[
        ~is_a_home, ["team_b", "team_a", "off_epa_b", "off_epa_a", "plays_b", "plays_a"]
    ].values
    matchups = matchups.rename(columns={"team_a": "home_team_x", "team_b": "away_team_x"})
    return matchups


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    df = build_dataset(seasons)
    engine = PowerRatingEngine()
    result = engine.run_walk_forward(
        df.rename(columns={"home_team_x": "team_a", "away_team_x": "team_b"})
    )
    out_path = DATA_PROCESSED / "layer1_games_with_ratings.parquet"
    result.to_parquet(out_path, index=False)
    print(f"wrote {len(result)} games to {out_path}")
    print(result[["season", "week", "team_a", "team_b", "pregame_rating_diff", "actual_margin"]].tail(10))
