"""Team-level walk-forward models for OREB/DREB/AST/TOV/STL/BLK -- the
macro anchor these 5 categories currently lack (see `usage_allocation.py`'s
module docstring: v1 deliberately leaves them UNANCHORED, a flagged fast-
follow, not an oversight, since Phase 1 never built a team-level
projection for them). Mirrors `team_strength.py`'s own pace x rating
architecture exactly: each stat gets a FOR (this team's own tendency) /
AGAINST (what this team allows the opponent) pair, walk-forward shrunk via
the SAME `shrinkage.add_walk_forward_rate` primitive Phase 1 already uses
for OFF_RATING/DEF_RATING, then combined via the SAME multiplicative-ratio
idiom `team_strength.project_game` already uses for pace x rating.

Built entirely from `BoxScoreTraditionalV3` player box scores already
cached for the six per-player rate models -- no new ingest needed, just
aggregated to the TEAM level (summed across every player on a team in a
game). Regular season only, matching `team_strength.build_team_game_log`'s
own convention (playoffs are a downstream prediction target, not
dev-training signal).
"""

import pandas as pd

from src.ingest.fetch_schedule import season_str
from src.models.shrinkage import add_walk_forward_rate
from src.utils.paths import DATA_RAW

# label -> raw box-score column to sum per team per game
STAT_COLUMNS = {
    "oreb": "reboundsOffensive", "dreb": "reboundsDefensive",
    "ast": "assists", "tov": "turnovers", "stl": "steals", "blk": "blocks",
}

# Placeholder priors (games) -- each swept and confirmed empirically before adoption in
# `validate_team_stat_rates.py`, NOT assumed to transfer from OFF/DEF_RATING's own prior or from
# each other (this project's standing discipline: every category earns its own check).
PRIOR_GAMES = {
    "oreb": 20.0, "dreb": 20.0, "ast": 20.0, "tov": 20.0, "stl": 20.0, "blk": 20.0,
}

# Categories cleared for LIVE anchoring, per the one-time confirmatory holdout check
# (`run_team_stat_holdout_check.py`, see MODEL_DOCUMENTATION.md Sec23). All six beat the naive
# floor on dev; AST/TOV/STL additionally showed a real (not noise) dev-vs-holdout MAE gap -- the
# same already-documented era-driven league-wide trend/regime-change phenomenon found at the
# PLAYER level in Sec16-19 -- but still beat naive IN ABSOLUTE TERMS on real holdout data, so are
# adopted anyway (exactly how player-level OREB/AST priors were handled in Sec15: "the gap didn't
# close, but absolute performance still held up"). OREB is the one genuine holdout failure: its
# model/naive comparison flips sign on holdout (model LOSES to naive, 3.0661 vs 3.0401 MAE) --
# a real veto, not just a widening gap -- so OREB is deliberately excluded here and stays
# unanchored (bottom-up player-sum only) until a different approach is found, mirroring how
# OREB/STL/BLK's matchup-difficulty gap was left unadopted in Sec20 rather than force-adopting a
# fix that doesn't actually generalize.
ADOPTED_CATEGORIES = ("dreb", "ast", "tov", "stl", "blk")


def build_team_stat_game_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per team per game, with each stat's `{label}_for` (this
    team's own total) and `{label}_against` (what the opponent recorded
    that game) -- the for/against pairing `shrinkage.add_walk_forward_rate`
    needs, mirroring OFF_RATING/DEF_RATING's own construction in
    `team_strength.py`."""
    schedule_frames, box_frames = [], []
    for y in range(start_year, end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_trad_player_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule_frames.append(pd.read_parquet(sched_path))
        box_frames.append(pd.read_parquet(box_path))
    if not schedule_frames:
        return pd.DataFrame()

    schedule = pd.concat(schedule_frames, ignore_index=True)
    schedule = schedule[schedule["seasonType"] == "Regular Season"]
    box = pd.concat(box_frames, ignore_index=True)

    team_totals = box.groupby(["gameId", "teamId"], as_index=False)[list(STAT_COLUMNS.values())].sum()
    team_totals = team_totals.rename(columns={col: label for label, col in STAT_COLUMNS.items()})

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        for_totals = team_totals.rename(columns={label: f"{label}_for" for label in STAT_COLUMNS})
        against_totals = team_totals.rename(columns={label: f"{label}_against" for label in STAT_COLUMNS})

        merged = schedule.merge(
            for_totals, left_on=["gameId", f"{side}TeamId"], right_on=["gameId", "teamId"], how="inner")
        merged = merged.merge(
            against_totals, left_on=["gameId", f"{other}TeamId"], right_on=["gameId", "teamId"], how="inner",
            suffixes=("", "_opp"))

        row = {
            "gameId": merged["gameId"], "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": merged["season"], "team": merged[f"{side}TeamId"], "opponent": merged[f"{other}TeamId"],
            "is_home": side == "home",
        }
        for label in STAT_COLUMNS:
            row[f"{label}_for"] = merged[f"{label}_for"]
            row[f"{label}_against"] = merged[f"{label}_against"]
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def add_team_stat_ratings(log: pd.DataFrame, cross_season_weight: float = 0.0) -> pd.DataFrame:
    """Adds, per category, f"{label}_attack_rate"/f"{label}_defense_rate"
    (walk-forward shrunk toward the trailing league average, same
    primitive and season-reset convention as OFF_RATING/DEF_RATING).

    `cross_season_weight` (default 0.0, i.e. the original full within-
    season reset): applies UNIFORMLY across all 6 categories -- added to
    test the same lever that gave `team_strength.add_team_ratings` a real
    win (Sec33) via the identical `shrinkage.add_walk_forward_rate`
    primitive, never previously tested for any team_stat_rates category.
    Pass a nonzero value to blend in each team's own immediately-preceding
    season rate for early-season games instead of resetting fully to the
    league average."""
    log = log.copy()
    for label, prior in PRIOR_GAMES.items():
        log = add_walk_forward_rate(log, f"{label}_for", f"{label}_against", prior, prefix=label,
                                     cross_season_weight=cross_season_weight)
    return log


def to_wide_games(log: pd.DataFrame) -> pd.DataFrame:
    """One row per real game, home_/away_ prefixed columns -- shared by
    `validate_team_stat_rates.py` and `run_team_stat_holdout_check.py` so
    both build predictions from the exact same game-shaping logic."""
    home = log[log["is_home"]].copy()
    away = log[~log["is_home"]].copy()
    games = home.merge(away, on=["gameId", "gameDate", "season"], suffixes=("_home", "_away"))
    return games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)


def project_team_stat(home_attack: float, home_defense: float,
                       away_attack: float, away_defense: float, league_avg: float) -> tuple[float, float]:
    """Same multiplicative-ratio combine as `team_strength.project_game`'s
    pace x rating, generalized to an arbitrary for/against stat pair --
    returns (projected_home_total, projected_away_total) for this category."""
    home_proj = league_avg * (home_attack / league_avg) * (away_defense / league_avg)
    away_proj = league_avg * (away_attack / league_avg) * (home_defense / league_avg)
    return home_proj, away_proj


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season

    current = current_nba_season()
    log = build_team_stat_game_log(FIRST_DEV_SEASON, current)
    print(f"team-stat game log: {len(log)} rows ({log['gameId'].nunique()} games)", flush=True)
    rated = add_team_stat_ratings(log)
    sample = rated.dropna(subset=["ast_attack_rate"]).tail(5)
    print(sample[["gameId", "team", "ast_for", "ast_attack_rate", "tov_for", "tov_attack_rate"]].to_string(), flush=True)
