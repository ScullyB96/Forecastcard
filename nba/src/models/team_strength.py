"""Walk-forward shrunk team-level PACE/OFF_RATING/DEF_RATING, and the
closed-form pace x rating combine that turns two teams' ratings into a
projected final score for a game.

`build_team_game_log` merges the cached schedule (real final scores, home/
away, date/season) with the cached advanced-box-score team rows (that
game's realized pace/ORtg/DRtg) into one row per team per game -- the shared
input every walk-forward rating in this module is built from.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_str
from src.models.shrinkage import add_walk_forward_mean, add_walk_forward_rate
from src.utils.paths import DATA_PROCESSED, DATA_RAW

PRIOR_GAMES_RATING = 15.0  # placeholder, un-calibrated -- see MODEL_DOCUMENTATION.md
PRIOR_GAMES_PACE = 15.0


def build_team_game_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per team per game, for every season in [start_year, end_year]
    that has cached schedule + box score data. Regular season only (a team's
    rating shouldn't be trained on playoff-intensity/rotation-shortened
    games mixed in with regular-season ones -- playoffs are a downstream
    prediction target, not dev-training signal, same principle as excluding
    spring training from the MLB sibling's true-talent fit)."""
    schedule_frames, box_frames = [], []
    for y in range(start_year, end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_adv_team_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule_frames.append(pd.read_parquet(sched_path))
        box_frames.append(pd.read_parquet(box_path))

    if not schedule_frames:
        return pd.DataFrame()

    schedule = pd.concat(schedule_frames, ignore_index=True)
    schedule = schedule[schedule["seasonType"] == "Regular Season"]
    box = pd.concat(box_frames, ignore_index=True)

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        merged = schedule.merge(
            box, left_on=["gameId", f"{side}TeamId"], right_on=["gameId", "teamId"], how="inner")
        rows.append(pd.DataFrame({
            "gameId": merged["gameId"],
            "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": merged["season"],
            "team": merged[f"{side}TeamId"],
            "opponent": merged[f"{other}TeamId"],
            "is_home": side == "home",
            "actualScore": merged[f"{side}Score"],
            "opponentScore": merged[f"{other}Score"],
            "pace": merged["pace"],
            "offRtg": merged["offensiveRating"],
            "defRtg": merged["defensiveRating"],
            "possessions": merged["possessions"],
        }))

    log = pd.concat(rows, ignore_index=True)
    return log.sort_values(["gameDate", "gameId", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


def add_team_ratings(log: pd.DataFrame, cross_season_weight: float = 0.0,
                      league_avg_halflife_games: float | None = None) -> pd.DataFrame:
    """Adds walk-forward shrunk pace_shrunk_mean, rtg_attack_rate (OFF),
    rtg_defense_rate (DEF), and their league-average companions.

    `league_avg_halflife_games` (default None, i.e. the original flat
    infinite-memory league average): passed straight through to
    `shrinkage.py`'s primitives -- see their docstrings for the
    scoring-era-drift motivation (Sec9.5/Sec24). None preserves the
    exact original, already-validated Phase 1 behavior."""
    log = add_walk_forward_mean(log, "pace", PRIOR_GAMES_PACE, prefix="pace",
                                 cross_season_weight=cross_season_weight,
                                 league_avg_halflife_games=league_avg_halflife_games)
    log = add_walk_forward_rate(log, "offRtg", "defRtg", PRIOR_GAMES_RATING, prefix="rtg",
                                 cross_season_weight=cross_season_weight,
                                 league_avg_halflife_games=league_avg_halflife_games)
    return log


def project_game(home_pace: float, away_pace: float, league_avg_pace: float,
                  home_oRtg: float, home_dRtg: float, away_oRtg: float, away_dRtg: float,
                  league_avg_rtg: float, home_court_mult: float) -> tuple[float, float, float]:
    """The Phase 1 closed-form combine (see MODEL_DOCUMENTATION.md/the plan
    for the derivation): a multiplicative-ratio idiom, one home-court
    multiplier applied symmetrically (home's projected rating scaled up,
    away's scaled down by the same factor) rather than as an additive bonus,
    so it composes cleanly with the ratio structure of the rest of the
    combine. Returns (projected_pace, projected_home_score, projected_away_score)."""
    projected_pace = league_avg_pace * (home_pace / league_avg_pace) * (away_pace / league_avg_pace)

    projected_home_rtg = (league_avg_rtg * (home_oRtg / league_avg_rtg) * (away_dRtg / league_avg_rtg)
                          * home_court_mult)
    projected_away_rtg = (league_avg_rtg * (away_oRtg / league_avg_rtg) * (home_dRtg / league_avg_rtg)
                          / home_court_mult)

    projected_home_score = projected_home_rtg * projected_pace / 100.0
    projected_away_score = projected_away_rtg * projected_pace / 100.0
    return projected_pace, projected_home_score, projected_away_score


if __name__ == "__main__":
    from src.ingest.fetch_schedule import current_nba_season
    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    print(f"team-game log: {len(log)} rows ({log['gameId'].nunique()} games) "
          f"across seasons {sorted(log['season'].unique())}", flush=True)
    if not log.empty:
        rated = add_team_ratings(log)
        out_path = DATA_PROCESSED / "team_game_log_rated.parquet"
        rated.to_parquet(out_path, index=False)
        print(f"saved {out_path}", flush=True)
