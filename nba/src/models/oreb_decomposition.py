"""OREB miss-decomposition primitive (task #55, Fable 5 external-review
critique item 6): `team_stat_rates.py`'s OREB category has a documented,
four-times-confirmed structural ceiling (shrinkage strength, adaptive
league average, cross_season_weight, own_halflife_games all converged to
the same "statistical tie with naive" result -- see MODEL_DOCUMENTATION.md
Sec30/34/37) because it fits raw OREB COUNT as an independent FOR/AGAINST
rate, conflating two genuinely different things: shot-VOLUME (how many
misses does this team's game generate at all -- already well-modeled by
this project's scoring machinery) and offensive-rebounding SKILL (of the
misses that happen, what share does this team grab).

This module separates them:

1. `misses_for`/`misses_against` (`fieldGoalsAttempted - fieldGoalsMade`,
   summed per team per game) -- walk-forward shrunk via the SAME
   `shrinkage.add_walk_forward_rate` primitive `team_stat_rates.py` already
   uses for every other category, combined via the SAME
   `team_stat_rates.project_team_stat` ratio idiom to get
   `projected_team_misses`.
2. `own_oreb_rate` (`oreb_for / misses_for` -- this team's own offensive-
   rebounding SKILL per own miss) and `own_dreb_rate` (`dreb_for /
   misses_against` -- this team's own defensive-rebounding SKILL per
   opponent's miss): EXPOSURE-normalized walk-forward rates (numerator/
   exposure, not a for/against COUNT pair), reimplemented here at
   (team, season) grain rather than cross-imported from
   `player_rate_shrinkage.py` -- this project's existing convention of
   separate-but-similar files per domain rather than a shared cross-
   subsystem import (see that module's own docstring).

`own_oreb_rate` (league-wide ~0.25-0.30) and `own_dreb_rate` (league-wide
~0.70-0.75) are COMPLEMENTARY but numerically DIFFERENT baselines (unlike
oRtg/dRtg, which legitimately share one scale by definition) -- so they
can't be combined via `project_team_stat`'s single-shared-league-avg
signature unchanged. `project_oreb_share` below is the same multiplicative-
ratio-DEVIATION idiom, adapted to take two distinct baselines.

`projected_team_OREB = projected_oreb_share * projected_team_misses`. This
gives an EXACT accounting identity for free, not an approximation: a team's
own projected OREB plus the opponent's projected DREB *on that same miss
set* equals that team's own projected misses by construction (the
opponent's DREB share is literally `1 - this team's OREB share`, never
independently re-estimated) -- the game-level coherence the current
independent FOR/AGAINST fit does not guarantee.
"""

import pandas as pd

from src.ingest.fetch_schedule import season_str
from src.models.shrinkage import add_walk_forward_rate
from src.models.team_stat_rates import build_team_stat_game_log, project_team_stat
from src.utils.paths import DATA_RAW

PRIOR_GAMES_MISSES = 20.0  # un-calibrated placeholder, same convention as every other new prior in
# this project -- swept and confirmed empirically in validate_oreb_decomposition.py before adoption.
PRIOR_MISSES_OREB_RATE = 200.0  # exposure unit is MISSES here, not games -- placeholder, swept.
PRIOR_MISSES_DREB_RATE = 200.0


def _build_misses_log(start_year: int, end_year: int) -> pd.DataFrame:
    """`team_stat_rates.build_team_stat_game_log`'s output (already has
    oreb_for/dreb_for/etc.) with `misses_for`/`misses_against` added
    (`fieldGoalsAttempted - fieldGoalsMade`, summed per team per game,
    joined on (gameId, team) / (gameId, opponent))."""
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
    box = pd.concat(box_frames, ignore_index=True)

    team_totals = box.groupby(["gameId", "teamId"], as_index=False)[
        ["fieldGoalsAttempted", "fieldGoalsMade"]].sum()
    team_totals["misses"] = team_totals["fieldGoalsAttempted"] - team_totals["fieldGoalsMade"]
    misses_lookup = team_totals[["gameId", "teamId", "misses"]]

    log = build_team_stat_game_log(start_year, end_year)
    log = log.merge(misses_lookup.rename(columns={"teamId": "team", "misses": "misses_for"}),
                     on=["gameId", "team"], how="inner")
    log = log.merge(misses_lookup.rename(columns={"teamId": "opponent", "misses": "misses_against"}),
                     on=["gameId", "opponent"], how="inner")
    return log


def _trailing_team_exposure_rate(log: pd.DataFrame, numerator_col: str, exposure_col: str) -> pd.Series:
    """Team-grain analog of `player_rate_shrinkage._trailing_league_rate`:
    trailing (strictly prior games only) POOLED league rate =
    cumsum(numerator)/cumsum(exposure), collapsed to one row per gameId
    FIRST (summing both team-rows) so a game's own two rows can never see
    each other, same leak guard as `shrinkage._trailing_league_stat`."""
    per_game = log.groupby("gameId", as_index=False).agg(
        gameDate=("gameDate", "first"), _num=(numerator_col, "sum"), _exp=(exposure_col, "sum"))
    per_game = per_game.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    cum_num = per_game["_num"].shift(1).expanding().sum()
    cum_exp = per_game["_exp"].shift(1).expanding().sum()
    per_game["_trailing_rate"] = cum_num / cum_exp
    return log.merge(per_game[["gameId", "_trailing_rate"]], on="gameId", how="left")["_trailing_rate"]


def add_walk_forward_exposure_rate(log: pd.DataFrame, numerator_col: str, exposure_col: str,
                                    prior_exposure: float, prefix: str) -> pd.DataFrame:
    """Team-grain walk-forward shrinkage of a single numerator_col/
    exposure_col rate (e.g. oreb_for/misses_for), season-reset -- same
    algebra as `player_rate_shrinkage.add_walk_forward_player_rate`
    (`shrunk_rate = (numerator_before + prior_exposure * league_avg) /
    (exposure_before + prior_exposure)`), reimplemented at (team, season)
    grain. Adds f"{prefix}_league_avg_rate" and f"{prefix}_shrunk_rate"."""
    log = log.sort_values(["gameDate", "gameId"]).reset_index(drop=True)
    league_avg_col = f"{prefix}_league_avg_rate"
    log[league_avg_col] = _trailing_team_exposure_rate(log, numerator_col, exposure_col)

    grp = log.groupby(["team", "season"])
    numerator_before = grp[numerator_col].cumsum() - log[numerator_col]
    exposure_before = grp[exposure_col].cumsum() - log[exposure_col]

    league_avg = log[league_avg_col]
    log[f"{prefix}_shrunk_rate"] = (numerator_before + prior_exposure * league_avg) / (exposure_before + prior_exposure)
    return log


def add_oreb_decomposition_ratings(log: pd.DataFrame) -> pd.DataFrame:
    """Adds `misses_attack_rate`/`misses_defense_rate` (for projected_team_misses,
    via `project_team_stat`) and `oreb_rate_shrunk_rate`/`dreb_rate_shrunk_rate`
    (for `project_oreb_share`) to a log built by `_build_misses_log`."""
    log = add_walk_forward_rate(log, "misses_for", "misses_against", PRIOR_GAMES_MISSES, prefix="misses")
    log = add_walk_forward_exposure_rate(log, "oreb_for", "misses_for", PRIOR_MISSES_OREB_RATE, prefix="oreb_rate")
    log = add_walk_forward_exposure_rate(log, "dreb_for", "misses_against", PRIOR_MISSES_DREB_RATE, prefix="dreb_rate")
    return log


def project_oreb_share(home_oreb_rate: float, home_dreb_rate: float,
                        away_oreb_rate: float, away_dreb_rate: float,
                        league_avg_oreb_rate: float, league_avg_dreb_rate: float) -> tuple[float, float]:
    """Same multiplicative-ratio-DEVIATION idiom as
    `team_stat_rates.project_team_stat`, adapted to two distinct league
    baselines (own_oreb_rate and own_dreb_rate are complementary but
    numerically different quantities, unlike oRtg/dRtg which share one
    scale). Returns (home_share_of_home_misses, away_share_of_away_misses)
    -- each team's own projected share of ITS OWN misses that it rebounds;
    the opponent's DREB share on that same miss set is `1 - this value`,
    never independently estimated (the source of this decomposition's
    exact game-level coherence)."""
    home_share = (league_avg_oreb_rate * (home_oreb_rate / league_avg_oreb_rate)
                  * (away_dreb_rate / league_avg_dreb_rate))
    away_share = (league_avg_oreb_rate * (away_oreb_rate / league_avg_oreb_rate)
                  * (home_dreb_rate / league_avg_dreb_rate))
    return home_share, away_share


def project_team_oreb(home_misses_attack: float, home_misses_defense: float,
                       away_misses_attack: float, away_misses_defense: float, league_avg_misses: float,
                       home_oreb_rate: float, home_dreb_rate: float,
                       away_oreb_rate: float, away_dreb_rate: float,
                       league_avg_oreb_rate: float, league_avg_dreb_rate: float) -> tuple[float, float]:
    """Full pipeline for one game: projected_team_misses (via
    `project_team_stat`) x projected_oreb_share (via `project_oreb_share`)
    -> (projected_home_oreb, projected_away_oreb)."""
    home_misses, away_misses = project_team_stat(
        home_misses_attack, home_misses_defense, away_misses_attack, away_misses_defense, league_avg_misses)
    home_share, away_share = project_oreb_share(
        home_oreb_rate, home_dreb_rate, away_oreb_rate, away_dreb_rate,
        league_avg_oreb_rate, league_avg_dreb_rate)
    return home_share * home_misses, away_share * away_misses


if __name__ == "__main__":
    from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season

    current = current_nba_season()
    log = _build_misses_log(FIRST_DEV_SEASON, current)
    print(f"misses log: {len(log)} rows ({log['gameId'].nunique()} games)", flush=True)
    rated = add_oreb_decomposition_ratings(log)
    sample = rated.dropna(subset=["oreb_rate_shrunk_rate"]).tail(5)
    print(sample[["gameId", "team", "misses_for", "oreb_for", "oreb_rate_shrunk_rate",
                   "dreb_for", "dreb_rate_shrunk_rate"]].to_string(), flush=True)
