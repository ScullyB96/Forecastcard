"""Residuals-first check (§11.2's proposed next cycle), run BEFORE building
anything: is there a REAL season-to-season correlation in team strength that
the current model's season-reset priors are throwing away? Current
production (`shrinkage.add_walk_forward_toi_rate`, used by
`team_strength_situational.py`) resets every team to the pure trailing
league average at game 1 of each new season -- zero team-specific
information until current-season games accumulate. If last season's team
quality has no real predictive power for this season's EARLY games (beyond
league average), a cross-season prior is pointless. If it does, §11's
front-loaded market gap has a concrete, buildable explanation.

Uses whole-game xG differential (xGF-xGA per game, MoneyPuck `situation=="all"`)
as the summary team-strength stat -- simplest real test, not the full
4-channel situational split, since this check only needs to establish
whether the EFFECT exists at all, not calibrate its final shape.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.ingest.team_codes import normalize_moneypuck_team_code, normalize_nhl_api_team_code
from src.models.baseline_naive_poisson import build_team_game_log
from src.utils.paths import DATA_RAW

EARLY_SEASON_GAMES = 15


def build_whole_game_xg_log(schedule: pd.DataFrame, moneypuck: pd.DataFrame) -> pd.DataFrame:
    goals_log = build_team_game_log(schedule)
    goals_log["season_start"] = goals_log["season"].astype(str).str[:4].astype(int)
    goals_log["team"] = [normalize_nhl_api_team_code(t, s) for t, s in zip(goals_log["team"], goals_log["season_start"])]
    goals_log = goals_log.drop(columns=["season_start"])

    xg = moneypuck[moneypuck["situation"] == "all"].copy()
    xg["team"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(xg["team"], xg["season"])]
    xg = xg[["gameId", "team", "xGoalsFor", "xGoalsAgainst"]]

    merged = goals_log.merge(xg, on=["gameId", "team"], how="inner")
    return merged.sort_values(["team", "season", "gameDate", "gameId"]).reset_index(drop=True)


if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")

    log = build_whole_game_xg_log(schedule, moneypuck)
    log["xg_diff"] = log["xGoalsFor"] - log["xGoalsAgainst"]

    league_avg_xg_diff_by_season = log.groupby("season")["xg_diff"].transform("mean")
    log["xg_diff_resid"] = log["xg_diff"] - league_avg_xg_diff_by_season

    # Each team-season's FULL-season average xG-diff residual (the "last season rating").
    season_end = log.groupby(["team", "season"], as_index=False)["xg_diff_resid"].mean().rename(
        columns={"xg_diff_resid": "last_season_xg_diff_resid"})
    season_end = season_end.sort_values(["team", "season"]).reset_index(drop=True)
    season_end["season"] = season_end["season"].astype(int)
    # Map to the NEXT real season (skip any gap season, e.g. no 2020-21 NHL API "season" oddities
    # would just not match and get dropped -- fine, real games are the only accepted match).
    all_seasons = sorted(log["season"].unique())
    season_order = {s: i for i, s in enumerate(all_seasons)}
    season_end["season_idx"] = season_end["season"].map(season_order)
    season_end["next_season_idx"] = season_end["season_idx"] + 1
    idx_to_season = {i: s for s, i in season_order.items()}
    season_end["next_season"] = season_end["next_season_idx"].map(idx_to_season)

    prior = season_end[["team", "next_season", "last_season_xg_diff_resid"]].rename(
        columns={"next_season": "season"})

    log["team_game_number"] = log.groupby(["team", "season"]).cumcount() + 1
    early = log[log["team_game_number"] <= EARLY_SEASON_GAMES].copy()
    early = early.merge(prior, on=["team", "season"], how="inner")

    n = len(early)
    r, p = pearsonr(early["last_season_xg_diff_resid"], early["xg_diff_resid"])
    print(f"n={n} team-game rows (first {EARLY_SEASON_GAMES} games of a season, team has a matched "
          f"immediately-preceding season)")
    print(f"correlation(last season's full-season xG-diff residual, this season's early-game "
          f"xG-diff residual): r={r:.4f}, p={p:.2e}")

    # Compare against the SAME correlation measured on LATE-season games (41+), where current-season
    # data has already accumulated -- current model captures this part already; the interesting
    # comparison is whether the correlation is markedly stronger early than late.
    late = log[log["team_game_number"] >= 41].copy()
    late = late.merge(prior, on=["team", "season"], how="inner")
    r_late, p_late = pearsonr(late["last_season_xg_diff_resid"], late["xg_diff_resid"])
    print(f"\nSAME correlation, late-season (41+) games instead, n={len(late)}: "
          f"r={r_late:.4f}, p={p_late:.2e}")
    print(f"\nif r (early) >> r (late), last season's rating carries real information current-season "
          f"data hasn't yet replaced by game 41+ -- exactly what a cross-season prior would exploit.")
