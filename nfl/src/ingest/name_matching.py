"""Player/QB name <-> gsis_id matching, shared by every script that needs to
join an external, name-keyed source (Clay's PDF projections, a schedule's
home_qb_name/away_qb_name columns) against our own ID-keyed data.

Extracted 2026-07 (review #3.4 housekeeping audit) from predict_2026.py and
predict_props_2026.py, where this cascade was originally built as a one-off
for the 2026 preseason bootstrap. weekly_update.py (the live, ongoing
pipeline) had been importing these functions FROM those two reference/
one-time scripts -- backwards, since weekly_update.py is what's meant to keep
running long after predict_2026.py/predict_props_2026.py are stale, and would
break the moment either got tidied up or archived. Both now import from here
instead.
"""

import re

import pandas as pd


def norm_name(name: str) -> str | None:
    if pd.isna(name):
        return None
    s = name.lower().strip()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


def build_name_to_id(rosters: pd.DataFrame) -> dict:
    ros = rosters[["player_name", "player_id"]].dropna().copy()
    ros["name_norm"] = ros["player_name"].apply(norm_name)
    return ros.drop_duplicates("name_norm", keep="last").set_index("name_norm")["player_id"].to_dict()


def build_lastname_team_pos_to_id(rosters: pd.DataFrame, season: int) -> dict:
    """Fallback index for nickname mismatches (Ken/Kenneth, Cam/Cameron) that
    the exact-name lookup misses. Keyed on (last_name, team, position) from
    the most recent season's roster -- tight enough (team+position narrows
    the field to a handful of players at most) that it won't accidentally
    match two different people, unlike matching on last name alone."""
    ros = rosters[(rosters["season"] == season) & rosters["player_name"].notna()].copy()
    ros["last_name"] = ros["player_name"].apply(norm_name).str.split().str[-1]
    lookup = ros.groupby(["last_name", "team", "position"])["player_id"].nunique()
    unambiguous_keys = lookup[lookup == 1].index
    ros = ros.set_index(["last_name", "team", "position"]).loc[lambda d: d.index.isin(unambiguous_keys)]
    return ros["player_id"].to_dict()


def build_lastname_pos_to_id(rosters: pd.DataFrame, season: int) -> dict:
    """Third-tier fallback for players who changed teams in the offseason
    (e.g. Kenneth Walker III: SEA in 2025, KC per Clay's 2026 projection) --
    team+position no longer matches, so drop team and require the last
    name+position combo to be unique LEAGUE-WIDE, which is true for the
    large majority of surnames at a given position."""
    ros = rosters[(rosters["season"] == season) & rosters["player_name"].notna()].copy()
    ros["last_name"] = ros["player_name"].apply(norm_name).str.split().str[-1]
    lookup = ros.groupby(["last_name", "position"])["player_id"].nunique()
    unambiguous_keys = lookup[lookup == 1].index
    ros = ros.set_index(["last_name", "position"]).loc[lambda d: d.index.isin(unambiguous_keys)]
    return ros["player_id"].to_dict()


def build_qb_name_to_id(schedules: pd.DataFrame) -> dict:
    reg = schedules[schedules["game_type"] == "REG"]
    h = reg[["home_qb_name", "home_qb_id"]].rename(columns={"home_qb_name": "name", "home_qb_id": "qb_id"})
    a = reg[["away_qb_name", "away_qb_id"]].rename(columns={"away_qb_name": "name", "away_qb_id": "qb_id"})
    both = pd.concat([h, a]).dropna(subset=["name", "qb_id"])
    return both.drop_duplicates("name", keep="last").set_index("name")["qb_id"].to_dict()


def find_qb_changes(schedules: pd.DataFrame, clay_players: pd.DataFrame, prior_season: int = 2025) -> pd.DataFrame:
    """Compare each team's most-used 2025-season starter against Clay's 2026
    projected starter (by last name) to flag likely offseason QB changes.
    NOTE: prior_season is hardcoded to 2025 by every current caller -- this
    whole offseason-bootstrap path is 2026-specific (see predict_2026.py's
    and weekly_update.py's docstrings) and will need a real generalization,
    not just a parameter default bump, before reuse in a future season."""
    reg_prior = schedules[(schedules["season"] == prior_season) & (schedules["game_type"] == "REG")]
    h = reg_prior[["home_team", "home_qb_name"]].rename(columns={"home_team": "team", "home_qb_name": "qb_name"})
    a = reg_prior[["away_team", "away_qb_name"]].rename(columns={"away_team": "team", "away_qb_name": "qb_name"})
    starts = pd.concat([h, a])
    top_prior = (
        starts.groupby(["team", "qb_name"]).size().reset_index(name="starts")
        .sort_values("starts", ascending=False).drop_duplicates("team")
    )
    # sort by pass attempts, not 'games' -- 'games' means games on the active roster (a
    # backup stays at 17 all season too), so it doesn't discriminate the real starter.
    # Attempts does: a true starter is projected for 450-550+, a handcuff/insurance QB for <100.
    clay_qb = clay_players[clay_players["position"] == "QB"].sort_values("pass_att", ascending=False).drop_duplicates(
        "team"
    )
    comp = top_prior.merge(clay_qb[["team", "player"]], on="team", how="outer")
    last_name = lambda s: str(s).split()[-1].lower() if pd.notna(s) else ""
    comp["likely_change"] = comp["qb_name"].apply(last_name) != comp["player"].apply(last_name)
    return comp.rename(columns={"qb_name": f"starter_{prior_season}", "player": "projected_2026"})
