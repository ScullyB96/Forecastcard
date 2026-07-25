"""Detect players who won't play this week and reallocate their projected
usage to healthy teammates -- validated against real historical instances,
not assumed.

Two independent detection sources, since neither alone is enough:
  1. Real roster status ("ACT"/"PUP"/"RES"/"SUS"/... from weekly_rosters,
     already fetched every pipeline run but never used for this). This is
     the automated, no-manual-intervention path -- but it only reflects
     nfl_data_py's own data lag. A season's roster snapshot (PUP lists,
     final cuts) typically isn't published until right before Week 1, so
     this alone can't catch a fresh transaction the moment it happens.
  2. A manual override file (data/manual_overrides/known_outs_2026.json)
     for verified breaking news the automated data hasn't caught up to yet
     -- the same stopgap pattern already used for the Clay-PDF offseason
     QB-swap bootstrap elsewhere in this project. Entries here should be
     web-verified before adding, and removed once real data catches up
     (confirmed via weekly_rosters status or the player actually missing
     real games in weekly_player_stats).

Reallocation is validated POSITION-SPECIFIC, not a blanket rule -- tested
against real historical "season-long lead back/receiver ruled Out" instances
(injury reports, 2016-2025):
  - RB carries: proportional reallocation among remaining active RBs is a
    real, significant improvement over leaving teammates' shares untouched
    (MAE 4.49 -> 3.71, ~17% better, p=0.0003, n=443 backup-week observations).
    Bell-cow succession is real: a clear next-man-up typically absorbs the
    vacated workload.
  - WR/TE targets: the SAME reallocation makes predictions WORSE (MAE 1.93
    -> 2.32, p=0.0001, n=513) -- vacated targets don't redistribute cleanly
    across a receiving corps the way carries do. For this group, simply
    excluding the injured player (current model behavior) is already the
    validated-better choice; do NOT reallocate.
"""

import json

import pandas as pd

from src.utils.paths import PROJECT_ROOT

INACTIVE_STATUSES = {"PUP", "RES", "SUS", "CUT", "INA", "RET", "DEV", "EXE", "RSN"}
MANUAL_OVERRIDES_PATH = PROJECT_ROOT / "data" / "manual_overrides" / "known_outs_2026.json"


def load_manual_overrides() -> list[dict]:
    if not MANUAL_OVERRIDES_PATH.exists():
        return []
    with open(MANUAL_OVERRIDES_PATH) as f:
        return json.load(f)


def out_player_ids_for_team(team: str, current_roster: pd.DataFrame) -> set[str]:
    """current_roster must include a 'status' column (from weekly_rosters) and
    'team'/'player_id'/'player_name'. Combines real roster status with any
    verified manual override for this team."""
    team_roster = current_roster[current_roster["team"] == team]
    out_ids = set(team_roster.loc[team_roster["status"].isin(INACTIVE_STATUSES), "player_id"])

    for override in load_manual_overrides():
        if override["team"] != team:
            continue
        match = team_roster[team_roster["player_name"] == override["player_name"]]
        if not match.empty:
            out_ids.add(match["player_id"].iloc[0])

    return out_ids


def reallocate_shares(raw_shares: dict[str, float], out_keys: set[str]) -> dict[str, float]:
    """Generic version: given every RB's raw share (real-history players via
    their own engine, PLUS any rookie/no-history fallback entries already
    folded in by the caller -- see rookie_fallback_rb_shares below) and which
    keys are ruled out, proportionally renormalize the active remainder so
    the vacated share gets redistributed rather than left unclaimed.
    Validated -- see module docstring. Only use for RB carry share; WR/TE
    targets should just exclude the out player with no reallocation."""
    active = {k: v for k, v in raw_shares.items() if k not in out_keys}
    total = sum(raw_shares.values())
    active_total = sum(active.values())
    if active_total <= 0 or total <= 0:
        return active
    scale = total / active_total
    return {k: v * scale for k, v in active.items()}


def rookie_fallback_rb_rates(
    team: str,
    clay_players: pd.DataFrame,
    name_to_id: dict,
    lastname_team_pos_to_id: dict,
    lastname_pos_to_id: dict,
    carry_engine,
    rush_attempts: float,
    league_ypc: float,
    league_rush_td_rate: float,
) -> dict[str, dict]:
    """Clay-projected RBs for this team with NO real NFL history in our own
    engines. Triggers on "zero real engine history" (carry_engine.predict()
    == 0), not "absent from the roster snapshot" -- a rookie can be visible
    in weekly_rosters (nflverse's roster data can update mid-week as real
    transactions get published) with zero actual career touches, which is
    the more common case in practice, not the exception. Same fallback
    pattern already validated in predict_props_2026.py, ported here since it
    was never wired into the live weekly pipeline. Keyed by a synthetic
    "clay:<name>" string so callers can fold the 'share' value directly into
    reallocate_shares' raw_shares pool, then use the other fields to build
    the full prop row. The normal per-roster-row loop elsewhere already
    skips anyone with zero share on both targets and carries, so this never
    produces a duplicate row for the same player."""
    from src.models.predict_props_2026 import norm_name

    team_players = clay_players[(clay_players["team"] == team) & (clay_players["position"] == "RB")]
    out = {}
    for p in team_players.itertuples():
        name_key = norm_name(p.player)
        pid = name_to_id.get(name_key)
        if pid is None:
            last_name = name_key.split()[-1] if name_key else None
            pid = lastname_team_pos_to_id.get((last_name, team, "RB"))
        if pid is None:
            pid = lastname_pos_to_id.get((last_name, "RB"))
        has_history = pid is not None and carry_engine.predict(pid) > 0
        if has_history or rush_attempts <= 0 or p.games <= 0:
            continue
        out[f"clay:{p.player}"] = {
            "player_name": p.player,
            "share": (p.rush_att / max(p.games, 1)) / rush_attempts,
            "ypc_rate": (p.rush_yds / p.rush_att) if p.rush_att > 0 else league_ypc,
            "rush_td_rate": (p.rush_td / p.rush_att) if p.rush_att > 0 else league_rush_td_rate,
        }
    return out
