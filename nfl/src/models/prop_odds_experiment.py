"""The prop-odds experiment (review round 2 #4.4, reframed by round 3 #9 as small and
decisive): does the already-validated RB carry-reallocation result
(injury_reallocation.py's `reallocate_shares`, MAE 4.49->3.71, p=0.0003 on real historical
"lead RB ruled Out" instances -- this project's strongest single result) beat a REAL posted
prop line, or only our own prior estimate? That's never been checked. Twenty to thirty
observations (a few weeks of a cheap current-week odds API, ~2-5 qualifying player-weeks a
slate) gives a first read.

Same snapshot-then-reconcile pattern already validated for CLV (weekly_update.py's
`line_snapshots.parquet`, §11.2.1): log our projection against the posted line the moment
both exist (well before kickoff), then join real outcomes back on once games complete. Never
score against a line fetched or reconstructed after the fact.

MOCK-DATA DISCIPLINE: every function here works identically whether `fetch_prop_odds`'s real
API path or its mock path is active -- but a run using mock data must never be reported as a
real result. `run_weekly_snapshot`'s output/prints say plainly which mode produced a given
snapshot (via `data_source` column), and no summary statistic should be computed across a
mix of real and mock rows.
"""

import numpy as np
import pandas as pd

from src.ingest.fetch_prop_odds import fetch_all_rb_prop_odds
from src.ingest.name_matching import norm_name
from src.models.injury_reallocation import out_player_ids_for_team
from src.utils.paths import DATA_PROCESSED

LEAD_BACK_MIN_SHARE = 0.15  # informal "lead or clear co-lead back" cutoff -- see module
# docstring: the original §6.2 validation didn't gate on a specific share threshold (it
# tested the reallocation mechanism generically on every OUT RB instance), but this
# experiment is deliberately scoped to the decision-relevant case the review describes
# ("a lead back was just ruled out"), so a real, adjustable, and stated threshold is used
# rather than silently including every third-string OUT RB.

SNAPSHOT_PATH = DATA_PROCESSED / "prop_odds_snapshots.parquet"


def identify_lead_rb_outs(team: str, current_roster: pd.DataFrame, carry_engine) -> pd.DataFrame:
    """Which of this team's OUT players are a real lead-or-co-lead RB by our own engine's
    pre-injury share (>= LEAD_BACK_MIN_SHARE) -- the population this experiment cares about.
    Returns one row per qualifying OUT player: player_id, player_name, lead_share."""
    out_ids = out_player_ids_for_team(team, current_roster)
    team_rbs = current_roster[(current_roster["team"] == team) & (current_roster["position"] == "RB")]
    rows = []
    for p in team_rbs.itertuples():
        if p.player_id not in out_ids:
            continue
        share = carry_engine.predict(p.player_id)
        if share >= LEAD_BACK_MIN_SHARE:
            rows.append({"team": team, "player_id": p.player_id, "player_name": p.player_name, "lead_share": share})
    return pd.DataFrame(rows)


def target_backup_projections(team: str, props_df: pd.DataFrame) -> pd.DataFrame:
    """This team's backup RBs who actually inherited reallocated carries this week --
    directly from the live pipeline's own props output (weekly_update.py's
    props_{season}_wk{week}.parquet or its in-memory prop_rows), no recomputation. Excludes
    the OUT row itself (status_note=="OUT", proj_carries==0 by construction) and anyone with
    zero projected carries (not actually a reallocation beneficiary)."""
    team_props = props_df[(props_df["team"] == team) & (props_df["position"] == "RB")]
    return team_props[(team_props["status_note"] != "OUT") & (team_props["proj_carries"] > 0)][
        ["team", "player", "proj_carries", "proj_rush_yards", "status_note"]
    ].copy()


def match_projection_to_posted_line(projections: pd.DataFrame, posted_odds: pd.DataFrame) -> pd.DataFrame:
    """Joins our backup-RB projections (keyed by our own `player` display name) to the
    posted-odds table (keyed by the book's `player_name_raw`) via normalized-name matching
    (name_matching.norm_name -- the same exact-normalized-name join used throughout this
    project's own name-matching cascade, e.g. injury_reallocation.py's rookie fallback)."""
    proj = projections.copy()
    proj["name_norm"] = proj["player"].apply(norm_name)
    odds = posted_odds.copy()
    odds["name_norm"] = odds["player_name_raw"].apply(norm_name)
    return proj.merge(odds, on="name_norm", how="inner", suffixes=("", "_odds"))


def our_implied_side(our_rush_yards: float, point: float) -> str:
    """Which side of the posted line our own projection implies betting -- 'over' if our
    number sits above the book's line, 'under' if below, 'push' if exactly equal (should be
    rare with continuous projections)."""
    if our_rush_yards > point:
        return "over"
    if our_rush_yards < point:
        return "under"
    return "push"


def run_weekly_snapshot(team_props_by_team: dict, current_roster: pd.DataFrame, carry_engine,
                         season: int, week: int, market_key: str = "player_rush_yds",
                         api_key: str | None = None) -> pd.DataFrame:
    """Full snapshot step for one week: identify target teams (a real lead RB ruled out),
    pull their backup RBs' reallocated projections, fetch+match the posted prop line, and
    append one row per matched player to prop_odds_snapshots.parquet (append, not overwrite --
    same discipline as line_snapshots.parquet, §11.2.1). `team_props_by_team` maps team ->
    that team's slice of the current week's props DataFrame (weekly_update.py already has
    this available per-team in its props loop).

    Returns the newly-appended rows (empty if no qualifying team this week -- expected most
    weeks, since a real lead-RB-out instance is rare by design, ~2-5 player-weeks/slate per
    the review's own estimate)."""
    import os

    data_source = "real" if os.environ.get("ODDS_API_KEY") else "MOCK"
    posted_odds = fetch_all_rb_prop_odds(market_key=market_key, api_key=api_key)

    rows = []
    for team, props_df in team_props_by_team.items():
        leads_out = identify_lead_rb_outs(team, current_roster, carry_engine)
        if leads_out.empty:
            continue
        backups = target_backup_projections(team, props_df)
        if backups.empty:
            continue
        matched = match_projection_to_posted_line(backups, posted_odds)
        for r in matched.itertuples():
            rows.append({
                "season": season, "week": week, "team": team,
                "backup_player": r.player, "proj_rush_yards": r.proj_rush_yards,
                "posted_point": r.point, "over_price": r.over_price, "under_price": r.under_price,
                "our_side": our_implied_side(r.proj_rush_yards, r.point),
                "bookmaker": r.bookmaker, "data_source": data_source,
            })
    new_rows = pd.DataFrame(rows)
    if new_rows.empty:
        print(f"prop-odds experiment: no qualifying lead-RB-out instances this week ({season} wk{week}) -- nothing to snapshot")
        return new_rows

    if SNAPSHOT_PATH.exists():
        existing = pd.read_parquet(SNAPSHOT_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_parquet(SNAPSHOT_PATH, index=False)
    print(f"prop-odds experiment: snapshotted {len(new_rows)} matched player-week(s) "
          f"[{data_source}] ({len(combined)} total in log)")
    return new_rows


def reconcile_and_score(weekly_stats: pd.DataFrame) -> dict:
    """Joins prop_odds_snapshots.parquet against real weekly_player_stats (rushing_yards) for
    any completed game, and reports win rate (did our_side match the actual over/under
    outcome?) -- ONLY on rows where data_source=='real'. Mock rows are excluded from any
    reported statistic; they exist purely to exercise this code path, not to produce a number."""
    if not SNAPSHOT_PATH.exists():
        print("no prop_odds_snapshots.parquet yet -- run run_weekly_snapshot first")
        return {}
    snap = pd.read_parquet(SNAPSHOT_PATH)
    real_snap = snap[snap["data_source"] == "real"]
    if real_snap.empty:
        print(f"prop-odds experiment: {len(snap)} MOCK row(s) logged, 0 REAL rows yet -- "
              f"nothing to score. (Mock rows are for exercising the pipeline only.)")
        return {"n_real": 0, "n_mock": len(snap)}

    snap_named = real_snap.copy()
    snap_named["name_norm"] = snap_named["backup_player"].apply(norm_name)
    ws = weekly_stats.copy()
    ws["name_norm"] = ws["player_display_name"].apply(norm_name) if "player_display_name" in ws.columns else ws["player_name"].apply(norm_name)
    merged = snap_named.merge(
        ws[["season", "week", "name_norm", "rushing_yards"]], on=["season", "week", "name_norm"], how="left"
    ).dropna(subset=["rushing_yards"])
    if merged.empty:
        print("no completed games yet to reconcile against")
        return {"n_real": len(real_snap), "n_reconciled": 0}

    merged["actual_side"] = np.where(merged["rushing_yards"] > merged["posted_point"], "over",
                             np.where(merged["rushing_yards"] < merged["posted_point"], "under", "push"))
    decided = merged[merged["actual_side"] != "push"]
    win_rate = (decided["our_side"] == decided["actual_side"]).mean() if len(decided) else float("nan")
    print(f"prop-odds experiment: {len(decided)} reconciled real player-week(s), "
          f"win rate={win_rate:.3f}" if len(decided) else "no decided (non-push) rows yet")
    return {"n_real": len(real_snap), "n_reconciled": len(decided), "win_rate": win_rate}
