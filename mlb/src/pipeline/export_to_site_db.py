"""Read this pipeline's own latest daily output and upsert it into the
combined multi-sport site's shared Postgres tables (site/db/schema.sql).

This is the ONLY place MLB's model touches the site -- it reads
already-written parquet files, never model internals, and the site never
imports MLB code back. Run after generate_daily_props.py for the same
date (that's what writes daily_props_{date}.parquet /
batter_props_{date}.parquet / pitcher_props_{date}.parquet in the first
place); intended to be chained in the same Railway cron command, e.g.
`python -m src.pipeline.generate_daily_props && python -m
src.pipeline.export_to_site_db`.

Known gap (not solved here, scoped for a later pass if it matters): batter/
pitcher prop rows don't carry a team column -- generate_game_props's
batter_props/pitcher_props output has no team field to read, only
game_pk/batter_id/pitcher_id, so `team` is left NULL in the props table.
The site can still group props by game (game_id) without it.
"""

import datetime as _dt
import json
import os
import sys

import pandas as pd
import psycopg2
import psycopg2.extras
import pybaseball as pb
import requests

from src.utils.paths import DATA_PROCESSED
from src.utils.tz import default_slate_date

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

SPORT = "mlb"

# column -> (display market name, kind); "prob" columns populate over_prob,
# "mean" columns populate proj_mean. Matches the real column names
# _batter_props/_pitcher_props produce (src/models/props.py).
BATTER_MARKETS = {
    "p_1plus_hit": ("1+ Hit", "prob"),
    "p_2plus_hits": ("2+ Hits", "prob"),
    "p_1plus_hr": ("1+ HR", "prob"),
    "p_1plus_bb": ("1+ BB", "prob"),
    "p_1plus_rbi": ("1+ RBI", "prob"),
    "mean_total_bases": ("Total Bases", "mean"),
    "mean_hits": ("Hits", "mean"),
    # "Batter Strikeouts" (not just "Strikeouts") -- a two-way player (e.g.
    # Ohtani) can have both a batter AND a pitcher row for the same game,
    # and props' primary key is (..., player_id, market) with no role
    # column, so an identical market NAME for both would silently collide
    # on ON CONFLICT and drop one row. Confirmed against real 2026-07-25
    # data: 6 real two-way-appearance collisions before this rename.
    "mean_k": ("Batter Strikeouts", "mean"),
}
PITCHER_MARKETS = {
    "mean_k": ("Strikeouts", "mean"),
    "mean_bb": ("Walks Allowed", "mean"),
    "mean_hits_allowed": ("Hits Allowed", "mean"),
    "mean_runs_allowed": ("Runs Allowed", "mean"),
    "mean_batters_faced": ("Batters Faced", "mean"),
    "p_6plus_k": ("6+ K", "prob"),
}

GAME_EXTRA_COLUMNS = [
    "p_over_8.5", "p_under_8.5", "home_covers_minus_1_5", "away_covers_plus_1_5",
    "real_weather", "postseason", "used_debut_fallback",
    "home_lineup_source", "away_lineup_source",
]


def _statsapi_name(player_id: int) -> str | None:
    """Live MLB Stats API lookup for one player -- fallback for ids the
    Chadwick register hasn't caught up with yet (e.g. very recent debuts).
    Only called for the handful of ids the bulk lookup below misses, so one
    request per id per day is negligible. Returns None on any failure
    (network, 404, unexpected shape) so the caller can fall back further
    rather than the whole export erroring."""
    try:
        resp = requests.get(f"{MLB_API_BASE}/people/{player_id}", timeout=5)
        resp.raise_for_status()
        people = resp.json().get("people", [])
        return people[0]["fullName"] if people else None
    except Exception:
        return None


def _player_names(player_ids: list[int]) -> dict[int, str]:
    """MLBAM id -> "First Last" via pybaseball's Chadwick register lookup
    (cached locally after first call), with a live MLB Stats API fallback
    for anything the register doesn't resolve (e.g. very recent debuts) --
    falls back to the id itself only if that also fails, rather than
    failing the whole export."""
    if not player_ids:
        return {}
    lookup = pb.playerid_reverse_lookup(player_ids, key_type="mlbam")
    names = {
        int(row.key_mlbam): f"{row.name_first.title()} {row.name_last.title()}"
        for row in lookup.itertuples()
    }
    unresolved = [pid for pid in player_ids if pid not in names]
    for pid in unresolved:
        name = _statsapi_name(pid)
        if name:
            names[pid] = name
    return {pid: names.get(pid, str(pid)) for pid in player_ids}


def _prop_rows(df: pd.DataFrame, id_col: str, market_map: dict, role: str,
               name_lookup: dict[int, str]) -> list[dict]:
    rows = []
    for row in df.itertuples():
        raw_id = getattr(row, id_col)
        player_id = str(raw_id)
        player_name = name_lookup.get(int(raw_id), player_id) if str(raw_id).isdigit() else player_id
        extra_common = {"role": role, "pa_per_game": getattr(row, "pa_per_game", None)}
        for col, (market, kind) in market_map.items():
            value = getattr(row, col, None)
            if value is None or pd.isna(value):
                continue
            rows.append({
                "game_id": str(row.game_pk), "player_id": player_id, "player_name": player_name,
                "team": None, "market": market,
                "over_prob": value if kind == "prob" else None,
                "proj_mean": value if kind == "mean" else None,
                "extra": extra_common,
            })
    return rows


def export(target_date: str, database_url: str) -> None:
    daily_path = DATA_PROCESSED / f"daily_props_{target_date}.parquet"
    if not daily_path.exists():
        print(f"no daily_props file for {target_date} at {daily_path} -- nothing to export", flush=True)
        return

    summary = pd.read_parquet(daily_path)
    games = []
    for _, row in summary.iterrows():
        away_team, home_team = row["matchup"].split(" @ ")
        # "debug" is stored in the parquet as a JSON string (see
        # generate_daily_props.py's own comment on why -- pyarrow struct
        # inference breaks across games whose factor dicts don't all share
        # identical nested shapes), so it needs parsing back here rather
        # than a plain column select like GAME_EXTRA_COLUMNS. Missing/older
        # parquet files without this column (pre-dating this feature) fall
        # back to {} -- same graceful-degradation convention as extra.
        debug_raw = row.get("debug") if "debug" in summary.columns else None
        games.append({
            "game_id": str(row["game_pk"]), "home_team": home_team, "away_team": away_team,
            "matchup": row["matchup"], "home_win_prob": row["home_win_prob"],
            "away_win_prob": row["away_win_prob"], "projected_home_score": row["mean_home_score"],
            "projected_away_score": row["mean_away_score"], "projected_total": row["mean_total"],
            "extra": {c: row[c] for c in GAME_EXTRA_COLUMNS if c in summary.columns},
            "debug": json.loads(debug_raw) if isinstance(debug_raw, str) else {},
        })

    batter_path = DATA_PROCESSED / f"batter_props_{target_date}.parquet"
    pitcher_path = DATA_PROCESSED / f"pitcher_props_{target_date}.parquet"
    bp = pd.read_parquet(batter_path) if batter_path.exists() else pd.DataFrame()
    pp = pd.read_parquet(pitcher_path) if pitcher_path.exists() else pd.DataFrame()

    # position-player-pitching placeholder ids ("TEAM_POSITION_PLAYER", see
    # generate_daily_props.py) aren't real MLBAM ids -- exclude them from the
    # reverse-lookup batch, _prop_rows falls back to the raw string for them.
    all_ids = (
        bp["batter_id"].tolist() if not bp.empty else []
    ) + (
        pp.loc[pp["pitcher_id"].str.isdigit(), "pitcher_id"].astype(int).tolist() if not pp.empty else []
    )
    name_lookup = _player_names(sorted(set(all_ids)))

    props = []
    if not bp.empty:
        props += _prop_rows(bp, "batter_id", BATTER_MARKETS, "batter", name_lookup)
    if not pp.empty:
        props += _prop_rows(pp, "pitcher_id", PITCHER_MARKETS, "pitcher", name_lookup)

    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            for g in games:
                cur.execute(
                    """
                    INSERT INTO games (sport, slate_key, game_id, home_team, away_team, matchup,
                                        home_win_prob, away_win_prob, projected_home_score,
                                        projected_away_score, projected_total, extra, debug, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (sport, slate_key, game_id) DO UPDATE SET
                        home_team = EXCLUDED.home_team, away_team = EXCLUDED.away_team,
                        matchup = EXCLUDED.matchup, home_win_prob = EXCLUDED.home_win_prob,
                        away_win_prob = EXCLUDED.away_win_prob,
                        projected_home_score = EXCLUDED.projected_home_score,
                        projected_away_score = EXCLUDED.projected_away_score,
                        projected_total = EXCLUDED.projected_total,
                        extra = EXCLUDED.extra, debug = EXCLUDED.debug, updated_at = now()
                    """,
                    (SPORT, target_date, g["game_id"], g["home_team"], g["away_team"], g["matchup"],
                     g["home_win_prob"], g["away_win_prob"], g["projected_home_score"],
                     g["projected_away_score"], g["projected_total"],
                     psycopg2.extras.Json(g["extra"]), psycopg2.extras.Json(g["debug"])),
                )
            for p in props:
                cur.execute(
                    """
                    INSERT INTO props (sport, slate_key, game_id, player_id, player_name, team,
                                        market, line, over_prob, under_prob, proj_mean, extra, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (sport, slate_key, game_id, player_id, market) DO UPDATE SET
                        player_name = EXCLUDED.player_name, team = EXCLUDED.team,
                        line = EXCLUDED.line, over_prob = EXCLUDED.over_prob,
                        under_prob = EXCLUDED.under_prob, proj_mean = EXCLUDED.proj_mean,
                        extra = EXCLUDED.extra, updated_at = now()
                    """,
                    (SPORT, target_date, p["game_id"], p["player_id"], p["player_name"], p["team"],
                     p["market"], None, p["over_prob"], None, p["proj_mean"],
                     psycopg2.extras.Json(p["extra"])),
                )
            cur.execute(
                """
                INSERT INTO runs (sport, slate_key, run_at, status, notes)
                VALUES (%s, %s, now(), %s, %s)
                ON CONFLICT (sport, slate_key) DO UPDATE SET
                    run_at = now(), status = EXCLUDED.status, notes = EXCLUDED.notes
                """,
                (SPORT, target_date, "ok", f"{len(games)} games, {len(props)} prop rows"),
            )
        print(f"exported {len(games)} games, {len(props)} prop rows for {target_date}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else str(default_slate_date())
    database_url = os.environ["DATABASE_URL"]
    export(target_date, database_url)
