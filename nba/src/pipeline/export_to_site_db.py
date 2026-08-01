"""Read this pipeline's own latest daily output and upsert it into the
combined multi-sport site's shared Postgres tables (site/db/schema.sql).

This is the ONLY place NBA's model touches the site -- it reads
already-written parquet files, never model internals, and the site never
imports NBA code back. Run after generate_predictions.py + generate_props.py
for the same date; intended to be chained in the same Railway cron command,
e.g. `python -m src.pipeline.generate_predictions && python -m
src.pipeline.generate_props && python -m src.pipeline.export_to_site_db`.

Schema-drift note: daily_props_{date}.parquet has gained columns over this
project's history (proj_var/family/family_param were added at some point --
older on-disk files don't have them). This script reads every optional
column via `.get(col)`-style checks rather than assuming they exist.

No fixed betting line exists in either output (this pipeline emits pure
mean/variance projections, not over/under probabilities against a line) --
over_prob/under_prob/line are left NULL in the props table; proj_mean is
the only populated numeric field per prop row.
"""

import datetime as _dt
import json
import os
import sys

import pandas as pd
import psycopg2
import psycopg2.extras
from nba_api.stats.static import players as _static_players

from src.utils.paths import DATA_PROCESSED

SPORT = "nba"

GAME_EXTRA_COLUMNS = ["projected_pace", "spread_80pct_lo", "spread_80pct_hi", "lineup_source"]
PROP_EXTRA_COLUMNS = ["opponent", "family", "family_param", "anchored_to_team_total", "matchup_adjusted",
                      "minutes_tier_tag"]

# stat_name -> display market name (the raw column values are already a
# clean, real box-score vocabulary -- just title-cased for display).
MARKET_NAMES = {
    "points": "Points", "2pt_made": "2PT Made", "3pt_made": "3PT Made", "ft_made": "FT Made",
    "oreb": "Offensive Rebounds", "dreb": "Defensive Rebounds", "ast": "Assists",
    "tov": "Turnovers", "stl": "Steals", "blk": "Blocks",
}

_NAME_CACHE: dict[int, str] = {}


def _player_name(player_id: int) -> str:
    """nba_api PERSON_ID -> full name via the static player list (no HTTP
    call, same list used by src/ingest/player_name_crosswalk.py for the
    reverse direction). Falls back to the id itself for anything not found
    rather than failing the whole export."""
    if player_id in _NAME_CACHE:
        return _NAME_CACHE[player_id]
    info = _static_players.find_player_by_id(player_id)
    name = info["full_name"] if info else str(player_id)
    _NAME_CACHE[player_id] = name
    return name


def export(target_date: str, database_url: str) -> None:
    pred_path = DATA_PROCESSED / f"daily_predictions_{target_date}.parquet"
    if not pred_path.exists():
        print(f"no daily_predictions file for {target_date} at {pred_path} -- nothing to export", flush=True)
        return

    predictions = pd.read_parquet(pred_path)
    games = []
    for _, row in predictions.iterrows():
        home_prob = row.get("home_win_prob")
        games.append({
            "game_id": str(row["gameId"]), "home_team": row["homeAbbrev"], "away_team": row["awayAbbrev"],
            "matchup": f"{row['awayAbbrev']} @ {row['homeAbbrev']}",
            "home_win_prob": home_prob, "away_win_prob": (1 - home_prob) if pd.notna(home_prob) else None,
            "projected_home_score": row["pred_home"], "projected_away_score": row["pred_away"],
            "projected_total": row["pred_home"] + row["pred_away"],
            "extra": {c: row[c] for c in GAME_EXTRA_COLUMNS if c in predictions.columns},
        })

    props_path = DATA_PROCESSED / f"daily_props_{target_date}.parquet"
    props = []
    if props_path.exists():
        pp = pd.read_parquet(props_path)
        for _, row in pp.iterrows():
            player_id = int(row["playerId"])
            market = MARKET_NAMES.get(row["stat_name"], row["stat_name"])
            extra = {c: row[c] for c in PROP_EXTRA_COLUMNS if c in pp.columns}
            if "family_param" in extra and isinstance(extra["family_param"], str):
                try:
                    extra["family_param"] = json.loads(extra["family_param"])
                except (ValueError, TypeError):
                    pass
            props.append({
                "game_id": str(row["gameId"]), "player_id": str(player_id),
                "player_name": _player_name(player_id), "team": row["team"], "market": market,
                "proj_mean": row["proj_mean"], "extra": extra,
            })

    conn = psycopg2.connect(database_url)
    try:
        with conn, conn.cursor() as cur:
            for g in games:
                cur.execute(
                    """
                    INSERT INTO games (sport, slate_key, game_id, home_team, away_team, matchup,
                                        home_win_prob, away_win_prob, projected_home_score,
                                        projected_away_score, projected_total, extra, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (sport, slate_key, game_id) DO UPDATE SET
                        home_team = EXCLUDED.home_team, away_team = EXCLUDED.away_team,
                        matchup = EXCLUDED.matchup, home_win_prob = EXCLUDED.home_win_prob,
                        away_win_prob = EXCLUDED.away_win_prob,
                        projected_home_score = EXCLUDED.projected_home_score,
                        projected_away_score = EXCLUDED.projected_away_score,
                        projected_total = EXCLUDED.projected_total,
                        extra = EXCLUDED.extra, updated_at = now()
                    """,
                    (SPORT, target_date, g["game_id"], g["home_team"], g["away_team"], g["matchup"],
                     g["home_win_prob"], g["away_win_prob"], g["projected_home_score"],
                     g["projected_away_score"], g["projected_total"],
                     psycopg2.extras.Json(g["extra"])),
                )
            for p in props:
                cur.execute(
                    """
                    INSERT INTO props (sport, slate_key, game_id, player_id, player_name, team,
                                        market, line, over_prob, under_prob, proj_mean, extra, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (sport, slate_key, game_id, player_id, market) DO UPDATE SET
                        player_name = EXCLUDED.player_name, team = EXCLUDED.team,
                        proj_mean = EXCLUDED.proj_mean, extra = EXCLUDED.extra, updated_at = now()
                    """,
                    (SPORT, target_date, p["game_id"], p["player_id"], p["player_name"], p["team"],
                     p["market"], None, None, None, p["proj_mean"], psycopg2.extras.Json(p["extra"])),
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
    target_date = sys.argv[1] if len(sys.argv) > 1 else str(_dt.date.today())
    database_url = os.environ["DATABASE_URL"]
    export(target_date, database_url)
