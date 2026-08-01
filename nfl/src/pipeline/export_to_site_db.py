"""Read this pipeline's own latest weekly output and upsert it into the
combined multi-sport site's shared Postgres tables (site/db/schema.sql).

This is the ONLY place NFL's model touches the site -- it reads
already-written parquet files, never model internals, and the site never
imports NFL code back. Run after weekly_update.py for the same
season/week; intended to be chained in the same Railway cron command,
e.g. `python -m src.pipeline.weekly_update && python -m
src.pipeline.export_to_site_db 2026 1`.

Known gap (not solved here, scoped for a later pass if it matters):
props_{season}_wk{week}.parquet has no player_id column -- weekly_update.py
only ever writes the player's display name, not their gsis_id, into this
output (confirmed real duplicate names exist across different teams in the
same week, e.g. two different "Geno Smith"s). Since the props table's
primary key needs a stable player_id, this script derives one as
f"{team}:{player}" (team+name is unique within a single week's roster
snapshot, per the survey that found this gap) -- stable across re-runs of
the SAME week, but not a real cross-season player identity.
"""

import os
import sys

import pandas as pd
import psycopg2
import psycopg2.extras

from src.utils.paths import DATA_PROCESSED

SPORT = "nfl"


def _json_safe(d: dict) -> dict:
    """NaN (real -- e.g. `wind` before a game's weather is known, or any
    sim_* column when no market line has posted yet, see module docstring)
    serializes as the bare token `NaN`, which is not valid JSON -- Postgres'
    jsonb column rejects it outright. Replace with None (SQL NULL)."""
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in d.items()}

GAME_EXTRA_COLUMNS = [
    "market_spread", "market_total", "sim_margin_std", "sim_home_moneyline",
    "sim_away_moneyline", "sim_total_std", "home_cb_flag", "away_cb_flag",
    "qb_swap_flag", "wind", "roof",
]

# column -> display market name. All "mean" (pure projections) except
# td_probability, which is the one real over/under-style prop this
# pipeline emits (an anytime-TD probability).
PROP_MARKETS = {
    "proj_pass_attempts": "Pass Attempts", "proj_completions": "Completions",
    "proj_passing_yards": "Passing Yards", "proj_passing_tds": "Passing TDs",
    "proj_interceptions": "Interceptions", "proj_targets": "Targets",
    "proj_carries": "Carries", "proj_receptions": "Receptions",
    "proj_rec_yards": "Receiving Yards", "proj_rush_yards": "Rushing Yards",
}


def _games_rows(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in predictions.iterrows():
        home_prob = row.get("sim_home_win_prob")
        rows.append({
            "game_id": row["game_id"], "home_team": row["home"], "away_team": row["away"],
            "matchup": f"{row['away']} @ {row['home']}",
            "home_win_prob": home_prob, "away_win_prob": (1 - home_prob) if pd.notna(home_prob) else None,
            "projected_home_score": row["our_home_pts"], "projected_away_score": row["our_away_pts"],
            "projected_total": row["our_total"],
            "extra": _json_safe({c: row[c] for c in GAME_EXTRA_COLUMNS if c in predictions.columns}),
        })
    return rows


def _prop_rows(props: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in props.iterrows():
        player_id = f"{row['team']}:{row['player']}"
        extra_common = {"position": row.get("position"), "status_note": row.get("status_note") or None}
        for col, market in PROP_MARKETS.items():
            value = row.get(col)
            # 0.0 for these columns means "not this player's stat side"
            # (a QB's proj_targets, a WR's proj_pass_attempts, etc. -- see
            # module docstring's gotcha #3) rather than a real projection
            # of zero, so skip rather than show a meaningless "0" market.
            if value is None or pd.isna(value) or value == 0.0:
                continue
            rows.append({
                "game_id": row["game_id"], "player_id": player_id, "player_name": row["player"],
                "team": row["team"], "market": market, "over_prob": None, "proj_mean": value,
                "extra": extra_common,
            })
        td_prob = row.get("td_probability")
        if td_prob is not None and pd.notna(td_prob):
            rows.append({
                "game_id": row["game_id"], "player_id": player_id, "player_name": row["player"],
                "team": row["team"], "market": "Anytime TD", "over_prob": td_prob, "proj_mean": None,
                "extra": extra_common,
            })
    return rows


def export(season: int, week: int, database_url: str) -> None:
    slate_key = f"{season}-wk{week}"
    pred_path = DATA_PROCESSED / f"predictions_{season}_wk{week}.parquet"
    if not pred_path.exists():
        print(f"no predictions file for {slate_key} at {pred_path} -- nothing to export", flush=True)
        return

    predictions = pd.read_parquet(pred_path)
    games = _games_rows(predictions)

    props_path = DATA_PROCESSED / f"props_{season}_wk{week}.parquet"
    props = _prop_rows(pd.read_parquet(props_path)) if props_path.exists() else []

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
                    (SPORT, slate_key, g["game_id"], g["home_team"], g["away_team"], g["matchup"],
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
                        line = EXCLUDED.line, over_prob = EXCLUDED.over_prob,
                        under_prob = EXCLUDED.under_prob, proj_mean = EXCLUDED.proj_mean,
                        extra = EXCLUDED.extra, updated_at = now()
                    """,
                    (SPORT, slate_key, p["game_id"], p["player_id"], p["player_name"], p["team"],
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
                (SPORT, slate_key, "ok", f"{len(games)} games, {len(props)} prop rows"),
            )
        print(f"exported {len(games)} games, {len(props)} prop rows for {slate_key}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m src.pipeline.export_to_site_db SEASON WEEK -- "
            "pass the same season/week weekly_update.py just predicted (its own "
            "current_nfl_season()/find_next_week() logic needs real schedule data "
            "this script has no reason to duplicate)"
        )
    season, week = int(sys.argv[1]), int(sys.argv[2])
    database_url = os.environ["DATABASE_URL"]
    export(season, week, database_url)
