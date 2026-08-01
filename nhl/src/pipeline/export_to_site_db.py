"""Read this pipeline's own latest daily output and upsert it into the
combined multi-sport site's shared Postgres tables (site/db/schema.sql).

This is the ONLY place NHL's model touches the site -- it reads
already-written parquet files, never model internals, and the site never
imports NHL code back. Run after generate_predictions.py for the same
date; intended to be chained in the same Railway cron command, e.g.
`python -m src.pipeline.generate_predictions && python -m
src.pipeline.export_to_site_db`.

No props table population here: this project has no player-props
pipeline (confirmed -- no props module/function anywhere in src/), so the
site's props table simply stays empty for NHL, matching the plan.

Reads the daily_predictions_{date}.parquet "convenience snapshot", not the
canonical hash-chained live_prediction_log.jsonl -- the parquet already has
every field this export needs (home/away team, lambda_home/lambda_away,
home_win_prob_full, gameId/gameType) and is far cheaper to read than
replaying the JSONL's per-game embedded ~100-row injury snapshots.
"""

import datetime as _dt
import os
import sys

import pandas as pd
import psycopg2
import psycopg2.extras

from src.utils.paths import DATA_PROCESSED

SPORT = "nhl"

GAME_EXTRA_COLUMNS = ["home_starter_goalie_id", "away_starter_goalie_id", "gameType"]


def export(target_date: str, database_url: str) -> None:
    pred_path = DATA_PROCESSED / f"daily_predictions_{target_date}.parquet"
    if not pred_path.exists():
        print(f"no daily_predictions file for {target_date} at {pred_path} -- nothing to export", flush=True)
        return

    predictions = pd.read_parquet(pred_path)
    games = []
    for _, row in predictions.iterrows():
        home_prob = row.get("home_win_prob_full")
        games.append({
            "game_id": str(row["gameId"]), "home_team": row["home_team"], "away_team": row["away_team"],
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "home_win_prob": home_prob, "away_win_prob": (1 - home_prob) if pd.notna(home_prob) else None,
            "projected_home_score": row["lambda_home"], "projected_away_score": row["lambda_away"],
            "projected_total": row["lambda_home"] + row["lambda_away"],
            "extra": {c: row[c] for c in GAME_EXTRA_COLUMNS if c in predictions.columns},
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
            cur.execute(
                """
                INSERT INTO runs (sport, slate_key, run_at, status, notes)
                VALUES (%s, %s, now(), %s, %s)
                ON CONFLICT (sport, slate_key) DO UPDATE SET
                    run_at = now(), status = EXCLUDED.status, notes = EXCLUDED.notes
                """,
                (SPORT, target_date, "ok", f"{len(games)} games, no props pipeline for this sport"),
            )
        print(f"exported {len(games)} games for {target_date}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else str(_dt.date.today())
    database_url = os.environ["DATABASE_URL"]
    export(target_date, database_url)
