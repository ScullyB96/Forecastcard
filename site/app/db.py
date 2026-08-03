"""Thin psycopg2 query layer against the shared schema (db/schema.sql).

No ORM -- this project's sport pipelines are all plain-pandas/plain-SQL by
convention (see e.g. mlb/src/pipeline/export_to_site_db.py), and the query
surface here is small enough that an ORM would add indirection without
buying anything.
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

SPORTS = ["mlb", "nfl", "nba", "nhl"]


@contextmanager
def _cursor():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
    finally:
        conn.close()


def latest_slate_key(sport: str) -> str | None:
    """The most recently-run slate for a sport, per its `runs` row -- lets
    every route show "whatever the latest run produced" without the site
    needing its own per-sport notion of "today"/"this week" (NFL's
    "{season}-wk{week}" slate_key isn't a date, so a generic date-based
    default wouldn't work for it anyway).

    Only considers slates with at least one real game row. A dual-run
    cron can create a `runs` row for tomorrow (the night-before pass)
    hours before that day's schedule/lineups are confirmed and real
    games get exported -- without this filter, the site would default to
    that still-empty slate the moment its run_at became the newest,
    which is exactly the "predictions are gone" symptom this project's
    timezone fix was built to eliminate."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT r.slate_key FROM runs r
            WHERE r.sport = %s AND EXISTS (
                SELECT 1 FROM games g WHERE g.sport = r.sport AND g.slate_key = r.slate_key
            )
            ORDER BY r.run_at DESC LIMIT 1
            """,
            (sport,),
        )
        row = cur.fetchone()
        return row["slate_key"] if row else None


def games_for_slate(sport: str, slate_key: str) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            """
            SELECT * FROM games WHERE sport = %s AND slate_key = %s
            ORDER BY updated_at DESC, game_id
            """,
            (sport, slate_key),
        )
        return cur.fetchall()


def props_for_game(sport: str, slate_key: str, game_id: str) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            """
            SELECT * FROM props WHERE sport = %s AND slate_key = %s AND game_id = %s
            ORDER BY player_name, market
            """,
            (sport, slate_key, game_id),
        )
        return cur.fetchall()


def props_for_slate(sport: str, slate_key: str) -> list[dict]:
    """Every prop row for the whole slate in one query -- powers both the
    per-game panels (grouped by game_id in Python, replacing what used to
    be one props_for_game query per game) and the slate-wide player
    leaderboard, off the same fetch."""
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM props WHERE sport = %s AND slate_key = %s ORDER BY player_name, market",
            (sport, slate_key),
        )
        return cur.fetchall()


def slate_keys_for_sport(sport: str) -> list[str]:
    """Every slate_key with a real run AND at least one real game for
    this sport, newest first -- powers the history/browse picker on the
    sport page. Every past date's games/props rows already persist
    forever (slate_key is part of each table's primary key, so a new
    date's export can only ever collide with rows sharing that same
    slate_key) -- this just needs to enumerate which ones exist.

    Excludes a slate the pipeline has touched but not yet populated
    (e.g. tomorrow's night-before run before schedule/lineups post) --
    not a real, browsable slate yet, so it shouldn't appear as a pickable
    option until an actual export lands."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT r.slate_key FROM runs r
            WHERE r.sport = %s AND EXISTS (
                SELECT 1 FROM games g WHERE g.sport = r.sport AND g.slate_key = r.slate_key
            )
            ORDER BY r.run_at DESC
            """,
            (sport,),
        )
        return [r["slate_key"] for r in cur.fetchall()]


def run_for_slate(sport: str, slate_key: str) -> dict | None:
    with _cursor() as cur:
        cur.execute(
            "SELECT * FROM runs WHERE sport = %s AND slate_key = %s", (sport, slate_key)
        )
        return cur.fetchone()


def latest_runs() -> list[dict]:
    """One row per sport: its most recent run, for a simple health/"last
    updated" view. LEFT JOIN-free by construction -- just the max run_at
    per sport via DISTINCT ON, which also naturally omits a sport with no
    runs yet rather than needing a placeholder row."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (sport) sport, slate_key, run_at, status, notes
            FROM runs ORDER BY sport, run_at DESC
            """
        )
        return cur.fetchall()
