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


def top_hr_props(sport: str, slate_key: str, limit: int = 10) -> list[dict]:
    """The `limit` highest projected-mean-HR player rows for this slate,
    each joined to its game's `matchup` string for context (a prop row's
    own `team` column is always NULL by design -- see export_to_site_db's
    _prop_rows -- so the matchup is the only per-player team context
    available without a schema change). Ranked by `proj_mean` (the real
    projected HR count for that game, e.g. 0.42), not a probability --
    the site currently only exports "mean" markets (see export_to_site_db
    .py's BATTER_MARKETS comment), so there is no p_1plus_hr column to
    rank by instead. Empty for any sport/slate with no "HR" market rows
    (every non-MLB sport, and MLB slates before props exist yet) -- no
    sport-name check needed, the data itself decides."""
    with _cursor() as cur:
        cur.execute(
            """
            SELECT p.player_name, p.proj_mean, g.matchup
            FROM props p
            JOIN games g ON g.sport = p.sport AND g.slate_key = p.slate_key AND g.game_id = p.game_id
            WHERE p.sport = %s AND p.slate_key = %s AND p.market = 'HR' AND p.proj_mean IS NOT NULL
            ORDER BY p.proj_mean DESC
            LIMIT %s
            """,
            (sport, slate_key, limit),
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


def mark_notified(sport: str, slate_key: str) -> bool:
    """Records that this (sport, slate_key) has been Slack-notified (see
    schema.sql's slack_notifications table / app/slack_notify.py).
    Returns True only if THIS call actually inserted the row -- i.e. this
    is genuinely the first notification for this slate. Returns False if
    a row already existed, meaning a caller should treat this as a
    silent no-op rather than posting to Slack again."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO slack_notifications (sport, slate_key)
            VALUES (%s, %s)
            ON CONFLICT (sport, slate_key) DO NOTHING
            """,
            (sport, slate_key),
        )
        return cur.rowcount == 1


def get_discord_message_id(sport: str, slate_key: str) -> str | None:
    """The Discord message id of the last digest posted for this slate
    (see schema.sql's discord_message_id column), or None if nothing was
    ever posted (e.g. Discord wasn't configured yet, or the row predates
    this column) -- used to delete-then-repost a corrected digest."""
    with _cursor() as cur:
        cur.execute(
            "SELECT discord_message_id FROM slack_notifications WHERE sport = %s AND slate_key = %s",
            (sport, slate_key),
        )
        row = cur.fetchone()
        return row["discord_message_id"] if row else None


def set_discord_message_id(sport: str, slate_key: str, message_id: str) -> None:
    """Records the id of the Discord message just posted for this slate,
    upserting the (sport, slate_key) row if mark_notified hasn't already
    created it (a force-repost can outlive the row's original insert)."""
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO slack_notifications (sport, slate_key, discord_message_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (sport, slate_key) DO UPDATE SET discord_message_id = EXCLUDED.discord_message_id
            """,
            (sport, slate_key, message_id),
        )


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
