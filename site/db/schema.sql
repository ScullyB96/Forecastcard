-- Shared schema for the combined multi-sport prediction site.
--
-- Owned by site/ (not any individual sport project) since this is the one
-- piece of shared state across otherwise-fully-decoupled sport pipelines.
-- Each sport's own pipeline keeps running unchanged on its own schedule;
-- a small per-sport export_to_site_db.py (one per sport, living in that
-- sport's own src/pipeline/) reads that sport's latest parquet output and
-- upserts rows here. No sport project imports another's code, and none of
-- them import anything from site/ either -- this SQL file is the only
-- contract between them.
--
-- slate_key is whatever each sport's own output filename already uses as
-- its date/period identifier, so the export script's job is a straight
-- passthrough, not a new normalization scheme:
--   MLB / NBA / NHL: an ISO date string, e.g. '2026-08-01'
--   NFL:              '{season}-wk{week}', e.g. '2026-wk1'
--
-- game_id is stored as TEXT (not INTEGER) even though some sports use a
-- numeric id internally (MLB's game_pk, NBA's game id) -- this keeps the
-- schema uniform across sports without needing a sport-specific id column,
-- and NFL's own game ids are already string-shaped.
--
-- extra JSONB on both games and props holds whatever doesn't generalize
-- across all 4 sports (MLB's over/under-8.5 and spread-cover probabilities,
-- NFL's market lines, etc.) rather than forcing a false common schema onto
-- fields only some sports have.

CREATE TABLE IF NOT EXISTS games (
    sport                   TEXT NOT NULL,
    slate_key               TEXT NOT NULL,
    game_id                 TEXT NOT NULL,
    home_team               TEXT NOT NULL,
    away_team               TEXT NOT NULL,
    matchup                 TEXT NOT NULL,
    home_win_prob           DOUBLE PRECISION,
    away_win_prob           DOUBLE PRECISION,
    projected_home_score    DOUBLE PRECISION,
    projected_away_score    DOUBLE PRECISION,
    projected_total         DOUBLE PRECISION,
    extra                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- Per-game factor breakdown (park/weather/umpire/HFA/lineup+pitcher
    -- summaries/bullpen/etc.) powering the site's "Factors" debug panel.
    -- Separate from `extra` (small flag-shaped values) because this is a
    -- larger structured payload and, unlike `extra`, only some sports
    -- populate it -- keeping it its own column lets a sport opt in without
    -- reshaping extra's existing contract. Empty object (not present) means
    -- this sport/row hasn't wired the factor breakdown through yet.
    debug                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, slate_key, game_id)
);

-- Safe to re-run: adds `debug` to a games table that already existed
-- before this column was introduced (schema.sql has no separate migration
-- runner -- this file IS the migration, applied idempotently).
ALTER TABLE games ADD COLUMN IF NOT EXISTS debug JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_games_sport_slate ON games (sport, slate_key);

-- One row per player x market (e.g. "1+ HR", "6+ K", "total bases") rather
-- than one row per player -- each sport's export script explodes its own
-- wide per-player prop table (batter_props_{date}.parquet's several
-- p_1plus_hr / p_1plus_hit / mean_total_bases-style columns, etc.) into
-- this long shape so the site can render an arbitrary, sport-specific set
-- of markets without a schema change. Stays empty for NHL (no props
-- pipeline exists there today -- not a bug, just nothing to export yet).
CREATE TABLE IF NOT EXISTS props (
    sport                   TEXT NOT NULL,
    slate_key               TEXT NOT NULL,
    game_id                 TEXT NOT NULL,
    player_id               TEXT NOT NULL,
    player_name             TEXT NOT NULL,
    team                    TEXT,
    market                  TEXT NOT NULL,
    line                    DOUBLE PRECISION,
    over_prob               DOUBLE PRECISION,
    under_prob              DOUBLE PRECISION,
    proj_mean               DOUBLE PRECISION,
    extra                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, slate_key, game_id, player_id, market)
);

CREATE INDEX IF NOT EXISTS idx_props_sport_slate ON props (sport, slate_key);
CREATE INDEX IF NOT EXISTS idx_props_game ON props (sport, slate_key, game_id);

-- Powers a "last updated" indicator and a simple per-sport health view --
-- one row per (sport, slate_key) run, overwritten on re-run via the same
-- upsert pattern as games/props.
CREATE TABLE IF NOT EXISTS runs (
    sport                   TEXT NOT NULL,
    slate_key               TEXT NOT NULL,
    run_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    status                  TEXT NOT NULL,
    notes                   TEXT,
    PRIMARY KEY (sport, slate_key)
);

CREATE INDEX IF NOT EXISTS idx_runs_sport ON runs (sport, run_at DESC);

-- Dedup ledger for the Slack daily-picks digest (see app/slack_notify.py).
-- A sport's own cron can call the /internal/notify trigger several times
-- a day (MLB fires 4x, NBA/NHL 2x) as it refreshes the SAME slate_key
-- intraday -- without this, a "daily" digest would spam the channel once
-- per firing. app/db.mark_notified INSERTs with ON CONFLICT DO NOTHING
-- and reports back whether THIS call was the one that actually inserted
-- the row, so only the first firing to touch a given (sport, slate_key)
-- ever posts to Slack; every later same-day firing for that same slate
-- is a silent no-op, no per-sport cron-time gating needed.
CREATE TABLE IF NOT EXISTS slack_notifications (
    sport                   TEXT NOT NULL,
    slate_key               TEXT NOT NULL,
    notified_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, slate_key)
);
