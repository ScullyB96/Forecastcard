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
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, slate_key, game_id)
);

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
