# Combined Multi-Sport Prediction Site

FastAPI + Jinja2 app showing game predictions and player props across all
4 sport models (MLB, NFL, NBA, NHL), deployed on Railway. Full plan and
architecture: see the plan under `/Users/brettscully/.claude/plans/` from
the 2026-08 planning session (this README covers only what's needed to run
`site/` itself).

## Contract with each sport project

`site/` never imports MLB/NFL/NBA/NHL code directly. Each sport's own
pipeline (unchanged, running on its own Railway cron schedule) is followed
by a small `export_to_site_db.py` script living in that sport's own
`src/pipeline/`, which reads that sport's latest parquet output and upserts
into 3 shared Postgres tables owned by `db/schema.sql`:

- `games` — one row per (sport, slate_key, game_id)
- `props` — one row per (sport, slate_key, game_id, player_id, market)
- `runs` — one row per (sport, slate_key), for a "last updated" indicator

`slate_key` is whatever each sport's own output filename already uses: an
ISO date (MLB/NBA/NHL) or `"{season}-wk{week}"` (NFL). See the comments in
`db/schema.sql` for the full column-by-column rationale.

This keeps every sport project fully decoupled from the site — a change to
how a model works internally can never break the site, and the site can be
rebuilt from scratch without touching any model.

## Local dev database

```bash
brew install postgresql@16   # one-time
brew services start postgresql@16
createdb sports_site_dev
psql -d sports_site_dev -f db/schema.sql
```

Local connection string (trust auth, no password, matches Railway's
`DATABASE_URL` shape): `postgresql://$(whoami)@localhost:5432/sports_site_dev`

## Stack

FastAPI + Jinja2 (not Next.js — keeps the whole monorepo, including the
site, in one language). `requirements.txt` and the `app/` package land in
a later phase of this build.
