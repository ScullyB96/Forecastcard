# Combined-site Railway deployment runbook

Everything on the code side is built, tested locally (Postgres 16 via Homebrew,
uvicorn, real data from all 4 sports), and committed. This is what's left, all
of which needs your own Railway account — I can't create paid cloud resources
on your behalf. Follow these steps in order; each has a way to confirm it
worked before moving to the next.

## 0. Prerequisites

- A Railway account (railway.app) with billing set up (Postgres + 6 services
  will incur real cost — check Railway's current pricing before proceeding).
- The Railway CLI, optional but recommended: `brew install railway` (or
  `npm i -g @railway/cli`), then `railway login`.
- This repo pushed to a Git remote (GitHub) that Railway can deploy from —
  Railway's standard flow links a GitHub repo per service.

## 1. Create the project + Postgres

1. railway.app → New Project → empty project.
2. Add a service → Database → PostgreSQL. Railway provisions it and exposes
   `DATABASE_URL` automatically to every other service in the same project
   once you reference it (step 3 below).
3. Apply the schema once, from your machine, against Railway's Postgres:
   ```bash
   railway connect postgres   # opens a psql shell against Railway's DB
   ```
   then paste the contents of `site/db/schema.sql`, or run it directly:
   ```bash
   railway run --service postgres psql -f site/db/schema.sql
   ```
   **Verify**: `\dt` inside that psql shell should list `games`, `props`, `runs`.

## 2. Add the 6 services

For each service: New Service → Deploy from GitHub repo → select this repo →
set **Root Directory** as listed below. Railway auto-detects Python via
Nixpacks and reads `requirements.txt` + the service's `railway.toml`.

| Service name | Root directory | Config file path (only if ≠ default) |
|---|---|---|
| `mlb-worker` | `mlb/` | (uses `mlb/railway.toml`) |
| `nfl-worker-tuesday` | `nfl/` | `nfl/railway.tuesday.toml` |
| `nfl-worker-sunday` | `nfl/` | `nfl/railway.sunday.toml` |
| `nba-worker` | `nba/` | (uses `nba/railway.toml`) |
| `nhl-worker` | `nhl/` | (uses `nhl/railway.toml`) |
| `web` | `site/` | (uses `site/railway.toml`) |

The two NFL services share a root directory but need **different** config
files — set each one's "Config File Path" explicitly in that service's
Settings tab (Railway dashboard, not something set via the repo). Without
this, both would default to reading the same `nfl/railway.toml`, which
doesn't exist (there's only `.tuesday.toml`/`.sunday.toml` — deliberate, so a
misconfigured service fails loudly instead of silently picking one).

For each of the 5 worker services, add the env var `DATABASE_URL` = a
reference to the Postgres service's connection string (Railway's dashboard
has a "reference a variable from another service" picker — do this rather
than copy-pasting the raw string, so it stays correct if Postgres ever
changes). Same for `web`.

**Verify**: each service's Settings tab shows the right root directory and
(for NFL) the right config file path; each has `DATABASE_URL` set.

## 3. First manual run, one service at a time

Don't wait for cron. For each of the 5 workers, use Railway's "Deploy" /
"Run" button (or `railway run --service <name> <startCommand>` from the CLI)
to trigger one real run, then check its logs.

**What a good run looks like** (matches what this session verified locally):
- MLB: ends with `exported N games, M prop rows for YYYY-MM-DD`
- NFL: ends with `exported N games, M prop rows for YYYY-wkW`
- NBA: ends with `exported N games, M prop rows for YYYY-MM-DD`
- NHL: ends with `exported N games for YYYY-MM-DD` (no prop-row count — NHL
  has no props pipeline, this is expected, not an error)

If a worker fails here, it's almost always one of: missing system package
Nixpacks didn't auto-detect (check the build log), a network egress issue
fetching that sport's own data source (Statcast/nflverse/nba_api/NHL API),
or `DATABASE_URL` not actually resolving (echo it in the service's shell to
confirm before assuming the export script is broken).

**Verify** (via `railway connect postgres` or any Postgres client):
```sql
SELECT sport, slate_key, run_at, status, notes FROM runs ORDER BY run_at DESC;
```
Should show one fresh row per sport you just ran.

## 4. Verify the web service

Once at least one worker has run successfully, open the `web` service's
public URL (Railway assigns one automatically, or attach a custom domain).

**Verify**:
- `/` shows the sport(s) you just ran, with real games and (if applicable)
  an expandable props panel per game.
- `/mlb`, `/nfl`, `/nba`, `/nhl` each load without error, even ones with no
  data yet (should show "No runs yet for X" rather than crashing).
- `/healthz` returns `{"status": "ok"}` — this is also what Railway's own
  healthcheck polls, so if the service shows unhealthy in the dashboard,
  hit this path directly first to isolate whether it's the app or something
  else (bad `DATABASE_URL`, port binding).

## 5. Turn on cron

Once satisfied with a manual run, set each worker's schedule in its
Settings tab (or it's already read from `railway.toml`'s `cronSchedule` —
confirm the dashboard shows the schedule you expect). **Re-check the UTC
times in each `railway.toml`** against whichever DST regime is active on
the actual date you're reading this — every config file has a comment
explaining the EDT/EST conversion, but the comments were written assuming
you'd sanity-check them, not as a substitute for doing so.

**Verify** over the following days: `runs.run_at` timestamps update on
schedule without you triggering anything manually.

## 6. Lock it down (optional, whenever you want)

Set `SITE_PASSWORD` on the `web` service's env vars (any value). No
redeploy needed — verified locally this session that the gate takes effect
immediately: unset = fully public, set = every route (except `/login`,
`/static`, `/healthz`) redirects to a login form until a valid session
cookie is presented. Unset it again any time to go back to fully public.

## If something's wrong and you don't know where to start

Check the `runs` table first (`SELECT * FROM runs ORDER BY run_at DESC`) —
it's the fastest way to see which sport last exported successfully and
when. Note the limit of this: each export script only writes its `runs`
row after games/props are written, in the same transaction, so a sport
missing from `runs` (or showing a stale `run_at`) means that service's last
attempt either hasn't happened yet or crashed before finishing — it does
NOT log a "failed" row to check, so for the actual error you need that
service's Railway logs directly, not just this table.
