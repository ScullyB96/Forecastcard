# NBA Prediction Model

Build in progress (started 2026-07-24). Mirrors the `nhl/`/`mlb/` projects' shape so the same
conventions (walk-forward validation, no-lookahead discipline, a genuine holdout split,
paired-bootstrap significance testing, a continuously-updated research log) carry over, but
this is a fully independent project — own venv, own data cache, own source tree. Nothing here
imports from or depends on `nhl/`, `mlb/`, or `nfl/`.

Predicts game scores from a closed-form pace x team-rating combine, with a play-by-play-derived
RAPM-lite layer that adjusts each team's rating for tonight's actual active lineup. See
`MODEL_DOCUMENTATION.md` for the full research log, current status, and every design decision's
rationale.

## Setup

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

- `src/ingest/` — raw data fetching from `nba_api` (stats.nba.com), RotoWire, and the static
  player/team lookups: `fetch_schedule.py`, `fetch_boxscores.py`, `fetch_playbyplay.py`,
  `fetch_rotation.py`, `fetch_rotowire_lineups.py`, `build_stints.py` (lineup-stint construction
  from play-by-play + rotation data), `team_codes.py`, `player_name_crosswalk.py`
- `src/models/` — rating engines and validation:
  - `shrinkage.py`, `team_strength.py`, `home_court.py`, `rest_schedule.py` — Phase 1, the
    walk-forward team-level pace/rating engine
  - `rapm_lite.py`, `player_priors.py`, `garbage_time.py`, `lineup_rating.py` — Phase 2, the
    play-by-play-derived RAPM-lite active-lineup adjustment
  - `score_distribution.py` — Phase 3, win probability / spread / total from the projected score
  - `final_holdout_check.py`, `bootstrap_significance.py`, `metrics_ledger.py` — the shared
    validation harness (dev/holdout split, paired bootstrap, append-only results log)
  - `validate_*.py` — one acceptance-gate script per phase, each runnable standalone
- `src/pipeline/` — live orchestration: `refresh_data.py` (incremental current-season refresh),
  `generate_predictions.py` (the daily entry point — see its docstring for what's wired up so far)
- `src/utils/paths.py` — shared path constants (`DATA_RAW`/`DATA_PROCESSED`/`DATA_EXTERNAL`)
- `data/raw/`, `data/processed/` — gitignored, regenerable from `src/ingest`
- `data/external/` — any manually-acquired reference data (gitignored if large)
- `tests/` — regression guards for any bug class found during validation
- `MODEL_DOCUMENTATION.md` — the research log: every cycle, what worked, what didn't, why
