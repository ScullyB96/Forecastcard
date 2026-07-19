# MLB Prediction Model

Not started yet. Mirrors the `nfl/` project's shape so the same conventions
(walk-forward validation, layered rating engines, an automated weekly/daily
pipeline, a standardized predictions export for `site/`) carry over, but this
is a fully independent project — own venv, own data cache, own source tree.
Nothing here imports from or depends on `nfl/`.

## Setup (when starting this project)

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # create this as the project takes shape
```

## Structure

- `src/ingest/` — raw data fetching (source TBD — e.g. pybaseball, MLB Stats API)
- `src/models/` — rating engines, prediction logic
- `src/pipeline/` — automated orchestration (refresh data -> rebuild -> predict)
- `data/raw/`, `data/processed/` — gitignored, regenerable from `src/ingest`
- `data/external/` — any manually-acquired reference data (gitignored if large)
