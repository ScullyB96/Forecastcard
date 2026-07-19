# Sports Prediction Models

Independent, self-contained prediction projects, one per sport, plus a
shared frontend that presents all of them in one place.

- [`nfl/`](nfl/) — NFL game/score/prop prediction model. Layered (team power
  ratings -> QB/injury adjustments -> real market-line blend -> player props),
  walk-forward validated throughout. Automated weekly pipeline.
- [`mlb/`](mlb/) — not started yet.
- [`site/`](site/) — not started yet. The eventual unified predictions
  frontend; reads a standardized JSON export from each sport project, never
  their internals.

## Why this shape

Each sport project has its own venv, own data cache, and own source tree —
no shared Python package between them, so a bug or dependency change in one
can never bleed into another. The only thing they have in common is a small
standardized predictions-export format that `site/` consumes. See each
project's own README for setup.
