# NBA Prediction Model

Built 2026-07-24 through 2026-08-01. Mirrors the `nhl/`/`mlb/` projects' shape so the same
conventions (walk-forward validation, no-lookahead discipline, a genuine holdout split,
paired-bootstrap significance testing, a continuously-updated research log) carry over, but
this is a fully independent project — own venv, own data cache, own source tree. Nothing here
imports from or depends on `nhl/`, `mlb/`, or `nfl/`.

Two layers, both done (v1 scope) and validated on real data:

- **Game scores**: a closed-form pace x team-rating combine, with a play-by-play-derived RAPM-lite
  layer that adjusts each team's rating for tonight's actual active lineup, plus a score-
  distribution layer for win probability / spread / total. Live entry point:
  `python -m src.pipeline.generate_predictions [YYYY-MM-DD]`.
- **Player props**: a full per-player stat-line projection (points split 2PT/3PT/FT, rebounds
  split OREB/DREB, assists, turnovers, steals, blocks) built from six independently-validated
  walk-forward rate models, a matchup-difficulty layer (built, not yet live-wired), and a
  macro-anchor + micro-reallocation composition rule that ties points back to the game-score
  model's own team totals without double-counting. Live entry point:
  `python -m src.pipeline.generate_props [YYYY-MM-DD]`.

See `MODEL_DOCUMENTATION.md` for the full research log, current status, and every design
decision's rationale — including several real smoothing-family regressions found and fixed along
the way (the recurring lesson: two superficially similar stats often need OPPOSITE treatment,
confirmed empirically every time, never assumed by analogy).

## Setup

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

- `src/ingest/` — raw data fetching from `nba_api` (stats.nba.com), RotoWire, and the static
  player/team lookups:
  - `fetch_schedule.py`, `fetch_boxscores.py`, `fetch_playbyplay.py`, `fetch_boxscore_traditional.py`,
    `build_stints.py` (lineup-stint reconstruction from play-by-play substitution events),
    `fetch_rotowire_lineups.py`, `team_codes.py`, `player_name_crosswalk.py` — game-score inputs
  - `fetch_player_track.py` (`BoxScorePlayerTrackV3` — touches, rebound chances), `fetch_boxscore_matchups.py`
    (`BoxScoreMatchupsV3`/`BoxScoreDefensiveV2`, 2017-18+ only), `fetch_team_rosters.py`
    (`CommonTeamRoster`, per-season positions) — player-props inputs
- `src/models/` — rating engines and validation:
  - `shrinkage.py`, `team_strength.py`, `home_court.py`, `rest_schedule.py` — Phase 1, the
    walk-forward team-level pace/rating engine
  - `rapm_lite.py`, `player_priors.py`, `garbage_time.py`, `lineup_rating.py` — Phase 2, the
    play-by-play-derived RAPM-lite active-lineup adjustment
  - `score_distribution.py` — Phase 3, win probability / spread / total from the projected score
  - `player_rate_shrinkage.py` — the shared walk-forward primitive every per-player rate model
    below is built on (generalizes `shrinkage.py` to arbitrary exposure units and player-level
    grouping)
  - `player_minutes.py`, `player_scoring_rates.py`, `player_rebounding_rates.py`,
    `player_playmaking_rates.py`, `player_defensive_event_rates.py` — the six per-player
    rate-category models (minutes; 2PT/3PT/FT attempts+makes; OREB/DREB; AST/TOV; STL/BLK)
  - `matchup_difficulty.py` — the 3-level defender/team-scheme difficulty hierarchy (built,
    validated, not yet wired into the live props pipeline — see `MODEL_DOCUMENTATION.md`)
  - `usage_allocation.py` — the macro-anchor + micro-reallocation composition rule (points
    anchored to the game-score model's own team total; REB/AST/TOV/STL/BLK left unanchored, v1)
  - `prop_distribution.py` — player-level predictive distributions (Normal/Student-t for
    high-count stats, Poisson/Negative-Binomial for low-count ones) and `over_under_prob`
  - `final_holdout_check.py`, `bootstrap_significance.py`, `metrics_ledger.py` — the shared
    validation harness (dev/holdout split, paired bootstrap, append-only results log)
  - `validate_*.py` — one acceptance-gate script per model, each runnable standalone
- `src/pipeline/` — live orchestration:
  - `refresh_data.py` — incremental current-season refresh
  - `active_roster.py` — shared active-lineup resolution (`games_on_date`, `resolve_active_lineup`,
    `build_team_history`), used by both entry points below so they can't silently drift
  - `generate_predictions.py` — the daily game-score entry point
  - `generate_props.py` — the daily player-props entry point
- `src/utils/paths.py` — shared path constants (`DATA_RAW`/`DATA_PROCESSED`/`DATA_EXTERNAL`)
- `data/raw/`, `data/processed/` — gitignored, regenerable from `src/ingest`
- `data/external/` — any manually-acquired reference data (gitignored if large)
- `tests/` — regression guards for any bug class found during validation
- `MODEL_DOCUMENTATION.md` — the research log: every cycle, what worked, what didn't, why
