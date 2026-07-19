# Unified Predictions Site

Not started yet. Planned as a proper component-based frontend (not the
quick server-templated HTML used for the NFL project's standalone page) so
there's real room to grow: per-sport sections, team branding, history,
filters — a clean, professional presentation across every sport model.

## Contract with each sport project

`site/` never imports NFL or MLB code directly — it only reads a small,
standardized predictions export that each sport project writes after its own
pipeline runs (e.g. `nfl/data/processed/predictions_latest.json`,
`mlb/data/processed/predictions_latest.json`). Proposed shape:

```json
{
  "sport": "nfl",
  "generated_at": "2026-07-19T12:00:00Z",
  "events": [
    {
      "id": "2026_01_NE_SEA",
      "date": "2026-09-10",
      "home_team": "SEA",
      "away_team": "NE",
      "home_score_pred": 24.0,
      "away_score_pred": 21.0
    }
  ]
}
```

This keeps every project fully decoupled — a change to how the NFL model
works internally can never break the site, and the site can be rebuilt from
scratch without touching either model.

## Stack (TBD)

Leaning toward a static-export React/Next.js (or similar) app that reads
these JSON exports at build time — no server to run or maintain, while still
getting full component/design flexibility for a polished UI. Revisit when
we actually start building this.
