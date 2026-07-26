# NHL Model — Day-Zero Operational Runbook

Written 2026-07-25, before the 2026-27 season, per MODEL_DOCUMENTATION.md §38's freeze
discipline: the defense against a bad opening week is having written down, in July, what a bad
week does and doesn't mean — not improvising it in October under real pressure. If you're reading
this during the season because something looks wrong, start here before touching any code.

## What runs, and when

- **Daily**: `python -m src.pipeline.generate_predictions [YYYY-MM-DD]` (defaults to today).
  This refreshes all data (`refresh_data.py`), fetches the real RotoWire injury report, prices
  every real game scheduled that day, prints results, and appends each prediction to the
  immutable log (`data/processed/live_prediction_log.jsonl`) **before puck drop**.
- **Every 4 weeks** (rolling): compute the degradation-budget check (§37.4) and the live-vs-market
  gap (§37.5/§37.6) against `live_prediction_log.jsonl` once real results are known. Not yet
  automated as of this writing — a manual run against the log, cross-referenced with real final
  scores, until/unless that's worth scripting.
- **One time, Sep 19-20**: the dress rehearsal (§39.3/§41-42) — a reminder is scheduled
  (`trig_01WBTUeWnFKqaUwsPrC9zS47`) for that window. Run `generate_predictions.py` for real on
  those dates even though they're preseason (predictions don't mean anything for preseason
  rosters — the point is exercising the machinery), fix `fetch_rotowire_odds.py`'s unverified
  parser against real populated odds, and investigate `nhl-lineups.php`'s real structure now that
  games exist.

## What the canaries look like when they fire

`refresh_data.py` calls `assert_schedule_schema`/`assert_moneypuck_situational_schema`
(`src/ingest/schema_guards.py`) on every fresh fetch, **before** overwriting the cache. If either
raises `SchemaGuardError`:

- **The cache is NOT corrupted** — the guard runs before the write, so yesterday's known-good
  data is still there. Nothing crashes silently; the daily run just stops with a loud message.
- Read the error message itself — it names exactly which column or value is unexpected and why
  it matters (which downstream model piece depends on it). This is the "provider broke" case
  §38's emergency-fix path exists for.
- **Do not react by tuning the model.** This is a data problem, not a model problem. Fix the
  ingest code (or wait out a transient upstream issue) and re-run.
- A NEW `gameType` value (outside the already-seen {1,2,3,4,6,7,8,9,12,19,20}) is fine and
  expected — the guard only checks `gameType==2` rows' own `gameState`/`lastPeriodType`, and a
  brand-new value there (or a missing/new MoneyPuck `situation`) is what actually fires.

## What "degradation budget breached" concretely triggers (§37.4)

Compute the paired-bootstrap gap between live and backtest-holdout performance, per rolling
4-week window (~220-240 games), for Brier, **margin-MAE** (the primary metric — the one axis with
a real, confirmed goalie-information sensitivity), and total-MAE.

| | routine budget (realistic) | escalation ceiling (worst case) |
|---|---|---|
| margin-MAE | ≤ 0.00382 | ≤ 0.01371 (this one is a REAL, confirmed number, not a noise bound) |
| Brier | ≤ 0.00088 | ≤ 0.00110 |
| total-MAE | ≤ 0.00610 | ≤ 0.00491 |

- **Within routine budget on everything**: no action. Expected, budgeted variation.
- **Margin-MAE exceeds routine but stays under escalation**: investigate the goalie-data pipeline
  FIRST (`fetch_rotowire_injuries.py` returning stale/missing/low-confidence data more often than
  expected) before anything else — this is the one pattern this project can name in advance.
- **Anything exceeds its escalation ceiling, or Brier/total-MAE alone show a real gap without a
  matching margin-MAE story**: genuine, unbudgeted anomaly. Check for a specific, nameable cause
  (rule change, realignment, a real data-quality regression) before assuming the model itself
  needs refitting. **Recalibration is the LAST resort, not the first response** — this project's
  standing preference for root-cause fixes over parameter adjustment (§29.3.1) applies here too.
- **Live outperforms backtest on everything**: no action, just note it — mildly surprising, not
  itself a problem.

## The live-vs-market improvement channel (§37.5-37.6) — separate from the above, diagnostic only

Track the live model-vs-market Brier gap against **0.00247** (§36.5's frozen reference), segmented
by season phase (early season / mid-season stable baseline / trade-deadline window / final two
weeks, §37.6). This does NOT feed the degradation trigger above and should not be reacted to
mid-season — it's read once enough games accumulate per phase. A narrower-than-0.00247 live gap
in the trade-deadline/final-weeks buckets specifically (vs. the mid-season baseline) would
confirm the informational (lineup/injury) hypothesis for the residual gap; a flat gap across all
four phases would argue against it. Either result is informative; neither triggers automated action.

## Known residuals — do not panic about these

- **Mean predicted total sits ~0.09 goals/game low** (the EV-bucket residual, §18.3/§28-29) — a
  real, understood, currently-unfixable-by-tracker undershoot. §29.3.1's signal-to-noise argument
  says no trailing tracker at any halflife can safely estimate a signal this small; only a future
  root-cause mechanism fix could close it. This is priced in, not a live-season surprise.
- **Single-week straight-up accuracy anywhere from ~45% to ~70% is pure noise** at NHL variance
  levels on a ~50-60-game week — do not read anything into one bad (or good) week. The §38 freeze
  exists specifically so a bad opening week doesn't trigger reactive tinkering; this line is the
  concrete reminder of why.
- **The live single-game predictor's own residual vs. backtest is ~0.3-1%** on real historical
  validation (§41.3) — a known, small, honestly-logged gap in the tie-mass calibration ratio
  specifically, not fully traced. Don't mistake this for new live-season drift if it shows up
  again; it was already there before the season started.

## The immutable prediction log

`data/processed/live_prediction_log.jsonl` — hash-chained, append-only (`src/models/
prediction_log.py`). Verify integrity any time with `python -m src.models.prediction_log`
(`VALID: N entries, hash chain intact`, or a specific entry index if something's wrong). This is
the only record that should ever be used to compute the degradation budget or the live-vs-market
gap — never `daily_predictions_*.parquet` (a mutable convenience snapshot that gets overwritten
on re-run).

## Documenting an emergency fix (§38's second permitted change path)

If §37.4's rule fires a real escalation, OR a confirmed data-pipeline failure requires an
in-season code change (the ONLY two paths §38 permits touching the frozen configuration): write
down, before making the change, (1) what broke and how it was confirmed real (not assumed),
(2) the specific fix, (3) whether it changes any frozen constant's value and by how much, (4) a
re-run of the market-benchmark/holdout-adjacent check most relevant to what changed, to confirm
the fix doesn't quietly regress something else. This mirrors every other adoption decision this
project has ever made — an emergency in-season fix does not get a lower evidentiary bar just
because it's urgent.
