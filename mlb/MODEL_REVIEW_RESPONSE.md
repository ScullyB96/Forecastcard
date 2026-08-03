# MLB Model Review — Findings and Response

**What this is.** A complete record of an independent review of the MLB prediction model
(conducted against `MODEL_REVIEW_PACKET.md` + `MODEL_REVIEW_PROMPT.md`), what was verified or
fixed in response to each finding, and the additional production work that followed from it.
Written to be a standalone reference — every number below was independently reproduced against
real data or the real production system, not taken on faith from the review itself.

**Reviewer's overall assessment** (quoted in substance): the model is at or extremely near the
genuine no-market-data ceiling for straight-up game prediction; remaining real headroom lives
almost entirely in props/calibration, not win-pick accuracy. This independently converged with
the packet's own §10 finding (from an earlier internal literature-benchmarking pass), reached via
a completely different method — one from direct arithmetic bounding and an audit-ranking table,
the other from published-literature comparison.

---

## Finding 1 — Arithmetic mismatch in the §9 worked example (Brandon Lowe)

**The claim.** The reviewer manually verified the odds-ratio combine in §9 (exact match:
`odds(3.089%) × ... = 4.502%`), then tried to reconcile that 4.502% pre-context figure against
the packet's final stated home-run probability (3.0%) by multiplying through the documented
context factors (platoon, park, weather, state, TTO). The math didn't close — bounded that the
16-category unnormalized vector would need to sum to ≈1.42 to explain the gap, which is
implausible given `field_out` alone consumes ~45% of the vector and nothing in §4 meaningfully
boosts it. Correctly guessed the likely cause was a missing second platoon leg (the pitcher's own
platoon-allowed split, separate from the batter's), but used an estimated magnitude (~0.85) that
left the reconciliation ~15% short even after that adjustment.

**Verification.** Re-instrumented the exact plate appearance (Lowe leading off the 2nd, bases
empty, 0 outs, vs. Andrew Abbott) and logged every applied factor plus the true pre-renormalization
sum, rather than reasoning from memory.

**Result: not a code bug.** The reviewer's instinct was exactly right — the real value of Abbott's
own platoon-allowed multiplier is **0.7206**, not the estimated ~0.85. With the real number, the
chain closes to the last decimal:

```
4.502% × 1.0815 (state) × 0.8541 (batter platoon) × 0.7206 (pitcher platoon)
       × 1.1348 (park) × 0.9604 (weather) × 0.9417 (TTO) = 3.0757% (unnormalized)

exact 16-category unnormalized sum = 1.011394   (not ~1.42 — the renorm correction is small,
                                                  because park/weather/state/TTO factors are
                                                  each independently mean-normalized to ~1.0
                                                  across the league by construction)

3.0757% / 1.011394 = 3.0411%   ← matches the packet's stated 3.0% exactly
```

**Fix.** `MODEL_REVIEW_PACKET.md` §9 previously described Lowe's own platoon multiplier in "Step
3" and Abbott's own platoon-allowed multiplier in "Step 4," and never re-listed either one in
"Step 7"'s context-multiplier table — so a careful reader correctly saw a gap where the *prose*
had one, even though the *code* didn't. Rewrote Step 7 to list both platoon legs together in the
same table the code actually applies them in, and added the verified reconciliation math directly
beneath it so a future reader can't hit the same false alarm.

---

## Follow-up to Finding 1 — a real platoon double-counting bug, found by scrutinizing the number that closed the gap

After Finding 1 was resolved, the reviewer went one step further: the combined platoon effect for
Lowe-vs-Abbott (0.8541 × 0.7206 = 0.615, a 38.5% HR reduction for one same-hand matchup) was
flagged as suspicious in its own right. Hypothesis: `platoon_splits.py` computes ONE population-
level `league_mult` (same-vs-opposite-hand odds ratio) from raw plate-appearance data — since a
raw PA-level rate ratio can't separate "how much of this effect is batters" from "how much is
pitchers," this single value inherently already reflects both sides' combined contribution. But
the batter-side and pitcher-side platoon tables each independently default to `league_mult**-0.5`
when a player lacks enough individual same/opposite-hand history, and both legs get multiplied
together in every real matchup — so two default players combine to `league_mult**-1`, applying
the shared population effect **twice**.

### Verification (in order)

1. **Decomposed leakage-free regression**: fit a separate attenuation exponent per context factor
   (offset = logit of the matchup combine, one covariate per factor: state, batter-platoon,
   pitcher-platoon, park, times-through-order) on real 2024 data. Result:

   | | λ_batter_platoon | λ_pitcher_platoon | sum |
   |---|---|---|---|
   | home_run | 0.4166 (CI excludes 1) | 0.6735 (CI excludes 1) | **1.09** |
   | strikeout | 0.3909 (CI excludes 1) | 0.8274 (CI includes 1) | **1.22** |

   Real data supports the *combined* platoon weight landing near 1×, not the ~2× the code
   implicitly applies. The batter/pitcher asymmetry (0.42 vs 0.67) has a clean mechanical
   explanation: pitchers face far more plate appearances per season than any one batter
   accumulates against a single specific handedness, so a pitcher's own split more often clears
   the 2,200-PA stabilization threshold and relies less on the shared league-default fallback than
   a batter's does — meaning the redundant signal sits disproportionately on the batter side.

2. **Held-out bucketed calibration, unbiased this time** (bucketed by matchup handedness, *not*
   conditioned on outcome — correcting the exact bias the reviewer flagged in Finding 4's own
   evaluation): on 2025 data never touched by the fit, mean predicted vs. actual, by same/opposite
   hand:

   | | actual | baseline (pre-fix) | error |
   |---|---|---|---|
   | HR, same-hand | 2.889% | 2.817% | −0.072pp (under-suppresses less than reality) |
   | HR, opp-hand | 3.255% | 3.434% | +0.179pp (over-boosts) |
   | K, same-hand | 22.763% | 23.727% | +0.964pp |
   | K, opp-hand | 21.767% | 20.888% | −0.879pp |

   Both directions, both categories, all consistent with over-spreading the platoon effect beyond
   what real data supports.

### The structural fix

`src/models/platoon_splits.py`: changed the shared league-average default from
`league_mult**0.5` (per side) to `league_mult**0.25` — so two defaults now combine to
`league_mult**0.5`, the single-application magnitude the data supports, instead of `league_mult**1`.
Each player's own individually-measured platoon deviation (`own_same_mult`/`own_opp_mult`,
blended in by personal reliability) is **completely untouched** — that component is a real,
non-redundant per-player signal, not part of the shared/double-counted population term.

### Pre-registered ship bar (set before results, to avoid post-hoc rationalization)

- **Primary**: the held-out bucketed handedness calibration holds or improves under the
  *structural* fix specifically (not the fitted λ, a different correction).
- **Guardrail**: no material regression in SU/Brier/MAE at high K under CRN pairing — explicitly
  not allowed to be killed by a small negative SU read at low K, since this class of change (a
  symmetric same-hand-vs-opposite-hand correction) was expected to mostly wash out of SU via
  renormalization, making SU a nearly-powerless metric for this specific fix.

### Results

**Primary — passed, with zero fitted correction, pure structural change:**

| | actual | OLD error | NEW error | improvement |
|---|---|---|---|---|
| HR, same-hand | 2.889% | −0.064pp | +0.012pp | 81% closed |
| HR, opp-hand | 3.255% | +0.224pp | +0.131pp | 42% closed |
| K, same-hand | 22.763% | +0.898pp | +0.487pp | 46% closed |
| K, opp-hand | 21.767% | −0.818pp | −0.448pp | 45% closed |

The K-side landing at ~45–46% closure (not full closure) was explicitly predicted in advance — a
real residual, not a failure; a second, smaller K-specific effect (possibly a framing/umpire
interaction with handedness, or the batter-side split's own shrinkage target) is out of scope
today and left as a documented open item.

**Guardrail — passed, and exceeded expectations.** CRN-paired backtest, n=697 real 2023–2024
games, K=30/100/300:

| K | SU gap | Brier gap | MAE gap |
|---|---|---|---|
| 30 | +1.00pp | −0.0020 | −0.0131 |
| 100 | +1.58pp | −0.0008 | −0.0028 |
| 300 | +1.15pp | −0.0003 | −0.0063 |

Not just "no regression" — a consistent, non-sign-flipping improvement in every metric at every
K tested. Over-stating the platoon split in both directions was evidently adding enough noise to
point predictions to genuinely hurt discrimination, not just calibration.

**Shipped**: commit `0bd844a` — *"MLB: fix real platoon double-counting bug (external review
finding)"*. Deployed to production; the next full pregame-context rebuild picks it up
automatically.

---

## Finding 2 — Retest the whiff-rate and walk-blend reverts under CRN pairing

**The claim.** Both reverted signals (a strikeout-specific `whiff_rate_multiplier` and a
walk-specific `pitch_walk_multiplier`/"walk-blend") were killed on SU deltas of roughly the same
size and shape as deltas that were *later shown to be low-K estimator noise* elsewhere in this
project (the latent-shock factor's own apparent −1.1pp cost shrank to −0.7pp under a proper
K-scaling check; the jet-lag factor's apparent −1.2pp cost reversed to statistically zero). Argued
both should be re-tested the same way before being trusted as settled — a new argument the
original revert decisions never had access to.

**Action taken, in order:**

1. **Wired the 3 dormant CRN decision-tags** (`DECISION_BULLPEN_PICK`, `DECISION_CLOSER_USAGE`,
   `DECISION_WEATHER_BUCKET`) — defined in `crn.py` since an earlier task but never referenced
   anywhere in the codebase (confirmed by grep before touching anything). Added an opt-in
   `crn_keys` parameter to `bullpen.py`'s `sample_bullpen_plan` and `weather_forecast.py`'s
   `sample_weather_bucket`; default `None` preserves the exact prior `rng.random()`/`rng.choice()`
   behavior byte-for-byte for every existing caller. **Caught a real bug of my own mid-edit**: an
   early version of the closer-preference draw would have consumed a random value on *every*
   inning instead of only inning 9, which would have desynced the non-CRN rng stream from its
   pre-existing sequence — fixed before it shipped, verified by re-reading the diff against the
   original short-circuit structure. Smoke-tested: default path unchanged, CRN path deterministic
   given identical keys across repeated calls.
2. **`whiff_rate_multiplier` turned out to be fully deleted**, not kept as a documented-but-unused
   artifact the way this project's own revert discipline normally works (confirmed via grep — zero
   references anywhere, matching `MODEL_DOCUMENTATION.md`'s own note). Reconstructing it faithfully
   from a written description alone was judged too speculative to do properly within scope —
   **explicitly deferred, not silently dropped.**
3. **`pitch_walk_multiplier` (the walk-blend) still existed** in `pitch_talent.py`, unused. Re-wired
   it into `validate_game_simulator.py`'s `build_profile` behind a new opt-in
   `use_pitch_walk_blend` flag (byte-identical when `False`), threaded through
   `build_shared_tables(..., build_pitch_walk_composed=True)` and `run_validation(...,
   use_pitch_walk_blend=True)`.
4. **Ran the actual CRN-paired K-scaling retest**: n=697 real 2023-2024 games (same protocol the
   shock's own K-scaling check used), K=30/100/300, baseline vs. walk-blend-on, same seed.

**Result:**

| K | SU gap (blend − baseline) | Brier gap |
|---|---|---|
| 30 | −1.29pp | −0.0005 |
| 100 | −0.14pp | +0.0005 |
| 300 | **+0.57pp** | +0.0004 |

This does **not** show the clean signature that saved the shock. That case had a consistent-sign
gap shrinking monotonically toward zero as K grew. This one **flips sign** (−1.3pp → +0.6pp)
rather than converging, and the Brier score — this project's own preferred calibration metric —
stays essentially flat (~0.0004–0.0005) across every K tested.

**Verdict.** The original documented −1.4pp rejection was very likely noise-inflated too, matching
the reviewer's own hypothesis. But the corrected picture isn't "confirmed, un-revert it" — it's
"no detectable effect in either direction at n=697." **Recommendation: leave the walk-blend
multiplier reverted**, unchanged from before — just now on the basis of a properly re-tested
current null, not a stale one.

---

## Finding 3 — A safer clip design for the bat-speed HR-share extension

**The claim.** The packet's §8.4 (an evidence-based ranking of where real headroom exists) had
flagged a previously-rejected bat-speed extension to `hr_share_multiplier` as "the single most
evidence-backed concrete lead" — rejected for a 5.5× real-data-limit-test blowup, which the
reviewer argued was an *implementation* problem (a hard clip), not a *signal* problem, and
proposed a more principled fix: shrink the two input shares toward league average by reliability
weight *before* the odds-ratio, rather than clipping the output after the fact.

**Discovery.** This was already fully investigated and resolved **one day before this review
session began** (task #159, referenced directly in the git log). The real root cause was found to
be a pre-existing defect in the *base* 2-term HR-share formula's clip floor (0.02), which produced
up to **5.02×** on real zero-HR-in-sample batters *even with no bat-speed term involved at all*.
The fix (floor raised 0.02 → 0.035) tamed both the base formula (max 2.88×) and the bat-speed
extension (max 2.86×, zero cases above 3×) at once. With the safety issue genuinely fixed, a proper
**full-stack A/B ran at n=8,711 real games**: SU delta +0.0068pp (95% CI includes zero), Brier
delta −0.0004 (95% CI includes zero) — a decisive, honest null, not a blowup-driven rejection. The
bat-speed/pulled-air 4-term extension was reverted again on that basis; only the clip-floor retune
itself (a standalone robustness fix, no accuracy claim) shipped.

**Why the reviewer's design likely wouldn't change the outcome.** The R² gain from the bat-speed
signal *survived* being made safe and still didn't move a full-stack metric — this points to a
real-but-small effect size, not a clip-design artifact. The existing fix's floor only touches the
bottom ~5% of batters; a smoother reliability-weighted shrinkage would touch a similar population
differently, but the null was attributed to effect size, not to *which* batters get touched.

**Fix.** Corrected `MODEL_REVIEW_PACKET.md` §8.4, which had cited a one-day-stale intermediate
state (the audit-table's finding, before task #159 resolved it the next day) — explained the full
resolution and why a different clip mechanism is unlikely to reopen this category. No new compute
was needed; this closed via investigation of already-existing history alone.

---

## Finding 4 — Is the stacked context-multiplier combination well-calibrated?

**The claim.** §10 flagged this as a genuinely open question: does straight multiplication of
independent contextual factors (park × weather × platoon × state × TTO, etc.) stay calibrated in
the tails, or does stacking several boosts/dampers compound into over- or under-confidence? No
published claim on this survived the earlier literature-verification pass in either direction.

**Test built and run.** Fit a per-category attenuation exponent λ via
`unnormalized_probability ∝ p0 × M^λ`, where `p0` is the odds-ratio-combined base probability and
`M` is the product of 5 core context factors (base-out state, batter's own platoon split,
pitcher's own platoon-allowed split, park, times-through-order). Catcher/umpire/defense/weather/HFA
were excluded from `M` for tractability — an explicit, documented scope limit, not a silent one.
Fit via logistic regression with an offset (`logit(p0)`) and single covariate (`ln(M)`) on real
2024 plate-appearance data only — 2023 was deliberately excluded to route around a newly-discovered
cold-start leak (see the bonus finding below) — then evaluated held-out on 2025.

**Result:**

- **home_run**: λ = 0.598, 95% CI (0.451, 0.746) — **clearly excludes 1**.
- **strikeout**: λ = 0.869, 95% CI (0.774, 0.964) — **also excludes 1**.
- **single**: λ = 0.845, 95% CI (0.672, 1.018) — does not exclude 1 (not significant).

Straight multiplication genuinely **is** measurably overconfident in aggregate for the first two
categories — the open question has a real answer, and it's "not calibrated."

**But applying the correction is not an obvious win.** Held-out, full-vector-renormalized
multiclass log-loss improves only trivially in aggregate (1.605455 → 1.605145). The important
part: restricted specifically to the plate appearances that were **actual home runs**, the
λ-correction makes log-loss slightly **worse** (3.4088 → 3.4164) — same worsening direction for
strikeout's own true-positive rows. Only `single` (whose λ wasn't even significant) showed a real
improvement on its own true positives.

**Interpretation.** The aggregate overconfidence is real and is driven by the vast majority of
true-*negative* plate appearances; a single global λ per category doesn't clearly help — and may
slightly hurt — the rare true-*positive* predictions that a prop bettor or calibration check
actually cares about most.

**A sharper, still-untested follow-up this surfaces**: check whether λ<1 is a uniform effect
across the whole range of `M`, or driven specifically by the most extreme-`M` tail — a "cap the
boost, don't dampen everything" design, closer to this project's existing clip-based discipline
than a blanket exponent. This packet's own fit can't distinguish those two shapes.

**Fix.** Updated `MODEL_REVIEW_PACKET.md` §10 with the full real result, replacing the "genuinely
open" framing with the actual, nuanced answer and the sharper follow-up lead.

---

## Bonus finding (not part of the original review) — a second, unfixed cold-start leak

While building Finding 4's factor tables, direct code inspection turned up a 5th instance of the
exact cold-start look-ahead pattern that an earlier correctness audit (task #160) had found and
fixed in four sibling modules:

```python
ref = prior if len(prior) else X[X["season"] == season]   # the buggy pattern
```

`game_simulator.py`'s `build_state_factors_by_season` still has it — apparently missed when the
audit fixed the same pattern in `catcher_framing.py`, `weather.py`, `umpire_factor.py`, and
`ttop.py`. Confirmed by direct code read; **not yet fixed or quantified on real data** (tracked as
an open task — the true first season's state factor may still be leaking a small amount of
look-ahead information into that season's own predictions).

---

## Documents updated as part of this review response

| File | Change |
|---|---|
| `MODEL_REVIEW_PACKET.md` §9 | Corrected Step 7's context-multiplier table (both platoon legs listed together); added the verified reconciliation math |
| `MODEL_REVIEW_PACKET.md` §8.4 | Corrected the bat-speed HR-share "open lead" — now documents the full resolution (task #159) and why it's closed |
| `MODEL_REVIEW_PACKET.md` §10 | Replaced "genuinely open" framing with the real, nuanced λ-fit result and a sharper follow-up lead |

---

## Separately, in production: the "why only 5 of 8 games" investigation and fix

Triggered by a live discrepancy report (site showing 5 real MLB games when 8 were actually
scheduled), traced independently of the review above but resolved in the same session.

### Root cause

`generate_daily_props.py` requires **both** teams' MLB-official probable-pitcher announcements
before generating a game's props; real starters post **2–4 hours before each game's own first
pitch** (research: [lineups.com](https://www.lineups.com/mlb/lineups/),
[sportsbettingdime.com](https://www.sportsbettingdime.com/mlb/probable-pitchers-starting-lineups/)),
not on a fixed daily clock — so a single once-a-day check structurally misses some games every
day. Confirmed live: Washington @ Philadelphia had both pitchers posted by early afternoon, hours
after that morning's cron had already run and skipped it.

### Fix 1 — RotoWire expected-pitcher fallback

- Added `resolve_rotowire_pitcher_id` / `rotowire_pitcher_for_team` to
  `src/ingest/fetch_rotowire_lineups.py` — resolves RotoWire's own listed starter name to a real
  MLBAM player id via `pybaseball.playerid_lookup`, memoized per run, best-effort (returns `None`
  on any lookup failure or ambiguity — never guesses).
- Wired into `generate_daily_props.py`: before giving up on a game for a missing MLB
  probable-pitcher id, tries the RotoWire expected/confirmed starter for that team. Flags the
  source explicitly (`home/away pitcher rotowire-<status>`), persisted via new
  `home_pitcher_source` / `away_pitcher_source` columns (added to `export_to_site_db.py`'s
  `GAME_EXTRA_COLUMNS`).
- **Verified real**: the day's slate went from 5/8 to **8/8** games — the two fully-missing games
  (LAD@CHC, SF@TEX) resolved entirely through RotoWire, correctly flagged as such.

### Fix 2 — light-refresh cadence architecture

The pipeline previously re-did its full, expensive rebuild (re-fetch years of Statcast data,
rebuild every walk-forward rate table) on *every* invocation, making a more frequent cron
prohibitively wasteful. Split it into two modes:

- **`full`** (unchanged behavior): refreshes yesterday's completed games, rebuilds the derived PA
  table, rebuilds every walk-forward rate table (`build_pregame_context`) — then **caches** that
  context to disk (`pregame_context_cache_{date}.pkl`, ~285–300MB) for same-day light runs to reuse.
  Stale-date caches are auto-pruned on every full run (only today's is ever needed).
- **`light`** (new): skips the entire expensive rebuild, reuses the cached context — but *always*
  still refreshes the things that genuinely change intraday: today's schedule/probable pitchers
  (`fetch_schedule_day`, unconditional either way), RotoWire lineups/pitchers, and specifically
  `ctx["game_weather"]` (new `refresh_game_weather()` helper, since real weather posts closer to
  game time and must never be served stale from an old cache).
- **Self-healing**: if `light` mode is requested but no cache exists, falls back to a full rebuild
  automatically — never worse than before, just slower that one time.
- Added `default_run_mode()` to `src/utils/tz.py` — the same hour≥17-ET convention the existing
  `default_slate_date()` already uses, so an `"auto"` sentinel resolves the right mode purely from
  the real Eastern hour at firing time. Also extended `target_date` parsing to accept the same
  `"auto"` sentinel, so a single explicit, self-documenting invocation works:
  `generate_daily_props.py auto 1000 auto`.
- **Verified real**: light mode completed in **~19 seconds** vs. several minutes for a full
  rebuild — roughly a 20× cost reduction per intraday check.
- **`railway.toml`** updated: `startCommand` is now explicit and under version control for the
  first time (previously only set via the Railway dashboard) —
  `python -m src.pipeline.generate_daily_props auto 1000 auto && python -m src.pipeline.export_to_site_db`.
  `cronSchedule` changed from `0 1,14 * * *` (2×/day) to **`0 1,14,17,20 * * *`** (4×/day: 9pm ET
  full rebuild, then 10am/1pm/4pm ET light refreshes) — bracketing when day-game vs. night-game
  lineups/pitchers actually post, rather than one blind morning check.

**Commit**: `f78e33e` — *"MLB: RotoWire expected-pitcher fallback + intraday light-refresh
cadence"* — 5 files changed (`railway.toml`, `src/ingest/fetch_rotowire_lineups.py`,
`src/pipeline/export_to_site_db.py`, `src/pipeline/generate_daily_props.py`, `src/utils/tz.py`),
256 insertions, 40 deletions. Deployed; confirmed live via the Railway API that the deployed
commit's `startCommand`/`cronSchedule` match exactly what was intended.

### Manual one-off fix for that day's already-stale slate

The redeploy finished *after* that day's remaining light-refresh cron slots (1pm/4pm ET) had
already passed — meaning the next *scheduled* cron run would target the following day, never
re-touching the already-stale slate at all. Manually triggered the real production pipeline:
`railway run` (to get the correct production environment variables locally) to regenerate props
with the new code, then a direct Postgres tunnel (Railway's internal DB hostname
`postgres.railway.internal` isn't reachable from outside Railway's own network) to run
`export_to_site_db.py` against the actual live database. **Confirmed directly via `psql`**: all 8
real games were written to production for that slate.

---

## Separately: NFL cron investigation (informational — no code changed)

Triggered by: "why do you have `nfl-worker-sunday` and `nfl-worker-tuesday` — do we need both? Is
`nfl-worker-sunday` broken?"

**Both services are needed by design**, documented in `nfl/MODEL_DOCUMENTATION.md` §11.2: they run
the *identical* `weekly_update.py` + `export_to_site_db.py` pipeline, but at two different points
in the week specifically to support **CLV (closing-line-value) tracking** — Tuesday captures an
"early-week line" snapshot (right after the previous week's Monday-night game wraps and injury
designations start posting), Sunday captures a "gameday-refresh"/closing-line snapshot (catching
Friday's "Final" injury report and closer-to-kickoff weather, both structural no-ops earlier in
the week). The CLV log explicitly *appends* both snapshots rather than overwriting one with the
other. Two separate Railway services (rather than one service with two schedules) exist purely
because Railway doesn't support multiple cron schedules per service — same `nfl/` root directory,
two different config-as-code paths (`railway.sunday.toml` / `railway.tuesday.toml`).

**"Broken" could not be reproduced**, after direct verification:
- Ran `weekly_update.py` locally end-to-end — completes cleanly, correctly auto-detects the next
  unplayed week, generates real Week 1 2026 predictions for every real matchup.
- `export_to_site_db.py` only fails locally due to a missing `DATABASE_URL` (expected outside
  Railway's network) — confirmed via `railway variables` that the real service has it correctly
  configured.
- No failed or crashed deployments found anywhere in `nfl-worker-sunday`'s history.
- Queried the production database directly: real data exists — the `runs` table has one row for
  `2026-wk1`, the `games` table has all 16 real Week 1 games correctly written.
- The one genuine oddity: that single successful run's timestamp (Aug 2, 18:43 UTC) doesn't match
  the configured cron time (13:21 UTC) — most likely an initial bootstrap execution from when the
  service was first deployed, not a scheduled misfire. Railway's own `nextCronRunAt` showed the
  *next* real cron-triggered firing as the following Sunday, meaning the cron mechanism itself
  hadn't actually fired on schedule even once as of the investigation.

No code was changed as a result of this investigation — it answered the question directly.

---

## Separately: Players-tab decimal precision

Small UI change to `site/app/templates/_leaderboard.html`: stat-line columns (`mean_hr`,
`mean_hits`, etc.) went from one decimal to two (`%.1f` → `%.2f`, e.g. `1.6` → `1.62`);
probability columns (e.g. NFL's Anytime TD market) went from a whole percent to one decimal
(`round()`% → `%.1f`%, e.g. `34%` → `34.2%`). Scoped specifically to the Players/leaderboard tab,
not the per-game card view, per the request. **Commit `999e435`** — deployed, confirmed live.

---

## Net summary of what changed, by file

| File | What changed |
|---|---|
| `mlb/MODEL_REVIEW_PACKET.md` | 3 sections corrected with verified real numbers (§9, §8.4, §10) |
| `mlb/src/ingest/fetch_rotowire_lineups.py` | New RotoWire pitcher-name → MLBAM-id resolution |
| `mlb/src/pipeline/generate_daily_props.py` | RotoWire pitcher fallback; full/light/auto mode split; context caching + pruning |
| `mlb/src/pipeline/export_to_site_db.py` | New `home_pitcher_source`/`away_pitcher_source` columns |
| `mlb/src/utils/tz.py` | New `default_run_mode()` |
| `mlb/railway.toml` | Explicit `startCommand`; cron cadence 2×/day → 4×/day |
| `mlb/src/models/platoon_splits.py` | **Live correctness fix**: shared league-average platoon default corrected from `league_mult**0.5` to `league_mult**0.25`, resolving a real double-counting bug (Finding 1 follow-up) |
| `mlb/src/models/bullpen.py` | CRN-keyed opt-in for closer/bullpen-pick sampling (Finding 2 infra) |
| `mlb/src/models/weather_forecast.py` | CRN-keyed opt-in for weather-bucket sampling (Finding 2 infra) |
| `mlb/src/models/validate_game_simulator.py` | Walk-blend multiplier re-wired behind an opt-in flag (Finding 2 retest infra) |
| `site/app/templates/_leaderboard.html` | One more decimal of precision, Players tab only |

## Open items for a future session

- **Bonus finding**: fix `build_state_factors_by_season`'s cold-start leak (5th instance of an
  already-fixed pattern elsewhere) and quantify its real-data impact, same discipline as the
  original 4-module fix. It contaminates 2023, which sits inside the canonical 2023–2025 backtest
  window — but it affects both arms of any paired A/B identically, so it's neutral to comparisons
  like the platoon fix's own guardrail above; fix it before quoting any absolute (non-paired)
  backtest number.
- **Platoon fix's own residual**: the structural fix closed only ~45–46% of the strikeout
  calibration gap (vs. ~81%/42% on the home_run side). Expected, not a failure — likely a second,
  smaller strikeout-specific effect (framing/umpire interaction with handedness, or the
  batter-side split's own shrinkage target) out of scope for this fix specifically.
- **Finding 2, deferred half**: `whiff_rate_multiplier` was fully deleted, not kept dormant — a
  from-scratch reconstruction (from its documented description) would be needed before it could be
  retested the same way the walk-blend was.
- **Finding 4's sharper follow-up**: test whether the stacked-multiplier overconfidence is
  concentrated in the extreme-`M` tail specifically (a "cap the boost" design) rather than uniform
  across the whole range — the current fit can't distinguish the two shapes.
