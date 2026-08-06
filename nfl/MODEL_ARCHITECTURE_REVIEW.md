# NFL Prediction Model — Architecture & Methodology Review

**Prepared for external technical review (Fable 5, research mode). Written ~1 month ahead of
2026 season kickoff.**

This document is a standalone technical reference to how this NFL prediction model actually
works today: the math, the algorithms, the data flow, the sources, the assumptions, and —
explicitly — what's validated versus provisional. It assumes no prior context. The project's
own living record, `MODEL_DOCUMENTATION.md` (~2,100 lines), is the chronological source of
truth this document is distilled from; that file also contains the full historical narrative
(what was tried, in what order, and why) that this document deliberately compresses into a
"kept vs. rejected" ledger (§14) rather than retelling in full.

---

## 1. Executive summary

The model predicts, for every NFL regular-season game: **margin, total, moneyline/win
probability, and a full player-props statline for every rostered skill player and QB on both
teams.** It runs automatically twice a week (Tuesday early-week refresh, Sunday
gameday-refresh) via a scheduled pipeline, and is deployed on Railway as of 2026-08.

**What's real and validated:**
- Team power ratings, player usage-share engines, and TD-rate engines are all walk-forward
  validated with real, positive, statistically meaningful signal (§4, §8).
- Margin and total both use the **real market closing line directly** as their base — not a
  blend — because a rigorous statistical test (ATS%/O-U% with bootstrap confidence intervals,
  not raw MAE) showed that blending in this model's own prediction added no measurable edge
  over the market alone, and in margin's case, added a real negative bias (§5).
- A Monte Carlo drive-state game simulator produces a full predictive *distribution* over
  final scores (not just a point estimate), recentered onto the market-based point estimate so
  the displayed spread and the derived moneyline never contradict each other (§10).
- Player props have real, validated volume/scoring signal (target share, carry share, TD-rate
  engines) and a real, validated injury-driven reallocation mechanism for running backs (§9).

**The one honest, open caveat, stated plainly because it's the model's entire specifically-claimed
game-side betting edge:** a cornerback-injury margin adjustment (§7) survived four rounds of
increasingly skeptical statistical review. Its direction is real and its lookahead risk has
been ruled out by direct experiment, but the strongest available test — a permutation test
that re-runs the *entire specification search* that produced the coefficient, not just the
winning configuration — puts the result at only the **95th percentile** of the correct null
distribution, not comfortably beyond it. This is held as a provisional, real-looking effect,
not a proven one, and the production coefficient has been deliberately shrunk toward the most
conservative honest estimate available as a result.

**A current, real, and temporary data gap**, as of this writing: 2026 real injury reports and
snap-count data are not yet published by the upstream data source (still returns HTTP 404).
Practical effect: the cornerback-injury mechanism above is **completely inert for Week 1**
except for one verified manual override. This is not a bug — the pipeline degrades gracefully,
with no false positives — but it means this week's predictions don't yet reflect any real
in-season injury news beyond that single entry. This should resolve on its own as the season
approaches and the upstream data source begins publishing.

---

## 2. System architecture

```
                          ┌─────────────────────────────────────────┐
                          │            RAW DATA SOURCES              │
                          │  nflreadpy: schedules (incl. real        │
                          │  closing lines), play-by-play, weekly    │
                          │  rosters, injury reports, snap counts,   │
                          │  draft picks                             │
                          │  + one manually-curated offseason PDF    │
                          │    (rookie/depth-chart projections)      │
                          │  + a small manual-override JSON file     │
                          │    (verified breaking news)               │
                          └───────────────────┬───────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   ┌───────────────────┐        ┌─────────────────────────┐      ┌────────────────────────┐
   │ LAYER 1: team      │        │ PLAYER USAGE ENGINES     │      │ MARKET LINE (real      │
   │ power ratings      │        │ (ShareEngine, TdRateEngine,│    │ closing spread/total)  │
   │ (recursive EPA,    │        │  QbRatingEngine)          │      │                        │
   │ opponent-adjusted) │        │                          │      │                        │
   └─────────┬──────────┘        └────────────┬─────────────┘      └───────────┬────────────┘
             │                                │                               │
             │ (total-only input now;         │                               │
             │  margin fallback only if       │                               │
             │  no market line published)     │                               │
             ▼                                │                               ▼
   ┌────────────────────┐                     │                  ┌─────────────────────────┐
   │ Total calibration    │                    │                  │  MARGIN = market line     │
   │ (regression on Layer1 │                   │                  │  + QB-swap adjustment      │
   │  total signal + roof) │                   │                  │  + CB-injury adjustment    │
   └──────────┬───────────┘                    │                  └──────────────┬────────────┘
              │                                │                                 │
              └──────────────┬─────────────────┘                                 │
                              ▼                                                   │
                  ┌────────────────────────┐                                     │
                  │ GAME-SCRIPT PASS RATE   │                                     │
                  │ (market-implied team    │◄────────────────────────────────────┘
                  │ total conditions pass   │
                  │ rate & drives simulator)│
                  └───────────┬─────────────┘
                              ▼
        ┌──────────────────────────────────────────┐
        │        PLAYER PROPS PIPELINE               │
        │  volume (targets/carries) → per-touch rate  │
        │  engines → TD probability → auto-refit      │
        │  linear calibration → injury reallocation   │
        └──────────────────────┬───────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                     ▼
  ┌────────────────────────┐          ┌───────────────────────────┐
  │ MONTE CARLO GAME         │          │ MONTE CARLO PROPS           │
  │ SIMULATOR (drive-state,  │          │ SIMULATOR (bootstrap        │
  │ recentered on market-    │          │ standardized residuals,     │
  │ based margin/total)      │          │ touch-count distributions)  │
  └────────────┬─────────────┘          └──────────────┬───────────────┘
               │                                        │
               ▼                                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                            OUTPUTS                                │
  │  predictions_{season}_wk{week}.parquet  (margin/total/moneyline)  │
  │  props_{season}_wk{week}.parquet         (full statlines)         │
  │  predictions_page_{season}_wk{week}.html (rendered UI)            │
  │  line_snapshots.parquet                  (self-generated CLV log) │
  └─────────────────────────────────────────────────────────────────┘
```

Everything downstream of the market line is **walk-forward, no-lookahead**: every engine's
`.predict()` is called *before* `.update()` for any given game, and historical validation
always uses a strict TRAIN/TEST split with TEST touched deliberately, not accidentally.

---

## 3. Data sources

| Source | What it provides | Refresh behavior |
|---|---|---|
| `nflreadpy` schedules | Game metadata, **real historical closing lines** (`spread_line`, `total_line`) with 100% coverage 2016-2025, real final scores once played | Fetched fresh every pipeline run, `force=True` |
| `nflreadpy` play-by-play | Every play, EPA per play, drive-level fields (`fixed_drive`, `fixed_drive_result`, `drive_time_of_possession`) | Fetched fresh; falls back to `seasons[:-1]` if the newest season has no data yet (handled gracefully) |
| `nflreadpy` weekly rosters | Real per-team, per-week roster status (`ACT`/`PUP`/`RES`/`SUS`/etc.) | Same fallback behavior |
| `nflreadpy` injury reports | `Out`/`Doubtful`/`Questionable` designations | Same fallback behavior — **still no 2026 data as of this writing**, see §1 |
| `nflreadpy` snap counts | Real per-player snap percentages, used to rank a team's top cornerbacks by usage | Same fallback behavior — **still no 2026 data as of this writing** |
| `nflreadpy` draft picks | Real draft position (round, pick) for every rookie, linked by real player ID (`gsis_id`), no fuzzy matching needed | Fetched fresh every run |
| *(Migrated from `nfl_data_py` 2026-07-08 — `nfl_data_py` was archived read-only 2025-09-25, deprecated in favor of `nflreadpy`; see MODEL_DOCUMENTATION.md §2.2.1)* | | |
| A manually-extracted offseason PDF (one analyst's preseason player projections) | Bootstraps true rookies / new-team signings before any real in-season data exists for them; also detects offseason starting-QB changes | One-time per-season extraction, frozen thereafter — **an explicitly known annual maintenance dependency**, not automated |
| `data/manual_overrides/known_outs_2026.json` | Verified breaking roster news the automated data hasn't caught up to yet (e.g., a real, dated PUP-list entry, web-verified before being added) | Manually maintained, reviewed each pipeline run for staleness |

Every fetch is idempotent (skips the network call if a cached file already exists, unless a
fresh pull is explicitly requested) and every one of the season-completeness-dependent sources
degrades gracefully — if the current season's data isn't published yet, the pipeline falls back
to the prior complete season range rather than crashing or silently guessing.

---

## 4. Layer 1: recursive opponent-adjusted EPA power ratings

Each team carries two live numbers: an **offensive rating** (EPA/play generated, relative to
league average) and a **defensive rating** (EPA/play allowed, relative to league average).

**Prediction** (`nets`): for a game between `home` and `away`,

```
home_net = off_rating[home] + def_rating[away]
away_net = off_rating[away] + def_rating[home]
rating_diff = home_net - away_net
total_signal = home_net + away_net
```

**Update**, after observing the real per-team offensive EPA/play in a game (`off_epa_a`,
`off_epa_b` for the two teams):

```
target_off_a = off_epa_a - def_rating[b]      # this game's implied offensive rating for A
target_def_b = off_epa_a - off_rating[a]      # this game's implied defensive rating for B
target_off_b = off_epa_b - def_rating[a]
target_def_a = off_epa_b - off_rating[b]

new_off[a] = (1 - α) * off_rating[a] + α * target_off_a      # exponential recency weighting
new_def[b] = (1 - α) * def_rating[b] + α * target_def_b
new_off[b] = (1 - α) * off_rating[b] + α * target_off_b
new_def[a] = (1 - α) * def_rating[a] + α * target_def_a
```

At the start of each new season, every team's ratings shrink toward zero (league average):
`off_rating *= (1 - off_shrink)`, `def_rating *= (1 - def_shrink)`.

**Current tuned hyperparameters** (nested grid search: fit on a training window, select on a
validation season, confirm once on held-out test — never re-tuned on the same data twice):
`α = 0.06`, `off_shrink = 0.20`, `def_shrink = 0.50`. A later continuous re-optimization
(`scipy`, not the discrete grid) was tested and **rejected** — the coarse grid generalized
better than the fine continuous search, a real overfitting-to-a-small-validation-season
finding.

**Layer 1's remaining live role, post-market-line-adoption**: since margin now uses the real
market line directly (§5), Layer 1's own margin prediction only fires as a fallback when no
market line has been published yet for a game (rare, only very-far-future games). Its live,
routine role is as one input (alongside roof type) to the **total** calibration:

```
predicted_total = total_coefs[0] + total_coefs[1] * total_signal + total_coefs[2] * is_indoor
```

fit by ordinary least squares on the forward-looking calibration window (most recent four
complete seasons). Four separate signal-quality improvements to the underlying EPA calculation
(early-down-only EPA, winsorizing extreme plays, turnover-luck neutralization, a success-rate
co-input) were tested against this specific role and came back **null or actively harmful**
(winsorizing in particular: real explosive-play/turnover tails are a persistent team trait, not
noise — clipping them destroyed signal). None of the four are live.

---

## 5. Margin & total: the market-line decision

**Current production formula** — no blend of any kind:

```
margin_base = spread_line                (real market closing line, home-team-signed)
total_base  = total_line                 (real market closing line)
             (falling back to Layer 1's own base_pred/predicted_total ONLY if no market
              line is published yet for this specific game)

our_margin = margin_base + SWAP_B_MARKET * (home_qb_swap − away_qb_swap)     [§6]
             + injury_adjustment                                             [§7]
our_total  = total_base
```

**Why there's no blend.** Earlier versions of this model blended the market line with this
model's own Layer-1-based prediction via a Ridge-regularized regression
(`intercept + our_weight·our_pred + market_weight·market_line`). A rigorous evaluation panel —
ATS win-rate with a bootstrap 95% confidence interval, signed bias with its own CI, and CRPS
(a proper scoring rule rewarding both accuracy *and* honest uncertainty) — replacing raw MAE as
the decision metric, found:

| | MAE | ATS% vs. closing line | Signed bias (95% CI) |
|---|---|---|---|
| Our model alone (margin) | 10.000 | 49.05% [46.1, 52.1] | −1.240 [−1.97, −0.47] |
| Market alone (margin) | 9.494 | — (trivial) | −0.574 [−1.27, +0.20] |
| Blend (former production) | 9.533 | **49.62% [46.6, 52.6]** | **−0.986 [−1.70, −0.22]** |

MAE alone made the blend look nearly indistinguishable from the market. The panel showed two
things MAE hid entirely: the blend's ATS win rate is statistically indistinguishable from a
coin flip and *below* the ≈52.4% breakeven against standard vig, and the blend carries a real,
CI-excludes-zero **negative** bias that market-alone does not. **Decision: for margin, the
blend is removed entirely.**

The identical panel was then run on **total**, closing an asymmetry (totals had only ever been
graded on MAE + a single p-value). A follow-up review correctly caught that the total result
had been mis-characterized: the total blend's own weight (~0.06–0.17) implies a
disagreement-with-the-line standard deviation of only ~0.3 points, and even a perfectly
informative signal of that size could only move O/U% to ~50.6% — a value the observed
confidence interval [47.2%, 53.0%] comfortably contains. **The test had no statistical power to
distinguish real signal from noise at that weight** — a materially different, weaker claim than
"decisively failed." The decision to remove the total blend still stands (no measurable
contribution to MAE/CRPS either, and it beat market-alone in only 2 of 4 rolling-origin folds —
a coin-flip rate), but the *stated reasoning* was corrected. This produced a general rule now
applied to any future use of this kind of test: **before treating an ATS/O-U result as
decisive, compute the component's best-case win rate at its actual weight; if that best-case
sits inside the observed confidence interval, the test lacked the power to be decisive, and the
decision must rest on bias or fold-consistency instead.**

---

## 6. QB-swap adjustment

**Diagnosis**: Layer 1's team rating updates slowly by design (`α=0.06`, meant to be sticky), so
when a team's starting QB changes — injury, benching, a bye-week return — the very next game is
predicted using a rating that still reflects the *old* starter. Measured: MAE on QB-change games
is 10.70 vs. 9.79 for no-change games, with a systematic +4.2-point bias specifically when the
*away* team's QB changes.

**Mechanism** (`QbRatingEngine`): a per-QB, recency-weighted EPA/dropback rating,
`α_QB = 0.15`, season-shrink `0.15` (gentler than team-level shrink, since QB skill is more
persistent than team context), gated on a minimum of 5 attempts so mop-up snaps don't move it.
Ground-truth starters come from the schedule's own `home_qb_id`/`away_qb_id`, not a snap-count
heuristic. A one-game **swap delta** —

```
swap_delta = new_starter_rating − previous_starter_rating
```

— is applied only on the first game after a detected change (zero otherwise; Layer 1's own
recursive update naturally catches up after that).

```
our_margin += SWAP_B * (home_qb_swap − away_qb_swap)
```

**Two different values of `SWAP_B` exist for the same physical quantity, deliberately, and this
is documented explicitly to prevent them from ever being confused or "synced":**
- `SWAP_B_LAYER1 = 6.616` — the coefficient as originally fit, against the **pure Layer-1
  residual**. Used only by a frozen, one-time historical reference script that never blends
  with the market.
- `SWAP_B_MARKET = 2.970` — refit against the **market-line-based residual**, since applying
  the original 6.616 value on top of an already market-based margin would double-count
  information the market already prices in. This is the live production value.

Both constants are named for their residual basis rather than left as an ambiguous bare
`SWAP_B` in two different files — this exact "same name, different meaning depending on
context" shape is the identical bug class as a well-known `np.polyfit` argument-order mistake
that bit this project three separate times before a shared, sign-asserting helper eliminated it
project-wide (see §13).

An **offseason bootstrap** (Week 1 only, using the one-time analyst PDF projection as the source
of incoming-starter information, since the schedule data has no starter QB for any unplayed
game) detects likely offseason QB changes by comparing each team's most-frequent 2025 starter
against the analyst's projected 2026 starter. This naturally stops mattering the moment a team
actually plays its first real 2026 game — `build_starter_sequence` picks up the real starter
from the schedule data at that point.

---

## 7. Injury adjustment — the cornerback mechanism (the model's one claimed game-side edge)

This is, after §5's blend removals, **the entire remaining live betting edge of the game-side
margin model** — the closing line contributes no edge by construction, and everything layered
on top of it is either this mechanism or §6's QB-swap adjustment (which itself did not clear
statistical significance on a direct ATS test — see below).

### 7.1 Current production formula

```
margin_adjustment = intercept + coef · (away_cb_out − home_cb_out)

intercept   = −0.018
coef        = ±2.446           (symmetric: one coefficient, opposite sign per side)
```

`away_cb_out`/`home_cb_out` are 0/1 flags: does this team have at least one of its **top-3
cornerbacks by real, season-to-date snap share** currently listed with an "Out" designation on
the real injury report?

### 7.2 How the coefficient was arrived at, and why it changed three times

- **v1** (pooled fit against the pure Layer-1 residual): included separate coefficients for
  away-skill-player-out, away-OL-out, home-CB-out, and away-CB-out. Home-skill and home-OL
  terms were dropped early (unstable sign/noise).
- **v2** (refit against the *blend* residual, since a review correctly identified that applying
  v1's coefficients on top of an already market-blended prediction double-counts information):
  every coefficient shrank; only the two CB terms remained individually significant
  (home_cb t=−2.43, away_cb t=+3.27).
- **v3** (symmetric constraint): reparameterizing the two independent CB terms as **one**
  coefficient on `(away_cb_out − home_cb_out)` — forcing home/away effects to be equal and
  opposite, which is the theoretically correct constraint and roughly doubles the effective
  sample size for that one coefficient — raised the pooled t-statistic from ±2.43/3.27
  (independent) to **+4.01** (symmetric), and a strict walk-forward check confirmed this
  generalizes better, not just fits better in-sample (CB-flagged holdout MAE 9.968→9.836, signed
  bias +1.596→+0.542). Production value at this point: **±2.977**.

### 7.3 The resubstitution correction (the most important methodological event in this project)

The v3 coefficient (±2.977) was fit on **pooled 2018–2025 data** — which includes the
2022–2025 window that an ATS evaluation subsequently scored it against. That is
**resubstitution**: the coefficient was fit on (part of) the very data used to evaluate it. The
original evaluation reported **62.4% ATS, 95% CI [55.4%, 68.8%]** and treated this as decided.
It was not an honest out-of-sample test, and the CI was computed as though it were one.

A **rolling-origin walk-forward refit** — refitting the coefficient using only strictly-prior
seasons for each of four test-season folds — gives the only honest, out-of-sample estimates:

| Train seasons | Test season | Fold coefficient |
|---|---|---|
| 2018–2021 | 2022 | 1.590 |
| 2018–2022 | 2023 | 2.353 |
| 2018–2023 | 2024 | 2.539 |
| 2018–2024 | 2025 | 2.629 |

**Every single honest fold estimate sits below the pooled production value of 2.977.**

Scoring each fold's own test season using *only that fold's own* coefficient (true
out-of-sample), pooled across all four folds, gives a **walk-forward ATS of 61.8%** — nearly
identical to the flawed resubstitution number (62.4%). This closeness was initially
misinterpreted as "vindication." It is not: ATS is a pure sign-of-disagreement statistic — for a
single-signed adjustment on top of a market line, it is *entirely* magnitude-independent, and
every fold coefficient above has the same sign. Walk-forward and resubstitution ATS are very
nearly the *same statistic* (the raw win rate of "bet against the team missing a top-3
cornerback"), not independent evidence that the in-sample fit was sound.

### 7.4 The permutation tests — pricing in the specification search

A first permutation test shuffled each team's real CB-out flags across its own played weeks
*within season* (preserving that team's real annual flag count exactly), re-ran the identical
pooled-fit-then-score procedure 1,000 times, and located the real number in the resulting null
(mean 54.3%, std 3.9%). The real resubstitution ATS sat at the **97.7th percentile** of that
null. This was treated as settling the question. It does not: that null only prices in "fit and
scored on the same games," not the **specification search** that chose this exact configuration
(top-N cornerbacks for N ∈ {2,3,4,5} × three injury-report designation sets × a
symmetric-vs-independent choice) out of the family actually explored. The back-of-envelope
expected *maximum* of 10–15 independent noise draws from that null lands at 60.3–61.1% — close
enough to the observed 61.8–62.4% to be genuinely uncomfortable.

**Extending the permutation test to re-run the full specification search per shuffle** (for
each of 300 shuffles: re-run a 12-configuration grid search — corner-count × designation-set,
selected by the same |t-statistic| criterion that historically chose the production
configuration — then score *that shuffle's own winning configuration's* ATS) gives the correct
reference: the null of "the best result this whole search procedure finds on pure noise." That
null has mean 55.3%, std 3.8%, **95th percentile 61.8%**. The real walk-forward number (61.8%)
sits at **almost exactly the 95th percentile** — right at the edge of conventional
significance, not comfortably beyond it.

A follow-up check on why the simpler null centers at 54.3% rather than the 50% a true null
"should" show: re-running the permutation with walk-forward (not resubstitution) fitting moves
the null's center to 52.4% — confirming the resubstitution advantage explains *part* of the
effect, but a smaller residual gap from 50% remains unexplained (a real, dated, between-team
variation in CB-flag incidence is a plausible partial contributor, not fully pinned down).

### 7.5 Staking implication

Because ATS is magnitude-insensitive, shrinking the point-estimate coefficient (see below)
could not and did not move the observed ATS number — even though the shrink was explicitly
motivated by concern about overstating edge for stake sizing. The coefficient itself (fit on
the *magnitude* of ~2,127 pooled residuals) is the right basis for sizing a bet, with its own
confidence interval — not the sign-only ATS statistic, which uses only 186 binary outcomes and
therefore contains far less information. **±2.446 × ~3% ATS-per-point ≈ 57.3% implied edge** is
the number that should inform staking, not the observed 61.8% ATS.

### 7.6 Current production decision and honest status

**Coefficient shrunk from ±2.977 to ±2.446** — the rolling-origin fold *median* — as an
asymmetric-downside precaution on the point estimate specifically (every honest fold sits
below the pooled value; using an inflated coefficient risks a systematic ~0.5-point error on
the ~46 CB-flagged games/season this mechanism fires on). This was checked directly for a
side-effect cost and found to have essentially none: computed against the *current* production
formula, signed bias on the CB-flagged holdout is −0.282 at ±2.977 vs. −0.290 at ±2.446 —
functionally unchanged.

A snap-share-rank discriminator (does the effect concentrate on a team's #1 cornerback and
taper for #2/#3, as the stated "cornerback quality" mechanism would predict?) found **no such
pattern** — two-thirds of flags fire on the *third*-ranked corner, and the effect doesn't
concentrate on the first. This is genuinely weak evidence against the specific stated mechanism,
not merely inconclusive, though the per-rank samples (37–103 games) are small.

A direct lookahead audit — truncating the underlying snap-count data to only weeks ≤10 of a
real season and confirming the resulting week-10 cornerback rankings are byte-identical to the
same computation run on the full season (zero mismatches across 96 team-week-player rows) —
confirmed the ranking is genuinely season-to-date, not using future information.

**Honest bottom line**: real-looking, direction supported, lookahead risk ruled out by direct
experiment, but the evidence that corroborates it *independent of the resubstitution error* is
weaker than earlier reporting claimed. Held provisionally.

---

## 8. Player usage engines

### 8.1 Share engines (target share, carry share)

A short-memory, per-player exponentially-recency-weighted share:

```
new_share[player] = (1 − α) · share[player] + α · observed_share_this_week
```

`α = 0.30` (roughly a 3–4 game effective memory — usage roles change far faster than team
strength), season-shrink `0.35`. Critically, the update is **gated**: a week where the player
had zero statistical involvement (bye, injury, healthy scratch) leaves the rating unchanged
rather than wrongly treating it as "this player's role went to zero."

Validated on held-out data: target share correlation **0.74**, carry share correlation
**0.88** — both well above a flat naive baseline.

### 8.2 TD-rate engine — Bayesian/Laplace shrinkage

```
predicted_rate(player) = (career_TDs + prior_weight · league_rate) / (career_touches + prior_weight)
```

`prior_weight` controls how many "pseudo-touches" of league-average the estimate is anchored
by — a player with few career touches reverts almost entirely to league average; a
high-volume veteran's own real rate dominates.

### 8.3 Empirical-Bayes derivation of `prior_weight`

Originally a swept constant (raised from an unvalidated 15 to 30 after a concrete real-world
failure: a receiver with 5 TDs on 22 career targets still projected at 2.6× league average even
after shrinkage — an overconfident, small-sample outlier the sweep should have caught but
didn't, because the sweep only checked an *aggregate* calibration metric that stays flat across
a huge range of `prior_weight` values and is blind to individual-outlier overconfidence).

The current method replaces the swept constant with a closed-form, method-of-moments
derivation (a DerSimonian-Laird-style random-effects estimator, applied per-player rather than
per-study):

```
μ̂           = Σ(hits) / Σ(touches)                              # league rate
v_obs        = touch-weighted variance of each player's own rate  # total observed variance
v_sampling   = touch-weighted average of (per_touch_variance / touches)   # expected NOISE alone
v_signal     = max(v_obs − v_sampling, ε)                         # real between-player variance
prior_weight = per_touch_variance / v_signal
```

Interpretation: **"how many touches of league-average pseudo-data is one real touch worth,"**
derived directly from the real shape of the population rather than swept against a metric that
can't see the failure mode it's meant to catch. `per_touch_variance` is `μ(1−μ)` for a
Bernoulli-shaped rate (TD rate, catch rate).

**Current fitted values** (refit fresh from real historical data every pipeline run — this is
one of several constants deliberately *not* frozen, unlike §7's coefficient, because it's a
cheap, mechanical, closed-form recomputation with no judgment call involved): receiving TD rate
≈ **269**, rushing TD rate ≈ **194**, catch rate ≈ **41**. All roughly 6–9× the old swept
constants. A follow-up check (does this over-shrink high-volume players, since the population
is dominated by low-touch players whose own variance is mostly noise?) found the *opposite*:
restricting the fit to players with ≥100 career touches moved the estimate **up**, not down —
if anything the current fit is mildly conservative for the high-volume tail, not aggressive.

### 8.4 QB passing statline & wind adjustment

A parallel structure (completions, yards, TDs, interceptions) uses the same shrinkage pattern
with swept (not yet empirical-Bayes-refit) prior weights.

**Wind adjustment** (`rate = intercept + slope · wind_speed`, fit by ordinary least squares on
outdoor games only, applied as a no-op indoors or when no forecast exists yet): the fit itself
operates on the rate (yards/attempt, yards/target); validated on the resulting projected total
yards for the game (rate × volume) for QB passing yards (MAE 44.72→44.09, p=0.0022) and WR/TE
receiving yards (MAE 13.55→13.31, p<0.0001, much larger sample). Completion rate showed the
same correctly-signed raw correlation but did not clear significance (p=0.34) and was **not**
shipped — a real, if directionally suggestive, negative result kept out of production rather
than added on a hunch.

---

## 9. Player props pipeline

### 9.1 Volume: game-script-adjusted pass rate

Each team has a recency-weighted **neutral-script pass rate** (α=0.15, season-shrink 0.5,
anchored at league average 57%), adjusted for this specific game's expected script:

```
adj_pass_rate = neutral_pass_rate + game_script_adjustment
game_script_adjustment = intercept + margin_coef·team_margin + total_coef·implied_team_total
```

fit by OLS on real historical (team_margin, implied_team_total, actual pass-rate residual)
triples. `implied_team_total` (`total_line/2 ± spread_line/2`) — the market's own implied
expectation for this specific team's scoring, continuous and already opponent-adjusted —
replaced an earlier, coarser 3-bucket EPA-tercile signal, a real, validated improvement
(MAE 0.07141→0.07080, p=0.0406, same-direction in all four individual test seasons).

```
pass_attempts = LEAGUE_AVG_PLAYS · adj_pass_rate
rush_attempts = LEAGUE_AVG_PLAYS · (1 − adj_pass_rate)
proj_targets  = pass_attempts · player_target_share
proj_carries  = rush_attempts · player_carry_share
```

`LEAGUE_AVG_PLAYS` is recomputed fresh from the most recent four complete seasons on every
pipeline run (confirmed real drift over time: full-history mean 62.40 plays/team-game vs.
61.04 for the two most recent seasons alone — a real pace-of-play trend, not noise; this
constant was found hardcoded and never refreshed in three separate files during a systematic
audit, and fixing it is documented as one of the highest-value single findings of the entire
review cycle).

### 9.2 TD probability and its calibration

```
raw_td_prob = 1 − exp(−(proj_targets·rec_td_rate + proj_carries·rush_td_rate))
td_prob     = clip(A + B · raw_td_prob, 0, 1)
```

`(A, B)` is an **auto-refit linear calibration**, refit fresh every pipeline run from real
historical (raw_td_prob, actual_td) pairs, decile-bucketed the same way the very first version
of this correction was originally derived. Current live values: **A ≈ 0.042, B ≈ 0.820**.

This replaced two prior approaches, in order, each rejected for a specific, documented reason:
1. A **hand-fit, frozen** linear map — a real, live bug: it was fit when the upstream TD-rate
   `prior_weight` was still the old swept value (15), and silently went stale once §8.3's
   empirical-Bayes refit changed the upstream shrinkage — exactly the "downstream calibration
   coupled to an upstream shrinkage it was fit against" bug class this project has now
   generalized into a standing audit practice (§13).
2. **Isotonic regression** — more flexible than a 2-parameter line, so it can represent the
   monotone-but-nonlinear miscalibration a straight line structurally cannot. Tested honestly on
   a strict walk-forward split (both fit on train seasons, scored on held-out test seasons):
   isotonic *underperformed* the linear refit by roughly 2× (top-decile overshoot +0.032 vs.
   +0.015) — a real, surprising negative result, plausibly isotonic overfitting a small
   training sample's specific tail noise. It was nonetheless kept for one round on a
   full-history comparison where the test data was inside the fit set — **the same
   resubstitution-flavored evidence structure §7 explicitly rejected for the cornerback
   coefficient**, caught as a real self-consistency problem and corrected. The genuinely strong
   argument for either mechanism — automatically retiring the stale-coupling bug class — comes
   from refitting *every run*, not from isotonic specifically; a 2-parameter linear refit
   captures the same structural benefit with better demonstrated out-of-sample behavior, so
   that's the current default. (Beta calibration — 2–3 parameters, monotone by construction, a
   deliberate middle ground — was tested as an upgrade candidate and did not beat linear either.)

### 9.3 Injury detection and reallocation

Two independent detection paths, since neither alone suffices:
1. **Real roster status** (`PUP`/`RES`/`SUS`/etc. from weekly rosters) — automatic, no manual
   step, but reflects the upstream data source's own publication lag.
2. **A manual override file** for verified breaking news the automated data hasn't caught up to
   yet — each entry web-verified before being added, removed once real data catches up.

**Position-specific reallocation**, validated against real historical "season-long lead player
ruled Out" instances, not assumed:
- **RB carries**: proportional reallocation among remaining active backs is a real, significant
  improvement (MAE 4.49→3.71, ~17% better, p=0.0003, n=443) — this project's single strongest
  validated result. Renormalization is proportional: `scale = total_share / active_share`,
  applied to every remaining active back.
- **WR/TE targets**: the *same* mechanism makes predictions **worse** (MAE 1.93→2.32,
  p=0.0001, n=513) — vacated targets don't redistribute cleanly the way carries do. Simply
  excluding the injured player, with no reallocation, is the validated-better default here.

A **draft-capital rookie prior** (Bayesian-shrunk expected share by `(position, round)` bucket,
fit on every real rookie 2016–2025, linked by real draft-pick data — no fuzzy name matching
needed) closed a real, previously-existing gap: a true rookie with zero NFL history and no
analyst-PDF coverage previously received *zero* projected volume, silently. Validated via
leave-one-season-out: beats a naive position-mean baseline by ~18% on WR/TE target share and
~16% on RB carry share (both p<0.0001).

---

## 10. Monte Carlo game simulator

A drive-state simulator, architecturally modeled on (not copying) a sibling MLB project's
Monte Carlo pattern: combine context into a distribution → bootstrap-resample a real historical
state transition → loop to a terminal condition → aggregate across many trials.

**Mechanics**: a game is a strict alternating sequence of drives (every real drive-ending event
changes possession, so only *start field position* needs modeling, never "who's up next").
Each drive's outcome (result category, points, time elapsed) is bootstrap-resampled from real
historical drives matching (start field-position bucket, this team's market-implied-total
quantile) — continuous and already opponent-adjusted, replacing an earlier, coarser
EPA-tercile conditioning signal that was tested and found to underperform.

**The half-boundary mechanism** (the most mechanically subtle piece): drives that end because
the clock expired (not a score or turnover) have two properties, and conflating them caused a
real, documented back-and-forth:
- Their **duration** is a genuine artifact (mean 42.6s vs. 165.8s for every other result
  category, ~7% of the real drive pool) — an early fix correctly excluded these rows from the
  *duration* sampling pool.
- Their **outcome** (nearly always zero points) is a completely real, common NFL event — a
  first attempt at the fix excluded these rows *entirely*, which fixed the duration artifact
  (simulated drive count fell from 24.22 toward the real 22.11) but then made simulated
  **total** accuracy slightly *worse*, because the surviving pool's average points-per-drive
  rose along with its now-accurate duration.

The corrected mechanism keeps sampling **duration** from the clean, duration-uncontaminated
pool, but samples **outcome** from the full pool including the clock-truncated drives, gated by
an explicit clock check: if a candidate drive's own sampled duration would exceed the half's
remaining time, the real generating process is "this drive got cut off" — so it's resolved from
a *separate*, real empirical end-of-half point distribution instead of a normal independent
draw. Result: total MAE improved from 11.62 (original) through 12.28 (the interim,
duration-only fix, genuinely worse) to **10.49** (the corrected mechanism) — close to the
top-down model's own 10.20, with margin essentially unchanged throughout.

**Recentering** (the most important production-facing fix): the simulator's own drive dynamics
land wherever they land (margin MAE 9.69 vs. the market-based point estimate's 9.49 — worse at
*location*), while being genuinely better at *shape* (real variance, skew, key-number mass —
the entire reason a simulator exists over a Gaussian approximation). Directly measured before
the fix: the mean absolute gap between the simulator's own implied margin and the displayed
point-estimate margin was **1.58 points** across a full live slate — meaning the displayed
spread and the derived moneyline implied two different games. The fix shifts every simulated
margin/total sample by a constant (`actual_sample − raw_simulated_mean + point_estimate`) before
computing win probability or moneyline — inheriting the market-based estimate's superior
location while keeping the simulator's own variance and shape intact. The same fix unblocked
shipping the simulated **total** distribution, which had been held back purely because of the
same location problem, not a shape problem.

**Overtime** is simplified to sudden-death (first score of any kind ends the game), capped at a
guard limit against a pathological scoreless loop — a documented approximation, not a claim of
exact modern-rule fidelity, affecting a small minority of games.

**Moneyline**: standard American-odds conversion from the recentered simulated win probability.

---

## 11. Player props Monte Carlo simulator

The originally-proposed specification called for a fully parametric generative chain (plays ~
NegBinomial, targets ~ Dirichlet-Multinomial, yards ~ Gamma/lognormal, TDs ~ Poisson-binomial).
**Before building any of it, the core premise was tested directly**: is a player's real
week-to-week touch count uniformly overdispersed relative to pure multinomial sampling — which
is exactly what a single Dirichlet-Multinomial concentration parameter models?

**It is not.** Standardized residuals, `(actual_touches − predicted_touches) / multinomial_sd`,
show mean **0.38**, std **1.71** (a well-calibrated multinomial model would show ~0/1), a 95th
percentile of 2.96 and a 99th of 6.0 — a real, heavy right tail, present within every usage
tier (fringe/role-player/primary), not an artifact of mixing tiers together. The real
data-generating process looks like a genuine mixture — mostly-tight weeks plus a persistent
minority of large-surprise weeks (almost certainly real injury-driven role changes, blowout
game scripts, coaching decisions no share engine can anticipate) — not a single-parameter
overdispersion the proposed spec could represent.

Rather than layer more unverified parametric assumptions on top (a Beta-Binomial mixture, a
contaminated normal), the simulator **bootstrap-resamples the real empirical distribution of
these standardized residuals**, stratified by usage tier, reproducing whatever the true shape
actually is — skew, heavy tails, everything — without assuming a family. This is the same
successful pattern reused from §10's game simulator.

---

## 12. Rookie priors (see also §9.3)

`fit_draft_capital_prior`: a Bayesian-shrunk mean share per `(position, round)` bucket,

```
bucket_estimate = (n · bucket_mean + prior_weight_games · position_mean) / (n + prior_weight_games)
```

`prior_weight_games = 8.0`. Fit on every real rookie season 2016–2025, linked directly by real
draft-pick data. Falls through to Clay's analyst-projection PDF when it's available (treated as
more precise for a specific landing spot/scheme fit when it exists), with the draft-capital
prior as the always-on fallback underneath — this closes a real gap the analyst-PDF-only
approach had for WR/TE specifically (that source only ever covered running backs).

---

## 13. Validation methodology

- **Walk-forward, no-lookahead, everywhere**: every engine's `.predict()` executes strictly
  before its own `.update()` for a given row; every walk-forward run sorts input chronologically
  first.
- **Two deliberate season windows**: a fixed historical `TRAIN` (2018–2021) / `TEST`
  (2022–2025) split for honest backtesting comparable across the whole project's history, and a
  separate, forward-looking `CALIBRATION_SEASONS` window (the four most recent complete seasons)
  for anything meant to reflect current, live conditions rather than a fixed historical
  snapshot.
- **Rolling-origin cross-validation**: refit on everything strictly before a given season,
  score that season, walk the origin forward one season at a time — reveals whether an effect
  (a coefficient's sign, an MAE improvement) is stable fold-to-fold or fragile, which a single
  point estimate cannot show. This is precisely what caught an earlier market-blend
  coefficient's own documented instability (swinging from −0.06 to +0.17 depending on the exact
  fit window) directly from its fold-to-fold spread.
- **Permutation testing on the full search procedure, not just the winning result** (§7.4): the
  most rigorous tool in active use, developed specifically because a simpler permutation test
  (shuffle the outcome, refit-and-score once) was found to price in only "fit and scored on the
  same data," not the wider specification search that produced the thing being tested.
- **The statistical-power check** (§5): before treating any ATS/O-U-style test as a decisive
  kill criterion for a low-weight component, compute that component's best-case win rate at its
  actual weight; if the observed confidence interval contains that best-case value, the test had
  no power to be decisive and the real decision must rest on bias, CRPS, or fold-consistency.
- **Multiple-comparisons awareness**: an explicit running count of distinct TEST-window
  experiments is maintained in the living documentation (currently in the 85–95+ range) — at a
  conventional significance threshold, a handful of false positives are statistically expected
  purely by chance across that many tests, which is exactly why items with a thin statistical
  margin (§7's cornerback coefficient chief among them) are flagged and held provisionally
  rather than presented with unqualified confidence.
- **A self-generated CLV (closing-line value) log**: every pipeline run now appends (never
  overwrites) a snapshot of the line, this model's prediction, and the relevant flags to a
  running log, with automatic post-game reconciliation against the real final closing line and
  score once a game completes. This is the only evaluation instrument with enough statistical
  efficiency to ever resolve §7's provisional CB-edge question on a realistic timescale — at
  ~46 flagged games/season, resolving a small ATS edge from outcomes alone would need on the
  order of 2,400 bets, while CLV's much lower per-observation variance gives a usable read in a
  single season.

---

## 14. Full ledger — kept, rejected, and investigated-not-built (condensed)

**Kept, live in production:**
Layer 1 opponent-adjusted EPA ratings (§4); market-line-based margin/total, no blend (§5);
QB-swap adjustment (§6); the symmetric cornerback injury adjustment, shrunk and held
provisionally (§7); empirical-Bayes usage/TD-rate priors (§8.3); wind adjustment for QB
yards/attempt and WR/TE yards/target (§8.4); market-implied-team-total game-script conditioning
(§9.1); auto-refit-linear TD-probability calibration (§9.2); position-specific injury
reallocation (§9.3); draft-capital rookie prior (§12); the Monte Carlo game simulator's
recentered margin/total/moneyline (§10); the self-generated CLV log (§13).

**Tested and rejected, with a specific documented reason each — not simply "not gotten to
yet":** a continuous re-optimization of Layer 1's hyperparameters; four separate EPA
signal-quality improvements (early-down-only, winsorized, turnover-neutralized, success-rate
co-input); a declining-gain rating-update alternative; an nfelo-style dynamic per-team-error
blend; a kicker-quality total-points input; air-yards/WOPR/aDOT as a receiving-yards
correction (the *computations* are kept as reusable building blocks; the point-estimate
correction is not); isotonic and Beta calibration for TD probability; a parametric
Dirichlet-Multinomial props specification; a shared-game-environment-percentile correlation
mechanism for the simulator's total (superseded by the half-boundary fix); volume-based
opponent effects on props (as opposed to rate-based, tested three separate times previously and
also rejected).

**Investigated, not built — a real limitation, not a rejected hypothesis:** real player-prop
market odds data (no source exists in the current data pipeline; a specific, small, decisive
experiment — testing the validated RB-reallocation result against real posted lines — is fully
built and tested against mock data, gated on an actual paid API subscription decision);
historical opening-line data for a longer CLV baseline (the self-generated current-week version
needs no such gate and is already live); a true per-kickoff-window refresh (measured directly
and found not to address the real gap, which is mostly healthy-scratch decisions with no injury
report entry to refresh in the first place, not stale timing).

---

## 15. Known limitations & honest open questions

1. **The cornerback-injury edge (§7) is the model's entire specifically-claimed game-side
   betting edge, and it is held provisionally, not proven**, at roughly the 95th percentile of
   the most rigorous available null. This is the single most important caveat in this document.
2. **A real, current, temporary data gap**: 2026 injury-report and snap-count data are not yet
   published upstream (HTTP 404 as of this writing) — the §7 mechanism is presently inert except
   for one manually-verified override. Expected to resolve as the season approaches; worth
   reconfirming once real data starts flowing.
3. **Two purchase decisions remain unmade**, each fully scoped and, in the props-odds case,
   fully engineered against mock data already: a historical opening-line archive, and a month
   of real current-week player-prop odds to test the RB-reallocation result (this project's
   single strongest validated finding) against a real posted line rather than only its own
   prior estimate.
4. **Recurring bug classes this project has specifically found, and now structurally guards
   against**: (a) a linear-fit argument-order mistake that silently swapped slope and intercept,
   caught three separate times before a shared, sign-asserting helper eliminated the class
   project-wide; (b) a downstream calibration constant silently invalidated by an upstream
   shrinkage change, now addressed by a standing "check every hardcoded calibration's upstream
   dependency" audit practice; (c) a real-world name collision (two different real players
   sharing an exact full name) that caused a rookie fallback mechanism to silently resolve to
   the wrong person, fixed by preferring a team-and-position-scoped name match over a bare
   global one; (d) resubstitution — evaluating a fitted coefficient on data that was part of
   its own fit — caught and corrected twice in this project's history (§5's total-blend
   power-check, §7's cornerback coefficient), now an explicit item on this project's own
   statistical checklist before trusting any in-sample-adjacent evaluation.
5. **The QB-swap adjustment's own coefficient sits just under conventional statistical
   significance** (t≈1.52), kept on mechanism plausibility (a stale-team-rating-catches-up
   effect is well-established) rather than a clean p<0.05 — a documented judgment call, not a
   settled result.
6. **This document, and the model it describes, will continue to change.** The living record
   (`MODEL_DOCUMENTATION.md`) is updated after every phase of work closes; this document should
   be regenerated from it rather than hand-patched if it goes stale.

---

## Appendix: file map

| Section | Primary source file(s) |
|---|---|
| §4 Layer 1 | `src/models/ratings.py`, `src/models/tune.py` |
| §5 Margin/total, market line | `src/models/market_blend.py` (kept as a utility, unused in production), `src/pipeline/weekly_update.py` |
| §6 QB-swap | `src/models/qb_adjustment.py` |
| §7 Injury adjustment | `src/models/injury_adjustment.py`, `src/models/validate_adjustment_layer.py` |
| §8 Usage engines | `src/models/player_usage.py`, `src/models/qb_passing_stats.py` |
| §9.1 Game script / pass rate | `src/models/game_environment.py` |
| §9.2 TD probability / calibration | `src/pipeline/weekly_update.py`, `src/models/validate_props_pipeline.py` |
| §9.3 Injury reallocation | `src/models/injury_reallocation.py`, `src/models/rookie_prior.py` |
| §10 Game simulator | `src/models/game_simulator.py`, `src/models/drive_transitions.py`, `src/models/validate_game_simulator.py` |
| §11 Props simulator | `src/models/props_simulator.py`, `src/models/validate_props_simulator.py` |
| §13 Validation utilities | `src/utils/rolling_origin_cv.py`, `src/utils/stats.py`, `src/models/scoring.py` |
| Wind adjustment | `src/models/weather_adjustment.py` |
| Live pipeline orchestration | `src/pipeline/weekly_update.py`, `src/pipeline/build_predictions_page.py` |
| Data ingestion | `src/ingest/fetch.py`, `src/ingest/name_matching.py`, `src/ingest/build_weekly_stats_from_pbp.py`, `src/ingest/parse_clay_pdf.py` |
| Prop-odds experiment (built, not yet run against real data) | `src/ingest/fetch_prop_odds.py`, `src/models/prop_odds_experiment.py` |
| Full chronological record | `MODEL_DOCUMENTATION.md` |
