# MLB Prediction Model — Independent Review Packet

**What this document is.** A self-contained technical brief on a from-scratch MLB game/prop
prediction model, written for an independent reviewer with no prior exposure to this codebase
and no access to it — everything you need to form your own opinion is below. This is a
distillation of a much larger internal engineering log (`MODEL_DOCUMENTATION.md`, ~4,300
lines), reorganized around one goal: **let you find real, net-new opportunities to improve
prediction accuracy, without wasting your first ideas on things already tried.**

**How to use it.** Section 8 ("Ledger") is the single most important section for you,
specifically — it lists every signal/mechanism ever investigated, whether it was kept,
reverted, or rejected before building, and *why*, with real numbers. If your first instinct is
one of these, that's fine — reread the entry and either (a) find a genuinely new argument the
original test didn't consider, or (b) move on. Sections 1–7 are the architecture, in enough
depth to reason about correctness and methodology. Section 9 is a fully worked numeric example
of the entire pipeline for one real player in one real game — use it to sanity-check your own
understanding of the mechanics before critiquing them. Section 10 is what's already been
independently checked against published sabermetric literature (a prior external review pass),
so you know where the field's own ceiling has already been benchmarked.

This document deliberately does **not** include the team's own opinions about what to build
next. That's intentional — the point of bringing in an outside reviewer is to get an
independently-derived answer, not a rubber stamp on an internal roadmap.

---

## 0. One-paragraph summary

For a single plate appearance (PA), the model estimates a batter's and a pitcher's true-talent
rate for each of ~16 mutually-exclusive outcomes (strikeout, walk, single, double, triple, home
run, sac fly, etc.) via Marcel-style shrinkage of multi-season Statcast data, applies a
batter-only "contact quality" correction from Statcast expected-stats (xBACON, barrel rate, bat
speed, sprint speed, pulled-air rate), combines batter/pitcher/league rates via an odds-ratio
("log5"-style) formula, multiplies in independently-validated contextual factors (base-out
state, platoon, park, weather, times-through-the-order, catcher framing + umpire + defense,
latent per-appearance "stuff" shock), renormalizes to a valid probability distribution, and
samples an outcome. A bootstrap-resampled transition table converts that outcome into runs
scored and the next base-out state. A full game is simulated as a sequence of half-innings
until 27 outs using real/projected lineups, real/predictive bullpen usage, and real/forecast
weather. Monte Carlo aggregation over many full-game trials (200–1000+ depending on context)
produces the final score distribution, win probability, and every player prop.

**Current best trustworthy accuracy figure**: a genuine, zero-influence-on-any-modeling-decision
holdout of 1,474 real 2026 games gives **straight-up (SU) win/loss accuracy 58.3%, Brier score
0.2407** — see §7 for the full protocol discussion and important caveats about trial-count
sensitivity before treating any single number here as more precise than "roughly this, ±1–2
points."

---

## 1. Architecture overview

```
raw MLB Stats API + Statcast
        │
        ▼
per-plate-appearance table (one row per completed PA, 2023–2026, ~675k rows)
        │
        ▼
walk-forward true-talent rate engine (Marcel-style shrinkage + age curve)
        │
        ▼
batter-only contact-quality correction (Statcast expected-stats)
        │
        ▼
odds-ratio ("log5") matchup combine: batter × pitcher × league, independently per outcome
        │
        ▼
contextual multipliers, applied per-outcome-category (all independently validated):
  base-out state · platoon · park · weather · times-through-order ·
  catcher framing + umpire + defense · latent per-pitcher-appearance "stuff" shock
        │
        ▼
renormalize → single-PA outcome distribution (sums to 1 across the ~16 categories)
        │
        ▼
sample outcome → bootstrap transition table → runs scored + next base-out state
        │
        ▼
repeat per PA until 3 outs × 2 halves × 9+ innings → one simulated game
        │
        ▼
Monte Carlo aggregation (200–1000+ trials) → score distribution / win prob / player props
```

**The validation discipline** (this matters for how to read every number in this document):
every contextual factor and every batter-quality signal was validated **leakage-free** before
being wired in — built from data available strictly *before* the target season/game, tested
against the *real* target-season outcome, and required to add predictive value *beyond what the
current production signal already captures* (an incremental test, not a standalone
correlation). Then, for anything touching the simulator's outcome distribution, a **full-stack
isolated A/B test** (identical random seed, ~600–7,200 real historical games, one temporarily
disabled copy of the file vs. the real one) is the final arbiter — SU accuracy, Brier score, and
mean-absolute-error on total/margin runs, evaluated together. A statistically real,
component-level-validated signal does **not** automatically survive this bar; §8 documents as
many rejections as acceptances.

---

## 2. Core statistical engine

### 2.1 Marcel-style true-talent rate estimation

The foundational rate estimator for every batter/pitcher/outcome-category pair. Produces a
walk-forward-safe ("pregame") rate — at any point in a season, reflects only data available
strictly before that PA.

1. **Recency-weighted 3-season prior**: combines a player's most recent 3 seasons with weights
   `[5, 4, 3]` (most recent weighted highest — the standard Marcel scheme).
2. **Regression to league average**: the weighted 3-season rate is blended with league average
   using a per-`(player role, outcome)` stabilization constant sourced from Russell Carleton's
   published stabilization-point research (batters generally need fewer PA to stabilize than
   pitchers for the same outcome).
3. **In-season Bayesian blend**: once the target season is underway, the preseason prior is
   further blended with the player's own current-season-to-date counts:
   `(observed + PRIOR_WEIGHT * prior_rate) / (observed_n + PRIOR_WEIGHT)`.
4. **Age adjustment**: a Tango-style age curve (peak age 29, asymmetric slopes below/above peak)
   nudges the rate based on the player's age that season; the sign of the generic curve is
   flipped per outcome category where domain logic requires it (contact-outcome rates vs.
   power-outcome rates age in different directions).

Rates are **park-neutralized before** this stage (see §3.2) so the per-game park multiplier
applied later doesn't double-count the same effect.

### 2.2 Odds-ratio ("log5"-generalization) matchup combine

```
odds(p) = p / (1 - p)              [p clipped away from exactly 0/1 first — see §6]

matchup_odds = odds(league_rate) × (odds(batter_rate) / odds(league_rate))
                                  × (odds(pitcher_rate) / odds(league_rate))

matchup_prob = matchup_odds / (1 + matchup_odds)
```

Interpretation: start from league-average odds, multiply in the batter's own odds-ratio
*relative to league average*, then the pitcher's own odds-ratio *relative to league average*.
This is the standard sabermetric "log5" combination, generalized from a binary (win/lose)
formulation to an arbitrary rate, and applied **independently to each of the ~16 outcome
categories** — not just to one headline stat. (A prior external literature review — §10 —
confirmed this matches Baseball Prospectus's own published PECOTA methodology.)

### 2.3 Base-out transition table

Converts a sampled PA outcome (e.g. "single" with runners on 1st/2nd, 1 out) into runs scored
and the next base-out state via **nonparametric bootstrap resampling of real historical
transitions** — not a hardcoded rule table, so real-world variance (runner-advancement
decisions, error scoring) is captured directly. Three-tier fallback for sparse states: exact
base-out-state match → pooled by out-count only → pooled by outcome only, league-wide.

---

## 3. Batter-side contact-quality correction

Applied to a batter's own `rates` dict at profile-build time — a fixed, opponent-independent
adjustment at the same tier as the Marcel-shrunk rate itself, *not* another contextual
multiplier in the matchup combine. **Batter-side only** — every pitcher-side analog investigated
(CSW%, fastball velocity/spin, contact suppression) failed to survive the full-stack bar or
showed no real signal (see §8; this matches DIPS theory, which predicts pitchers have little
control over contact quality allowed even though they do control some batted-ball mix).

Built as four layers, all **live in production**:

1. **xBACON-based contact quality**: real BACON (hits-per-ball-in-play) correlates only
   0.25–0.30 season-to-season; Statcast's `estimated_ba_using_speedangle` ("xBACON") correlates
   0.39 with *next-season* real BACON. Applied to all hit-type categories (single/double/
   triple/home_run) via an odds-ratio multiplier: `odds(predicted_hit_share) /
   odds(existing_hit_share)`, both clipped to `[0.10, 0.60]` before the ratio.
2. **Barrel-rate → HR-share multiplier**: `home_run` specifically, further adjusted via a fitted
   regression (`HR-share-among-hits ≈ 0.0442 + 0.3616 × existing_share + 0.53 × barrel_rate`,
   both existing and predicted shares clipped to `[0.02, 0.35]`) — barrel rate's own
   coefficient exceeds real-outcome-history's in a leakage-free incremental-R² test (0.40→0.43
   and 0.41→0.44 across two season-pairs).
3. **Sprint-speed → groundball-single multiplier**: `single` only, scaled by the batter's own
   groundball rate (a fly-ball hitter gets almost no adjustment) — real incremental R² gain
   0.088→0.141 and 0.024→0.043 across two season-pairs.
4. **Bat speed extension** (Statcast bat-tracking data, 2023+): extends both the xBACON layer
   (`R² 0.139→0.183`, ~32% relative gain) and, separately, a **pulled-air-ball-rate extension**
   to the HR-share layer (a *direction* signal, distinct from barrel rate's power/quality
   signal — 17.8% real HR rate on pulled-air balls-in-play vs. 3.96% on everything else, and
   67.5% of all real home runs are pulled-air balls). Both are the two most recent confirmed
   full-stack wins (§8).

**Every one of these multipliers is clipped both sides before its odds-ratio** — see §6, a
recurring and important discipline in this codebase after multiple real production blowups.

**One pitcher-side signal did survive** the full-stack bar: a GB/FB-rate-based HR-allowed-share
multiplier (mirrors the batter-side barrel-rate mechanism, but for pitchers) — the first
pitcher-side win after 6 prior pitcher-side attempts failed. See §8's "refined finding" for why
this one succeeded where the others didn't.

---

## 4. Contextual multipliers (applied per-PA, per-outcome-category)

Every factor below is multiplicative, computed independently ahead of time from real data, and
validated leakage-free before being wired in. All factors for a given PA are applied in
sequence to each outcome's odds-ratio-combined base probability, then the full 16-category
vector is renormalized to sum to 1. Where a factor is irrelevant to a category (e.g. catcher
framing has no effect on home runs, since a real home run can't be affected by fielding), its
multiplier for that category is fixed at neutral (1.0) — it is simply absent from that factor's
dict, not present-and-equal-to-1.

- **Platoon splits**: same-hand/opposite-hand batter-vs-pitcher effect, computed separately for
  the batter's own split AND the pitcher's own split-allowed, both applied.
- **Park factors**: per-outcome-category (not one overall "runs" factor), mean-normalized to
  1.0 across the league each season.
- **Weather**: temperature/wind-bucketed, further split by the batter's own handedness AND
  their individual career pull-tendency tercile (wind blowing out to a batter's own pull field
  matters more than a generic "wind out" factor). For a future game with no posted conditions,
  a climatological distribution is resolved and **sampled fresh per Monte Carlo trial**
  (propagating real day-to-day weather uncertainty), not collapsed to a single expected value.
- **Times-through-the-order penalty (TTOP)**: a pitcher facing the same batter for the Nth time
  in a game becomes progressively less effective (well-established sabermetric effect), capped
  at 3 (4th+ time treated identically — sample sizes get too sparse past that to trust further
  differentiation).
- **Catcher framing**: catcher's real framing skill as a multiplier on `strikeout`/`walk` only.
- **Umpire tendency**: home-plate umpire's own real called-strike-zone tendency, same mechanism
  as catcher framing, merged multiplicatively into the same factor dict (one umpire calls the
  whole game).
- **Defense (Outs Above Average)**: team's real per-game defensive alignment (infield/outfield
  split), applied to `single`/`double`/`triple` (batted balls convertible to outs by good
  defense) — merged into the same dict as catcher/umpire (disjoint outcome keys, safe to merge).
- **Home-field advantage**: a league-wide pooled residual factor, applied only to the home
  team's own at-bats (not symmetric) — specifically the edge that survives after park factors'
  own per-team normalization cancels out most of the park effect itself.
- **Latent per-pitcher-appearance "stuff" shock**: once per real pitcher appearance per Monte
  Carlo trial, draws `g ~ N(0, σ²)` and multiplies the odds of on-base-allowed outcomes (walk,
  HBP, single, double, triple, home_run — *not* intentional walk, a strategic decision rather
  than a "stuff" effect) by `exp(g)/exp(σ²/2)` (a mean-1 lognormal multiplier), then
  renormalizes. This exists because real per-start outcome variance is measurably wider than
  the model without it produces (§7's dispersion discussion) — **σ = 0.40 is the current frozen
  production value**, arrived at via full-stack sweep, not a claim about how much real
  pitcher "stuff" varies day-to-day (the simulator's own game-flow logic attenuates the raw
  effect, so this is a simulator-internal calibration constant, not a physical one).

Stolen-base/baserunning (real per-runner attempt/success rate, keyed by lineup slot) and
blowout/position-player-pitching substitution (8+ run deficit, 8th inning or later) round out
the game-flow mechanics but aren't per-outcome-category multipliers in the same sense.

---

## 5. Bullpen and lineup handling

- **Predictive bullpen model** (the live/deployable path, as opposed to a backtest's use of the
  *real* historical sequence): each starter's own walk-forward-projected expected innings, then
  a **specific reliever sampled per remaining inning** — resampled fresh every trial, since
  which reliever actually pitches is itself part of the uncertainty a Monte Carlo simulation
  should propagate — weighted by recent usage frequency and discounted by rest-day
  availability, with closer identification for save situations.
- **Starter-hook frailty**: a per-start correlated-hazard mechanism modeling *when* a starter is
  actually pulled (as opposed to a fixed expected-innings cutoff) — confirmed as a real,
  large improvement to the distribution of starter outs/innings specifically (the quantity
  props actually price), even though it showed no separate game-level win-probability effect.
  Deployed in the live props path only, not the oracle backtest (a deliberate, documented
  scope decision, not an inconsistency).
- **Lineup projection** (for a future game without a posted real lineup yet): baseline = each
  team's most recent complete 9-batter lineup; a platoon-aware variant swaps a slot only when
  there's a *strong, asymmetric* signal that another player specifically starts against
  today's pitcher-hand (an earlier one-sided version over-swapped; the current two-sided
  weak-share/strong-share test fixed that).

---

## 6. Recurring engineering disciplines (context for judging "is this a bug")

- **Clip both sides before any odds-ratio, always.** Every module computing `odds(p) = p/(1-p)`
  clips `p` away from exactly 0/1 first. This is not defensive boilerplate — real documented
  blowups occurred without it: platoon splits produced a **146,611×** multiplier, weather **73×**,
  park factors `inf`, one HR-share variant **13.8×**, another rejected-before-shipping variant
  **5.5×**. Any new multiplier in this codebase is expected to follow this pattern.
- **Bayesian shrinkage everywhere**: the same `(observed + PRIOR_WEIGHT × reference) /
  (observed_n + PRIOR_WEIGHT)` pattern, from the core Marcel prior down to platoon splits to
  every expected-stats layer. Prior weights are always fit or sourced from published research,
  never guessed.
- **Leakage-free, incremental-value validation is the standard for every new signal**: build
  from strictly-prior data, predict the real target-season/game outcome, and require the new
  signal to add value *beyond the current production signal* — not just beyond raw history.
- **Full-stack isolated A/B is the final arbiter, always.** A real, statistically valid
  component-level signal has failed this bar more often than it's passed (§8).
- **Revert discipline**: when a factor fails the full-stack bar, all wiring is cleanly removed
  from every consumer, but the underlying investigation code is deliberately left in place,
  clearly marked as a documented-but-unused artifact — never silently deleted. This is why §8
  can cite exact numbers for rejected ideas.

---

## 7. Validation protocol and current accuracy — read the caveats, not just the headline number

**Primary backtest**: real historical games (2023–2025, ~7,200 games at the current canonical
protocol), each player's real walk-forward pregame snapshot, real lineups/starters, real park
factors, real recorded weather. Two bullpen modes exist: an *oracle* mode (replays the game's
real historical bullpen sequence — the cleanest "does the core simulator work" test) and a
*predictive* mode (the same bullpen-sampling logic actually used live). Metrics: total/margin
score MAE, and **straight-up (SU) win/loss accuracy**, weighted most heavily as the practical
full-stack arbiter throughout this project's development.

**A 2026 holdout** (1,474 real games, 2026-03-25 through 2026-07-19) was run **exactly once**
against the frozen production stack, with an explicit rule against iterating against it —
2023–2025 remain the only seasons ever used for any fitting/selection/keep-revert decision. This
is the first genuinely out-of-sample check in this project's history. Result: **SU 58.3%,
Brier 0.2407, total MAE 3.561, margin MAE 3.521** — statistically indistinguishable from the
in-sample figure (57.9% SU at the time), which is read correctly as "no detectable
degradation out-of-sample," not "the model is better in 2026" (the ±0.4pp gap is noise-level at
this n). **Published pre-game MLB classifiers in the broader literature cluster mid-50s to
low-60s%** across decades of published work, and historical betting-market favorites themselves
win only 55.9–57% of games — this model's real, walk-forward, market-data-free accuracy is at
or above that documented band (see §10 for the full literature-comparison context).

**Important protocol caveat, specifically about trial count (K).** The canonical backtest
protocol runs K=50–200 trials per game. Straight-up accuracy is the *sign* of a K-trial-estimated
mean margin — its sampling noise scales with per-trial variance / √K. This means:

- Any change that specifically **increases per-trial variance** (e.g. the latent pitcher shock,
  §4) will mechanically look worse on a low-K point-metric even when the underlying model
  genuinely improved, because a wider per-trial distribution flips more close-call games at low
  K. A CRN-paired K-scaling check (same random draws, increasing K) is the only way to
  distinguish "this change costs real accuracy" from "this change costs estimator precision at
  low K." For the shock specifically: the raw canonical K=50 read showed a −1.1pp SU cost, but a
  K-scaling check (K=30/100/300) found the gap shrinking monotonically toward ~−0.7pp at K=300,
  the textbook signature of an estimator artifact rather than a real cost — and **production
  itself runs K=1,000+ trials**, roughly 3–20× further along that same shrinking curve.
- A dispersion diagnostic (PIT/z-score coverage across real games) independently confirmed the
  model was measurably **under-dispersed** before the shock (std(z) ≈ 1.08–1.14 across multiple
  independent measurements, should be 1.0) and the shock fixed most, though not quite all, of
  that gap (std(z) → 1.015 in the canonical full run). Two earlier attempted fixes — resampling
  each rate from its own Bayesian posterior per trial, and deterministically "widening" every
  rate away from league average — were both tested and **did not help** (the second one, tested
  across a `w ∈ [1.0, 2.0]` sweep, made the diagnostic *worse* at every value tried, for a
  specific, understood mechanical reason: stretching a rate toward an extreme *lowers* its
  binomial variance `p(1-p)`, the opposite of what dispersion needed). **If you're about to
  propose a dispersion fix, read this paragraph twice — two plausible-sounding approaches
  already failed for non-obvious reasons.**

**Bottom line for reading any number in this document**: treat SU/Brier as "roughly this, ±1–2
points," always prefer the highest-K or holdout-season number available over a raw low-K
canonical read, and assume any number describing a variance-changing mechanism specifically
needs the K-scaling caveat above before being taken at face value.

---

## 8. Ledger: what's live, what was tried and reverted, what was investigated and never built

**Read this section closely before proposing anything.** This project's practice is to keep
failed and rejected ideas visible rather than erase them — most of the entries below represent
real engineering time and a real statistical test, not a hunch.

### 8.1 Currently live in production

| Factor | Status |
|---|---|
| Marcel true-talent rates, odds-ratio matchup combine, bootstrap transition table | core engine |
| Platoon splits (batter + pitcher sides) | live |
| Park factors (per-outcome, mean-normalized) | live |
| Weather (temp/wind × handedness × pull-tercile) | live |
| Times-through-order penalty | live |
| Catcher framing + umpire tendency + defense (OAA) | live |
| Baserunning/stolen-base rates, blowout substitution, predictive bullpen | live |
| xBACON contact-quality, barrel-rate HR-share, sprint-speed groundball-singles | live |
| Bat-speed extension to contact quality | live, though a later re-test at a larger sample (n=7,237) found the SU delta's confidence interval includes zero — kept as "plausible, not statistically re-confirmed at the larger n," not a clean-cut win |
| Pulled-air-ball-rate extension to HR-share | live, same caveat as bat speed above — original n=597 test showed a clean win, the larger re-test's CI includes zero |
| Home-field advantage (league-wide pooled) | live (a correctness addition, not a hypothesis test) |
| Park-neutralized true-talent rates | live (correctness addition) |
| Latent per-pitcher-appearance "stuff" shock (σ=0.40) | live — see §7 for the full dispersion story |
| Pitcher GB/FB-rate → HR-allowed-share multiplier | live — the only pitcher-side signal to survive a full-stack test, in 7 attempts |
| Zombie-runner / auto-runner extra-innings rule (real MLB rule since 2023) | live (a correctness fix, kept despite a small net-negative point-metric delta because the underlying rule is certain, not a hypothesis) |

### 8.2 Built, then reverted (wiring fully removed; investigation code kept as a documented, unused artifact)

| Idea | Full-stack result | Likely reason |
|---|---|---|
| Whiff-rate-on-swings strikeout multiplier | Total MAE improved; margin MAE and **SU accuracy worsened (−1.3pp)** | Strikeout-specific batter signal |
| Full pitch-by-pitch outcome-composition mechanism (replacing K/BB/HBP resolution with a pitch-level simulation) | **SU −3.7pp** | The composed distribution was measurably variance-compressed vs. real spread — correct on average, worse at separating strong/weak matchups |
| Pitcher fastball velocity+spin → strikeout multiplier | Net negative on SU despite a real, positive component-level R² gain | Pitcher-side/strikeout-specific signal (matches DIPS theory) |
| Runner-speed-conditioned base-out transitions | Real, large component-level effect, but **failed full-stack even after a more rigorous re-test** (Brier score regression, CI excludes zero) | Conditioning resampling on runner identity adds within-game heterogeneity that hurts calibration |
| Batter-side walk-rate blend multiplier | Total MAE improved; margin MAE worse; **SU −1.4pp** | Same failure shape as whiff-rate, despite being a walk (not strikeout) signal |
| Jet-lag / circadian fatigue team-level multiplier | Original test: SU −1.2pp. **Retested with a more rigorous protocol: the delta reversed to statistically indistinguishable from zero** | The original "regression" was itself noise, not a real negative effect — still not deployed (no proven benefit either), but for a different reason than first believed |

### 8.3 Investigated and never built (failed a component-level or safety check before reaching a full-stack test)

| Idea | Why not built |
|---|---|
| Pitcher-arsenal-tercile matchup adjustment | Negative in leakage-free component testing across three outcome categories |
| HR-share-with-bat-speed extension (a different variant than the one shipped) | Real incremental R² gain, but produced up to a 5.5× real-data blowup with no safe clip design found |
| Pitcher-side contact-quality-allowed signals (xBACON-allowed, barrel-rate-allowed) | No real incremental signal (R² ~0.005–0.07) — matches DIPS theory: pitchers show almost no control over contact quality allowed |
| Bat-tracking for double/triple-share prediction | No real incremental signal (R² ~0.01–0.03) |
| Double/triple split via xSLG | Same — doubles/triples resist prediction from available Statcast inputs generally (see §8.4's ranked table) |
| EV90 and squared-up rate as contact-quality stabilizers | Both essentially zero incremental value once xBACON + bat speed are already known — redundant signals |
| Extra-bases-taken-rate-conditioned baserunning transitions | Real, split-half-validated component effect, but structurally the same intervention type as the rejected speed-conditioned transitions above, which already failed decisively — not re-tested on strong analogical grounds |
| Park-factor × batter-pull-tercile interaction | Clean, decisive null (incremental R² = 0.00000, interaction p=0.97) — a batter's own pull tendency does not measurably amplify or dampen their home park's own HR factor |
| Defensive shift/positioning as a new signal | The strongest cited real-world effect (a 2023 shift-ban study) measures banning the OLD illegal shift — a mechanism that has never existed anywhere in this project's own 2023+ data window, by construction. The *remaining legal* "infield shade" positioning was directly measured on 665,659 real PAs: negligible and inconsistent in direction |
| Ballpark humidor/ball-conditioning as a distinct modeled attribute | Real and physics-grounded in general, but this project's park factors are purely empirical (measured from real outcome rates) — any real humidor effect for a park with sufficient history is already baked into that park's own factor whether or not it's explicitly labeled |

### 8.4 Where does real headroom still live? (an evidence-based ranking, not a hunch)

A standing, re-runnable audit tool ranks every batter/pitcher × outcome-category combination by
`(1 − R²) × real per-PA run-value leverage`, where R² is measured strictly between each
player's own preseason Marcel estimate and their real full-season outcome (i.e., this
specifically measures the *base-rate estimator's* own explanatory power — it does **not**
credit any of the downstream contextual multipliers in §3–4, so a high score does not mean
"nothing has been done here"). Top of the list, cross-referenced against what's already been
tried:

- **Pitcher `field_out` rate** (highest score): this is DIPS — a pitcher's own year-to-year
  control over batted-ball-out rate specifically is a well-established, largely irreducible
  sabermetric near-null. High score reflects real variance, not a neglected opportunity.
- **`double`, both sides** (lowest R² on the entire table, ~0.01–0.02): doubles resist
  prediction from every Statcast-derived input tried so far — a genuinely hard, already-
  investigated category, not a neglected one.
- **`single`, `strikeout`, both sides**: already have deployed contextual corrections
  (contact-quality/xBACON generally, the age-adjusted Carleton-K stabilization specifically for
  strikeouts) — a high base-rate score here is expected and largely already addressed
  downstream.
- **`home_run`, both sides — flagged as an open lead by this ranking, then closed one day
  later; corrected here after an external review caught the packet citing the stale
  intermediate state.** This ranking originally reconfirmed real R² was on the table for a
  previously-rejected bat-speed extension to HR-share (rejected for a 5.5× real-data-limit-test
  blowup, an implementation problem, not a signal problem) — and a follow-up investigation acted
  on exactly that lead: it found the true root cause was a pre-existing, already-shipped
  defect in the *base* 2-term HR-share formula's clip floor (0.02, producing up to **5.02×** on
  real zero-HR batters even without any bat-speed term involved), fixed the floor (0.02→0.035,
  capping the base formula at 2.88× and the bat-speed-extended version at 2.86×), refit the
  coefficients, and ran the **full-stack A/B properly this time (n=8,711 real games)**: SU delta
  +0.0068pp (CI includes zero), Brier delta −0.0004 (CI includes zero) — a **decisive, honest
  null**, not a blowup-driven rejection. The R² gain survived being made safe and still didn't
  move a full-stack metric — strong evidence this is a real-but-small effect size problem, not a
  clip-design problem. **What's actually still live**: only the clip-floor retune itself (a
  standalone robustness fix, independently justified, carrying no accuracy claim) — the
  bat-speed/pulled-air 4-term HR-share extension is not deployed. A materially different
  input-shrinkage clip design (weighting by reliability before the odds-ratio, rather than a hard
  floor) is more principled in isolation, but is very unlikely to reverse this verdict — the
  floor only touches the bottom ~5% of batters, and the null was attributed to effect size, not
  to which ~5% get touched. **There is currently no open, evidence-backed "just fix the clip"
  lead on HR-share** — a genuinely different mechanism (not another clip variant) would be
  needed to reopen this category.

### 8.5 The cross-cutting pattern this project has found in its own results (useful for judging your own proposals in advance)

Every **kept** factor is a refinement to an *existing* per-category odds-ratio multiplier the
simulator already applies (contact quality, barrel rate, sprint speed, bat speed, pulled-air
rate, and the one pitcher-side win all extend the *same* two mechanisms — `contact_quality_
multiplier` / `hr_share_multiplier` — with better-fit coefficients or an added input). Every
**failed** factor introduces a genuinely new mechanism or a new axis of within-game
heterogeneity into the joint outcome distribution — six separate examples, six failures (the
pitch-level composition swap, whiff-rate, pitcher-stuff, the walk-blend, speed-conditioned
transitions, jet-lag). The one pitcher-side exception (GB/FB → HR-allowed-share) fits the rule
exactly: it's structurally a refinement to the *existing* home_run category, not a new
mechanism, and DIPS theory independently predicts pitchers *do* control batted-ball type even
though they don't control contact-quality outcome — distinguishing it from the failed
strikeout/stuff attempts.

**Practical implication for your review**: a proposal that reads as "refine an existing
per-category multiplier with a better-fitted coefficient or a genuinely new, cheap input" has
historically had good odds here. A proposal that reads as "add a new mechanism/heterogeneity
axis to the simulator" has a real, demonstrated track record of failing here even when its
own component-level statistics look clean — that's not a reason to withhold such an idea, but
it is a reason to propose an unusually rigorous test design for it (the K-scaling caveat in §7
is exactly this kind of lesson-already-learned).

### 8.6 Known bugs found and fixed (for calibrating how much to trust "this looks fine at a glance")

A full internal correctness audit (independent per-subsystem review, personally verified against
real data before any fix) found a dozen real, verified defects, including:

- A live, user-facing prop (`away_covers_plus_1_5`) computed an entirely wrong condition —
  effectively "home doesn't lose by 2+ runs" instead of the actual spread-cover condition —
  silently wrong since the file's first commit.
- A cold-start data-leakage bug in four independent modules (catcher framing, weather, umpire
  tendency, times-through-order), all sharing one copy-pasted fallback pattern that let the
  dataset's true first season see its own future-in-season data.
- The same reliability-formula units bug (weighted-count used where an unweighted count was
  required, inflating apparent statistical confidence for any player without a full 3-season
  track record) found independently copy-pasted into 5 separate modules, having only been fixed
  in its original location years earlier.
- One confirmed, kept, live full-stack-win multiplier (the pitcher GB/FB HR-allowed-share
  factor) had never received the real-data limit test every sibling multiplier explicitly
  requires, and its clip floor was measurably too loose (real thin-sample pitchers could hit a
  5.2× multiplier before the fix; capped at 2.82× after).

**Also flagged, real but lower-priority, not yet fixed** (included for completeness — none of
these affect the live production path): a dead/reverted jet-lag module groups travel by team
only (not team+season), which could misattribute a season-opener's jet-lag status to the
*previous* season's road trip; a gated-off, never-enabled pitcher double-play multiplier has the
widest real-data tail in its file (6.33×) plus a documented coefficient-sign instability — a
real landmine if ever flipped on without first hardening its clip; three CRN (deterministic
random-draw pairing) decision-tags are defined but never wired up, which affects the
*precision* of certain paired A/B comparisons involving bullpen sampling, not the correctness of
the base model.

This list exists so you can calibrate: real, non-trivial bugs have been found in this codebase
before, more than once, by exactly the kind of independent review you're being asked to do now.
Don't assume something is correct just because it's already shipped.

---

## 9. Worked example: one real player, one real game, every intermediate number

This section exists so you can verify your own understanding of the mechanics against a
concrete case before critiquing them — every number below was produced by actually running the
live production code (not reconstructed from memory) against real cached Statcast data for a
real completed game.

**Game**: Pittsburgh Pirates @ Cincinnati Reds, August 1, 2026, Great American Ball Park. Real
posted weather: Overcast, 76°F, wind 10 mph in from center field. **Batter**: Brandon Lowe
(traded from Tampa Bay to Pittsburgh at the deadline days earlier — the batter-side true-talent
rate estimator is team-agnostic, keyed only by player ID across all of a player's own real PAs
regardless of which team he played for, so a mid-season trade requires no special handling on
the batter side). Bats left-handed. **Opposing starter**: Andrew Abbott, throws left-handed.

**Step 1 — raw Marcel rate** (age-adjusted, park-neutralized, recency-weighted shrinkage),
before any contact-quality overlay:

| Outcome | Rate | Effective sample size |
|---|---|---|
| home_run | 5.313% | 625 |
| single | 12.381% | 745 |
| double | 4.311% | 1,869 |
| triple | 0.283% | 1,869 |
| walk | 8.882% | 575 |
| strikeout | 27.654% | 515 |

**Step 2 — after the contact-quality + HR-share + bat-speed overlay** (real 2026 Statcast
inputs: xBACON 0.354, barrel rate 12.1%, pulled-air rate 26.4%, bat speed 71.7 mph):

| Outcome | Adjusted rate | Change vs. raw |
|---|---|---|
| home_run | 3.954% | **−25.6%** |
| single | 11.552% | −6.7% |
| double / triple | scaled by the same contact multiplier | −8.3% |
| walk, strikeout | unchanged | 0% (contact quality doesn't touch these) |

The large HR cut reflects this batter's actual current-season barrel/exit-quality profile
pulling below what his multi-season outcome-based shrinkage rate alone would imply — the
overlay doing exactly the job it's designed for, not an error.

**Step 3 — platoon, BOTH legs** (§4 documents this, easy to read past on a first pass: the
batter's own same/opposite-hand split AND the pitcher's own same/opposite-hand-*allowed* split
are two separately-estimated real effects, and **both are applied multiplicatively to the same
category** — this is not double-counting the same effect twice, it's two different players'
own independently-measured tendencies): Lowe's own platoon multiplier vs. a lefty pitcher (his
side) is HR ×0.8541; Abbott's own platoon-allowed multiplier vs. a lefty batter (his side,
separately estimated from his own career same/opposite-hand-allowed splits) is HR ×0.7206. Both
numbers are real and both apply to this one PA.

**Step 4 — the opposing pitcher's own rate** (Andrew Abbott): HR 3.221% raw → 3.521% adjusted
(GB/FB-mix adjustment), single 14.396%, walk 9.830%, strikeout 18.916%, effective sample size
1,825 for HR (a large, well-established sample).

**Step 5 — league average** (2026 season to date): HR 3.089%, single 14.204%, walk 8.115%,
strikeout 22.510%.

**Step 6 — odds-ratio matchup combine, HR specifically**:
`league=3.089%, batter(adj)=3.954%, pitcher(adj)=3.521% → combined (pre-context) HR probability
= 4.502%`.

**Step 7 — every context multiplier applied to this exact PA, in the order the code applies
them** (Lowe leading off the 2nd inning, bases empty, 0 outs, first time facing Abbott this
game, Pittsburgh is the visiting team so no home-field-advantage factor applies to this at-bat).
This table is the corrected, complete version — an earlier draft of this packet omitted the
pitcher's own platoon row here (it was mentioned only in Step 3 above, disconnected from this
table), which is exactly the kind of gap an external reviewer should — and did — catch; see the
verified reconciliation immediately below the table.

| Factor | Value | Direction check |
|---|---|---|
| Base-out state (bases empty, 0 outs) | ×1.0815 | correct — empirically HR-favorable state |
| Batter's own platoon split (Lowe vs. LHP) | ×0.8541 | correct — standard same-hand disadvantage |
| Pitcher's own platoon-allowed split (Abbott vs. LHB) | ×0.7206 | correct — same-hand advantage for the pitcher, a *separate* real effect from the row above |
| Park (Great American Ball Park, HR) | ×1.1348 | correct — GABP is one of MLB's most homer-friendly parks |
| Weather (wind in from CF) | ×0.9604 | correct — wind blowing in suppresses fly balls |
| Times-through-order (1st time) | ×0.9417 | correct — pitchers are typically sharpest their first time through |
| Catcher + umpire + defense (HR) | no key present | correct — a real home run can't be affected by fielding, so it's absent, not neutral-and-present |

**Verified reconciliation** (re-ran the instrumented code rather than trusting hand arithmetic,
after an external reviewer correctly flagged that the numbers as originally drafted didn't
close): `4.502% × 1.0815 × 0.8541 × 0.7206 × 1.1348 × 0.9604 × 0.9417 = 3.0757%` unnormalized —
confirmed to match the model's own internal unnormalized value bit-for-bit. The exact
unnormalized sum across all 16 categories for this PA is **1.011394** (not close to the ~1.4 a
plausible-looking manual estimate might suggest — renormalization here is a small correction,
not a large one, because park/weather/state/TTO factors are each independently mean-normalized
to ~1.0 across the league by construction, so most PAs' own unnormalized vectors land close to
1 already). `3.0757% / 1.011394 = 3.0411%` — matches Step 8's stated home_run figure exactly.

**Step 8 — full renormalized 16-category outcome distribution for this exact PA**: strikeout
26.8%, field_out 46.8%, single 12.5%, walk 6.9%, double 2.9%, **home_run 3.0%**, everything else
small — sums to exactly 100.00% (verified: 1.011394 total, each category divided by that sum).

**Step 9 — the actual displayed prop** (1,000-trial full-game Monte Carlo simulation, averaging
~4.78 plate appearances per game across trials, against Abbott and then whichever Reds
relievers/closer enter later): mean home runs 0.162, mean hits 0.906, mean walks 0.411, mean
RBI 0.607, mean strikeouts 1.258, mean total bases 1.562, P(1+ HR) 14.9%, P(1+ hit) 60.0%.
Sanity check: 0.162 HR over 4.78 PA ≈ 3.4%/PA average, consistent with the 3.0% single-PA figure
above (the game average is a little higher since it blends in more favorable bullpen matchups
later in the game). **What actually happened in this real game**: 1 hit, 1 walk, 1 strikeout, no
home run across 5 real plate appearances — broadly consistent with the pregame simulated
average (a single game is not itself calibration evidence either way, just a plausibility
check).

---

## 10. Prior independent benchmarking (already done — for context, not to be re-litigated from scratch)

A previous research pass (adversarially verified, ~25 sources, ~25 claims independently
confirmed) checked this architecture against published sabermetric and machine-learning
literature and found:

- The core odds-ratio/"log5" combine matches Baseball Prospectus's own published PECOTA
  methodology — not an outdated or naive choice.
- A head-to-head test found regularized logistic regression matched or beat gradient-boosted
  trees/random forests on Brier score for this class of problem; several published
  "89–95% accuracy" ML papers were checked and found to leak the game's own outcome-linked
  stats as inputs (not genuine pre-game forecasting). No verified source showed a neural-network
  or GBM approach beating this model class on genuine pre-game MLB prediction.
- Published pre-game MLB classifiers cluster mid-50s to low-60s% SU across decades of
  literature; one well-cited XGBoost study topped out at 55.5% (its own authors concluding it
  couldn't beat bookmakers); historical betting favorites themselves win only 55.9–57% of games;
  a 155,563-game study found no significant exploitable inefficiency anywhere in posted odds.
  This model's real, walk-forward, **zero-market-data** 58.3% SU sits at or above that
  documented band.
- Two literature-suggested candidate signals (defensive shift/positioning, ballpark humidor
  effects) were investigated directly against this project's own data and came back null or
  already-subsumed — see §8.3.
- **The stacked-context-multiplication question — raised as open here, then actually tested;
  answer is real but nuanced, not a clean yes/no.** A follow-up fit a per-category attenuation
  exponent λ (unnormalized probability ∝ p0 × M^λ, where M is the product of state/platoon-
  both-sides/park/times-through-order factors for that PA; catcher/umpire/defense/weather/HFA
  were excluded from M for tractability, a real scope limit, not a silent one) via a logistic
  regression with an offset, fit on one season (2024, chosen specifically to avoid a separately-
  discovered cold-start leak in the base-out-state factor's own first-season data — see the new
  finding below), evaluated held-out on 2025. Result: **home_run's λ = 0.598, 95% CI (0.451,
  0.746) — clearly excludes 1. Strikeout's λ = 0.869, CI (0.774, 0.964) — also excludes 1.**
  Straight multiplication genuinely IS measurably overconfident in aggregate for these two
  categories — the open question has a real answer, and it's "not calibrated." **But applying
  the fix is not an obvious win**: held-out full-vector-renormalized multiclass log-loss improves
  only trivially in aggregate (1.605455 → 1.605145), and — the important part — restricted
  specifically to the PAs that were REAL home runs, the λ-correction makes log-loss slightly
  *worse* (3.4088 → 3.4164), same direction for strikeout's own true-positive rows. Only
  `single` (whose λ=0.845 CI included 1, i.e. not even significant) showed a real improvement on
  its own true-positive rows. **Interpretation**: the aggregate overconfidence is real and is
  driven by the vast majority of true-negative PAs; a single global λ per category doesn't
  clearly help on the rare true-positive predictions that a prop bettor / calibration check
  actually cares about most, and may hurt them slightly. **A more promising, still-untested
  refinement of this same idea**: check whether λ<1 is a UNIFORM effect across the whole range
  of M, or driven specifically by the most extreme-M tail (a "cap the boost, don't dampen
  everything" design, closer to this project's existing clip-based discipline than a blanket
  exponent) — this packet's own fit can't distinguish those two shapes and would need to.

- **New finding surfaced while building the above, not in the original ledger**:
  `game_simulator.py`'s `build_state_factors_by_season` still has the exact cold-start
  look-ahead leak (`ref = prior if len(prior) else X[X["season"]==season]`) that a prior
  correctness audit found and fixed in four sibling modules (catcher framing, weather, umpire,
  times-through-order) — this 5th instance was apparently missed. Confirmed by direct code read;
  not yet quantified on real data or fixed. A real, small, verifiable defect, offered here as an
  example of exactly the kind of finding this packet's Section 8.6 warns you not to assume is
  already closed just because it's shipped.

**What this means for you**: don't spend your first pass re-deriving "is log5 the right
combination method" or "should this be gradient-boosted trees instead" — both were already
checked against real literature. The stacked-multiplication question above is now a checked,
nuanced finding, not an open invitation — a fresh proposal here should engage with why the naive
global-λ version didn't clearly help on true positives, not repeat the same fit. Spend your
first pass instead on (a) anything in §9's worked example that looks mathematically or
statistically off to you, (b) the extreme-tail-only version of the λ idea above, or (c)
something none of this project's own reviews have thought to check yet.

---

## 11. What would make your review most useful

- **Cite a specific mechanism, section, or number above** — not a general architectural
  critique that doesn't engage with what's actually implemented.
- **If your idea resembles anything in §8, say so explicitly and explain what's different this
  time** — a new argument, a new dataset, a materially different implementation — rather than
  re-proposing the same test that already failed.
- **Prefer falsifiable, scoped proposals** ("test whether X, measured via Y, on real data, would
  predict Z beyond what's already captured by W") over general direction ("consider incorporating
  more advanced defensive metrics").
- **It is a legitimate, valuable conclusion to say a given area is already well-modeled or near a
  real ceiling** — §10 found real evidence the model overall may already be close to a
  literature-documented limit for this problem class without market data as an input. A
  manufactured "everything can always be improved somehow" answer is less useful than an honest
  "I don't see a real gap here" for any specific area you check.
