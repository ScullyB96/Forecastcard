# Full Adversarial Model Audit — 2026-08-03

**What this is.** A fresh, comprehensive audit of the entire MLB model — every assumption,
fitted number, data source, and statistical/probabilistic decision — run as 8 parallel
adversarial subsystem audits (ingest, rate estimation, park/weather/HFA, situational factors,
simulator core, bullpen/lineup, props/pipeline, and a dedicated cross-factor double-counting
hunt). **Every finding below was independently verified against the actual code and/or real
data by the supervising session before inclusion** — quoted lines were re-read, empirical
magnitudes were independently reproduced where feasible (several exactly), and agent claims
that didn't survive verification were downgraded or corrected in place. Checked-clean results
are summarized at the end so coverage is auditable.

**Headline.** The model is *not yet* world-class-correct. The audit found **4 critical
issues, ~22 verified major defects, and ~20 minors**, spanning real data loss, walk-forward
violations, double-counting (two new instances of the platoon-bug class), mechanical
simulator errors producing MLB-impossible outcomes, and production/validation configuration
divergence. Crucially, several of the largest biases **mutually compensate** — see "The
compensating-bias problem" below — which is why aggregate calibration looks good while
individual mechanisms are measurably wrong, and why the fix plan must be sequenced.

---

## CRITICAL

### C1. The measured platoon shared term leaks the target season's own outcomes
`measure_platoon_shared_term.py` + `platoon_splits.py:_measured_shared_term_for`.
Season S's shared-term cell is measured from season S's own realized outcomes (the baseline
prediction is walk-forward; the *numerator* is not), and `platoon_splits` looks up the same
season's cell when building that season's multipliers. The configuration that passed the
2025 holdout read used **2024-only** cells applied to 2025; the productionized version
silently changed to same-season keys — the shipped mechanism is not the validated one, the
module docstring's "using only PRIOR seasons' data" claim is false, and every backtest from
2024 onward (including the same-day guardrail's "fixed" arm on its 2024 games) is
contaminated on the platoon term. Live 2026 use is in-season-to-date information (legal in
the live sense) but the design violates the project's walk-forward convention.
**Fix**: look up the latest measured season strictly **before** the target season
(2025→2024 cells, 2026→2025 cells, 2024→exponent fallback) — exactly the validated shape.

### C2. Past dates in the schedule cache are never re-fetched after games finish
`fetch.py:251-253` — incremental fetch starts at `max(cached date) + 1`, and
`fetch_schedule_day(tomorrow)` (every evening full run) pushes that max to tomorrow, so
yesterday/today are never revisited after the last 4pm-ET light run. **Verified: 76
regular-season rows before 2026-08-03 permanently frozen at Pre-Game/In Progress etc., 40
with NULL weather and no lineup rows.** Real posted lineups and weather for those games are
lost, silently degrading lineup-projection history, weather climatology, and every
schedule-status-gated backfill (see C3/minors).

### C3. The Statcast pipeline can permanently lose night-game PAs on mixed slates
`fetch.py:87-88` (incremental assumes max cached game_date is complete) + `fetch.py:138`
(the backfill safety net only rescues games whose schedule row is "Final" — which, per C2,
evening games' rows never become). The 9pm-ET full-run cron (shipped 2026-08-03) fetches
Statcast mid-slate; if Savant returns completed-afternoon-game rows for a date whose night
games are still playing, `max(game_date)` advances past the date and the night games are
permanently absent with no rescue path. Verified live that Savant returns 0 rows for
in-progress games (protects all-night slates); the day/night mixed slate is the real
exposure. Compounding: **pybaseball's 7-day per-date-URL cache** (M22) re-serves a partial
day's response to every retry/backfill for a week.

### C4. The per-pitcher expected-innings model is completely inert in production
`props.py:550-551` — `ctx["expected_innings"].get((pitcher_id, game_pk), 5.4)` is keyed by
`(pitcher, game_pk)` built **from historical PA rows**; a live/future game_pk never exists,
so every starter, every game, gets exactly 5.4 expected innings. **Verified on the real
2026-08-03 output: all 16 starters = 5.4.** Backtests use historical game_pks where the
lookup works — so the validated configuration (per-pitcher innings) is not the deployed one
(constant). Everything keyed off the cutoff — bullpen arrival schedule, hook anchor, closer
timing (M16), starter volume props — runs on a flat wrong assumption. Two audit agents
converged on this independently.

---

## MAJOR — statistical estimation

**M1. Three surviving instances of the weighted-vs-raw reliability bug** (task #53's class)
in `expected_stats.py` — xBACON (~:180), barrel (~:287), pulled-air (~:442) priors all
compute reliability from the 5/4/3-weighted BIP sum against a raw-units constant; the GB/FB
function in the same file was already fixed with a comment citing the earlier audit.
Independently reproduced: 898 batters, mean reliability 0.657 vs correct 0.435 (~4× inflated
denominator) — short-history batters' noisy contact-quality/barrel/pulled-air inputs trusted
~2.5× too much, feeding live HR/hit props.

**M2. Prior-weight monotonicity inversion** — `true_talent.py:366` (`min(raw_prior_pa, K)`)
vs `:460` (`fillna(K)`): a player with 1–59 raw prior PA gets a *weaker* prior than a player
with zero history. More information should never produce a less sticky prior. 67 real 2025
batters affected.

**M3. `effective_n` posterior-sampling incoherence** — `true_talent.py:477`: the Beta
posterior width uses the K-capped blend denominator, so a veteran with 1,471 prior PA
samples his strikeout talent with n=60 (±4.6pp/trial, ~5× too wide), and fast-stabilizing
categories (best-known talents) get the *most* parameter noise — inverted vs. the
stabilization literature the constants come from. Caveat: this width was deliberately tuned
in to fix under-dispersion — a fix must re-validate dispersion (see fix plan Phase D).

**M4. TTOP factors double-count pitcher-talent composition and survivor selection**
(`ttop.py:66-79`) — three agents converged independently; magnitudes reproduced. The pooled
TTO baseline is 56.4% reliever PAs at TT1, so every reliever gets ×1.055 K on top of rates
that are already reliever-high; decomposition on the 2023-24 fit window shows the TTO2 HR
bump (+7.5%) is ~100% composition and the TTO3 walk reduction (−5.8%) is *entirely*
selection (causal within-pitcher: +0.8%). The same shared-component signature as the platoon
bug. Also interacts with the hook model, which re-models the same day-level selection (F6).

**M5. Stolen-base volume is double-counted** — `build_pa_table.py:119-122` defines
`post_state` as the *next PA's* start state, so every mid-PA steal/CS/pickoff/WP is already
embedded in the resampled transitions; `game_simulator.py:572-600` then fires an explicit
pre-PA SB attempt layer calibrated to the full real attempt rate. Embedded movement measured
at 11–14% per (runner-on-1st, strikeout) cell + explicit 6.3%×80% on top ≈ ~2× real steal
volume, plus doubled CS outs. Code-verified at both sites.

**M6. The pitcher "stuff" shock is not mean-preserving after renormalization** —
`game_simulator.py:271-275`: the mean-1 lognormal multiplier is applied in probability space
then renormalized; the renorm map is concave in the multiplier (Jensen), so shocked
(on-base) categories are systematically suppressed. Independently reproduced: −3.34% at
σ=0.40 ≈ −0.23 runs/team/game hidden mean shift. The docstring's "not a hidden bias shift"
claim is mathematically false post-renorm, and the frozen σ=0.40 was selected by an A/B
whose arms differed in mean, not just variance.

**M7. Opponent-quality contamination is real, measurable, and was never actually evaluated**
— no opponent-quality term exists anywhere in rate fitting; measured std of opponent quality
faced is 7–15% of the std of players' own rates (±1pp K-rate at schedule extremes before
Marcel pooling dilutes it). The presumed "investigated and declined" (task #60) note could
not be found — the closest artifact evaluated a different question. Modest but undocumented.

## MAJOR — data/environmental factors

**M8. ATH/OAK team-code mismatch breaks the Athletics' park factors** — PA table codes the
2023-24 A's `ATH`, schedules code `OAK` (verified by set-diff). Two breaks: the
venue-continuity merge yields NaN (the relocation fix never operated on the team it was
built for), and the backtest's `("OAK", 2024)` park lookup misses → **every 2023-24 Coliseum
backtest game ran with no park adjustment** at one of the league's strongest pitcher parks
(~160 games).

**M9. A zero-history venue gets factor 1/group_mean, not neutral 1.0** —
`park_factors.py:268-274`: a new park's raw factor is exactly 1.0 by construction, then
divided by the group mean (1.0326 in 2025) — Steinbrenner Field (short-porch Yankee clone)
was simulated all 2025 as a ~3% HR-*suppressing* park on zero observations, and the
placeholders contaminate the normalizing mean for the other 28 teams.

**M10. Weather factors are absolute, not park-relative** — each park's average climate is
counted in both the park factor and the weather factor. Two agents measured independently
and consistently: per-venue mean applied HR weather factor spans 0.979 (Petco/SF) to 1.020
(TB/Steinbrenner), a systematic ~±2% HR-scale bias at climate-extreme venues.

**M11. Defense composite imputes missing-OAA fielders at their covered teammates' average
instead of league-average 0** — `defense_factor.py:76-78` NaN-skipping mean; only 75.3% of
2025 infield slots have prior OAA, 27% of team-games have ≤2 of 4 infielders covered — the
thinnest-information lineups get the most magnitude-inflated factors.

**M12. RotoWire suffix normalization drops entire team lineups** —
`fetch_rotowire_lineups.py:113-121`: `_normalize_name`'s docstring claims Jr./II handling;
the code never strips suffixes (verified). One unmatched name ("Luis Garcia" vs "luis garcia
jr", live today for NYY) rejects the whole 9-man RotoWire lineup down to the lower-fidelity
projection tier.

## MAJOR — simulator mechanics

**M13. Non-HR walkoffs overcount runs** — `game_simulator.py:677-681`: the game-ending play
credits ALL sampled runs; MLB counts only through the winning run unless it's a HR. ~0.70%
of all simulated games end with an MLB-impossible margin — distorts run-line/margin and
totals at the 1-vs-2-run boundary.

**M14. Outs can decrease** — the state-factor shrinkage floor makes mechanically-impossible
outcomes (2-out sac bunt/fly) drawable (~2.6e-3/PA), and the transition fallback chain's
4th tier (`_by_outcome_only`, below the outs-keyed tier — verified present) serves them
post-states from 0/1-out contexts: outs go backwards ~1 per 26 games, un-ending innings and
inflating scoring. Tier-2/3 also materialize runners that never existed for a few cells.

**M15. Mid-inning hook handoff PAs are attributed to the starter** — `props.py:693`
attributes by whole inning; the hook fires per-PA, so the incoming reliever's PAs in the hook
inning credit the starter. Committed 150-game calibration parquet shows sim starter mean_k
5.28 vs real 4.79 (+0.49 K systematic) — a headline exported market.

**M16. `_shift_bullpen_after_hook` scrambles the closer's inning-9 role** — the pure timing
shift moves the closer to inning 7 on an early hook (then the fallback makes him pitch 3
innings) or to inning 11 on a late one (a middle reliever takes the 9th). The measured
"closer takes the 9th 56%" calibration only survives when the stochastic hook lands exactly
on the cutoff — which C4 pins at 5 for everyone.

**M17. A starter can be drawn as his own reliever** — the 45-day relief pool never excludes
the day's probable starter; 13.0% of 2025 starts (633/4,860) had the starter qualifying for
his own team's pool (illegal re-entry, double-credited props).

**M18. Projected lineups can contain the same batter twice** —
`lineup_projection.py:138-142`: the baseline-retention branch never checks `used_players`.
2.56% of real projected 2025 lineups (10/390 sampled) contain a duplicate — an 8-man lineup
with one batter double-drafted, on exactly the tier-3 fallback games decided earliest.

## MAJOR — pipeline/output

**M19. Prop calibration coefficients are stale post-platoon-fix** — `BATTER_PROP_CALIBRATION`
fit 2026-07-23/25; the Aug 3-4 platoon fixes changed the simulated distributions those
shrinkage slopes were fit to correct; no refit occurred (git-verified). The old slopes now
over-shrink the less-overconfident raw probabilities. (Pre-registered as this audit's top
suspect; confirmed.)

**M20. Upsert-only export leaves phantom player rows** — `export_to_site_db.py` has zero
DELETE statements (verified); when a lineup or probable starter changes between the 4
intraday exports, replaced players' prop rows persist in the site DB alongside the new ones —
the site shows props for players who aren't playing.

**M21. Slate-day context never contains the previous night's games** — the only full rebuild
runs at ~9pm ET on D-1 (mid-slate), so reliever rest-day availability, closer logs, and all
snapshots reflect D-2; the three same-day light runs never repair it. A closer who threw 25
pitches last night is sampled as fully rested — directly contradicting the module's own
"rest-day availability for this exact date" claim.

**M22. Reliever prop rows are conditional-on-appearance means presented as unconditional**
— `props.py:851` groupby only includes trials where the pitcher appeared; 271 of 287
exported pitcher rows on 2026-08-03 are relievers/placeholders whose "Strikeouts 1.6" is
E[K | appeared] with no appearance probability anywhere in the export. (Plus M22b: the
pybaseball 7-day URL cache staleness described under C3.)

---

## MINOR (verified; grouped)

- **Data**: SB/umpire backfills skip non-Final + "Completed Early" games that ARE in the PA
  table (78 games with PAs but no SB/ump rows → SB rates biased down); ~0.24% of PAs dropped
  with the *preceding* PA's post_state absorbing the gap (~370 contaminated transitions in
  2025); 2-strike foul bunt misclassified as no-op (15/yr, real strikeouts);
  `sac_bunt_double_play` unmapped (0 occurrences to date); ambiguous Chadwick lookups
  resolved by "most recent activity" with arbitrary tie-break (real live tie: two Max
  Muncys); trade-override fetch window anchors at today not target_date (1-day skew), and
  the "175 pitchers" log line counts all traded persons (~⅓ are pitchers).
- **Estimation**: debut-cohort prior pools debutants' entire careers (survivorship mutes the
  below-average correction; live only on the rare last-resort path); age-sign evidence
  standard applied inconsistently (noise → 0.0 for intent_walk/triple but noise → keep +1.0
  for pitcher walk/HR — documented decision, internally inconsistent); `widen_rate` clip
  asymmetry (dormant, w=1.0); pitch-model stabilization constants still labeled "STARTING
  VALUES," never updated from the promised sweep (mostly reverted paths); PULL_TERCILE
  cutoffs fit on 2023-25 leak boundary info into 2023-24 backtests (one-tercile boundary
  effects only).
- **Environment**: PARK_FACTOR_PRIOR_PA=5000 is ~35-45% of a full window — common-outcome
  park effects heavily attenuated (Coors HR 1.127 vs industry ~1.3+); one global prior sized
  for triple_play tames HR too; climatological weather split across venue renames
  (Daikin/Minute Maid, Rate/Guaranteed Rate — halves the roof-frequency sample); pooled HFA
  includes bottom-9 truncation/walk-off selection (untested); stale
  `outcome_park_factors.parquet` artifact predates multiple fixes (not read by production;
  poisonous to any future consumer).
- **Simulator**: compound rare-category blowup survives renormalization silently (state 30
  triple_play 24× × park ~6× × HFA 1.4 clip *binding* = 202× → ~0.9% triple plays in that
  cell); hook hazard stays active while the blowout position-player pitches (tallies
  polluted, tier re-draw inversion); CRN tier-selection key reused across two draws in the
  same (inning, side) (paired-A/B precision only); 18-inning ties leave probability mass
  unassigned and are scored as home losses / away covers (observed mass ≈ 0 today; latent);
  one reliever pitches all extras on the props path (workload unreal, no re-entry violation);
  BULLPEN_FALLBACK innings attributed to a placeholder while the sim actually used the
  starter (never observed in real output); switch hitters get a blended LR/RL platoon cell
  vs LHP instead of the true RL (~0.5% on ~9% of batters); fatigue uses only the most recent
  appearance date (back-to-back streaks not modeled).
- **Output**: totals threshold hardcoded at 8.5 regardless of market line; displayed
  precision (0.1pp) exceeds Monte Carlo precision at n=1000 (SE ≈ 1.1-1.6pp) by ~10-30×;
  `validate_prop_calibration` collects post-calibration probabilities with no raw bypass — a
  refit-circularity trap for exactly the refit M19 requires.

---

## Protocol context (not bugs, but load-bearing for interpreting every quoted number)

The canonical backtest is **oracle-conditioned three ways**: real posted lineups (live:
projections/RotoWire), each game's actual per-inning pitcher usage (live: predictive
bullpen), and posted-actual weather (live: forecast for evening games). Documented and
deliberate; the weather share was measured ≈0 (CRN-paired ablation) and the bullpen share
was measured (task #143), but every canonical SU/Brier figure is an upper bound on live
performance. Additionally, C1/C4 mean the validated configuration ≠ deployed configuration
in two separate places — a class of failure the packet should track explicitly going forward.

## The compensating-bias problem (why the fix plan is sequenced)

Aggregate calibration currently looks good (sim total 8.82 vs real 8.86) **with** the shock
suppressing ~0.45 runs/game (M6) while the SB double-count (M5), walkoff overcount (M13),
outs-decrease (M14), and TTOP reliever-K inflation (M4, run-suppressing) push in offsetting
directions — and σ=0.40, the state prior, and the prop calibration were all tuned with these
present. **Fixing any single mechanism in isolation will likely worsen headline calibration.**
Per-fix acceptance must therefore use distribution-level and mechanism-level diagnostics
(bucketed calibration, PIT dispersion, component invariants), with SU/Brier judgment reserved
for the post-retune re-baseline.

## Sequenced fix plan

- **Phase A — data layer (independent, no retuning needed):** C2 schedule re-fetch of
  non-Final past dates; C3 Statcast completeness guard (don't advance past a date until its
  schedule says all its games are Final) + backfill Final-filter widened + pybaseball cache
  bypass for backfills; M12 suffix stripping; M8 ATH/OAK canonical team mapping; M20 export
  DELETE-then-upsert (or stale-row sweep); minors: backfill status filters, dropped-PA
  handling, foul-bunt classification, transactions window anchor.
- **Phase B — mechanical correctness (ship on invariants, not point metrics):** C1
  prior-season platoon lookup; C4 live per-pitcher innings projection (key by pitcher, not
  game_pk); M13 walkoff run truncation; M14 zero impossible outcomes in state factors +
  outs-preserving fallback guard; M5 SB de-double-count (strip embedded mid-PA advancement
  from transitions, or retire the explicit layer — decide by A/B); M15 per-PA pitcher
  stamping in events; M16 closer-anchored (not sequence-shifted) bullpen re-timing; M17
  starter exclusion from own pool; M18 lineup dedup; M11 league-average imputation; M22
  appearance-rate column for conditional means; M21 light-mode refresh of relief/closer logs.
- **Phase C — statistical re-estimation:** M1 three reliability-unit fixes; M2 monotone
  prior weight; M4 within-pitcher TTOP refit + reliever TT1 neutralization; M10
  park-relative weather rebuild; M9 zero-history venue neutrality; M3 effective_n redesign
  (jointly with the dispersion story); M7 opponent-quality evaluation (measure, then decide).
- **Phase D — re-tune and re-baseline on the fixed stack:** re-sweep σ (M6 says its frozen
  value embeds a mean shift); refit prop calibration (M19) from RAW probabilities (fixing the
  circularity trap first); re-run `measure_platoon_shared_term` under the new keying;
  re-fit park prior per-outcome; then one full re-baseline + 2026-H1 holdout re-run + new
  ledger entry as the single point where SU/Brier verdicts are read.

## Coverage: what was checked and found clean (abbreviated)

Marcel weighting math; park-neutralization exactness (both directions, per-PA venue keying
for traded players); in-season snapshot walk-forward (verified to the PA on real data);
odds-ratio combine + renormalization mechanics; age formula vs Tango's published values; age
data completeness (zero nulls); outcome taxonomy completeness on all real data; pre_state
encoding (0 mismatches in 676,748 rows); doubleheader/suspended-game dedup; spring-training
exclusion; MLB lineup structure; framing/umpire slope fitting and quasi-random assignment
(measured, no material double-count); state×TTO overlap (measured ≤1%); HFA×park exact
cancellation math; xBACON×barrel level/share factorization (exact); park×GB/FB; frailty
correlation structure (one z per start, Gauss-Hermite mean-preservation); hook walk-forward
and no-foresight; tier-draw margin signs; batting-order PA weighting (realistic 4.54→3.65);
spread-cover definitions (current code correct, partition exact); RBI attribution definition
consistency; export column mapping and fan-out arithmetic (2,299 rows reconcile exactly);
two-way player collision fix; CRN statistical quality; zombie-runner mechanics vs the real
MLB rule; transition-table outs/runs invariants on all tier-1 data.

**Origin note:** this audit was commissioned after an external review found a real platoon
double-counting bug that every prior internal audit had missed. Its methodology — parallel
adversarial subsystem passes with mandatory independent verification of every claim, plus a
dedicated cross-factor overlap hunt with measured (not argued) overlaps — is the direct
generalization of what that review did by hand, and it found two more instances of the same
bug class (M4, M5) plus the two validated-vs-deployed divergences (C1, C4) that no
metric-level check could see.

---

## Resolution log (updated as fixes land — see git history for full verification evidence)

Committed, each individually verified before the next (2026-08-03/04):

- **Phase A (data layer)**: C2 schedule re-fetch of non-terminal past dates; C3 Statcast
  completeness guard; M8 ATH/OAK canonicalization; M12 RotoWire suffix stripping; M20 export
  phantom-row DELETE sweeps; minors (SB/umpire backfill status filters, PA-table
  post_state-before-drops, foul-bunt-with-2-strikes, trade-lookback anchor). C1 platoon
  prior-season keying (+ MIN_CELL_EVENTS=50 floor + deviation clips — two collateral
  regressions caught and fixed during verification).
- **Phase B (mechanical)**: C4 live expected-innings 3-tier lookup; M13 walkoff run
  truncation; M14 impossible base-out/outcome cells zeroed + outs-decrease fallback filter;
  M15 per-PA pitcher stamping (replaces whole-inning attribution); M16 closer-anchored
  bullpen re-timing (LATE_INNING_ANCHOR=8); M17 starter excluded from own relief pool;
  M18 lineup dedup with widening fallback tiers; M11 defense composite league-average (0)
  imputation; M21 light-run refresh of relief/appearance/closer logs + expected_innings_live;
  M22 appearance_rate column on pitcher props.
- **Phase C (statistical)**: M1 raw-unit reliability for xBACON/barrel/pulled-air with
  re-derived constants (200/75/800 — pulled-air held-out MSE −62%); M2 monotone prior weight
  (constant K floor, all 7 sites); M9 zero-history venue → neutral 1.0; M10 park-relative
  weather (per-venue mean applied HR factor spread 0.0144 → 0.0035 std); M4 within-pitcher
  starter-only TTOP + reliever apply_ttop=False (TT2 HR 1.075→0.997, TT3 walk 0.942→1.007,
  matching this audit's decomposition).
- **M7 (opponent quality): evaluated, NOT built.** Additive Marcel-style schedule correction
  (opponent window rates shrunk with production constants), held-out against realized 2025
  rates (≥200 PA): batter side K +2.7% MSE improvement but walk/HR flat-to-worse; pitcher
  side 0 of 3 improved (K −1.3%). Pre-registered bar (≥2 of 3 categories >1% on a side)
  fails on both sides. Root cause of smallness: the per-PA odds-ratio matchup combine already
  conditions on the actual opponent, so schedule bias enters only through the Marcel input,
  where shrinkage dilutes it. Task #60's missing artifact is now replaced by this real one
  (scratchpad evaluate_m7.py, output in the M7 commit message).

- **M5 (SB layer): RETIRED.** CRN-paired A/B (n=697 common games, K=300, 2023-24; OFF arm =
  empty pregame SB table): SU 0.5997 vs 0.5882 at K=300 but 0.5552 vs 0.5667 at K=100 (sign
  flips inside noise), Brier gap +0.0009, MAE tied — the pre-registered keep-bar (SU loss
  >1.5pp at K=300 or Brier loss >0.002) not met, so volume correctness wins: the embedded
  transition movement is now the single source of steals. baserunning.py + SB fetch retained
  for a future de-embed-and-re-add-identity redesign.
- **Platoon shared term re-measured** on the fixed stack (its _meta requires this after any
  handedness-adjacent change — M4/M10 qualified). First pass exposed and fixed a measurement
  baseline mismatch with M4 (TTOP applied to reliever PAs shifted every K cell uniformly
  ~4%); corrected table keeps validated-category structure with <1% mean drift.

Open: M3 (effective_n redesign, joint with dispersion re-validation), M6 (renorm-aware
mean-preserving shock — implemented + verified locally, held for the paired SHOCK_SIGMA
re-sweep so they land atomically), M19 coefficient refit, full re-baseline + 2026-H1
holdout + ledger entry.
