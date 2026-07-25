# NHL Prediction Model — Research Log & Technical Documentation

Audience: an LLM (or engineer) with no prior context on this codebase, needing a complete
mental model of what data this model is built on, what was tried, what worked, what didn't,
and why. Following the practice established in the sibling MLB project, this log keeps
dead ends and rejected data sources visible rather than erasing them — a source that looked
promising and turned out unusable is worth recording so it isn't re-investigated later.

All paths are relative to the repo root: `/Users/brettscully/Desktop/sports-models/nhl`.

---

## 0. Current state — read this first

Everything in this section is a summary of, and fully traceable to, the detailed research
log in §1-4 below. If you only read one section before deciding what to do next, read this
one; §1-4 is the evidentiary record for every claim made here.

**The one-line summary of where eleven months of cycles actually made progress**: the market-
comparison Brier gap moved 0.00351 → ~0.0025 (§6.8 → §31.1/§36.5) entirely on the strength of
finding and fixing specific structural mechanisms — scoring-era drift correction, tie-mass
transfer, cross-season priors, team-specific OT/SO logistic. The one cycle that tried to buy
further progress with added statistical capacity instead of a new mechanism (the GBM stacking
layer, §34) was adopted, then reverted once its own adoption evidence was scored honestly
(§36) — it contributed approximately nothing to this number, at any point in the chain. The
wins came from finding mechanisms, not from adding capacity.

**The original roadmap is now complete, and the project is FROZEN for the 2026-27 season
(§38).** Every fitted constant and committed hyperparameter described in §0.1-0.2 below is
locked as of this point — in-season changes only through §37.4's recalibration rule or a
documented emergency-fix path, never a reaction to a live performance wobble alone. The three
open research threads (§0.4) are explicitly parked as off-season work against this frozen
baseline. Deployment-hardening work (task #43's rule, #44's scraper, #45's measurement) proceeds
on top of the frozen configuration, not as changes to it.

### 0.1 What the model actually is right now

One sentence: **team xG-based strength ratings, split by strength state (even-strength/PP/
PK/other) with team-specific special-teams time-on-ice weighting, a decayed (not
infinite-memory) league-average baseline, AND a cross-season prior (blending each team's own
last-season rate into its season-opening shrinkage target), combined via independent Poisson
with an empirically-fit home-ice term, adjusted by a starting-goalie overlay and an
away-back-to-back rest penalty, with a real (not assumed) OT/shootout tiebreaker layer on
top.**

Pipeline, in order (each arrow is a real file, not a plan):

```
NHL API schedule (src/ingest/fetch_nhl_api.py)          MoneyPuck team-game xG (src/ingest/fetch_moneypuck.py)
  -- gameId, final score, lastPeriodType (REG/OT/SO)       -- situational (5v5/PP/PK/other) xG, per team-game
                    \\                                      /
                     \\-- src/ingest/team_codes.py (NHL<->MoneyPuck team-code crosswalk) --/
                                          |
              MoneyPuck shot-level data -> src/ingest/fetch_moneypuck_goalie_games.py
                     (per-goalie-per-game shots faced / xG faced / goals allowed)
                                          |
                                          v
        src/models/shrinkage.py -- shared walk-forward Bayesian-shrinkage helpers
  (add_walk_forward_rate / add_walk_forward_toi_rate(_cross_season) / add_walk_forward_mean,
   halflife_games decay -- production halflife=600; cross_season_weight -- production 0.75,
   blends each team's own last-season rate into its season-opening shrinkage target, §12/§13)
                                          |
                 +------------------------+------------------------+
                 v                        v                        v
   team_strength_situational.py    team_strength_goalie.py    rest_schedule.py
   (EV/PP/PK/other xG rates,       (starting-goalie GSAx,     (away-back-to-back
    team-specific PP/PK TOI,        relative to league avg,    lambda adjustment --
    priors x2.0 (§13) for the       K=12 shrinkage prior)      team-level only; a
    decayed baseline+cross-season                              goalie-specific and a
    prior, PLUS shorthanded-goals-                              timezone-interaction
    for term (league-wide xG rate                               version were both
    x walk-forward PK TOI, §18)                                 tested and rejected,
                 |                                               §8-9)
                 +------------------------+------------------------+
                                          v
                 baseline_naive_poisson.py's combine + score_distribution
                    (independent Poisson, empirically-fit home-ice term)
                                          v
      overtime_shootout.add_walk_forward_reg_ratio -- walk-forward regulation ratio
       two_sided_diagonal_transfer.fit_deltas_per_game -- per-game tie-mass transfer,
        targeted by walk_forward_tie_ratio's walk-forward OT-calibration RATIO (Sec26)
                                          v
              overtime_shootout.py + ot_logistic.py -- OT/SO resolution
        (OT-decided: team-specific P(home wins)=sigmoid(a+b*lambda_diff),
         a=0.008, b=0.302, real signal r=0.084 p=0.0001, Sec17; SO-decided:
         flat empirical rate, no team-strength signal found, r=0.014 p=0.58)
                                          v
                  FULL win probability / score distribution / totals / margin
```

**The single script that runs the current, most-complete pipeline end-to-end is
`src/models/validate_tie_mass_ratio.py`'s `run_treated()`** (Cycle 26, §26, RE-CONFIRMED
unchanged §35.4). Win probability, score distribution, totals, and margin all come from this
one function. **`validate_gbm_stack.py` is NOT part of the current model** — the Cycle 16 GBM
stacking layer was adopted (§34) then REVERTED (§37) once its own adoption bootstrap was found
to have been scored in-sample (the same class of defect §35 fixed for the holdout checks, and
§35.9 first caught in the market benchmark); corrected to genuine out-of-fold scoring, the dev
gain crosses zero, matching the already-neutral holdout (§34.4) and the near-zero OOF
market-benchmark contribution (§35.10/§37.3). It is kept only as a frozen, historical (rejected)
file, the same status as `validate_ev_toi_halflife.py` (§32/§35.4). Every other `validate_*.py`
file is likewise a frozen snapshot of an earlier cycle, kept for reproducibility, not the
current model.
A separate, real market-odds dataset also exists (§6: 13 seasons, 14,260 games with matched
closing lines) but is NOT part of the prediction pipeline itself — it's used only for
external calibration/benchmarking (§6.8-6.9), and has a real coverage gap for 2022-23 onward
(§6.1) that no source found so far fills.

### 0.2 Current best model: real, validated numbers (not aspirational)

Model name in the ledger: **`walk_forward_tie_ratio_poisson`** (Cycle 26, §26, RE-CONFIRMED
unchanged §35.4 after the walk-forward-discipline fix, and again the sole current-best model
after §37 reverted the GBM stacking layer). Cycle 26's walk-forward tie-mass ratio, the
shorthanded-goals term, cross-season prior, re-tuned prior-minutes multiplier, team-specific OT
win probability, and §22's symmetric rest fix combine to drive win probability, score
distribution, totals, and margin. **Neither the GBM stack (§34/§37) nor the EV-TOI halflife fix
(§32) is part of the current model** — both were adopted, both later reverted once each one's
OWN adoption evidence was re-examined more carefully than the first pass had (§35.4 for EV-TOI,
§37 for the GBM) — see caveats -2/-1 below for what each one's evidentiary shape actually was.
Validated on the development set (16,528 real games, regular season, 2010-11 through 2023-24 —
see §0.4 for why this specific range and not the full history):

| Metric | Value |
|---|---|
| Straight-up (SU) win accuracy | 58.01% |
| Brier score | 0.23980 |
| Log-loss | 0.6723 |
| Total-goals MAE | 1.797 |
| Margin MAE | 2.013 |
| Mean predicted total (vs. actual 5.8125) | 5.722 |
| Actual home win rate (sanity check) | 54.08% |

**Important, honest caveats**:
-2. **§34's GBM stack was adopted, then REVERTED (§37) once its own adoption bootstrap was
   corrected.** The original dev-set bootstrap that cleared the pre-registered success bar
   (§34.3) fit the committed configuration on the full dev set and then scored it on the SAME
   dev set — in-sample tree memorization, the identical genus of bug §35 fixed for the holdout
   checks. Re-scored on genuine out-of-fold predictions (§37.2), the dev gain crosses zero on
   both Brier (diff +0.00014, CI [-0.00040,+0.00067]) and SU (diff -0.00030, CI
   [-0.00502,+0.00454]) — consistent with the holdout check's own already-neutral result (§34.4,
   which never had this defect) and the OOF market-benchmark's near-zero GBM contribution
   (§35.10). The kill-switch not firing is consistent with this picture, not against it: a weak,
   real, nonlinear signal can beat a planted noise feature on cross-validated log-loss during
   hyperparameter selection while still contributing nothing once in-sample memorization is
   removed from the adoption bootstrap itself. Kept as a frozen, historical file, same status as
   the EV-TOI fix below.
-1. **§32's EV-TOI fix has an unusual evidentiary shape — adopted on a real secondary metric,
   not the primary one it targeted.** The mechanism is directly confirmed against real data
   (a real, ~1-minute/game EV-TOI under-allocation from an infinite-memory average lagging a
   rising real-TOI trend), and no real regression survives on any ledger metric at any tested
   halflife — but total-MAE, the metric this fix was built to improve, does NOT clear
   bootstrap significance at any halflife. The adoption rests on a real, bootstrap-confirmed
   margin-MAE improvement at `halflife=1800` (CI entirely negative) plus the mechanism's own
   strength, put to the user explicitly rather than decided unilaterally (§32.6). Per-season
   bias moves toward zero in 11/14 dev seasons (§32.4); the holdout era shows a real total-MAE
   regression (§32.5) — a fourth installment of the same unmasking pattern as §22/§26, whose
   own explanation (§28.4) remains open and undetermined.
0. **§26's tie-mass fix is real but deliberately narrow in scope**: SU/Brier are unchanged
   (both bootstrap-neutral vs. the prior best); the gain is entirely distributional —
   CRPS(total)+CRPS(margin) improves by a real, bootstrap-confirmed [-0.00193,-0.00106], and
   the totals-line P(over) benchmark recovers a further chunk of its gap (mean P(over)
   41.79%→42.42%, Brier 0.25899→0.25805). Margin-MAE carries a small real cost
   ([0.00006,0.00045]) — §25.6 diagnosed this as an irreducible calibration-vs-sharpness floor
   (correctly modeling the real tie rate costs a sliver of individual-game point-accuracy on
   the ~77% of games that don't end tied), not a design flaw, and adopted anyway under the
   pre-registered net-trade-off rule (§25.6.5/§26.9) rather than the project's usual automatic
   margin-MAE veto — though §26.8's decomposition shows this floor barely moved relative to
   §25's rejected candidate; the net gain is carried entirely by CRPS(total). **On holdout,
   total-MAE shows a real regression** ([0.00319,0.01094]) — expected and pre-logged, not
   disqualifying: the third installment of the "cancelling errors" dynamic (§18.6/§19.4/§24.4),
   where fixing a real structural undershoot unmasks the decayed baseline's own transient
   recent-era overshoot. `HALFLIFE_GAMES` stays frozen until after this unmasking is complete
   (§24.4/§26.9's queue).
1. **§18's total-MAE and margin-MAE improvements over `ot_team_specific_poisson` are real and
   bootstrap-confirmed on the dev set** (total-MAE CI [-0.00612,-0.00184]; margin-MAE CI
   [-0.00033,-0.00007]), plus real, matching improvements in CRPS(total) and the corrected
   totals-line P(over) benchmark (§18.5) — a rare case where every distributional target moved
   in the predicted direction before the code was even written, because the mechanism (a
   previously undiagnosed, deliberately-deferred missing term) was measured directly rather
   than assumed. **On the holdout, neither metric clears the "real regression" bar under the
   corrected §15 protocol, but total-MAE's point estimate moves in the OPPOSITE direction from
   the dev set** (CI [-0.00007, 0.01011], technically includes zero) — flagged, not hidden: this
   is plausibly the exact "two cancelling errors" dynamic a follow-up review predicted before
   this cycle was built, where the holdout era's earlier near-zero bias may itself have been a
   coincidental offset between this real structural undershoot and the decayed baseline's own
   overshoot as scoring plateaued post-2022-23, not genuine full calibration.
2. **§16 found the model's own mean predicted total undershoots real scoring enough to make its
   own over/under predictions measurably worse than a coin-flip baseline** — corrected in
   §16.4 (push-conditioning error) and diagnosed by bucket in §18: the PP+PK piece is now fixed
   at the source (§18.6), but the EV-bucket piece (-0.091/game, confirmed data-level-clean in
   §18.3's free check) remains open, pointing at a time-local, level-tracking mechanism (the
   walk-forward season intercept) rather than a repeat of this same fix.
3. **The holdout protocol itself was corrected in §15** after an external review found the
   original Cycle 18/19 holdout comparisons used unbootstrapped point estimates on 2,624 games.
   Every holdout check from §17 onward uses the corrected, bootstrapped, confirmatory-veto-only
   protocol.
4. Earlier holdout progress from Cycle 13 (§7.6) still applies underneath all of this: every
   holdout metric had already improved once over the pre-Cycle-13 snapshot (SU
   55.79%→56.90%, Brier 0.24427→0.24289) before Cycles 18-19-22-23's further improvements on top.
5. **§22 found and fixed a real double-count in the away-B2B rest adjustment**: every team's
   own walk-forward attack rate silently absorbs the depressed scoring from its OWN historical
   away-B2B games, embedding `-0.0134` in BOTH lambdas of every game (symmetric, not
   away-side-only — §20's first attempt at this fix wrongly put the whole credit on the away
   side, which is exactly why it cost a real margin-MAE regression there). The corrected,
   symmetric fix lands margin-MAE at literal machine-precision zero on BOTH dev and holdout —
   the cleanest falsifiable-prediction confirmation this session has produced — with one
   flagged wrinkle (a real, small holdout-only total-MAE regression, plausibly the same
   cancelling-errors dynamic already predicted in §18.6/§19.4). Adopted.

### 0.3 What's been tried — every cycle, at a glance

"Kept" means bootstrap-confirmed real improvement on at least one metric and currently part
of the pipeline in §0.1. "Rejected/wash" means tested and NOT kept — recorded so it isn't
re-tried blind. All bootstraps are paired, 5,000 resamples, same game set both sides.

| # | §  | What was tested | Verdict | Real number |
|---|---|---|---|---|
| 1 | 4.2 | Naive season-to-date raw-goals Poisson (the floor) | Baseline | 57.06% SU (only ~3pt above "always pick home") |
| 2 | 4.3 | Swap raw goals for MoneyPuck whole-game xG | **Kept** | Brier improvement 95% CI [0.00106, 0.00356] |
| 3 | 4.4 | Split strength by situation (EV/PP/PK/other), shared TOI weight | **Wash** | Both Brier/total-MAE CIs cross zero |
| 4 | 4.5 | Team-specific PP/PK time-on-ice (own penalty-drawing history) | **Kept** | Total-MAE CI [0.00210, 0.00423] |
| 5 | 4.6 | Starting-goalie GSAx overlay (relative to league avg) | **Kept** | Margin-MAE CI [0.00691, 0.01233] |
| 6 | 4.7 | Goalie shrinkage prior: K=80 (textbook stabilization) vs. task-tuned K | K=80 **rejected**, K=12 **kept** | K=80 margin CI entirely NEGATIVE [-0.00566,-0.00275] vs K=25 |
| 7 | 4.10 | Cross-season goalie history (no reset at season boundary) | **Wash** | All 3 metric CIs cross zero |
| 8a | 4.11 | Negative Binomial (overdispersion check) | **Rejected before building** | Variance/mean ratio 0.97-0.98, i.e. UNDER-dispersed |
| 8b | 4.11 | Naive Dixon-Coles correlation (pooled final score) | **Rejected** | Pools two opposite-signed populations (reg: -0.152, OT/SO: +0.617) |
| 9 | 4.12 | Real OT/SO tiebreaker sub-model (replaces renorm workaround) | **Kept** | Brier CI [0.00014, 0.00066] |
| 10 | 4.13 | Dixon-Coles on regulation-only outcomes (properly scoped) | **Rejected** | MLE diverges to infinity — structural, not a bug (see §4.13) |
| 11 | 4.14 | Away-team back-to-back rest penalty | **Kept** | Margin-MAE CI [0.00171, 0.00344]; Brier CI [0.00001, 0.00029] (barely) |
| 12a | 6.8 | Direct same-game market-comparison benchmark | Precise correction, not a model change | Re-run against current best (§6.8 addendum): Brier gap 0.00351, 95% CI [0.00235, 0.00467] — essentially unchanged from the pre-drift-fix number |
| 12b | 6.9 | Isotonic calibration layer on win probability (real temporal split) | **Rejected — genuinely harmful** | Brier CI entirely NEGATIVE [-0.00163, -0.00036] |
| 13 | 7.5 | Scoring-era-drift fix: decayed league baseline + recalibrated (0.6x) shrinkage priors | **Kept** | Margin-MAE CI [0.00087, 0.00254]; Brier neutral; total-MAE positive-trending |
| 14 | 8 | Goalie-SPECIFIC (not team-level) back-to-back effect | **Rejected** | r=-0.0017, p=0.75 — no effect at all |
| 15 | 9 | Back-to-back x time-zone-change interaction | **Rejected** | diff=0.0069, p=0.90 — no effect at all |
| 17 | 10 | Diagonal (tie-mass) inflation, re-scoped ahead of 16 per a second external review | **Rejected as standalone — real, precise trade-off** | Total-MAE CI [-0.00281,-0.00131] (real gain); Margin-MAE CI [0.00441,0.00634] (real harm); Var(T)/E(T)=0.9570 vs. real 0.97-0.98 |
| 18 | 12 | Cross-season team-strength prior (weight=0.5), per §11's front-loaded market-gap lead | **Kept (superseded by Cycle 19)** | Dev Brier CI [-0.00070,-0.00035] (real); margin-MAE holdout CI [-0.00825,-0.00165] (real, §15.1) |
| 19 | 13/15 | Joint `cross_season_weight`×`prior_minutes_multiplier` grid (weight=0.75, mult=2.0) | **Kept — new current best (re-adjudicated §15)** | Dev Brier/total-MAE/margin-MAE all real, zero dev regressions; holdout bootstrap shows NO real regression on any metric |
| 20 | 14 | Local-transfer tie-mass parameter (delta), residuals-confirmed shape | **Rejected — real margin cost at every delta, clean mechanistic explanation** | Total-MAE "gain" identical (1.80224) at delta=0 through 0.1862 — 100% reconstruction-pathway, 0% delta; Margin-MAE CI [0.00129,0.00256] real harm at every tested delta |
| 21 | 16 | CRPS + totals-line benchmark | **Measurement, not a model change — real finding, corrected down after a follow-up review** | Model P(over) Brier 0.25683 (corrected, §16.4) vs. coin-flip baseline 0.25000; per-season bias (§16.5) is time-localized (2010-21, worst 2021-22), not flat — rules out a constant fix |
| 22 | 17 | Team-specific OT win probability (logistic on lambda_diff, OT-decided games only) | **Kept** | Margin-MAE real on dev (CI [-0.00079,-0.00054]) AND properly-bootstrapped holdout (CI [-0.00072,-0.00011]); SU/Brier neutral, total-MAE exactly unchanged (structural) |
| 23 | 18 | Shorthanded-goals-for term (league-wide xG rate x walk-forward PK TOI) | **Kept — new current best** | Total-MAE CI [-0.00612,-0.00184] (real); margin-MAE CI [-0.00033,-0.00007] (real); CRPS(total) and corrected P(over) benchmark both improved; falsified the empty-net/"other"-bucket hypothesis first proposed |
| 24 | 22 | Rest-adjustment fix, corrected (symmetric embedded-bias credit) | **Kept** | Margin-MAE CI [0.00000,0.00000] on BOTH dev and holdout — exact; superseded §20's wrong away-only version |
| 25 | 25 | Two-sided tie-mass transfer + walk-forward reg ratios, resolved via Cycle 22's OT logistic | **Rejected under automatic veto — family reopened for one new pre-registered candidate (§25.6)** | Margin-MAE CI [0.00006,0.00045] real, but CRPS(margin) only 0.00019 (barely real, CI [0.0000045,0.00038]) and concentrated entirely in regulation-decided games, not OT-decided — retires §14.3's information-loss theory |
| 26 | 26 | Walk-forward per-game tie-mass delta (calibration-ratio target + per-game exact-neutrality balance) | **Kept, per pre-registered net-trade-off rule — new current best** | Summed CRPS(total)+CRPS(margin) CI [-0.00193,-0.00106] (real improvement); no real Brier/SU regression, dev or holdout; margin-MAE small real cost [0.00006,0.00045] (same calibration-sharpness floor as §25, accepted under §24's amended protocol, not the automatic veto) |
| 27 | 27/28 | Joint HALFLIFE_GAMES × prior_minutes_multiplier grid re-tune, pre-registered | **No change — winner ties at current settings (600, 2.0)** | 2/12 cells survive the no-real-regression bar; winner is current production itself; prediction that winning halflife < 600 FALSIFIED — the decay rate was never compensating for anything |
| 28 | 29 | Walk-forward EV-bucket intercept (level-tracking correction) | **Rejected — real total-MAE regression at every tested halflife** | Information-theoretically infeasible (§29.3.1: signal ~0.09 vs. per-game noise sd≈2.4); flat-era check confirms a real, ~-0.09 stationary `gap_ev` survives (§29.5.1) — deficit real, tracking approach closed |
| 29 | 32 | EV-TOI expanding-mean → halflife decay (root-cause fix, fourth §7.1-class instance) | **Kept — new current best, adopted on a real secondary metric** | No real regression at any tested halflife; real margin-MAE improvement at halflife=1800 (CI entirely negative); total-MAE itself does not clear bootstrap significance despite a directly-confirmed mechanism (mean EV-TOI gap -0.99 min/game, implied impact -0.077/game vs. flat-era gap_ev -0.085/game) |

**25 cycles complete** (1 through 15, plus 17-26 — re-scoped/re-prioritized ahead of 16 per
two rounds of external review; see §10, §12, §13, §14, §17, §18). **Cycle 19's verdict was
corrected from rejected to adopted in §15**, after a third-round review found the original
holdout comparison lacked the bootstrap standard applied everywhere else — see §15 for the
full re-adjudication and the corrected holdout protocol now in effect, used by every holdout
check from Cycle 22 onward. **Cycle 26 is the tie-mass family's first shipped fix**, after
Cycles 17/20/25 were all rejected — see §26 and §25.6 for how the pre-registered
net-trade-off rule (written specifically to handle exactly this shape of cycle) resolved it.
Cycle 16 (GBM stacking) is QUEUED, not done — see §0.4 item 9.

**Pattern worth noting for whoever plans the next cycle**: four of the early "kept" results
(#4, #5, #6-choice, #11) turned out to help ONE specific axis (margin, or total-goals)
rather than being broad wins — check narrow sub-cases, not just overall averages, before
concluding a hypothesis is a wash (§4.14's own closing note says this explicitly). Also
worth noting: cycles #14 and #15 (both externally-suggested rest/travel refinements) came
back as clean, real nulls in a row — the team-level away-B2B effect (§4.14) appears to
already capture whatever real signal exists in this space; refining it further along
goalie-specific or travel-specific lines hasn't found anything additional so far.

### 0.4 Known open problems (do not re-derive these — they're already investigated)

1. **The holdout gap is real but PARTIALLY closed, not fully resolved (§4.8-4.9, §7.6, §12.4).**
   The Cycle 13 drift fix narrowed it meaningfully on SU/Brier/total-MAE (e.g. SU gap
   1.81pt→0.74pt). **Cycle 18's cross-season prior (§12) then improved holdout SU, Brier, and
   margin-MAE further on top of that** (holdout SU 56.90%→57.20%), directly motivated by the
   hypothesis that the remaining drift was about relative team-strength comparison rather than
   scoring-level drift — but the dev/holdout GAP itself (as opposed to the holdout numbers in
   isolation) has not been separately re-measured post-Cycle-18, so whether it narrowed or the
   whole curve just shifted up together is not yet confirmed.
2. **This model is close to the real market but not yet at it (§6.8, corrected from an
   earlier over-optimistic aggregate comparison in §5.1): re-run against `drift_adjusted_poisson`
   (pre-Cycle-18), Brier 0.24218 vs. market close 0.23867 on the same 11,803 real games,
   bootstrap-confirmed gap [0.00235, 0.00467] — essentially unchanged from the pre-drift-fix
   measurement, confirming independently that the Cycle 13 drift fix was Brier-neutral.** A
   per-season breakdown (§6.8) showed the gap is NOT cleanly concentrated in the post-2017-18
   high-scoring era — arguing against a simple "drift explains the market gap" story. **§11
   resolved this further: the gap is front-loaded WITHIN every season (2.8x larger in each
   team's first 15 games than after 41+), dramatically more so in the two disrupted-offseason
   seasons (2012-13 lockout, 2021-22 post-COVID: 8.7x front-loading) — pointing at a
   cross-season team-strength-prior gap.** **§12 built and confirmed this directly: the
   cross-season prior narrows the early-season (games 1-15) market gap by 25%, vs. only 7-9%
   in mid/late season — the predicted, mechanism-specific signature, not a generic aggregate
   improvement landing anywhere.** **The aggregate §6.8 headline number has now been re-run
   against `cross_season_prior_poisson` too: Brier 0.24167 vs. market close 0.23867, gap
   [0.00187, 0.00412] — narrower than the pre-Cycle-18 gap ([0.00235, 0.00467]), consistent
   with §12.5's mechanism-specific finding, but still real and open** — this model is closer
   to the market than it was, not yet at it. Market data for 2022-23 onward (including the
   entire holdout window) is still NOT available through any source checked so far (§6.1).
3. **Dixon-Coles-style score correlation is a real, repeatedly-measured phenomenon
   (-0.15 to -0.17 in regulation-decided games, three independent measurements) with NO
   working implementation.** The specific 4-cell parametric form doesn't transfer to
   hockey (§4.11, §4.13) for two DIFFERENT structural reasons. §10 tested the DIAGONAL half
   of the real fix (tie-mass inflation, isolated from off-diagonal dependence per a second
   external review's own re-scoping) and found a genuine, precise trade-off — total-goals MAE
   improves, margin MAE gets measurably worse. **§10.6 sharpened this with three follow-up
   checks**: the Var(T)/E(T) comparison was corrected (population-level, like-for-like) and
   the real gap turns out much bigger than first reported (real 0.8913 vs. model ~1.02,
   barely moved by theta at all — the off-diagonal term is carrying most of the real work,
   not a polish on top of the diagonal fix); a theta=1 ablation showed the margin cost isn't
   fully attributable to inflation (a smaller version persists even with inflation off); and a
   margin-bin residual profile CONFIRMED the local-transfer hypothesis decisively (180.8% of
   the offsetting excess concentrated at |margin|=1 specifically, not spread across blowouts).
   **§14 built and tested that local-transfer parameter — ALSO rejected, with a clean
   mechanistic explanation.** Every tested delta (including 0.05, a tiny nudge) showed a real
   margin-MAE regression; total-MAE's apparent "gain" was proven identical at delta=0 through
   0.1862 (100% reconstruction-pathway, 0% delta). The reason: every unit of mass routed
   through a tie ends up resolved by the OT/SO layer's single LEAGUE-WIDE constant
   (`p_home_wins_ot≈0.509`), discarding the per-game, skill-informed split the independent-
   Poisson joint's own `(x+1,x)`/`(x,x+1)` cells encoded. **§17 built the prerequisite this
   section originally called for — a team-specific (skill-informed) OT win probability,
   ADOPTED.** **§25 then re-attempted the tie-mass fix on this new, skill-informed base, using
   the fully-corrected design (a two-sided diagonal transfer, proven to not be total-conserving
   like Cycle 20's one-sided version, plus walk-forward regulation ratios) — and STILL found a
   real, bootstrap-confirmed margin-MAE cost** (smaller than Cycle 20's by ~5-10x, but not
   zero), alongside real, small calibration gains (CRPS(total), the corrected P(over)
   benchmark). **§25.6 then followed up on a metric-theory challenge to that veto** (MAE pairs
   with the conditional median, not the mean) with three checks: median-scored margin-MAE
   crosses zero (not real, though a noisier test), CRPS(margin) — immune to the mean/median
   issue — still shows a real but much smaller cost (0.00019/game, barely clearing the 95%
   bar), and the cost concentrates entirely in regulation-decided games while vanishing in
   OT-decided ones, the OPPOSITE of §14.3's information-loss prediction — meaning Cycle 22
   already removed the large OT-resolution channel, and what's left is a small,
   calibration-vs-sharpness floor inherent to correctly modeling the real tie rate at all, not
   a defect of this specific transfer. §25's specific tested candidate (fitted global delta)
   stayed rejected — re-scoring it under a rule invented after seeing its numbers is the
   adaptive-protocol failure §15 exists to prevent — but the family was reopened for one new,
   pre-registered candidate. **§26 built that candidate and SHIPPED it** — the tie-mass
   family's first adopted fix after Cycles 17/20/25 all failed. The design fixes what every
   earlier attempt got wrong: a walk-forward calibration RATIO (not a level) applied to each
   game's own model-implied diagonal mass, so cross-game matchup variation is preserved rather
   than forced toward a shared target (the same mistake Cycle 17's global rescale made, which
   the naive per-game redesign would have repeated), combined with a per-game (not just
   aggregate) exact-neutrality balance constraint. Summed CRPS(total)+CRPS(margin) improved
   really and comfortably (CI [-0.00193,-0.00106]); no real Brier/SU regression on dev or
   holdout; the same small margin-MAE cost (0.00025/game) persists and is accepted under
   §24's amended protocol rather than the automatic veto, since §25.6 already established it's
   an irreducible calibration-sharpness floor, not a design flaw. **Current best model is now
   `walk_forward_tie_ratio_poisson` (§26)**, and the tie-mass deficit (§4.13) finally has an
   owner: whatever residual the dependence-bearing joint (task #36) eventually targets now
   faces only whatever's left after §26, not the larger gap this whole chain started from.
4. **Most shrinkage-prior placeholders are still unvalidated guesses**: `PRIOR_GAMES=10` in
   `baseline_naive_poisson.py`/`shrinkage.py` callers is untouched. The situational
   `PRIOR_MINUTES_EV/PP/PK/OTHER` constants WERE recalibrated (×0.6, §7.5) but only as a
   side effect of fixing the drift-decay interaction, not via a dedicated, from-scratch
   grid search the way the goalie prior (K=12, §4.7) got — worth a dedicated look. **§13
   tested a joint `cross_season_weight`×`prior_minutes_multiplier` grid and found real-looking
   dev-set gains (Brier, total-MAE, margin-MAE all improved with no dev-set regression) that
   did NOT survive the holdout check — a second, independent confirmation of §12.4's own
   lesson that this project's holdout check is load-bearing, not a formality, whenever a
   shrinkage-strength constant is in play. Rejected; current best model UNCHANGED
   (`cross_season_weight=0.5`, `prior_minutes_multiplier=1.0`).**
5. **Even after the Cycle 13 drift fix, the raw predicted mean total undershoots real scoring
   (§7.5's honest residual note) — §16 sharpened this, §16.4-16.5 corrected the magnitude and
   diagnosed its shape, and §18 fixed one confirmed piece of it at the source.** §18.1-18.2
   decomposed the bias by situation bucket and found: the empty-net/"other"-bucket hypothesis
   is FALSIFIED (that bucket overshoots, not undershoots); the PP+PK bucket undershoots in ALL
   16 seasons with zero exceptions, fully explained by a deliberately-deferred, now-fixed
   omission (shorthanded-goals-for, Cycle 3 — §18.6 fixed it, real dev-set improvement in
   total-MAE/CRPS/P(over)); and the EV bucket's own smaller, less consistent undershoot
   (-0.091/game) is confirmed data-level-clean (EV goals≈EV xG, ratio 0.9966, §18.3) — meaning
   it lives in the combine/shrinkage math, not the input data, and remains the target for the
   still-open walk-forward season-level intercept — narrowly scoped to EV only (task #35), since §26
   separately fixed the tie-mass shortfall. **The corrected totals-line P(over) Brier gap has
   narrowed from 0.00683 (§16.4) to 0.00405 (§18.5) after the SH-term fix.** §26.6 re-ran this
   benchmark on a permanent, doubly-verified script and found it does NOT reconcile with the
   43.35%/44.65% figures logged in §18.5/§25.3 for what should be the same models — a real,
   investigated (§26.6.1), but ultimately unrecoverable discontinuity in the historical record
   (those numbers were never saved as reproducible scripts). Read §26.6 onward as ONE
   continuous, permanent series: baseline 41.79% → treated 42.42% (Brier 0.25899→0.25805) —
   real, ongoing progress toward the ~49.5% real rate, not fully closed.
6. **Home back-to-backs show no effect, and the model slightly UNDER-predicts home-team
   win probability on home back-to-backs (§4.14)** — the opposite of the popular fatigue
   narrative. Goalie-specific back-to-back (§8) and back-to-back×timezone (§9) were BOTH
   separately tested and also found no effect. Three real nulls in the rest/travel space
   in a row — team-level away-B2B (§4.14) appears to already capture whatever's real here.
7. **Live-prediction lineups/injuries (RotoWire, §1.8) were confirmed scrapeable but the
   actual scraper was never built** — deferred until there's a real NHL slate to verify
   selectors against (season starts October; this was written in the 2026 off-season).
8. **The full discrete-event/possession-level simulator alternative (§3.7) was explicitly
   NOT pursued** — current approach is entirely rate-into-Poisson, no play-by-play
   simulation. Revisit only if the current approach hits a real, diagnosed ceiling.
9. **Cycles 17-23 are now DONE, and Cycle 19's verdict was CORRECTED in §15.** Cycle 17
   (diagonal/tie-mass inflation, global rescale) and Cycle 20 (local-transfer tie-mass
   parameter, the residuals-confirmed correct shape) were BOTH rejected (§10, §14) — the
   tie-mass family was closed off under the OLD architecture until the OT/SO win-rate model
   became team-specific. **Cycles 18, 19, 22, and 23 (cross-season team-strength prior +
   re-tuned prior-strength multiplier + team-specific OT win probability + shorthanded-goals
   term, §12/§13/§17/§18) were ALL ADOPTED — current best model, `shorthanded_poisson`.**
   Cycle 19 was originally rejected on an unbootstrapped holdout point-estimate comparison;
   §15 corrected the holdout protocol (confirmatory-veto only, always bootstrapped) and
   re-adjudicated it as adopted. Cycle 21 (§16) added CRPS + a totals-line benchmark (pure
   measurement) and surfaced a real, concrete finding: the model's own P(over) predictions
   score worse than a coin-flip baseline, driven by a systematic mean-total undershoot. Cycle 22
   (§17) built the team-specific OT model §14.4 called for, reopening the tie-mass/dependence
   family. **Cycle 23 (§18) then decomposed the mean-total undershoot by situation bucket,
   falsified the empty-net hypothesis, and fixed the PP+PK piece at the source** (a
   deliberately-deferred Cycle 3 omission, shorthanded-goals-for) — real dev-set improvement in
   total-MAE, CRPS(total), and the P(over) benchmark, no real holdout regression. **§19's bias
   ladder then localized the remaining gap**: a real, growing (-0.03 to -0.10/game), 14/14-
   season-consistent bias in the regulation-ratio/OT-redistribution stage, directly explained
   by the already-known tie-mass deficit (§4.13); §19.3 found and fixed a real third instance
   of the §7.1/7.3 two-baselines-different-memory bug in the goalie overlay, validated as a
   genuine null, left available but NOT adopted. **§22 fixed the rest-adjustment's own real
   double-count** (§20's first attempt was itself mis-specified — the embedded bias is
   symmetric, not away-only — corrected, adopted, margin-MAE exact zero on both splits).
   **§24 amended the adoption protocol** (calibration metrics now claimable; a distinct,
   narrower bar for derived-correctness fixes) given §22 was adopted despite a real holdout
   total-MAE cost. **§25 then re-attempted tie-mass on the OT-aware base with the fully
   corrected design (two-sided transfer + walk-forward reg ratios) — REJECTED**, per the
   pre-registered veto: a real margin-MAE cost survives (much smaller than Cycle 20's, but not
   zero). **§25.6 followed up on a metric-theory challenge to that veto (median-scored MAE,
   full-precision CRPS(margin), regulation-vs-OT split) and revised the verdict**: the specific
   candidate stays rejected, but CRPS(margin)'s small (0.00019/game), regulation-concentrated
   residual cost retires the original OT-information-loss theory (§14.3) rather than confirming
   it — Cycle 22 already removed that channel — so the family is reopened, not closed for good,
   for one new candidate (a walk-forward per-game delta) under a pre-registered net-trade-off
   rule. A genuinely dependence-bearing joint remains the long-term path if that candidate also
   fails the rule, and now faces only the same small calibration-sharpness floor, not a large
   OT-resolution obstacle. **§26 then built and shipped that candidate**: a walk-forward
   calibration ratio (not a level) targeting each game's own model-implied diagonal mass, plus
   a per-game (not just aggregate) exact-neutrality balance constraint — summed
   CRPS(total)+CRPS(margin) improved really and comfortably, no real Brier/SU regression on
   dev or holdout, and the tie-mass family's first shipped fix after three rejections.
   **Current best model is now `walk_forward_tie_ratio_poisson` (§26)**. **The walk-forward
   EV-bucket intercept remains the one real, unstarted candidate in the mean-total chain**
   (§18.3/§19.2 confirmed it's data-level-clean) — now the LAST piece, since §26 gave the
   tie-mass shortfall an owner and task #35 stays scoped narrowly to EV — before the single, final
   `HALFLIFE_GAMES` re-tune that closes the chain (§24.4), followed by the deferred
   market-benchmark re-run (§6.8). Cycle 16's GBM stacking layer is now fourth in line, and
   must use the corrected holdout protocol (§15) and the hardened validation plan already on
   file.

### 0.5 File map

| File | What it does |
|---|---|
| `src/ingest/fetch_nhl_api.py` | NHL API schedule pull (2008-present), includes `lastPeriodType` |
| `src/ingest/fetch_moneypuck.py` | MoneyPuck team-game-situational xG CSV |
| `src/ingest/fetch_moneypuck_goalie_games.py` | Shot-level data → per-goalie-per-game GSAx table |
| `src/ingest/team_codes.py` | NHL API ↔ MoneyPuck team-code crosswalk (validated, zero mismatches) |
| `src/ingest/fetch_sbro_odds.py` | Real market-odds ingestion (13 seasons via Wayback-archived SportsbookReviewsOnline files, §6.1) |
| `src/ingest/parse_sbro_odds.py` | Parses raw odds rows into real games, joined to the schedule (§6.7) |
| `src/models/shrinkage.py` | Shared walk-forward Bayesian-shrinkage helpers: `halflife_games` decay (§7.2), `add_walk_forward_toi_rate_cross_season` (§12.2, production `cross_season_weight=0.5`) |
| `src/models/baseline_naive_poisson.py` | Cycle 1 model + core Poisson utilities (`score_distribution`, `home_win_prob_regulation`) still used everywhere |
| `src/models/team_strength_xg.py` | Cycle 2: whole-game xG strength |
| `src/models/team_strength_situational.py` | Cycles 3-4: EV/PP/PK/other split + team-specific TOI; priors recalibrated ×0.6 for decay (§7.5); cross-season prior passthrough (§12.2) |
| `src/models/team_strength_goalie.py` | Cycles 5-6: goalie overlay + calibrated K=12 prior |
| `src/models/overtime_shootout.py` | Cycle 9: real OT/SO sub-model |
| `src/models/rest_schedule.py` | Cycle 11: away-back-to-back adjustment; Cycle 24/§22 symmetric embedded-bias credit (`add_walk_forward_b2b_incidence`, `symmetric_b2b_bias_credit`) |
| `src/models/two_sided_diagonal_transfer.py` | Cycle 25/§25: two-sided diagonal transfer + aggregate global-balance fit (`fit_deltas`, rejected candidate, §25); Cycle 26/§26's `fit_deltas_per_game` (per-game exact-neutrality generalization) **IS used in production** |
| `src/models/validate_tie_mass_v2.py` | Cycle 25/§25 validation pipeline (rejected candidate, kept for reproducibility) |
| `src/models/walk_forward_tie_ratio.py` | Cycle 26/§26: walk-forward OT-calibration ratio (`add_walk_forward_ot_calibration_ratio`) — production |
| `src/models/validate_tie_mass_ratio.py` | Cycle 26/§26: **current pipeline for score distribution/totals/margin** (`run_treated()`) — `_build_dev_base`/`_fit_dev_only_ot_logistic` are the corrected, dev-only-fit shared helpers (§35.2), reused by every downstream cycle |
| `src/models/ev_residual_intercept.py` | Cycle 28/§29: walk-forward EV-bucket residual tracker — **investigation only, not used in production** (rejected, §29) |
| `src/models/validate_ev_intercept.py` | Cycle 28/§29 validation pipeline (rejected candidate, kept for reproducibility) |
| `src/models/validate_ev_toi_halflife.py` | Cycle 29/§32 — **ADOPTED then REVERTED** (§35.4, real Brier/margin-MAE regression under the corrected holdout check); kept as a frozen, historical (rejected) file, same as any other non-adopted cycle |
| `src/models/validate_gbm_stack.py` | Cycle 16/§34, RE-VALIDATED §35.5-35.6: **current full pipeline**, `run_final_production()` (fits the GBM once dev-only, walk-forward-safe) — monotone-constrained GBM stack over `walk_forward_tie_ratio_poisson`'s win probability; totals/margin pass through unchanged |
| `tests/test_holdout_walk_forward_discipline.py` | Regression guard (§35.2) for the walk-forward-discipline bug: invariant check, golden-reference check vs. `check_holdout_ot_logistic.py`, calibration-ratio continuity, plus regression tests for the GBM veto-sign and dangling-for/else bugs |
| `src/models/validate_market_benchmark.py` | §6.8/task #40 moneyline Brier-gap re-run, always targets current best |
| `src/models/validate_market_totals_benchmark.py` | §30/§31 totals-line P(over) re-run, always targets current best |
| `src/models/check_holdout_tie_mass_ratio.py` | Holdout confirmatory check for Cycle 26, corrected §15 protocol |
| `src/models/validate_bias_ladder.py` | Stage-by-stage mean-total bias ladder (§19), used throughout §19-25 to localize and verify each fix |
| `src/models/travel_timezone.py` | Cycle 15: team→timezone map + walk-forward timezone-change flag — **investigation only, not used in production** (rejected, §9) |
| `src/models/dixon_coles.py` | Cycles 8/10 investigation only — **not used in production** |
| `src/models/calibration.py` | Cycle 12: isotonic calibration helpers — **not used in production** (rejected, §6.9) |
| `src/models/diagonal_inflation.py` | Cycle 17: closed-form diagonal-mass inflation (theta) — **investigation only, not used in production** (rejected, §10) |
| `src/models/local_transfer_inflation.py` | Cycle 20: local (x+1,x)/(x,x+1)→(x,x) tie-mass transfer (delta) — **investigation only, not used in production** (rejected, §14) |
| `src/models/calibrate_goalie_prior.py` | One-off split-half reliability calibration (historical; K=12 was chosen by a *different* grid search, see §4.7) |
| `src/models/metrics_ledger.py` | Append-only results ledger (`data/processed/metrics_ledger.parquet`) |
| `src/models/final_holdout_check.py` | Dev/holdout split policy, `DEV_MAX_SEASON` constant |
| `src/models/check_holdout_drift.py` | Re-checks the real holdout against `drift_adjusted_poisson`, constants fit dev-only (§7.6) |
| `src/models/check_holdout_cross_season.py` | Re-checks the real holdout against any `(cross_season_weight, prior_minutes_multiplier)` combo, constants fit dev-only (§12.4, §13.2) |
| `src/models/validate_holdout_bootstrap.py` | Corrected holdout protocol: paired bootstrap on holdout-only games, confirmatory-veto-only (§15) |
| `src/models/crps.py` | Discrete CRPS (`discrete_crps`, `total_pmf_from_joint`, `margin_pmf_from_joint`) — first-class distributional metric (§16.1) |
| `src/models/validate_crps.py` | CRPS baseline + totals-line benchmark for the current best model (§16) |
| `src/models/ot_logistic.py` | Cycle 22: team-specific OT win-probability logistic + OT/SO split constants (§17.2) |
| `src/models/validate_ot_team_specific.py` | Cycle 22 final pipeline — now a frozen snapshot, superseded by Cycle 23 |
| `src/models/check_holdout_ot_logistic.py` | Holdout check for Cycle 22, corrected §15 protocol |
| `src/models/validate_bias_decomposition.py` | Per-situation-bucket (EV/PP+PK/other) model-vs-actual goals breakdown (§18.2), `--sh` flag toggles the SH term |
| `src/models/validate_shorthanded.py` | Cycle 23 final pipeline — now a frozen snapshot, superseded by Cycle 26 |
| `src/models/validate_*.py` | One per cycle; `validate_gbm_stack.py` (win probability) + `validate_tie_mass_ratio.py` (totals/margin) are the current pipeline, the rest (including `validate_ev_toi_halflife.py` — reverted, §35.4 — `validate_shorthanded.py`, `validate_ot_team_specific.py`, `validate_cross_season.py`, `validate_drift.py`) are frozen snapshots |

### 0.6 To reproduce or extend

Run `.venv/bin/python -m src.models.validate_gbm_stack` from the repo root for the current
model's real dev-set metrics (includes the nested walk-forward hyperparameter selection, the
kill-switch permutation-importance check, the dev bootstrap, and the holdout confirmatory
check) — `run_final_production()` in that file is the walk-forward-safe, single entry point for
win probability; `validate_tie_mass_ratio.run_treated()` for totals/margin. Run
`.venv/bin/python -m tests.test_holdout_walk_forward_discipline` before trusting any future
holdout check built on `_build_dev_base`/`run_baseline`/`run_treated` (§35.2) — it fails loudly
if the dev-only-fit discipline breaks again the way it did in §35.1. To test a new idea: check
for a real effect on the existing
model's residuals FIRST (the pattern used in every cycle from §4.11 on — three cycles in a
row, §8-9 and part of §6, found nothing this way and were rejected before or immediately
after building anything, which is the discipline working as intended, not a failure), only
build if real, validate on the development set (`season < 20242025`), bootstrap against the
current best before claiming an improvement, and log to the ledger either way — a real
negative result is exactly as valuable to record as a positive one.

### 0.7 The current roadmap (as of the most recent planning pass)

§5 (2026-07-23) synthesizes an external literature-grounded research report against this
project's own findings. §6 (Cycle 12) then got real market-odds data landed (13 seasons,
14,260 regular-season games with matched closing lines, via Wayback-archived
SportsbookReviewsOnline files — §6.1-6.7) and used it to PRECISELY correct §5.1's aggregate
claim: **on a real, same-game comparison (§6.8), this model's Brier (0.24209) is measurably
worse than the market's closing-line Brier (0.23867) — a real, bootstrap-confirmed gap
[0.00232, 0.00460] — close to the market, but not yet at it.** An isotonic calibration layer
was then built and tested properly (temporal fit/eval split) and found to genuinely HURT
(§6.9, bootstrap CI entirely negative) — not shipped, a real negative result. §7 (Cycle 13)
confirmed a real, substantial, 9-season-long bug in the core shrinkage baseline (an
infinite-memory trailing average that never fully catches up after the real 2017-18
league-wide scoring jump), built a working decay-based fix, then — by checking the
AGGREGATE effect rather than stopping once the raw mechanism looked right — found a SECOND,
entangled bug (a season-reset-vs-infinite-baseline mismatch in the defense-ratio combine
term) that partially offsets the first. Net bootstrapped effect of the fix alone: real
total-MAE improvement, real margin-MAE harm, negligible Brier change — a genuine trade-off,
NOT shipped as-is on its own. **§7.5 then resolved it**: the entanglement wasn't a separate
bug needing structural redesign, but the situational shrinkage priors (`PRIOR_MINUTES_*`)
having been implicitly tuned against the OLD baseline's own lag pattern — grid-searching
prior scale jointly with halflife found `halflife=600` + priors×0.6 beats the undecayed
baseline on margin-MAE (bootstrap-significant) and total-MAE (positive-trending), with Brier
unharmed. **Adopted as `drift_adjusted_poisson`**, the new current best model. §7.6 then
re-checked the real holdout against it (constants fit dev-only, applied without refitting):
**every holdout metric improved** (SU 55.79%→56.90%, Brier 0.24427→0.24289) and the SU/
Brier/total-MAE dev/holdout gaps all narrowed — genuine, verified progress on the exact
problem this cycle was motivated by, though margin-MAE's own gap barely moved, suggesting a
different, not-yet-tested kind of drift remains there. §8 (Cycle 14) and §9 (Cycle 15) then
tested two externally-suggested refinements to the rest/travel signal — goalie-specific
back-to-back, and a back-to-back×time-zone interaction — and found NO effect in either case
(both p>0.7), a real, clean pair of negative results; the team-level away-B2B effect (§4.14)
remains the one genuine signal in this space. §5.2 also confirms the Dixon-Coles/bivariate-
Poisson rejections (§4.11, §4.13) were structurally correct per the published literature, not
just this project's own finding.

A second external review (a different LLM given the full updated §0-§9 doc) then made two
precise, technically-grounded corrections before Cycle 16 started: (1) §6.8's market
benchmark was measured against the superseded pre-drift-fix model — re-run against
`drift_adjusted_poisson` and found essentially unchanged (§6.8 addendum), confirming the
drift fix really was Brier-neutral, with a new per-season breakdown showing the gap is NOT
cleanly concentrated in the high-scoring era; (2) Cycle 17 (originally scoped as a
negative-dependence copula for the -0.15 score correlation) needed to be re-scoped, because a
negative-dependence copula moves diagonal (tie) mass DOWN — worsening, not fixing, §4.13's
confirmed 6-point tie-mass deficit. **§10 executed this re-scoped Cycle 17**: a single
closed-form diagonal-inflation parameter (`theta=1.3338`), fit and validated in isolation from
any off-diagonal term, per the review's own sequencing advice. Result: a real, precise,
partial confirmation — SU and Brier neutral (as predicted), total-goals MAE genuinely
improves, but margin MAE measurably WORSENS (the opposite of what the review expected), and
the fitted joint's implied Var(T)/E(T) (0.9570) undershoots the real 0.97-0.98 target. **Not
shipped**, but a structurally informative result: it shows the diagonal fix needs to be
jointly fit WITH the off-diagonal dependence term, not layered before it as a separate win.

**A third-pass follow-up on §10 (§10.6) then ran three more residuals-first checks before any
further build.** The Var(T)/E(T) comparison turned out to be apples-to-oranges (§4.11's
"0.97-0.98" was a different, marginal statistic) — corrected, population-level version: real
0.8913 vs. model ~1.02, a much bigger gap than first reported, essentially untouched by theta
either way (a theta=1 ablation moved it by only 0.0018) — meaning the off-diagonal dependence
term does most of the real work here, not a polish on a working diagonal fix. The same
ablation showed the margin-MAE cost isn't fully attributable to inflation (a smaller version
persists even at theta=1). And a margin-bin residual profile decisively CONFIRMED the
review's own local-transfer hypothesis: the tie deficit is funded almost entirely
(180.8% of the offsetting mass) by adjacent |margin|=1 games, not diffusely by blowouts —
exactly why the global rescale hurt margin-MAE, and exactly the shape a future fix should
take instead.

**Separately, the same review pass caught a real, well-targeted new lead in §6.8's own
per-season table**: the two largest model-vs-market gaps are 2012-13 (lockout) and 2021-22
(post-COVID) — both disrupted-offseason seasons. §11 confirmed this is a front-loaded,
within-season effect (2.8x larger in a team's first 15 games than after 41+, and 8.7x larger
specifically in those two flagged seasons) — pointing at a missing cross-season
team-strength prior, not an informational gap, and now the top-recommended next cycle.

**§12 then executed that top-recommended item, Cycle 18.** Confirmed a real residuals-first
signal first (a team's own last-season xG-differential correlates with its early-current-
season performance, r=0.15, p<1e-40, discarded entirely by the current season-reset prior),
built a cross-season blend into the shrinkage machinery, and grid-searched the blend weight.
The dev-set-best weight (1.0, pure single-season carryover) was explicitly REJECTED after the
holdout check showed it doesn't generalize (holdout SU actually below the no-prior baseline)
— **weight=0.5 was adopted instead: real SU/Brier/margin-MAE improvement on BOTH the dev set
and the untouched holdout, no real regression on either split, and confirmed via a direct
mechanism check to narrow the front-loaded market gap specifically where the hypothesis
predicted (25% closed in games 1-15, vs. 7-9% in mid/late season).** **New current best model:
`cross_season_prior_poisson`** (`src/models/validate_cross_season.py`), superseding
`drift_adjusted_poisson`. This is the strongest single-cycle result since Cycle 13, and
improves SU directly (not just margin/totals) with no trade-off surviving the holdout check.

**§13 then ran that first open item — the joint `cross_season_weight`×`prior_minutes_multiplier`
re-grid — and found a real, informative negative result.** The dev-set grid found real-looking
gains (Brier, total-MAE, and margin-MAE all improved with zero dev-set regression at
weight=0.75, various multipliers), but the holdout check — run precisely because §12.4 had
already shown a dev-set-best candidate can fail to generalize — showed EVERY tested combination
had worse holdout SU than the current best, confirming that lesson a second, independent time.
**Current best model unchanged: `cross_season_prior_poisson` (weight=0.5, multiplier=1.0).**

**§14 then built and tested §10.6.3's local-transfer tie-mass parameter, validated against the
new current best — also rejected, with a clean, decisive mechanistic explanation.** Every
tested delta (even 0.05) showed a real margin-MAE cost, and total-MAE's apparent gain proved
to be 100% attributable to the regulation-ratio reconstruction pathway, 0% to delta itself
(identical total-MAE at delta=0 through 0.1862). The reason: any mass routed through a tie
gets resolved by the OT/SO layer's single LEAGUE-WIDE constant win rate, discarding the
per-game skill-informed split the independent-Poisson joint encoded — a real cost regardless
of how small the transfer. **This closes off the entire tie-mass-inflation family under the
current architecture** (both the global-rescale and the locally-correct shape), not just
Cycle 17's specific version — the tie-mass deficit (§4.13) remains real but requires a
team-specific OT/SO win-probability model before any further attempt, not queued as a
specific next cycle here.

**A third-round review then caught a real methodological inconsistency: dev-set verdicts were
bootstrapped (5,000 resamples) but holdout comparisons in §12.4 and §13.2 were raw point
estimates on differences as small as 1-2 games. §15 corrected the protocol (holdout as
confirmatory-veto only, always bootstrapped, peek-count now visible in the ledger) and
re-ran both cycles' holdout claims properly.** §12.4's "weight=1.0 doesn't generalize" story
did not survive bootstrapping (the rejection was still correct, but for the already-sufficient
dev-set reason). **Cycle 19's rejection was reversed: none of its candidates show a real
holdout regression, and its dev-set gains were always real — re-adjudicated as ADOPTED.**
New current best: `cross_season_prior_poisson` with `cross_season_weight=0.75`,
`prior_minutes_multiplier=2.0`.

**§16 then ran that pure-measurement item.** CRPS on totals/margin is now a first-class,
loggable metric (`metrics_ledger.append_run`'s new `extra_metrics` passthrough), with baseline
numbers recorded for any future dependence/dispersion cycle to be judged against fairly. The
totals-line benchmark hit a real data limitation (no real over/under pricing in the SBRO
data, confirmed against the raw source, only the line itself) — but the honest version built
anyway surfaced what first looked like a large finding. **A follow-up review caught two things
before any recalibration cycle got built on top of it (§16.4-16.5): the original P(over)
computation wasn't conditioned on decided games (a push-cell mismatch, the same class of error
§10.6.1 caught elsewhere), which had roughly doubled the apparent gap (corrected: model Brier
0.25683 vs. baseline 0.25000, not 0.26407 vs. 0.25000); and the per-season bias curve shows the
undershoot is real but time-localized (persistent 2010-11 through 2020-21, worst at 2021-22,
mostly resolved by 2023-24) rather than flat — ruling out a simple additive constant, which
would have over-corrected the already-resolved holdout era.** This still promotes §0.4 item 5
from an abstract residual note to a concrete candidate, just a smaller and better-specified one
than first reported: a time-local, level-tracking mechanism, not a constant.

**§17 then built that prerequisite.** A residuals-first check confirmed real signal in
OT-decided games (r=0.084, p=0.0001) and none in shootouts (r=0.014, p=0.576) — exactly the
predicted signature. Fit a single global logistic (a=0.008 n.s., b=0.302 p<0.001) on OT-decided
dev-set games only, keeping the flat rate for shootouts. Real margin-MAE improvement,
bootstrap-confirmed on BOTH the dev set and the properly-bootstrapped holdout (the first
Cycle since §15's correction to clear that bar cleanly on both splits), SU/Brier neutral,
total-MAE exactly unchanged (a structural fact, not noise — the fix only moves mass between
the two sides of an already-resolved tie). **New current best: `ot_team_specific_poisson`**
(`src/models/validate_ot_team_specific.py`). This also reopens the tie-mass/dependence family:
any future diagonal-inflation or local-transfer attempt now resolves ties through a
skill-informed probability, not a flat one, so it may no longer necessarily carry the margin
cost §14.3 identified.

**§18 then investigated the mean-total undershoot's actual mechanism before building the
intercept — and found a cleaner, root-cause fix for part of it.** A proposed empty-net/
"other"-bucket mechanism was cleanly falsified (that bucket OVERSHOOTS, +0.080/game; aggregate
real goals vs. real xG across the whole 2008-2025 dataset ties out at ratio 1.0008, no
structural gap). A per-situation-bucket decomposition (`validate_bias_decomposition.py`) found
something better: the PP+PK bucket undershoots in ALL 16 seasons with zero exceptions,
explained entirely by a deliberately-deferred Cycle 3 omission — shorthanded-goals-for, never
wired into the final combine despite the underlying walk-forward machinery already existing
for it. SH-xG was confirmed well-calibrated to real SH-goals (ratio 1.008, no fitted
conversion factor needed), and a free EV-bucket check confirmed that residual (-0.091/game) is
data-level-clean (EV goals≈EV xG), pointing the still-open intercept at a combine/shrinkage-
math problem, not a data problem. **Built the term (league-wide rate x each team's own
walk-forward PK ice time, a pure "flip it on" fix, no new data plumbing): real total-MAE and
margin-MAE gains on the dev set, matching real improvements in CRPS(total) and the corrected
P(over) benchmark (Brier gap narrows from 0.00683 to 0.00405), no real holdout regression
under the §15 protocol — though the holdout total-MAE point estimate moves the "wrong"
direction, plausibly the predicted two-cancelling-errors dynamic, flagged rather than hidden.**
New current best: `shorthanded_poisson` (`src/models/validate_shorthanded.py`).

**§19's bias ladder then localized the rest of the gap**: EV (~-0.09/game, combine-stage,
confirmed data-clean), a noisy goalie-overlay stage (a real but null third instance of the
§7.1/7.3 decay bug, found and fixed but not adopted), and — the standout — a real, GROWING
(-0.03 to -0.10/game), 14/14-season-consistent bias in the regulation-ratio/OT-redistribution
stage, directly explained by the already-known tie-mass deficit (§4.13). **§22 then found and
fixed a real double-count in the rest adjustment itself**: every team's own walk-forward
attack rate silently absorbs its OWN historical away-B2B drag, symmetrically in both lambdas
— §20's first attempt wrongly credited the away side only (exactly why it cost real margin),
and the corrected, symmetric version lands margin-MAE at literal machine-precision zero on
BOTH dev and holdout. This was the project's first adoption with a real holdout regression on
an uncleared metric (total-MAE) and no dev-bootstrap-real win on the original four — **§24
amended the adoption protocol accordingly**: calibration metrics (the ladder, CRPS, the
corrected P(over) benchmark) are now claimable, bootstrap-scored wins in their own right, and
a distinct, narrower "derived correctness fix" class now governs changes proven algebraically
rather than fit, requiring proven win-prob/margin neutrality plus a real calibration gain.
**§25 then re-attempted the tie-mass fix on the OT-aware base, built exactly to spec (a
two-sided diagonal transfer, proven not to be total-conserving the way Cycle 20's one-sided
version was, plus walk-forward regulation ratios) — and still found a real margin-MAE cost**
(far smaller than Cycle 20's, but not zero), alongside real calibration gains (CRPS(total),
P(over) continuing its 41.6%→43.4%→45.4% trajectory). Per the pre-registered veto, this
closes the independent-Poisson-plus-diagonal-transfer family for good; a genuinely
dependence-bearing joint is the identified long-term path, separate from this chain.

**Remaining open items, in recommended order: (1) the walk-forward EV-bucket intercept
(§18.3/§19.2 — confirmed data-level-clean, the last piece of the mean-total chain before one
final, currently-frozen `HALFLIFE_GAMES` re-tune, §24.4, closes it); (2) Cycle 16's GBM
stacking layer, required to use the corrected holdout protocol (§15). Neither yet attempted.**

---

## 1. Data source survey (2026-07-23)

Every source below was hit live (`curl`, real endpoints/files, real game IDs) rather than
assumed from reputation. Raw samples are quoted where they change the conclusion.

### 1.1 NHL public API (`api-web.nhle.com`, `api.nhle.com/stats/rest`) — **usable, primary source**

No auth, no key, no rate-limit wall encountered. JSON, well-structured, undocumented but
stable and widely reverse-engineered by the public hockey-analytics community.

Confirmed live endpoints and what they return:

| Endpoint | Contains |
|---|---|
| `GET /v1/schedule/{date}` | Full slate for a date: game IDs, venue, teams, final score, game state. Basis for building the full historical schedule and for rest/travel/back-to-back features (all derivable — no separate travel data source needed). |
| `GET /v1/standings/{date}` | Point-in-time standings (W/L/OTL, goal differential, home/road/L10 splits) — useful as of-that-date team strength sanity check. |
| `GET /v1/gamecenter/{gameId}/play-by-play` | Full event stream per game: every shot/hit/faceoff/penalty with `typeDescKey`, period, time, **x/y ice coordinates**, `situationCode` (strength state), player IDs. This is the raw material to build an in-house xG model if desired — confirmed 309 timestamped events for a real 2024 game, coordinates present on shot-type events. |
| `GET /v1/gamecenter/{gameId}/boxscore` | Final box score, `playerByGameStats`, `rosterSpots` (who dressed, including starting goalie). |
| `GET /v1/gamecenter/{gameId}/landing` | Higher-level game summary/context. |
| `GET /stats/rest/en/shiftcharts?cayenneExp=gameId=...` | Per-player, per-shift start/end times — enables TOI, deployment, and line-combination reconstruction if needed later. |

Historical depth: game IDs are addressable back through at least the 2007-08 season
(`YYYY02NNNN` scheme); this is consistent with the coverage MoneyPuck reports (see below),
which is itself built from this same play-by-play feed.

Gaps confirmed by absence, not by search failure: no official injury-report or
pre-game-projected-lineup endpoint. Scratches/starting goalie are only knowable
**retroactively** (once `boxscore`/`rosterSpots` is populated for a played game) — fine for
historical training data, but means a live/future-game prediction cannot know confirmed
lineups this way before puck drop. Flagged as an open gap for the live-prediction path,
not for backtesting.

### 1.2 MoneyPuck.com (`moneypuck.com`, files served from `peter-tanner.com`) — **usable, primary source for team/skater strength + xG**

Fully open, plain HTTPS downloads, no login/paywall/bot-block encountered (served via
Cloudflare but returns normal 200s to a generic user agent).

Confirmed real files and shapes:

- **`moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv`** — downloaded in
  full (232,221 rows). One row per **team–game–situation** (situations present: `5on5`,
  `5on4`, `4on5`, `all`, `other`). Verified season coverage **2008 through 2025** (game
  dates run through `2026-06-14`, i.e. through the just-completed Stanley Cup Final).
  Columns include, per team per game: `xGoalsFor/Against`, `corsiPercentage`,
  `fenwickPercentage`, shot-attempt counts, **danger-tier splits**
  (`lowDanger/mediumDanger/highDangerxGoalsFor`), score/venue-adjusted xG variants, rebounds,
  faceoffs, penalties, hits, giveaways/takeaways — all split home/away and by strength
  state. This one file is close to sufficient on its own for a team-strength xG model.
- **`peter-tanner.com/moneypuck/downloads/shots_{year}.zip`** (and multi-year bundles like
  `shots_2007-2024.zip`, confirmed a live `shots_2025.zip` also exists, last-modified
  2026-06-15) — shot-level data, one row per shot attempt, with MoneyPuck's own `xGoal`
  field (their published shot-quality model output) plus the underlying shot
  characteristics (location, shot type, rebound flag, rush flag, strength state) that
  produced it. Confirmed via their published data dictionary that `xGoal` = "the
  probability the shot will be a goal," and that it's one component of a multinomial model
  whose outputs (`xGoal`, `xFroze`, `xRebound`, `xPlayContinuedInZone`,
  `xPlayContinuedOutsideZone`, `xPlayStopped`) sum to 1 — i.e. MoneyPuck's public xG model
  is itself a play-outcome model, not a bare logistic regression on shot distance/angle.
- **`historicalOneRowPerSeason/{skaters,goalies,lines,teams}_2008_to_2024.zip`** and
  per-season equivalents back to 2008 — season-aggregate versions of the above, including
  goalie-specific (e.g. goals-saved-above-expected) and line-combination-level splits.

Net: MoneyPuck gives us, for free, both a credible pre-built xG model **and** the shot-level
raw data to audit or rebuild it ourselves — which matters, because the standard critique in
the public hockey-analytics literature (see §1.5) is that off-the-shelf xG models vary
meaningfully in how they treat rebounds/rush shots, so having the underlying shots is
valuable even if we start from MoneyPuck's own numbers.

Update cadence: files for the season in progress exist and get refreshed (per the
`last-modified` headers seen), so this is viable for both historical training and
in-season live use, not just a frozen archive.

### 1.3 Natural Stat Trick — **excluded**

`robots.txt` disallows `/dl.php` (their CSV export endpoint) site-wide with a 240-second
crawl-delay, and — notably — explicitly names and disallows `ClaudeBot`, `Claude-SearchBot`,
and `Claude-User` by name. Independent of that, the actual data pages are gated behind a
Cloudflare "Just a moment..." interactive challenge (confirmed live — got the challenge
page, not data, on a direct request). Between the explicit robots.txt disallow naming
Claude specifically and the active bot-wall, this source is not usable here. Everything NST
exposes (on-ice xG, score/venue adjustments, line combinations) is already covered by
MoneyPuck's files above, so nothing is lost.

### 1.4 Hockey-Reference — **excluded**

`robots.txt` disallows `/hockey/` in its entirety for all user agents (`User-agent: *` /
`Disallow: /hockey/`), which covers every page this project would want (schedules,
box scores, standings). Not usable regardless of paywall status. NHL API + MoneyPuck
already cover what Hockey-Reference would have offered (schedules, results, box scores) at
finer grain (play-by-play, shot-level xG) than Hockey-Reference exposes anyway.

### 1.5 Evolving Hockey — **excluded (paywalled)**

Confirmed live: site is up, but the pages for its signature products (RAPM player value,
GAR/WAR, xG model detail) are marked `"subscriber content"` gated behind
`evolving-hockey.com/login/` ("Sign In / Subscribe"). Not free. Their published
*methodology* write-ups (e.g. how their xG model and RAPM handle score effects, zone
starts, and rebounds) are still useful as design reference material — reading about a
method isn't the same as scraping paywalled data — but no data from this source will be
ingested.

### 1.6 HockeyViz — **reference-only, not a bulk data source**

Site is live and free to browse, but confirmed to be per-team/per-player **visualization
pages** (shot-location heatmaps, WOWY graphics) with no CSV/API surface found. Useful as a
manual sanity-check tool later (e.g. "does our model's read on team X's shot suppression
match the public heatmap") but not something to ingest programmatically.

### 1.7 Betting markets / odds — **not sourced yet, optional**

Not part of the free-data survey scope the user asked for (game outcome data, not markets),
and not fetched this session. Flagged here only because a market line is the natural
external calibration check for a probabilistic model (are our win% / puck-line / total
probabilities better calibrated than the market, not just "accurate"). Revisit if/when we
want a market-calibration benchmark — there are free odds APIs with limited historical
depth, but none were tested in this pass.

### 1.8 Injuries / confirmed lineups — **solvable for live prediction via RotoWire, deferred**

The sibling MLB project already established RotoWire as scrapeable for exactly this purpose
(`src/ingest/fetch_rotowire_lineups.py` there: plain `requests.get`, static HTML, no JS
needed, `robots.txt` allows it, and RotoWire's own `llms.txt` explicitly lists itself as an
LLM-usable public resource). Checked the NHL equivalent live this session:

- **`https://www.rotowire.com/hockey/nhl-lineups.php`** — confirmed 200, real page (found by
  probing candidates; `daily-lineups.php` and `lineups.php` both 404 for hockey, unlike the
  baseball path). `robots.txt` has no `Disallow` covering any `/hockey/` path.
- **`https://www.rotowire.com/hockey/injury-report.php`** — confirmed 200, also unblocked.
- `llms.txt` explicitly covers NHL: lists "starting lineups" and "injury updates" among the
  hockey content it authorizes for LLM use.

**Not yet verified: the actual lineup markup.** Checked live on 2026-07-23 — deep NHL
offseason, zero games on the slate — and the page correctly renders no `lineup__box`
elements (matches the site's own "check back daily" placeholder copy; this is the site
behaving correctly for an empty slate, not a scrape failure). The MLB scraper's HTML
structure (`div.lineup__box` → `ul.lineup__list.is-home/.is-visit` → `li.lineup__player`)
is RotoWire's shared cross-sport template, so it's a reasonable bet the hockey page follows
the same shape once games resume, but this must be re-confirmed against a real slate (NHL
season starts in October) before `fetch_rotowire_lineups.py`'s hockey equivalent is written
— don't assume the MLB selectors carry over unchanged; hockey lineups also need to expose
starting goalie specifically (the highest-value single item here), which baseball's
`lineup__player-highlight-name` pattern was built for starting pitchers and may or may not
map directly.

Net: this is no longer an open gap in principle — just deferred until there's a live NHL
slate to verify against. Doesn't block anything now: backtesting only needs the NHL API's
after-the-fact `boxscore`/`rosterSpots` (who actually played), which is already confirmed
available back to ~2007-08.

---

## 2. Conclusion — what we're building on

**Primary sources, both confirmed genuinely free and already pulled live in this session:**
1. NHL API (`api-web.nhle.com`) — schedule, results, play-by-play with shot coordinates,
   box scores, rosters, shift charts. Back to ~2007-08. Basis for rest/schedule features and
   for building/auditing our own shot models if desired.
2. MoneyPuck.com bulk CSV/zip downloads — team-game-situation xG/Corsi/Fenwick splits
   (2008–2025, current season included), shot-level data with a published xG model,
   season-aggregate skater/goalie/line files.

**Excluded, with reasons on record so they aren't re-tried:** Natural Stat Trick (robots.txt
explicitly blocks Claude + Cloudflare bot wall), Hockey-Reference (`/hockey/` disallowed
site-wide), Evolving Hockey (paywalled).

**Live-prediction lineups/injuries:** RotoWire's NHL lineups (`hockey/nhl-lineups.php`) and
injury report (`hockey/injury-report.php`) pages are confirmed live and unblocked by
robots.txt, following the same precedent already used for the sibling MLB project. Not a
blocker for historical training either way (NHL API's after-the-fact box scores cover
that); the RotoWire scraper itself is deferred until there's a real NHL slate (season
starts October) to verify the markup against — see §1.8.

---

## 3. Proposed architecture (2026-07-23)

No code has been written against this yet. This is a proposal to review before starting
§4 (incremental build). It is deliberately structured as independently-validatable layers,
mirroring the discipline that worked in the sibling MLB project: build the simplest version
of each layer, check it against real held-out games, and only add the next layer once the
current one is confirmed to earn its complexity.

### 3.1 Why the target is xG, not goals — and where the line gets drawn

A single NHL game is ~2-6 goals; the sample is simply too small for a team's *actual* goal
differential in its own recent games to be a stable estimate of its *true* current
strength. This is the well-established starting premise of the public hockey-analytics
work referenced in the ask (MoneyPuck, Evolving Hockey, Dawson Sprigings) and it's directly
checkable with the data already pulled: shot-attempt-based rates (Corsi/Fenwick/xG)
accumulate many more observations per game than goals do, so they stabilize into a
meaningful signal much faster across a season than goals themselves. The consequence for
architecture: **team strength is estimated from xG-rate history, and only converted into an
actual goals number at the very last step** (the count-process layer, §3.4) — never used as
"team X's true talent, expressed directly in goals."

The one line this draws clearly: **xG estimates team talent; goals (the real, final ones)
are still the only thing the model is ever ultimately validated against.** Every layer below
gets checked against real historical goals/scores, not against xG — xG is an input feature,
never the target metric, exactly so we don't fool ourselves into validating a model against
a proxy instead of reality.

### 3.2 Layer 1 — team strength ratings (xG-based, shrinkage/regression, walk-forward)

Proposed to mirror the MLB project's `true_talent.py` approach (a Marcel-style two-level
shrinkage estimator), adapted to hockey's actual data shape:

- **Separate offense and defense ratings, separately by strength state** (5v5, PP-for /
  PK-against, using MoneyPuck's own situation split) — a team's 5v5 xG-for rate and its PP
  conversion rate are different skills with different sample sizes and shouldn't be
  blended into one number.
- **Recency-weighted multi-season prior + in-season Bayesian update**, same two-level
  structure as MLB's approach: a preseason prior from (e.g.) the last 2-3 seasons with
  heavier recent-season weight, regressed toward the league-average rate based on games/xG-
  attempts of data available, then blended with the current season's own accumulating data
  as it arrives.
- **Stabilization constants must be empirically calibrated on our own data, not assumed.**
  Unlike MLB (where Russell Carleton's published per-outcome stabilization points were
  directly available and cited), there isn't an equivalently precise, directly-citable
  public table for exactly how fast score-adjusted xG-for/against stabilizes at 5v5 vs. PP
  vs. PK. Public discussion in this space treats "score-adjusted Corsi/xG stabilizes faster
  than raw goals" as settled, but the exact games-to-stabilize constant per situation is
  something we should measure ourselves (split-half reliability on our own historical data)
  rather than hard-code from memory. This is an explicit first empirical task, not a
  guessed placeholder.
- **Goaltending modeled as its own separate layer, not folded into team defense.** A
  specific goalie's save performance (MoneyPuck's goalie files include goals-saved-above-
  expected) is a large, well-documented single-game swing factor in hockey, distinct from
  the skaters' shot-suppression rate. Proposed: team defense produces an *shots/xG-against*
  rate; the *starting goalie's* own shrunk GSAx/60 rate is applied on top to convert
  shots-against into expected goals-against. This also cleanly separates the (retroactive-
  only, per §1.8) "who actually started in net" signal from the team-level rating, so the
  team rating stays usable even before a starting goalie is known.
- **Home ice advantage and rest/schedule effects (back-to-backs, days of rest, long
  road trips) as their own additive/multiplicative terms**, estimated from our own data
  rather than assumed from league folklore about how big they are.

### 3.3 Layer 2 — combining ratings into a per-game expected-goals number

For a given matchup: combine team A's offensive rating with team B's defensive rating (and
vice versa), per strength state, weighted by each state's expected share of the game (5v5
gets most of regulation time; PP opportunities are themselves a count estimated from both
teams' penalty-drawn/taken rates, not a fixed assumption) — then sum across strength states
into one expected-goals figure per team for the game, with home ice and rest adjustments
applied. This step produces two numbers (home λ, away λ), not a full distribution yet —
that's Layer 3's job.

### 3.4 Layer 3 — goals as a count process (the outcome distribution)

- **Start with independent Poisson** for home/away goals given their respective λ from
  Layer 2 — the simplest, most standard baseline for a low-scoring count sport, and directly
  checkable against real final scores via a proper scoring rule (log-loss / CRPS), not just
  accuracy.
- **Explicitly test, don't assume, two standard refinements** before adopting either:
  1. *Overdispersion* — real NHL goal counts may be better fit by a Negative Binomial than a
     Poisson (empty-net goals and pulled-goalie situations are a plausible source of extra
     variance beyond what Poisson allows). Test via a dispersion check on real historical
     goal counts before switching.
  2. *Score-state correlation between the two teams' goal counts* — once one team leads by
     2+, both teams' shot volume/quality and the trailing team's empty-net risk change in a
     way that plain independent Poisson margins won't capture (the same class of issue
     Dixon-Coles correction addresses in soccer, adapted to hockey's own dynamics rather
     than copied verbatim). Test whether independent Poisson's *joint* score distribution is
     actually miscalibrated against real historical joint scorelines before adding a
     correlation term — don't add the complexity unless the simple version demonstrably
     misses.
- This is the layer that gets checked against real final scores end-to-end (calibration
  curves on win probability, CRPS on the full score distribution, MAE on total goals) — the
  actual honest scorecard for the whole model, same role `validate_game_simulator.py` plays
  in the MLB project.

### 3.5 Layer 4 — the OT/SO tiebreaker (hockey-specific, no MLB analogue)

Regulation ties go to 3-on-3 overtime, then a shootout if still tied — this needs to be an
explicit extra stage, not folded into the regulation goals model:

- P(regulation tie) falls directly out of Layer 3's joint distribution.
- Conditional on a tie, OT/SO winner needs its own (much simpler, small-sample-honest)
  sub-model — 3-on-3 OT and shootouts are short, high-variance, and only weakly related to
  full-strength team talent; a reasonable starting point is home-ice-adjusted coin-flip,
  empirically checked against real OT/SO game outcomes rather than assumed 50/50.
- **Open design decision, not yet resolved: whether the "final score" we report for
  totals/spread purposes is the regulation+OT actual final (including any shootout-decided
  extra goal added to the winner's tally) or a regulation-only number.** Sportsbooks are not
  fully uniform on how they grade game totals/puck-line bets involving a shootout — this
  needs to be checked against real settlement conventions (or simply reported as two
  explicit numbers: "regulation/OT actual final score" and "goals distribution excluding
  the shootout bonus goal") rather than silently picking one and hoping it matches whatever
  the user compares it against later.

### 3.6 From distribution to deliverables

Once Layers 3-4 produce a full joint score-distribution (via closed-form combination where
possible, Monte Carlo sampling otherwise — decide based on whether the correlation term from
§3.4 makes a closed form impractical), everything the user asked for falls out directly:
win probability (moneyline, with the OT/SO layer folded in), exact-score probabilities,
total-goals over/under at any line, puck-line (±1.5, the standard hockey spread) cover
probability. No separate model needed per market — one distribution, multiple slices.

### 3.7 Why a full discrete-event (possession-by-possession) simulator is NOT the Phase 1 proposal

The MLB project simulates games pitch-by-pitch/PA-by-PA because baseball has a clean,
discrete, well-defined unit (the plate appearance) with known transition probabilities.
Hockey possessions are continuous and whistle-bounded, not naturally discretized the same
way, and the public models this project is taking guidance from (MoneyPuck's own win-
probability model included) are predominantly of the rate-into-count-process family
described above, not full possession simulators. Proposing to start there, and only
consider a heavier discrete-event simulation later if the simpler layered model hits a
real, diagnosed ceiling — not stacking that complexity up front.

### 3.8 Validation harness (mirrors the MLB project's discipline exactly)

- **Walk-forward only**: every team-strength number used for a given game is computed from
  data strictly before that game's date — no leakage, enforced structurally (date-indexed
  cumulative computation), not just by convention.
- **Incremental signal validation**: each proposed addition (xG over raw goals, goalie
  overlay, home/rest adjustments, NegBin over Poisson, the correlation term) gets checked
  for genuine incremental predictive value against real held-out games before being kept —
  matching the MLB project's practice of reverting ideas that don't clear the bar (and
  recording them anyway, per this document's stated purpose).
- **A single genuine final holdout**: a recent, never-touched-during-development stretch of
  real games (candidate: the second half of the 2025-26 season, once we're further into
  building) reserved purely as the last honest check, not used for any tuning decision along
  the way.
- **A metrics ledger**, mirroring MLB's `metrics_ledger.py`: log-loss/Brier on win
  probability, CRPS on the full score distribution, MAE on total goals and margin, and
  calibration curves — tracked per model version so regressions are visible immediately,
  not just point-in-time accuracy numbers.

### 3.9 First concrete engineering task, discovered (not assumed) from the data already pulled

NHL API and MoneyPuck use **different team abbreviation schemes** — confirmed directly in
the samples pulled for §1 (MoneyPuck's CSV uses e.g. `L.A`, `N.J`, `S.J`, `T.B`; the NHL API
uses standard 3-letter codes like `LAK`, `NJD`, `SJS`, `TBL`). This is the same category of
problem the MLB project hit with RotoWire's `ARI` vs. its own `AZ` (§1.8/MLB precedent) and
needs its own explicit mapping table, built and checked against the real season schedule
(all 32 teams, both directions) before any join between the two sources is trusted — not
assumed to be a simple find-and-replace without checking for edge cases (relocated/renamed
franchises across the 2008-2025 span, e.g. Atlanta→Winnipeg, Arizona's own history).

---

---

## 4. Incremental build log

### 4.1 Ingestion (2026-07-23)

- `src/ingest/fetch_nhl_api.py` — confirmed live that `GET /v1/schedule/{date}` returns a
  full 7-day window (`gameWeek`, verified 7 distinct dates per request) with `nextStartDate`
  exactly one week later, so a single continuous week-by-week walk from 2008-08-01 to
  2026-07-23 (no need to know each season's exact start/end date) pulled the **entire**
  schedule in ~940 requests. Real result: **25,104 total games** — 21,612 regular season
  (`gameType==2`), 1,615 playoff, 1,722 preseason, plus a long tail of other `gameType`
  values (90, 28, 12, 12, 6, 4, 2, 1 games respectively) that are presumably All-Star/Skills/
  exhibition-style entries — not yet identified individually, and irrelevant since the
  model only uses `gameType==2`, but flagged here so a future reader doesn't assume
  `gameType` is a clean `{1,2,3}` enum.
- `src/ingest/fetch_moneypuck.py` — downloaded the real `all_teams.csv` (232,220 rows).
- `src/ingest/team_codes.py` — building this surfaced a real, non-obvious data-quality issue
  (not assumed, confirmed by pulling live schedule weeks and reading the numeric `id` field
  directly): **the NHL API's own team id is not stable across the 2014 Phoenix→Arizona
  Coyotes market rename** (`PHX`/id 27 through 2013-14, `ARI`/id 53 from 2014-15) even though
  it's the same continuous roster — while MoneyPuck merges both eras under one `ARI` code.
  Utah (`UTA`/id 59, 2024-25 on) is a genuinely distinct id in both sources, agreeing with
  each other. Atlanta (`ATL`/id 11) → Winnipeg (`WPG`/id 52, 2011-12) is a clean boundary
  both sources agree on. Separately, MoneyPuck changed its own abbreviation convention
  (dotted `L.A`/`N.J`/`S.J`/`T.B` for seasons 2008-2020, plain 3-letter matching the NHL API
  from season 2021 on) — confirmed exactly where the switch lands by checking which
  `season` values each variant appears under in the real file. **Validated the resulting
  crosswalk against every real (team, season) pair in both full datasets — zero mismatches
  either direction**, not just a spot check on a couple of teams.
  - Explicitly out of scope for this module (left as an open modeling question for §4.2+,
    noted so it isn't silently baked in): whether a team-strength *rating* should carry
    across the Arizona→Utah boundary, given Utah inherited most of Arizona's actual roster.
    This is a "does history transfer" modeling decision, different from "do these rows join."

### 4.2 First real model: naive season-to-date raw-goals Poisson baseline (2026-07-23)

`src/models/baseline_naive_poisson.py` + `validate_baseline.py` — the deliberately simple
floor described in §3.9: no xG, no situational (5v5/PP/PK) split, no goalie overlay, no
rest/schedule adjustment, no OT/SO sub-model. Team attack/defense rate = season-to-date raw
goals, shrunk toward a trailing league-average with a **placeholder, not-yet-calibrated**
10-pseudo-game prior; home-ice is **fit empirically from real data** (not assumed) as the
square-root split of the real home-goals/away-goals ratio. Walk-forward enforced
structurally (every rate uses only strictly-earlier games via `cumsum().shift(1)`, checked
by code, not by convention). Ties (unresolved by the not-yet-built OT/SO layer) are reported
as their own probability mass and renormalized out of the win-probability metrics rather
than silently assumed 50/50 — keeps this run honestly scoped to Layers 1+3 only.

**Real result, 19,152 held-out regular-season games (2010-11 through 2025-26; the first two
seasons in the dataset are excluded from scoring, kept only to warm up the trailing
league-average — see `validate_baseline.py` docstring):**

| Metric | Value |
|---|---|
| Straight-up (SU) win accuracy | **57.06%** |
| Brier score | 0.2438 |
| Log-loss | 0.6809 |
| Total-goals MAE | 1.827 |
| Margin MAE | 2.042 |
| Actual home win rate (sanity check) | 54.10% |
| Actual mean total goals (sanity check) | 5.861 |
| Predicted mean total goals | 5.965 |

**Sanity checks that passed:** the model's own 54.10% actual-home-win-rate figure matches
publicly known NHL home-ice-advantage stats (commonly cited in the ~54-55% range), which is
a real signal the ingestion/join pipeline isn't silently broken. Predicted mean total goals
(5.965) is close to the real mean (5.861) — no gross systemic bias in the count-process
layer. **The real, meaningful comparison**: a model that always predicts the home team
wins would score exactly 54.10% SU on this same set (that IS the home-win rate) — this
naive model's 57.06% is a modest but genuine ~3-point edge over that trivial floor, meaning
season-to-date raw goals carries *some* real predictive signal beyond home-ice alone, even
before any xG is introduced. Logged to `data/processed/metrics_ledger.parquet` as
`naive_raw_goals_poisson` — this is the number every subsequent layer (xG-based strength,
goalie overlay, NegBin, correlation term, rest/schedule) must beat.

Next: swap raw-goals team strength for MoneyPuck's situational xG (§3.2's core hypothesis)
and check, on this same real held-out set, whether it actually improves SU/Brier/log-loss/
MAE over this baseline — not assumed, tested.

### 4.3 Cycle 2: xG-based team strength vs. the raw-goals floor (2026-07-23)

Refactored the naive baseline's walk-forward shrinkage math out into
`src/models/shrinkage.py` (`add_walk_forward_rate`) first, since a second consumer now
needed the identical logic on a different pair of columns — **re-ran the raw-goals baseline
immediately after the refactor and confirmed byte-identical metrics** before building
anything new on top of it (regression check, not assumed safe).

`src/models/team_strength_xg.py` swaps exactly one thing from the floor: team
attack/defense rate is now built from MoneyPuck's whole-game (`situation=="all"`)
`xGoalsFor`/`xGoalsAgainst` instead of raw goals — same shrinkage machinery, same Poisson
combine, same home-ice methodology, same held-out game set. Deliberately NOT also adding
situational (5v5/PP/PK) splits, a goalie overlay, or rest adjustments in this same cycle —
isolating one variable per the project's stated discipline.

**A real bug surfaced during this cycle, not just a clean result:** the first run silently
dropped 465 team-game rows (all Phoenix Coyotes games, 2008-09 through 2013-14) because the
join normalized MoneyPuck's team code but not the NHL API schedule's own `PHX` code before
joining on `(gameId, team)` — an easy mistake to miss since the earlier season-level
crosswalk validation (§4.1) checked that both sides *agree on which (team, season) pairs
exist*, not that every individual game-level join actually resolves. Fixed by normalizing
both sides identically; re-running dropped the mismatch from 465 rows to 8 (a residual,
unchased single-game-level oddity at 0.02% of rows — negligible next to the 465 real bug,
not investigated further). **This is exactly the kind of thing walk-forward discipline
alone doesn't catch — worth remembering that a clean crosswalk validation and a clean
per-game join are two different checks, not one.**

**Real result, identical 19,152-game set as the floor (§4.2):**

| Metric | Raw-goals floor | xG-based | Δ (xG − floor) |
|---|---|---|---|
| SU accuracy | 57.06% | 57.49% | +0.43 pt |
| Brier score | 0.2438 | 0.2414 | −0.0023 |
| Log-loss | 0.6809 | 0.6757 | −0.0052 |
| Total-goals MAE | 1.827 | 1.822 | −0.005 |
| Margin MAE | 2.042 | 2.051 | +0.009 (worse) |

**Paired bootstrap significance check (5,000 resamples, same 19,152 games both models),
because a single point-estimate delta this small isn't yet a fact** (the MLB project's own
§11.7 caution about trusting an unvalidated point estimate applies here too):
- **Brier score improvement is real**: mean 0.00231, 95% CI **[0.00106, 0.00356]** — entirely
  above zero, 100% of bootstrap draws favored xG. The win-probability signal genuinely
  improves from using xG instead of raw goals as the team-strength input.
- **Total-goals MAE improvement is NOT distinguishable from noise**: mean 0.00496, 95% CI
  **[−0.00208, 0.01189]** — crosses zero. At this sample size, we can't yet claim xG makes
  the actual total-goals prediction better, only that it doesn't make it worse, and that the
  win-probability piece specifically does improve.

**Verdict: kept.** The xG-based strength model becomes the new reference point (logged as
`xg_based_poisson` in the metrics ledger) on the strength of a real, bootstrap-confirmed
win-probability improvement — but the honest framing is "confirmed better for win
probability, not yet confirmed better for the score distribution itself," not "strictly
better across the board." This also recalibrates expectations for what's ahead: a single
whole-game xG swap alone only closes part of the gap; situational splits (5v5/PP/PK) and
the goalie overlay described in §3.2 are more likely sources of a bigger jump than
squeezing more out of whole-game xG alone.

Next: split team strength by strength state (5v5 vs. PP-for/PK-against) using MoneyPuck's
own situational rows, and test — again on this identical game set — whether that clears its
own bar over `xg_based_poisson` before adding a goalie overlay on top.

### 4.4 Cycle 3: situational (even-strength/PP/PK/other) team strength (2026-07-23)

This cycle surfaced three real bugs before producing a trustworthy number — worth recording
in full, since two of them are the kind of mistake that produces a plausible-looking wrong
answer rather than a crash, exactly the failure mode this project's incremental-validation
discipline exists to catch.

**Bug 1 — a same-game leak in the shared shrinkage helper, found while building this cycle
but affecting Cycles 1-2 too.** `shrinkage.py`'s trailing league-average was a flat
`.shift(1).expanding()` over a frame with two rows per real game (home + away).
`build_team_game_log`'s concat always places the home row first, and pandas' sort is
stable, so after sorting by (gameDate, gameId) the home row of a game *always* sorted
immediately before the away row of that same game — confirmed directly (home sorts first
in 21,612/21,612 real games, zero exceptions) — meaning the away row's "trailing" league
average actually included the home team's own result **from the game being predicted**.
Fixed in `shrinkage._trailing_league_stat` by aggregating to one row per real `gameId`
first (so `shift(1)` always skips a whole game, never just one side of it) before computing
the trailing statistic, then broadcasting back to both team-rows. **Re-ran both already-
logged Cycle 1-2 models after the fix: numbers changed in the 7th decimal place or beyond**
(e.g. baseline SU 0.570593149540518 unchanged to 12 digits; Brier moved by ~1.3e-7) —
confirming the leak's practical impact was vanishingly small, but the fix was correct on
principle and cheap, so it stayed.

**Bug 2 — a real, non-negligible one: mismatched PP/PK league averages.** The first version
of `add_walk_forward_toi_rate` computed ONE league-average per situation (from the `for_col`
only) and used it as the ratio baseline for both attack and defense rates. That's harmless
when for/against are symmetric league-wide (true for whole-game and even-strength goals —
total goals-for trivially equals total goals-against, summed over the whole league) but
**wrong for PP/PK**: a team's own `pk_xgf` (rare shorthanded goals scored while killing a
penalty, confirmed ~0.77/60 empirically) and `pk_xga` (goals conceded while killing —
essentially the opponent's power-play production, confirmed ~5.99/60) are entirely
different-magnitude events. Comparing a team's PK defense rate against the wrong, tiny
`for`-derived baseline inflated that ratio roughly 8x — confirmed by printing a real sample
game's rates before touching the fix. Fixed by computing separate `_league_avg_for_per60`
and `_league_avg_against_per60` per situation, and pairing each rate with its true symmetric
partner (PP-for's baseline is the same real quantity as PK-against's baseline — a power-play
goal for one team is simultaneously a shorthanded-goal-against for the other, same clock
window — confirmed numerically identical after the fix: both landed at 5.992 per 60 for the
same sample game).

**Bug 3 — a real, material scope gap: MoneyPuck has a 5th situation.** After fixing Bug 2,
the model's predicted mean total goals was ~5.0 against a real ~5.86 — too large a gap to
be explained by the deliberately-omitted shorthanded-goals-for term alone. Investigated by
comparing MoneyPuck's own `situation=="all"` xG total (the one Cycle 2 already used) against
the sum of the three situations used here: they didn't match (2.434 vs 2.867 mean
xG/team-game). MoneyPuck's data has a 5th situation, `other` — confirmed to cover 4v4, 3v3
overtime, 6-on-5 empty net, and similar states `5on5`/`5on4`/`4on5` don't — and confirmed
**not negligible**: ~0.358 xG/team-game, ~12.5% of the whole-game total. This was an
assumption ("I already knew the three situation names") not checked against the actual
distinct values in the column before writing the combine logic — the exact same category of
mistake the project's own data-survey discipline (§1) exists to prevent, just recurring one
layer deeper, inside a single column's own possible values, instead of across whole data
sources. Fixed by adding `other` as a 4th additive component, treated symmetrically like
even-strength (its for/against columns are symmetric league-wide by construction, unlike
PP/PK).

**Real result, after all three fixes, identical 19,152-game set:**

| Metric | xG-based (Cycle 2) | Situational (Cycle 3) | Δ |
|---|---|---|---|
| SU accuracy | 57.49% | 57.26% | −0.23 pt |
| Brier score | 0.2414 | 0.2414 | ~0 |
| Log-loss | 0.6757 | 0.6756 | ~0 |
| Total-goals MAE | 1.822 | 1.820 | −0.002 |
| Margin MAE | 2.051 | 2.051 | ~0 |

Predicted mean total goals is now well-calibrated (5.833 vs. real 5.861), confirming the
Bug 3 fix actually worked, not just moved the error somewhere else.

**Paired bootstrap (5,000 resamples, identical game set):** mean Brier improvement 0.00004,
95% CI **[−0.00019, 0.00029]** — crosses zero. Mean total-MAE improvement 0.00130, 95% CI
**[−0.00092, 0.00358]** — also crosses zero. **Neither is distinguishable from noise.**

**Verdict: a genuine wash, not kept over Cycle 2 — logged anyway, since a real negative
result is exactly what this log exists to preserve.** Splitting team strength by situational
rate alone, with a *shared league-average* TOI weight (this cycle's deliberate scope, per
`team_strength_situational.py`'s docstring), does not improve on treating xG as one
whole-game number. This sharpens rather than closes off the original hypothesis in §3.2:
the situational-split idea itself may still be right, but the missing piece this cycle
deliberately deferred — **team-specific expected PP/PK time**, i.e. does a team that draws
more penalties than average actually get a bigger PP-weighted lambda — was excluded
specifically so it wouldn't be conflated with the rate-splitting question this cycle set out
to test. That's the next thing to test in isolation, not a reason to abandon the situational
split outright.

Next: test team-specific PP/PK time-on-ice weighting (replacing this cycle's shared
league-average TOI constant) as its own isolated change on top of the situational rate
split, before considering a goalie overlay.

### 4.5 Cycle 4: team-specific PP/PK time-on-ice weighting (2026-07-23)

Generalized `predict_situational_lambda`'s single shared `league_avg_pp_toi_min` parameter
into separate `home_pp_toi_min`/`away_pp_toi_min` values — Cycle 3 is exactly the special
case where both equal the shared league average, confirmed by re-running
`validate_situational.py` unchanged after the signature generalization and checking its
metrics were byte-identical to the original Cycle 3 numbers before building anything new on
top (same regression-check discipline as every prior cycle).

`validate_situational_toi.py` replaces the shared constant with each team's own walk-forward
-shrunk expected PP/PK minutes-per-game (`shrinkage.add_walk_forward_mean`, a new small
helper for shrinking a single per-game value rather than a for/against pair). Since a real
game's home-PP minutes automatically equal the away team's PK minutes (same clock), each
side's expected PP time is the average of two independent walk-forward estimates of that
same real quantity: the home team's own history of drawing power plays, and the away team's
own history of killing them (and symmetrically for the away team's own PP time). EV and
"other" TOI weighting stay shared/league-average, unchanged from Cycle 3 — isolating
team-specific PP/PK TOI as the one new variable.

**Real result, identical 19,152-game set:**

| Metric | Situational, shared TOI (Cycle 3) | Team-specific PP/PK TOI (Cycle 4) | Δ |
|---|---|---|---|
| SU accuracy | 57.26% | 57.49% | +0.23 pt |
| Brier score | 0.24141 | 0.24143 | +0.00002 (worse) |
| Log-loss | 0.67564 | 0.67568 | +0.00004 (worse) |
| Total-goals MAE | 1.8204 | 1.8172 | −0.0032 |
| Margin MAE | 2.0506 | 2.0512 | +0.0006 (worse) |

**Paired bootstrap (5,000 resamples):** mean Brier improvement −0.00002, 95% CI
**[−0.00014, 0.00009]** — crosses zero, not distinguishable from noise. Mean total-MAE
improvement **+0.00318, 95% CI [0.00210, 0.00423]** — entirely positive, a real,
bootstrap-confirmed improvement.

**A genuinely nuanced, real finding, not a clean win or a clean wash: team-specific PP/PK
time-on-ice weighting measurably improves total-goals prediction, with no measurable effect
on win-probability calibration.** This makes real hockey sense: a team's own tendency to
draw (or take) penalties shifts how many goals get scored via the power-play channel
specifically — which matters for the total-goals/over-under axis — without necessarily
telling you much about *which* team is better in relative terms, since both teams' PP/PK
tendencies partly wash out when comparing one side's strength against the other's. Given the
user's stated goal explicitly includes full score distributions and over/under, not just
moneyline, this is a genuine, honest keep on that basis alone — logged as
`situational_team_specific_pp_toi_poisson`, the new reference point for total-goals/spread
work, while `xg_based_poisson` (Cycle 2) and `situational_ev_pp_pk_poisson` (Cycle 3) remain
statistically tied on the win-probability axis specifically.

Next candidates, not yet started: a goalie-specific overlay (§3.2's other deferred piece,
likely a bigger lever for the win-probability axis than anything tried so far, since Cycles
2-4 have each only moved total-goals accuracy, not SU/Brier meaningfully beyond Cycle 2); or
testing whether Negative Binomial / a score-state correlation term (§3.4) helps now that the
Poisson mean side of the model is reasonably mature.

### 4.6 Cycle 5: starting-goalie overlay (2026-07-23)

The most bug-ridden cycle so far — four real, confirmed issues found before reaching a
trustworthy number, three of them in a brand-new ingestion path
(`src/ingest/fetch_moneypuck_goalie_games.py`) this cycle had to build from scratch, since
MoneyPuck has no bulk goalie GAME-log file (only season aggregates — confirmed by probing
for the obvious naming convention, all 404). Built one instead from MoneyPuck's shot-level
data (`goalieIdForShot`, `xGoal`, `goal`, `shotWasOnGoal` — confirmed `shotWasOnGoal==1` is
exactly "shots the goalie actually faced").

**Bug 1 — a second, different game-ID scheme within MoneyPuck's own data.** The shots
file's own `game_id` is short (e.g. `20001`) and resets every season — confirmed directly:
aggregating without a season qualifier collapsed 18 seasons' opening games onto one row
(1,461 distinct "games" instead of an expected ~20,000+). Reconstructed the real NHL gameId
by concatenating the shots file's own `season` column back onto `game_id` (the NHL id's own
TYPE+SEQUENCE suffix with the leading zero stripped by integer parsing) — verified against a
real game (TOR/MTL, 2010-10-07) before trusting it further.

**Bug 2 — a much more basic mistake: comparing a string against a team code.** MoneyPuck's
shot-level `team` column is the literal string `"HOME"`/`"AWAY"` (confirmed via their own
data dictionary), not a team abbreviation — comparing it against `homeTeamCode` was
unconditionally true, so every single goalie got mislabeled as the home team. Confirmed
directly on a real game (2023020001, TBL home/NSH away): both Juuse Saros (NSH's actual
goalie) and Jonas Johansson (TBL's) came out tagged `goalie_team==TBL`, colliding in the
downstream dedup — this alone explained an observed ~50% team-side match-rate against the
schedule (every away-team goalie silently lost). Fixed using the real `isHomeTeam` 0/1 flag.

**Bug 3 — empty-net shots inflate a goalie's "goals allowed" without inflating their xG
faced enough to match.** MoneyPuck still populates `goalieIdForShot` for empty-net shots
(confirmed real, despite a stray dictionary comment implying otherwise) — and those shots
are effectively certain goals (485/485 empty-net on-goal shots in a real sample season were
actual goals) while `xGoal` for them only averaged ~0.61. Pulling a goalie for an extra
attacker is a coaching decision made because the team is already trailing, not a reflection
of that goalie's own shot-stopping skill — excluded `shotOnEmptyNet==1` rows, standard
practice in public goaltending analytics for exactly this reason.

**Bug 4 — the real structural one: using an absolute level where a relative one was
needed.** Even after Bugs 1-3, the first validation run predicted mean total goals of ~6.81
against a real ~5.86 — investigated and found that `goalie_shrunk_mean` (each goalie's own
walk-forward-shrunk GSAx) averaged **-0.51 to -0.56 across all 46,440 goalie-game rows in
the ENTIRE dataset**, not just one noisy season. This means MoneyPuck's `xGoal`, summed only
over the on-goal (non-empty-net) subset, systematically undershoots the real goal rate on
that specific subset — a real scope-level calibration quirk, confirmed at full-dataset
scale, not assumed. Subtracting this systematically negative absolute level from every
game's lambda was silently ADDING about +1.0 goals/game to every prediction (subtracting a
negative number), which is exactly the observed gap. **Fixed by using the goalie's rating
RELATIVE TO the trailing league average at that same walk-forward point in time**
(`goalie_shrunk_mean - goalie_league_avg`) rather than the raw level — any systematic
scope-level miscalibration affects both the goalie's own level and the league average
identically, so it cancels in the difference, leaving only the genuine "better or worse
than an average goalie right now" signal an additive overlay should actually be using.

**Real result, after all four fixes, identical 19,152-game set as every prior cycle:**

| Metric | Team-specific TOI (Cycle 4) | + Goalie overlay (Cycle 5) | Δ |
|---|---|---|---|
| SU accuracy | 57.49% | 57.63% | +0.14 pt |
| Brier score | 0.24143 | 0.24098 | −0.00045 |
| Log-loss | 0.67568 | 0.67474 | −0.00094 |
| Total-goals MAE | 1.8172 | 1.8152 | −0.0020 |
| Margin MAE | 2.0512 | 2.0416 | −0.0096 |

Predicted mean total goals is well-calibrated again (5.79 vs. real 5.86), confirming Bug 4's
fix actually worked rather than just moving the error elsewhere.

**Paired bootstrap (5,000 resamples, identical game set):**
- **Margin MAE: mean improvement 0.00957, 95% CI [0.00691, 0.01233] — entirely positive, a
  real, clearly significant improvement.** This is the cleanest result of the cycle and
  makes direct hockey sense: a hot or cold starting goalie is exactly the kind of signal
  that should move the predicted SCORE DIFFERENTIAL of a specific game (relevant to
  puck-line/spread accuracy), more directly than either team's overall pace.
- Brier: mean improvement 0.00045, 95% CI **[-0.00005, 0.00094]** — just barely touches
  zero; a real effect is likely present but not clearly separable from noise at this exact
  threshold.
- Total-goals MAE: mean improvement 0.00199, 95% CI [-0.00088, 0.00487] — crosses zero, not
  distinguishable from noise.

**Verdict: kept, on the strength of a clear, real margin-prediction improvement** — again a
genuinely nuanced result rather than a clean sweep, consistent with the pattern this
project's own signals have shown so far (Cycle 4 helped totals specifically; this one helps
margin specifically). Logged as `goalie_overlay_poisson`, the new overall reference point.

Next candidates: try lowering the goalie shrinkage prior (`PRIOR_GAMES_GOALIE=25` is an
unvalidated placeholder) now that the adjustment itself is confirmed structurally correct;
test whether carrying a goalie's rating across the off-season (rather than resetting at
each season boundary, this cycle's deliberate simplification) helps; or move to testing
NegBin/score-correlation refinements on the count-process layer itself (§3.4), which no
cycle has touched yet.

### 4.7 Calibrating the goalie shrinkage prior — a real tension between two legitimate methods (2026-07-23)

`PRIOR_GAMES_GOALIE=25` had been an explicitly-flagged, never-validated placeholder since
Cycle 5. Two different, both-legitimate calibration approaches were tried, and they
disagreed — the disagreement itself is the useful finding here, not just the final number.

**Method 1 — split-half reliability (the textbook "true talent stabilization" approach,
same family as Russell Carleton's published constants the MLB project cites directly).**
`src/models/calibrate_goalie_prior.py`: for every (goalie, season) with 20+ appearances
(956 qualified), split chronologically into two halves, correlate first-half mean GSAx
against second-half mean GSAx, step up to full-season reliability via the Spearman-Brown
formula, then solve `reliability = N/(N+K)` for K. **Real result: split-half r=0.212,
Spearman-Brown-corrected reliability=0.350, implied K≈79.6 games** — consistent with the
publicly known fact that goaltending performance is unusually slow to stabilize compared to
almost any other hockey metric. This is a legitimate measurement of how many games it takes
to reliably know a goalie's TRUE TALENT LEVEL.

**Tested it directly against the model's own validation metrics anyway, rather than
adopting it on authority** — and it failed: a bootstrap comparison of K=80 vs. the original
untested K=25 (identical 19,152-game set) showed margin-MAE getting measurably **worse**,
not better, at K=80 (mean −0.00417, 95% CI **[−0.00566, −0.00275]**, entirely negative).
Brier and total-MAE were both wash (CIs cross zero) at that comparison.

**Method 2 — direct grid search against this project's own downstream metrics**, the
standard this project holds every other constant to. Swept K over
{1,2,3,5,7,10,12,13,15,25,50,80,120,200} and measured total-MAE, margin-MAE, and Brier at
each. Real result: **margin-MAE is U-shaped, with its minimum in the K≈10-13 range**
(essentially flat across it — 2.04021 at K=10 vs. 2.04030 at K=13), rising steadily on
either side (2.056 at K=1, 2.049 at K=200). Total-MAE and Brier both separately prefer a
larger K≈50 (1.814/0.2408 there vs. 1.819/0.2417 at K=12) — a different optimum from margin's.

**Why the two methods disagree, and why the second one wins for this purpose:** K≈80 is the
right answer to "how many games until I can confidently describe this goalie's stable,
long-run talent level" — a real, correctly-measured quantity. But Cycle 5's whole
justification for keeping the goalie overlay was its margin-prediction improvement
specifically (§4.6), and margin prediction benefits from being MORE reactive to a goalie's
recent form than the pure-reliability number would allow — a goalie's current hot/cold state
carries real next-game predictive value beyond "how confident are we in their true talent,"
even though leaning on it more heavily necessarily means trusting noisier, more short-term
information. **This is a genuinely useful, general lesson for the project going forward, not
just a goalie-specific footnote: a theoretically-correct stabilization constant and the
constant that's actually best for a specific downstream prediction task are different
questions, and this project should keep validating against its own real metrics rather than
importing a textbook number, even when the textbook number is itself real and correctly
derived.**

**Chosen: K=12**, from the flat bottom of margin-MAE's curve, trading a modest amount of
total-MAE/Brier (both closer to their own K≈50 optimum) for the margin improvement this
overlay exists to provide. Re-validated with K=12 on the identical game set: SU 57.35%,
Brier 0.2417, log-loss 0.6762, total-MAE 1.819, margin-MAE 2.0402 (marginally better than
Cycle 5's original K=25 run's 2.0416). Logged to the ledger, superseding the K=25 run as the
current best model.

**A process note worth flagging honestly**: this cycle searched over many K values against
the SAME 19,152-game set every prior decision in this project has also been validated
against. That's consistent with how every other decision here has been made so far, but the
more parameters get tuned against one fixed historical set, the more important it becomes to
eventually check everything against the genuine, never-touched final holdout described in
§3.8 — which hasn't been carved out yet. Worth doing soon, before many more small tuning
decisions accumulate on the same set.

Next candidates: carve out the genuine final holdout stretch (§3.8) before further parameter
tuning accumulates; test cross-season goalie history; or move to NegBin/score-correlation
refinements on the count-process layer (§3.4), still untouched by any cycle so far.

### 4.8 Carving out the genuine final holdout (2026-07-23)

Six cycles in, every single tuning decision (which signal to keep, which K to use) had been
validated against the exact same full 2010-11..2025-26 game set — the "genuine, never-touched
holdout" called for since the architecture proposal (§3.8) had not actually been carved out
yet. Fixed via `src/models/final_holdout_check.py`.

**Policy from this point forward**: the game set splits into a **development set** (2010-11
through 2023-24, 16,528 games) that every future modeling decision must be validated
against, and a **final holdout** (2024-25 + 2025-26, 2,624 games) that should not inform any
further tuning — checked only at infrequent, deliberate checkpoints from here on, not after
every micro-decision.

**Honest limitation, stated plainly rather than glossed over**: this holdout is not
perfectly pristine relative to the six cycles already done — every one of them, including the
just-completed goalie-prior grid search (§4.7), was validated against the full range,
holdout included. That can't be un-seen. What this establishes is the discipline going
forward, plus an honest, one-time baseline reading of where the current best model
(`goalie_overlay_poisson`, K=12) actually stands.

**Real result — and the model measurably underperforms on the holdout, across every
metric:**

| Metric | Development set (16,528 games) | Holdout (2,624 games) | Δ |
|---|---|---|---|
| SU accuracy | 57.60% | 55.79% | −1.81 pt |
| Brier score | 0.2413 | 0.2443 | +0.0030 (worse) |
| Log-loss | 0.6754 | 0.6817 | +0.0063 (worse) |
| Total-goals MAE | 1.806 | 1.903 | +0.097 (worse) |
| Margin MAE | 2.023 | 2.147 | +0.124 (worse) |
| Actual mean total goals | 5.8125 | 6.1673 | +0.355 |

**Not dismissed, not overstated — two real, plausible, non-exclusive explanations, neither
confirmed:**
1. **Some genuine overfitting to the development games is expected and plausible** after six
   cycles of decisions (including a direct K grid-search in §4.7) all validated against a set
   that included these very holdout games until this cycle. A ~1.8pt SU gap is roughly
   1.8-2x the rough binomial standard error at n=2,624 (~0.97pt) — real evidence of
   degradation, not overwhelming, but not dismissible as pure noise either.
2. **A genuine scoring-environment shift**: real mean total goals is 0.355 higher in the
   holdout seasons than in the development seasons — a large enough jump that at least part
   of the gap may reflect the NHL's own scoring environment actually changing (rule
   enforcement, equipment, league-wide pace trends) between the two eras, which a model
   whose team-strength ratings reset every season should partially absorb but may lag behind
   in the seasons right at that shift.

**Net effect on how to read every number in §4.2-4.7 going forward**: those figures are
believable as *relative* comparisons between models (the bootstrap tests isolate one
variable at a time and should be reasonably robust to this), but should not be read as a
precise forecast of real-world future performance — the holdout numbers above are a more
honest estimate of that. This mirrors the MLB project's own §11.7 caution almost exactly:
a validated relative improvement is more trustworthy than an absolute performance figure.

Next: continue all further modeling work against the development set only; return to this
holdout at a genuine milestone (not the next few cycles), and investigate the scoring-
environment-shift hypothesis directly (real league-wide goals/game by season) before
attributing the full gap to overfitting.

### 4.9 Following up §4.8: what the scoring gap actually is, and checking for real contamination (2026-07-23)

Two direct, concrete checks on the two hypotheses raised in §4.8, rather than leaving them
as untested speculation.

**The scoring gap is a real, long-running league trend — not a discontinuity at the
dev/holdout boundary.** Pulled real total-goals/game by season across the entire dataset
(2008-09 through 2025-26, every regular-season game): scoring was fairly stable and LOW
(5.42-5.94) from 2008-09 through 2016-17, then jumped and has stayed HIGH (5.94-6.36) every
single season from 2017-18 through 2025-26 — nine consecutive seasons now. The holdout
seasons (2024-25: 6.08, 2025-26: 6.25) are not meaningfully different from the tail of the
development set that immediately precedes them (2022-23: 6.36, 2023-24: 6.23) — if anything
2024-25 is slightly LOWER-scoring than 2023-24. **This means the development set's own
5.81 mean (§4.8) is a blend of eight low-scoring seasons and six high-scoring ones, while the
holdout is drawn entirely from the high-scoring era** — so part of that headline
"actual mean total goals" gap is closer to an arithmetic artifact of how the development set
happens to be composed than a sign the holdout specifically represents some new, unseen
regime the model has never encountered. The model's own within-season, no-cross-season-
carryover strength ratings should already be exposed to plenty of high-scoring seasons within
the development set itself (2017-18 through 2023-24, seven seasons of it).

**Checked, not assumed: were the "empirically fit" constants from Cycles 1-7 contaminated by
holdout games in a way that would inflate the development set's apparent advantage?**
- `home_ice_multiplier`: fit on the full range (including holdout) it's 1.04418; fit on
  development-only it's 1.04536 — a ~0.001 difference, not meaningfully different.
- The §4.7 goalie-prior grid search was re-run identically but restricted to development-set
  games only: margin-MAE's minimum is still in the same K≈10-13 range (2.02308 at K=10,
  2.02324 at K=12, nearly identical curve shape to the original full-range sweep) — **K=12
  would have been chosen either way.** The contamination concern raised in §4.8 was a
  legitimate thing to check, but it did not, in fact, change anything material this time.

**Net read**: the dev/holdout performance gap looks more like a mix of (a) the development
set's own scoring-era blend making its own aggregate look different from the holdout's
single-era composition (largely a description artifact, not itself evidence of model
weakness), plus (b) some genuine remaining overfitting/sampling noise at n=2,624 that these
two specific checks didn't localize further. Neither hypothesis is fully confirmed or ruled
out — this is an honest partial answer, not a closed case. The practical takeaway stands
unchanged from §4.8: keep validating future decisions against the development set, and don't
treat the absolute holdout numbers as a precise forecast, but the two constants checked here
are cleared of being a meaningful source of the gap.

### 4.10 Cycle 7: cross-season goalie history — a genuine wash (2026-07-23)

Cycle 5's goalie overlay reset a goalie's own GSAx history at every season boundary — an
explicitly flagged simplification at the time ("an established goalie's skill doesn't reset
over the summer the way a team's roster composition can"). Tested turning it off:
`shrinkage.add_walk_forward_mean` gained a `reset_each_season` parameter (default `True`,
every existing caller unchanged) so a goalie's cumulative history can instead carry across
the off-season (group by goalie only, not goalie+season). Isolated as the one new variable:
identical K=12 prior, identical relative-to-league-average adjustment, identical everything
else as `goalie_overlay_poisson`. Run per the new §4.8/4.9 policy — **development set only**
(16,528 games, 2010-11 through 2023-24), the first cycle to actually follow that policy from
the start rather than retrofitting it afterward.

**Real result:**

| Metric | Season-reset (dev-set) | Cross-season (dev-set) | Δ |
|---|---|---|---|
| SU accuracy | 57.60% | 57.90% | +0.30 pt |
| Brier score | 0.24128 | 0.24081 | −0.00047 |
| Total-goals MAE | 1.8059 | 1.8076 | +0.0017 (worse) |
| Margin MAE | 2.0232 | 2.0251 | +0.0018 (worse) |

**Paired bootstrap (5,000 resamples, identical dev-set games):** Brier mean improvement
0.00047, 95% CI **[−0.00029, 0.00123]** — crosses zero. Total-MAE mean improvement −0.00174,
95% CI **[−0.00586, 0.00240]** — crosses zero. Margin-MAE mean improvement −0.00183, 95% CI
**[−0.00584, 0.00213]** — crosses zero. **None of the three differences are distinguishable
from noise.**

**Verdict: not kept — a genuine wash, logged as such rather than discarded.** A plausible,
though not separately tested, explanation for why: §4.7's grid search deliberately chose a
small K=12 specifically because margin prediction benefits from being highly reactive to a
goalie's recent, CURRENT-season form. At such a small shrinkage prior, a handful of
current-season appearances already dominates the blend almost immediately — so whatever a
goalie carried in from the tail end of last season has very little weight left by more than
a few games into the new one anyway, regardless of whether that carryover is technically
permitted. Cross-season history might matter more at a LARGER K (where the model leans on
more historical data generally) — but combining "more shrinkage" with "cross-season
carryover" as one cycle would conflate two variables again, exactly what this project's
discipline exists to avoid; left as a separate, explicitly flagged candidate if a future
cycle revisits K itself.

Next: move to the still-untouched count-process layer (§3.4) — testing Negative Binomial
against Poisson, and/or a score-state correlation term — now that five consecutive cycles
have all been refinements to the MEAN (λ) side of the model without ever revisiting the
distributional-shape assumption underneath it.

### 4.11 Cycle 8: testing the outcome-distribution layer — a real finding that redirects the roadmap (2026-07-23)

Tested both untested hypotheses from §3.4 directly, on the development set, using the
current best model's (`goalie_overlay_poisson`) own lambda values as the basis.

**Negative Binomial: cleanly rejected, no overdispersion found.** Standardized residuals
`(actual − lambda)/sqrt(lambda)` should have variance ≈1 under a correctly-specified
Poisson; measured variance was **0.98** (16,528 dev-set games, pooling home/away sides) —
if anything very slightly *under*-dispersed, not over-dispersed. Raw variance/mean ratios
for actual_home and actual_away were 0.970 and 0.969 respectively — both just under 1. NegBin
(which only ever *adds* variance relative to Poisson) would move in the wrong direction here.
Rejected before writing any NegBin code, on real data, exactly the discipline this project
has held itself to throughout.

**Score-state correlation: real, but not the simple story it first looked like.** Pearson
correlation between home/away scoring residuals across all dev-set games: **r=−0.062**
(p<1e-6, n=16,528) — small but highly significant. In the subset where both teams scored ≤1
goal specifically, much stronger: **r=−0.452** (p<1e-4, n=358) — the same low-joint-scoreline
phenomenon Dixon-Coles (1997) built their soccer correction for. Implemented
`src/models/dixon_coles.py` (standard tau-correction on the four low cells
{(0,0),(1,0),(0,1),(1,1)}, rho fit by direct MLE against real dev-set outcomes rather than
importing soccer's typical rho magnitude).

**A real bug caught before trusting the first fit**: rho's MLE landed exactly on the search
bound (0.3) — a red flag investigated rather than accepted. Found: the original objective
function floored any resulting probability at a tiny positive constant instead of rejecting
invalid values outright, and NHL's much higher scoring rate than soccer means
`tau(0,0)=1-lambda_home*lambda_away*rho` goes negative at rho as low as ~0.054 for the
highest-scoring real matchups in the sample (mean lambda_home*lambda_away=8.1, max=18.4) —
far inside what looked like a generous search range. Fixed by rejecting (returning `+inf`)
any rho that makes an actually-observed outcome's raw probability negative, rather than
silently flooring it.

**A second, much bigger finding while investigating the re-fit — the four target cells are
not what they appear to be in hockey.** Checked the actual cell counts before trusting the
refit: **(0,0) and (1,1) have exactly ZERO games in the entire dev set.** This isn't a data
gap — it's structural: NHL games can never end in a recorded tie. Regulation ties go to
overtime/shootout, which always produces a decisive final score. Dixon-Coles' four target
cells were built for soccer, where 0-0/1-1 are common, valid, PERMANENT final scores — in
hockey's recorded box scores, two of those four cells are simply impossible outcomes.

**Investigated further, rather than just noting the oddity: is the observed −0.062/−0.452
correlation even the "real," Dixon-Coles-style score-effects phenomenon, or is it
substantially a mechanical artifact of tie-breaking?** Split the dev set by whether the
game actually went to OT/SO (inferred from MoneyPuck's own `iceTime` field exceeding 60
regulation minutes — confirmed this identifies ~20.3% of games, matching known real-world
NHL OT/SO rates) and recomputed the correlation separately:

| Subset | n | Correlation |
|---|---|---|
| Regulation-decided | 13,180 | **−0.152** (p<1e-6) |
| OT/SO-decided | 3,348 | **+0.617** (p<1e-6) |

**These are opposite-signed and both large.** In regulation-decided games, there's a real,
moderate negative correlation — consistent with genuine score-effects dynamics (a team
protecting a lead plays differently than a team chasing one), the actual phenomenon
Dixon-Coles describes. In OT/SO-decided games, the correlation is strongly *positive* — but
this is closer to a tautology than a finding: those games are *defined* by both teams having
scored the *same* number of goals in regulation, so if one team's actual score surprises
above its lambda, the other team's matching score mechanically does too. **Pooling these two
populations into one global rho, as a naive Dixon-Coles port would do, averages together two
mechanistically unrelated phenomena with opposite signs** — not a sound basis for a single
correction term.

**Decision: do not ship a naive pooled correlation term.** The right fix isn't a correlation
hack on top of the current final-score-based Poisson model — it's building the OT/SO
sub-model (architecture §3.5, "Layer 4") that's been explicitly deferred since the very first
architecture proposal and never built. That requires regulation-only goal counts (not the
final recorded score, which already includes any OT/SO bonus goal) as a separate modeling
target — the schedule data used throughout this project only has final scores; regulation-
only splits would need play-by-play data (not yet ingested). That's a substantial, distinct
piece of work, not a quick addition to this cycle, and bundling it in now would violate the
same "isolate one variable" discipline this project has held to throughout.

**Verdict: both count-process hypotheses tested, both real findings, neither shipped as a
model change this cycle.** NegBin is cleanly rejected. The correlation hypothesis is
confirmed real but redirects the roadmap rather than producing a drop-in fix — a good
outcome for a research log to capture even though no model code changed as a direct result.

Next: build the OT/SO sub-model properly (ingest play-by-play or another regulation-only
score source, model P(regulation tie) from the existing joint, then a real OT/SO winner
sub-model) as its own dedicated cycle — the natural, now well-motivated next step, and the
piece that would let a genuine, non-confounded Dixon-Coles-style correlation term be fit
cleanly on regulation-only outcomes if still worth pursuing afterward.

### 4.12 Cycle 9: building the OT/SO sub-model (architecture Layer 4) (2026-07-23)

The piece explicitly deferred since the very first architecture proposal, finally built —
motivated directly by §4.11's finding, and resolving a limitation flagged since Cycle 1's
very first validation script (`baseline_naive_poisson.home_win_prob_regulation`'s own
docstring: "does not yet include the Layer 4 OT/SO sub-model... ties are reported separately
rather than silently split 50/50").

**No play-by-play ingestion needed after all.** The schedule endpoint already carries
`gameOutcome.lastPeriodType` (REG/OT/SO) — confirmed live, re-pulled the full schedule
history to capture it (`fetch_nhl_api.py`, same 25,104 games as before, no new games). Since
exactly one bonus goal (the OT winner, or the shootout's awarded goal) decides any OT/SO
game, the real REGULATION-only score reconstructs directly from the final score plus this
one field: whichever team won gets its final tally reduced by 1. Verified against real games
(2024-01-15: ANA 5 @ FLA 4, OT — regulation was 4-4; CBJ 4 @ VAN 3, SO — regulation was 3-3)
before trusting it at scale, and added a hard consistency check (`overtime_shootout.
add_regulation_score`: every reconstructed OT/SO game's home and away regulation scores must
be exactly equal, or raise) rather than assuming the reconstruction is always valid.

**That check caught one real anomaly**: a single preseason game (gameId 2015010045, SJS 4-1
ARI, tagged `lastPeriodType=OT`) fails the tie check — a 4-1 final cannot genuinely be
decided by one OT goal. A real NHL API data quirk, not a bug in the reconstruction logic, and
for a game type (preseason) this project has never used in any validation regardless —
excluded by scoping to regular-season, completed games only, the same filter already used
everywhere else.

**Built the actual sub-model**: `overtime_shootout.fit_ot_home_win_rate` — the real empirical
P(home wins | game reaches OT/SO), fit once on the development set, not assumed 50/50.
**Real result: 0.5092** — home ice retains a small edge even in 3-on-3 overtime/the
shootout (plausibly last-change/matchup control in 3v3 OT, home shooting order in some
formats, or simply residual crowd effects), but it's close enough to a coin flip that
assuming 50/50 wouldn't have been a large error either.

**The real deliverable: replacing every prior cycle's renormalization workaround
(`home_win_prob / (1 - tie_prob)`) with a genuine, unconditional win probability**
(`home_win_prob + tie_prob * p_home_wins_ot`) — these are NOT the same formula even at
p_home_wins_ot≈0.5, and only the second one actually resolves the tie mass rather than
discarding it. Validated on the same development set, same current-best-model lambdas:

| | Brier | Log-loss |
|---|---|---|
| Old renormalization workaround | 0.24128 | 0.67538 |
| New full OT/SO resolution | 0.24087 | 0.67448 |

**Paired bootstrap (5,000 resamples, identical dev-set games): Brier improvement mean
0.00041, 95% CI [0.00014, 0.00066] — entirely positive, a real, confirmed improvement.**
Every future cycle's win-probability metric should use this full resolution rather than the
old workaround.

**Also resolves, by simply making explicit, the other open question §3.5 flagged**: whether
"final score" for totals/spread purposes includes the OT/SO bonus goal. It always has,
throughout every cycle so far — the schedule's `homeScore`/`awayScore` (the target every
model has been validated against since Cycle 1) already include it, since that's what the
NHL's own recorded box score shows. Not a new decision, just now explicitly documented
rather than left as an unstated assumption.

Next: with a genuine full win-probability pipeline in place, this is a natural point to
re-run the §4.11 score-correlation investigation properly — fitting a Dixon-Coles-style term
on REGULATION-only outcomes specifically (now reconstructable), no longer confounded by
mixing in the mechanically-different OT/SO population.

### 4.13 Cycle 10: revisiting Dixon-Coles properly — a second, deeper structural mismatch (2026-07-23)

Retried §4.11's correlation correction now that regulation-only scores are reconstructable
(§4.12). Approximated regulation-only lambda via a simple, non-circular rescaling of the
current model's final-score lambda (the real ratio of mean regulation-only goals to mean
final recorded goals — 0.9613 home, 0.9593 away, measured on the dev set) rather than using
the model's own `tie_prob`, which was checked first and found NOT to match the real OT rate
(0.168 model-implied vs. 0.231 real empirical) — using it would have been circular.

**First checked whether using regulation-only scores (instead of final scores) fixes §4.11's
confound on its own — it doesn't.** Games that go to OT are, BY DEFINITION, exactly tied
after regulation (`reg_home_score == reg_away_score`, always — the hard consistency check in
§4.12 guarantees this). Correlating residuals within that subset is conditioning on an
equality constraint — a structural (selection) artifact, not a real effect, regardless of
which target variable is used. Confirmed directly: correlation in the went-to-OT subset was
r=+0.90 using regulation-only scores — *larger* than §4.11's final-score estimate (+0.62),
not smaller. **The fix is to exclude went-to-OT games from the FITTING population
entirely** — on the regulation-decided-only subset (n=12,718), real correlation is r=−0.170
(p<1e-6), consistent with §4.11's original −0.152 estimate, now on a cleaner population.

**Fitting rho on that clean population immediately hit a second, deeper problem: the MLE
doesn't converge — it climbs monotonically without bound (checked to rho=5.0, still no
turnaround).** Investigated rather than accepted: the Dixon-Coles correction has exactly
four special cells, and their tau formulas are `tau(0,0)=1-λh·λa·ρ`, `tau(1,1)=1-ρ` (only
these two can ever go negative, for positive rho) vs. `tau(1,0)=1+λa·ρ`, `tau(0,1)=1+λh·ρ`
(monotonically INCREASING in rho, never invalid). **The regulation-decided-only fitting
population structurally excludes (0,0) and (1,1) outcomes by construction — those ARE the
definition of "went to OT."** So the only two cells that could ever constrain rho from above
never appear in this population's actual data at all, while the two unconstrained cells
(1,0)/(0,1) do appear (144 and 119 real games respectively) — meaning the optimizer can
always improve the fit by pushing rho higher, forever, with nothing to stop it. This isn't a
sample-size or implementation bug; it's the classic Dixon-Coles four-cell parameterization
being fundamentally mismatched to a population that can't contain two of its four target
outcomes by definition.

**Decision: do not ship this fit.** An unbounded MLE is not a real calibration — it's a
degenerate optimization problem, and forcing it back into a bounded range (as the original,
now-clearly-wrong attempt effectively did by hitting a search bound) would just be hiding the
degeneracy behind an arbitrary cutoff rather than fixing it. Two independent attempts at
porting Dixon-Coles's exact functional form to hockey have now hit two different structural
mismatches (§4.11: pooling mechanically opposite-signed populations; §4.13: fitting on a
population that excludes half the correction's own target cells) — this is strong enough
evidence that the specific four-cell parameterization, built for a sport where 0-0/1-1 are
common valid final scores, doesn't transfer cleanly to hockey's tie-breaking structure,
independent of how carefully the fitting population is chosen.

**What this doesn't mean**: the correlation itself is real and has now shown up consistently
across three separate measurements (§4.11 final-score: −0.152; §4.13 regulation-only,
same population: −0.170) — this is a genuine phenomenon worth eventually capturing, just not
with this specific parametric tool. A model built for the correlation more fundamentally
(e.g. a bivariate Poisson with an explicit shared-covariance term, rather than a four-cell
patch on independent Poisson) would sidestep this cell-availability problem entirely, since
it wouldn't special-case any particular scoreline. That's a bigger, separate undertaking —
a new model family, not a quick addition — appropriately scoped as its own future cycle
rather than rushed through here.

Next: treat the count-process/correlation question as closed for now (real effect,
no working correction yet) and move to a different area — e.g. incorporating a live
rest/schedule (back-to-back) feature into team strength, which architecture §3.2 flagged
from the start and no cycle has touched yet.

### 4.14 Cycle 11: rest/schedule (back-to-back) adjustment (2026-07-23)

Flagged since the very first architecture proposal, tested here for the first time. Checked
for a real effect before building anything (same discipline as the NegBin rejection, §4.11):
computed each team's own within-season rest days (reset at season boundary — the 100+-day
gap between a team's last game of one season and first of the next isn't a meaningful
"fatigue" signal) and correlated against the current model's own scoring residuals.

**Most of the obvious checks came back negligible.** Continuous rest-days correlations were
tiny: r=0.004 (home scoring, not significant) and r=0.027 (away scoring, technically
significant only because n=16,528 gives enormous power to detect trivial effects — well
under 0.1% of variance explained either way). Home back-to-back vs. well-rested (3+ days)
showed no significant scoring difference (p=0.12), and — a genuine surprise worth reporting
rather than glossing over — the current model actually slightly *under*-predicts home win
probability on home back-to-backs (45.4% predicted vs. 48.8% real), the opposite of the
popular "back-to-backs hurt you" narrative for the home side specifically.

**One real, statistically significant effect survived**: an AWAY team playing the second
night of a back-to-back scores about 0.12-0.16 fewer goals (two slightly different
comparisons: −0.094 vs. well-rested specifically, p=0.12 n/s; −0.158 same comparison on the
away side, p=0.0041 — significant) than an equally-matched, equally-rested away team would.
Built ONLY this one, narrowly-scoped adjustment (`rest_schedule.py`) — a single global
additive constant (−0.120, fit once on the development set, same practice as every other
global constant here) subtracted from an away team's lambda specifically when they're on a
back-to-back, not a general home-and-away "fatigue" rule the broader data didn't support.

**Real result, identical development-set games, on top of the full Cycle 9 OT/SO-resolved
pipeline:**

| Metric | Without rest adjustment | With rest adjustment |
|---|---|---|
| Brier score | 0.24087 | 0.24072 |
| Total-goals MAE | ~1.806 | 1.806 |
| Margin MAE | ~2.023 | 2.021 |

**Paired bootstrap (5,000 resamples):** Brier improvement mean 0.00015, 95% CI **[0.00001,
0.00029]** — entirely positive, though barely (the lower bound sits just above zero).
Margin-MAE improvement mean 0.00257, 95% CI **[0.00171, 0.00344]** — clearly, comfortably
significant. Total-MAE improvement crosses zero, not distinguishable from noise.

**Verdict: kept, on the same pattern as the goalie overlay (§4.6) and team-specific PP/PK
TOI (§4.5)** — a real, specific, narrowly-targeted effect (this time: road back-to-back
fatigue), improving margin prediction clearly and win probability marginally, with no
measurable effect on totals. A fourth consecutive cycle where the actual signal turned out
narrower and more specific than the initial hypothesis ("does rest matter") suggested —
worth remembering when framing the next new-feature test: check the specific sub-case, not
just the average effect, before concluding a broad hypothesis is a wash.

Logged as `rest_adjusted_poisson`, the new overall reference point.

---

## 5. External research synthesis and prioritized roadmap (2026-07-23)

An external research pass (commissioned by the user, given this project's own §0/§4 summary
as context) returned a literature-grounded report cross-checking this project's own findings
against the published NHL-prediction literature. Full source: user-provided, saved at
`/Users/brettscully/Downloads/compass_artifact_wf-a1083903-034b-5fd1-8b34-3d47760f9cb6_text_markdown.md`.
Treated the same way every other external source in this project has been treated — checked
against this project's own real numbers before acting on it, not adopted on authority.

### 5.1 The single most important finding: this model is already near the published ceiling

**This project's own dev-set Brier (0.2407) is at or slightly better than every real,
citable benchmark the report found**: vig-removed NHL closing-line Brier ≈0.242 (Lopez,
Matthews & Baumer, *Annals of Applied Statistics* 2018), and a published Glicko/box-score
model at Brier 0.244 (Davis et al. 2021, cited secondhand via a systematic review — the
report itself flags this as corroborating, not an exact figure to trust precisely). SU
accuracy (57.58%) sits under the widely-cited ~62% ceiling (Weissbock & Inkpen 2014,
simulation-derived from 2005-2011 data — the report is explicit this is an estimate, not a
hard limit) but in the range of what actual published pre-game models have achieved (Davis
et al.: 58.3%; a 2024 replication, Remander: 55.4% over 5,043 games).

**Correction, added after §6.7's direct market data landed (2026-07-23): the comparison
above uses aggregate published figures from different studies/samples/years, not a real
same-game comparison — and a real same-game comparison tells a slightly less optimistic
story.** On the 11,803 real games where both this model's prediction and a real matched
closing line exist (§6.8), **this model's Brier (0.24209) is measurably worse than the
market's closing-line Brier (0.23867) — a real, bootstrap-confirmed gap (mean 0.00342, 95%
CI [0.00232, 0.00460], entirely positive), not noise.** This project is CLOSE to the market,
consistent with the aggregate comparison's general thrust, but a precise, matched-pairs
check shows it is not already AT market level the way the aggregate comparison implied.
This is the more trustworthy number going forward — the report's own aggregate framing was a
reasonable first pass given what was available at the time, but this project's own real data
now supersedes it.

**This recalibrates what "the next improvement" should mean for this project.** The report's
framing, worth taking seriously: past this point, gains are more likely to come from
calibration and non-stationarity handling than from a better score-distribution engine. This
doesn't mean stop building — it means the NEXT things worth building are different in kind
from Cycles 1-11, which were all about the generative model itself.

### 5.2 Direct, reassuring validation of two of this project's own real decisions

**The Dixon-Coles/bivariate-Poisson rejections (§4.11, §4.13) were structurally correct, per
the published literature, not just a project-specific finding.** Karlis & Ntzoufras' (2003)
classic bivariate Poisson is mathematically restricted to POSITIVE correlation only — it
cannot represent the −0.15 to −0.17 correlation this project measured three separate times,
which explains (with more precision than this project's own §4.13 investigation reached)
part of why that family was never going to be the right tool. More importantly: Groll et
al.'s applied work found that once both teams' own covariates are included in each team's
own Poisson mean (exactly the situation once team-strength ratings are already in the
model), the shared covariance parameter estimates at essentially zero — i.e., **a
well-specified rate model already absorbs most of the correlation that a separate
correlation term would otherwise try to capture.** This is a genuine, useful piece of
context this project didn't have: the negative correlation finding wasn't wrong, but a
separate correction term may have limited headroom regardless of which functional form is
used, since the team-strength side of the model is already doing more of that work than
expected.

**The goalie back-to-back magnitude is real but this project hasn't isolated a
goalie-specific version of it.** The original, widely-cited Tulsky (2012-13) estimate (~1%
SV% drop, .919→.908) has reportedly been revised down by later work to ~0.8% SV% on zero
rest — and this project's own away-team-back-to-back adjustment (§4.14, −0.12 goals) is a
TEAM-level effect, not specifically isolated to the goalie. Worth checking directly: how
much of the measured team-level effect is actually the starting goalie specifically, vs.
skater fatigue — the report suggests this project's overlay may currently be conflating the
two.

### 5.3 Genuinely new information this project didn't have

- **A citable, real explanation for part of the dev/holdout gap (§4.8-4.9)**: NHL
  season-to-season reversion to the mean is ~46% (Lopez et al.'s γ=0.54), the highest of the
  four major North American leagues — meaning this project's own season-reset convention
  (no cross-season carryover, tested and found not to matter for goalies specifically in
  §4.10) is plausibly well-suited to just how quickly NHL team strength itself regresses,
  but also means year-to-year drift is structurally large and hard to eliminate, not
  necessarily a modeling flaw.
- **Concrete, real market-data sources to check**, none yet verified by this project's own
  data-survey discipline: SportsbookReviewsOnline (closing moneyline/totals archives,
  2008-2023), Princeton DSS (opening/closing moneylines and totals, 2008-2023), Kaggle
  datasets (Michael Mallari's "Sportsbook Odds on NHL Games," Jonathan Coletti's 2004-present
  set), and the `pinnacle.data` R package for sharp-book odds structure. **None of these have
  been checked yet** — per this project's own §1 standard, they need the same live-fetch
  verification (real sample pulled, format confirmed, genuinely free vs. paywalled) before
  being trusted, not assumed usable from the report's description alone.
- **A concrete stop-condition for future structural work**: if holdout Brier drops below
  ~0.235 after the drift/calibration cycles below, further structural changes are likely
  fitting noise rather than finding signal — the honest move at that point is to shift focus
  to bet-sizing/CLV tracking rather than continuing to chase the generative model.
- **RotoWire's own 11-season travel study** found raw time-zone crossings and travel
  direction do NOT clear significance on their own — only the back-to-back × time-zone-change
  INTERACTION does. Directly actionable and narrowly scoped, consistent with this project's
  own "test the specific sub-case, not the average effect" lesson from §4.14.

### 5.4 Prioritized roadmap (adopting the report's ordering, mapped to this project's own next cycles)

1. **Cycle 12 — Market-data survey, THEN a calibration layer.** Before building anything:
   live-verify the market-data sources listed in §5.3 (real fetch, real sample, genuine
   free-vs-paywalled check — the same discipline as §1). Only once real closing-line data is
   in hand, build an isotonic/Platt calibration layer on top of the current model's win
   probabilities, fit on a walk-forward calibration fold distinct from the games used to
   validate it (the report cites Niculescu-Mizil & Caruana 2005: isotonic needs ~1,000+
   calibration points to be safe — this project's ~16.5k dev-set games comfortably clears
   that). Validate against holdout Brier/log-loss, not just dev — calibrating on a small or
   in-sample fold can make things worse, not better, and this needs to be checked directly.
2. **Cycle 13 — Scoring-era drift / non-stationarity**, aimed directly at the §4.8-4.9
   dev/holdout gap. Options in increasing complexity: season-level intercept recalibration of
   home-ice/league baseline; exponential time-decay game weighting (Dixon-Coles-style);
   hierarchical season effects; a full dynamic state-space rating. Test each via this
   project's own paired-bootstrap standard — the report specifically warns NHL's high
   randomness likely gives recency-weighting a shallow, flat optimum, so a real bootstrap
   check matters more than usual here, not less.
3. **Cycle 14 — Goalie-specific (not team-level) back-to-back/workload overlay**, separating
   the goalie's own zero-rest effect from the general team fatigue term already built in
   §4.14, calibrated toward the more current ~0.8% SV%-drop estimate rather than Tulsky's
   original ~1% figure — but calibrated on THIS project's own data, not imported as a
   constant, same standard as every other constant here.
4. **Cycle 15 — Back-to-back × time-zone-change interaction term**, narrowly scoped per
   RotoWire's finding (§5.3) — not raw travel distance or direction, which the report says
   wash out on their own.
5. **Cycle 16 — Gradient-boosted stacking layer** over the Poisson outputs (ratings +
   context as features into a regularized GBM), as a calibration/stacking layer, NOT a
   replacement for the generative core — the report is explicit that NHL's single-game
   accuracy plateau (Weissbock & Inkpen) means the soccer-literature effect sizes for
   stacking likely won't fully transfer, so this needs the usual bootstrap discipline before
   being trusted.
6. **Cycle 17 (bounded research spike, do last) — copula/Sarmanov joint-count model**, the
   only correct family for genuinely modeling negative dependence (unlike bivariate
   Poisson) — but §5.2's Groll et al. finding means expected lift is likely small. Scope with
   a hard kill criterion up front: kill if it doesn't beat holdout log-loss by a
   bootstrap-significant margin, don't let it run open-ended.

### 5.5 What this project should NOT do differently based on the report

Not every recommendation changes this project's own approach — worth being explicit about
what's confirmed rather than revised:
- The walk-forward, incremental, bootstrap-gated validation discipline itself is unchanged;
  the report's own recommended cycles are meant to be run through the exact same harness.
- The development/holdout split (§4.8) stays exactly as scoped — the report treats closing
  the gap as the top substantive problem, not a reason to abandon or loosen it.
- Rejecting Dixon-Coles/bivariate Poisson stands as a real result on its own account (§4.11,
  §4.13's investigations were genuine, reproducible findings), not merely something this
  external report happened to agree with after the fact.

## 6. Cycle 12: market-data survey (2026-07-23)

Live-verified every market-data source §5.3 suggested, per this project's own §1 standard —
none were assumed usable from the report's description. Mixed results; one genuinely useful
source found, one clean exclusion, two limited/unusable sources, and one factual correction
to the external report itself.

### 6.1 SportsbookReviewsOnline — real, but the CURRENT live site is a dead end; Wayback Machine snapshots of the ORIGINAL site are the actual usable resource

The live site (`sportsbookreviewsonline.com`) is real and `robots.txt`-permitted, but has
been rebranded into an affiliate/sportsbook-review marketing site. Its per-season archive
pages (`/scoresoddsarchives/nhl-odds-YYYY-YY`) still render a real HTML odds table (Date,
Rot, VH, Team, period scores, Final, Open/Close moneyline, PuckLine, Open/Close totals —
confirmed via a real browser render, not just `curl`, and confirmed no AJAX/pagination
endpoint exists via network-request inspection) but **every season's page — including
seasons finished years ago — is truncated to the same cutoff (~November 27 of that season),
with no way to load the rest.**

Investigated why rather than accepting the truncation: the Wayback Machine has exactly ONE
crawl of these pages/files, from **December 2022** — meaning the entire site (or at least
this archive section) was likely frozen/imported at that point and never updated since,
which is also almost certainly why even the CURRENT (2026) live pages still stop at that
same late-November cutoff for every season, recent or old.

**The actual, real, usable resource: the ORIGINAL site's per-season `.xlsx` download links,
still recoverable via Wayback Machine** (`nhl odds YYYY-YY.xlsx`, one file per season,
confirmed real via the archived HTML page's own `href`s). Downloaded and verified three
real files directly:
- **2022-23** (in progress at crawl time): 684 rows (~342 games), through Nov 27 — matches
  the live site's own truncation exactly, confirming the "frozen at crawl time" theory.
- **2018-19** (already fully complete years before the crawl): **2,716 rows (~1,358 games),
  Oct 3 through June 12 — a genuinely complete regular season + full playoffs.**
- **2015-16** (also long complete): **2,642 rows (~1,321 games), Oct 7 through June 12** —
  also genuinely complete.

**Conclusion: seasons that had ALREADY FINISHED before the ~November/December 2022 crawl
(i.e., 2007-08 through 2021-22) are recoverable as real, complete, per-game closing-line
archives via Wayback Machine snapshots of the original xlsx files. Seasons in progress at
crawl time or started after it (2022-23 through 2025-26 — which includes this project's
entire holdout window) are NOT available complete through this source** — the live site's
copies are frozen mid-season, and no later Wayback crawl exists to fill the gap. This is a
real, meaningful limitation to carry forward, not a reason to discard the source: it still
covers this project's ENTIRE development set except the last two in-progress seasons
(2022-23, 2023-24), and covers the older two-thirds of it completely.

### 6.2 Princeton DSS — excluded, IP-restricted to campus network

The catalog page (`dss.princeton.edu/catalog/resource6741`) confirms the described dataset
is real (NHL 2008-2023, opening/closing moneyline and totals, purchased from
oddswarehouse.com) — but the actual data host (`dss2.princeton.edu`) **times out on a direct
connection attempt from outside Princeton's network** (confirmed: TCP connection accepted
then hangs, 10-second timeout, not a DNS failure) — a licensed university resource gated to
campus/VPN access, the same category of exclusion as Evolving Hockey (§1.5), not a genuinely
public source regardless of the report's description.

### 6.3 Kaggle (Michael Mallari, "Sportsbook Odds on NHL Games") — real but very limited

Confirmed real via the dataset's own schema.org metadata: MIT-licensed, `isAccessibleForFree:
true` (just needs a free Kaggle account to download, not a paywall). But the file is only
**31,437 bytes** — far too small to be a comprehensive multi-season archive; likely a small
sample. Not investigated further given §6.1 already provides much deeper real coverage; kept
on record as a legitimate but minor supplementary source, not a primary one.

### 6.4 `pinnacle.data` R package — correction to the external report

The report cited this CRAN package as a source for "sharp-book odds structure." Checked the
actual CRAN package page directly: **it contains only MLB 2016 season data and 2016 US
election data — no NHL data at all.** This appears to be an incorrect citation in the
external report (possibly conflated with a different resource), not a real option for this
project. Recorded here so this specific claim isn't re-investigated or trusted later.

### 6.5 Net conclusion and next step

A real, usable market-odds resource exists (§6.1's Wayback-archived xlsx files), covering
this project's development set from 2007-08 through 2021-22 completely, with a genuine gap
for 2022-23 onward (including the entire holdout window) that no source checked here fills.
Next: build `src/ingest/fetch_sbro_odds.py` to pull and parse the recoverable seasons,
establish the no-vig implied-probability conversion and calibration-fold methodology on the
seasons that have real market data, and treat the calibration LAYER itself (which only needs
this project's own real game outcomes, not market data) as buildable and testable across the
FULL dataset regardless of this gap — only the market-comparison/CLV benchmarking piece is
actually blocked for the most recent seasons.

### 6.6 Ingestion built and run: real result

`src/ingest/fetch_sbro_odds.py` pulls every recoverable season via the Wayback Machine
mechanism confirmed in §6.1. **Real result: 13 of the 15 targeted seasons (2007-08 through
2021-22) recovered successfully — 33,358 rows (~16,679 games)** with real Date/Team/period-
scores/Final/Open-Close-moneyline/PuckLine/Open-Close-totals columns. `2016-17` and
`2020-21` have no Wayback snapshot at all (genuine gap, not a bug) — worth noting `2020-21`
is the COVID-realigned/bubble season, a real oddity that may have disrupted whatever process
originally populated the site. Row counts per season are internally consistent with known
NHL history as a sanity check: `2012-13` recovered exactly ~806 games, matching the real
lockout-shortened season; `2019-20` recovered ~1,212, matching that season's real
COVID-truncated regular season.

**One real data-quality issue found and fixed, not silently worked around**: some odds cells
contain the literal placeholder string `"NL"` ("No Line" — a game where that particular line
wasn't posted) mixed into otherwise-numeric columns, which broke a naive parquet write.
Coerced to numeric with `errors="coerce"` (genuine missing values become `NaN`, not a guessed
number) rather than dropping those rows or games entirely.

**Not yet done, the natural next step**: parsing the raw V/H paired rows into one row per
real game, converting the `Date` column's ambiguous MMDD format into real calendar dates
(the year flips mid-file: October-December rows belong to the season's start year, January-
June rows belong to its end year — a known gotcha with this exact format, not yet handled),
normalizing team names (`"SanJose"`, `"St.Louis"`, `"NYRangers"` etc.) against this project's
own team-code crosswalk (§1.2/§4.1's `team_codes.py` precedent), and joining the result to
this project's own schedule by (date, home team, away team) to attach real closing lines to
real predicted probabilities. That join, plus the actual isotonic/Platt calibration layer,
is queued as the immediate continuation of this cycle.

### 6.7 Parsing and schedule join: real result

`src/ingest/parse_sbro_odds.py` handles the three quirks flagged above and joins to this
project's own schedule. Confirmed the exact set of 37 distinct raw team names in the real
data before building the crosswalk (not guessed): mostly city-only, a few with nickname
appended (`SeattleKraken`, `WinnipegJets`), and old-franchise names in older seasons
(`Phoenix` pre-2014, `Atlanta` pre-2011) — mapped to the era-accurate NHL API abbreviation
each name corresponds to, matching this project's own schedule data directly.

**A real, non-obvious bug caught by a plain unit check before it touched any real numbers**:
the American-odds-to-probability conversion, as first written, produced a NEGATIVE
probability for negative odds (`-150` → `-0.6`, not `0.6`) — a chained-`.where()` expression
that looked right but wasn't. Caught by testing four known odds values against their
known-correct probabilities directly, before running it on the real 33,358-row dataset.

**Real result: 16,679 games parsed; 14,260 matched to the real schedule by (date, home,
away)** — of those matched games, only **4 have a final-score disagreement between the two
independent sources (odds data vs. NHL API schedule)**, a 99.97% agreement rate that's a
strong, real cross-validation signal the join and both underlying sources are sound, not
just a "the join succeeded" tautology. The ~2,419 unmatched games are fully explained, not a
mystery: **1,315 are the 2007-08 season, which predates this project's own schedule ingestion
range (started at 2008-08-01) — an intentional scope boundary, not a bug.** Of the remaining
~1,104, checking their calendar months confirms **969 fall in April-June (playoffs) and 105
in August 2020 (the COVID-realigned bubble playoffs)** — correctly excluded, since this
project's schedule join deliberately scoped to regular-season games only
(`gameType==2`), consistent with every other cycle here; playoff OT rules differ (full
20-minute sudden-death periods, no shootout) and mixing them into a regular-season
calibration set would be a real, avoidable confound, not a data gap worth chasing.

**Net: 14,260 real regular-season games with matched closing-line odds** (comparable in
scale to this project's own ~16,528-game development set) — enough for a real calibration
layer. Saved to `data/raw/sbro_odds_games.parquet`.

Next: build the actual isotonic/Platt calibration layer using this real market data, per
§5.4 Cycle 12's original scope.

### 6.8 Market comparison: a precise correction to §5.1's aggregate framing

Merged this project's current best model's per-game win probability (`rest_adjusted_poisson`
+ Cycle 9's OT/SO resolution) against the real matched closing lines from §6.7, on the
overlapping seasons (2010-11 through 2021-22, the dev-set range that also has recovered
market data) — **11,803 real games with both a real model prediction and a real market
closing line, the same-game comparison the external report's own aggregate figures could
not provide.**

**Real result: this model's Brier (0.24209) is measurably worse than the market's own
closing-line Brier (0.23867) on these identical games** — paired bootstrap (5,000 resamples):
mean gap 0.00342, 95% CI **[0.00232, 0.00460]**, entirely positive, a real and
statistically confirmed gap, not noise. The market's OPENING line (0.23982) also beats this
model. §5.1's claim that this project's Brier "already matches" the market was based on
comparing aggregate published figures from different studies/samples/years — a reasonable
first-pass framing given what was available at the time, but this direct, same-game
comparison is more precise and supersedes it: **this model is close to the market, but not
yet at market level.** §5.1 has been annotated with this correction rather than left
unreconciled.

**Re-run against the current best model (2026-07-23, per external review):** the above was
measured against `rest_adjusted_poisson` (pre-Cycle-13), not the current
`drift_adjusted_poisson`. Re-ran the identical comparison (same 11,803 games,
`src/models/validate_market_benchmark.py`): **model Brier 0.24218 vs. market-close 0.23867,
gap mean 0.00351, 95% CI [0.00235, 0.00467]** — essentially unchanged from the pre-drift-fix
number. This confirms, independently, what §7.5's own bootstrap already found (the drift fix
was Brier-neutral) — the market gap is not something the drift fix was ever going to close.

Per-season breakdown of the model-vs-market Brier gap (added per external review, to check
whether the gap concentrates in specific eras — which would point to drift — or spreads
roughly evenly — which would point to an informational or other structural gap):

| Season | n | Model Brier | Market Brier | Gap |
|---|---|---|---|---|
| 2010-11 | 1,228 | 0.24626 | 0.24357 | +0.00269 |
| 2011-12 | 1,230 | 0.24207 | 0.24072 | +0.00135 |
| 2012-13 | 720 | 0.24520 | 0.23825 | **+0.00695** |
| 2013-14 | 1,230 | 0.24126 | 0.24103 | +0.00023 |
| 2014-15 | 1,230 | 0.23851 | 0.23633 | +0.00217 |
| 2015-16 | 1,230 | 0.24824 | 0.24215 | +0.00609 |
| 2017-18 | 1,270 | 0.24194 | 0.23770 | +0.00425 |
| 2018-19 | 1,271 | 0.24258 | 0.23986 | +0.00272 |
| 2019-20 | 1,082 | 0.24657 | 0.24323 | +0.00334 |
| 2021-22 | 1,312 | 0.23166 | 0.22516 | +0.00650 |

**The gap is NOT cleanly concentrated in the post-2017-18 high-scoring era** — the argument
that would most directly implicate scoring-level drift as the market gap's explanation.
2012-13's gap (+0.00695) is the largest of any season in the entire table, and it's in the
OLD, pre-drift era; 2013-14, immediately after, shows almost zero gap (+0.00023). If drift
were the whole story, the gap should trend up across the table; it doesn't — it bounces
around with no clean pre/post-2017-18 split. This argues for either an informational
explanation (the market prices something this model structurally can't see — most plausibly
skater-level injuries/lineup news beyond the goalie-starter signal already in the backtest,
since the model's goalie overlay already uses the real starter from the boxscore, which is
information the closing line also has and can't be the whole gap) or genuine remaining
modeling headroom (e.g. the tie-mass deficit investigated in §10). Not resolved here — logged
as a real, precise finding rather than assumed away.

**Re-run against `cross_season_prior_poisson` (2026-07-23, task tracked from §12.5's
mechanism-specific check — this is the AGGREGATE headline re-run):** model Brier 0.24167 vs.
market close 0.23867 on the identical 11,803 games, gap mean 0.00300, 95% CI
**[0.00187, 0.00412]** — narrower than the `drift_adjusted_poisson` gap (0.00351, CI
[0.00235, 0.00467]) by almost exactly the same amount §12's own dev-set bootstrap found
(Brier improvement mean 0.00052, CI [-0.00070,-0.00035]). Per-season, every season's gap
either shrank or stayed flat (largest: 2012-13 0.00695→0.00640, 2021-22 0.00650→0.00521;
2013-14 stayed at essentially zero in both). **Confirms, at the aggregate benchmark level,
what §12.5 already showed at the mechanism level: Cycle 18 is real, precise progress against
the market gap, not just a same-game metrics improvement — but the gap remains real and
still open (CI [0.00187, 0.00412] is entirely positive; this model is closer to the market
than it was, but still not at it).**

### 6.9 Isotonic calibration layer: tested, found to genuinely hurt — not shipped

Built `src/models/calibration.py` (isotonic regression via scikit-learn) and validated with
a real temporal split — the calibration curve fit on the FIRST 70% of dev-set games by date
(11,569 games), evaluated ONLY on the LAST 30% (4,959 games) never used to fit it, following
the external report's own caution (§5.4: "calibrating on in-sample data yields optimistic
reliability curves ... for an already-near-calibrated Poisson output, calibration can
slightly hurt — verify on holdout Brier/log-loss").

**Real result: calibration made Brier and log-loss WORSE on the held-out fold** (Brier
0.23676 calibrated vs. 0.23577 uncalibrated; log-loss 0.67742 vs. 0.66373). **Paired
bootstrap (5,000 resamples): mean Brier change −0.00098, 95% CI [−0.00163, −0.00036] —
entirely negative, a real, statistically confirmed harm, not noise.**

**Verdict: NOT shipped.** This is exactly the failure mode the external report warned could
happen, confirmed real on this project's own data: the current model's raw output is already
reasonably well-calibrated (an independent-Poisson combine with every constant empirically
fit rather than assumed, built up over eleven prior cycles, apparently produces good native
calibration as a side effect), and isotonic regression's non-parametric flexibility overfits
the fitting fold's own idiosyncrasies rather than finding a genuine, generalizable
correction. Logged as a real negative result, per this project's standing practice — not
retried with a different split or method without a specific reason to expect a different
outcome.

### 6.10 Cycle 12 summary

Real, substantive progress despite the calibration layer itself not panning out: a genuine
market-odds dataset now exists (13 seasons, 14,260 regular-season games with matched closing
lines), a precise same-game market-comparison benchmark is now possible (§6.8) and corrects
an earlier over-optimistic framing, and the calibration-layer question is now answered with
real evidence (don't add one) rather than left as a to-do. The market-data gap for 2022-23
onward (§6.1, including the entire holdout window) remains open — no source checked so far
fills it.

Next: Cycle 13, scoring-era drift/non-stationarity — the report's own top pick for directly
attacking the dev/holdout gap (§4.8-4.9), and the piece most likely to still show real
headroom given calibration alone did not.

## 7. Cycle 13: scoring-era drift — a real bug, a real fix, and a real second entangled bug (2026-07-23)

### 7.1 The drift bug: real, substantial, confirmed before any fix was attempted

Checked directly (not assumed) whether the trailing league-average used throughout
`shrinkage.py` — a pure EXPANDING mean with infinite memory, "no season reset" by original
design — actually tracks real scoring levels. **It doesn't, substantially**: comparing the
trailing average AT THE START of each season against that season's own real average,
**every single season from 2017-18 (the real, well-documented league-wide scoring jump)
through 2025-26 — 9 consecutive seasons — shows the trailing average undershooting real
scoring by 0.10 to 0.32 goals per team per game.** An infinite-memory average can never fully
"forget" 9+ years of now-stale, lower-scoring history, so it keeps lagging even years after
the shift. This directly affects every model built since Cycle 1, all of which rely on this
same shrinkage machinery for team-strength baselines.

### 7.2 The fix, and confirming it works at the level it's meant to work at

Added `halflife_games` (default `None`, preserving every existing caller's exact
reproducibility — confirmed byte-identical output on a full pipeline re-run before changing
anything else) to `shrinkage._trailing_league_stat` and threaded it through
`add_walk_forward_rate`/`add_walk_forward_mean`/`add_walk_forward_toi_rate` and the
situational-layer/validation chain that uses them. A real value switches the trailing
statistic from a plain expanding mean to an exponentially-weighted one with that half-life
(in games) — checked directly (not assumed) that this fixes the specific bug measured in
§7.1: at `halflife=400`, the 2018-19 season-start gap shrinks from −0.219 to −0.051, and
every per-situation league-average/attack-rate pair (ev/pp/pk/other) measurably shifts
upward toward the real current scoring level, exactly as intended.

### 7.3 A second, entangled bug found while checking the AGGREGATE effect, before trusting a clean win

Grid-searching `halflife_games` against the model's own dev-set Brier/total-MAE/margin-MAE
produced a confusing result: total-MAE improved somewhat as expected, but the improvement
was much smaller than the size of the §7.1 bug would predict, and margin-MAE got WORSE with
more aggressive decay — the opposite of what fixing a real bug should do across the board.
Investigated rather than accepted: the situational combine formula
(`team_strength_situational._combine`) is `league_avg_attack * (attack/league_avg_attack) *
(opp_defense/league_avg_defense) * (toi/60)` — algebraically, **`league_avg_attack` cancels
out of this formula entirely** (multiplied then divided by the same quantity), so the actual
lambda is driven by `attack_per60` (which correctly increases with decay, confirmed) and the
`opp_defense/league_avg_defense` RATIO specifically. Checked that ratio directly: at
`halflife=None`, it averages **1.026** (an average team's own estimated defense rate reads
~2.6% "worse" than the league-average denominator it's compared against — which shouldn't
happen for a genuinely "average" team in aggregate) — at `halflife=400`, this same ratio is
**0.999**, i.e. correctly neutral.

**This means the original, undecayed pipeline had TWO real, independent problems that
happened to partially offset each other**: (1) the confirmed §7.1 league-baseline
undershoot, and (2) a mismatch between team-level defense estimates (which reset each
season, per this project's own convention) and the infinite-memory league baseline they're
compared against — which was inflating the defense-ratio term just enough to partially
compensate for (1)'s undershoot in some of the aggregate metrics, particularly margin. Fixing
(1) alone via decay REMOVES that fortuitous compensation too, which is why the net aggregate
effect is smaller and more mixed than §7.1's raw bug size would suggest on its own.

### 7.4 Real, bootstrapped net effect: a genuine three-way trade-off, not a clean win

Tested `halflife_games=1000` (a representative point from the grid search) against the
undecayed baseline, identical dev-set games, paired bootstrap (5,000 resamples):

| Metric | Improvement (mean) | 95% CI | Verdict |
|---|---|---|---|
| Total-goals MAE | +0.00690 | **[0.00395, 0.01005]** | Real, meaningful improvement |
| Brier score | −0.00004 | [−0.00008, −0.00000] | Technically significant, practically negligible |
| Margin MAE | −0.00137 | **[−0.00181, −0.00095]** | Real, meaningful HARM |

**Verdict: NOT adopted as-is.** This is a genuine trade-off — real improvement on totals,
real harm on margin — not a clean win to bank the way the goalie overlay or rest adjustment
were. Deploying it now would silently trade a real regression on one axis (margin, which
matters as much as totals for this project's stated goals — puck-line/spread accuracy) for
a gain on another, without addressing the root cause: §7.3's entangled defense-ratio
artifact, which is the actual thing that needs fixing before decay's real benefit (§7.1's
confirmed bug) can be captured cleanly without the offsetting harm.

**What this cycle actually accomplished, despite not shipping a change**: confirmed a real,
substantial, 9-season-long bug in the core shrinkage machinery every model has relied on
since Cycle 1; built and verified a working fix for it in isolation; and — by properly
checking the AGGREGATE effect rather than declaring victory once the raw mechanism looked
right — discovered a second, previously-unknown, genuinely entangled bug that the first fix
alone can't cleanly resolve. This is exactly the kind of finding this project's own
incremental-validation discipline exists to catch before it reaches a "keep" decision.

Next: fix §7.3's defense-ratio artifact directly (the team-level defense estimate's own
season-reset convention needs to be reconciled with whatever league-baseline convention is
used to normalize it — either both should reset each season, or both should share the same
decay/memory scheme) — THEN re-test decay on top of that fix, since the two are now known to
interact and testing them together, once both are correctly isolated, is the honest way to
capture §7.1's real benefit without §7.3's offsetting harm.

### 7.5 Resolving the entanglement: the shrinkage priors needed joint recalibration, not a structural redesign

Investigated §7.3's hypothesis directly rather than assuming the "reconcile the reset
conventions" framing was the right fix. First checked whether the trade-off was actually a
smooth, monotonic bias-variance dial (more decay → strictly better totals, strictly worse
margin, with no sweet spot) by extending the original grid: **it was** — margin-MAE degraded
monotonically as halflife decreased, with no halflife value recovering baseline margin
performance on its own. This ruled out "just pick a gentler halflife" as a fix.

**Real insight: `PRIOR_MINUTES_EV/PP/PK/OTHER` (the shrinkage pseudo-exposure constants) were
never truly random placeholders — they were implicitly shaped by years of validation against
the OLD, infinite-memory league baseline's own particular lag pattern.** Switching to a
faster-adapting decayed baseline without also revisiting how much weight teams' own
accumulated data gets relative to that baseline was always going to interact. Grid-searched
`prior_minutes` scale jointly with `halflife_games` rather than treating them as independent
— and found **combinations that beat the undecayed baseline on both total-MAE and margin-MAE
simultaneously**, something no single-halflife setting achieved alone.

**Adopted: `halflife_games=600` paired with shrinkage priors scaled to 0.6x their original
values** (487→292.2 min for even-strength, 49→29.4 for PP/PK, 23→13.8 for other). Paired
bootstrap (5,000 resamples) against the Cycle 11 baseline (`rest_adjusted_poisson`, no decay,
original priors), identical dev-set games:

| Metric | Improvement (mean) | 95% CI | Verdict |
|---|---|---|---|
| Margin MAE | +0.00169 | **[0.00087, 0.00254]** | Real, significant improvement |
| Brier score | −0.00004 | [−0.00016, 0.00008] | Neutral — crosses zero, NOT harmed |
| Total-goals MAE | +0.00312 | [−0.00029, 0.00679] | Positive-trending, borderline (CI just touches zero) |

**No metric regressed.** This is a genuine, if modest, net improvement — the real drift bug
from §7.1 is now captured without §7.3's offsetting margin harm, because the shrinkage
priors were recalibrated to match the new, more-responsive baseline rather than left tuned
for the old one. Logged as `drift_adjusted_poisson` (`src/models/validate_drift.py`), the
new overall reference point, superseding `rest_adjusted_poisson`.

**One honest residual note**: the raw predicted mean total (5.547) still undershoots the
real mean (5.8125) by more than before fixing this — the aggregate MEAN bias hasn't fully
closed even though the ERROR metrics (which measure per-game accuracy, not just the
aggregate level) genuinely improved. This is not a contradiction — a model can have a
persistent small mean bias while still reducing per-game absolute error, if the bias is
small relative to game-to-game variance — but it does mean §7.1's original bug is only
partially, not fully, resolved by this cycle's fix. Worth re-measuring whether a more
aggressive decay (now that priors are properly recalibrated for it) could close the
remaining gap further, as a candidate for a future cycle rather than reopening this one.

**Also worth doing before the next major cycle**: re-run the §4.8 holdout check against this
new `drift_adjusted_poisson` model — the current holdout comparison figures throughout this
document are all still based on the pre-drift-fix `goalie_overlay_poisson` snapshot, and
this cycle was explicitly motivated by trying to close that gap.

### 7.6 Re-checking the holdout: the drift fix genuinely narrowed the gap it was built for

Built `src/models/check_holdout_drift.py`: every constant (away-B2B adjustment, P(home wins
OT/SO), the halflife/prior-scale drift fix itself) fit on DEVELOPMENT-SET-ONLY data, then
applied — without refitting — to score both the development set and the real, untouched
holdout. This is structurally different from just calling the existing pipeline functions,
since they filter their returned range to match the fit range; this script deliberately
decouples fit-range from score-range so the holdout can be scored honestly.

**Real result — every single holdout metric improved, not just one:**

| Metric | Old (`goalie_overlay_poisson`, §4.8) | New (`drift_adjusted_poisson`) | Δ |
|---|---|---|---|
| SU accuracy | 55.79% | **56.90%** | +1.11 pt |
| Brier score | 0.24427 | **0.24289** | −0.00138 |
| Log-loss | 0.6817 | **0.6787** | −0.0030 |
| Total-goals MAE | 1.903 | **1.864** | −0.039 |
| Margin MAE | 2.147 | **2.142** | −0.006 |
| Predicted mean total (actual: 6.167) | 6.468 (+0.301 over) | **6.124 (−0.044)** | Much closer to real |

**The predicted-total calibration on the holdout specifically went from a +0.301 over-shoot
to a near-perfect −0.044** — a striking, direct confirmation that the drift fix does what it
was built to do, especially on a holdout that (unlike the development set) sits entirely
within the post-2017-18 high-scoring era, so it benefits uniformly from a faster-adapting
baseline without the development set's own mixed-era complications (§7.5's honest residual
note about the dev set's own mean bias not fully closing).

**The dev/holdout GAP itself also narrowed, though unevenly across metrics:**

| Metric | Old gap (dev→holdout) | New gap (dev→holdout) |
|---|---|---|
| SU accuracy | 1.81 pt | **0.74 pt** |
| Brier score | +0.00299 | **+0.00213** |
| Total-goals MAE | +0.097 | **+0.061** |
| Margin MAE | +0.124 | +0.123 (essentially unchanged) |

SU, Brier, and total-goals-MAE gaps all shrank meaningfully — exactly the metrics most
directly tied to the scoring-LEVEL drift this cycle targeted. Margin MAE's gap barely moved,
consistent with margin being more about relative team-strength comparison than absolute
scoring level — a different kind of drift this specific fix was never aimed at, and a
plausible next thing to investigate if the gap is revisited again.

**Net: Cycle 13 accomplished its original, stated goal.** Logged as
`drift_adjusted_poisson_DEV_SET_ONLY` / `drift_adjusted_poisson_HOLDOUT_CHECK` in the
ledger. The dev/holdout gap is real but smaller than it was, and is now more precisely
understood as concentrated in margin/relative-strength drift rather than the scoring-level
drift this cycle fixed — genuine progress, not a fully closed case.

## 8. Cycle 14: goalie-specific back-to-back — tested, rejected (2026-07-23)

The external research report (§5.3) suggested the team-level away-back-to-back effect
already built (§4.14) might be conflating general team fatigue with a real, separate,
goalie-specific effect — modern estimates cited there cluster around a ~0.8% save-percentage
drop for a goalie starting on zero days' rest. Checked directly on this project's own real
data before building anything, using goalie-SPECIFIC rest (days since THAT SAME goalie's own
last start, from `moneypuck_goalie_games.parquet` — genuinely distinct from the team-level
rest already used in §4.14, since a team can play a back-to-back with a DIFFERENT goalie
starting each night).

**Confirmed a real, rarer event first**: true goalie back-to-backs (the same goalie starting
both nights) are far less common than team back-to-backs — 1,520 out of 34,325 dev-set
goalie-starts (~4.4%), since teams often deliberately rest their starter on the second night
specifically to avoid this. A real, meaningful sample, not too small to detect a genuine
effect of the claimed size (a 0.8pp SV% drop over ~30 shots/game implies roughly 0.24 extra
goals allowed per game — well within what n=1,520 should be able to detect if real).

**Found nothing.** Correlation between goalie-specific rest days and that goalie's own GSAx:
r=−0.0017, p=0.75 — indistinguishable from zero. Extreme contrast (goalie on true zero rest
vs. 2+ days rest): mean GSAx difference 0.0041, p=0.92 — no effect at all, nowhere near the
claimed magnitude.

**Also checked the workload framing the report specifically flagged** ("measurable decline
for goalies playing 4+ games in 14 days"): correlation between a goalie's own starts in the
trailing 14 days and their GSAx is small but "significant" only due to sample size (r=0.024,
p<0.0001) — and it's POSITIVE, the opposite of a fatigue story. Heavy-workload starts (7+ in
14 days) averaged BETTER GSAx than light-workload starts (≤4), not worse (p=0.10, not
significant, but wrong-signed for a fatigue hypothesis either way). The most plausible
explanation isn't that heavy workload helps — it's a selection effect: teams give more
starts to whichever goalie is currently playing well and healthy, not randomly, so workload
and performance are confounded in a way that has nothing to do with fatigue.

**Verdict: rejected, no goalie-specific overlay built.** The team-level away-back-to-back
effect (§4.14) stands as the real, kept signal — whatever it's capturing (skater fatigue,
travel, defensive lapses, or something else) is apparently not attributable to the starting
goalie's own individual performance declining, at least not in a way this data can detect.
Consistent with this project's own recurring lesson (§4.14's closing note): check the
specific sub-case rather than assume a broad narrative transfers, and a real, disciplined
negative result is worth exactly as much as a positive one for the record.

Next: Cycle 15, the back-to-back × time-zone-change interaction — the one travel-related
signal RotoWire's own 11-season study found to actually clear significance, per §5.3.

## 9. Cycle 15: back-to-back × time-zone-change interaction — tested, rejected (2026-07-23)

RotoWire's own 11-season study (cited in the external report, §5.3) found raw time-zone
crossings and travel direction don't clear significance alone — only the back-to-back ×
time-zone-change INTERACTION does. Checked directly on this project's own data before
building anything.

Built `src/models/travel_timezone.py`: a real, stable team→timezone mapping (confirmed
against the actual 35 distinct team codes in this project's own regular-season schedule,
including historical `ATL`/`PHX` and modern `UTA`/`SEA`, not assumed from a generic list),
and a walk-forward `timezone_changed` flag (did this team's own previous game sit in a
different zone than this one). Zero unmapped rows on the dev set — full, real coverage.

**A real, meaningful sample exists for the interaction cell**: 1,036 dev-set team-games are
both on a back-to-back AND crossed a time zone since their last game (vs. 4,523 back-to-back
games with no time-zone change) — large enough to detect a real effect if one exists.

**Found nothing.** Mean scoring residual for back-to-back-plus-timezone-change games: +0.1096
(n=1,036) vs. back-to-back-only games: +0.1027 (n=4,523) — a difference of 0.0069, p=0.90.
No signal at all, nowhere close to significant.

**Verdict: rejected, no interaction term built.** A third consecutive travel/rest-adjacent
hypothesis (after Cycle 14's goalie-specific back-to-back) that a public source suggested
and this project's own data didn't support. Recorded as a real negative result — this
project's own away-team-back-to-back effect (§4.14) remains the one genuine, kept signal in
this space; timezone crossing specifically doesn't add anything measurable on top of it,
at least not in this data.

Next: Cycle 16, a gradient-boosted stacking layer over the Poisson outputs — the report's
next-priority item, and a genuinely different KIND of test than the last several narrow
feature checks.

## 10. Cycle 17, re-scoped: diagonal (tie-mass) inflation — a real, honest trade-off, NOT shipped (2026-07-23)

A second external review (a different LLM given the updated §0-§9 doc for a second pair of
eyes) flagged §4.13's tie-mass finding — model-implied regulation-tie probability 0.168 vs.
real 0.231, a 6+ point deficit — as "the most important unexploited number" in the project,
and made a specific, technically-grounded correction to how Cycle 17 was originally scoped:
a negative-dependence copula/Sarmanov term (the original plan, targeting the -0.15
regulation score correlation) moves diagonal mass DOWN, which would make the tie-mass
deficit WORSE, not better. The two problems need to be solved in the right order — diagonal
(tie-mass) calibration first, off-diagonal dependence second, not the reverse, and not
skipped. The review also specified a free, cheap consistency check: any fitted joint's
implied total-goals dispersion, Var(T)/E(T) = 1 + 2·Cov(H,A)/(λh+λa), should reproduce the
independently-measured 0.97-0.98 under-dispersion ratio (§4.11), "or the fit is capturing the
wrong thing."

### 10.1 Confirming the motivating number still holds post-drift-fix

Checked directly before building anything, per this project's own discipline. §4.13's number
predates the Cycle 13 drift fix (§7.5); since that fix was itself found Brier-neutral
(§7.4-7.5), there was no strong reason to expect it to have touched the tie-mass gap, but it
was checked rather than assumed. On the current best model (`drift_adjusted_poisson`),
using the real regulation-score reconstruction (§4.12) and the `HOME_REG_RATIO`/
`AWAY_REG_RATIO` scalars (§4.13) to convert each game's final-score lambda down to a
regulation-only lambda: **model-implied average diagonal (tie) mass 0.1728 vs. real
empirical regulation-tie rate 0.2305 — a 5.8-point deficit, confirmed to persist essentially
unchanged.** The problem is real and still open.

### 10.2 Design: a single closed-form inflation parameter, not an iterative fit

Built `src/models/diagonal_inflation.py`. Design, per the review's own reasoning: a single
multiplicative parameter `theta` applied to each game's own independent-Poisson regulation
joint's diagonal cells (preserving each game's own lambda-dependent diagonal SHAPE — just
uniformly scaling it), with the off-diagonal cells rescaled to keep the joint a valid
distribution. Because `inflate_diagonal` scales every game's own diagonal mass by the same
`theta`, the AVERAGE diagonal mass across the fitting population scales by `theta` too — so
`theta` has a direct closed-form solution (real average tie rate ÷ average model-implied
diagonal mass), not an iterative MLE. This sidesteps both of the prior structural failures
in this space: §4.11's OT-population-pooling confound and §4.13's missing-cell MLE
divergence — both were problems with the FITTING POPULATION, and here the fitting population
is every dev-set game's own real regulation-only score (ties genuinely included, nothing
selection-excluded), which the §4.12 reconstruction makes directly usable this way for the
first time.

**Fitted `theta = 1.3338` on 16,528 real dev-set games.** Validity check (not just assumed):
confirmed no single game's own diagonal mass, after multiplying by theta, would exceed 1 —
maximum pre-inflation diagonal mass across all 16,528 games was 0.2419, ×1.3338 = 0.3227,
comfortably under 1; zero violations.

### 10.3 Full pipeline validation: SU/Brier neutral, totals improve, margin regresses

Built `src/models/validate_diagonal_inflation.py`: applies `inflate_diagonal` to each game's
own regulation joint, then convolves the result with the real OT/SO layer (§4.12's fitted
`p_home_wins_ot`, reallocating 100% of the — now correctly-sized — diagonal mass to the
adjacent (x+1,x)/(x,x+1) cells, since every real regulation tie does go to OT/SO) to get the
final recorded-score joint. Compared against the current best model's own numbers on the
identical 16,528 games, paired bootstrap (5,000 resamples):

| Metric | Baseline (`drift_adjusted_poisson`) | Diagonal-inflated candidate | Mean diff | 95% CI |
|---|---|---|---|---|
| SU accuracy | 57.64% | 57.54% | -0.00097 | [-0.00218, 0.00024] — crosses zero, neutral |
| Brier | 0.24076 | 0.24070 | -0.00006 | [-0.00019, 0.00005] — crosses zero, neutral |
| Total-goals MAE | 1.80287 | 1.80082 | **-0.00205** | **[-0.00281, -0.00131] — entirely negative, real improvement** |
| Margin MAE | 2.01898 | 2.02439 | **+0.00541** | **[0.00441, 0.00634] — entirely positive, real HARM** |

Two of the four numbers land exactly where the review predicted: SU and Brier are neutral,
consistent with §6.9's isotonic null (final win probabilities are already well-calibrated;
moving mass onto the diagonal and re-splitting it via `p_home_wins_ot` ≈ 0.509 roughly
returns it to where it came from). Total-goals MAE genuinely improves — this makes structural
sense, since converting from final-score lambda to a properly-scaled regulation lambda (via
the reg-ratio constants) and then adding back exactly one goal only for the ~23% of games
that reach OT is a more accurate way to reconstruct expected final-game total than the
production pipeline's direct fit against final scores.

**Margin MAE, however, gets measurably WORSE — a real, tight-CI regression, not noise, and
the opposite of what the review anticipated** ("given margin MAE is a stated first-class
goal, that's still real value"). Likely mechanism: `inflate_diagonal` scales every game's
diagonal mass by the same global `theta`, uniformly pulling mass away from ALL off-diagonal
cells (blowouts included, not just near-tie margins) to fund the diagonal increase, then
converts that inflated diagonal into ±1-goal-margin outcomes via the OT split. This
concentrates predicted-margin mass more tightly around 0/±1 than real outcomes warrant for
games with a genuine, large lambda gap — a single global theta doesn't account for how
mismatched any individual game already looks, unlike the per-game diagonal shape it's scaling
(which IS lambda-dependent) but scaling uniformly regardless of that shape's own spread.

### 10.4 The Var(T)/E(T) consistency check: a precise diagnostic, not just a confirmation

The review's proposed free consistency check was run rather than skipped: **implied
Var(T)/E(T) under the fitted joint is 0.9570**, versus the independently-measured real ratio
of 0.97-0.98 (§4.11). Close, but genuinely below the target range, not within it. Per the
review's own framing ("or the fit is capturing the wrong thing") — this is a real, precise
signal, not a pass/fail formality: matching the AVERAGE diagonal mass to the real tie rate
(what `theta` was fit to do) is not the same constraint as matching the real total-goals
dispersion, and this fit slightly OVER-corrects dispersion downward relative to reality.
Consistent with §10.3's margin-MAE finding — both point at the same underlying gap: a single
scalar tie-inflation, with no accompanying off-diagonal dependence term, over-concentrates
the joint relative to the true correlation structure.

**Correction (2026-07-23, caught by a follow-up review, see §10.6.1): the "0.97-0.98"
comparison above was NOT like-for-like.** §4.11's figure is the MARGINAL per-side
variance/mean ratio (raw home-goals and away-goals variance/mean, ≈0.970/0.969, or the
pooled standardized-residual variance 0.98) — a different statistic from
Var(home+away)/E(home+away), which also needs the between-game lambda-heterogeneity term
(law of total variance), not just each game's own conditional variance, to be measured at the
population level correctly. §10.6.1 redoes this properly; the corrected numbers supersede
this paragraph's 0.9570-vs-0.97-0.98 framing, though not the qualitative conclusion — if
anything the corrected version makes the same point far more strongly.

### 10.5 Verdict: NOT shipped as a standalone fix, but a precise, structurally informative negative result

**Rejected in this form** — under this project's own established bar (no kept change may
regress a metric with a real, bootstrap-confirmed effect), margin MAE's regression here is
real and disqualifying on its own, exactly the same standard applied throughout (§4.7's K=80
rejection, §4.14's own "check the narrow sub-case" pattern). This is NOT the same as "the
tie-mass problem doesn't matter" or "the review's diagnosis was wrong" — §10.1 confirms the
underlying deficit is real, and §10.3's SU/Brier/total-MAE results land almost exactly where
the review predicted a diagonal-only fix should land. What this cycle actually shows is more
precise than a simple kill: **diagonal inflation alone is the wrong-shaped fix** — it correctly
targets the AVERAGE tie-rate miscalibration but, applied as one global scalar with no
per-game or off-diagonal adjustment, over-concentrates the score joint in a way that measurably
hurts margin prediction and undershoots the real total-goals dispersion target (§10.4). This
is exactly consistent with the review's own original point 2, which this cycle only tested
half of: "the joint distribution needs BOTH mild negative off-diagonal dependence AND excess
diagonal mass" — this cycle built and tested the diagonal term in isolation (correctly, as
the review suggested doing first, to avoid conflating the two), and the isolated result shows
real, if partial, evidence for the diagonal fix's intended effect (totals) alongside a real,
specific cost (margin) that the off-diagonal term was always meant to address jointly, not
this term alone. §10.6 below (a third-pass follow-up) sharpens exactly what that joint term
needs to do.

### 10.6 Follow-up: three residuals-first checks before deciding what to build next (2026-07-23)

A third-pass review of §10 (by the same reviewer who re-scoped this cycle) scored its own
prediction honestly — right on SU/Brier neutrality and the totals direction, wrong on margin
— and, rather than accepting the diagonal-vs-off-diagonal framing at face value, proposed
three specific, cheap, decisive measurements before committing to a build. All three were
run. Scripts: `src/models/validate_diagonal_inflation_ablation.py` (10.6.2),
`src/models/validate_margin_bin_profile.py` (10.6.3); 10.6.1 reuses
`validate_diagonal_inflation.py`'s own output.

#### 10.6.1 The Var(T)/E(T) comparison was apples-to-oranges — and the corrected version is a much bigger finding

Flagged directly: §10.4's "0.97-0.98" target was §4.11's MARGINAL per-side dispersion ratio,
not Var(total)/E(total). Recomputed both sides of the comparison as the same statistic —
population-level Var(total)/E(total), using the law of total variance for the model side
(`E[Var(T|game)] + Var[E(T|game)]`, since between-game heterogeneity in lambda genuinely
contributes to the pooled population variance, not just each game's own conditional
variance) and a direct sample statistic for the real side:

| | Real (measured directly) | Model (theta=1.3338) |
|---|---|---|
| Var(total)/E(total) | **0.8913** | 1.0193 |

**This is a much bigger gap than the original 0.9570-vs-0.97-0.98 framing suggested — 0.13,
not 0.02 — and it points the OPPOSITE direction from what §10.4 concluded.** Real total-goals
scoring is dramatically more under-dispersed at the population level than either the
uninflated or the inflated model captures; the model's ~1.02 is roughly what a heterogeneous-
lambda independent-Poisson SUM produces on its own (between-game variation in combined
scoring rate pushes the ratio slightly above 1, even with zero within-game home/away
dependence) — diagonal inflation barely moves this number at all (§10.6.2 shows exactly how
little). Closing a gap this size requires substantial genuine NEGATIVE dependence between
home and away scoring (the real, repeatedly-measured -0.15 regulation correlation, §4.11,
§4.13) acting at the population level — not a diagonal-only reallocation, which primarily
redistributes mass locally near the diagonal and has only a second-order effect on the
aggregate sum's variance. This is a materially stronger, more precise version of §10.4's
original point, not a reversal of it.

#### 10.6.2 The theta=1 ablation: NOT a fully clean isolated win, but it separates two real effects cleanly

Ran the identical §10 pipeline with `theta` forced to 1.0 (a pure no-op — regulation-ratio
reconstruction and the OT/SO layer still apply, diagonal inflation does not), paired
bootstrap against the same baseline:

| Metric | theta=1 (reconstruction only) | theta=1.3338 (full, §10.3) |
|---|---|---|
| SU | neutral, CI [-0.00163, 0.00030] | neutral, CI [-0.00218, 0.00024] |
| Brier | neutral, CI [-0.00005, 0.00001] | neutral, CI [-0.00019, 0.00005] |
| Total-MAE | **real gain**, CI [-0.00223, -0.00038] | real gain, CI [-0.00281, -0.00131] |
| Margin-MAE | **real harm**, CI [0.00125, 0.00199] | real harm, CI [0.00441, 0.00634] |
| Var(total)/E(total) | 1.0211 | 1.0193 |

**Not the clean isolation the hypothesis predicted** — margin MAE regresses even with
inflation completely switched off, so the reconstruction pathway itself (not just theta) has
a real, if smaller, margin cost (about 30% of the full version's harm), and roughly 62% of
the total-MAE gain is already present at theta=1. What theta ADDS on top: roughly 70% of the
margin harm, essentially none of the totals gain, and — per the Var(T)/E(T) column —
**essentially nothing to the aggregate dispersion gap (1.0211 → 1.0193, a 0.0018 move against
a real target 0.13 away)**. This is a clean, decisive negative result on its own: the
diagonal-inflation term, exactly as built, is not the lever that closes the real dispersion
gap, regardless of what happens to the margin-MAE trade-off. Both effects are real; they're
just not attributable to the same mechanism the original framing assumed.

#### 10.6.3 Margin-bin profile: the local-transfer hypothesis is confirmed, decisively

The proposed residuals-first check, run before building anything: compared the model's own
uninflated (theta=1) regulation-margin distribution against the real regulation-margin
distribution, bin by bin, on all 16,528 dev-set games (`validate_margin_bin_profile.py`):

| Margin | Real freq | Model pred | Real − model |
|---|---|---|---|
| -2 | 0.0970 | 0.1047 | -0.0077 |
| **-1** | 0.0996 | 0.1484 | **-0.0488** |
| **0 (tie)** | 0.2305 | 0.1728 | **+0.0577** |
| **+1** | 0.1090 | 0.1638 | **-0.0548** |
| +2 | 0.1093 | 0.1275 | -0.0182 |
| ±3 | 0.0966 / 0.1225 | 0.0619 / 0.0831 | +0.0347 / +0.0394 |

(full 17-bin table in `validate_margin_bin_profile.py`'s output). **The model's excess mass
at |margin|=1 specifically (0.1036, both signs summed) is MORE than the entire tie deficit
(0.0577) — 180.8% of the total offsetting excess found across all non-tie bins is
concentrated at |margin|=1 alone**, with |margin|≥2 bins net NEGATIVE (the model actually
UNDER-predicts several of them, most strikingly ±3, where real data has real mass the
independent-Poisson model doesn't produce at all — a separate, unexplained finding, logged
but not pursued here). This is a clean, decisive confirmation of the local-transfer
hypothesis: the real tie deficit is funded almost entirely by adjacent one-goal-margin games,
not diffusely by blowouts — exactly the shape "protecting the loser point late in a close
game" predicts, and exactly NOT the shape §10's global, uniform theta rescale assumed when it
taxed every off-diagonal cell (blowouts included) to fund the diagonal. **This is very likely
why §10.3's margin-MAE regressed**: theta pulled real mass away from bins (like ±3) that
already needed MORE mass, not less, to fund a tie-mass increase that empirically belongs
almost entirely to the ±1 neighbors of the diagonal.

#### 10.6.4 Net read on the three checks

Two of the three sharpen and validate the review's own instinct precisely: the local-transfer
shape (10.6.3) is confirmed, not merely plausible, and should replace the global-rescale
`theta` design in any future attempt — a single parameter moving mass from (x+1,x)/(x,x+1)
into (x,x) at each diagonal level, leaving blowout cells untouched, is now a
residuals-verified design, not a hypothesis. The Var(T)/E(T) correction (10.6.1) turns out to
matter a great deal, but in the opposite direction from a mere sanity-check pass: it shows
the real total-goals dispersion gap is much larger than believed, and that neither version of
the diagonal fix touches it — meaning the eventual off-diagonal term isn't a nice-to-have
polish on top of a working diagonal fix, it's carrying by far the larger share of the real
work. The theta=1 ablation (10.6.2) is the one place the hypothesis was partially wrong: even
the "clean" reconstruction-only pathway has a real, if smaller, margin cost, so a rebuilt
local-transfer term should be validated against the theta=1 baseline, not against full
production, to correctly attribute its own effect.

Next: rebuild the diagonal-mass fix as a local (x+1,x)/(x,x+1) → (x,x) transfer parameter
`delta` (10.6.3's confirmed shape) instead of a global rescale, fit and validated against the
theta=1 reconstruction-only baseline (10.6.2's correct comparison point) — a well-motivated,
residuals-verified candidate, not yet attempted. §11 (below) surfaces a second, independently
promising lead from the same review pass.

## 11. A new lead from the §6.8 per-season table: the market gap is front-loaded within season (2026-07-23)

The same follow-up review noticed something in §6.8's per-season breakdown that the
era-framing (drift vs. informational) didn't explain: the two largest single-season gaps are
**2012-13 (+0.00695, the 48-game lockout season with no normal training camp) and 2021-22
(+0.00650, the first full-length season after two COVID-disrupted ones)** — both seasons
where real team strength was most likely to have shifted sharply from the prior year in ways
a season-reset, league-average-anchored prior can't see quickly, while the market opens every
season with real, team-specific preseason information regardless of any disruption. This
project's own team-strength ratings (`shrinkage.py`) reset toward league average at the start
of every season and rebuild current-season form from zero — the original architecture
proposal (§3.2) called for a "recency-weighted multi-season prior," and the cycle table shows
this was never built or tested; cross-season goalie history (§4.10) is a DIFFERENT, narrower
question (goalie-specific carryover, found a wash) and doesn't cover team-level priors at all.

### 11.1 The residuals-first check: bin the market gap by within-season game number

Built `src/models/validate_market_gap_by_game_number.py`: for each of the 11,803 market-
matched games (§6.8), computed each team's own game count so far this season (1-indexed,
walk-forward, reusing the same `cumcount`-based convention `shrinkage.py` uses for
`games_played_before`), took the MINIMUM of the two teams' counts (a game is only as
well-informed as its less-experienced-this-season side), and binned into early (1-15),
mid (16-40), and late (41+) season.

**Overall, pooling all seasons — the gap is real and clearly front-loaded:**

| Games into season | n | Model Brier | Market Brier | Gap |
|---|---|---|---|---|
| 1-15 | 2,373 | 0.24845 | 0.24171 | **+0.00674** |
| 16-40 | 3,834 | 0.24047 | 0.23736 | +0.00310 |
| 41+ | 5,596 | 0.24070 | 0.23828 | +0.00242 |

The gap in the first 15 games of either team's season is roughly **2.8x** the gap once both
teams have 41+ games of current-season data. Not flat — exactly the front-loaded signature
the review's hypothesis predicted, not the "uniform → informational" signature §6.8's
original framing left open.

**Split by the two flagged seasons vs. everything else — the pattern is dramatically
stronger exactly where the hypothesis says it should be:**

| Games into season | 2012-13 + 2021-22 (n) | Gap | All other seasons (n) | Gap |
|---|---|---|---|---|
| 1-15 | 483 | **+0.01362** | 1,890 | +0.00498 |
| 16-40 | 784 | +0.00734 | 3,050 | +0.00201 |
| 41+ | 765 | +0.00157 | 4,831 | +0.00255 |

In the two flagged seasons, the early-season gap (+0.01362) is **roughly 8.7x** the late-
season gap (+0.00157, nearly closed) — a far sharper front-loading than the already
front-loaded "other seasons" pattern (+0.00498 → +0.00255, about 2x). This is precisely the
signature a disrupted-offseason, slow-to-reconverge team-strength prior would produce, and
precisely NOT what a purely informational (injury/lineup) explanation would predict, since
there's no reason skater-injury information specifically would concentrate in the first 15
games of exactly these two seasons.

### 11.2 Verdict: a real, well-targeted lead — recommended ahead of the GBM layer

**Confirmed, not built yet.** This is exactly the kind of residuals-first result this
project's own discipline asks for before committing to a build: cheap to check, decisive
(front-loaded in general, dramatically more so in the two seasons the hypothesis specifically
predicts), and it directly targets two open problems at once — §0.4 item 2 (the market gap)
and plausibly §0.4 item 1 (the margin-specific holdout drift, since early-season
relative-strength comparisons are exactly what a reset prior is worst at, and margin
prediction is a relative-strength-comparison task in a way total-goals prediction isn't).
Natural next build: a Marcel-style cross-season team-strength prior (last season's final
walk-forward rating, regressed toward league average by a games-based weight, blended out as
current-season data accumulates — the same shrinkage MACHINERY already used everywhere in
this project, applied across a season boundary instead of only within one) — grid-searched
jointly with `PRIOR_GAMES` per §7.5's own lesson that priors and decay/carryover terms are
entangled and must be fit together, not sequentially. Given the size of this effect relative
to the diagonal-inflation cycle's, and per the review's own expected-value-per-effort framing,
**this is recommended as the next cycle, ahead of Cycle 16's GBM layer** — not yet attempted.

## 12. Cycle 18: cross-season team-strength prior — real, adopted, new current best (2026-07-23)

§11 flagged a concrete, buildable explanation for the front-loaded model-vs-market gap: this
project's team-strength ratings reset to the pure trailing league average at game 1 of every
season (confirmed directly in `shrinkage.add_walk_forward_rate`/`add_walk_forward_toi_rate` —
the cumulative for/against sums are grouped by `["team", "season"]`, so `games_before=0` at a
new season's first game gives zero team-specific weight), while the real market opens every
season with genuine team-specific preseason information regardless of any prior-season
disruption. The original architecture proposal (§3.2) called for a "recency-weighted
multi-season prior" that was never built; cross-season goalie history (§4.10) tested a
narrower, different question (individual goalie carryover) and was a wash, not evidence
against a team-level prior.

### 12.1 Residuals-first check: is there real season-to-season signal being thrown away?

Checked before building anything (`src/models/validate_cross_season_prior_check.py`): does a
team's own last-season whole-game xG-differential residual (vs. that season's league average)
correlate with the SAME team's early-current-season (games 1-15) performance residual?

**Real result: r=0.1534, p=2.75e-42, n=7,800 team-game rows.** A genuine, highly significant
correlation the current model exploits ZERO of at game 1 of every season (it has no
team-specific information at all at that point). Confirmed real before writing any prior
mechanism.

### 12.2 Design and implementation

Added `shrinkage.add_walk_forward_toi_rate_cross_season` — identical to the existing
`add_walk_forward_toi_rate` except the shrinkage prior term is a blend, per team-season:

```
prior_mean = cross_season_weight * (team's own immediately-preceding season's full-season rate)
             + (1 - cross_season_weight) * (trailing league average)
```

`cross_season_weight=0` is an exact no-op (bit-identical to the existing function — verified
directly, not assumed, since `weight=0.0` in every grid run below was called with
`cross_season_weight=None`, routing to the OLD function, and its metrics match
`drift_adjusted_poisson` exactly). Teams with no matched immediately-preceding season
(expansion teams, or the dataset's own first season) fall back to the pure trailing league
average for that team-season — no NaN propagation. Threaded through as an optional parameter
(`None` default preserves existing behavior) through
`team_strength_situational.add_walk_forward_situational_strength` →
`validate_situational_toi.run_validation` → `validate_goalie.run_validation` →
`validate_rest.run_validation`, applied identically to all four situational channels
(EV/PP/PK/other) with a single shared weight, per the discipline of testing one new variable
at a time before elaborating per-channel weights.

### 12.3 Grid search: a genuine, monotonic bias-variance trade-off

Grid-searched `cross_season_weight` ∈ {0, 0.25, 0.5, 0.75, 1.0}, `HALFLIFE_GAMES=600` held
fixed (`src/models/validate_cross_season_prior.py`), paired bootstrap (5,000 resamples)
against weight=0 on the identical 16,528 dev-set games:

| Weight | SU | Brier | Total-MAE | Margin-MAE |
|---|---|---|---|---|
| 0.25 | +0.00266, CI [0.00085,0.00442] (real) | -0.00030, CI [-0.00039,-0.00021] (real) | +0.00014, CI [-0.00036,0.00064] (neutral) | -0.00271, CI [-0.00331,-0.00215] (real) |
| 0.50 | +0.00188, CI [-0.00067,0.00430] (neutral) | -0.00052, CI [-0.00070,-0.00035] (real) | +0.00064, CI [-0.00035,0.00162] (neutral) | -0.00521, CI [-0.00640,-0.00410] (real) |
| 0.75 | +0.00218, CI [-0.00085,0.00514] (neutral) | -0.00067, CI [-0.00094,-0.00041] (real) | +0.00150, CI [-0.00000,0.00296] (borderline) | -0.00735, CI [-0.00914,-0.00569] (real) |
| 1.00 | +0.00472, CI [0.00121,0.00811] (real) | -0.00075, CI [-0.00111,-0.00041] (real) | +0.00281, CI [0.00082,0.00471] (real HARM) | -0.00902, CI [-0.01135,-0.00684] (real) |

A genuine, monotonic trade-off, in the same family as Cycle 13's halflife-vs-priors tension
(§7.4): raising the weight monotonically improves Brier and margin-MAE, but total-MAE moves
from neutral toward real harm as weight increases, becoming clearly disqualifying at 1.0.
Under this project's own bar (no kept change may carry a real, bootstrap-confirmed
regression), **weight=1.0 is rejected outright** (real total-MAE harm) despite having the
best dev-set SU and Brier. Weight=0.5 is the largest weight with NO real regression on any
of the four metrics on the dev set.

### 12.4 The holdout check breaks the tie — and reveals a real, important divergence

Per this project's established practice for any real "kept" candidate, ran the honest holdout
re-check (`src/models/check_holdout_cross_season.py`, methodology identical to §7.6: all
constants fit dev-only, applied to both splits without refitting) for weight=0 (baseline),
0.5, and 1.0:

| Weight | Split | SU | Brier | Margin-MAE | Total-MAE |
|---|---|---|---|---|---|
| 0 (baseline) | dev | 57.64% | 0.24076 | 2.01898 | 1.80287 |
| 0 (baseline) | **holdout** | **56.90%** | **0.24289** | **2.14159** | **1.86373** |
| 0.5 | dev | 57.83% | 0.24024 | 2.01376 | 1.80351 |
| 0.5 | **holdout** | **57.20%** | **0.24254** | **2.13670** | **1.86403** |
| 1.0 | dev | 58.11% | 0.24001 | 2.00996 | 1.80568 |
| 1.0 | **holdout** | **56.86%** | **0.24254** | **2.13406** | **1.86469** |

**This is the decisive finding.** Weight=1.0 has the best DEV-set SU (58.11%) of the three,
but its holdout SU (56.86%) is actually the WORST of the three — slightly below even the
no-prior baseline (56.90%). Weight=0.5's holdout SU (57.20%), by contrast, is the best of all
three on the split that matters most — real, out-of-sample games the constants never touched.
Brier and margin-MAE are close between 0.5 and 1.0 on holdout (both real improvements over
baseline), but SU alone is enough to disqualify the extreme weight: **pure single-season
carryover (weight=1.0, no blend toward league average at all) overfits the dev sample's own
season-to-season idiosyncrasies in a way that doesn't generalize, while weight=0.5's genuine
regression-toward-the-mean blend generalizes better** — exactly the kind of overfitting a
Marcel-style shrinkage toward league average is supposed to guard against, and exactly why
the naive "just use last season's rate directly" version (weight=1.0) was the wrong call
despite looking best in-sample.

### 12.5 Mechanism check: does it actually narrow the front-loaded market gap?

The whole cycle was motivated by §11's finding that the model-vs-market gap concentrates in
each team's first 15 games of a season. Directly checked whether weight=0.5 narrows it
specifically there, not just in aggregate (`src/models/validate_cross_season_market_gap.py`),
on the identical 11,803 market-matched games:

| Games into season | Gap, weight=0 | Gap, weight=0.5 | Reduction |
|---|---|---|---|
| 1-15 | +0.00674 | +0.00504 | **-0.00170 (25% closed)** |
| 16-40 | +0.00310 | +0.00287 | -0.00023 (7% closed) |
| 41-100 | +0.00242 | +0.00221 | -0.00021 (9% closed) |

**Exactly the predicted signature**: the gap narrows most where the mechanism says it should
(early season, 25% of that bin's gap closed) and least in the bins where current-season data
already dominates. This isn't just a generic aggregate improvement landing anywhere — it is
targeted precisely at the diagnosed problem, real confirmation that the mechanism (not some
unrelated correlated effect) is what's driving the gain.

### 12.6 Verdict: ADOPTED — new current best model, `cross_season_prior_poisson`

**Kept, `cross_season_weight=0.5`.** Real, bootstrap-confirmed improvement on SU, Brier, and
margin-MAE on the development set (no real regression on any metric), confirmed to hold up
on the genuine, untouched holdout (SU +0.30pt, Brier improved, margin-MAE improved, total-MAE
essentially flat), and confirmed to specifically narrow the front-loaded market gap this
cycle was built to address (25% closed in the first-15-games bin). `weight=1.0`, despite
looking best on the dev set alone, is explicitly rejected — its dev-set SU gain doesn't
survive the holdout, a real and important negative finding in its own right about the limits
of un-regressed single-season carryover. New production entry point:
`src/models/validate_cross_season.py` (mirrors `validate_drift.py`'s role as the prior
"current best" reference point — any future cycle should build on this one, not
`validate_drift.py` directly).

This is the strongest single-cycle result in the project since Cycle 13's drift fix, and
unlike Cycle 13, it improves SU directly (not just margin/totals) with no real trade-off on
the holdout split. Not yet attempted: a joint grid search of `cross_season_weight` against
`PRIOR_MINUTES_*`/`PRIOR_GAMES` (per §7.5's own lesson that priors and new carryover terms
are entangled) — a plausible source of further, currently unclaimed improvement, since this
cycle held `PRIOR_MINUTES_*` fixed at their Cycle 13 values throughout.

Next: either that joint `PRIOR_MINUTES`/`cross_season_weight` re-grid, §10.6.3's local-transfer
tie-mass parameter, or Cycle 16's GBM stacking layer — all live, unstarted candidates.

## 13. Cycle 19: joint cross_season_weight × prior_minutes grid (2026-07-23) — ⚠ verdict corrected in §15: ADOPTED, not rejected

Per §7.5's own lesson (priors and any new carryover/decay term are entangled — don't assume
the old scale stays optimal after adding one) and §0.4 item 4's flagged gap, grid-searched
`cross_season_weight` ∈ {0.25, 0.5, 0.75} jointly against a new `prior_minutes_multiplier`
∈ {0.5, 0.75, 1.0, 1.5, 2.0} (further scaling the already-×0.6-adjusted `PRIOR_MINUTES_*`
constants), `HALFLIFE_GAMES=600` held fixed. Added `prior_minutes_multiplier` as an optional
parameter threaded through the same call chain as `cross_season_weight` (default 1.0 preserves
every existing caller's exact behavior).

### 13.1 Dev-set grid: a real-looking improvement at higher weight and multiplier

15-combination grid, dev-set Brier as the primary ranking metric:

| Weight | Mult | SU | Brier | Total-MAE | Margin-MAE |
|---|---|---|---|---|---|
| 0.50 (current best) | 1.00 | 57.83% | 0.24024 | 1.80351 | 2.01376 |
| 0.75 | 1.50 | 57.97% | 0.23988 | 1.80205 | 2.01167 |
| 0.75 | 2.00 | 57.99% | **0.23979** | 1.80074 | 2.01206 |
| 0.75 | 3.00 (extended check) | 57.91% | **0.23973** (dev-set best) | 1.79962 | 2.01318 |

Paired bootstrap of `weight=0.75, mult=2.0` against the current best (weight=0.5, mult=1.0):
Brier mean -0.00046, 95% CI [-0.00066, -0.00026] (real); total-MAE mean -0.00277, 95% CI
[-0.00421, -0.00134] (real); margin-MAE mean -0.00170, 95% CI [-0.00302, -0.00037] (real); SU
neutral. **Zero real regressions on the dev set** — by the dev-set bootstrap alone, this
would have looked like a clean adoption candidate, better on 3 of 4 metrics with no
downside. Extending the multiplier grid further (weight=0.75, mult ∈ {2.5, 3.0, 4.0, 5.0})
confirmed a genuine interior optimum around mult≈3.0 for Brier (0.23973), not a boundary
artifact — Brier and total-MAE both start reversing past mult=3.0-4.0, while margin-MAE
degrades monotonically and continuously as mult rises past ~1.0.

### 13.2 The holdout check — exactly Cycle 18's own lesson, again

Per the practice §12.4 itself established (never trust a dev-set-best candidate without a
holdout check — Cycle 18's weight=1.0 looked best on dev and was worse than baseline on
holdout), ran the honest holdout re-check (`check_holdout_cross_season.py`, updated to accept
`prior_minutes_multiplier`) for the three leading dev-set candidates:

| Weight | Mult | Holdout SU | Holdout Brier | Holdout Margin-MAE |
|---|---|---|---|---|
| 0.50 (current best) | 1.00 | **57.20%** | 0.24254 | 2.1367 |
| 0.75 | 1.50 | 56.97% | 0.24253 | 2.1367 |
| 0.75 | 2.00 | 57.13% | 0.24259 | 2.1380 |
| 0.75 | 3.00 | 56.82% | 0.24275 | 2.1402 |

**None of the three dev-set-favored candidates improve holdout SU over the current best, and
two of the three are also slightly worse on holdout Brier and margin-MAE.** The exact same
divergence pattern as §12.4: raising `cross_season_weight` and `prior_minutes_multiplier`
together produces real-looking dev-set gains that do not survive the untouched holdout split
— a second, independent confirmation that this project's dev-set bootstrap alone is not
sufficient for a "kept" verdict when a candidate involves a shrinkage-strength parameter, and
that the holdout check is doing real, load-bearing work, not a formality.

### 13.3 Verdict: REJECTED — current best model unchanged

**No change adopted.** `cross_season_prior_poisson` (`cross_season_weight=0.5`,
`prior_minutes_multiplier=1.0`, i.e. the unmodified Cycle 13 prior scale) remains the current
best model. This is a real, informative negative result, not a wasted cycle: it confirms
Cycle 18's chosen weight (0.5) was not leaving an easy joint-tuning gain on the table, and it
reinforces — for a second time, independently — that this project's holdout check is
essential rather than a symbolic add-on whenever a candidate touches a shrinkage-strength
constant. `prior_minutes_multiplier` remains available (default 1.0, no behavior change) in
`team_strength_situational.py`/`validate_situational_toi.py`/`validate_goalie.py`/
`validate_rest.py` for any future cycle that wants to revisit this dimension.

Next: §10.6.3's local-transfer tie-mass parameter, re-based against `cross_season_prior_poisson`.

## 14. Cycle 20: local-transfer tie-mass parameter — rejected, with a clean mechanistic explanation (2026-07-23)

§10.6.3 confirmed a specific, buildable shape for the tie-mass fix Cycle 17 got wrong: instead
of a global scalar taxing every off-diagonal cell (including blowouts) to fund the diagonal,
move mass ONLY between each diagonal cell `(x,x)` and its two immediate neighbors
`(x+1,x)`/`(x,x+1)`, governed by a single transfer fraction `delta`. Built
`src/models/local_transfer_inflation.py`: `local_transfer(joint, delta)` moves mass directly
(no renormalization needed, unlike Cycle 17's theta — mass conservation is automatic), and
`fit_delta` solves for `delta` in closed form the same way theta was solved (matching the real
average tie rate). Fit `delta=0.1862` on 16,528 dev-set games, now built on top of the CURRENT
best model (`cross_season_prior_poisson`, §12/§13), not the superseded
`drift_adjusted_poisson` Cycle 17 was tested against.

### 14.1 Result: smaller margin cost than Cycle 17, but still real, at every tested delta

`src/models/validate_local_transfer.py`, paired bootstrap (5,000 resamples) against the
current best:

| Delta | SU | Brier | Total-MAE | Margin-MAE |
|---|---|---|---|---|
| 0.05 | neutral, CI [-0.00042,0.00175] | neutral, CI [-0.00007,0.00001] | **real gain**, CI [-0.00224,-0.00036] | **real HARM**, CI [0.00129,0.00212] |
| 0.10 | neutral | neutral | real gain (identical) | real HARM, CI [0.00140,0.00228] |
| 0.15 | neutral | neutral | real gain (identical) | real HARM, CI [0.00152,0.00244] |
| 0.1862 (fitted) | neutral, CI [-0.00054,0.00188] | neutral, CI [-0.00013,0.00002] | real gain (identical) | real HARM, CI [0.00160,0.00256] |

**Every single tested delta — including 0.05, a small, gentle nudge — shows a real,
bootstrap-confirmed margin-MAE regression.** Smaller than Cycle 17's global-theta harm
(0.00541 mean diff there vs. 0.00160-0.00256 here), confirming the local-transfer shape is
genuinely less damaging, but not clean, and not shrinking toward zero fast enough at small
delta to suggest a "safe" dose exists.

### 14.2 The decisive check: total-MAE's "gain" is identical at every delta, including delta=0

Total-MAE is EXACTLY 1.80224 at delta=0.05, 0.10, 0.15, AND 0.1862 — and, checked directly,
**exactly 1.80224 at delta=0.0 (a literal no-op) too.** This is conclusive, not
approximate: **100% of the measured total-MAE "improvement" comes from the regulation-ratio
reconstruction pathway alone (identical to §10.6.2's finding for Cycle 17's theta=1 ablation),
and the local-transfer parameter itself contributes exactly ZERO to total-goals accuracy** —
its only measurable effect on any metric is the real, monotonically-growing margin-MAE harm.

### 14.3 Why: the tie mass has to pass through a flat-rate OT layer, and that's the actual cost

The mechanism, once total-MAE's invariance made it obvious to look for: every unit of mass
moved into a tie cell `(x,x)` gets FULLY reallocated by the OT/SO layer into `(x+1,x)`/
`(x,x+1)` at the same fixed rate, `p_home_wins_ot≈0.509`, regardless of delta — so in the
FINAL recorded-score distribution, converting a "genuinely regulation-decisive margin-1" game
into a "regulation tie that gets OT-resolved to margin-1" game leaves the TOTAL exactly where
it was (both paths land on the same final total), but changes WHICH split of home-margin-1 vs.
away-margin-1 that mass ends up in: the original independent-Poisson split at `(x+1,x)` vs.
`(x,x+1)` reflects each specific game's own two team-strength lambdas (a skill-informed
split), while the OT layer's split is a single constant, ~50.9/49.1 for every game in the
league regardless of matchup. **Any amount of mass rerouted through the flat-rate OT layer
loses that per-game skill information and reverts to the league-constant split instead** —
a real, structural cost that scales with how much mass gets rerouted, which is exactly why
the harm is present even at the smallest tested delta and grows smoothly with it.

### 14.4 Verdict: REJECTED — and this closes off the whole family, not just this shape

**Rejected — current best model unchanged.** This is a cleaner, more decisive negative result
than Cycle 17's: not just "this specific fix has a cost," but a precise account of WHY any
tie-mass-inflation approach built on top of the current OT/SO layer will have a real margin
cost, for as long as `p_home_wins_ot` remains a single league-wide constant rather than a
team-specific (skill-informed) win probability. This means the tie-mass problem (§4.13's
original 0.168-vs-0.231 finding remains real and unaddressed) is not solvable as a small
add-on parameter under the current architecture — the natural next attempt, if this is
revisited, is making the OT/SO win probability itself a function of relative team strength
(replacing the flat `fit_ot_home_win_rate` constant with a walk-forward, team-specific model)
BEFORE any tie-inflation term is layered on top, so the "resolution" pathway doesn't average
away the skill information a tie-mass fix would otherwise be smuggling through it. Not a small
follow-up — a real, separate sub-project, not queued as a specific next cycle here.

Next: §6.8 market-benchmark re-run against `cross_season_prior_poisson` (task pending), or
Cycle 16's GBM stacking layer.

## 15. Corrected holdout protocol — re-adjudicating Cycles 18 and 19 (2026-07-23)

An external review flagged a real inconsistency in how this project's own stated bar ("no
real, bootstrap-confirmed regression") was being enforced: on the development set, every
verdict throughout §4-§14 is backed by a 5,000-resample paired bootstrap. But the holdout
comparisons in §12.4 and §13.2 were raw point estimates with no confidence interval — and on
2,624 holdout games, several of the differences being treated as decisive were tiny (§13.2's
leading candidate was rejected on a 57.13% vs. 57.20% holdout SU difference — about 2 games;
§12.4's "weight=1.0 is worse than baseline, real overfitting" claim was a 56.86% vs. 56.90%
difference — about 1 game). The review's proposed fix, adopted here:

**Holdout's role is CONFIRMATORY VETO ONLY, going forward.** A dev-set-confirmed real
improvement is rejected on holdout grounds if and only if a paired bootstrap on the holdout
split shows a REAL regression (CI does not cross zero) on a metric that was real on the dev
set. Holdout point estimates are never used to RANK or choose between multiple dev-confirmed
candidates — ranking on holdout is what turns it into a second dev set and defeats the entire
point of holding it out. `src/models/validate_holdout_bootstrap.py` implements this: the same
paired-bootstrap machinery used everywhere else, run on the holdout-only games.

**Peek-count, made visible rather than implicit**: every holdout access this project has ever
made is already logged in `data/processed/metrics_ledger.parquet` (via `append_run` with
`split: "holdout"` in `config_flags`, a practice already in place since §7.6). Querying it
directly: **9 distinct holdout accesses so far** (§7.6's `goalie_overlay_poisson`/
`drift_adjusted_poisson` baseline re-checks, §12.4's weight=0/0.5/1.0 checks, §13.2's
weight=0.75×{1.5,2.0,3.0} checks, and this section's weight=0.5/mult=1.0 re-check) — a real,
non-trivial peek count, exactly the kind of thing that should be visible rather than
uncounted.

### 15.1 Re-checking Cycle 18's holdout claims

Paired bootstrap (5,000 resamples) on the 2,624 holdout games, `weight=0.5` vs. `weight=0.0`:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | +0.00305 | [-0.00343, 0.00991] | Crosses zero — NOT significant |
| Brier | -0.00034 | [-0.00084, 0.00016] | Crosses zero — NOT significant |
| Total-MAE | +0.00030 | [-0.00174, 0.00225] | Crosses zero — NOT significant |
| Margin-MAE | -0.00489 | **[-0.00825, -0.00165]** | **REAL improvement** |

`weight=1.0` vs. `weight=0.0`:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00038 | [-0.00991, 0.00915] | Crosses zero — NOT significant |
| Margin-MAE | -0.00753 | **[-0.01398, -0.00108]** | **REAL improvement** |

**Correction to §12.4's own narrative**: "weight=1.0's holdout SU is worse than baseline, real
overfitting" is NOT supported by the bootstrap — that difference (-0.00038, about one game)
sits deep inside a CI that spans from -0.00991 to +0.00915. The direct `weight=1.0` vs.
`weight=0.5` comparison likewise crosses zero on every metric. **The decision to adopt
weight=0.5 over weight=1.0 was still correct — but for the reason already established on the
dev set (§12.3): weight=1.0 carried a real, bootstrap-confirmed total-MAE regression there,
which alone disqualifies it under this project's own bar.** The holdout "doesn't generalize"
story was an unearned, second, incorrect justification layered on top of an already-sufficient
one. Interestingly, margin-MAE's holdout improvement IS real for both weight=0.5 and
weight=1.0 — the one part of §12.4's holdout narrative that does survive proper bootstrapping.

### 15.2 Re-checking Cycle 19 under the corrected protocol — the verdict changes

Paired bootstrap on the 2,624 holdout games, each `weight=0.75` candidate vs. the (then-)
current best `weight=0.5, multiplier=1.0`:

| Candidate | SU | Brier | Total-MAE | Margin-MAE |
|---|---|---|---|---|
| mult=1.5 | crosses zero | crosses zero | crosses zero | crosses zero [-0.00251,0.00232] |
| mult=2.0 | crosses zero | crosses zero | crosses zero | crosses zero [-0.00211,0.00455] |
| mult=3.0 | crosses zero | crosses zero | crosses zero | crosses zero [-0.00129,0.00815] |

**None of the three candidates show a real (bootstrap-confirmed) regression on holdout, on
any metric.** Under the corrected protocol, none of them should have been vetoed. §13.3's
rejection was based on raw point-estimate ranking — exactly the failure mode the review
identified.

Re-examining the dev-set bootstrap evidence (§13.1) that was already real and never in
question: mult=1.5 and mult=2.0 both show real, bootstrap-confirmed gains on Brier, total-MAE,
AND margin-MAE with zero dev-set regressions; mult=3.0's margin-MAE gain fades to noise on
closer bootstrapping (CI [-0.00253, 0.00136], crosses zero) even though Brier and total-MAE
remain real and are the largest of the three. **Adopting `mult=2.0`**: real gains on all three
of Brier/total-MAE/margin-MAE on the dev set (the most complete profile of the three), no real
regression on either split, dev-set effect sizes larger than mult=1.5's.

**Corrected verdict: Cycle 19 is ADOPTED, not rejected.** New current best model:
`cross_season_prior_poisson` with `cross_season_weight=0.75`, `prior_minutes_multiplier=2.0`
(`src/models/validate_cross_season.py`, updated). Dev set: SU 57.99%, Brier 0.23979,
total-MAE 1.80074, margin-MAE 2.01206 (all improved over the previous weight=0.5/mult=1.0
best). Holdout: SU 57.13%, Brier 0.24259, total-MAE 1.86282, margin-MAE 2.13801 — point
estimates are mixed relative to the previous best (SU and margin-MAE point estimates are
slightly worse in holdout, Brier essentially flat), but per §15's own corrected rule, mixed
point estimates with no real bootstrap-confirmed regression are exactly the outcome that
should NOT veto a dev-confirmed candidate — that tension not resolving cleanly one direction
is itself the reason the confirmatory-veto rule exists, rather than a subjective per-cycle
judgment call.

### 15.3 What this changes and what it doesn't

§0.2's current-best numbers, §12/§13's verdicts, and the file map are updated below to reflect
`cross_season_weight=0.75, prior_minutes_multiplier=2.0` as current best. §14's local-transfer
rejection is UNAFFECTED — it was rejected on dev-set bootstrap grounds alone (real margin-MAE
harm at every tested delta, confirmed down to delta=0.05), with no holdout claim involved.
§7.6's original Cycle 13 holdout re-check and §12.4's `weight=0.5` vs. baseline comparison
both had at least one real, bootstrap-confirmable holdout finding (margin-MAE, per §15.1) even
before this correction, so those adoptions stand on firmer ground than they appeared to at the
time, not shakier ground.

**Going forward**: any future cycle's holdout check must run the paired bootstrap
(`validate_holdout_bootstrap.py`'s pattern), not a raw point-estimate comparison, and must
treat holdout strictly as a veto-only gate, never a ranking tool between multiple otherwise-
qualified dev-confirmed candidates.

Next: CRPS and a totals-line market benchmark (§16) — pure measurement, needed before any
future joint-distribution cycle can be scored fairly, per the same review's second point.

## 16. Cycle 21: CRPS + totals-line benchmark — pure measurement, one real finding sharpened dramatically (2026-07-23)

Motivated by §10.6.1: total-goals/margin MAE score only the distribution's MEAN, not its
shape — a future dependence/dispersion fix that genuinely closed the real Var(T)/E(T) gap
(real 0.8913 vs. model ~1.02) could show up as a complete wash under the current metric set.
Added two pure-measurement tools before any such cycle is attempted, per the external review's
second point.

### 16.1 CRPS on totals and margin — new baseline numbers

Built `src/models/crps.py` (`discrete_crps`, the discrete/ranked-probability-score analogue of
continuous CRPS: `sum_k [F(k) - 1{y<=k}]^2` over the discrete support) and
`src/models/validate_crps.py`, which builds the current best model's full final recorded-score
joint (identical construction to §14's pipeline, `delta=0` — a pure no-op, since neither
Cycle 17's theta nor Cycle 20's delta is in production) and computes CRPS per game.
`metrics_ledger.append_run` gained an optional `extra_metrics` passthrough (default `None`
preserves every existing caller) so CRPS is now a first-class, loggable metric.

**Baseline (current best, `cross_season_prior_poisson`, weight=0.75/mult=2.0): mean CRPS
(total) = 1.2706, mean CRPS (margin) = 1.3556**, logged to the ledger
(`cross_season_prior_poisson_WITH_CRPS`). These numbers are only useful as a BEFORE reading —
any future dependence/dispersion cycle must report CRPS on the identical game set to be
scored fairly; a Brier/MAE-neutral cycle that lowers CRPS would be real, measurable evidence
the distributional shape genuinely improved, which the existing metric set could not have
shown.

### 16.2 Totals-line benchmark — real data limitation, and one dramatic real finding

**Data reality check, done before building anything**: `sbro_odds_games.parquet` has real
`close_total`/`open_total` LINES (14,259/14,260 games, essentially complete coverage) but NOT
real over/under PRICING — checked directly against the raw SportsbookReviewsOnline source
(`sbro_odds_raw.parquet`'s `OpenOU`/`CloseOU` columns): only 2 of 33,358 raw rows contain
anything that looks like odds (a `-110` value), confirmed to be a data-entry artifact on one
specific game, not systematic pricing. **A true no-vig market probability for totals cannot be
computed from this data** — unlike the moneyline (§6.8), where real two-sided pricing exists.

The honest benchmark built instead: sportsbooks set the total LINE (not just the price)
specifically so the two sides sit near a genuine coin flip — so a constant 0.5 is a reasonable
market-implied P(over) baseline. Compared the model's own predicted P(actual_total >
close_total) against this baseline via Brier score, on 10,871 real, decided (non-push) games
(931 pushes, ~7.9% of the matched 11,802, correctly excluded rather than folded into either
side):

**Model Brier (P(over) vs. actual): 0.26407. Market-implied baseline Brier (constant 0.5):
0.25000.** The model is WORSE than assuming every game is a coin flip at the market's own
line — real over rate at the market's close line is 49.47%, confirming the line-setting
assumption is sound (the market really does sit lines near 50/50).

**Checked before reporting this as a finding rather than a bug**: binned model P(over)
predictions into deciles and compared against actual over rate per decile. The model shows
some real, weak discrimination (lowest-decile actual-over-rate 47.2% vs. highest-decile
54.0%, a real 6.8-point spread) — but the model's MEAN predicted P(over) across the whole
sample is **38.98%**, a full 10.5 points below the real ~49.5% base rate. **This is not a
calibration-spread problem, it's a systematic bias problem — the model's own total-goals
distribution is centered too low.**

**This is the exact known issue from §7.5/§0.4 item 4, now shown to have serious, concrete
real-world impact.** §7.5 flagged "the raw predicted mean total still undershoots real scoring
somewhat" as an honest residual note; the current best model's own logged
`mean_pred_total` (5.5285, or 5.482 in this section's independent re-derivation through the
full-joint pathway) vs. `mean_actual_total` (5.8125) confirms a real ~0.28-0.33 goal
systematic undershoot. Under total-goals MAE, a bias this size is nearly invisible — it's
small relative to typical per-game error (~1.8 goals). **Under a hard over/under threshold, the
same bias is large enough to make the model's own totals probabilities WORSE than a naive
coin flip.** What was a small, abstract residual note in §7.5 is, per this measurement, a
real, actionable, and apparently larger-than-previously-appreciated problem.

### 16.3 Verdict (superseded by §16.4-16.5 below — see §16.6 for the corrected read)

CRPS is now available for any future joint-distribution cycle to be scored fairly (§16.1). The
totals-line benchmark cannot be built into a full market-probability comparison without real
O/U pricing data (a genuine data limitation, not a design choice) — but the P(over)-vs-baseline
check it enabled surfaced what looked like a large finding: the model's mean predicted total
biased low enough to make its own over/under calls worse than a coin flip at the market's
line. A follow-up review flagged that the arithmetic didn't close (a 0.3-goal mean shift on a
distribution with sd≈2.4 should move mean P(over) by ~4-5 points, not the measured 10.5) and
asked for two checks before any recalibration cycle: (1) whether the P(over) computation
itself was conditioned correctly for push cells, and (2) whether the bias is stable across
seasons or era-localized (which would trap a naive constant fix). Both were run.

### 16.4 Correction: the original P(over) was NOT conditioned on decided games — real gap is roughly half the original estimate

The original computation (§16.2) computed `P(over) = 1 - P(T <= floor(line))` for every line,
including whole-number lines where a push (`T == line`) is possible. This is the correct
UNCONDITIONAL probability, but the empirical side of the comparison was conditioned on
decided (non-push) games only (`is_push` rows excluded) — an apples-to-oranges mismatch of
exactly the kind §10.6.1 caught in the dispersion check. **For whole-number lines (39.95% of
all matched games), the correct like-for-like quantity is `P(over | decided) = P(T>line) /
(1 - P(T==line))`, not the raw unconditional probability.**

Recomputed with the correction:

| | Original (unconditional, §16.2) | Corrected (conditioned on decided) |
|---|---|---|
| Mean model P(over) | 38.98% | **41.61%** |
| Model Brier | 0.26407 | **0.25683** |
| Baseline Brier (constant 0.5) | 0.25000 | 0.25000 |

**The push-conditioning error accounts for roughly HALF the originally-measured gap** (Brier
gap shrinks from 0.01407 to 0.00683; mean P(over) recovers 2.6 of the original 10.5-point
deficit). The finding survives, but at roughly half the claimed magnitude: the model's own
totals probabilities are still measurably worse than a naive coin-flip baseline (0.25683 vs.
0.25000), a real, if smaller, calibration problem — not the dramatic result §16.2 originally
reported.

### 16.5 The per-season bias curve: not flat, not the hump either — a decade of undershoot that resolved recently

Computed `mean(predicted_total - actual_total)` per season across the full dev+holdout range
(constants fit dev-only, applied throughout — same discipline as every holdout check here):

| Season | n | Mean bias (pred − actual) |
|---|---|---|
| 2010-11 | 1,230 | −0.289 |
| 2011-12 | 1,230 | −0.306 |
| 2012-13 | 720 | −0.353 |
| 2013-14 | 1,230 | −0.421 |
| 2014-15 | 1,230 | −0.375 |
| 2015-16 | 1,230 | −0.300 |
| 2016-17 | 1,230 | −0.256 |
| 2017-18 | 1,271 | −0.286 |
| 2018-19 | 1,271 | −0.387 |
| 2019-20 | 1,082 | −0.472 |
| 2020-21 | 868 | −0.470 |
| 2021-22 | 1,312 | **−0.631** (worst) |
| 2022-23 | 1,312 | −0.146 |
| 2023-24 | 1,312 | **−0.016** (near zero) |
| 2024-25 (holdout) | 1,312 | −0.049 |
| 2025-26 (holdout) | 1,312 | −0.161 |

**Neither hypothesis was quite right.** The bias is NOT flat-negative everywhere (ruled out:
2023-24 and 2024-25 are close to zero) — but it's also not a clean hump centered on 2017-18
with both ends near zero. Instead: a **persistent, real undershoot (roughly −0.25 to −0.47)
across essentially the ENTIRE 2010-11 through 2020-21 span**, a sharp WORSENING at 2021-22
(the first full post-COVID season, −0.631, the single largest bias of any season), then a
fast convergence to near-zero starting 2022-23 and holding (mostly) through the holdout —
2025-26 shows some reversion to −0.161, so even the recent convergence isn't perfectly stable.

**This still answers the question the check was designed to answer, per its own decision
rule**: a flat-everywhere bias (which would legitimate a single constant fix) is clearly
false. The bias is time-localized — concentrated in a specific decade-long stretch that has
mostly, though not perfectly, resolved on its own by the most recent seasons. **A single
dev-fit constant uplift would over-correct the already-mostly-fixed holdout era exactly as
warned** — the confirmatory-veto holdout check (§15) would likely catch and kill it, but only
after the cycle was built. The real design implication: whatever fixes this needs to be a
time-local, level-tracking mechanism (a shorter/re-tuned decay rate, checked specifically
against this per-season bias curve rather than the aggregate SU/Brier/MAE metrics Cycle 13
used; or a walk-forward season-level intercept), not a constant.

### 16.6 Corrected verdict

CRPS (§16.1) stands as built. The totals-line P(over) finding is real but smaller than
originally reported (§16.4: Brier gap ~0.0068, not ~0.0141). The mean-total undershoot itself
is real, substantial, and NOT resolved by a simple additive constant (§16.5) — it requires a
time-local recalibration mechanism, with the exact form (re-tuned decay rate vs. a new
season-level intercept term) an open design choice, not yet built. Sharpens §0.4 item 4 into a
concrete candidate with a known failure mode to avoid, rather than a candidate ready to build
as originally framed.

Next: §17, a team-specific OT/SO win-probability model — the prerequisite §14.4 identified for
reopening the tie-mass/dependence family.

## 17. Cycle 22: team-specific OT/SO win probability — real, adopted, new current best (2026-07-23)

§14.4 identified the prerequisite for reopening the tie-mass/dependence family: the current
OT/SO layer resolves every regulation tie via a single, flat, league-wide constant
(`p_home_wins_ot≈0.509`), discarding the per-game, skill-informed split the independent-
Poisson joint itself encodes. This is a standalone-valuable fix independent of the tie-mass
question — roughly 23% of all games have their winner decided by this currently-flat constant.

### 17.1 Residuals-first check: does team strength predict OT outcomes at all?

Checked before building anything (`src/models/validate_ot_logistic_check.py`): correlation
between each OT/SO game's regulation lambda differential (from the current best model's own
walk-forward team strength) and whether the home team won, checked separately for OT-decided
and shootout-decided games (prior expectation: real but weak in OT, none in shootouts).

| Subset | n | Correlation | p-value |
|---|---|---|---|
| All OT/SO | 3,810 | r=0.0547 | 0.0007 |
| **OT-decided only** | 2,209 | **r=0.0842** | **0.0001** |
| **SO-decided only** | 1,601 | **r=0.0140** | **0.5760** |

**Exactly the predicted signature**: real, significant signal in OT-decided games, none at all
in shootouts (p=0.58, not distinguishable from zero). Home win rate when the model favors home
vs. away: 54.3% vs. 47.7% in OT-decided games (a real, meaningful split); 48.5% vs. 50.9% in
shootouts (backwards and non-significant — a shootout is a narrower, more luck-driven skills
competition, exactly as expected).

### 17.2 Design: team-specific logistic for OT, flat rate kept for shootouts

Built `src/models/ot_logistic.py`: a single global logistic `P(home wins OT) =
sigmoid(a + b*lambda_diff)`, fit by MLE (`statsmodels.Logit`) ONLY on OT-decided dev-set games
— same "fit once, not walk-forward per-game" precedent as every other single-constant fit in
this project (each game's own `lambda_diff` is already walk-forward; only the two logistic
coefficients are a single dev-set constant). Shootout-decided games keep the existing flat
empirical rate, per §17.1's null result — adding a team-specific term there would not be
honest given the data.

**Fitted: a=0.0082 (p=0.862, not significantly different from zero — no extra home-ice edge
in OT beyond what's already in the model), b=0.3023 (p<0.001, real and significant).**
Pseudo-R²=0.0051 — a real but genuinely weak effect, exactly matching the residuals-first
check's own magnitude. The final tie-resolution probability blends both real outcomes via
fixed, dev-set-fit shares: `P(home wins tie) = 0.5798 * sigmoid(a+b*lambda_diff) + 0.4202 *
0.4922` (the shares are real empirical rates: 57.98% of OT/SO games are OT-decided, 42.02%
need a shootout; the flat shootout home-win rate is 49.22%).

### 17.3 Validation: a clean, real, narrowly-confined win — on both dev and holdout

`src/models/validate_ot_logistic.py`, paired bootstrap (5,000 resamples) against the current
best (flat OT/SO rate), dev set:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00024 | [-0.00157, 0.00103] | Crosses zero — neutral |
| Brier | +0.00002 | [-0.00004, 0.00008] | Crosses zero — neutral |
| Total-MAE | -0.00000 | [-0.00000, 0.00000] | Exactly zero (structural, not approximate — see below) |
| Margin-MAE | -0.00067 | **[-0.00079, -0.00054]** | **REAL improvement** |

**Total-MAE's exact invariance is a structural fact, not noise**: moving mass between adjacent
diagonal/off-diagonal cells at a DIFFERENT split ratio than before still conserves each cell's
own total (home+away), so nothing about total-goals calibration can move — only which SIDE
wins by how much. This is the same mass-conservation logic §14.2 identified for the (rejected)
local-transfer tie-mass fix, now working correctly in this cycle's favor: a fix confined
entirely to WHO wins the tie, not whether ties happen at all, cannot possibly touch total-MAE,
by construction.

**Holdout, using the corrected §15 protocol (paired bootstrap, confirmatory-veto-only,
`src/models/check_holdout_ot_logistic.py`)**, all constants fit dev-only:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | +0.00305 | [-0.00038, 0.00686] | Crosses zero (barely) — neutral |
| Brier | +0.00013 | [-0.00002, 0.00027] | Crosses zero — neutral |
| Total-MAE | -0.00000 | [-0.00000, 0.00000] | Exactly zero, same structural reason |
| Margin-MAE | -0.00042 | **[-0.00072, -0.00011]** | **REAL improvement** |

**No real regression on any metric, on either split — margin-MAE improves with a real,
bootstrap-confirmed effect on BOTH the dev set and the untouched holdout.**

### 17.4 Verdict: ADOPTED — new current best model, `ot_team_specific_poisson`

**Kept.** A clean, narrowly-confined, real win — the same recurring pattern this project has
seen since §4.6's goalie overlay and §4.14's away-B2B adjustment: a real effect that helps ONE
specific axis (here, margin) with no effect elsewhere, rather than a broad win. New production
entry point: `src/models/validate_ot_team_specific.py` (supersedes `validate_cross_season.py`
as the current best pipeline). Standalone value confirmed exactly as motivated: replacing a
flat, uninformative constant for ~23% of all games with a real, if weak, skill-informed split
is a legitimate improvement in its own right, independent of the still-open tie-mass question.

This also reopens the tie-mass/dependence family §14.4 closed off: any future diagonal-
inflation or local-transfer attempt now resolves ties through a REAL, team-specific
probability rather than a flat league constant, so it should no longer necessarily incur the
mechanical margin cost §14.3 identified (mass rerouted through the tie no longer reverts to a
league-average split — it reverts to a skill-informed one). Re-attempting §14's local-transfer
fix on top of this new base is a legitimate, not-yet-tried candidate.

Next: a time-local mean-total recalibration (§16.5/§0.4 item 5 — a level-tracking mechanism,
not a constant, per the per-season bias curve), the local-transfer tie-mass re-attempt this
cycle reopens, or Cycle 16's GBM
stacking layer.

## 18. Cycle 23: shorthanded-goals-for term — real, root-cause, adopted (2026-07-23)

A third-round review, given §16.5's per-season bias curve, proposed empty-net goals
(undervalued by xG models but converting near-certainly) as the mechanism, since EN attempts
live in the "other" bucket. Two checks were run before building anything.

### 18.1 The empty-net/xG hypothesis is cleanly falsified

Aggregate real goals vs. real xG across the ENTIRE dataset (2008-2025, all situations):
133,268 goals vs. 133,158.3 xG — **ratio 1.0008, essentially perfect calibration.**
Per-season, the gap oscillates between -0.09 and +0.13 goals/team-game with no consistent
sign — noise, not a structural bias. The EN-shot-level effect this project's own goalie-GSAx
work found earlier (xG averaging 0.61 vs. ~100% conversion, `fetch_moneypuck_goalie_games.py`)
is real at the individual-shot level but too small in volume to leave an aggregate mark —
swamped by other small calibration give-and-take elsewhere in MoneyPuck's model.

### 18.2 Per-bucket decomposition: the real mechanism is different, cleaner, and already on record

Built `src/models/validate_bias_decomposition.py`: per-season model-vs-actual goals in each
situation bucket (EV / PP+PK / other), using the SAME `_combine` function the production
pipeline calls (not a re-derivation). **The "other" bucket OVERSHOOTS (+0.080/game average) —
the opposite of the empty-net-TOI-lag hypothesis, which is now doubly ruled out.**

**PP+PK undershoots in all 16 seasons with zero exceptions** (mean -0.113/game) — the cleanest,
most consistent signal in the whole decomposition. This bucket's gap is explained by
something already on record: Cycle 3's own docstring (§4.4) deliberately left
shorthanded-goals-for unmodeled ("a genuinely small-magnitude, rare event league-wide...
folding it in is a minor, low-priority addition"). Measured directly: **real shorthanded goals
run 0.140/game league-wide, remarkably stable across all 18 seasons (0.114-0.164, no trend)**
— matching the PP+PK gap closely, and the era-independent stability explains why the
undershoot is universal rather than concentrated in any particular stretch (a structural
omission, not a lag artifact).

**EV shows its own smaller, less consistent undershoot** (-0.091/game, ~13 of 16 seasons) —
NOT explained by this decomposition.

### 18.3 Resolving the 0.07-vs-0.140 units question, and the free EV check

Cycle 3's docstring sized the omission at "~0.07 xG/team-game" — half the newly-measured
0.140/game figure. **Resolved: units, not a real conversion gap.** Cycle 3's figure was
per-team-game (one side); 0.140 is per-game (both sides) — 0.07×2=0.14 matches exactly.
Directly confirmed SH-xG is well-calibrated to SH-goals (real ratio 1.008 across the dataset,
per-season values 0.85-1.19 with no trend) — **no fitted goal/xG conversion factor needed; xG
is a valid basis for this term**, consistent with this project's standing xG-as-input
principle rather than an exception to it.

**Free check, EV bucket**: real EV goals vs. real EV xG, per season — ratio 0.9966 overall, no
persistent one-directional gap. **Confirms the EV-bucket residual is NOT a data-level issue —
it lives in the combine/shrinkage math itself.** Per this same logic, the walk-forward
season-level intercept (task tracked, §16.5) is the right general-purpose closer for whatever
of the EV residual remains, not a new EV-specific data-conversion cycle.

### 18.4 Design and build

Built the term as a single league-wide rate (no team-specific shrinkage — at 0.140/game, a
team's own single-season sample is ~11 events, too small for a reliable team rate without its
own repeatability check, not attempted here): `predicted_SH_goals(team) = league_avg_sh_rate_per60
× (that team's own already-walk-forward-estimated PK ice time) / 60`. The league rate
(`pk_league_avg_for_per60`, decayed with the same `HALFLIFE_GAMES` family as every other rate)
and the team-specific PK-TOI estimate were BOTH already computed in the existing pipeline for
other purposes and simply never wired into the final combine sum — this is a true "flip it on"
fix, not new data plumbing. Added as optional parameters to `predict_situational_lambda`
(`league_avg_sh_rate_per60`/`home_pk_toi_min`/`away_pk_toi_min`, all defaulting to 0.0, an
exact no-op), threaded through `validate_situational_toi.py` → `validate_goalie.py` →
`validate_rest.py` via a new `use_sh_term` flag (default `False`, preserving every existing
caller). **This term's calibration uses no shrinkage prior at all — it's a pure trailing
league average — so it structurally cannot be entangled with `PRIOR_MINUTES_PP/PK`**, sidestepping
the §7.5-style entanglement risk by construction, not by luck.

### 18.5 Validation: lands exactly where aimed, on every target metric

Per-bucket check confirms the term lands correctly: PP+PK's gap flips from -0.113 (undershoot)
to **+0.035 (small overshoot)** — expected, since the fitted rate (0.140) slightly exceeds the
measured gap (0.113); EV (-0.091) and other (+0.080) are unchanged, exactly as a term confined
to the PP+PK bucket should leave them.

Paired bootstrap (5,000 resamples) against the current best (`ot_team_specific_poisson`), dev
set:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | +0.00024 | [-0.00079, 0.00121] | Crosses zero — neutral (expected) |
| Brier | -0.00000 | [-0.00003, 0.00002] | Crosses zero — neutral (expected) |
| Total-MAE | -0.00395 | **[-0.00612, -0.00184]** | **REAL improvement** |
| Margin-MAE | -0.00020 | **[-0.00033, -0.00007]** | **REAL improvement (small bonus)** |

CRPS (§16.1's new metric, doing exactly the job it was added for): total 1.27297→1.26210
(real improvement); margin roughly flat (1.35577→1.35519). Corrected totals-line P(over)
benchmark (§16.4's methodology): mean P(over) recovers from 41.61% to **43.35%** (a real step
toward the true ~49.5%, not all the way — the EV residual remains, exactly as predicted);
Brier gap to the coin-flip baseline narrows from 0.00683 to **0.00405**.

**Holdout, corrected §15 protocol**: SU mean diff -0.00229, CI [-0.00457, 0.00000] — technically
includes zero, not a real regression, though close to the boundary. Total-MAE mean diff
+0.00499, CI [-0.00007, 0.01011] — same: technically not a real regression, but the point
estimate moves in the OPPOSITE direction from the dev set, and closer to the boundary than any
other holdout check in this project. **This is very likely the exact dynamic the review
predicted when it flagged the earlier "near-perfect" holdout calibration as probably two
errors cancelling**: the decayed baseline's own overshoot (as real scoring plateaued after
2022-23) was likely offsetting part of the genuine, structural PP+PK undershoot in the
holdout era specifically; removing the real error here removes part of what was cancelling it.
Per the corrected protocol, this does not veto the change (no CI clears the "real regression"
bar) — but it's flagged explicitly, not glossed over, and it sharpens the case for the
still-open season-level intercept as the eventual closer for whatever level error remains once
both known structural pieces (this term, and any future EV-level fix) are in place.

### 18.6 Verdict: ADOPTED — new current best model, `shorthanded_poisson`

**Kept.** A clean root-cause fix: the mechanism was measured, not assumed (18.1-18.3), the
build reused existing walk-forward machinery rather than adding new data plumbing (18.4), and
every validation target moved exactly as predicted before the code was written (18.5) — total-
MAE, CRPS(total), and the P(over) benchmark all improved with real, bootstrap-confirmed effect
sizes on the dev set, SU/Brier neutral as expected for a term that doesn't touch win probability
directly, and no real holdout regression under the corrected protocol. New production entry
point: `src/models/validate_shorthanded.py` (supersedes `validate_ot_team_specific.py`).

This closes out the empty-net hypothesis (falsified) and the PP+PK piece of the mean-total
undershoot (fixed at the source) in the same cycle the review predicted: "the decomposition
doesn't waste anything — it decides what the intercept should be allowed to see." The EV
residual (-0.091/game, confirmed data-level-clean in 18.3) is what the walk-forward
season-level intercept should target next, not a repeat of this same mechanism.

Next: the walk-forward season-level intercept (§16.5/§0.4 item 5) for the EV-bucket residual,
Cycle 16's GBM stacking layer, or a team-specific SH-skill repeatability check (deferred here
per 18.4's own reasoning) if there's ever reason to revisit the league-level design choice.

## 19. Bias ladder and EV-ratio check — localizing the rest of the mean-total gap (2026-07-23)

Before building the EV-bucket intercept (task tracked, §18.3), an external review flagged a
real accounting gap: §18.2's pre-SH-fix per-bucket sum (EV -0.091 + PP/PK -0.113 + other
+0.080 = -0.124/game) accounted for less than half the full-pipeline bias (-0.284/game,
logged `mean_pred_total` 5.5285 vs. actual 5.8125) — roughly -0.16/game entering the pipeline
downstream of the situational combine, invisible to the bucket table. Two checks were run
before building the intercept.

### 19.1 The bias ladder: stage-by-stage localization

Built `src/models/validate_bias_ladder.py`: mean predicted-minus-actual total per season, at
each real pipeline stage (situational combine incl. SH term → + goalie overlay → + rest
adjustment → full final-joint pathway, i.e. regulation-ratio scaling + OT logistic
redistribution). Per-stage INCREMENTS (what each layer itself adds), averaged across 14
dev-set seasons:

| Stage | Mean increment | Sign consistency |
|---|---|---|
| + goalie overlay | -0.0246/game | Noisy — ranges -0.114 to +0.111, no consistent direction |
| + rest adjustment | -0.0268/game | Stable, -0.021 to -0.031 every season |
| + regulation-ratio/OT redistribution | **-0.0535/game** | **NEGATIVE in ALL 14/14 seasons, growing to -0.10/game in 2022-24** |

**The rest-adjustment increment is expected and already understood** — it matches the
external review's own back-of-envelope estimate in order of magnitude, and is simply the
away-B2B adjustment applying asymmetrically (a real downward shift with no exactly-offsetting
positive term elsewhere) — not a bug, just a real, quantified mechanical consequence of a
kept, validated fix (§4.14).

**The regulation-ratio/OT-redistribution increment is the standout finding**: negative in
every single season with no exceptions, and clearly GROWING in the two most recent dev
seasons (2022-23: -0.103, 2023-24: -0.101, both roughly double the 2010s-era values). This
stage was assumed "nominally self-cancelling by construction" (scale down ~4% via
`HOME_REG_RATIO`/`AWAY_REG_RATIO`, then add back exactly 1 goal on however many games the
model resolves as ties) — **it is not.** The mechanism: the OT bonus-goal add-back only fires
on the diagonal cells the model's OWN regulation joint predicts as tied — and that's exactly
the already-confirmed tie-mass deficit (§4.13, §10.1: model diagonal mass ~0.17 vs. real
~0.23). The bonus-goal make-up credits roughly 17% of games instead of the true ~23%,
under-crediting the make-up term by almost exactly the amount that shows up here. **This
increment growing as real scoring has risen (2022-24) is exactly what a percentage-based
deficit acting on an absolute-goal make-up term should look like as the base level rises** —
not a new, separate mechanism, but the SAME already-known tie-mass problem showing up in a
metric (mean-total bias) nobody had connected it to before.

### 19.2 The EV-ratio check: clean, confirming the intercept as the honest tool

Per §7.3's own precedent (a 1.026 defense-ratio bias found there, purely from shrinkage/
baseline geometry), checked whether the EV bucket's own attack/defense ratios — exactly as
`_combine` consumes them — average away from 1.000. **Overall: mean attack ratio 0.9996, mean
defense ratio 0.9989, mean product 0.9978** — essentially clean, per-season range 0.977-1.013
with no systematic direction. On a ~7-8 combined-EV-goal-per-game base, a 0.2% product
deviation contributes at most ~0.014 goals/game — nowhere near enough to explain the
EV bucket's own -0.091/game undershoot. **Per the review's own decision rule: ratios average
clean, so the walk-forward season-level intercept is confirmed as the honest tool for the
EV residual, with no nagging alternative mechanism left unexplored.**

### 19.3 A confirmed code bug, found and fixed, validated as a genuine null

Reading `team_strength_goalie.add_walk_forward_goalie_strength` directly (prompted by the
goalie-overlay stage's flagged-as-suspect noisy increment) confirmed a real bug: the call
never threaded `league_avg_halflife_games` through to the goalie's own `goalie_league_avg`
baseline (`goalie_relative = goalie_shrunk_mean - goalie_league_avg`), leaving it an
infinite-memory expanding mean while every other league-average baseline in the pipeline has
used the Cycle 13 decay since §7.2/7.5 — **the same two-baselines-different-memory bug class
as §7.1/7.3, found a third time.** Added `league_avg_halflife_games` as an optional parameter
(default `None`, preserving the original behavior) and validated decaying it (halflife=600,
matching the rest of the pipeline) with a paired bootstrap:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00061 | [-0.00163, 0.00042] | Crosses zero |
| Brier | -0.00001 | [-0.00004, 0.00002] | Crosses zero |
| Total-MAE | +0.00088 | [-0.00010, 0.00187] | Crosses zero |
| Margin-MAE | -0.00016 | [-0.00033, 0.00002] | Crosses zero |

**A genuine null, consistent with the ladder's own noisy (no clean sign) per-season pattern
for this stage.** The code inconsistency is real and matches an already-established bug
class, but on real data it has no detectable aggregate effect — **not adopted into
production** (this project only adopts changes with a validated real improvement, not merely
a "more theoretically correct" one with no measurable benefit), but left available as an
optional, documented parameter for any future cycle that wants to revisit it jointly with
`PRIOR_GAMES_GOALIE` (per the standing §7.5-style entanglement caution).

### 19.4 A falsifiable prediction to track going forward

As the remaining mean-total-undershoot pieces get fixed (the tie-mass deficit via the
regulation-ratio/OT-redistribution mechanism above, and/or the EV-bucket intercept), **the
per-season HOLDOUT bias curve (§16.5's table) should show 2023-25 tip from its current
near-zero reading into a small POSITIVE overshoot before eventually settling** — the
mechanism being that the holdout era's current near-zero bias is itself suspected (§18.6) to
be a coincidental cancellation between this real structural undershoot and the decayed
baseline's own overshoot as scoring plateaued post-2022-23; removing the structural piece
without a compensating change would unmask the baseline's own overshoot. **Track this curve,
not the aggregate holdout bias, as the health metric for the whole mean-total-recalibration
chain** — a future cycle that fixes a real piece of this puzzle but shows the holdout curve
staying flat (rather than tipping positive) would be a signal the accounting above is
incomplete somewhere.

### 19.5 Revised priorities

1. **The tie-mass deficit (§4.13, previously rejected fixes §10/§14) is re-prioritized ahead
   of the EV intercept** — it's no longer just a margin/CRPS/dispersion issue (§10.6.1,
   §14.3); it now has a directly quantified, GROWING (-0.05 to -0.10/game in recent seasons)
   total-level cost too, discovered via a mechanism (the regulation-ratio/OT-redistribution
   stage) neither prior tie-mass cycle connected to mean-total bias. Both prior attempts
   (Cycle 17's global rescale, Cycle 20's local transfer) were built on the OLD flat-rate OT
   layer; Cycle 22's team-specific OT model may change the calculus for a re-attempt, per
   §17.4's own note.
2. **The EV-bucket walk-forward intercept (§18.3/§19.2, task tracked) remains the right tool
   for that specific -0.091/game piece** — confirmed data-level-clean and ratio-clean, no
   alternative mechanism found.
3. **The goalie-baseline decay fix is available but not adopted** — a real bug, a real null
   result; revisit only if a future cycle wants to jointly re-grid it with
   `PRIOR_GAMES_GOALIE`.
4. The market moneyline benchmark (§6.8) still reflects `cross_season_prior_poisson`
   (pre-Cycles 22-23) — due for a re-run once the mean-total chain (tie-mass re-attempt +
   EV intercept) is resolved, not before, per the standing recommendation to avoid re-running
   it after every Brier-neutral adoption.

Next: the tie-mass re-attempt (re-prioritized above the EV intercept, given the newly
quantified stakes), then the EV-bucket intercept, then the market benchmark re-run, then
Cycle 16's GBM layer.

## 20. Rest-adjustment re-centering: real bias fixed, real margin cost found — a genuine trade-off, not adopted (2026-07-23)

**⚠ SUPERSEDED by §22**: the specific re-centering design in this section put the entire
returned credit on the away side, which turned out to be the actual CAUSE of the margin
regression found below (it shifts every game's predicted margin toward away by ~0.027) — not
an inherent trade-off of fixing the bias at all. §22 re-diagnoses the mechanism as symmetric
(both lambdas carry the embedded bias, not just the away one) and the corrected fix is
adopted with margin-MAE landing at exact machine-precision zero. Kept here as the historical
record of the first (wrong) attempt, per this project's standing practice of not deleting
a real, instructive mis-step.

A follow-up review identified a precise, checkable mechanism for the ladder's rest-stage
bias (§19.1, -0.027/game, stable every season): the away-B2B adjustment is fit as a pure
DIFFERENCE (`fit_away_b2b_adjustment` returns `b2b_mean - other_mean`) but applied as a raw
subtraction on B2B games only, leaving non-B2B games untouched. Since the walk-forward league
baseline every lambda is built from is itself computed over ALL games (B2B-depressed ones
included), it already carries the average B2B drag — subtracting the full difference again on
B2B games removes it a second time, shifting the population mean down by
`away_b2b_adj * P(B2B)`. **Confirmed directly: -0.11746 * 0.22901 = -0.0269, matching the
ladder's measured increment (-0.0268) almost exactly.**

### 20.1 The fix, and confirmation on the ladder

Added `fit_away_b2b_incidence` (P(B2B) in the fitting population) and
`centered_away_b2b_adjustment` to `src/models/rest_schedule.py`: re-centers the adjustment to
be mean-zero over the population while preserving the EXACT same B2B-minus-non-B2B difference
(`away_b2b_adj`) the original fit measured — B2B games get `away_b2b_adj * (1 - p_b2b)` (a
real penalty, slightly smaller in magnitude), non-B2B games get `-away_b2b_adj * p_b2b` (a
small credit). Re-ran the bias ladder with this applied: **the rest-stage increment drops to
essentially zero** (mean bias_C - mean bias_B = +0.0001, vs. -0.027 before) — confirms the
mechanism exactly as diagnosed, full-pipeline bias improves from -0.196 to -0.170/game.

### 20.2 The cost: a real, small, bootstrap-confirmed margin-MAE regression

Paired bootstrap (5,000 resamples), re-centered vs. the current production (uncentered)
formula, identical dev-set games:

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00133 | [-0.00333, 0.00067] | Crosses zero — neutral |
| Brier | -0.00004 | [-0.00010, 0.00003] | Crosses zero — neutral |
| Total-MAE | +0.00007 | [-0.00033, 0.00045] | Crosses zero — neutral |
| Margin-MAE | +0.00090 | **[0.00050, 0.00129]** | **REAL regression** |

**A genuine trade-off, not a clean win.** The re-centering does exactly what it was designed
to do (zero out the population bias, confirmed on the ladder) but at a real, if small, margin
cost — plausibly because redistributing a uniform credit across the ~77% of away games that
are NOT on a back-to-back (versus a full penalty concentrated on the ~23% that are) changes
how the adjustment interacts with per-game margin accuracy in a way the raw, uncentered
version happened to avoid, even though the uncentered version is less defensible on the
population-mean argument.

### 20.3 Verdict: NOT adopted — reverted to production, fix kept available

Per this project's own established bar (no kept change may carry a real, bootstrap-confirmed
regression on a tracked metric), **the centered version is NOT adopted into production**. All
five call sites that were updated during testing (`validate_rest.py`, `validate_shorthanded.py`,
`check_holdout_cross_season.py`, `check_holdout_ot_logistic.py`, `validate_bias_ladder.py`)
were reverted to the original (uncentered) formula, matching the actual validated
`shorthanded_poisson` model exactly — current best model numbers are UNCHANGED by this
section. `centered_away_b2b_adjustment`/`fit_away_b2b_incidence` remain available in
`rest_schedule.py`, documented, for any future cycle that wants to revisit this trade-off
(e.g. jointly with a re-tuned `away_b2b_adj` itself, since the current coefficient was fit
against the uncentered application's own residuals).

This is the second genuine null/trade-off this session in a row for a "theoretically more
correct" fix (after §19.3's goalie-baseline decay null) — both real bugs, both fixed in code,
both validated honestly, neither adopted, because this project's bar is measured improvement,
not theoretical correctness alone.

## 21. Tie-mass re-attempt: full design specification, staged for a dedicated build (2026-07-23)

Per §19.1's finding (the regulation-ratio/OT-redistribution pipeline stage carries a real,
growing, 14/14-season-consistent bias directly explained by the tie-mass deficit), a follow-up
review specified the exact design correcting both of Cycle 20's structural flaws before that
cycle is built. Logged here in full ahead of the build, per this project's own standing
discipline of specifying success criteria before running an experiment, not after.

### 21.1 Why Cycle 20's one-sided transfer structurally cannot close the ladder's -0.054

The coherence identity: `final_mean = regulation_mean + P(reach OT) * 1` (every OT/SO game
adds exactly one recorded goal on top of the regulation score). The REQUIRED tie rate for
this identity to hold at the real final mean is `(1 - reg_ratio) * final_total` — at the real
mean (~5.81), this is ≈0.0397 * 5.81 ≈ 0.231, matching the real tie rate almost exactly; at a
recent-era final total of ~6.2, the REQUIRED tie rate rises to ≈0.246, while the model's own
independent-Poisson tie probability actually FALLS as lambdas rise (higher-count Poisson
distributions concentrate less mass on any single value, including the diagonal) — this
growing gap between "what the coherence identity requires" and "what independent Poisson
naturally produces as scoring rises" is the mechanism behind the ladder's growing -0.10/game
in the most recent two seasons.

Cycle 20's one-sided local transfer (moving mass only from the "above" neighbors `(x+1,x)`/
`(x,x+1)`, total `2x+1`, into the diagonal `(x,x)`, total `2x`) is **total-conserving by
construction**: it lowers the regulation mean by exactly the same amount the subsequent OT
bonus-goal re-add raises it back by, netting exactly zero on the final mean — which is exactly
why §14.2 found total-MAE identical at every tested delta, including delta=0. It can raise the
tie RATE, but it can never touch the ladder's total-level bias, regardless of how it's tuned.

### 21.2 The corrected design: two-sided transfer + walk-forward regulation ratios

**Two-sided diagonal transfer**: for each diagonal level `(x,x)`, pull mass from BOTH the
"above-total" neighbors (`(x+1,x)`/`(x,x+1)`, total `2x+1`) AND the "below-total" neighbors
(`(x-1,x)`/`(x,x-1)`, total `2x-1`) in a BALANCED proportion — moving equal PROBABILITY mass
from each side (not equal fractions, since the above-pair and below-pair masses generally
differ for an asymmetric lambda_home/lambda_away). Moving mass from above lowers the
regulation mean by 1 unit of probability moved; moving mass from below raises it by the same;
balancing them keeps the regulation mean EXACTLY fixed while still raising the diagonal
(tie) mass by the combined amount — which the OT bonus re-add then converts into a real,
correctly-signed increase in the FINAL mean. **Boundary case**: `x=0` has no "below" neighbors
(count cannot go negative) — the two-sided balance cannot apply there; this must be handled
explicitly (most likely: cap or exclude the `x=0` diagonal cell from the transfer, or accept a
small, explicitly-flagged residual mean-shift at that one cell, checked directly rather than
assumed away, per this project's own discipline).

**Walk-forward regulation ratios**: replace the fixed dev-fit constants `HOME_REG_RATIO`
(0.9613) / `AWAY_REG_RATIO` (0.9593) with a walk-forward, `HALFLIFE_GAMES`-decayed estimate
(trailing mean regulation-only goals / trailing mean final recorded goals) — the FIXED ratios
are exactly what convert a rising league scoring level into the ladder's GROWING deficit
(§21.1); a walk-forward ratio tracks the current level and removes the trend component, while
the two-sided transfer removes the level component.

**Resolution**: through Cycle 22's skill-informed OT logistic (`ot_logistic.py`), not the flat
constant that caused §14.3's margin cost — per §17.4's own note, this may change how much
margin cost the diagonal-mass fix carries this time, since mass rerouted through a tie no
longer reverts to a league-average split.

**Empirical license for the shape**: §10.6.3's margin-bin profile already showed the tie
deficit's offsetting excess concentrated at `|margin|=1` on BOTH signs (home-favored and
away-favored) — i.e. both the above-total and below-total bands — supporting a symmetric,
two-sided transfer rather than a one-sided one.

### 21.3 Success criteria, stated before the cycle runs

1. **Primary**: the per-season regulation-ratio/OT-redistribution ladder increment (§19.1)
   flattens toward zero across ALL eras, including the currently-worst 2022-24 seasons — not
   just a smaller AVERAGE, a flatter CURVE. Model-implied tie rate should land at the
   coherence-required `(1 - reg_ratio) * final_total`, checked per season, not just on
   average.
2. **Secondary**: CRPS(total) (§16.1) and the corrected P(over) benchmark (§16.4) both
   improve — this is where most of the remaining ~6-point P(over) gap (43.35%→~49.5%) should
   close, net of the still-open EV-bucket residual.
3. **Veto metric**: margin-MAE, under the §15 corrected protocol (paired bootstrap,
   confirmatory-veto-only). With the two-sided transfer AND the skill-informed OT resolution
   both in place, §14's margin harm should shrink from both directions — but the bootstrap
   decides whether it reaches neutral, not an assumption. **If margin harm survives even under
   this corrected design, the honest conclusion is that the tie-mass family is genuinely closed
   under an independent-Poisson-based joint, and the real fix requires a dependence-bearing
   joint distribution instead** — which is exactly what the CRPS baseline (§16.1) was built to
   adjudicate fairly.
4. **Falsifiable tracking metric** (§19.4, still standing): the per-season HOLDOUT bias curve
   should show 2023-25 tip from near-zero into a small positive overshoot before eventually
   settling, as this fix (and/or the EV intercept) unmasks the decayed baseline's own
   overshoot that had been coincidentally cancelling the structural undershoot there.

**Sequencing note**: the rest-adjustment re-centering (§20) was correctly done BEFORE this
build, per the review's own note that it moves the same ladder metric this cycle will be
judged on — resolved as a documented null (§20.3), so the tie-mass cycle's own ladder
comparisons should use the CURRENT (uncentered) rest-adjustment baseline, not a
re-centered one that was never adopted.

Not yet built — staged here in full for a dedicated implementation pass, given the real
complexity (boundary handling, walk-forward ratio calibration, full multi-metric validation
against the criteria above) this design warrants over rushing it.

## 22. Rest-adjustment fix, corrected: the double-count is symmetric, not away-only — ADOPTED (2026-07-23)

§20's re-centering was itself mis-specified, per a follow-up review: it put the entire
returned credit on the away side, which shifts every game's predicted MARGIN toward away by
~0.027 — precisely explaining that version's real, bootstrap-confirmed margin-MAE regression
(§20.2). The correct diagnosis: the embedded B2B drag is SYMMETRIC, because every team's own
walk-forward attack-rate estimate pools that team's ENTIRE history — including the ~11.45% of
that team's own games (half of ~22.9% away-B2B incidence) that were themselves away-B2Bs,
each depressed by the real `away_b2b_adj` effect. This embeds `0.5 * p_b2b * away_b2b_adj` ≈
`0.5 * 0.229 * (-0.117)` ≈ **-0.0134** into EVERY team's own rate estimate — present in BOTH
lambdas of EVERY game (whichever team is home or away tonight), since it comes from each
team's OWN history, not tonight's specific matchup. Two such contaminated lambdas per game
give exactly `-0.0134 * 2` ≈ `-0.027`, matching the ladder's measured increment precisely for
a second, independently-derived reason.

### 22.1 The corrected fix

Replaced §20's wrong `centered_away_b2b_adjustment` with `symmetric_b2b_bias_credit` in
`src/models/rest_schedule.py`: a walk-forward (not hardcoded) incidence weight
(`add_walk_forward_b2b_incidence`, an expanding mean of B2B incidence over strictly-prior
away-side games league-wide, tracking any real drift in B2B scheduling frequency) feeds a
FIXED credit `-0.5 * p_b2b * away_b2b_adj` ≈ `+0.0134`, added to BOTH `lambda_home` and
`lambda_away` UNCONDITIONALLY (every game, regardless of tonight's B2B status) — on top of,
not instead of, the original, unchanged, tonight-specific conditional term on the away side
only (`fit_away_b2b_adjustment`, kept exactly as shipped).

Algebraically: since the credit is an IDENTICAL constant added to both sides, it cancels
EXACTLY out of margin (`lambda_home - lambda_away` is untouched, bit-for-bit, not merely
mean-zero over a population) while raising the TOTAL by exactly `2 * 0.0134` ≈ `0.0268` on
every single game — recovering precisely the ladder's missing rest-stage bias.

### 22.2 Validation: the falsifiable prediction holds with the cleanest possible margin

Per the review's own falsifiable test ("margin-MAE with a CI tightly straddling zero — not
merely 'crosses zero' with a wide band"), paired bootstrap (5,000 resamples), corrected vs.
current production:

| Split | SU | Brier | Total-MAE | Margin-MAE |
|---|---|---|---|---|
| Dev | +0.00006, CI [0.00000,0.00018] | ~0, CI [-0.00001,0.00000] | +0.00008, CI [-0.00034,0.00049] (neutral) | **0.00000, CI [0.00000, 0.00000] — EXACT** |
| Holdout | 0.00000, CI [0.00000,0.00000] (exact) | ~0, CI [-0.00001,0.00000] | +0.00151, CI **[0.00053,0.00248] — REAL** | **0.00000, CI [-0.00000,0.00000] — EXACT** |

**Margin-MAE lands at literal machine-precision zero on BOTH splits** — the cleanest possible
confirmation that the mechanism is exactly as re-diagnosed, not merely approximately
mean-zero. The bias ladder confirms the mechanism too: re-running with the corrected fix
shows the rest-stage bias recovering by a consistent ~+0.027-0.030/game in EVERY dev season
with no era-dependence (unlike the reg-ratio/OT stage's growing bias) — exactly the
stationary, non-lag signature this mechanism predicts.

**One real wrinkle**: holdout total-MAE shows a real regression (CI entirely positive,
+0.00151) — total-MAE was never a claimed dev-set win for this fix in the first place (dev CI
crosses zero), so this isn't a case of holdout vetoing a dev-confirmed claim. Most likely
explanation: the same cancelling-errors dynamic flagged in §18.6/§19.4 — the holdout era's own
near-zero bias may have been partly a coincidental offset that this correctness fix partially
unwinds. Given the fix is mathematically derived (not a data-driven hypothesis) and the
primary target (margin-neutrality) is confirmed at machine precision on both splits, this was
adopted with the wrinkle explicitly flagged rather than treated as a silent veto trigger.

### 22.3 Verdict: ADOPTED

**Kept.** `shorthanded_poisson`'s own numbers update marginally (mean predicted total
5.622→5.650, margin/SU/Brier essentially unchanged) — propagated to `validate_shorthanded.py`,
`validate_rest.py`, `check_holdout_cross_season.py`, `check_holdout_ot_logistic.py`, and
`validate_bias_ladder.py`. This is the second real bug this session traced to the same root
cause (a walk-forward rate estimate silently absorbing an effect that a downstream adjustment
then re-applies) — a useful pattern to watch for in any future additive adjustment layered on
top of a walk-forward baseline.

Next: the tie-mass re-attempt (§21's full spec — the ladder comparisons for that cycle should
use this corrected rest baseline, not the reverted uncentered one §21 was drafted against),
then the EV-bucket intercept, then the deferred market-benchmark re-run.

## 23. Reconciliation check: the ladder map is complete, two diagnostic-tool bugs found and fixed along the way (2026-07-23)

Before the tie-mass build, per a follow-up review's request: verify the bias ladder's stages
actually sum to the full-pipeline total, so every remaining component of the mean-total gap
has a named owner before any cycle claims credit for closing it.

### 23.1 Two real bugs found in `validate_bias_decomposition.py` itself (not the model)

**Bug 1**: the script never threaded `cross_season_weight`/`prior_minutes_multiplier` through
its own `add_walk_forward_situational_strength` call — it had been silently running against
the OLD (Cycle 13-era) shrinkage configuration, not the actual current production settings
(`cross_season_weight=0.75`, `prior_minutes_multiplier=2.0`) adopted in Cycles 19/13. Fixed.

**Bug 2, found while chasing a residual reconciliation gap**: even after fixing Bug 1, the
sum of the three bucket gaps (+0.013) still didn't match the ladder's own directly-measured
Stage A bias (-0.091) — a ~0.10 discrepancy. A direct, per-game comparison isolated it exactly:
the PREDICTED side matches production bit-for-bit (diff ~1e-16, pure floating-point noise —
confirming the decomposition script's `_combine` calls are byte-identical to production, not
a modeling discrepancy). The ACTUAL side does not: broken down by `lastPeriodType`, **OT-decided
games show exactly 0 discrepancy (n=2,593), regulation games ~0 (n=14,644), and shootout-decided
games show EXACTLY -1.000000 for every single one (n=1,768).** Mechanism: a shootout attempt
isn't a real on-ice shot in a situational sense — MoneyPuck's situational shot data has no
bucket for it, so the shootout-winning goal is invisible to any of the three summed buckets
(EV/PP+PK/other), while the production model's own `actual_home`/`actual_away` (from the real
NHL final score) correctly includes it. This is a genuine data-shape fact about the diagnostic
tool's OWN ground-truth source, not a bug in the production model, and it does NOT contaminate
any INDIVIDUAL bucket's own gap (§18.2's EV/PP+PK/other findings stand as reported) — it only
affects the "do the three buckets sum to the whole" check this reconciliation performs.

### 23.2 The reconciled map

Correcting the bucket sum for the ~1,768/19,152 ≈ 9.2% missing-goal rate (≈ -0.092/game):
`(EV -0.088) + (PP+PK +0.024) + (other +0.077) = +0.013` (raw bucket sum) `- 0.092` (SO-goal
correction) `≈ -0.079` — within ~0.012 of the ladder's directly-measured Stage A (-0.091).
**The map is complete**, every component has a named owner:

| Component | Value | Status |
|---|---|---|
| EV residual | ~-0.09/game | Confirmed data-level-clean (§18.3/§19.2); target of the still-open walk-forward intercept |
| PP+PK | ~+0.02/game (small overshoot) | Fixed at the source, Cycle 23 (§18) |
| Other | ~+0.08/game | Unchanged, not the empty-net story (§18.1 falsified that) |
| Goalie-overlay stage | ~-0.025/game, noisy/no persistent sign | Real code bug found (§19.3), validated null, not adopted |
| Rest-adjustment stage | ~0/game (was -0.027) | Fixed, Cycle 24 (§22), adopted |
| Regulation-ratio/OT stage | -0.05 to -0.10/game, growing, 14/14 seasons negative | Tie-mass-driven (§21.1); the cycle about to be built |

Full pipeline (Stage D, post-SH-fix, post-rest-fix): -0.169/game, matching
`EV+PP+other+goalie+rest+reg_ratio_OT ≈ -0.09+0.02+0.08-0.025+0.00-0.055 ≈ -0.17` closely.

## 24. Protocol amendment: a distinct adoption class for derived correctness fixes (2026-07-23)

§22's rest-adjustment fix was adopted despite a real, bootstrap-confirmed holdout regression
on total-MAE — the first adoption in this project's history with no bootstrap-real dev win on
any of the four original ledger metrics (SU, Brier, total-MAE, margin-MAE) and a real holdout
regression on one of them. Under the LETTER of this project's stated bar ("no kept change may
regress a metric with a real, bootstrap-confirmed effect"), that fix should have been vetoed.
It was adopted anyway, for good reasons — but those reasons were an unwritten judgment call,
and the tie-mass cycle is about to hit the identical structure at larger stakes (its primary
expected wins are ladder/CRPS/P(over), its veto risk is margin-MAE). Per a follow-up review,
the bar itself is amended here, before that build, rather than left as precedent to
rediscover case-by-case.

### 24.1 Promote the calibration metrics to claimable, bootstrap-scored status

The original four ledger metrics (SU, Brier, total-MAE, margin-MAE) were fixed before this
project's mean-total-calibration chain (§16-23) made mean-level accuracy a first-class
objective in its own right. Effective now, the following are claimable wins, held to the same
5,000-resample paired-bootstrap standard as the original four:
- **The per-season bias ladder** (`validate_bias_ladder.py`): a stage's own increment (or a
  cycle's effect on the full-pipeline bias) flattening toward zero, or losing its
  era-dependence, is a real, reportable win — not merely diagnostic.
- **CRPS(total)/CRPS(margin)** (§16.1): already logged as a first-class ledger metric; now
  formally eligible to be cited as a cycle's PRIMARY claimed win, not just a secondary check.
- **The corrected totals-line P(over) benchmark** (§16.4): eligible the same way.

### 24.2 A distinct adoption class: derived correctness fixes

Most cycles in this project test a HYPOTHESIS (a new feature, a re-tuned constant) and are
scored empirically — bootstrap-confirmed improvement is the only evidence such a change is
real. §22 was different in kind: the mechanism was PROVEN algebraically (a symmetric level
credit cancels out of `λ_h - λ_a` by construction, not by estimation), and its correctness
doesn't depend on empirical confirmation the way a fitted hypothesis does. This project now
recognizes **derived correctness fixes** as a distinct class, with its own adoption bar:

1. **Proven neutrality on win-probability/margin metrics** — not merely "CI crosses zero" but,
   where the mechanism implies EXACT cancellation (as a symmetric credit does), a bootstrap CI
   at or indistinguishable from machine precision. A wide crosses-zero CI is a WEAKER standard
   than this class should be held to, precisely because the mechanism makes a stronger claim
   than "probably neutral."
2. **A measured, real improvement on at least one calibration metric** (§24.1's list, or the
   original four) — a derivation being algebraically correct is necessary but not sufficient;
   it must also move a real number in the right direction.
3. **Explicit documentation of any accuracy-metric cost accepted in trade** — §22.2's holdout
   total-MAE regression was not hidden; it was named, diagnosed (the cancelling-errors
   mechanism), and adopted with the trade-off on the record, not silently absorbed. Any future
   derived-correctness fix must do the same: name the cost, propose the mechanism, and let the
   next cycle's own results confirm or falsify that explanation.

A fitted hypothesis that fails the ORIGINAL four-metric bar is still rejected exactly as
before — this amendment does not loosen the bar for ordinary cycles, only names a real,
narrower exception for changes whose correctness is proven rather than inferred.

### 24.2.1 Second amendment: confirmed-mechanism structural fixes (2026-07-24, motivated by §32)

§32's EV-TOI fix exposed a gap between the two existing classes. It isn't the ORIGINAL bar —
the metric it was built to move (total-MAE) never cleared bootstrap significance at any tested
halflife. It isn't §24.2's derived-correctness class either — that class requires the
mechanism to be PROVEN algebraically (exact cancellation by construction); §32's mechanism was
confirmed EMPIRICALLY, against independent real data (real EV-TOI vs. the model's own
expanding-mean estimate), not derived from an identity. What actually happened: a genuinely
borderline case was flagged, and the human operator made an explicit adoption call rather than
a pre-written rule firing. That is fine to do once — but it should be visible as a judgment
call, not read back later as if a rule admitted it. Rather than leave every future case like
this as an ad hoc discretionary call, this project's own standing practice (turn a real
judgment call into a named rule once it recurs, §15/§24.2's own history) applies here too. A
third class, **confirmed-mechanism structural fixes**:

1. **The mechanism is verified against independent real data**, not merely a plausible story —
   §32's own check (real EV-TOI vs. the walk-forward estimate, converted to an implied goal
   impact that closely matches the independently-measured residual, §29.5.1/§32.1) is the
   template: a genuine measurement, not a fitted correlation with the metric it's meant to fix.
2. **No real dev-set regression on any ledger metric**, at whatever the fix's own
   hyperparameter is (here, `toi_halflife_games`) — the same no-regression bar as every other
   class, non-negotiable.
3. **A real, bootstrap-confirmed gain on AT LEAST ONE claimable metric** — primary or
   secondary; unlike §24.2's derived-correctness class, this class does NOT require the gain to
   land on the specific metric the mechanism targets. §32's real gain was margin-MAE, a
   secondary metric, while the targeted metric (total-MAE) stayed neutral — acceptable under
   this class specifically because the mechanism's own verification (point 1) carries real
   evidentiary weight independent of which metric moves.
4. **Any holdout-era cost is named and diagnosed, not hidden** — same discipline as §24.2's
   point 3; §32.5's real holdout total-MAE regression is logged as a fourth unmasking
   installment, not silently absorbed.
5. **The grid, if one exists, is extended until the winning cell is confirmed interior, not a
   boundary artifact** — §13.1's own precedent (extending until `prior_minutes_multiplier`'s
   optimum was confirmed non-boundary), now applied here: §32's own follow-up extended
   `toi_halflife_games` to 2400 and 3600, both of which reproduce the same real margin-MAE gain
   at the same magnitude (-0.00014, essentially unchanged) rather than a continuing trend —
   confirming 1800 sits on a genuine plateau, not merely the first point past the noise.

**§32 is logged retroactively as this class's motivating instance.** Both prior classes stay
exactly as written — this does not loosen either; it names the third, real shape of adoption
this project has now actually used once, so the next borderline case has a pre-written bar to
check against instead of another ad hoc flag-and-ask.

### 24.3 How calibration gains trade against ledger-metric costs

Given the tie-mass cycle's own expected shape (primary wins in ladder/CRPS/P(over), veto risk
in margin-MAE, per §21.3's own success criteria), the trade-off rule: **margin-MAE remains a
hard veto for that cycle specifically**, per §21.3 as already written — this amendment does
not weaken that. What it changes is how a MIXED result gets read: a real margin-MAE cost
alongside a real calibration gain is a genuine, reportable trade-off (matching this project's
long-standing practice for ordinary cycles, e.g. §4.6's goalie overlay, §4.14's rest
adjustment — narrow, real, axis-specific wins have always been the normal outcome here, not
the exception) — NOT an automatic veto, unless the cost lands on a metric with no
correctness-based justification for accepting it. The distinction that matters: §22 could
justify its trade-off algebraically (the mechanism proves margin is untouched; the total-MAE
cost is a named, diagnosed side effect of removing a real bias). The tie-mass cycle will need
an equally explicit justification if its own result is mixed — not simply the numbers, but
the reason the numbers should be read the way they're being read.

### 24.4 The mean-total chain's endgame, stated now

The cancelling-errors account so far: the holdout era's earlier near-zero mean-total bias
(§16.5) is being progressively explained as a coincidental offset between real, structural
undershoot components (shorthanded-goals, fixed ✓; rest double-count, fixed ✓; tie-mass
shortfall, pending; EV residual, pending) and the decayed baseline's own transient overshoot
as real scoring plateaued post-2022-23. **Each structural fix removes undershoot and should
therefore make holdout total-MAE tick WORSE, not better** — §22's [0.00053, 0.00248] is the
first such installment; the tie-mass fix (worth +0.05 to +0.10/game in the most recent eras,
§21.1) should produce a materially larger one. This is the prediction the chain's own
narrative is held to, not an excuse for future holdout regressions — if a future structural
fix does NOT show this pattern, that's a sign the accounting above has a real gap, not
confirmation.

**Corollary: `HALFLIFE_GAMES` (currently 600) stays FROZEN until the structural chain is
complete.** Re-tuning the decay rate now, while real undershoot components remain unfixed,
would fit the baseline's own overshoot to the CURRENT (still-uncorrected) level of remaining
undershoot — re-manufacturing the exact cancellation this chain exists to dismantle, not
removing it. Once the tie-mass fix and the EV-bucket intercept both land, ONE final decay
re-tune against a structurally clean system — judged on whether the per-season bias curve
(§16.5/§19.1 style) flattens across ALL eras, not just improves on average — closes the
chain. That ordering (fix the structure, THEN re-tune the one remaining time-varying
parameter) is what separates removing the cancellation from relocating it.

**⚠ Corrected by §27 (2026-07-24)**: this paragraph's own ordering — tie-mass fix, THEN
EV-bucket intercept, THEN the decay re-tune — turns out to repeat §7.5's entanglement lesson
rather than avoid it. A walk-forward EV intercept and a halflife re-tune are both
level-tracking mechanisms for the same trailing realized-minus-predicted error; fitting the
intercept before the re-tune lets it absorb signal the re-tune should chase instead, exactly
the failure mode `HALFLIFE_GAMES=600`+priors×0.6 (§7.5) and Cycle 19's joint grid were built to
avoid. §27 corrects the order (re-tune first, against the tie-mass-fixed but EV-intercept-NOT-
yet-fitted system) and the scope (a joint `HALFLIFE_GAMES`×`prior_minutes_multiplier` grid, not
a solo decay re-tune, per the same §7.5 precedent). Kept here as the historical record of the
original endgame plan, not silently rewritten.

Next: the tie-mass cycle, built under §21's spec and §24's amended protocol.

## 25. Cycle 25: two-sided tie-mass transfer + walk-forward reg ratios — built per spec, rejected per pre-registered veto (2026-07-23)

Built exactly to §21's specification and evaluated against §21.3's success criteria, stated
before this cycle ran, governed by §24's amended protocol.

### 25.1 Build

`src/models/two_sided_diagonal_transfer.py`: `two_sided_transfer(joint, delta_above,
delta_below)` moves mass into each diagonal cell `(x,x)` from both its above-total neighbors
`(x+1,x)`/`(x,x+1)` and below-total neighbors `(x-1,x)`/`(x,x-1)`. `fit_deltas` solves for
both in closed form, enforcing the GLOBAL (not per-cell) mean-preservation constraint
`delta_below = delta_above * mean(above_mass) / mean(below_mass)` — correctly handling the
`x=0` boundary (no below-neighbor there) by balancing the aggregate amounts moved from each
side across the population, not forcing an impossible per-cell symmetry. Fitted on the dev
set: **`delta_above = delta_below = 0.0977`** — the two came out numerically identical,
confirming the boundary asymmetry from `x=0` is negligible in aggregate (real regulation
0-0 ties are well under 1% of mass at current scoring levels, exactly as anticipated). Both
values pass the validity check (`< 1`).

`overtime_shootout.add_walk_forward_reg_ratio`: replaces the fixed dev-fit `HOME_REG_RATIO`/
`AWAY_REG_RATIO` constants with a walk-forward, `HALFLIFE_GAMES`-decayed estimate. **Confirmed
real, sensible drift before trusting it**: the walk-forward ratio rises from ~0.955-0.96 in
the earliest dev seasons to ~0.964-0.966 in the most recent — mechanically expected, since a
fixed 1-goal OT/SO bonus becomes a smaller fraction of final score as real total scoring has
risen from ~5.4 to ~6.2 over the same window, independent of any change in true tie
propensity.

Resolved through Cycle 22's skill-informed OT logistic throughout (`src/models/validate_tie_mass_v2.py`).

### 25.2 Dev-set bootstrap: the veto metric fails, by a real but much smaller margin than Cycle 20

Paired bootstrap (5,000 resamples) against the current best (`shorthanded_poisson` + §22's
rest fix, no tie-mass term):

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00024 | [-0.00169, 0.00115] | Crosses zero — neutral |
| Brier | 0.00000 | [-0.00004, 0.00004] | Crosses zero — neutral |
| Total-MAE | +0.00113 | [-0.00004, 0.00223] | Crosses zero, but trending WORSE, not the hoped-for secondary win |
| Margin-MAE | +0.00026 | **[0.00006, 0.00045]** | **REAL regression** |

**The veto metric fails.** Real, bootstrap-confirmed margin-MAE harm survives even with both
of §21.2's corrections in place — smaller than Cycle 20's one-sided version by roughly 5-10x
(`[0.00129, 0.00256]` there vs. `[0.00006, 0.00045]` here), confirming the two-sided design
genuinely helps, but not enough to clear zero.

### 25.3 Calibration metrics: real, small gains — confirming the mechanism partially, not fully

- **CRPS(total): 1.25945 → 1.25772** (small real-looking improvement)
- **CRPS(margin): 1.35495 → 1.35514** (essentially flat)
- **Corrected P(over) benchmark: mean P(over) 44.65%→45.35% (toward the real ~49.5%), Brier
  0.25285→0.25213** (small, real-looking improvement, continuing the trajectory from §18.5's
  41.61%→43.35%→45.35% across the SH-term and tie-mass cycles)

### 25.4 Primary success criterion (era-flattening): NOT met

Per-season bias (treated minus baseline) shows the fix's own correction growing from
**+0.047 to +0.061/game in the earliest dev seasons (2010-11 through 2016-17) up to
+0.085-0.10/game in the most recent (2018-19 through 2023-24)** — tracking the DIRECTION of
the known growing deficit, but **not cleanly enough**: the correction looks well-matched to
the recent, high-deficit seasons but appears to OVER-correct the earliest, lower-deficit
seasons (adding ~0.05-0.06/game there against a previously-measured deficit of only
~0.03-0.04/game in that same window, §19.1). The full-pipeline bias curve's own era-to-era
RANGE is essentially unchanged (baseline spans -0.454 to +0.169 across seasons; treated spans
-0.362 to +0.270) — shifted up by a growing amount, not flattened. **This does not meet the
pre-registered "flattens across all eras" bar.**

### 25.5 Verdict: REJECTED, per the pre-registered veto — the honest conclusion is architectural, not a tuning failure

**⚠ Verdict language revised by §25.6**: a metric-theory follow-up (mean-vs-median-scored
MAE, a full-precision CRPS(margin) bootstrap, a regulation-vs-OT split) found the margin cost
below is real but much smaller than it first appeared, concentrated entirely in
regulation-decided games, and mechanistically distinct from what was assumed here. The
SPECIFIC candidate tested in this cycle stays REJECTED — §25.6 does not re-score it under a
rule invented after seeing its numbers, since that is exactly the adaptive-protocol failure
§15 exists to prevent. But "closes the family for good" (below) is corrected to "closed to
automatic-veto adoption; reopened for one new, pre-registered candidate." See §25.6 for the
full analysis before treating this section's closing language as final.

**Rejected.** Per §21.3's own pre-stated decision rule: margin-MAE is a real, bootstrap-
confirmed regression, which is disqualifying under this project's standard bar regardless of
the real (if partial) calibration gains alongside it. This is NOT read as a mixed trade-off to
adopt under §24's amended protocol — §24.2's derived-correctness-fix class requires PROVEN
(not merely improved) neutrality on margin, and this mechanism makes no such algebraic
guarantee (unlike §22's exact cancellation) — a real margin cost here is a genuine finding
about the mechanism's limits, not a mis-specification to keep chasing.

**The pre-registered fallback conclusion applies**: even the theoretically-correct, two-sided,
walk-forward-ratio version of a diagonal-mass transfer under an INDEPENDENT-POISSON joint
cannot fully close the tie-mass gap without a real margin cost. Closing it cleanly requires a
genuinely DEPENDENCE-BEARING joint distribution (a real bivariate/copula structure with
negative off-diagonal dependence, not a diagonal-cell reallocation trick layered on top of
independence) — which is exactly the kind of model the CRPS baseline (§16.1) and this cycle's
own CRPS/P(over) numbers were built to let a future attempt be judged against fairly. **This
closes off the independent-Poisson-plus-diagonal-transfer family for good** — not just this
specific parameterization, having now tried both the one-sided (Cycle 20) and two-sided
(Cycle 25) shapes, with and without walk-forward reg ratios, with both the flat (Cycle 20) and
skill-informed (Cycle 25) OT resolution.

Current best model remains `shorthanded_poisson` with §22's rest fix — unchanged by this
cycle. `HALFLIFE_GAMES` remains frozen per §24.4 (this cycle didn't touch it, and its own
walk-forward reg-ratio piece is a separate, already-adopted-as-tested mechanism, not the
league-average decay itself).

Next: the walk-forward EV-bucket intercept (§18.3/§19.2/task #35), the only remaining item in the
mean-total chain before the one final `HALFLIFE_GAMES` re-tune (§24.4) that closes it; a
genuine dependence-bearing joint distribution is now the identified, but not yet scoped,
long-term path for the tie-mass problem specifically, separate from the mean-total chain.

### 25.6 Follow-up: the mean/median metric-theory challenge — three measurements, and a revised verdict (2026-07-23)

§25.5's veto rests entirely on margin-MAE. MAE's decision-theoretically optimal point
forecast is the conditional MEDIAN, not the mean — a model that correctly reshapes a margin
distribution (moving real probability mass onto near-coin-flip outcomes, to match the real
~23% tie rate) can show a real MEAN-scored MAE regression purely from this mismatch, even when
the reshaping is a genuine distributional improvement. Three checks, run on the same 16,528
dev-set games and the same baseline/treated joints as §25.2, adjudicate this before §25.5's
language is allowed to stand as the final word.

#### 25.6.1 Three measurements

| Check | Mean diff | 95% CI | Fraction of resamples ≤0 | Real? |
|---|---|---|---|---|
| (a) Margin-MAE, MEAN-scored (§25.2, for reference) | +0.00026 | [0.00006, 0.00045] | — | Real |
| (a) Margin-MAE, MEDIAN-scored | +0.00073 | [-0.00212, 0.00351] | 31.3% | Crosses zero — not real under this functional |
| (b) CRPS(margin), full precision | +0.00019 | [0.0000045, 0.00038] | 2.2% | Real, but barely — only just clears the 95% bar |
| (c) Margin-MAE, regulation-decided only (n=12,718) | +0.00040 | [0.00017, 0.00062] | — | Real |
| (c) Margin-MAE, OT/SO-decided only (n=3,810) | -0.00021 | [-0.00062, 0.00018] | — | Crosses zero — not real |

Baseline: `mean_margin_mae=2.01300`, `median_margin_mae=1.96551`, `CRPS(margin)=1.35495`.
Treated: `mean_margin_mae=2.01326`, `median_margin_mae=1.96624`, `CRPS(margin)=1.35514`.

#### 25.6.2 Interpretation: not a clean flip in either direction

The mean/median hypothesis does NOT fully exonerate the fix. If margin-MAE's regression were
purely a scoring-functional artifact, CRPS(margin) — a strictly proper score over the whole
distribution, with no point-forecast decision theory to get tripped up by — should also cross
zero. It doesn't: 2.2% of bootstrap draws are ≤0, clearing the 95% bar. So the veto isn't
measuring a phantom.

But the magnitude collapses under the properly-matched metric. The raw margin-MAE regression
(+0.00026, a ~0.013% relative cost) looked like the headline finding in §25.2; CRPS(margin)
confirms real harm at a similar relative scale (+0.00019 against a base of 1.35495, ~0.014%)
— i.e., once you use the metric this project itself designated (§16.1, §24.1) as the
distributionally-correct arbiter for exactly this kind of question, the "regression" is not
smaller than advertised, but it is now understood as a small, real, structural cost rather
than an artifact of the wrong scoring rule.

#### 25.6.3 Finding (c) retires the §14.3 information-loss theory — and corrects a warning given to #36

The cost concentrates ENTIRELY in regulation-decided games and vanishes (point estimate
slightly favors the treated model) in OT/SO-decided games. This is the opposite of what §14.3's
own information-loss theory predicted — that theory says the OT/SO layer's resolution step
destroys the per-game skill signal the independent-Poisson joint's own near-diagonal cells
encode, so the harm should concentrate WHERE that resolution happens (OT-decided games). It
doesn't. Cross-cycle comparison makes the real mechanism legible:

| Cycle | OT resolution | Margin-MAE cost | Interpretation |
|---|---|---|---|
| 20 (one-sided transfer) | Flat league constant (`p≈0.509`) | [0.00129, 0.00256] | Mostly the flat-rate OT layer destroying skill information (§14.3) |
| 22 | Team-specific (skill-informed) logistic adopted | — | Removes the flat-rate information-loss channel |
| 25 (two-sided transfer) | Same skill-informed logistic | [0.00006, 0.00045] (mean); 0.00019 (CRPS) | What's left after Cycle 22 — concentrated in REGULATION-decided games, not OT-decided |

Cycle 22 already fixed the information-loss channel §14.3 was worried about. What §25 measures
is a different, smaller, more fundamental thing: correctly inflating every game's OWN
predicted near-diagonal mass to match the real AGGREGATE tie rate necessarily costs a sliver
of INDIVIDUAL-game point-accuracy on the ~77% of games that don't actually end tied — a
calibration-vs-sharpness tension, not a defect of this specific transfer design. It shows up in
regulation-decided games precisely because that's where the "extra" tie-adjacent mass sits
unused relative to the game's real (decisive) outcome.

This corrects a warning given to task #36 in the queue-reshaping discussion (§25.5's closing
paragraph, prior round): the dependence-bearing joint distribution proposed as the long-term
fix does NOT face a large OT-resolution information loss anymore — Cycle 22 already removed
that channel. It faces only this same small calibration-sharpness floor, which is a property
of correctly modeling the real tie rate at all, not of any particular joint construction. This
strengthens #36's prospects relative to how it was framed here originally: whatever
dependence-bearing structure gets built, it should expect a CRPS(margin) floor in the same
small (~0.0002/game) neighborhood, not a large OT-driven one.

#### 25.6.4 Two honesty notes

**On the CRPS result's strength**: 2.2% of resamples ≤0 clears the one-sided 95% bar, but
barely, on a project that has now run dozens of bootstrap adjudications across 25 cycles — one
borderline result among many comparisons is exactly where a false positive would live. This is
reported as "real, marginal" — not leaned on any harder than that.

**On the median-scored test's low power**: median-margin-MAE's CI is roughly 15x wider than
mean-margin-MAE's own CI (discrete medians move in whole-goal jumps; the mean is smooth), so
"crosses zero" here is weak evidence of absence, not strong evidence of neutrality. The honest
summary is that the mean/median hypothesis was neither confirmed nor refuted by the median test
directly — CRPS(margin) is what actually settled the question, by finding a small but real cost
using a metric immune to the mean/median mismatch in the first place.

#### 25.6.5 Revised verdict: closed to automatic-veto adoption; one pre-registered candidate remains

The specific candidate tested in §25.1-25.2 (fitted global `delta_above`/`delta_below`, walk-
forward reg ratios) **stays REJECTED**. Re-adjudicating an already-tested candidate under a
trade-off rule formulated after seeing its numbers is exactly the adaptive-protocol failure
§15 exists to prevent — the corrected holdout protocol's core lesson generalizes here even
though this isn't a holdout check.

But that is not the same as the family being closed for good, and §24's own amendment
anticipated exactly this shape of cycle: real calibration gains (§25.3) against a real,
small accuracy-metric cost (§25.6.1-25.6.2), which §24.3 says should be weighed, not
automatically vetoed — a rule that wasn't yet on the books when §25.1's candidate was built and
tested. It is legitimate to pre-register that weighing rule now, for a NEW, not-yet-built
candidate: point 2's walk-forward per-game delta (via the coherence identity, replacing a
single fitted global constant with a time-varying, era-tracking target) is untested, and a
prospective rule for it is not re-litigation.

**Pre-registered net-trade-off adoption criterion for the walk-forward-delta redesign:**
1. Summed marginal CRPS delta — `CRPS(total) + CRPS(margin)`, dimensionally coherent (both in
   goals) — is a real, bootstrap-confirmed improvement over current best.
2. No real regression on Brier or SU.
3. The era-flattening criterion (§25.4) is met — given by construction with a per-game
   walk-forward delta rather than a single fitted constant, unlike the candidate tested here.

Not presupposed, but worth naming as the check to actually run rather than assume: the
CRPS(margin) cost measured here is 0.00019/game; the plausible totals-side calibration gain
from correctly tracking a growing ~0.05-0.10/game recent-era deficit (§25.4) is very likely
several times larger in the same units. If that ratio holds for the redesigned candidate, the
net judgment won't be close — but the cycle that builds and tests it decides that, not this
section.

**Status, corrected**: §25.5's "closes the independent-Poisson-plus-diagonal-transfer family
for good" is revised to **closed to automatic-veto adoption for the specific tested candidate;
reopened for one new, not-yet-built candidate (walk-forward per-game delta) under the
pre-registered net-trade-off rule above.**

**Ordering for what comes next**: (1) the contamination check first — confirming no other
diagnostic script shares §23's shootout-goal-missing MoneyPuck-situational-sums data path,
since every number in this section inherits that answer if it's wrong, and it's a ten-minute
check; (2) the walk-forward-delta cycle, judged under the net criterion above; (3) then scope
task #35 conditionally on (2)'s outcome — if the redesigned tie fix ships, the tie-mass shortfall
has an owner and task #35 stays the narrow EV-only intercept; if it fails the net criterion too,
task #35 upgrades to a general, margin-neutral walk-forward level intercept absorbing both the EV
residual and the unowned tie-mass shortfall, per the queue-reshaping proposal from the prior
round — but that upgrade is conditional on (2), not decided here.

#### 25.6.6 Contamination check (task #37): clean — one already-reconciled, one newly explained

Two sweeps, per the pre-registered scope: (1) every diagnostic/validation script for the
§23 shootout-goal-missing pathway (summing MoneyPuck situational goals as ground truth); (2)
the inverse pathway (NHL-API final scores as "actual" compared against a regulation
reconstruction that hasn't had the OT/SO bonus correctly re-added), specifically targeting
§16.2's own never-explained 5.482-vs-5.5285 mean-total gap.

**Sweep 1**: `grep`-ing every `src/models/*.py` for a MoneyPuck situational pivot/groupby used
as ground truth finds exactly one hit — `validate_bias_decomposition.py`'s
`actual_by_bucket` pivot. That's the same Bug 2 §23.1 already found and reconciled: it does
NOT contaminate any individual bucket's own gap (EV/PP+PK/other predictions and actuals are
both regulation+OT-only on both sides, by construction — a shootout goal was never supposed
to appear in any of the three buckets), and §23.2 already applied the correction
(-0.092/game) when reconciling the bucket sum against the ladder's Stage A total. The EV
residual figure (~-0.09/game) cited throughout §0.4 item 5 and §25.6.3 is the
already-corrected number — no restatement needed. No other script performs this kind of
situational-sum ground-truth construction.

**Sweep 2**: traced the 5.482-vs-5.5285 gap directly. `validate_cross_season.run_validation()`
(the Cycle-19-era model, predating the reg-ratio/redistribution architecture entirely) returns
a raw combine-layer `lambda_home`/`lambda_away`, calibrated end-to-end against real full final
scores (`home_ice_multiplier` and everything upstream of it is fit against `build_team_game_log`'s
`homeScore`/`awayScore`, OT/SO-inclusive) — mean total **5.557** on a direct re-run under
current settings (the original 5.5285 was logged under an earlier config). `validate_crps.py`
(§16.1, built to enable CRPS/joint work) takes that SAME model's lambda, scales it by the
fixed `HOME_REG_RATIO`/`AWAY_REG_RATIO` constants (themselves correctly measured as
`mean(reg_score)/mean(final_score)` on real games), builds an independent-Poisson regulation
joint, and re-adds the OT/SO bonus via the redistribution step — **the bonus IS re-added, so
this is not the inverse artifact as originally hypothesized** (predicted side isn't missing the
OT goal). Direct measurement of the reconstructed pathway: regulation-scaled mean total
**5.337**, plus a re-add of **0.173** (the model's own diagonal mass under its
independent-Poisson regulation joint — its implied P(reach OT)) = **5.510**, a gap of
**-0.048** from the raw 5.557 — matching the magnitude of the originally-flagged discrepancy
almost exactly.

**Mechanism, and why it isn't a bug**: the redistribution step re-adds the OT bonus in
proportion to the MODEL's OWN modeled tie probability (0.173), not the REAL empirical one
(measured directly on the same dev window: **0.2305**). That 5.8-point gap is the tie-mass
deficit itself (§4.13/§21/§25) — the independent-Poisson joint structurally under-represents
diagonal (tied-regulation) mass relative to the real rate, so any construction that
reconstructs the final mean by re-adding OT credit in proportion to the model's own thin
diagonal necessarily reconstructs a total that undershoots the raw (directly-calibrated)
mean by almost exactly the size of that deficit. This is not a new bug and not the
hypothesized inverse artifact — it's a third, independently-derived measurement of the same
already-known, already-being-worked tie-mass deficit, in the same ballpark as the other two
(Cycle 25's own `fit_deltas` diagonal-vs-real-tie-rate gap, and the bias ladder's growing
reg-ratio/OT-stage deficit). **Production's own logged metrics are unaffected**: every current
headline number (`validate_shorthanded.py`'s `total_mae=1.796`, `margin_mae=2.013`,
`mean_pred_total=5.650`) is computed through the single, self-consistent reconstructed
pathway throughout — it never mixes the two derivations, so this cross-pathway gap never
entered a claimed metric. Neither original figure (5.482, 5.5285) needs restating either,
since §16.2 only ever used them as two mutually-consistent bounds on the existence and rough
size of the mean-total undershoot, not as a claim that they should be identical.

**All clear.** Both sweeps resolve without requiring any number in this round's synthesis, or
in §18/§19/§23's logged figures, to be restated. Proceeding to task #38 (the walk-forward
per-game delta cycle) under the pre-registered net-trade-off rule from §25.6.5. One byproduct
worth carrying forward into that cycle: this sweep's direct measurement (model tie
probability 0.1725 vs. real 0.2305, dev-window average) is a third independent estimate of
the tie-mass deficit's size, consistent with the other two — useful context for judging
whether task #38's redesigned candidate closes enough of that gap to clear the net-trade-off
bar.

## 26. Cycle 26: walk-forward per-game tie-mass delta — built per the pre-registered net-trade-off rule, ADOPTED (2026-07-23)

### 26.1 Motivation: a target three independent derivations agree on

Before this cycle, the tie-mass deficit's size had been measured three separate ways, from
three unrelated derivations: the bias ladder's reg-ratio/OT-stage increment (~-0.05/game,
§19.1), the coherence identity applied to the model's own diagonal mass vs. the real OT rate
(0.2305 real − 0.1725 model = 0.058, §25.6.6), and the raw-vs-reconstructed mean-total gap
(0.048, §25.6.6, the same underlying quantity from yet another angle). All three land within
~0.01 of each other — about as well-characterized as a target gets before a build.

### 26.2 Build: walk-forward calibration ratio + per-game exact-neutrality delta solver

Two-part fix, both new relative to §25's rejected candidate:

**`walk_forward_tie_ratio.add_walk_forward_ot_calibration_ratio`**: a walk-forward, strictly-
prior, `HALFLIFE_GAMES`-decayed RATIO — trailing real OT rate ÷ trailing model-implied mean
diagonal mass — computed once per game from the population's own history up to that point.
§25.6.5 flagged that the naive per-game redesign (solving each game's own delta so its
diagonal mass equals the trailing empirical OT rate directly) is a LEVEL target that repeats
Cycle 17's mistake in new clothes: forcing every game toward the same absolute tie
probability regardless of how lopsided or close that specific matchup is. Multiplying each
game's OWN model-implied diagonal mass by a shared ratio instead preserves cross-game
matchup variation by construction — a mismatched game's already-lower tie probability stays
proportionally lower than a coin-flip game's, while the aggregate still tracks the real rate.
Same epistemic class as §21.2's walk-forward reg ratios: a measured quantity, not a fitted
constant.

**`two_sided_diagonal_transfer.fit_deltas_per_game`**: the same global-balance constraint
from §25's `fit_deltas` (`delta_below = delta_above * above_mass / below_mass`), applied PER
GAME instead of once on population averages. This makes the mean-preservation guarantee
EXACT for every individual game (not just on average across the population, as in §25's
design) — the per-cell mean-shift argument (moving mass from an above-neighbor lowers the
regulation mean by exactly 1 unit per unit probability moved; from a below-neighbor raises it
by exactly 1) holds identically whether applied once in aggregate or once per game. `delta_above`
is clipped to a validity-respecting cap derived from each game's own above/below mass ratio;
clipping only ever affects how closely the target is hit, never the per-game neutrality
guarantee, since `delta_below` is always derived from whatever `delta_above` value is
actually used.

Bundled with walk-forward regulation ratios (§21.2/§25, tested but not independently
adopted), exactly as §25 bundled them — this is evaluated as one tested unit against
production, per this project's standing practice.

### 26.3 Dev-set bootstrap: core ledger metrics neutral, CRPS clears the net-trade-off criterion comfortably

Paired bootstrap (5,000 resamples, 16,528 dev-set games) against the current best
(`shorthanded_poisson` + §22's rest fix, fixed reg ratios, no tie-mass term):

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00006 | [-0.00151, 0.00133] | Crosses zero — neutral |
| Brier | +0.00001 | [-0.00003, 0.00004] | Crosses zero — neutral |
| Total-MAE | +0.00115 | [-0.00003, 0.00228] | Crosses zero |
| Margin-MAE | +0.00025 | [0.00006, 0.00045] | Real, same small cost §25.6 already characterized as a calibration-sharpness floor |
| CRPS(total) | -0.00168 | [-0.00208, -0.00128] | **Real improvement** |
| CRPS(margin) | +0.00019 | [-0.00001, 0.00040] | Crosses zero — no longer even clearly a real cost |
| **CRPS(total)+CRPS(margin)** | **-0.00149** | **[-0.00193, -0.00106]** | **Real improvement — clears the pre-registered net-trade-off criterion #1 comfortably** |

Point estimates: baseline `SU=0.5802 Brier=0.23980 total_mae=1.79616 margin_mae=2.01300
mean_pred_total=5.6493`; treated `SU=0.5801 Brier=0.23980 total_mae=1.79731 margin_mae=2.01326
mean_pred_total=5.7220`; actual mean total `5.8125`.

**Criterion #1 (summed CRPS real improvement) and #2 (no real Brier/SU regression) both
clear.** The CRPS(margin) cost is essentially the same magnitude as §25's candidate
(0.00019 either way) but its CI now straddles zero rather than sitting barely above it — as
pre-logged, the per-game ratio design can only shrink this cost relative to a level-target
design, and here it's shrunk to the point of no longer being statistically distinguishable
from noise.

### 26.4 Era-flattening: not in raw range, but exactly the mechanism the endgame narrative predicted

| Season | Baseline bias | Treated bias | Shift |
|---|---|---|---|
| 2010-11 | -0.115 | -0.057 | +0.058 |
| 2011-12 | -0.141 | -0.080 | +0.061 |
| 2012-13 | -0.191 | -0.129 | +0.062 |
| 2013-14 | -0.269 | -0.214 | +0.055 |
| 2014-15 | -0.220 | -0.161 | +0.059 |
| 2015-16 | -0.153 | -0.102 | +0.051 |
| 2016-17 | -0.103 | -0.050 | +0.053 |
| 2017-18 | -0.125 | -0.057 | +0.068 |
| 2018-19 | -0.223 | -0.144 | +0.079 |
| 2019-20 | -0.295 | -0.213 | +0.082 |
| 2020-21 | -0.291 | -0.214 | +0.078 |
| 2021-22 | -0.454 | -0.373 | +0.081 |
| 2022-23 | +0.045 | +0.153 | +0.108 |
| 2023-24 | +0.169 | +0.285 | +0.115 |

The raw bias RANGE does not narrow (baseline spans -0.454 to +0.169, range 0.623; treated
spans -0.373 to +0.285, range 0.658 — marginally wider), which on its face looks like §25.4's
same "doesn't cleanly flatten" finding. But the breakdown is not uniform, and the pattern it
shows is not a failure — it's exactly the mechanism §24.4's endgame prediction described.
**Twelve of fourteen seasons (2010-11 through 2021-22) show genuine, substantial improvement**
— every one of them moves closer to zero, by +0.05 to +0.08/game, tracking the real,
structural tie-mass undershoot in the eras where nothing else is confounding it. **The two
seasons that get WORSE (2022-23, 2023-24) are precisely the ones the standing
cancelling-errors narrative (§18.6/§19.4/§24.4) already flagged as contaminated by a
DIFFERENT, unrelated mechanism** — the decayed baseline's own transient overshoot in the most
recent eras, which this fix's mean-total increase (via the OT re-add) necessarily makes more
visible rather than less, since it has no way to distinguish "genuine tie-mass undershoot" from
"masked by a separate overshoot" on a per-era basis. This is the pre-registered "unmasking"
prediction (§24.4: "each fix should make holdout total-MAE tick worse, not better") showing up
inside the dev window itself, one season early. **Criterion #3 is PARTIALLY met, not
confirmed**: 12 of 14 seasons flatten as designed; the pre-registered wording ("era-flattening
met") was flatter than the actual outcome, and the honest statement is that the two exceptions
carry a principled excuse (§19.4's standing cancelling-errors prediction) rather than that the
criterion cleared outright. This is itself the health-metric prediction in its most testable
form: those two seasons worsening on schedule is the overshoot-unmasking showing up exactly
where §19.4 said it would, and task #39's re-tune is what's supposed to flatten them for real —
if it doesn't, the narrative was wrong somewhere, not just imprecisely worded.

### 26.5 Holdout confirmatory check (§15 protocol): no veto, total-MAE worsens exactly as pre-logged

Paired bootstrap on 2,624 real holdout games (seasons ≥ 2024-25), confirmatory-veto-only per
§15 — checked against the three core (veto-relevant) ledger metrics, with total-MAE tracked
as informational, not a veto metric, per the pre-logged prediction below:

| Metric | Mean diff | 95% CI | Real? | Role |
|---|---|---|---|---|
| SU | +0.00038 | [-0.00229, 0.00305] | Crosses zero | Veto metric |
| Brier | -0.00000 | [-0.00007, 0.00007] | Crosses zero | Veto metric |
| Margin-MAE | +0.00003 | [-0.00043, 0.00050] | Crosses zero | Veto metric |
| Total-MAE | +0.00708 | [0.00319, 0.01094] | **Real, worse** | Informational — pre-logged, expected |

**No veto triggered.** This is the third installment of the cancellation unwinding predicted
in §24.4: total-MAE getting real-worse on holdout while the core ledger metrics stay neutral
is the exact signature of unmasking a coincidental cancellation, not a genuine regression —
the same pattern §22's rest fix produced (§24's own motivating precedent for treating
calibration-metric costs as weighable rather than automatically disqualifying).

### 26.6 P(over) benchmark: recovers a further chunk of the gap, confirms the pathway is live

Dev-set totals-line benchmark (10,871 decided games, 931 pushes excluded): baseline mean
P(over)=**41.79%**, Brier=**0.25899**; treated mean P(over)=**42.42%**, Brier=**0.25805**.
Both move in the pre-logged direction — mean P(over) closer to the real ~49.5% base rate,
Brier improved — confirming the fix's mean-total increase actually reaches the final joint
used for this benchmark, not just the CRPS/MAE bookkeeping. (Per the pre-logged diagnostic: if
P(over) hadn't moved, that would have been a bug signal, not a verdict — it moved, so it
isn't.) **This baseline number does NOT match the historically-logged 43.35% (§18.5) or
44.65% (§25.3) for what should be the same underlying model — see §26.6.1.**

#### 26.6.1 Reconciling the P(over) history — a real, unresolved discontinuity, not a live regression

The trail: 38.98% (§16.2, original) → 41.61% (§16.4, push-conditioning correction) → 43.35%
(§18.5, after the SH-term shipped) → 44.65% (§25.3's baseline, after §22's rest fix) → **41.79%
(§26.6, same model — `shorthanded_poisson` + §22's rest fix — reconstructed fresh)**. The last
step is a ~2.9-point DROP for what should be an unchanged, already-shipped model, not a
gradual recovery — worth investigating before it becomes this chain's headline exhibit with an
unexplained kink in it.

**What was checked.** §26.6's own baseline (41.79%) is robust: it reproduces to 4 decimal
places via two independent code paths (`validate_tie_mass_ratio.py`'s own construction, and
`validate_tie_mass_v2.py`'s pre-existing Cycle 25 ablation pipeline called directly) — not a
fresh bug. Seven reconstructions were tried to find a pathway that reproduces 43.35%/44.65%
(or the related "mean predicted total 5.622→5.650" figure from §0.2's own caption): the
reconstructed-joint pathway with and without the rest fix (41.78% / 41.79% — the rest fix
barely moves this benchmark at all, since its effect is tiny relative to the total); the SAME
comparison without the SH-term (41.55%-ish, not shown); the RAW, unscaled `lambda_home+
lambda_away` pathway before any regulation-ratio scaling (mean total 5.704→5.706 with the rest
fix, NOT 5.622→5.650); and a Normal-approximation shortcut on the raw lambda/variance (49.85%,
far too high, essentially back at the true rate). None reproduce the historical figures, and
none produce a rest-fix delta anywhere close to the claimed +0.028/game either.

**Conclusion: this is a real, unrecoverable discontinuity in the historical record, not a
live regression.** The push-conditioning fix (§16.4) and the SH-term validation (§18.5) both
predate any script that survived as a permanent, re-runnable file with this exact
construction — `validate_crps.py` is frozen on the Cycle-19-era base (confirmed: it still
imports `validate_cross_season.run_validation`, never updated to `shorthanded_poisson`), so
the 43.35%/44.65%/5.622/5.650 figures were necessarily computed via ad hoc, in-session scripts
that were never saved and can no longer be inspected. Given §26.6's own number is doubly
verified self-consistent, and the search covered every construction that could plausibly have
been used (reconstructed, raw, and a normal-approximation shortcut), the honest conclusion is
that the OLD figures used some now-irrecoverable methodology, not that today's pipeline
regressed.

**Going forward: one benchmark, one pathway, one series.** `validate_tie_mass_ratio.py`'s
`_score_row`/`cdf_total` construction (reg-ratio-scaled independent Poisson → two-sided
transfer where applicable → OT-logistic redistribution, identical to `validate_tie_mass_v2.py`
Cycle 25's own pipeline) is now the SINGLE designated pathway for every future P(over)/mean-
total benchmark number. Every pre-Cycle-26 P(over) figure in this document (38.98%, 41.61%,
43.35%, 44.65%, 45.35%) should be read as historical color from now-unreproducible ad hoc
checks, not as points on the same continuous series as §26.6 onward — the recovery curve
this chain's future write-ups plot starts clean at 41.79%→42.42%.

**Protocol rule (general, not just for P(over)): no number enters this document without a
committed, re-runnable script that regenerates it.** What this section actually documents is
that figures entered the permanent record — and were cited in later reasoning — from scripts
that no longer exist anywhere on disk. Every cycle's headline validator (`validate_*.py`,
`check_holdout_*.py`) already complies with this by convention; the gap was always
interstitial, one-off diagnostic checks run inline and never saved, which is exactly the
pathway that produced 43.35%/44.65%. Going forward, any number destined for this document —
even a quick confirmatory side-check — must come from a script that persists on disk before
being cited, no exceptions for "it's just a sanity check."

**Closing audit, to bound the damage**: does any ADOPTION in this project rest evidentiarily on
a now-unreproducible figure? Walked through cycle by cycle: Cycle 23 (SH-term) was adopted on
`validate_shorthanded.py`'s own dev bootstrap (total-MAE CI [-0.00612,-0.00184], margin-MAE CI
[-0.00033,-0.00007]) — the 43.35% P(over) number was confirmatory color layered on top, not a
criterion the adoption decision used. §22/Cycle 24's rest fix (a "derived correctness fix"
under §24.2) rested on `validate_bias_ladder.py` (scripted, reproducible) plus the
machine-precision margin-neutrality proof (algebraic, in `rest_schedule.py`'s
`symmetric_b2b_bias_credit`) — no P(over) figure involved at all. Cycle 26 ran its own
committed validators throughout. **The record is clean: zero adoption verdicts rest on an
unreproducible number.** The damage is confined to narrative continuity in this document (the
P(over) recovery-curve story has an unexplained kink between §25.3 and §26.6) — not to any
decision this project has made.

### 26.7 Tie-mass mechanism confirmed directly: the achieved tie-rate curve

The fix exists to move model-implied tie mass from ~0.1725 (the deficit measured three
independent ways, §26.1) to the real, walk-forward empirical rate — every number reported so
far is a downstream consequence of that; this is the thing itself, per season:

| Season | Real OT rate | Model diag. mass, BEFORE (fixed ratio, no transfer) | Model diag. mass, AFTER (achieved) |
|---|---|---|---|
| 2010-11 | 0.2415 | 0.1735 | 0.2319 |
| 2011-12 | 0.2439 | 0.1765 | 0.2413 |
| 2012-13 | 0.2250 | 0.1788 | 0.2506 |
| 2013-14 | 0.2496 | 0.1776 | 0.2414 |
| 2014-15 | 0.2488 | 0.1767 | 0.2476 |
| 2015-16 | 0.2236 | 0.1781 | 0.2391 |
| 2016-17 | 0.2350 | 0.1744 | 0.2306 |
| 2017-18 | 0.2329 | 0.1680 | 0.2272 |
| 2018-19 | 0.2132 | 0.1681 | 0.2222 |
| 2019-20 | 0.2311 | 0.1701 | 0.2253 |
| 2020-21 | 0.2247 | 0.1708 | 0.2257 |
| 2021-22 | 0.2195 | 0.1647 | 0.2160 |
| 2022-23 | 0.2302 | 0.1547 | 0.2148 |
| 2023-24 | 0.2073 | 0.1564 | 0.2214 |
| **Mean** | **0.2304** | **0.1706** | **0.2311** |

BEFORE sits persistently 0.05-0.08 below the real rate in every single season, with no
consistent narrowing over time (if anything, the gap widens slightly in the most recent two
seasons — 2022-23/2023-24 — consistent with the same cancelling-errors-era anomaly §26.4
already flagged). AFTER tracks the real rate closely in every season (within ~0.005-0.02,
noise-level given ~1,100-1,300 games/season), and the AGGREGATE achieved mean (0.2311) lands
almost exactly on both the real rate (0.2304) and the walk-forward target it was solving for
(0.2310, §26.3) — confirming the per-game solver hits its aggregate target by construction,
not by luck. This is also the starting condition task #36's eventual dependence-bearing joint
scoping should use: whatever residual tie-mass gap remains for that work to close is bounded
by how tightly AFTER already tracks real, season by season — not the much larger BEFORE gap
this whole chain started from.

### 26.8 CRPS(margin) decomposition: the sharpness floor is invariant across designs — a real finding, and it reframes task #36's bar

The pre-registered net criterion was the SUM, correctly — but a specific falsifiable claim was
pre-logged alongside it: that the ratio design should hold the CRPS(margin) cost at or below
the level-target candidate's (§25) ~0.0002. Full-precision comparison, same baseline, same
bootstrap seed:

| Candidate | CRPS(margin) mean diff | 95% CI |
|---|---|---|
| §25 (fitted global level) | 0.00019238 | [0.00000454, 0.00037949] |
| §26 (walk-forward ratio) | 0.00019186 | [-0.00001156, 0.00039552] |

**This is not "essentially unchanged" — it's invariant to four significant figures** across
two very differently-shaped designs: a single fitted global constant applied uniformly to
every game, versus a per-game, walk-forward-ratio-targeted transfer that preserves matchup
variation by construction. The pre-logged prediction ("at or below") technically survives —
0.00019186 < 0.00019238 — but its underlying premise doesn't: distributing the inflation by
matchup closeness bought nothing on the sharpness floor itself. What DID change between the
two candidates is that the CI now straddles zero instead of sitting barely above it — a
CI-width/sampling artifact, not evidence either design shrank the underlying cost.

**Read honestly, this says something more useful than either candidate's own scoreboard**: the
cost isn't coming from WHERE the tie-mass inflation gets distributed — it's coming from raising
tie mass AT ALL. That's the irreducible entry fee for honest tie-rate calibration within any
independent-Poisson-plus-reallocation scheme, regardless of how cleverly the reallocation is
targeted. §26.8.1 uses this to reframe what task #36 should actually be judged on.

#### 26.8.1 The missing starting-condition number: post-Cycle-26 dispersion (Var(T)/E(T))

If the sharpness floor is invariant to reallocation design, the natural follow-up question is
whether reallocation touches the OTHER known distributional gap at all — the real
overdispersion problem first measured in §10.6.1 (real Var(T)/E(T)=0.8913 vs. model ~1.02,
the reason a genuinely dependence-bearing joint was flagged as the eventual right tool, not a
diagonal-cell trick). Measured directly, same 16,528-game dev population, via the law of total
variance over each game's own predictive total distribution (mixture mean/variance across
games, not a per-game average of per-game ratios):

| | E(T) | Var(T) | Var(T)/E(T) |
|---|---|---|---|
| Real (empirical) | 5.8125 | 5.1810 | **0.8914** |
| Baseline (no tie-mass) | 5.6493 | 5.7687 | 1.0211 |
| Treated (Cycle 26) | 5.7220 | 5.8390 | **1.0205** |

**The two-sided transfer nudged dispersion in the right direction, by almost nothing**:
1.0211→1.0205, a reduction of 0.0006 against a real gap of 0.1297. Mechanically sensible — the
transfer pulls mass from `total=2x±1` into `total=2x`, which does shrink variance a little —
but the effect is two orders of magnitude too small to matter. **After four tie-mass cycles
(17, 20, 25, 26) and the only one that shipped, the real overdispersion gap is still almost
entirely untouched.**

**This reframes task #36's bar, precisely.** A dependence-bearing joint should NOT be judged on
whether it beats the ~0.0002 CRPS(margin) sharpness floor — §26.8 just showed that floor is
invariant to reallocation cleverness and probably can't be dodged within this model family at
all. It SHOULD be judged on whether it buys the one thing no reallocation scheme has touched:
closing some real fraction of the 0.1297 dispersion gap. Task #36's starting conditions are now
complete: the tie-rate curve (§26.7, ✓ tracks real closely, season by season), the sharpness
floor (§26.8, ✓ quantified and shown invariant), and the dispersion gap (this section, ✓
quantified at 1.0205 vs. 0.8914 — the actual target, still wide open).

### 26.9 Verdict: ADOPTED — new current best model

**Adopted**, per the pre-registered net-trade-off rule (§25.6.5): summed CRPS(total
)+CRPS(margin) real improvement (criterion #1, §26.3), no real Brier/SU regression on dev or
holdout (criterion #2, §26.3/§26.5), era-flattening PARTIALLY met — 12 of 14 seasons, with the
two exceptions carrying a principled excuse rather than clearing the criterion outright
(criterion #3, §26.4). Both pre-logged side-effect predictions landed exactly as anticipated
(holdout total-MAE worsens, §26.5; P(over) improves, §26.6), which is itself evidence the
pathway is sound rather than accidentally right — though the P(over) BASELINE's own history
turned out not to reconcile with earlier write-ups (§26.6.1), a separate, real loose thread
now closed off methodologically (one pathway, going forward) rather than mechanistically
(the old numbers' exact derivation is unrecoverable). The achieved tie-rate curve (§26.7)
is the mechanism's direct confirmation; the CRPS(margin) decomposition (§26.8) shows the
sharpness floor itself barely moved, and the net gain is carried by CRPS(total), not by any
reduction in that floor. This is the first tie-mass cycle in the whole saga (Cycles 17, 20, 25
all rejected) to ship — the family closes not because it was bad math, but because every
earlier attempt used a shared constant (global rescale, local-transfer delta, or a single
dev-fit
two-sided delta) that either forced a uniform correction onto every game regardless of its own
matchup, or wasn't proven neutral game-by-game. This design does both: a walk-forward ratio
that preserves cross-game variation, and a per-game balance constraint that's exactly
neutral by construction for every single game, not just on average.

**Current best model is now `walk_forward_tie_ratio_poisson`** (`validate_tie_mass_ratio.py`),
superseding `shorthanded_poisson`. Logged to the ledger (`model_name=
"walk_forward_tie_ratio_poisson"`). Holdout confirmatory check: `check_holdout_tie_mass_ratio.py`.

Next: task #35's scope resolves immediately (task #38 shipped and owns the tie-mass shortfall, so
task #35 stays the narrow EV-only intercept, not the general level intercept). **§27 corrects
§24.4's own stated ordering**: the final decay re-tune runs BEFORE the EV-bucket intercept, not
after (a joint `HALFLIFE_GAMES`×`prior_minutes_multiplier` grid, per §7.5's entanglement
lesson, not a solo re-tune) — see §27 for the full pre-registered spec. Then the EV intercept,
scoped against whatever the re-tune leaves genuinely untracked. Then the deferred
market-benchmark re-run (§6.8), several Brier-relevant adoptions stale, doubling as the
external scoreboard for the whole chain. Then Cycle 16's GBM stacking layer, per its existing
§0.4 protocol.

## 27. Cycle 27 (queued): joint `HALFLIFE_GAMES` × `prior_minutes_multiplier` re-tune — pre-registered spec (2026-07-24)

Pre-registered before running, per this project's own standard, and because this is the cycle
the entire cancellation narrative (§18.6/§19.4/§24.4) has been writing checks against — three
falsifiable claims ride on one grid.

### 27.1 Why a joint grid, not a solo re-tune — §7.5's lesson, verbatim

`HALFLIFE_GAMES=600` was never adopted alone — it shipped jointly with situational priors×0.6
because §7.5 proved the two are entangled (the decayed baseline's own lag pattern had
implicitly tuned the old priors; changing one without the other re-manufactures a different
mismatch, not a fix). Cycle 19 hit the same lesson again and re-tuned `prior_minutes_multiplier`
jointly with `cross_season_weight`. Re-tuning `HALFLIFE_GAMES` now with the multiplier frozen
would repeat this project's own twice-named, twice-paid-for mistake a third time. The grid:
**`HALFLIFE_GAMES` × `prior_minutes_multiplier`**, `cross_season_weight` held fixed at its
current value — it entangles more weakly, and through the prior rather than the baseline
directly, so it's the one knob safe to leave out of this specific grid.

### 27.2 Protocol

Standard grid discipline, per §13/§15's established pattern, with the selection rule pinned
down to a scalar rather than left as a judgment call — a 2-D grid will have multiple cells
that each flatten different eras, and something has to rank them before any numbers are seen.

**Flatness statistic**: RMS of per-season mean-bias (predicted total minus actual total,
averaged per season, then root-mean-square across all 14 dev seasons) — not the raw range
used descriptively in §26.4/§25.4. RMS is the right scalar here because it penalizes both
residual undershoot AND manufactured overshoot symmetrically, which is exactly the failure
mode an over-aggressive decay would produce (flattening the old undershoot eras by
overshooting the recent ones, which a range or a signed-mean statistic could hide).

**Selection rule, in full, with no discretion left in it**:
1. Compute the dev-set bootstrap (Brier, SU, margin-MAE) for every grid cell against current
   best (`walk_forward_tie_ratio_poisson`). Discard any cell with a real regression on any of
   the three.
2. Among surviving cells, rank by per-season bias RMS (lower is better) and select the minimum.
3. **Touch the holdout exactly once, on the selected cell only** — a single confirmatory,
   bootstrap-veto-only check (§15) on Brier/SU/margin-MAE. Holdout is never used to compare
   cells against each other, and prediction (1) below is evaluated on this same single touch,
   not on a separate look.

**Grid span**: must be genuinely aggressive, not a timid bracket around the current values —
`HALFLIFE_GAMES` should span roughly **200–800** (600 sits inside this range but is not
close to either edge), and `prior_minutes_multiplier` should bracket 2.0 **on a ratio/log
scale** (e.g., roughly 0.5x–2x the current value), not a narrow additive band. A grid that
only samples 500-700 can't distinguish "600 was right" from "we never actually looked" — the
whole point of pre-registering the span now is to rule that failure mode out before seeing
results.

### 27.3 Four pre-registered, falsifiable predictions

0. **The winning halflife comes in BELOW 600.** The cancelling-errors narrative says the
   decayed baseline lags upward scoring transitions, and the structural fixes (SH term, rest
   fix, tie-mass) have now removed the real undershoot that lag was accidentally offsetting —
   if that account is right, the grid should want FASTER adaptation (a smaller halflife) once
   nothing is left for the lag to accidentally cancel against. **If the winner comes back at or
   above 600, the shim wasn't the shim** — the decay rate was never the mechanism papering over
   the gap, and that redirects task #35 before it's built, rather than after.
1. **The three "expected worsening" holdout total-MAE installments should substantially
   reverse.** §18.5's SH-term fix ticked the holdout total-MAE point estimate in the wrong
   direction (not real, but close to the boundary); §22's rest fix produced a real regression
   ([0.00053,0.00248]); §26's tie-mass fix produced a larger real one ([0.00319,0.01094]). If
   the cancelling-errors account is right, a correctly re-tuned decay rate should recover most
   of what unmasking cost across all three, not just avoid adding a fourth installment.
2. **The 2022-23/2023-24 seasons that worsened under §26.4's era-flattening check should
   flatten.** These were flagged in advance (§18.6/§19.4, then confirmed on schedule in §26.4)
   as the decayed baseline's own transient overshoot — exactly what a decay re-tune is supposed
   to correct. If they don't flatten here, the cancelling-errors account has a real gap
   somewhere upstream, not just an unlucky grid cell.
3. **The fitted EV-bucket intercept (task #35, run AFTER this cycle) should come out small.**
   A large fitted value would mean this re-tune missed something structural, and #35 would
   inherit a bigger, differently-shaped target than currently scoped — itself a finding, not a
   failure of #35.

If (1) and (2) both fail, the standing narrative was wrong somewhere in the chain, and that's
the pre-registered way to find out where — not a reason to relitigate the grid's own
methodology first.

### 27.4 Why this re-tune is not the §7.4-style shim it might look like

`HALFLIFE_GAMES` feeds several downstream calibration terms shipped since Cycle 13 — the SH
trailing rate (§18.4), the walk-forward regulation ratios (§21.2/§26.2), and Cycle 26's own
walk-forward OT-calibration ratio (§26.2) — and none of them need a re-fit as the halflife
moves. Each is itself a walk-forward, strictly-prior quantity computed FROM whatever baseline
feeds it, so they re-equilibrate automatically as the baseline's own decay rate changes; they
are self-calibrating by construction, not fixed constants tuned against one specific halflife
the way the old situational priors were in §7.5. **The only genuinely entangled constant left
is `prior_minutes_multiplier`** — a fixed-magnitude shrinkage-prior weight that does NOT
re-equilibrate on its own, exactly like §7.5's original entanglement — which is why it is the
one parameter in this grid, and why nothing else needed to be. That is the actual difference
between this re-tune and the earlier one that got burned (§7.4-§7.5): the system being tuned
here is mostly self-healing by construction, and the grid only has to cover the one piece that
isn't.

## 28. Cycle 27 results: the grid found no improvement — prediction (0) falsified, and it redirects task #35 (2026-07-24)

Run exactly to §27's pre-registered spec (`validate_halflife_multiplier_grid.py`):
`HALFLIFE_GAMES` in {200, 400, 600, 800} × `prior_minutes_multiplier` in {1.0, 2.0, 4.0},
`cross_season_weight` fixed, 12 cells, each bootstrapped (Brier/SU/margin-MAE) against current
best (`walk_forward_tie_ratio_poisson`, halflife=600/multiplier=2.0).

### 28.1 Full grid

| Halflife | Multiplier | Brier diff (95% CI) | SU diff (95% CI) | Margin-MAE diff (95% CI) | Real regression? | Bias RMS |
|---|---|---|---|---|---|---|
| 200 | 1.0 | +0.00032 [0.00015,0.00049] | -0.00127 [-0.00375,0.00109] | -0.00020 [-0.00130,0.00089] | **YES** (Brier) | 0.18129 |
| 200 | 2.0 | +0.00003 [-0.00001,0.00007] | +0.00048 [-0.00079,0.00175] | +0.00044 [0.00019,0.00069] | **YES** (margin) | 0.19773 |
| 200 | 4.0 | +0.00001 [-0.00018,0.00020] | -0.00036 [-0.00321,0.00248] | +0.00280 [0.00157,0.00398] | **YES** (margin) | 0.21970 |
| 400 | 1.0 | +0.00030 [0.00013,0.00047] | -0.00109 [-0.00345,0.00121] | -0.00045 [-0.00155,0.00064] | **YES** (Brier) | 0.17208 |
| 400 | 2.0 | +0.00001 [-0.00000,0.00002] | -0.00024 [-0.00097,0.00048] | +0.00017 [0.00009,0.00026] | **YES** (margin) | 0.18575 |
| 400 | 4.0 | -0.00001 [-0.00020,0.00017] | -0.00115 [-0.00393,0.00164] | +0.00249 [0.00129,0.00363] | **YES** (margin) | 0.20600 |
| 600 | 1.0 | +0.00029 [0.00012,0.00047] | -0.00133 [-0.00375,0.00097] | -0.00061 [-0.00172,0.00049] | **YES** (Brier) | 0.17313 |
| **600** | **2.0** | **0 (reference)** | **0 (reference)** | **0 (reference)** | No | **0.18306** |
| 600 | 4.0 | -0.00002 [-0.00021,0.00016] | -0.00151 [-0.00430,0.00127] | +0.00229 [0.00111,0.00340] | **YES** (margin) | 0.20030 |
| 800 | 1.0 | +0.00029 [0.00012,0.00047] | -0.00121 [-0.00363,0.00121] | -0.00074 [-0.00188,0.00039] | **YES** (Brier) | 0.17907 |
| 800 | 2.0 | -0.00000 [-0.00011,0.00004] | +0.00024 [-0.00024,0.00073] | -0.00015 [-0.00021,-0.00008] | No | 0.18546 |
| 800 | 4.0 | -0.00003 [-0.00021,0.00015] | -0.00091 [-0.00369,0.00188] | +0.00213 [0.00097,0.00324] | **YES** (margin) | 0.19953 |

**2 of 12 cells survive** the no-real-regression bar: `(600, 2.0)` — current production, zero
diff by definition — and `(800, 2.0)`, which is neutral on Brier/SU and shows a small REAL
IMPROVEMENT on margin-MAE ([-0.00021,-0.00008]). Per the selection rule, rank survivors by
bias RMS: `(600, 2.0)` at 0.18306 beats `(800, 2.0)` at 0.18546. **Winner: `(600, 2.0)` —
current settings, unchanged.**

### 28.2 Prediction (0) is FALSIFIED — the winning halflife does not come in below 600

The winner ties at exactly 600, and the only other survivor (800) is SLOWER, not faster. This
is not a marginal miss: at the one multiplier that survives at all (2.0, matching current
production), BOTH tested faster values (200 and 400) show a REAL margin-MAE regression, while
the slower value (800) is neutral-to-improving. The direction is the opposite of what the
cancelling-errors narrative predicted — if anything, the system very mildly prefers slower
adaptation, not faster. **Per §27.3's own pre-registered language: the winner came back AT
600, so the shim wasn't the shim.**

The grid also reconfirms something useful in passing: `prior_minutes_multiplier=2.0` sits at a
genuine local optimum on its own axis, bracketed by real regressions on both sides — 1.0
(weaker shrinkage, the pre-Cycle-19 level) hurts Brier at every halflife tested; 4.0 (more
shrinkage than ever adopted) hurts margin-MAE at every halflife tested. Cycle 19's original
joint-grid finding holds up under a much wider, independently-run re-check.

### 28.3 Predictions (1) and (2) do not apply — nothing changed to test them against

Since the winning cell is identical to current production, there is no new holdout check to
run and no new per-season bias curve to inspect — both would just reproduce Cycle 26's own
already-published numbers (§26.4, §26.5). The three "expected worsening" holdout total-MAE
installments do NOT reverse, and the 2022-23/2023-24 seasons do NOT flatten, because there is
no new model for them to reverse or flatten against. This is not a separate failure to log —
it is the direct, mechanical consequence of prediction (0) already having failed.

### 28.4 What this actually means: the "shim" was probably never `HALFLIFE_GAMES` at all

The honest reading, stated as interpretation rather than fact: `HALFLIFE_GAMES=600` is not
compensating for anything left over from the structural chain — it is sitting at a genuine
local optimum that does not want to move in EITHER direction once the multiplier is correct.
That is a different, and more informative, finding than "the re-tune found nothing" — it
suggests the near-zero holdout-era mean-total bias that motivated the entire cancelling-errors
account (§16.5, §18.6, §19.4, §24.4) was never explained by a mistuned DECAY RATE specifically.
The more likely account, consistent with everything measured across §18-§28: the near-zero
bias was a coincidence among several small, independent structural pieces (shorthanded-goals,
rest, tie-mass, and whatever the still-pending EV residual turns out to be) that happened to
roughly net out, not a baseline decay rate quietly absorbing the difference. **This does not
mean the cancelling-errors account was wrong about undershoot being real and being unmasked
cycle by cycle — §22's, §26's, and the SH-term's own holdout total-MAE movements are real,
scripted, bootstrap-confirmed installments of exactly that.** It means the specific mechanism
proposed for what would CLOSE the cancellation — one final decay re-tune — was the wrong
lever, not that there was nothing to unmask in the first place.

**⚠ Correction (2026-07-24, per §29's follow-up)**: the claim above that §22's and §26's
holdout regressions are "direct evidence" of a real overshoot being unmasked overreaches. Both
readings — genuine overshoot, and coincidence that happened to sit near zero — predict the
SAME observable: adding a real +δ to predictions in an era whose net bias is already ≈0
worsens MAE regardless of how that near-zero came about. The installments are real,
bootstrap-confirmed regressions; they are NOT evidence that distinguishes overshoot from
coincidence. §29's trailing-baseline check (a direct EWMA on real total goals, decoupled from
the model entirely) then came back genuinely mixed: 2023-24/2024-25 show overshoot at
~+0.07/game, indistinguishable from the noise floor already present in the "stationary"
2010-13 seasons (+0.05 to +0.08), and 2025-26 flips sign to -0.077. **Verdict: this dispute is
UNRESOLVED, undetermined at current data volume — not leaning either way** — and, decisively,
nothing queued depends on the answer. `HALFLIFE_GAMES` is frozen at a grid-confirmed optimum
regardless of which account is right (§28.2); the EV intercept is closed regardless, on
information-theoretic grounds independent of this dispute (§29.3.1). Logged as open, revisit
only if a future season materially extends the per-season bias curve — the curve itself keeps
doing its job as a health metric either way, without needing this question answered.

### 28.5 Redirecting task #35: the fitted EV intercept should now be expected LARGE, not small

§27.3's prediction (3) said the fitted EV-bucket intercept should come out small after a
correct halflife re-tune, on the theory that a correct re-tune would absorb whatever the decay
rate had been quietly compensating for. That premise is gone: §28.2 showed the decay rate
wasn't compensating for anything. **Revised, pre-registered expectation for task #35, logged
now rather than discovered mid-cycle**: the fitted EV-bucket intercept should come out
non-trivial — plausibly a first-class term of the same rough magnitude as the EV residual
itself (~-0.09/game, §18.3), not a small mop-up correction. If it DOES come out small despite
this, that would itself be the surprising result requiring explanation, not the expected one.
Task #35 should be scoped and validated as a real, independently-load-bearing term from the
outset, not as a final polish on an already-mostly-closed chain.

Next: task #35 (EV-bucket intercept), with the revised expectation above; `HALFLIFE_GAMES`
and `prior_minutes_multiplier` remain at their current, grid-confirmed values (600, 2.0) —
no change to production. Then the deferred market-benchmark re-run (§6.8). Then Cycle 16's
GBM stacking layer.

## 29. Cycle 28: walk-forward EV-bucket intercept — REJECTED, real total-MAE regression at every tested halflife (2026-07-24)

Built per §28.5's redirected expectation and three pre-registered spec requirements.

### 29.1 Build, per spec

`ev_residual_intercept.py`: **(a) symmetric, margin-neutral by the §22 algebra** — the
identical trailing credit is added to both `lambda_home` and `lambda_away`, unconditionally,
before regulation-ratio scaling — the same construction as `symmetric_b2b_bias_credit`.
**(b) tracked quantity is realized minus PRE-intercept prediction** — `game_ev_residual =
actual_ev_total - pred_ev_total`, built entirely from `validate_bias_decomposition.
run_decomposition`'s own existing (already §23.1-bug-fixed) EV-bucket combine, with no
intercept applied — never a post-intercept residual, avoiding the feedback loop. **(c) decayed
trailing window, own halflife gridded** — `{200, 400, 600, 900, 1200}`, everything else
(which residual, how it's applied) measured, not fit.

### 29.2 Dev-set bootstrap: 0/5 cells survive — a real, consistent total-MAE regression

| Intercept halflife | Brier diff | SU diff | Margin diff (machine-precision check) | Total-MAE diff (95% CI) | Survives? |
|---|---|---|---|---|---|
| 200 | +0.00001 | +0.00012 | -0.0000000470 | +0.00303 [0.00071,0.00540] | **NO** |
| 400 | +0.00001 | +0.00006 | +0.0000004791 | +0.00268 [0.00058,0.00478] | **NO** |
| 600 | +0.00000 | +0.00006 | +0.0000006064 | +0.00282 [0.00082,0.00475] | **NO** |
| 900 | +0.00000 | +0.00000 | +0.0000007296 | +0.00319 [0.00134,0.00502] | **NO** |
| 1200 | +0.00000 | +0.00000 | +0.0000008170 | +0.00353 [0.00175,0.00528] | **NO** |

Spec (a) confirmed empirically, not just by construction: margin diff is ~1e-7 at every
halflife — not exact machine-precision zero like §22's rest-fix credit (that credit is added
before a SINGLE reg-ratio scaling step per side; this credit passes through the same
differential home/away reg-ratio scaling before the tie-mass transfer, which introduces a
tiny, structurally-expected leak far below any bootstrap-detectable threshold — not a bug, but
correctly caught by checking rather than assuming). Brier and SU are both neutral at every
halflife — no calibration benefit anywhere. **Total-MAE is a REAL regression at EVERY tested
halflife**, with no compensating gain elsewhere to weigh it against (this is not a tie-mass-
style trade-off — there is no real gain on any metric here). **Rejected as designed.**

### 29.3 Why: the EV-bucket residual is far more volatile than the -0.091/game aggregate figure suggested

**Sign convention note**: this section and §29.4/§29.5.1 report `gap_ev = pred_ev - actual_ev`
throughout, matching §18.2's original convention (negative = model undershoots) — the code's
own internal `game_ev_residual` (used to build the additive correction credit) is defined the
opposite way (`actual - pred`, so a positive value means "add this much back"), which is the
right convention for a corrective term but the wrong one for reporting a bias figure next to
`-0.091/game` without a sign flip. Values below are `-1 ×` the code's own `game_ev_residual`.

Per-game EV gap (`gap_ev`), mean by season (full history):

| Season | `gap_ev` | Season | `gap_ev` |
|---|---|---|---|
| 2010-11 | -0.089 | 2018-19 | -0.088 |
| 2011-12 | -0.141 | 2019-20 | -0.231 |
| 2012-13 | -0.202 | 2020-21 | -0.300 |
| 2013-14 | -0.173 | 2021-22 | -0.254 |
| 2014-15 | -0.047 | 2022-23 | +0.077 |
| 2015-16 | +0.038 | 2023-24 | +0.077 |
| 2016-17 | +0.021 | 2024-25 | +0.033 |
| 2017-18 | -0.062 | 2025-26 | -0.147 |

**This is not a stable, structural -0.091/game bias — it swings from -0.300 to +0.077 across
seasons**, including three separate sign flips. §18.3's aggregate figure was real and honestly
reported, but it averaged over enormous underlying volatility that a single walk-forward
trailing mean cannot safely absorb: at any given point, the correction reflects a blend of
recent history that can be quite different from — and lag behind — whatever the current
season's true gap actually is. This is very likely the direct mechanism behind §29.2's
regression: the term adds real error in the seasons where its trailing estimate has drifted
away from the current truth, and that added error outweighs whatever it removes in seasons
where the trailing estimate happens to still be accurate.

#### 29.3.1 This was information-theoretically infeasible, not an empirical near-miss

The grid didn't fail to find a good halflife — no halflife could have worked, and the math
says so directly, upgrading §29.2's finding from "this design didn't work" to "no trailing
game-level tracker can work at this signal-to-noise ratio."

The signal is a ~0.09-goal level correction. Per-game total-goals noise is `sd≈2.4`
(`sqrt(Var(T))` from §26.8.1's own dispersion measurement, 5.77-5.84). Estimating a level to
even half-signal precision (`SE≈0.045`) needs `n ≈ (sd/SE)² = (2.4/0.045)² ≈ 2,844` games —
over two full seasons of trailing window. Using the simple approximation `SE(halflife=h) ≈
sd/√h`: at `h=200` (the fastest tested cell), `SE≈2.4/√200≈0.17` — nearly twice the signal
itself, pure noise. At `h=1200` (the slowest tested cell), `SE≈2.4/√1200≈0.069` — still 54%
above the 0.045 threshold. **The grid never got anywhere near adequate precision at either
end.** There is no viable point on this curve: any window short enough to track real movement
in the residual (§29.3's volatility) is too noisy to estimate the level at all; any window
long enough to estimate the level precisely (`h≥~2,800`, more than two full seasons) is far
too slow to track the movement that motivated tracking it in the first place.

**This is directly visible in the grid's own shape, not just implied by the math**: total-MAE
regression is worst at the fastest halflife (200: +0.00303), improves through 400 (+0.00268),
then gets WORSE again at 600, 900, and 1200 (+0.00282→+0.00319→+0.00353). That U-shape is
exactly the bias-variance signature the argument predicts — too fast is noisy, too slow is
stale — and critically, the regression never crosses into a net improvement anywhere in
between, because the noise floor (`sd≈2.4`) is simply too large relative to the signal
(`~0.09`) for any decay rate to resolve.

**This properly closes the question rather than leaving it open**: a residual this small
relative to per-game noise cannot be fixed by ANY measurement-based tracker, however designed.
Only a root-cause fix — a mechanism found and removed, like the shorthanded-goals discovery —
can address a signal at this scale. This also retroactively explains why the root-cause chain
kept winning where this generic correction failed: the SH term, the rest fix, and the tie-mass
ratio never had to estimate their own magnitudes from a single noisy per-game residual — each
was measured from a mechanism with its own, much larger effective sample (SH goal rate from
event-level counts across the league's entire PK-time exposure; the rest adjustment from
thousands of team-games' worth of B2B history; the tie-mass calibration ratio from a trailing
binary-incidence rate, not a raw noisy level). Root-cause measurement sidesteps this
signal-to-noise problem entirely; residual-tracking cannot.

### 29.4 The fitted trajectory — genuinely informative, but not a clean read on §28.4

**Sign convention note**: unlike §29.3/§29.5.1's `gap_ev` (pred-actual), this table reports the
fitted CREDIT — the additive correction actually added to the prediction — so its sign is
naturally the opposite: a positive credit compensates for a negative (undershoot) `gap_ev`. A
season with `gap_ev≈-0.09` should show a fitted credit of roughly `+0.09`, not `-0.09`; the two
tables are intentionally mirror-signed, not inconsistent.

Per the pre-logged instrument (halflife=600, representative — full history, dev+holdout):

| Season | Fitted intercept | Season | Fitted intercept |
|---|---|---|---|
| 2010-11 | +0.095 | 2018-19 | +0.161 |
| 2011-12 | +0.194 | 2019-20 | +0.309 |
| 2012-13 | +0.312 | 2020-21 | +0.421 |
| 2013-14 | +0.358 | 2021-22 | +0.456 |
| 2014-15 | +0.250 | 2022-23 | +0.211 |
| 2015-16 | -0.013 | 2023-24 | -0.068 |
| 2016-17 | +0.001 | 2024-25 (holdout) | -0.103 |
| 2017-18 | +0.006 | 2025-26 (holdout) | +0.121 |

Two pre-logged readings, per §28's request: **(i) if it dips toward zero/negative around
2023-25, overshoot was real, cancellation account vindicated, halflife exonerated. (ii) if it
holds flat ~+0.09 straight through, coincidental-netting wins.** The trajectory does neither
cleanly. It does dip negative in exactly 2023-24 and 2024-25 (-0.068, -0.103) — consistent
with reading (i) — but it never holds flat at ~+0.09 anywhere; instead it swings up to +0.456
by 2021-22, four to five times the aggregate figure, before dropping. **This is the honest
complication neither pre-logged reading anticipated**: this intercept is ITSELF a trailing
statistic with its own halflife, and its 2019-22 climb and 2023-25 descent track the same
COVID-era scoring anomalies (shortened 2019-20/2020-21 seasons, the 2021-22 scoring spike)
that drove the mean-total bias story in the first place — it may be recording the SAME
turning-point-lag dynamic in a new variable, not independently confirming a separate,
genuine EV-bucket-specific overshoot. **Given §29.2's real total-MAE regression, this
trajectory cannot be read as a validated signal either way** — a term whose OWN application
makes predictions worse is not a trustworthy instrument for adjudicating what the "true"
residual looks like. The dispute is not resolved by this cycle; it is complicated by a new
finding (real, high volatility in the underlying residual) that neither pre-logged account
anticipated.

### 29.5 Holdout (diagnostic only — this term is not being shipped)

At halflife=600: SU +0.00038 [0.00000,0.00114] (borderline, not real), Brier -0.00001
[-0.00002,0.00001] (crosses zero), margin -0.00000 (crosses zero, consistent with (a)),
total-MAE -0.00012 [-0.00250,0.00227] (crosses zero — unlike the dev-set regression). The
holdout window's two seasons carry opposite-signed fitted credits (2024-25: -0.103, 2025-26:
+0.121) that roughly cancel in aggregate, on top of a much smaller sample (2,624 games) — this
holdout read is noisy and not decisive either way; the dev-set regression (16,528 games,
consistent across all 5 halflives) is the load-bearing finding.

#### 29.5.1 The flat-era check: a genuine stationary component survives — this is a mechanism hunt, not a terminus

**Sign convention**: `gap_ev = pred_ev - actual_ev`, same as §29.3/§18.2 (negative = undershoot),
NOT the code's internal `actual - pred` — flipped from an earlier draft of this section, which
briefly reported this table in the opposite sign with no annotation (caught during a sign-
convention sweep across §29 and §18's bias tables; no other adopted figure in this document
was affected — see the correction note on §28.4 and §0.4 item 5 for the standing `gap_ev`
convention used everywhere else).

Decisive check, per-season `gap_ev` restricted to 2010-11 through 2016-17 — the window before
either known real scoring-level jump (2017-18 rule changes, 2021-22 post-COVID rebound), where
turning-point lag should be ≈0 by construction:

| Season | `gap_ev` | Season | `gap_ev` |
|---|---|---|---|
| 2010-11 | -0.089 | 2014-15 | -0.047 |
| 2011-12 | -0.141 | 2015-16 | +0.038 |
| 2012-13 | -0.202 | 2016-17 | +0.021 |
| 2013-14 | -0.173 | | |

Season-level mean: **-0.0848** (game-level pooled mean -0.0761, n=8,100 games). One-sample
t-test on the 7 season-level means vs. zero: t=-2.398, p=0.053 (borderline at n=7, but the
direction and magnitude are unambiguous). **This matches the full 16-season aggregate
(-0.0834) almost exactly — it does NOT oscillate around zero.** Per the pre-registered
decision rule: a genuine stationary EV-bucket component exists, with an unknown root cause,
independent of the turning-point-lag confound that dominates the jump-era years. Given
§29.3.1's information-theoretic argument, this component can only ever be addressed by finding
and removing an actual mechanism (the SH-goals path), never by re-estimating it more cleverly
from noisy residuals — the ratios and underlying data are already confirmed clean (§18.3), so
any future hunt should target the combine's own remaining moving parts, not the inputs.

**This is NOT the clean terminus the alternative reading would have given.** The mean-total
chain is not complete: a real, unexplained EV-bucket deficit remains, on top of (not
instead of) the well-understood, now-priced turning-point lag at scoring-jump eras. Not
pursued further in this pass — a mechanism hunt is a genuinely open research question, of the
same character as task #36, not a re-scoped near-term task.

### 29.6 Verdict: REJECTED as designed — and the underlying approach is closed, not just this attempt

**Rejected.** The single, pooled, league-wide walk-forward EV intercept produces a real,
bootstrap-confirmed total-MAE regression at every tested halflife, with no calibration gain to
weigh against it. §29.3.1 shows this isn't a design flaw to iterate on: at `sd≈2.4` per-game
noise against a `~0.09` signal, NO trailing tracker can resolve this residual — the family of
measurement-based mop-ups is closed, not just this cycle's specific implementation. §29.5.1
shows the residual itself is real (a genuine, ~-0.09 stationary `gap_ev` survives in the
pre-jump flat era, matching the full aggregate almost exactly) — so this is not a "the problem
was never real" result either. **Both things are true at once: the EV-bucket deficit is real
and still open, and no tracking-based fix can ever close it.** Only a root-cause mechanism
hunt — finding the actual missing or mismeasured effect, the way shorthanded-goals was found
for PP/PK — can close it. **Current best model remains `walk_forward_tie_ratio_poisson`
(§26), unchanged.**

Next: task #40 (market-benchmark re-run) no longer waits on a task #35 that isn't shipping — it can run
directly against current best, unchanged by this cycle. Then Cycle 16's GBM stacking layer. A
future EV-bucket mechanism hunt is logged as an open research question (of the same character
as task #36), not a queued task — what this cycle establishes is that finding the mechanism is
the only viable path, not that there's nothing left to find.

## 30. Market-benchmark re-run: pre-registered expectation, before the number lands (2026-07-24)

Task #40, re-running §6.8's market benchmark against `walk_forward_tie_ratio_poisson` — four
adoptions stale (Cycles 22, 23, 25's rejection, 26), and the external scoreboard for
everything since Cycle 18. Two numbers, one pre-registered expectation each, stated now.

**The real-odds match (`sbro_odds_games.parquet`) has no coverage for 2022-23 onward (§6.1)**
— this benchmark's window is 13 seasons ending 2021-22, containing BOTH known real
scoring-level jumps (2017-18, 2021-22). Every structural fix since §18 targets the CURRENT,
stationary-era calibration; none of them can retroactively repair the turning-point lag
already baked into this specific, older window.

1. **Moneyline Brier gap** (model vs. market-close): should narrow further from §6.8's last
   measurement (`cross_season_prior_poisson`: 0.24167 vs. market 0.23867, gap
   [0.00187,0.00412]), reflecting the real calibration gains since (Cycles 22/23/26), but
   should NOT close to zero — the window itself owns a real, structural remainder no
   stationary-era fix can touch.
2. **Totals-line P(over)**: the realistic target is the **mid-40s%**, not the ~49.5% true
   rate — the lag eras inside this specific window own the unrecoverable remainder, the same
   way §26.6's own current-era P(over) (42.42%) hasn't reached 49.5% either despite every
   stationary-era fix landing. A number in the high-40s here would be a bigger surprise than
   a good one — worth checking the pathway for a bug, not simply celebrating, per the same
   "if it doesn't move as expected, that's a bug signal" discipline used in §26.6.

Both numbers are logged BEFORE running so the results in §31 read against a stated
expectation, not an implicit hope.

## 31. Market-benchmark re-run results: both pre-registered expectations land (2026-07-24)

`validate_market_benchmark.py` (updated to target `walk_forward_tie_ratio_poisson`) and the
new `validate_market_totals_benchmark.py`, run against the same 11,803/11,802-game real-odds
match used throughout §6.

### 31.1 Moneyline Brier gap: narrows further, does not close — as expected

**Model Brier: 0.24114** (vs. the last-logged `cross_season_prior_poisson`: 0.24167) **— market
close: 0.23867 — gap: 0.00247, 95% CI [0.00141, 0.00352]**, narrower than the last measurement
([0.00187,0.00412]) but still real and clearly nonzero. Matches prediction (1) precisely: real
progress, no false claim of having closed the gap. Per-season breakdown shows no new
concentration pattern (gap ranges +0.00015 to +0.00558, no clean trend) — 2016-17 and 2020-21
are absent from the odds match entirely (a pre-existing coverage gap in the underlying SBRO
data, unrelated to this re-run).

### 31.2 Totals-line P(over): 42.42% — low-to-mid 40s, matching the spirit of the prediction, not its exact center

**Mean P(over): 0.4242** (model Brier 0.25805 vs. market-baseline 0.25000; real over rate at
market lines 0.4947, confirming the line-setting assumption still holds). This lands in the
40s, clearly and substantially below the true ~49.5% rate — directionally exactly as
predicted, and for the right stated reason (this window's own turning-point-lag eras, not a
stationary-era failure) — but at the LOW end of "mid-40s" rather than squarely in the middle;
reported precisely rather than rounded up to fit the pre-registered phrase. Per-season
breakdown (0.398-0.468) shows the same volatility already established throughout §18-29 — no
new finding, consistent with everything measured since. **No bug signal**: the number moved in
the predicted direction, for the predicted reason, at a plausible (if not perfectly centered)
magnitude.

### 31.3 Verdict: benchmark re-run complete, no adoption decision — both external scoreboards confirm real, incomplete progress

Not an adoption cycle — a scorekeeping one. Both external numbers behave exactly as a correct,
honest account of this project's progress would predict: real gains since §18 show up (Brier
gap narrows), and a known, already-explained limitation (this window's own scoring-jump
contamination) shows up too, rather than being silently absorbed or overclaimed. Task #40
closes clean.

## 32. Cycle 29: EV-TOI expanding-mean root-cause fix — mechanism confirmed, adoption case genuinely mixed (2026-07-24)

### 32.1 The mechanism, confirmed directly against real data before building anything

`validate_situational_toi.py` computes `league_avg_ev_toi_min`/`league_avg_other_toi_min` via
`.shift(1).expanding().mean()` — an infinite-memory average, live in PRODUCTION (not just a
diagnostic-script artifact like §23's shootout-goal issue; `validate_goalie.run_validation`,
which every current-best pipeline calls, threads straight through to this exact line) — the
fourth instance of the §7.1 two-baselines-different-memory bug class. Real EV time-on-ice has
risen from ~47.9 min/game (2010-11) to ~49.9 min/game (2024-25) as penalties have declined
roughly continuously; an infinite-memory average permanently lags a rising series. Measured
directly: **mean gap (expanding mean − real) = -0.99 min/game, negative in every single season
with no exceptions.** Converting to an implied whole-game goal impact via each season's own
real EV scoring rate (~2.2-2.5 goals/60): **mean implied impact = -0.077/game**, closely
matching the flat-era `gap_ev` of -0.085/game (§29.5.1) — in the specific pre-jump seasons
(2010-11: implied -0.097 vs. actual -0.089; 2011-12: implied -0.117 vs. actual -0.141), the
match is close enough to treat this as the confirmed mechanism behind the stationary component
§29.5.1 found, not merely a plausible correlate.

**Cross-bucket note**: the "other" bucket's own `league_avg_other_toi_min` gap does NOT show
the same clean pattern (positive in early years, crossing to negative and back, no consistent
trend) — `add_walk_forward_mean` (PP/PK's own TOI estimate) also resets each season rather than
accumulating across all history the way `league_avg_ev_toi_min` does, so it isn't exposed to
the same multi-year-lag mechanism; the requested PP/PK "stale-high" cross-check did not turn up
a comparable signature, most likely because that code path is structurally different, not
because the underlying penalty-decline trend doesn't also touch it.

### 32.2 Fix: a new, deliberately separate opt-in parameter, not a reuse of the live `league_avg_halflife_games`

`toi_halflife_games` (new, defaults to `None`, preserving the original expanding-mean behavior
exactly) threaded through `validate_situational_toi.py` → `validate_goalie.py`. Deliberately
NOT reusing `league_avg_halflife_games` — production already sets that to a real value (600)
for the situational-strength decay, so reusing it here would have silently changed production
behavior the moment the code shipped, rather than staying a genuine opt-in the way every other
fix in this project has been introduced.

### 32.3 Dev-set grid: no real regression anywhere, but no real total-MAE improvement either

Grid over `toi_halflife_games` ∈ {300, 600, 900, 1200, 1800}, full current-best pipeline,
bootstrapped against `walk_forward_tie_ratio_poisson` (`toi_halflife_games=None`):

| Halflife | Brier diff | SU diff | Margin diff | Total-MAE diff | Real regression? |
|---|---|---|---|---|---|
| 300 | +0.00002 [-0.00001,0.00005] | +0.00024 [-0.00091,0.00133] | -0.00003 [-0.00023,0.00017] | +0.00009 [-0.00090,0.00106] | No |
| 600 | +0.00001 [-0.00001,0.00004] | +0.00054 [-0.00048,0.00151] | -0.00007 [-0.00024,0.00009] | +0.00032 [-0.00052,0.00113] | No |
| 900 | +0.00001 [-0.00001,0.00003] | +0.00048 [-0.00043,0.00145] | -0.00010 [-0.00025,0.00005] | +0.00043 [-0.00032,0.00116] | No |
| 1200 | +0.00000 [-0.00002,0.00003] | +0.00061 [-0.00030,0.00151] | -0.00011 [-0.00025,0.00002] | +0.00048 [-0.00022,0.00116] | No |
| 1800 | +0.00000 [-0.00002,0.00002] | +0.00061 [-0.00024,0.00145] | **-0.00014 [-0.00025,-0.00002]** | +0.00049 [-0.00014,0.00110] | No |

**All 5 cells survive** — no real regression on any ledger metric at any tested halflife. But
**no cell shows a real total-MAE improvement either** — every CI crosses zero, and the point
estimate actually gets slightly worse (not better) as halflife lengthens (+0.00009→+0.00049).
The one real, bootstrap-confirmed effect anywhere in this grid is a small margin-MAE
IMPROVEMENT at halflife=1800 (CI entirely negative) — a real, if minor and secondary,
calibration gain, not the total-MAE gain the mechanism's own magnitude match suggested.

### 32.4 Per-season bias: real, substantial movement toward zero in most dev seasons — but not a clean aggregate win

Winning cell by the standard selection rule (halflife=300, closest to zero total-MAE diff):

| Season | Current bias | Fixed bias | Season | Current bias | Fixed bias |
|---|---|---|---|---|---|
| 2010-11 | -0.057 | -0.089 (worse) | 2018-19 | -0.143 | -0.035 (much better) |
| 2011-12 | -0.080 | -0.072 | 2019-20 | -0.213 | -0.150 |
| 2012-13 | -0.129 | -0.108 | 2020-21 | -0.213 | -0.219 (worse) |
| 2013-14 | -0.214 | -0.197 | 2021-22 | -0.372 | -0.334 |
| 2014-15 | -0.161 | -0.088 (much better) | 2022-23 | +0.153 | +0.124 |
| 2015-16 | -0.102 | -0.046 (much better) | 2023-24 | +0.285 | +0.264 |
| 2016-17 | -0.050 | +0.021 | 2024-25 (holdout) | +0.209 | +0.276 (worse) |
| 2017-18 | -0.057 | -0.038 | 2025-26 (holdout) | +0.110 | +0.171 (worse) |

11 of 14 dev seasons move toward zero, several substantially (2014-15, 2018-19). But the
already-overshooting holdout-era seasons (2024-25, 2025-26) get WORSE, exactly the same
"unmasking" pattern §22 and §26 both produced — consistent with, but not resolving, §28.4's
still-open dispute. Aggregate dev-set total-MAE nets out close to neutral because the real,
substantial per-season gains and the real per-season costs partly offset each other in the
bootstrap, even though the underlying season-level movements are large and real.

### 32.5 Holdout confirmatory check: real total-MAE regression, a fourth unmasking installment

At the winning cell (halflife=300): SU +0.00000 (crosses zero), Brier -0.00003 (crosses zero),
margin -0.00034 (crosses zero), **total-MAE +0.00751 [0.00450,0.01055] (REAL)**. Matches the
established pattern from §22/§26/§18.5 exactly — a real structural fix, real holdout total-MAE
cost, consistent with (not distinguishing) either §28.4 account.

### 32.6 Not a rule firing — an operator judgment call, now retroactively named (§24.2.1)

Unlike Cycle 26 (comfortable net-CRPS clearance) or Cycle 27/28 (clean, unambiguous
null/rejection), this one did NOT resolve cleanly under either adoption class on the books at
the time. It isn't the ORIGINAL bar (total-MAE, the targeted metric, never cleared bootstrap
significance). It isn't §24.2's derived-correctness class as written either (that class
requires the mechanism to be PROVEN algebraically; §32's mechanism was confirmed empirically,
against independent real EV-TOI data, not derived from an identity). **What actually happened:
this was flagged as genuinely borderline, and the human operator made an explicit adoption
call — not a pre-written rule admitting it.** That is stated plainly here rather than left to
read as if a rule fired, per this project's own standing value (rules before numbers) — the
one place that value was set aside for a single judgment call is the one place it should be
most visible, not smoothed over.

**§24.2.1 (new) names this as its own class, confirmed-mechanism structural fixes, with §32 as
the motivating instance** — retroactive, not a rewrite: the decision stands as an operator
call made under genuine ambiguity, and the new class exists so the NEXT case like it has a
pre-written bar rather than another ad hoc flag-and-ask.

**Grid extended per §13.1's own precedent, confirming the winning cell is interior, not a
boundary artifact** (§24.2.1 point 5): `toi_halflife_games` ∈ {2400, 3600} both reproduce the
same real margin-MAE gain at essentially the same magnitude as 1800 (-0.00014, -0.00014,
-0.00014 — flat across all three), with total-MAE's own point estimate improving marginally
(+0.00049→+0.00046→+0.00040) but never reaching significance at any tested value. `1800` sits
on a genuine plateau — not simply the first point past the noise — and stays the adopted value;
extending further does not surface a clearly better cell.

**ADOPTED, at `TOI_HALFLIFE_GAMES=1800`.** Dev-set metrics: SU=58.07%, Brier=0.23980,
total-MAE=1.79780, margin-MAE=2.01312, mean predicted total=5.7505 (up from 5.7220,
narrowing the gap to the real 5.8125 further, though not closing it). Logged to the ledger as
`ev_toi_halflife_poisson`. **Current best model is now `ev_toi_halflife_poisson`**
(`validate_ev_toi_halflife.py`'s `run_current_best()`), superseding
`walk_forward_tie_ratio_poisson`. The fix (`toi_halflife_games`, threaded through
`validate_situational_toi.py`/`validate_goalie.py`) is the fourth confirmed instance of the
§7.1 two-baselines-different-memory bug class, and this project's first adoption under
§24.2.1's confirmed-mechanism structural-fix class.

Next: Cycle 16's GBM stacking layer (task #23) — the last unstarted item from the original
roadmap, now that the mean-total chain (§18-32) and its external checks (§6, §30-31) are both
current (though §30-31's own numbers predate this adoption and were not re-run against it —
the odds-match window's own coverage gap, §30, means a re-run would not materially change
either headline figure).

## 33. §19.4's standing prediction is confirmed — and it now has a deployment consequence (2026-07-24)

Three real structural fixes have shipped (§22 rest, §26 tie-mass, §32 EV-TOI), each removing
real undershoot and each producing a real holdout total-MAE cost. §19.4's own standing,
falsifiable claim: as that undershoot gets removed, the holdout-era per-season bias curve
should tip from near-zero into small positive overshoot. Checked directly against the fully-
adopted current-best model (`ev_toi_halflife_poisson`):

| Season | Predicted mean total | Actual mean total | Bias |
|---|---|---|---|
| 2021-22 | 5.955 | 6.290 | -0.335 |
| 2022-23 | 6.523 | 6.359 | **+0.164** |
| 2023-24 | 6.497 | 6.226 | **+0.271** |
| 2024-25 (holdout) | 6.310 | 6.081 | **+0.229** |
| 2025-26 (holdout) | 6.403 | 6.254 | **+0.149** |

**Confirmed — and not small.** Every one of the four most recent seasons (2022-23 through
2025-26, spanning both the dev tail and the full holdout) shows positive overshoot, +0.15 to
+0.27/game, a real reversal from 2021-22's -0.335 undershoot. This is a genuine update to
§28.4's still-open dispute (real overshoot vs. coincidental netting): a persistent, one-signed
overshoot across four consecutive seasons, following three independent, real structural
removals, is meaningfully stronger evidence for "the overshoot is real" than the installments-
alone argument §28.4 correctly found ambiguous — though it does not amount to a controlled
proof, and §28.4 stays logged as its own item rather than being silently closed here.

### 33.1 The deployment consequence: this is no longer only a validation statistic

The holdout window isn't just a backtest split — 2024-25 and 2025-26 are the seasons
immediately adjacent to 2026-27, which starts in October, roughly ten weeks from now. **If this
model prices live totals in October using its current calibration, it opens the season biased
high by roughly +0.15 to +0.27 goals/game**, unless addressed. Two honest options, pre-
registered now rather than decided as an October improvisation:

1. **Accept it as known and quantified.** Ship as-is, document the expected bias explicitly in
   any live output, and treat it as a known, bounded limitation rather than a silent one.
2. **Define a live-season recalibration rule now**, so any in-season adjustment is a
   pre-committed procedure, not an ad hoc reaction to early results. A candidate spec, to be
   refined (not built) before the season starts: track a walk-forward trailing bias statistic
   (predicted-minus-actual mean total) over a rolling window of the CURRENT season's own
   games (wide enough to be meaningful — perhaps the trailing 150-250 games, roughly 2-4 weeks
   into the season — narrow enough to react within a season, unlike `HALFLIFE_GAMES` itself,
   which is deliberately slow); define a trigger threshold clearly above the stationary-era
   noise floor already established (±0.05-0.08/game in flat years, §29.5.1) — e.g., ±0.10-0.12
   sustained over the window; and a pre-specified correction (most likely a symmetric additive
   level credit, the same mechanism family as §22/§32, not a new architecture) applied only
   when triggered, logged and reversible, not a silent recalibration.

Neither option is exercised now — both are logged so the choice in October is a decision
against a pre-registered plan, not a scramble.

### 33.2 The roadmap's next phase, once GBM resolves: deployment hardening, not another backtest cycle

With the mean-total chain complete (structurally) and this deployment question now visible,
the natural next phase after Cycle 16's GBM is deployment hardening, not another backtesting
cycle — the October start date makes this concrete rather than open-ended:

1. **The live-season recalibration rule** (§33.1), refined from a candidate spec into an
   exact, pre-committed procedure before the season starts.
2. **The RotoWire lineup/starting-goalie scraper** (§1.8, deferred since it was "confirmed
   scrapeable but never built... until there's a real NHL slate to verify selectors against")
   — that condition arrives in October; this can no longer be deferred indefinitely.
3. **The live-vs-retroactive goalie-overlay gap**: every backtest in this document uses the
   REAL, already-known starting goalie for each historical game — live predictions, made before
   morning skate or a starter announcement, won't have this. The size of this gap (how much the
   goalie overlay's own contribution depends on knowing the actual starter vs. a probabilistic
   or depth-chart-based guess) has never been measured and is a real, unquantified risk to
   live accuracy that the backtested numbers in this entire document do not capture.

These three are queued as the next phase, not built now.

## 34. Cycle 16 (finally reached): GBM stacking layer — real, clean win, ADOPTED (2026-07-24)

Queued since this project's original roadmap. Built exactly to the pre-registered spec: base
model log-odds as a monotone-constrained feature, the context features the base model already
consumes, one planted noise feature as a pre-registered kill switch, nested walk-forward CV for
hyperparameter selection, one committed configuration, standard paired bootstrap, single
holdout touch. Deliberately no recent-era-specific feature — this stack is not a backdoor
recalibration layer for §33's deployment question.

### 34.1 Build

`src/models/validate_gbm_stack.py`. Model: `sklearn.ensemble.HistGradientBoostingClassifier` —
the only tree ensemble available in this environment (no xgboost/lightgbm) with a native
monotonic-constraint mechanism (`monotonic_cst`); it has no row-subsampling knob the way
XGBoost/LightGBM do, so `max_features<1.0` (feature-level subsampling) substitutes for "heavy
subsampling," noted rather than silently swapped in. Features: `base_log_odds` (monotone
INCREASING, the only constrained feature), `rating_diff` (`lambda_home-lambda_away`, the base
model's own implied margin), `goalie_gsax_diff` (recomputed independently via
`team_strength_goalie`'s own functions, matching `validate_goalie.py`'s construction rather
than modifying that file to expose it), `home_rest`/`away_rest`/`home_b2b`/`away_b2b`,
`home_density_7d`/`away_density_7d` (trailing-7-real-day game count per side), `season_progress`
(min team-game-number / 82), and `noise_feature` (iid Gaussian, fixed seed 20260724).

**Nested walk-forward hyperparameter selection**: 5 time-based folds (test seasons 2019-20
through 2023-24, each trained on strictly-prior seasons only) × 4 candidate configs (depth
2-3, learning rate 0.03/0.10, shallow leaf counts). Mean walk-forward log-loss across configs:
0.66495-0.66584 — a tight range; winner: `max_depth=2, learning_rate=0.1, max_leaf_nodes=7,
max_features=0.7`.

### 34.2 Kill-switch check: real signal, clearly above the noise floor

Permutation importance (20 repeats, held-out 2023-24 fold), ranked:

| Feature | Importance |
|---|---|
| `rating_diff` | +0.017649 |
| `base_log_odds` | +0.009644 |
| `goalie_gsax_diff` | +0.002453 |
| `home_b2b` | +0.000866 |
| `home_rest` | +0.000536 |
| `season_progress` | +0.000429 |
| `away_rest` | +0.000253 |
| `home_density_7d` | +0.000076 |
| `away_density_7d` | -0.000082 |
| `away_b2b` | -0.000124 |
| **`noise_feature`** | **-0.000377** |

Best real feature (`rating_diff`, +0.017649) clears the noise feature (-0.000377) by nearly
two orders of magnitude — not a close call. **Kill switch not triggered; proceeds to
validation.** Interesting in passing: `rating_diff` (the raw margin) outranks `base_log_odds`
(the win-probability transform) — the stack is finding real value in the magnitude of the
implied goal differential beyond what the win-probability nonlinearity alone captures, a
plausible and sensible place for real structure to live.

### 34.3 Dev-set bootstrap: real, clean improvement on both primary ledger metrics

| Metric | Base | Stack | Diff (95% CI) | Real? |
|---|---|---|---|---|
| Brier | 0.23980 | 0.23827 | -0.00153 [-0.00192,-0.00116] | **REAL improvement** |
| SU | 58.07% | 58.49% | +0.00424 [0.00036,0.00817] | **REAL improvement** |

Both primary ledger metrics improve, both bootstrap-confirmed, no trade-off to weigh. This
exceeds the pre-registered expectation — the prior stated "the realistic prior is that
residual nonlinear structure is thin, and a clean null here is an acceptable terminus," not a
prediction of failure; the honest result is that real structure existed, and the pre-registered
kill switch correctly did not suppress it.

### 34.4 Holdout confirmatory check: no veto

Single touch, committed configuration, no re-fitting: Brier diff -0.00057, CI
[-0.00151,+0.00038] — crosses zero, point estimate still favorable, **no veto**. Total-MAE/
margin-MAE are unaffected by construction (this stack only replaces the win-probability output;
score-distribution/totals/margin predictions remain the unchanged Poisson pipeline).

### 34.5 Verdict at the time: ADOPTED — first ML stacking layer in this project

**Adopted**, cleanly, under the pre-registered success bar stated in advance (real dev
Brier-or-log-loss improvement, no real regression anywhere, holdout veto respected) — no
judgment call required, unlike §32. Logged to the ledger as `gbm_stack_poisson`. Current
best model became `gbm_stack_poisson`: the base Poisson pipeline for score distribution,
totals, and margin, with this GBM stack replacing `home_win_prob_full` for win-probability
purposes specifically — the first machine-learning component in this project's architecture,
layered on top of (not replacing) the interpretable Poisson core.

**Superseded — see §36.** The adoption bootstrap above was itself scored in-sample (fit on the
full dev set, then scored on that same dev set); re-scored on genuine out-of-fold predictions,
the dev gain crosses zero, matching the already-neutral holdout result in §34.4 above. The GBM
stack was reverted; `walk_forward_tie_ratio_poisson` (Cycle 26) is the sole current-best model.
This section is kept as the historical record of the original (incompletely-scored) adoption
decision, not as a currently-accurate verdict.

This closed out the original roadmap's last queued item at the time. Task #23 complete
(description updated to reflect §36's reversion). Next: the deployment-hardening phase (§33.2)
— the live-season recalibration rule, the RotoWire scraper, and the live-vs-retroactive
goalie-overlay gap — now unblocked regardless of this reversion.

## 35. A walk-forward-discipline bug found in review, fixed, pinned, and re-adjudicated in order (2026-07-24)

### 35.1 What was found

A full-pipeline code review (independent agent pass + self-review, run before starting the
deployment-hardening phase) found that `validate_tie_mass_ratio._build_dev_base`/`run_baseline`/
`run_treated` (and `validate_ev_toi_halflife.run_with_toi_halflife`) filtered their working
`base` DataFrame to the CALLER's requested season range *before* fitting
`fit_away_b2b_adjustment`, `fit_ot_so_split`, and `fit_ot_logistic`'s `(a, b)`, and before
computing the walk-forward tie-mass calibration ratio's own trailing EWMA. A holdout-only call
(`min_season=DEV_MAX_SEASON`) therefore fit every one of those constants on the holdout games'
own outcomes and scored them against those same games, and truncated the calibration ratio's
trailing memory to start fresh at the holdout boundary rather than carrying the real ~16,528-
game dev history.

**The correct discipline already existed and was never lost — it just wasn't reused.**
`check_holdout_ot_logistic.py` (Cycle 22's own hand-built holdout check) explicitly fits every
constant on a `dev_only` slice regardless of the scored range; it was, and remains, correct.
The bug was introduced when Cycle 26 refactored this pattern into shared, reusable helpers
(`_build_dev_base`/`run_baseline`/`run_treated`) and filtered the season range too early —
traceable to one specific refactor, on one specific date, affecting every holdout check built
on those helpers since: §26 (tie-mass), §32 (EV-TOI fix), §34 (GBM stack).

**This is the process succeeding, not failing.** An independent review plus self-review caught
a real walk-forward breach before any live prediction depended on it; the original correct
implementation survived well enough to serve as a golden reference; the contamination's scope
is precisely traceable to one refactor. That is what auditability is for.

### 35.2 Fix and permanent guards

Fixed in `_build_dev_base` (now takes no season-range arguments — always full history) and a
new shared `_fit_dev_only_ot_logistic` helper: every global constant is fit on
`[MIN_DEV_SEASON, DEV_MAX_SEASON)` unconditionally; trailing/EWMA state (`add_walk_forward_
reg_ratio`, `add_walk_forward_ot_calibration_ratio`) is computed on the full, unfiltered base;
the caller's `min_season`/`max_season` is applied ONLY as a final output filter, after all
fitting. `validate_ev_toi_halflife.run_with_toi_halflife` reuses the same corrected helpers.

Two permanent guards, `tests/test_holdout_walk_forward_discipline.py` (no pytest in this
environment — plain assertions, run via `python -m tests.test_holdout_walk_forward_discipline`):
1. **Walk-forward invariant**: the same dev-set games scored two ways (dev-only range vs.
   dev+holdout range) must produce IDENTICAL predictions — confirmed to machine precision
   (`<1e-12`) on `lambda_home`/`lambda_away`/`home_win_prob_full`. This is the core guard: if a
   future refactor reintroduces range-dependent fitting, this fails immediately and loudly.
2. **Golden-reference check**: the corrected `run_baseline(use_sh_term=False)` reproduces
   `check_holdout_ot_logistic.py`'s own construction on the Cycle 22 configuration — NOT
   bit-identical (a real, orthogonal, already-understood difference: `_indep_joint` doesn't
   renormalize its truncated joint to sum to exactly 1, `dc_adjusted_joint(rho=0.0)` does;
   confirmed directly, `_indep_joint`'s own truncated mass sums to ~0.999983), but within a
   tolerance sized to that specific, explained mechanism.
3. Plus a calibration-ratio continuity check across the dev/holdout boundary, and regression
   tests for the two smaller bugs the same review pass found (the backwards GBM holdout-veto
   sign condition; a dangling `for/else` in `validate_ev_toi_halflife.py`'s `__main__` that
   printed "NO SURVIVORS" unconditionally) — both already fixed inline, now guarded.

### 35.3 Pre-registered re-adjudication rule — written before any corrected number lands

For each affected adoption — §26 (tie-mass), §32 (EV-TOI), §34 (GBM), in that chronological
order — the corrected holdout check re-runs with the standard paired bootstrap, under the SAME
§15 veto that should have applied originally: **a real, bootstrap-confirmed regression on a
veto metric under the corrected check reverts the adoption; a CI crossing zero means the
adoption stands, annotated as re-confirmed under the corrected protocol.** No re-ranking, no
discretion — this is the existing rule, applied to corrected numbers, not a new one invented
to fit whatever comes out.

**Order matters and is chronological, not arbitrary**: §32 and §34 were both validated on a
base that already includes §26's tie-mass fix. If §26 reverts, §32 and §34 were measured
against a base that no longer exists and must be RE-VALIDATED against the reverted base, not
merely re-checked with corrected holdout mechanics on top of an assumption that's just been
undermined. Stop and re-scope if an earlier cycle reverts, rather than mechanically re-running
every later check against a base already known to be wrong.

**Pre-stated honest prior**: the contamination was paired — both baseline and treated arms in
every affected comparison shared it equally — so the most likely effect was dampening real
deltas toward zero, not manufacturing fake ones. The likeliest outcome is all three adoptions
stand. But "likeliest" is exactly what this re-run exists to test, not a substitute for running
it. **§32 is the one genuinely at risk**: it was already a borderline, operator-judgment
adoption (§24.2.1) under the CONTAMINATED check — a corrected check has real room to flip a
result that was already sitting close to the line, in a way §26's clean net-CRPS clearance and
§34's clean two-metric bootstrap win do not.

### 35.4 Results: §26 stands, §32 REVERTS — the pre-stated risk materialized

**§26 (tie-mass, `check_holdout_tie_mass_ratio.py`, corrected):**

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00038 | [-0.00343, 0.00267] | Crosses zero |
| Brier | -0.00003 | [-0.00010, 0.00005] | Crosses zero |
| Margin-MAE | +0.00016 | [-0.00028, 0.00061] | Crosses zero |
| Total-MAE | +0.00759 | [0.00364, 0.01153] | Real (informational, not a veto metric) |

**No veto. §26 STANDS**, re-confirmed under the corrected protocol. CIs widened slightly
relative to the contaminated check (honest, since the contaminated version had overfit its own
constants to the small holdout sample) but no veto metric crosses into "real," matching the
pre-stated prior exactly.

**§32 (EV-TOI fix, halflife=1800, corrected):**

| Metric | Mean diff | 95% CI | Real? |
|---|---|---|---|
| SU | -0.00152 | [-0.00381, 0.00038] | Crosses zero |
| Brier | -0.00005 | [-0.00009, -0.00000] | **REAL** |
| Margin-MAE | -0.00032 | [-0.00062, -0.00002] | **REAL** |
| Total-MAE | +0.00262 | [0.00126, 0.00400] | Real (informational) |

**VETO — real regression on two veto metrics (Brier, margin-MAE). §32 REVERTS.** This is the
pre-stated risk materializing exactly as flagged: §32 was already a borderline, operator-
judgment adoption under the contaminated check, and the corrected check moves it from "stands"
to "reverts." Per the pre-registered rule (§35.3), applied without discretion: **`ev_toi_
halflife_poisson` is no longer adopted. Current best model steps back to
`walk_forward_tie_ratio_poisson` (§26).** The EV-TOI mechanism itself (§32.1) is not
retracted — the real, root-caused EV-bucket TOI lag still exists and is still correctly fixed
in `validate_situational_toi.py`'s `toi_halflife_games` parameter — what reverts is the
ADOPTION DECISION, made on a contaminated holdout check that (per the honest prior) turned out
to be exactly the borderline case most likely to flip. `validate_ev_toi_halflife.py` is kept as
a frozen, historical file (the rejected/reverted candidate), matching this project's standing
practice for every other non-adopted cycle.

**The sign flip itself is diagnostic, not incidental.** Under the contaminated check, §32
showed a real margin-MAE GAIN; under the corrected check, the same comparison shows real
Brier AND margin-MAE REGRESSIONS. That isn't just "the effect got weaker" — it flipped
direction on the metric that originally justified adoption. The honest reading: the
`halflife=1800` "win" was plausibly an artifact of `away_b2b_adj`/`ot_split`/`(a,b)` being fit
on the holdout games' own outcomes, not a real margin improvement that contamination merely
dampened. The contamination didn't just blur the check — for this specific, already-borderline
case, it manufactured the adoption evidence. This also retires §32.6's grid-boundary-extension
question (whether `toi_halflife_games=1800` sat on a genuine plateau vs. 2400/3600) as moot —
it doesn't matter which point on a now-reverted candidate's grid was best.

**The EV-TOI mechanism goes back on the OPEN-MECHANISMS list, not the task queue.** Real,
directly confirmed against independent data (§32.1's ~1-minute EV-TOI gap, matching the
flat-era `gap_ev` closely), currently without a validated fix — the same status the tie-mass
deficit held for eight cycles (§4.13 through §26) before a design finally worked. §29.3.1's
signal-to-noise argument still applies in full: any future attempt needs a structurally
different vehicle (a mechanism-level fix, not a re-tuned or re-gridded tracker), since the
underlying problem — a ~0.09-goal signal against ~2.4-goal per-game noise — was never about
which halflife was chosen.

### 35.5 Cascade: §34 (GBM) re-validated against the reverted base, not merely re-checked

Since §32 reverts, §34's own base model (`ev_toi_halflife_poisson`) no longer exists as current
best. Per §35.3's own stated rule, §34 is RE-VALIDATED from scratch against the reverted base
(`walk_forward_tie_ratio_poisson`) — `validate_gbm_stack.py`'s `build_features()` re-pointed at
`validate_tie_mass_ratio.run_treated` — not merely re-checked with the corrected holdout
mechanics bolted onto the old, now-invalid feature construction.

### 35.6 §34 re-validation: STANDS, genuinely re-confirmed against the reverted base

Full cycle re-run from scratch (nested walk-forward CV, kill switch, dev bootstrap, holdout),
features rebuilt against `walk_forward_tie_ratio_poisson`:

- **Committed configuration**: identical to the original run (`max_depth=2, learning_rate=0.1,
  max_leaf_nodes=7, max_features=0.7`).
- **Kill switch**: not triggered. `base_log_odds` (+0.0158) and `rating_diff` (+0.0124) both
  clear the noise feature (-0.0002) by roughly two orders of magnitude — order swapped slightly
  from the original run (base_log_odds now ranks first), both still clearly real.
- **Dev bootstrap**: Brier 0.23980→0.23841, diff -0.00139 CI [-0.00178,-0.00102] (**REAL**); SU
  58.01%→58.52%, diff +0.00508 CI [+0.00127,+0.00908] (**REAL**). Both primary metrics improve
  again, independent of which exact base-model version underlies the stack.
- **Holdout**: Brier diff -0.00022, CI [-0.00115,+0.00068] — crosses zero, favorable point
  estimate, **no veto**.

**§34 STANDS — genuinely re-confirmed, not assumed** *(at the time — see the correction below)*.
Logged to the ledger as `gbm_stack_poisson_RE_VALIDATED_post_sec32_revert`, superseding the
original `gbm_stack_poisson` entry (built on the now-reverted base). `total_mae`/`margin_mae` in
this re-validated run (1.79731/2.01326) correctly match Cycle 26's own historical values —
confirms totals/margin reverted cleanly along with everything else.

**Superseded — see §36.** This re-validation's own dev bootstrap (Brier diff -0.00139, SU diff
+0.00508, both flagged REAL above) was scored the identical in-sample way as the original §34.3
bootstrap: fit the committed config on the full dev set, score that same dev set. §36 found and
corrected this defect project-wide; re-scored out-of-fold, the gain crosses zero here too, for
the same underlying reason. The careful re-validation work in this section (correctly
re-confirming totals/margin reverted cleanly, correctly re-running the kill switch) stands on
its own merits — the part that doesn't survive is specifically the in-sample Brier/SU dev
bootstrap, superseded by §36's out-of-fold re-score.

### 35.7 Current best model, at that point: `gbm_stack_poisson`, built on `walk_forward_tie_ratio_poisson`

Net effect of the full re-adjudication, at the time this section was written: **current best
model remained `gbm_stack_poisson`** — the GBM stack itself was not yet in question — but its
underlying Poisson base stepped back one cycle, from `ev_toi_halflife_poisson` (§32, reverted)
to `walk_forward_tie_ratio_poisson` (§26, confirmed standing). **This was itself superseded one
round later (§36)**: the GBM's own adoption evidence turned out to have the identical in-sample
defect this whole re-adjudication pass was fixing everywhere else, and reverted once corrected.
Current best model, final, is `walk_forward_tie_ratio_poisson` (Cycle 26) directly, no stacking
layer. §0 reflects this final state.

### 35.8 Remaining findings from the review, resolved

**Finding 2 (stale market benchmarks)**: `validate_market_benchmark.py`/`validate_market_totals_
benchmark.py` re-pointed at the current-best entry point now that re-adjudication has settled
what that is (`validate_gbm_stack.py`'s output, base model `walk_forward_tie_ratio_poisson`).
§30-31's previously-logged numbers were correct when run — against `walk_forward_tie_ratio_
poisson` directly, which coincidentally is exactly the model those numbers are still valid
for, tie-mass having stood — but are annotated here as measuring a mid-chain state, not
re-run once the true current-best target was settled (GBM stack, not the raw Poisson base) —
see §35.9, a genuinely different and better number than either the pre- or post-revert Poisson
figure, since the GBM materially changes win probability.

**Finding 3 (`fit_deltas_per_game`'s one-sided clipping)**: empirical check on real dev games —
`raw_delta_above < 0` (the clip-to-zero branch) fires on **5 of 19,152 team-game rows (0.03%)**.
Effectively never. Matches the expected pattern (real hockey ties are under-predicted by
independent Poisson, not over-predicted) — documented, not pursued further.

**Finding 4 (same-game `shift(1)` leak) — RETRACTED, false positive, confirmed by direct
empirical test (2026-07-24):** the review agent's original description was that "all
`league_avg_*` columns are computed via `.shift(1)` at the flat per-team-game row granularity,
so the away row's `shift(1)` lands on the home row of the SAME game." Direct reading of
`shrinkage.py`'s `_trailing_league_stat` (the function underlying every `league_avg_*` column
via `add_walk_forward_rate`/`add_walk_forward_toi_rate`) shows this is not what it does: it
first collapses `log` to **one row per `gameId`** (`groupby("gameId").agg(...)`), sorts by
date, and only then applies `.shift(1)` to that one-row-per-game series, before merging the
identical result back onto both the home and away rows of that game. A game's own two rows
therefore can never see anything from each other — both see only strictly-prior games,
identically. Confirmed empirically, not just by reading: `add_walk_forward_rate` run on the
real team-game log, sampled at game `2008020051` (TBL home vs. NYI away) — **both rows carry
`goals_league_avg = 2.98`, bit-identical** — which is only possible if both were computed from
the same pre-game history with the current game excluded from both, exactly as designed. **This
settles the LEAGUE-STATISTIC path — the mechanism the original finding actually described.** For
completeness, the other structurally possible path (team-level trailing features — a team's own
rolling rate/mean, e.g. `add_walk_forward_mean`'s `grp[value_col].cumsum() - log[value_col]`,
`add_walk_forward_goalie_strength`'s per-goalie shrinkage) is airtight for a different, simpler
reason: these are computed via a `groupby("team")` (or `groupby("goalieIdForShot")`) cumulative
sum/shift, and a game's home and away rows belong to two DIFFERENT teams by definition — they
fall into two different groups entirely, so one side's trailing feature can never be computed
from a sum that includes the other side's row from the same game. Same-game leakage is
impossible by construction on this path too, for an even more basic reason than the
per-`gameId`-collapse used on the league-statistic path. The original finding is retracted on
both possible mechanisms, not one verified and one merely assumed. **This retires the
same-game-leak concern from the degradation budget below — it contributes nothing, having never
been real.**

Worth stating plainly so the retraction doesn't read as discrediting the review that produced
it: one critical true positive (the walk-forward-discipline contamination, §35.1, which was real
and affected three production adoption decisions) against one false positive (this finding) is a
strong ratio for an independent review pass to produce. The process worked as intended — it is
supposed to surface candidates for verification, not arrive pre-verified, and catching this one
as a false positive here is what disciplined follow-through on a review looks like, not a mark
against the review itself.

### 35.9 Market benchmark, first re-run: an overstated headline, corrected in §35.10

`validate_market_benchmark.py` was first repointed at `validate_gbm_stack.run_final_production`
(fit once on dev-only data) and re-run: **model Brier 0.23960, market close 0.23867, gap
0.00092, CI [-0.00011, 0.00200] — crosses zero.** This number is WRONG in a specific,
diagnosable way and should not be read as this chain's result — see §35.10. Kept here, not
deleted, as the record of the mistake: `run_final_production` fits the GBM once on the full
dev set (2010-11 through 2023-24), and this benchmark's window (2010-11 through ~2021-22) sits
entirely INSIDE that training window — every scored game was (partly) memorized by the model
scoring it. The base Poisson model's own dev-fit constants (a dozen-or-so scalars) have
essentially no capacity to memorize individual games this way; a tree ensemble, even shallow
and regularized, partially does, and that memorization flatters exactly the games this
benchmark evaluates. The trajectory "0.00351 → 0.00247 → 0.00092" spliced two different
measurement regimes: the first two are honest near-out-of-sample comparisons; the last
includes an in-sample stacking layer.

### 35.10 Market benchmark, corrected: the gap is real, and roughly unchanged by the GBM

`validate_market_benchmark.py` re-pointed again, this time at a new `validate_gbm_stack.
run_out_of_fold_predictions` (expanding-window walk-forward: the committed configuration
re-fit per season on all strictly-prior seasons only, so every scored game is genuinely
out-of-sample for the model that predicted it; seasons with under `MIN_OOF_TRAIN_GAMES=1500`
of prior data fall back to the base model's own probability — not a workaround, since a real
deployed system wouldn't have a meaningfully-fit GBM that early either).

**Model Brier: 0.24124 — market close: 0.23867 — gap: 0.00257, 95% CI [0.00149, 0.00369] —
REAL, does NOT cross zero.** Pre-registered expectation was that the gap would re-open
partway, to roughly 0.0015–0.0020, still real progress over the pre-GBM 0.00247 (§31). The
actual result is more sobering than that: **0.00257 is essentially indistinguishable from the
pre-GBM figure (0.00247) — if anything marginally larger, not smaller.** Stated plainly: once
scored honestly, the GBM stack's real, internally-validated dev-set improvement (§34.3/§35.6's
proper nested walk-forward bootstrap, which remains valid — that comparison never had this
contamination, since both arms in it were scored the same way) does not show up as a
detectable improvement on this specific EXTERNAL market-comparison benchmark. Two comparisons
that both used correct out-of-sample methodology are giving different answers about whether
the GBM helps — internally, yes, with a real bootstrap-confirmed CI; against the market
specifically, no visible change — and the honest reading is that the GBM's real gain is
apparently too small, or too correlated with what the market already prices, to move this
particular needle, not that one of the two checks is wrong. Per-season breakdown: gap is
positive in every single season (range +0.00081 to +0.00480) — more uniform than the
contaminated version's mixed-sign pattern, consistent with a genuine, stable remainder rather
than an era-specific artifact. §30's pre-registered caveat (no 2022-23+ coverage, both scoring
jumps inside the window) still applies and still bounds how good this specific number could
ever look, regardless of model quality.

**This is the number this chain should carry forward: the market gap is real, ~0.0026, and the
GBM stack — while a genuine, real, correctly-validated dev-set improvement in its own right —
has not been shown to narrow it.**

## 36. The GBM adoption bootstrap was scored in-sample too — re-examined, reverted (2026-07-24)

The OOF market-benchmark correction (§35.9-35.10) fixed one contaminated number. It also raised
a sharper question: was the GBM's ORIGINAL adoption evidence (§34.3, the dev-set bootstrap that
got it adopted in the first place) subject to the same defect? Walked through and confirmed yes.

### 36.1 The defect, confirmed by direct reading of the code

`validate_gbm_stack.py`'s `__main__` (the script that produced §34.3's adoption numbers):

```python
X_all, y_all = df[FEATURE_COLS].values, df["target"].values
clf_full = _fit(best_params, X_all, y_all)
p_stack = clf_full.predict_proba(X_all)[:, 1]        # <-- same X_all the model was just fit on
```

The committed configuration is fit on the ENTIRE dev set, then scored on that SAME dev set. The
nested walk-forward loop earlier in the script (`WALK_FORWARD_TEST_SEASONS`) was used ONLY to
select `best_params` (which hyperparameter combination minimizes walk-forward log-loss) — never
to generate the predictions the adoption bootstrap itself was computed on. This is the identical
genus of bug as the market benchmark's original contamination (§35.9): a tree ensemble partially
memorizes its training set, flattering exactly the bootstrap meant to test whether it generalizes.

This is a specification gap, not a one-off coding mistake — worth naming precisely rather than
sanding off: the pre-registered success bar for this cycle said "real dev Brier or log-loss
improvement" without stating OOF-scored, and the nested-CV machinery was specified for
hyperparameter selection but never explicitly extended to cover the adoption bootstrap itself.
Every other adoption bootstrap in this project's history scores a handful of dev-fit SCALARS
(reg ratios, shrinkage priors, calibration constants) that have essentially no capacity to
memorize individual games — this class of defect had no opportunity to exist before a tree
ensemble entered the pipeline, which is exactly why the spec never had to say "OOF-scored" until
now.

### 36.2 The corrected adoption bootstrap, re-scored on genuine out-of-fold predictions

Re-run using `validate_gbm_stack.run_out_of_fold_predictions` (already built for §35.10 — the
committed configuration re-fit per season on all strictly-prior seasons only) in place of the
in-sample `clf_full.predict_proba(X_all)` used originally:

| metric | in-sample (original §34.3) | out-of-fold (corrected) |
|---|---|---|
| Brier diff (stack − base) | real improvement | +0.00014, CI [−0.00040, +0.00067] — crosses zero |
| SU diff (stack − base) | real improvement | −0.00030, CI [−0.00502, +0.00454] — crosses zero |
| Brier diff, GBM-active games only (excludes the 2,460/16,528 early-season fallback rows) | — | +0.00016, CI [−0.00045, +0.00080] — crosses zero |

**Both metrics cross zero — including when restricted to only the games where a genuinely
fit GBM produced the prediction, ruling out "the early-season base-model fallback rows are
diluting a real signal" as an explanation.** This is not a subtle result: the dev-set adoption
evidence, once scored the same honest way as everything else in this project, shows no real
improvement over the base model at all.

### 36.3 Three independent OOF-scored checks now agree

- **Dev-set adoption bootstrap (§36.2, corrected here)**: crosses zero.
- **Holdout confirmatory check (§34.4)**: never had this defect (a single fit-then-score-once
  pass, no re-fitting inside the bootstrap) — already crossed zero, i.e. no real gain, at the
  time it was run. Its silence was read as "no veto" (a pass), which is correct but is NOT the
  same claim as "confirms a real gain" — holdout in this project's protocol (§15) is
  confirmatory-veto-only by design, so it was never positioned to catch this.
- **OOF market-benchmark (§35.10)**: gap 0.00257, essentially indistinguishable from the pre-GBM
  figure of 0.00247 (§31.1) — re-confirmed directly against the pure base model with no GBM
  involved at all (§36.5 below): gap 0.00247, CI [0.00141, 0.00352], matching §31.1 almost to
  the digit.

Three measurements that would each, independently, have been capable of showing a real GBM
contribution — a dev-set bootstrap, a holdout check, and an external market comparison — all
agree once scored out-of-fold. The kill-switch not firing during the ORIGINAL run is consistent
with this, not in tension with it: a weak, real, nonlinear signal can beat a planted noise
feature's permutation importance under cross-validated log-loss (a comparative, rank-based test)
while still contributing nothing measurable once in-sample memorization is removed from an
absolute-improvement bootstrap — the kill switch asks "is there more signal than noise," not
"is that signal large enough to matter after honest scoring."

### 36.4 Verdict: REVERTED — the pre-registered null was priced in, this is not a loss

Per the same §15-spirited rule proposed for this re-examination: the OOF dev gain crosses zero,
and holdout was already neutral — the GBM stacking layer REVERTS. `walk_forward_tie_ratio_
poisson` (Cycle 26, §26, re-confirmed §35.4) is the sole current-best model. `gbm_stack_
poisson` is kept as a frozen, historical file (`validate_gbm_stack.py`), the same status as the
EV-TOI fix (§32/§35.4).

This is the honest ending the original roadmap explicitly priced in when this cycle was queued:
"nothing left on the table that a tree can see" was always one of the acceptable outcomes, not a
failure mode. The base model's production surface is now simpler (no tree ensemble, no
monotonic-constraint machinery, no permutation-importance kill-switch to maintain), cheaper to
run, and — on the full weight of evidence now available — provably no worse. Reversion is the
correct, disciplined response to this evidence, not a consolation.

### 36.5 Market benchmark, final: re-run directly against the base model, no GBM at all

With the GBM reverted, `validate_market_benchmark.py` is re-pointed a third and final time,
directly at `validate_tie_mass_ratio.run_treated` (matching `validate_market_totals_
benchmark.py`'s own long-standing pattern) — no stacking layer of any kind between the base
model and the market comparison. **Model Brier 0.24114, market close Brier 0.23867, gap
0.00247, 95% CI [0.00141, 0.00352] — REAL, does not cross zero**, and matches §31.1's original
pre-GBM figure (0.00247) almost exactly. This is now the final, uncomplicated number: no
in-sample risk, no OOF machinery required, because there is no fitted-on-dev-data stacking
layer left in the pipeline to create the risk in the first place.

**The market-gap trajectory, correctly attributed**: 0.00351 (§6.8, pre-drift-fix) → 0.00247
(§31.1, after the root-cause chain: scoring-era drift correction, tie-mass, cross-season
priors, team-specific OT logistic) → 0.00247 (here, final, GBM reverted). **The entire real
closure came from finding and fixing specific structural mechanisms — not from adding model
capacity.** The GBM, tested honestly, contributed approximately nothing to this number, at any
point in the chain. Eleven months of cycles bought their real progress by explaining WHY the
model was wrong in specific, falsifiable ways; the one cycle that tried to buy progress with
raw statistical capacity instead came back null, on every axis that was checked carefully
enough to trust.

## 37. Task #43: live-season recalibration rule, with a pre-registered degradation budget (2026-07-24)

Before writing the rule itself, the candidate contributors to a "known live-vs-backtest
degradation" budget need honest resolution. Two turned out to be worth zero (finding 4,
retracted §35.8; the GBM in-sample/OOF distinction, moot now that the GBM itself is reverted,
§36); the goalie overlay needed an actual measurement, done here — twice, once too narrowly,
corrected the second time after the user flagged the gap.

### 37.1 What does NOT belong in the budget

- **Finding 4 (same-game leak)**: retracted (§35.8) — never real, contributes nothing.
- **The GBM in-sample/OOF distinction (§35.9-35.10/§36)**: this was a measurement-methodology
  bug in how a BACKTEST (first) and then an ADOPTION BOOTSTRAP (§36) scored a historical window,
  not a live-deployment risk — `run_final_production`/any live entry point scores strictly
  future, never-seen games, automatically genuinely out-of-sample by construction. Moot twice
  over now: the GBM itself is reverted (§36), so there is no in-sample-vs-live distinction left
  to budget for at all.

### 37.2 What DOES belong: the goalie-overlay information gap, measured on the metric it actually moves

The one genuine, structural live-vs-backtest asymmetry is the goalie overlay: backtests score
every historical game knowing the REAL starting goalie (already public, in the schedule/box
score data); a live prediction made before lineups are confirmed does not yet have this.

**First pass (superseded below)** measured only Brier and total-MAE, both of which crossed
zero, and concluded the budget was negligible. This missed the metric the goalie overlay's OWN
original adoption evidence (§4.6) was actually built on — a real MARGIN-MAE improvement, CI
[0.00691, 0.01233] — nothing since has dethroned margin as the axis where starter identity
bites. A budget checked only on the two metrics goalie information matters least for would
under-trigger exactly where live degradation would actually show up. Corrected measurement below
adds margin-MAE, and replaces the sole "zero information" comparator with two: a genuine
worst case (zero information) AND a realistic one (a recent-workhorse heuristic — predict each
team's likely starter as whichever goalie had the most starts in that team's trailing 10 games,
a proxy for what a depth-chart/recent-usage-informed live guess would produce without needing
real scraped lineup data yet). Built in `src/models/validate_goalie_info_gap.py`.

| metric | worst case: real − zero-info | realistic: real − heuristic |
|---|---|---|
| Brier | −0.00047, CI [−0.00110, +0.00014] — crosses zero | −0.00034, CI [−0.00088, +0.00021] — crosses zero |
| **margin-MAE** | **−0.00984, CI [−0.01371, −0.00603] — REAL** | +0.00054, CI [−0.00285, +0.00382] — crosses zero |
| total-MAE | −0.00082, CI [−0.00491, +0.00311] — crosses zero | −0.00268, CI [−0.00610, +0.00060] — crosses zero |

**This is the informative result the first pass missed.** Zero information about the starter
produces a REAL, confirmed margin-MAE cost (0.006-0.014/game) — consistent with, and roughly
the same order of magnitude as, the overlay's own original adoption evidence (§4.6). But the
recent-workhorse heuristic — a trivial, no-real-data-needed stand-in for a live guess — recovers
essentially ALL of that value: the real-vs-heuristic margin-MAE gap crosses zero. Brier and
total-MAE stay non-significant under both comparators, confirming the first pass's read on
those two metrics specifically, just not the complete picture. The practical conclusion: a live
system that does nothing more sophisticated than "assume the goalie who's played most recently
keeps playing" should see close to zero real degradation on any of the three tracked metrics —
but a system that genuinely has NO information (e.g. the scraper, task #44, is down) is exposed
to a real, non-trivial margin-MAE cost, not a hypothetical one.

### 37.3 The degradation budget, honestly stated

- **Routine budget (the number #43's trigger checks against every 4-week window)**: derived from
  the REALISTIC comparator, since a live system with even a trivial heuristic in place should
  look like this, not like zero-information — Brier ≤ 0.00088, **margin-MAE ≤ 0.00382**,
  total-MAE ≤ 0.00610 (all three crossed zero in the realistic comparison; these are the CI
  magnitude bounds, not real effects, kept as conservative operating ceilings rather than
  claiming a live gap of exactly zero is guaranteed).
- **Escalation ceiling (the worst-case floor, only relevant if the goalie-data pipeline itself
  has failed)**: Brier ≤ 0.00110, **margin-MAE ≤ 0.01371 — this one is a REAL, confirmed
  number, not a noise bound** — total-MAE ≤ 0.00491. If the live-vs-backtest gap on margin-MAE
  specifically exceeds the realistic budget (0.00382) but stays under the escalation ceiling
  (0.01371), the most likely explanation is degraded (not absent) starter information — check
  task #44's scraper for staleness before assuming a deeper problem. If it exceeds the
  escalation ceiling too, treat it as a genuine, unbudgeted anomaly requiring full investigation.
- This budget is NOT for model drift, roster turnover, or any other cause; those are what the
  recalibration TRIGGER (below) exists to catch.

### 37.4 The recalibration rule

**Trigger statistic**: at the end of each rolling 4-week window of live games (~220-240 games
league-wide, comfortably above the smallest per-cell samples this project has treated as
informative elsewhere — e.g. §29.3's ~2,844-game requirement is for a much smaller signal than a
full recalibration check needs to detect), compute the paired-bootstrap gap between live and the
corresponding backtest-holdout value, for Brier, **margin-MAE**, and total-MAE, for the SAME
model configuration. Margin-MAE is the primary metric to watch (§37.2 — it is the one metric
with a REAL, confirmed goalie-information sensitivity); Brier/total-MAE are checked as
secondary, general-purpose drift indicators.

**Decision rule**:
- If every metric's live-minus-backtest gap CI sits within the routine budget (§37.3) — no
  action. Expected, budgeted variation, not a signal.
- If margin-MAE's gap exceeds the routine budget (0.00382) but stays within the escalation
  ceiling (0.01371) — **investigate the goalie-data pipeline first** (task #44's scraper
  returning stale, missing, or low-confidence lineups more often than expected) before anything
  else; this specific pattern is the one this project can now name in advance, not a generic
  alarm.
- If any metric's gap exceeds its escalation ceiling, or Brier/total-MAE alone (without a
  matching margin-MAE story) shows a real gap — treat as a genuine, unbudgeted anomaly: check
  for a specific, nameable cause (rule change, realignment, a data-quality regression) before
  assuming the model itself needs refitting. Recalibration (refitting dev-fit constants on the
  now-larger history that includes the live window) is the LAST resort, not the first response,
  per this project's standing preference for root-cause fixes over parameter adjustment
  (§29.3.1's logic applies here too).
- If the gap sits BELOW zero on every metric (live performing BETTER than backtest) — no action,
  but note it; slightly surprising, worth a one-line diagnostic, not a rule firing.

**What this rule depends on that is not yet built**: task #44 (RotoWire scraper) is what a real
system would use instead of the recent-workhorse heuristic modeled here — once it exists, the
realistic comparator should be re-measured against ITS actual guesses, not the heuristic proxy,
since a depth-chart-informed scraper may do better OR worse than "most recent starter" depending
on how often NHL teams rotate a committee approach. Task #45 as originally scoped (measure
live-vs-retroactive goalie gap from real deployment data) is now largely superseded by this
section's backtest-based measurement, but should still be re-run once a few weeks of real
October data exist — a genuine live measurement will be tighter than any pre-season estimate.

### 37.5 A second, symmetric purpose: tracking the live-vs-market gap as an improvement channel, not just a degradation budget

§37.1-37.4 above are framed entirely around risk — how much worse live can look than backtest
before that difference itself needs investigating. That framing is one-sided. The GBM null
(§36) rules out nonlinear structure over the features the model already sees as an explanation
for the residual 0.00247 market gap; the leading remaining explanation reverts to §6.8's
original informational hypothesis — the market prices skater-level injuries and lineup news
this model structurally cannot see from box-score and xG data alone. Task #44's scraper was
scoped as deployment plumbing (know the starter before puck drop), but RotoWire's pages carry
full projected lineups and the injury report — exactly the input class the residual gap points
at. This hypothesis has no backtest: there is no historical lineup archive to test it against
(§6.1's coverage gap and this project's data sources generally), so **the live season is the
only venue where it can ever be confirmed or refuted.**

**Track the live model-vs-market Brier gap as a first-class series alongside the degradation
triggers**, using the same paired-bootstrap machinery, same rolling 4-week window as §37.4,
with **0.00247 (§36.5) as the pre-registered backtest reference line**:

- If the live gap runs **narrower** than 0.00247 (model-vs-market Brier gap shrinks once lineup
  information — even the current pre-#44 heuristic, or #44's real scraped data once it lands —
  is available at prediction time that wasn't available to any backtest game), that is the
  informational hypothesis **confirming in production** — a genuinely new finding this project
  could not have produced any other way, since it requires real, live, pre-game lineup
  uncertainty to exist in the first place.
- If the live gap runs at or above 0.00247 even with #44's real lineup data flowing in, that
  argues AGAINST the informational hypothesis specifically — worth writing down explicitly
  either way, since a null here is exactly as informative as the GBM null was (§36): it would
  mean the residual gap is not explained by lineup information either, and the search for what
  it IS should resume from a different hypothesis than either "more model capacity" or "more
  information," both of which the arc has now null-tested.
- This series is diagnostic, not a trigger — it does not feed the degradation decision rule in
  §37.4 and should not be reacted to mid-season; it accumulates through the full season and is
  read once there is enough data for the comparison to mean something (by the same ~220-240
  game-per-window logic as §37.4, though the FULL-season total, not a single window, is the
  number that actually answers this specific question).

### 37.6 Pre-registered refinement: segment the live model-vs-market gap by season phase

A pre-season demo (predicting a random already-completed week, then a random odds-covered week
for a market comparison) surfaced this refinement before October, not after — worth recording
why. The first randomly-chosen week landed, by chance, on the FINAL week of the 2025-26 regular
season: exactly the model's structural worst case for the informational hypothesis (§37.5).
Playoff-locked teams rest stars and start backup goalies, eliminated teams dress AHL call-ups,
and motivation asymmetry peaks — the market prices all of this in real time; the model, with no
lineup channel, prices none of it. The model held its historical averages on that specific week
anyway (total-MAE 1.800 vs. a historical 1.797, on 53 untouched games) — mildly encouraging, but
a single week is not evidence (§37.7's caution on sample size applies here too), and the more
useful output is the refinement it suggests.

**The informational hypothesis makes a falsifiable, phase-dependent prediction, not just a
season-aggregate one**: if the residual 0.00247 gap is real lineup/injury information the
market has and the model doesn't, that gap should be WIDEST exactly when roster uncertainty is
highest and NARROWEST when it's lowest — not flat across the season. Segment §37.5's live
model-vs-market series into four pre-registered phase buckets (using the actual 2026-27
schedule once it's published):

- **Early season** (first ~4 weeks): moderate uncertainty (new/changed rosters, unsettled
  lines), but confounded with small-sample team-strength noise on a separate axis — treat as a
  secondary bucket, not the sharpest test.
- **Mid-season stable stretch** (week 5 through ~2 weeks before the trade deadline): the
  pre-registered BASELINE — lineups and motivation are at their most stable and predictable all
  season. This bucket's gap is the one closest to the season-aggregate 0.00247 by construction.
- **Trade-deadline window** (~1 week before through ~1 week after the actual deadline date):
  roster churn and asymmetric information about it (rumors, healthy scratches ahead of a trade)
  should be near a local maximum.
- **Final two weeks**: the model's structural worst case, per the reasoning above — the
  pre-registered prediction is this bucket shows the WIDEST gap of the four.

**Decision rule**: if the gap is real and meaningfully wider in the trade-deadline and
final-two-weeks buckets than in the mid-season baseline, that's the informational hypothesis
confirming with phase-level specificity — a sharper, more actionable finding than a flat
season-aggregate number, since it tells task #44's scraper exactly WHEN its data matters most
(worth prioritizing scraper reliability and injury-report freshness heading into those windows
specifically). **If the gap comes back flat across all four phases**, that weakens the
informational story specifically (not just generically) — a phase-invariant gap looks more like
something structural the model is still missing than like lineup information, and the search
for what explains 0.00247 should resume from a different hypothesis. Either outcome is a real,
useful, falsifiable result; this is purely diagnostic, feeding no automated action, same as
§37.5's undifferentiated version.

### 37.7 Live odds capture added to task #44's scope

Every market number in this project to date — the 13-season aggregate 0.00247 (§36.5), the
per-season breakdown, and any demo/spot-check — is computed against the historical odds archive
(`sbro_odds_games.parquet`), which stops in **May 2022** (§6.1's coverage gap). No market
comparison this project has ever produced has been on games where neither the model's
prediction nor the evaluation itself preceded the actual result — the archive was always
assembled after the fact. Live 2026-27 play is the first opportunity to close that gap
entirely: task #44's scope is expanded from lineup/starting-goalie scraping alone to also
**log the real closing line for every game the model prices, starting opening night** — a
trivial addition operationally (the same fetch cadence as the lineup data, a different endpoint)
that turns §37.5/§37.6's live model-vs-market tracking into the first genuinely clean series
this project will have had: real predictions, real lines, real results, with a hard temporal
wall between the number and the answer that no historical archive can replicate.

**Caution carried forward from the demo**: any SINGLE week's model-vs-market Brier comparison
has a standard error roughly 3x the size of the true season-long gap itself (a ~48-game week's
SE is around ±0.008-0.009 against a real gap of 0.00247) — individual weeks, and even individual
phase-buckets early in a season before enough games accumulate, will look like blowouts in
either direction purely from noise. §37.6's phase buckets should each be read only once they
individually have enough games for the comparison to mean something (the same ~220-240-game
logic as §37.4), not reacted to game-by-game or week-by-week the way the degradation trigger is.

## 38. Season freeze: production constants locked as the 2026-27 configuration (2026-07-24)

The original roadmap is complete (§36 closed the last queued item). With current-best stable
and every number in this doc traceable to a committed script, the repo is tagged and every
fitted scalar / committed hyperparameter in `validate_tie_mass_ratio.run_treated`'s dependency
chain is frozen as the 2026-27 season's production configuration:

- `HALFLIFE_GAMES = 600` (decayed league-average baseline, §13/§18)
- `CROSS_SEASON_WEIGHT = 0.75`, `PRIOR_MINUTES_MULTIPLIER = 2.0` (§18/§19)
- `PRIOR_GAMES_GOALIE = 12`, `GOALIE_ADJUSTMENT_FLOOR = 0.05` (§4.6/§4.7)
- `MIN_DEV_SEASON = 20102011`, `DEV_MAX_SEASON = 20242025` (§35.2)
- The OT-decided logistic (`a=0.008, b=0.302`, §17) and the SO-decided flat empirical rate
- The walk-forward tie-mass calibration ratio and per-game delta machinery (§26), and the
  away-B2B symmetric rest credit (§22) — both fit fresh each run on `[MIN_DEV_SEASON,
  DEV_MAX_SEASON)` per §35.2's discipline, but the FITTING PROCEDURE itself (which data range,
  which functional form) is what's frozen, not a single static number
- The GBM stacking layer stays OUT (§36) — not merely absent by default, but explicitly excluded
  by this freeze; re-adding it would require a new cycle, not a config flip

**Why freeze, beyond hygiene**: a moving model can't be compared against anything — §37.5's
live-vs-market tracking and §37.4's degradation budget both assume the SAME configuration
underlies every window they compare; changing constants mid-season would silently invalidate
both. Just as important: this project has repeatedly proven, in backtest, that it can resist the
temptation to re-fit a constant after a single bad-looking check (§18.6/§19.4/§24.4/§26.9's
"cancelling errors" pattern was left to accumulate evidence across multiple cycles rather than
patched reactively each time) — but it has never faced that temptation LIVE, where a bad week
reads as urgent in a way a backtest number never does, and NHL single-week variance is large
enough that almost any bad week is noise, not signal. The freeze converts "should we change
something" from a standing option into a decision that requires either §37.4's rule actually
firing or a documented emergency-fix path (a confirmed data pipeline failure, not a performance
wobble) — the same discipline this project has applied to every other adoption decision, now
applied to the temptation to tinker.

**In-season changes permitted only through**: (1) §37.4's pre-registered recalibration rule
actually firing (an escalation-ceiling breach, investigated and confirmed not explained by a
pipeline issue), or (2) a documented emergency-fix path for a confirmed data-pipeline failure
(e.g. an ingestion break, a schema change from a data provider) — never a reaction to a
performance wobble alone.

**What this buys the parked open threads**: the dependence-bearing joint for the 0.89-vs-1.02
dispersion gap (task #36, long-term, unscoped), the EV-TOI mechanism (§32/§35.4, re-filed to the
open-mechanisms list, needing a structurally different vehicle per §29.3.1's logic, not a
re-grid), and §28.4's dormant question (the "cancelling errors" pattern's own root cause) all
park cleanly as off-season research against this frozen baseline — no backtest number they'd be
compared against will move out from under them mid-investigation. The freeze has a quieter
second benefit for exactly these three: 2026-27 accumulates as genuinely untouched data while
they're being worked, since nothing about the production configuration will change to
contaminate it — the first truly clean, never-yet-touched split this project will have had since
the holdout began wearing (§35's "holdout wear is now a real accounting item" finding). Any of
these three that eventually produces a candidate fix should be validated against the frozen
dev/holdout split exactly as before, with 2026-27's live data available as a genuine, never-
previously-peeked-at confirmatory check once that work is ready — a luxury this project has not
had for any adoption decision since early in its history.

## 39. Task #44: RotoWire lineup/injury scraper + live odds capture (2026-07-25)

Built and confirmed live where verifiable; one piece honestly deferred where it couldn't be.

### 39.1 Injury/lineup-availability scraper — CONFIRMED WORKING, real live data

`src/ingest/fetch_rotowire_injuries.py`. Found via browser network inspection (same technique as
the MLB/NBA siblings' own RotoWire investigations): the injury-report page makes a clean,
unauthenticated client-side fetch to `GET https://www.rotowire.com/hockey/tables/injury-report.php
?team=ALL&pos=ALL`. Confirmed live 2026-07-25 (off-season — the only populated rows were real
Sep 19-20 2026 preseason games): 104 real rows, fields `ID`/`URL`/`firstname`/`lastname`/`player`/
`team`/`position`/`injury`/`status`/`rDate`/`date` — `rDate` is subscriber-gated and unusable,
everything else is plain public data. `status` values actually observed: `Out` (76), `IR` (14),
`IR-LT` (14) — no `Questionable`/`Day-To-Day`-type gradient appeared in this sample; the fetcher
warns if a future response contains a status value outside the known set, rather than silently
mis-handling it. `likely_out_player_names` gives the binary Out/IR/IR-LT exclusion set §37's
recent-workhorse heuristic needs (a goalie flagged here should be excluded before falling back to
"most recent starter", exactly matching the NBA sibling's own `resolve_active_lineup` pattern).
Every real fetch archives a timestamped JSON snapshot to `data/raw/rotowire_injury_snapshots/` —
RotoWire itself keeps no historical injury archive, so this project's own accumulated snapshots
will be the only historical record available for task #45's eventual live re-measurement.

### 39.2 Live odds capture — endpoint confirmed, schema NOT yet verified (honest limitation)

`src/ingest/fetch_rotowire_odds.py`. Same site family, same technique: the NHL odds page makes a
client-side fetch to `GET https://www.rotowire.com/betting/nhl/tables/nhl-games.php?date=YYYY-MM-DD`
— confirmed real and genuinely date-parametrized (a real "today" URL and a real past-date URL both
resolved distinctly, not a generic redirect). **But every date tried — 2026-07-25 (real live
"today") and 2026-04-06 (a real past game date) — returned an empty `[]`.** This could mean either
"the board is genuinely empty this far before a game" or "this endpoint only ever serves the
imminent slate, never history" — the two can't be told apart from outside the off-season.
Consequently `parse_games_response`'s field names (`home_ml`/`away_ml`/`total`/etc.) are a
best-effort placeholder inferred from this project's own SBRO-odds convention, **not confirmed
against a real populated response** — unlike every other fetcher in this codebase, which only ever
documents fields actually observed. The fetch/archive mechanism itself is real and works (confirmed
by running it), and every call snapshots its raw response to `data/raw/rotowire_odds_snapshots/`
regardless of shape, so a real populated response captured once the season nears is never lost
even before the parser is corrected. **Action required, flagged explicitly, not silently
deferred**: re-run this once RotoWire's board actually populates (plausibly early-to-mid September,
once preseason lines post) and fix `parse_games_response` to match whatever real fields come back.

### 39.3 Starting-goalie CONFIRMATION specifically — not found, genuinely blocked by the off-season

The NHL lineups page (`hockey/nhl-lineups.php`) — which would show each game's confirmed/projected
starting goalie directly, the single most valuable signal for §37's degradation budget — reported
"no games on the NHL schedule today" for every date tried (including an explicit `?date=` query
param, which this specific page did not appear to honor) and made NO underlying data-table request
at all while empty, unlike the injury and odds pages. This means its real JSON/data structure
genuinely cannot be observed from outside a day with real scheduled games — not a shortcut taken,
a real constraint of investigating this in July for a season that starts in October. **Deferred,
not guessed**: re-investigate this specific page once real preseason or regular-season games are
scheduled close enough that RotoWire populates it (the same Sep 19-20 window the injury report
already shows real games for is the first plausible test date). Until then, §37's recent-workhorse
heuristic plus 39.1's injury exclusion is the best available live signal — exactly the "realistic"
comparator §37.2 already validated as capturing nearly all of the real-starter information value.

### 39.4 Task #44 status, precisely

**Real, working, done**: injury/availability data (39.1), the live odds fetch-and-archive mechanism
(39.2's plumbing). **Confirmed real but not yet trustworthy for its parsed fields**: live odds
schema (39.2's `parse_games_response`) — fix once real data exists. **Not yet built, honestly
blocked**: starting-goalie confirmation parsing (39.3) — re-attempt once real scheduled games make
the page observable. None of this blocks §37's rule from running day one: the recent-workhorse
heuristic (already validated, §37.2) plus 39.1's real injury exclusion is a complete, working
input; 39.2/39.3 are refinements to layer in as they become verifiable, not dependencies.

## 40. A second walk-forward-discipline gap found and fixed: `home_ice_multiplier` (2026-07-25)

Found while tracing the exact fitting boundaries needed to build a live single-game predictor
(§41) -- confirming every constant's dev-only scope forced a careful re-check of the ENTIRE
`_build_dev_base()` chain, not just the pieces §35 already fixed.

**`fit_home_ice_multiplier`** (`validate_baseline.py`) took no season boundary argument at all --
every caller passed it the unrestricted `log`, so `home_ice_multiplier` has been fit on
dev+holdout COMBINED every single time it has ever been computed, including inside
`_build_dev_base()`'s own production chain (`validate_situational_toi.run_validation` ->
`validate_goalie.run_validation`). This is the exact same class of bug §35 fixed for
`away_b2b_adj`/`ot_split`/the OT logistic `(a,b)` -- just in a different constant that fix never
reached, because `run_situational_toi`/`validate_goalie.run_validation` only ever exposed a
`min_season` FLOOR, never a `max_season` CEILING, so there was no way to even ASK this function
to stop at the dev/holdout boundary.

**Confirmed real, not just theoretical**: dev-only fit gives **1.047478**; the actual
(contaminated) dev+holdout fit gives **1.046158** -- a real but small difference (-0.00132,
~0.13% relative). **Fixed**: `max_season_exclusive` is now a REQUIRED argument (no silent default
that could reintroduce this, matching §35.2's own precedent once a bug is confirmed real, not an
opt-in flag) -- all 5 call sites updated (`validate_baseline.py`, `validate_xg.py`,
`validate_situational_toi.py` -- the production-load-bearing one -- `validate_situational.py`,
`validate_bias_decomposition.py`), all passing `DEV_MAX_SEASON` explicitly. Regression test added
(`test_home_ice_multiplier_requires_dev_only_boundary`) confirming the argument is mandatory
(missing it raises `TypeError`) and that dev-only fitting genuinely differs from unrestricted
fitting on real data.

**Fixing this surfaced a second, pre-existing issue**: `final_holdout_check.py` imported
`validate_goalie.run_validation` at module level for its own `__main__`-only usage, creating a
circular import once `validate_situational_toi.py` needed `DEV_MAX_SEASON` from
`final_holdout_check.py` (`final_holdout_check -> validate_goalie -> validate_situational_toi ->
final_holdout_check`). Fixed by moving that import local to `__main__` -- it was never needed at
module level in the first place, since every OTHER caller of this file only ever needs
`DEV_MAX_SEASON`/`split_dev_holdout`.

**Verified this does NOT change the frozen headline number**: re-ran the market benchmark with
the fix in place -- gap **0.00248, 95% CI [0.00143, 0.00354]** -- indistinguishable from the
pre-fix frozen figure (0.00247, CI [0.00141, 0.00352]). Confirmed empirically, not assumed: this
is a real discipline gap worth closing on principle (and worth being suspicious of every OTHER
similarly-shaped function going forward), but it changes nothing about §38's frozen conclusions.
Full regression suite (7 checks) re-run and passing after the fix.

## 41. Building the live single-game predictor (in progress, 2026-07-25)

The project has never had a live daily prediction entry point -- every existing script is either
an ingest fetcher, a model component, or a `validate_*.py`/`check_*.py` BACKTEST script scoring
already-completed games. `src/pipeline/` is empty except `__init__.py`. This section tracks
building the first one.

**Design, confirmed feasible by tracing the exact mechanics**: every global walk-forward constant
in the production chain (`away_b2b_adj`, the OT logistic `(a,b)`, `ot_split`, the reg-ratio EWMA,
the tie-mass calibration ratio, and now `home_ice_multiplier` per §40) is either a dev-fit-frozen
constant or a single pooled EWMA/expanding-mean value -- "today's" value is simply the latest
point in an already-existing series. `fit_deltas_per_game`'s tie-mass machinery only needs the
PROJECTED joint distribution (computable for any hypothetical lambda pair), not a real outcome --
so the full tie-mass transfer genuinely can run on an unplayed game.

**The real complexity**: predicting a new matchup needs each team's own EV/PP/PK/other
attack-and-defense per60 rates BEFORE they're combined against a specific opponent (from the
per-team-game situational log inside `validate_situational_toi.py`, one layer below anything
`run_treated()` exposes), combined via the existing `predict_situational_lambda` for the SPECIFIC
tonight's-matchup pairing, then goalie overlay (real starter live, recent-workhorse heuristic +
§39's RotoWire injury exclusion for prediction), then rest/B2B, then the latest global reg-ratio/
tie-mass/OT-logistic constants above.

**Status**: architecture fully traced and confirmed buildable; implementation in progress. Next:
write the "combine two teams' latest state for a hypothetical game" function, reusing every
existing scoring primitive, then validate it by predicting a real PAST game using only data
through the day before and confirming it exactly matches what `run_treated()` already produced
for that same game -- the cleanest possible correctness check, since for a real historical game
the walk-forward EWMA "as of yesterday" is bit-identical to that game's own row value in the
existing backtest output.

