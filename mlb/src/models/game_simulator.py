"""Chain the true-talent engine + odds-ratio matchup combination + empirical
base-out transition table into an actual Monte Carlo game simulator.

Deliberate FIRST-VERSION simplifications, clearly flagged rather than hidden:
  - A single pitcher is used unless a `bullpen` mapping ({inning: profile})
    is supplied (see GameSimulator._pitcher_for_inning) -- this class itself
    has no opinion on WHERE that mapping comes from; validate_game_simulator.py
    builds it from real historical per-inning pitcher-usage data for
    backtesting, but a live/future-game caller would need a genuinely
    predictive "expected bullpen usage" model instead.
  - The 9 non-PRIMARY, rare outcome categories (sac_fly, sac_bunt,
    field_error, fielders_choice, catcher_interf, triple_play) are included in
    the SAME odds-ratio + renormalization mechanism as everything else, not
    hard-coded to a fixed rate -- this works because their heavy shrinkage
    (large STABILIZATION_PA in true_talent.py) already pulls them close to
    league average for virtually every player, so no special-casing is
    needed for "don't let a rare event overfit to one player."
"""

import numpy as np
import pandas as pd

from src.models.base_out_transitions import TransitionTable
from src.models.true_talent import STABILIZATION_PA, build_pregame_rates, sample_posterior_rate
from src.models.matchup import odds, prob_from_odds
from src.models.platoon_splits import build_platoon_multipliers
from src.models.crn import (
    crn_bool, crn_choice, crn_normal, crn_uniform, half_inning_ordinal,
    DECISION_OUTCOME, DECISION_SB_ATTEMPT, DECISION_SB_SUCCESS, DECISION_TRANSITION,
    DECISION_PITCHER_SHOCK, DECISION_HOOK_FRAILTY, DECISION_TIER_SELECT,
)
# Pure hook-frailty/tier-selection math only (task #144/#145) -- deliberately from
# these leaf modules, NOT bullpen.py/bullpen_usage_policy.py, to avoid a circular
# import: bullpen_usage_policy.py -> bullpen.py -> game_simulator.py (for OUTCOMES).
# See hook_frailty.py's/tier_selection.py's own docstrings.
from src.models.hook_frailty import (
    apply_hook_frailty, bucket_inning, bucket_pa_count, bucket_runs_allowed, bucket_margin,
)
from src.models.tier_selection import sample_tier_conditioned_reliever
from src.utils.paths import DATA_PROCESSED

OUTCOMES = list(STABILIZATION_PA.keys())

# Task #137 (Phase 1, the latent per-pitcher-appearance "stuff" shock): the
# on-base-allowed categories a pitcher's day-to-day effectiveness swing
# plausibly affects -- deliberately excludes intent_walk (a strategic
# decision, not a "stuff" effect) and every out/situational category.
SHOCKED_CATEGORIES = {"walk", "hit_by_pitch", "single", "double", "triple", "home_run"}


def _build_shock_renorm_table(sigma: float, n_s: int = 99, n_gh: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Precomputed correction c(s) making the pitcher shock mean-preserving
    THROUGH combine_matchup_distribution's renormalization (2026-08-03
    audit, finding M6). The old exp(sigma^2/2) division made E[mult]=1 only
    BEFORE the renorm; post-renorm, a shocked category's probability is
    p*m / (s*m + 1-s) where s is the shocked-mass share -- concave in m
    (Jensen), so shocked (on-base) categories were systematically
    SUPPRESSED: measured -3.34% at sigma=0.40, a hidden ~-0.23 runs/team/
    game mean shift, and the frozen sigma itself was selected by an A/B
    whose arms differed in mean, not just variance.

    For each s on a grid, solves (Gauss-Hermite over g ~ N(0, sigma^2),
    bisection over c) for the c satisfying
        E_g[ (e^g / c) / (s * e^g / c + 1 - s) ] = 1,
    i.e. the post-renorm expected multiplier on shocked mass is exactly 1.
    Limits check out analytically: s->0 gives c = E[e^g] = exp(sigma^2/2)
    (the old pre-renorm correction), s->1 gives 1 for any c (renorm kills
    the shock entirely). Applied per PA via linear interpolation on the
    PA's own pre-shock shocked-mass share -- the same drawn g serves every
    PA of the appearance, only the correction adapts to context."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_gh)
    g = np.sqrt(2.0) * sigma * nodes
    w = weights / np.sqrt(np.pi)
    m = np.exp(g)
    s_grid = np.linspace(0.0, 0.99, n_s)
    c_grid = np.empty(n_s)
    for i, s in enumerate(s_grid):
        if s <= 0.0:
            c_grid[i] = float(np.sum(w * m))  # = exp(sigma^2/2)
            continue
        lo, hi = 1e-6, float(np.sum(w * m)) * 4.0
        for _ in range(80):
            c = 0.5 * (lo + hi)
            val = float(np.sum(w * (m / c) / (s * (m / c) + 1.0 - s)))
            if val > 1.0:
                lo = c
            else:
                hi = c
        c_grid[i] = 0.5 * (lo + hi)
    return s_grid, c_grid


def build_wide_pregame_rates(pa: pd.DataFrame, player_col: str,
                              park_factors_df: pd.DataFrame | None = None,
                              use_debut_prior: bool = False) -> pd.DataFrame:
    """One row per PA (game_pk, at_bat_number): a pregame_rate_{outcome}
    column for every outcome category, for whichever player role is asked for.

    park_factors_df: pass park_factors.build_outcome_park_factors(pa)'s
    output here to park-neutralize these rates before this project's own
    per-game park_factors multiplier (applied later, at simulate time) gets
    to act on them -- otherwise that multiplier double-counts a park effect
    already baked into the raw rate. See true_talent.build_pregame_rates.

    use_debut_prior: see true_talent.build_pregame_rates/build_debut_rate
    (task #119) -- defaults False (old behavior, unqualified league-average
    cold-start fallback) until a full-stack A/B confirms the debut-cohort
    prior is a real net win.

    Also carries an effective_n_{outcome} column alongside each
    pregame_rate_{outcome} (task #127, posterior-sampled rates) -- the
    Bayesian-blend denominator backing that rate, needed to sample from its
    implied Beta posterior instead of always using the point-estimate mean.
    Purely additive; existing callers that only read pregame_rate_* columns
    are unaffected."""
    base_cols = ["game_pk", "at_bat_number", "season"]
    result = None
    for outcome in OUTCOMES:
        r = build_pregame_rates(pa, player_col, outcome, park_factors_df, use_debut_prior)[
            base_cols + [player_col, "pregame_rate", "effective_n"]].rename(
            columns={"pregame_rate": f"pregame_rate_{outcome}", "effective_n": f"effective_n_{outcome}"}
        )
        result = r if result is None else result.merge(r, on=base_cols + [player_col], how="inner")
    return result


def resample_profile_rates(profile: dict, rng: np.random.Generator) -> dict:
    """Task #127 (posterior-sampled rates, the correctness fix for the
    diagnosed total-runs under-dispersion -- see true_talent.sample_posterior_rate
    for the full rationale): returns a NEW profile dict with `rates` replaced
    by ONE fresh Beta-posterior draw per outcome category, using the SAME
    `effective_n` every existing profile now carries (see build_profile).
    Every other key (hand, same_mult, opp_mult, pull_tercile, effective_n) is
    passed through unchanged. Call this ONCE PER TRIAL, not once per PA or
    per half-inning -- the whole point is that a player's TRUE talent is one
    unknown quantity for the whole game in that parallel-universe trial, not
    independent noise on every at-bat (which the existing PA-outcome sampling
    already provides via a completely different, already-validated mechanism)."""
    sampled_rates = {
        o: sample_posterior_rate(profile["rates"][o], profile["effective_n"][o], rng)
        for o in profile["rates"]
    }
    return {**profile, "rates": sampled_rates}


def build_wide_platoon_multipliers(pa: pd.DataFrame, player_col: str) -> pd.DataFrame:
    """One row per (player, season): a same_hand_mult_{outcome} and
    opp_hand_mult_{outcome} column for every outcome category."""
    base_cols = [player_col, "season"]
    result = None
    for outcome in OUTCOMES:
        r = build_platoon_multipliers(pa, player_col, outcome).rename(
            columns={"same_hand_mult": f"same_hand_mult_{outcome}", "opp_hand_mult": f"opp_hand_mult_{outcome}"}
        )
        result = r if result is None else result.merge(r, on=base_cols, how="outer")
    # players with no usable split history (e.g. rookies) get a neutral 1.0x multiplier --
    # their overall true-talent rate (from build_wide_pregame_rates) already reflects their
    # only available signal; no platoon adjustment is applied on top rather than guessing.
    return result.fillna(1.0)


def build_league_rates_by_season(pa: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Walk-forward-safe (prior-seasons-only) league rate per outcome, per season."""
    out = {}
    for season in sorted(pa["season"].unique()):
        prior = pa[pa["season"] < season]
        ref = prior if len(prior) else pa[pa["season"] == season]
        out[season] = {o: (ref["outcome"] == o).mean() for o in OUTCOMES}
    return out


STATE_FACTOR_PRIOR_PA = 5000  # same magnitude/role as park_factors.py's and weather.py's own priors


def _mechanically_impossible(state: int, outcome: str) -> bool:
    """Rule-level impossibilities (2026-08-03 audit, finding M14): the
    Bayesian floor below otherwise keeps every outcome drawable at
    ~STATE_FACTOR_PRIOR_PA/(n+PRIOR) x league rate even where the scoring
    rules make it impossible (a 2-out sacrifice can't exist -- the out
    would end the inning before the advance counts; a triple play needs 0
    outs and 2+ runners; sacrifice/DP/FC/TP all need a runner). Those
    impossible draws then fell through the transition table to its
    outcome-only last-resort tier, whose pooled rows come from 0/1-out
    contexts -- OUTS WENT BACKWARDS ~once per 26 simulated games,
    un-ending half-innings and inflating scoring. Every rule below is
    verified against all 676,748 real PAs: zero real occurrences of any
    flagged (state, outcome) combination."""
    bm, outs = state // 10, state % 10
    n_runners = bin(bm).count("1")
    needs_runner = {"sac_fly", "sac_bunt", "fielders_choice", "double_play", "triple_play"}
    if outcome in needs_runner and n_runners == 0:
        return True
    if outcome in {"sac_fly", "sac_bunt", "double_play", "triple_play"} and outs == 2:
        return True
    if outcome == "triple_play" and (outs != 0 or n_runners < 2):
        return True
    return False


def build_state_factors_by_season(pa: pd.DataFrame) -> dict[int, dict[int, dict[str, float]]]:
    """Walk-forward-safe (prior-seasons-only) state_factor[state][outcome] =
    rate(outcome | state) / rate(outcome | overall). Confirmed necessary on
    real data: e.g. sac_fly is mechanically ~0% with bases empty but ~13% with
    a runner on 3rd/0 outs -- combine_matchup_distribution alone (batter x
    pitcher x league, no state) can't capture this, and ignoring it was
    confirmed (via a league-average-only simulation) to inflate simulated
    scoring by about +0.48 runs/team/game vs. real 2025 data. A plain ratio
    (not odds-ratio) is used deliberately: several categories are exactly
    zero in some states (sac_fly/sac_bunt/fielders_choice/double_play all
    require specific baserunners), and odds(0) is undefined.

    Bayesian-shrunk (added 2026-07-21, matching park_factors.py's and
    weather.py's own fix for the identical failure mode) -- an audit found
    this table was the one "environmental multiplier" left as a completely
    unshrunk raw ratio, and confirmed it live on real data: `triple_play` at
    one real (state) had a factor of 56.6x from just 5 historical events,
    which combined with a real park-factor extreme (28.4x) compounds to
    ~1600x for that rare category in that specific situation -- exactly the
    kind of small-sample blowup this project has fixed everywhere else.

    Cold-start fix (2026-08-03, follow-up to the task #160 correctness audit
    -- this module was the one place that same "prior if available else
    THIS season's own data" leak pattern survived, missed when the audit
    fixed the identical bug in catcher_framing.py/weather.py/umpire_factor.py/
    ttop.py). Unlike those, this table has no caller-side "missing key falls
    back to neutral" convention (GameSimulator does `state_factors[state]`
    directly -- see combine_matchup_distribution), so an empty dict for the
    true cold-start season would KeyError instead of silently leaking. Fixed
    the correct way instead: every one of the 24 mechanically-possible
    (bases-bitmask x outs) states gets an explicit, fully-neutral 1.0 cell
    for that season -- the same "neutral by construction" resolution
    park_factors.py already uses for its own true-first-season case, just
    applied to a full lookup table instead of a lookup-with-default."""
    out = {}
    for season in sorted(pa["season"].unique()):
        prior = pa[pa["season"] < season]
        if prior.empty:
            out[season] = {
                bitmask * 10 + outs: {
                    o: (0.0 if _mechanically_impossible(bitmask * 10 + outs, o) else 1.0)
                    for o in OUTCOMES
                }
                for bitmask in range(8) for outs in range(3)
            }
            continue
        ref = prior
        overall = ref["outcome"].value_counts(normalize=True)
        factors = {}
        for state, g in ref.groupby("pre_state"):
            state_counts = g["outcome"].value_counts()
            n_state = len(g)
            cell = {}
            for o in OUTCOMES:
                if _mechanically_impossible(int(state), o):
                    # hard zero, not a Bayesian floor -- see
                    # _mechanically_impossible (finding M14)
                    cell[o] = 0.0
                    continue
                overall_rate = overall.get(o, 1e-9)
                shrunk_rate = (
                    (state_counts.get(o, 0) + STATE_FACTOR_PRIOR_PA * overall_rate)
                    / (n_state + STATE_FACTOR_PRIOR_PA)
                )
                cell[o] = shrunk_rate / overall_rate if overall_rate > 1e-9 else 1.0
            factors[int(state)] = cell
        out[season] = factors
    return out


def combine_matchup_distribution(batter_rates: dict[str, float], pitcher_rates: dict[str, float],
                                  league_rates: dict[str, float],
                                  state_factors: dict[str, float] | None = None,
                                  batter_platoon_mult: dict[str, float] | None = None,
                                  pitcher_platoon_mult: dict[str, float] | None = None,
                                  park_factors: dict[str, float] | None = None,
                                  weather_factors: dict[str, float] | None = None,
                                  ttop_factors: dict[str, float] | None = None,
                                  catcher_factors: dict[str, float] | None = None,
                                  hfa_factors: dict[str, float] | None = None,
                                  pitcher_shock_factors: dict[str, float] | None = None) -> dict[str, float]:
    """Odds-ratio-combine every outcome category independently vs. league
    average, then apply the current base-out state's multiplicative adjustment (mechanical/
    situational effects -- see build_state_factors_by_season), the batter's/
    pitcher's platoon-split multiplier for THIS specific same-hand/opposite-
    hand matchup (see platoon_splits.py -- already resolved to the right
    direction by the caller before being passed in here), the park's
    per-outcome factor (see park_factors.py -- a property of the stadium,
    applies identically to both teams batting there), the weather's
    per-outcome factor for THIS batter's actual side (see weather.py --
    already resolved to the right stand by the caller), the times-through-
    the-order penalty factor for how many times THIS batter has already
    faced the CURRENT pitcher this game (see ttop.py -- resolved by the
    caller below), the DEFENSIVE team's catcher framing factor (see
    catcher_framing.py -- only strikeout/walk are non-1.0; a property of
    whichever catcher is behind the plate, applies identically regardless of
    which batter is up, same as park_factors), and the league-wide home-
    field-advantage factor (see park_factors.build_hfa_factors) -- unlike
    every other factor here, this is NOT symmetric: the caller only ever
    passes hfa_factors for the HOME team's own at-bats (see
    simulate_half_inning/simulate_game), never the visiting team's, since
    it's specifically the residual home edge that build_outcome_park_factors'
    own per-team normalization step cancels out -- then renormalize the full
    vector to sum to 1 (mutually exclusive categories) -- the natural
    multi-category extension of the (inherently binary) odds-ratio method.
    pitcher_shock_factors: optional (task #137) -- this pitcher's CURRENT
    appearance's latent "stuff" shock as {"mult": e^g, "s_grid", "c_grid"}
    (see GameSimulator._draw_pitcher_shock / _build_shock_renorm_table).
    Unlike the other factor dicts, applied AFTER the deterministic factors
    with a per-PA renorm-aware correction so it is mean-preserving THROUGH
    the final renormalization (finding M6), not just before it."""
    unnorm = {}
    for o in OUTCOMES:
        lg, b, p = league_rates[o], batter_rates[o], pitcher_rates[o]
        matchup_odds = odds(lg) * (odds(b) / odds(lg)) * (odds(p) / odds(lg))
        prob = prob_from_odds(matchup_odds)
        if state_factors is not None:
            prob *= state_factors[o]
        if batter_platoon_mult is not None:
            prob *= batter_platoon_mult[o]
        if pitcher_platoon_mult is not None:
            prob *= pitcher_platoon_mult[o]
        if park_factors is not None:
            prob *= park_factors[o]
        if weather_factors is not None:
            prob *= weather_factors[o]
        if ttop_factors is not None:
            prob *= ttop_factors[o]
        if catcher_factors is not None and o in catcher_factors:
            prob *= catcher_factors[o]
        if hfa_factors is not None:
            prob *= hfa_factors[o]
        unnorm[o] = prob
    if pitcher_shock_factors is not None:
        # RENORM-AWARE shock application (2026-08-03 audit, finding M6):
        # the shock is applied AFTER every deterministic factor, scaled by
        # a correction interpolated at THIS PA's own shocked-mass share s,
        # so that E_g[renormalized shocked probability] equals the no-shock
        # probability exactly (see _build_shock_renorm_table). The old
        # exp(sigma^2/2) division was mean-preserving only pre-renorm; the
        # renorm map is concave in the multiplier (Jensen), which silently
        # suppressed on-base categories by -3.34% at sigma=0.40.
        total = sum(unnorm.values())
        s = sum(unnorm[o] for o in SHOCKED_CATEGORIES) / total
        c = float(np.interp(s, pitcher_shock_factors["s_grid"], pitcher_shock_factors["c_grid"]))
        mult = pitcher_shock_factors["mult"] / c
        for o in SHOCKED_CATEGORIES:
            unnorm[o] *= mult
    total = sum(unnorm.values())
    return {o: v / total for o, v in unnorm.items()}


# Which base the BATTER occupies after their own PA, for the outcome
# categories where they reach one (see _reassign_runners) -- everything not
# listed here (strikeout, home_run, sac_fly, sac_bunt, double_play,
# triple_play, field_out) means the batter does NOT occupy a base afterward
# (either an out, or a home run scoring immediately).
BATTER_REACHES_BASE = {
    "single": 1, "double": 2, "triple": 3,
    "walk": 1, "intent_walk": 1, "hit_by_pitch": 1,
    "field_error": 1, "fielders_choice": 1, "catcher_interf": 1,
}


BATTER_SCORES = {"home_run"}  # the only outcome where the batter's own PA is an immediate score, not a base


def _reassign_runners(pre_runners: dict[int, int], outcome: str, post_bases_bitmask: int,
                       runs_this_pa: int, batter_idx: int) -> dict[int, int]:
    """Approximates WHICH lineup slot ends up on which base after a PA, given
    only the pre-PA runner identities plus this PA's outcome and the REAL
    historical post-state bitmask/runs sampled from TransitionTable (which
    doesn't itself track individual identity -- see base_out_transitions.py).
    Uses the standard "runners can't pass each other, lead runner scores
    first" baseball convention. Not perfectly exact on rare misdirection
    plays (e.g. a trailing runner thrown out at home while the lead runner
    holds), but correct in the overwhelming majority of real cases -- which
    is what matters here: giving each stolen-base OPPORTUNITY to the right
    lineup slot, not perfectly reconstructing every play's exact mechanics.

    Every pre-existing runner ends this PA in exactly one of three states:
    scored, still on base, or retired on the bases WITHOUT scoring (a force
    out, a runner thrown out taking an extra base, etc. -- a real and common
    category since TransitionTable's post_state/runs pairs are resampled
    directly from real historical plays, not synthesized). An EARLIER version
    of this function only modeled the first two, which meant "still on base"
    silently absorbed anyone actually retired -- overfilling the post-PA base
    slots and starving the BATTER of a slot whenever an existing runner was
    put out without scoring (a common play: any single/double/fielder's-
    choice with a runner thrown out advancing). Fixed via headcount
    conservation: the retired count is exactly `n_pre - scored - still_on`,
    computed from the real bitmask/runs already sampled, rather than left
    implicit. When more than one non-scoring runner remains uncertain
    (retired vs. still on), the TRAILING-most are assumed retired first
    (real force/FC outs overwhelmingly retire the runner closest to the
    batter, not the lead runner) -- an approximation only when that specific
    ambiguity arises, not a guess about whether anyone was retired at all."""
    order = (3, 2, 1)
    existing = [(b, pre_runners[b]) for b in order if pre_runners.get(b) is not None]
    n_pre = len(existing)

    post_occupied = [b for b in order if post_bases_bitmask & (1 << (b - 1))]
    batter_base = BATTER_REACHES_BASE.get(outcome)
    batter_scores = outcome in BATTER_SCORES

    existing_scored = max(0, min(n_pre, runs_this_pa - (1 if batter_scores else 0)))
    after_scoring = existing[existing_scored:]  # still lead-to-trail order
    existing_still_on = max(0, min(len(after_scoring), len(post_occupied) - (1 if batter_base is not None else 0)))
    still_on = after_scoring[:existing_still_on]  # lead-most of what's left survive; trailing-most are retired

    new_runners: dict[int, int] = {}
    slots = list(post_occupied)
    for _, pid in still_on:
        if not slots:
            break
        new_runners[slots.pop(0)] = pid
    if slots and batter_base is not None:
        if batter_base in slots:
            new_runners[batter_base] = batter_idx
        else:
            new_runners[slots[0]] = batter_idx
    return new_runners


LATE_INNING_ANCHOR = 8  # innings >= this stay role-anchored through a hook shift (setup 8th, closer 9th)


def _shift_bullpen_after_hook(original_bullpen: dict[int, dict] | None, cutoff: int, hook_inning: int) -> dict[int, dict] | None:
    """Task #144/#145: one-time rebuild of a pregame {inning: profile} bullpen
    plan (see bullpen.sample_bullpen_plan) after a state-conditioned starter
    hook fires at `hook_inning` instead of the plan's assumed `cutoff`.

    MIDDLE innings (< LATE_INNING_ANCHOR) shift by (hook_inning - cutoff):
    an early hook pulls the same middle arms in sooner, a late hook pushes
    them later -- identity and order untouched, only timing.

    LATE innings (>= LATE_INNING_ANCHOR) stay ANCHORED at their planned
    inning (2026-08-03 audit, finding M16): the plan's inning-9 arm is the
    closer, sampled specifically for that role (see sample_bullpen_plan's
    CLOSER_INNING9_RATE logic, calibrated against the real "closer takes
    the 9th" rate), and inning 8 is likewise a late-leverage arm. Real
    managers absorb a starter's early exit with MIDDLE relief and still
    hand the 8th/9th to the same late arms -- the old uniform shift instead
    moved the closer to e.g. the 7th on an early hook (where the
    _pitcher_for_inning fallback then stretched him 3 innings) or to a
    nonexistent inning 11 on a late one (a mop-up arm taking the real 9th),
    silently destroying the closer calibration whenever the stochastic hook
    missed the cutoff exactly. A middle arm whose shifted inning would
    collide with the anchored late innings is dropped -- the starter (or
    the stretch of the arm before it) ate its innings. Gap innings an early
    hook opens up before the anchor are covered by _pitcher_for_inning's
    existing most-recent-arm fallback. Called ONCE per game per side,
    immediately after the hook fires -- see simulate_game."""
    if original_bullpen is None:
        return None
    shifted = {
        hook_inning + (orig_inning - cutoff): profile
        for orig_inning, profile in original_bullpen.items()
        if cutoff < orig_inning < LATE_INNING_ANCHOR
        and hook_inning + (orig_inning - cutoff) < LATE_INNING_ANCHOR
    }
    # anchored late entries are kept even when hook_inning >= LATE_INNING_ANCHOR
    # (they're only ever consulted for innings AFTER the hook, and dropping
    # them would make _pitcher_for_inning's empty-dict fallback hand extra
    # innings back to the STARTER -- an illegal re-entry)
    shifted.update({
        orig_inning: profile
        for orig_inning, profile in original_bullpen.items()
        if orig_inning >= max(LATE_INNING_ANCHOR, cutoff + 1)
    })
    return shifted


class GameSimulator:
    def __init__(self, transitions: TransitionTable, league_rates: dict[str, float], rng: np.random.Generator,
                 state_factors: dict[int, dict[str, float]] | None = None,
                 ttop_factors: dict[int, dict[str, float]] | None = None,
                 shock_sigma: float = 0.0):
        self.transitions = transitions
        self.league_rates = league_rates
        self.rng = rng
        self.state_factors = state_factors  # {state: {outcome: factor}}, see build_state_factors_by_season
        self.ttop_factors = ttop_factors  # {times_through(capped at 3): {outcome: factor}}, see ttop.py
        # Task #137 (Phase 1, latent per-pitcher-appearance shock): standard
        # deviation of the mean-preserving lognormal "stuff" shock drawn once
        # per pitcher-appearance per trial (see _draw_pitcher_shock,
        # _pitcher_shock_factors). Default 0.0 is an exact no-op -- confirmed
        # byte-identical to the pre-task-#137 codebase (no shock draw of any
        # kind happens, so the plain-rng sequential stream is untouched).
        self.shock_sigma = shock_sigma
        # finding M6 (renorm-aware mean preservation): precomputed once per
        # simulator -- see _build_shock_renorm_table. None when the shock is
        # off (exact no-op path preserved).
        if shock_sigma > 0:
            self._shock_s_grid, self._shock_c_grid = _build_shock_renorm_table(shock_sigma)
        else:
            self._shock_s_grid = self._shock_c_grid = None

    def _draw_pitcher_shock(self, crn_game_pk: int | None, crn_trial: int | None,
                             side_tag: int, appearance_idx: int) -> dict[str, float] | None:
        """Draws ONE pitcher-appearance's latent "stuff" shock g ~ N(0,
        shock_sigma^2) and returns the resulting per-outcome multiplier dict
        (see _pitcher_shock_factors), or None if shock_sigma<=0 (the
        byte-identical no-op path -- no random draw of any kind happens).

        side_tag: 0 for the home team's current pitcher, 1 for the away
        team's -- combined with game_pk/trial/appearance_idx this uniquely
        and reproducibly identifies one specific real pitcher's specific real
        appearance within one specific simulated trial, without needing to
        thread real MLB player IDs into GameSimulator (side+appearance_idx is
        already unambiguous given a fixed game_pk -- "home team's 2nd
        reliever in this game, this trial" has exactly one meaning).

        When crn_game_pk/crn_trial are both given (CRN-paired mode, see
        crn.py), the draw is a deterministic hash of
        (game_pk, trial, side_tag, appearance_idx, DECISION_PITCHER_SHOCK) --
        this is the ONE new stochastic decision task #137 introduces, and per
        the CRN discipline every other decision in the game (SB, PA outcome,
        transition resampling) stays synchronized between a shock_sigma=0 and
        a shock_sigma>0 run up until this shock actually changes an outcome.
        Otherwise (crn disabled, matching every existing non-CRN caller) the
        draw comes from self.rng.normal(...), consuming exactly one value
        from the sequential stream -- same discipline as every other
        plain-rng decision in this class."""
        if self.shock_sigma <= 0:
            return None
        if crn_game_pk is not None and crn_trial is not None:
            g = crn_normal(0.0, self.shock_sigma, crn_game_pk, crn_trial, side_tag, appearance_idx,
                            DECISION_PITCHER_SHOCK)
        else:
            g = self.rng.normal(0.0, self.shock_sigma)
        # finding M6: the raw e^g plus the shared renorm-aware correction
        # table -- the context-dependent correction is applied per PA inside
        # combine_matchup_distribution, not baked in here, because it depends
        # on each PA's own shocked-mass share.
        return {"mult": float(np.exp(g)), "s_grid": self._shock_s_grid, "c_grid": self._shock_c_grid}

    def simulate_half_inning(self, lineup: list[dict], start_idx: int,
                              pitcher: dict, walkoff_margin: int | None = None,
                              park_factors: dict[str, float] | None = None,
                              weather_factors: dict[str, dict[str, float]] | None = None,
                              thruorder_counts: dict[int, int] | None = None,
                              events: list[dict] | None = None,
                              catcher_factors: dict[str, float] | None = None,
                              sb_rates: dict[int, dict[str, float]] | None = None,
                              auto_runner: bool = False,
                              hfa_factors: dict[str, float] | None = None,
                              crn_keys: tuple[int, int, int] | None = None,
                              pitcher_shock_factors: dict[str, float] | None = None,
                              inning: int | None = None,
                              hook_state: dict | None = None,
                              pitching_score_entering: int = 0,
                              batting_score_entering: int = 0) -> tuple[int, int]:
        """lineup: 9 batter profiles in order, each {"rates": {...}, "hand": "L"/"R",
        "same_mult": {...}, "opp_mult": {...}, "pull_tercile": str|None} (see
        build_wide_platoon_multipliers and spray.py).
        pitcher: the same profile shape for the (single, whole-game) pitcher.
        park_factors: {outcome: factor} for the stadium this game is played at
        (see park_factors.py) -- the same for both teams batting there.
        weather_factors: {"{stand}_{pull_tercile}": {outcome: factor}, ...}
        for this game's weather bucket (see weather.py) -- keyed by the
        batter's ACTUAL side (resolved below the same way platoon's
        is_same_hand is -- a switch-hitter always bats opposite the pitcher's
        hand) combined with their own pull tendency (see spray.py -- a real,
        validated signal beyond stand alone: pull-heavy hitters get
        disproportionately more benefit from wind blowing to their own pull
        field than balanced/oppo hitters do).
        thruorder_counts: optional {lineup_idx: times already faced THIS
        pitcher this game} dict, mutated in place -- see ttop.py and
        simulate_game (which owns resetting this whenever the pitcher
        actually changes; this method only reads/increments it).
        events: optional list, appended in-place with one
        {"batter_idx": lineup index, "outcome": str, "runs": int} dict per PA
        -- purely additive (every existing caller passes nothing and gets
        identical behavior/return value to before; see props.py for the
        per-at-bat prop generator this feeds).
        catcher_factors: optional {"strikeout": mult, "walk": mult} for the
        DEFENSIVE team's (this pitcher's) catcher this game (see
        catcher_framing.py) -- applies identically to every batter in this
        half-inning, same as park_factors.
        sb_rates: optional {lineup_idx: {"attempt_rate": float, "success_rate":
        float}} for the BATTING team's own runners (see baserunning.py) --
        unlike every other factor here, this is the BATTING team's own skill,
        not a pitching/defense/park property. When given, resolved ONCE per
        PA (a discrete-time approximation of a real steal attempt happening
        on some earlier pitch of that PA, same "one adjustment per PA, not
        per pitch" simplification already used for catcher framing) for a
        runner on 1st with 2nd open or on 2nd with 3rd open, checked lead
        runner (2nd->3rd) first then trail runner (1st->2nd). When None
        (every existing caller), no runner-identity tracking happens at all
        -- byte-for-byte the same RNG call sequence and behavior as before
        this parameter existed.
        Returns (runs_scored, next_start_idx).

        walkoff_margin: if given, the number of runs (can go negative) the
        batting team needs to EXCEED to end the game immediately -- checked
        after every single PA, not just at the half-inning's natural end.
        This models a real walk-off correctly: the game ends the instant the
        home team takes the lead in the bottom of the 9th+, not after they've
        used all 3 outs. Only ever passed for a bottom-half in inning 9+.

        auto_runner: real MLB rule (permanent since 2023) -- every half-inning
        from the 10th on starts with a runner on 2nd, not bases empty. When
        True, seeds state to "runner on 2nd, 0 outs" (bitmask 2 -> state 20,
        a state that already occurs constantly in the real historical data
        this table is built from, so no new state-factor/transition-table
        coverage is needed). The runner's IDENTITY (for sb_rates tracking
        only) is approximated as this team's own last batter retired --
        i.e. lineup_idx (start_idx - 1) % 9 -- since start_idx is always
        "next batter up for this team," (start_idx - 1) % 9 is exactly
        "last batter who made an out for this team" whenever a new
        half-inning is being seeded for a team that has already batted this
        game (always true for extra innings).

        hfa_factors: optional {outcome: factor} league-wide home-field-
        advantage multiplier (see park_factors.build_hfa_factors) -- unlike
        every other factor param here, NOT symmetric between teams: the
        caller (simulate_game) only ever passes this for the HOME lineup's
        own half-innings, never the away lineup's.

        crn_keys: optional (game_pk, trial, half_inning_ordinal) tuple (see
        crn.py) -- when given, every stochastic decision in this half-inning
        (stolen bases, PA outcome, transition-table resampling) is drawn from
        a deterministic hash of this key plus a running at-bat counter,
        instead of consuming `self.rng`'s next draw. This is for paired A/B
        testing: two simulator runs that build the SAME crn_keys for the
        same game/trial/half-inning stay synchronized on every decision up
        until the actual point where their outcomes first differ, instead of
        diverging immediately the way a shared sequential RNG stream does.
        Every existing caller (crn_keys=None, the default) is byte-for-byte
        identical to before this parameter existed.

        pitcher_shock_factors: optional (task #137) per-outcome multiplier
        for THIS pitcher's current appearance's latent "stuff" shock (see
        GameSimulator._draw_pitcher_shock) -- constant for the whole
        half-inning (one shock per appearance, not per PA), applied
        identically to every batter faced, same as catcher_factors/park_factors.

        inning/hook_state/pitching_score_entering/batting_score_entering:
        optional (task #144/#145, the state-conditioned starter-hook mechanism)
        -- when hook_state is None (every existing caller), completely inert;
        `inning`/the two score params are read ONLY inside the hook check below.
        hook_state, when given, is a MUTABLE dict shared across every half-inning
        this specific starter pitches in this specific trial (simulate_game owns
        building/reading it): {"hook_table": (inning_b,pa_count_b,runs_allowed_b,
        margin_b)->p_exit lookup, "z": this trial's frailty latent (float, fixed
        for the whole start), "next_reliever": profile to swap to if hooked,
        "hooked": bool (latches True on the FIRST hook, permanently), "hook_inning":
        set to the actual inning the hook fired, "pa_count"/"cum_runs_allowed": this
        starter's own running tallies THIS trial (mutated here, NEVER reset once
        hooked -- his tenure is over). Evaluated once per PA, BEFORE that PA is
        simulated (using pre-PA state, matching exactly how the fitted table's
        cells were defined -- see bullpen_usage_policy.py): if the hazard fires,
        the reliever in "next_reliever" takes over STARTING WITH this exact
        batter (the starter never faces them), thruorder_counts is cleared for
        the new pitcher (a real batter's times-through-order restarts against a
        new arm), and "hooked" latches so no further checks happen (only ONE
        starter-to-reliever handoff is modeled per side per game in this stage --
        subsequent reliever-to-reliever changes still happen at existing inning-
        boundary granularity in simulate_game, unchanged, per the task #144 step-2
        scope: hook model wired ALONE, reliever selection/order still pregame).
        A KNOWN, documented simplification: pitcher_shock_factors does NOT get
        redrawn immediately for the new reliever mid-half-inning (it continues
        using the OLD pitcher's shock for the remainder of THIS half-inning only)
        -- the redraw happens naturally next inning via simulate_game's existing
        id()-based change detection, which correctly fires exactly once since
        simulate_game syncs its own bookkeeping the instant hook_state["hooked"]
        flips True (see simulate_game). Deliberately accepted for this stage
        rather than threading extra CRN/appearance-index plumbing through this
        method for a second-order interaction that task #144 step 4 (SHOCK_SIGMA
        refit) already expects to revisit."""
        state, outs, idx, runs = (20, 0, start_idx, 0) if auto_runner else (0, 0, start_idx, 0)
        runners: dict[int, int] = {}  # base (1/2/3) -> lineup_idx, only maintained when sb_rates is given
        if auto_runner and sb_rates is not None:
            runners[2] = (start_idx - 1) % 9
        at_bat_counter = 0  # per-half-inning at-bat index for crn_keys -- see docstring above
        while outs < 3:
            lineup_idx = idx % 9
            batter = lineup[lineup_idx]

            # stolen-base attempts, resolved once per PA (a discrete-time
            # approximation of a real attempt on some earlier pitch of this
            # PA -- see the docstring above) BEFORE this batter's own AB.
            # Lead runner (2nd->3rd) checked first, then trail (1st->2nd),
            # matching real double-steal sequencing. Entirely skipped (byte-
            # for-byte identical behavior to before this parameter existed)
            # when sb_rates is None.
            if sb_rates is not None:
                for from_base, to_base in ((2, 3), (1, 2)):
                    if outs >= 3:
                        break  # half-inning already ended on this PA's earlier caught stealing
                    runner_idx = runners.get(from_base)
                    if runner_idx is None or runners.get(to_base) is not None:
                        continue
                    rates = sb_rates.get(runner_idx)
                    if rates is None:
                        continue  # no usable walk-forward history -- no attempt, not a guess
                    if crn_keys is not None:
                        attempts = crn_bool(rates["attempt_rate"], *crn_keys, at_bat_counter,
                                             DECISION_SB_ATTEMPT, from_base)
                    else:
                        attempts = self.rng.random() < rates["attempt_rate"]
                    if attempts:
                        if crn_keys is not None:
                            succeeds = crn_bool(rates["success_rate"], *crn_keys, at_bat_counter,
                                                 DECISION_SB_SUCCESS, from_base)
                        else:
                            succeeds = self.rng.random() < rates["success_rate"]
                        if succeeds:
                            runners[to_base] = runners.pop(from_base)
                        else:
                            runners.pop(from_base)
                            outs += 1
                if outs >= 3:
                    break  # half-inning ended on a caught stealing
                state = sum(1 << (b - 1) for b in runners) * 10 + outs

            # Task #144/#145 state-conditioned starter-hook check, evaluated
            # BEFORE this PA (using pre-PA exposure: this would-be Nth batter
            # faced, runs allowed strictly before this PA, current margin) --
            # matches exactly how the fitted hook table's cells are defined
            # (see bullpen_usage_policy.build_starter_hook_events). Entirely
            # skipped (byte-for-byte identical to before this existed) when
            # hook_state is None or the starter has already been hooked.
            if hook_state is not None and not hook_state["hooked"]:
                pa_number = hook_state["pa_count"] + 1
                margin_now = pitching_score_entering - (batting_score_entering + runs)
                cell = (bucket_inning(inning), bucket_pa_count(pa_number),
                        bucket_runs_allowed(hook_state["cum_runs_allowed"]), bucket_margin(margin_now))
                p_baseline = hook_state["hook_table"].get(cell)
                if p_baseline is not None:
                    p_hook = apply_hook_frailty(p_baseline, hook_state["z"], inning)
                    if crn_keys is not None:
                        fires = crn_bool(p_hook, *crn_keys, at_bat_counter, DECISION_HOOK_FRAILTY)
                    else:
                        fires = self.rng.random() < p_hook
                    if fires:
                        hook_state["hooked"] = True
                        hook_state["hook_inning"] = inning
                        # Task #144 step 4: when tier selection is ALSO enabled for this side,
                        # draw the reliever LIVE, conditioned on the exact margin at this hook
                        # moment, instead of using a pregame-fixed next_reliever. Recorded back
                        # into hook_state["next_reliever"] either way, so simulate_game's
                        # post-call id()-sync (below) reads whichever reliever actually took over.
                        if hook_state.get("tier_draw_fn") is not None:
                            pitcher = hook_state["tier_draw_fn"](inning, margin_now)
                        else:
                            pitcher = hook_state["next_reliever"]
                        hook_state["next_reliever"] = pitcher
                        if thruorder_counts is not None:
                            thruorder_counts.clear()

            # a switch-hitter ("S") always bats opposite the pitcher's hand by
            # definition -- that's the entire point of switch-hitting, not a
            # fixed attribute to look up like a regular batter's side.
            is_same_hand = False if batter["hand"] == "S" else batter["hand"] == pitcher["hand"]
            batter_mult = batter["same_mult"] if is_same_hand else batter["opp_mult"]
            pitcher_mult = pitcher["same_mult"] if is_same_hand else pitcher["opp_mult"]
            sf = self.state_factors[state] if self.state_factors is not None else None
            if weather_factors is not None:
                batter_stand = ("L" if pitcher["hand"] == "R" else "R") if batter["hand"] == "S" else batter["hand"]
                pull_tercile = batter.get("pull_tercile") or "mid_pull"
                wf = weather_factors.get(f"{batter_stand}_{pull_tercile}")
            else:
                wf = None
            # apply_ttop (2026-08-03 audit, finding M4): the TTOP table is now
            # fit within-pitcher on STARTER PAs only -- a reliever's profile
            # rates already come from overwhelmingly-first-pass PAs, so
            # applying the starter TT1 factor to him double-counts (the old
            # pooled table gave every reliever x1.055 strikeout on top of
            # already-reliever-high rates). Reliever/fallback/blowout profiles
            # are stamped apply_ttop=False at their build sites; a profile
            # without the key (starters, older callers) keeps the factor.
            if thruorder_counts is not None and self.ttop_factors is not None \
                    and pitcher.get("apply_ttop", True):
                times_through = min(thruorder_counts.get(lineup_idx, 0) + 1, 3)
                ttop = self.ttop_factors.get(times_through)
            else:
                ttop = None
            dist = combine_matchup_distribution(
                batter["rates"], pitcher["rates"], self.league_rates, state_factors=sf,
                batter_platoon_mult=batter_mult, pitcher_platoon_mult=pitcher_mult,
                park_factors=park_factors, weather_factors=wf, ttop_factors=ttop,
                catcher_factors=catcher_factors, hfa_factors=hfa_factors,
                pitcher_shock_factors=pitcher_shock_factors,
            )
            outcomes, probs = zip(*dist.items())
            if crn_keys is not None:
                outcome = crn_choice(outcomes, probs, *crn_keys, at_bat_counter, DECISION_OUTCOME)
                transition_key = (*crn_keys, at_bat_counter, DECISION_TRANSITION)
                post_state, runs_this_pa = self.transitions.sample(state, outcome, self.rng, crn_key=transition_key)
            else:
                outcome = self.rng.choice(outcomes, p=probs)
                post_state, runs_this_pa = self.transitions.sample(state, outcome, self.rng)
            if thruorder_counts is not None:
                thruorder_counts[lineup_idx] = thruorder_counts.get(lineup_idx, 0) + 1
            # Walk-off run truncation (2026-08-03 audit, finding M13): under
            # MLB rules a game-ending play only counts runs through the
            # WINNING run -- the game is over the instant the home team
            # leads -- UNLESS the play is a home run, which counts every
            # runner (the "walk-off grand slam" exception). The sampled
            # historical transition carries the full runs_scored from a
            # non-walkoff context, so a non-HR walkoff play must be
            # truncated to exactly (margin + 1) runs or ~0.7% of simulated
            # games end with MLB-impossible margins, distorting run-line
            # and totals distributions at the 1-vs-2-run boundary.
            if (walkoff_margin is not None and outcome != "home_run"
                    and runs + runs_this_pa > walkoff_margin + 1):
                runs_this_pa = walkoff_margin + 1 - runs
            if events is not None:
                # "pid": the pitcher who ACTUALLY threw this PA (2026-08-03
                # audit, finding M15) -- the local `pitcher` variable is
                # reassigned at the exact hook PA above, so this per-PA stamp
                # is exact even mid-half-inning, unlike any whole-inning
                # cutoff rule a caller could reconstruct after the fact
                # (which credits the starter for post-hook reliever PAs in
                # the hook inning). None when the caller's profiles carry no
                # "pid" key (e.g. validate_game_simulator's profiles).
                events.append({"batter_idx": lineup_idx, "outcome": outcome, "runs": runs_this_pa,
                               "pid": pitcher.get("pid")})
            if hook_state is not None and not hook_state["hooked"]:
                hook_state["pa_count"] += 1
                hook_state["cum_runs_allowed"] += runs_this_pa
            runs += runs_this_pa
            idx += 1
            at_bat_counter += 1
            if walkoff_margin is not None and runs > walkoff_margin:
                break  # walk-off: game ends the instant the home team takes the lead
            if post_state is None:
                break
            if sb_rates is not None:
                runners = _reassign_runners(runners, outcome, post_state // 10, runs_this_pa, lineup_idx)
            state = post_state
            outs = state % 10
        return runs, idx % 9

    @staticmethod
    def _pitcher_for_inning(default_pitcher: dict, bullpen: dict[int, dict] | None, inning: int) -> dict:
        """bullpen: {inning_number: pitcher_profile}, the REAL historical
        sequence of whichever pitcher started that inning for this team in
        this actual game (see validate_game_simulator.py) -- a faithful
        backtest of real bullpen usage, not a predictive model of when a
        manager pulls a starter. Falls back to the most recent known pitcher
        if the simulated game runs past what real bullpen data covered
        (e.g. simulated extra innings beyond how long the real game went)."""
        if bullpen is None:
            return default_pitcher
        if inning in bullpen:
            return bullpen[inning]
        prior = [i for i in bullpen if i < inning]
        return bullpen[max(prior)] if prior else default_pitcher

    # Blowout/position-player-pitching thresholds -- matches MLB's real
    # 8-run rule, confirmed empirically in blowout.py (a pitcher's entire
    # career sample being tiny AND concentrated in extreme-deficit situations
    # is overwhelmingly innings 8-9). Kept as local constants (rather than
    # imported from blowout.py) to avoid a circular import, since blowout.py
    # itself imports OUTCOMES from this module.
    BLOWOUT_MARGIN = 8
    BLOWOUT_MIN_INNING = 8

    def simulate_game(self, home_lineup: list[dict], away_lineup: list[dict],
                       home_pitcher: dict, away_pitcher: dict,
                       innings: int = 9, park_factors: dict[str, float] | None = None,
                       weather_factors: dict[str, dict[str, float]] | None = None,
                       home_bullpen: dict[int, dict] | None = None,
                       away_bullpen: dict[int, dict] | None = None,
                       blowout_pitcher_profile: dict | None = None,
                       events: list[dict] | None = None,
                       home_catcher_factor: dict[str, float] | None = None,
                       away_catcher_factor: dict[str, float] | None = None,
                       home_sb_rates: dict[int, dict[str, float]] | None = None,
                       away_sb_rates: dict[int, dict[str, float]] | None = None,
                       hfa_factors: dict[str, float] | None = None,
                       crn_game_pk: int | None = None, crn_trial: int | None = None,
                       home_hook_context: dict | None = None,
                       away_hook_context: dict | None = None,
                       home_tier_context: dict | None = None,
                       away_tier_context: dict | None = None,
                       postseason: bool = False,
                       hook_result: dict | None = None) -> tuple[int, int]:
        """home_lineup/away_lineup/home_pitcher/away_pitcher: player profiles,
        see simulate_half_inning. park_factors: the home team's stadium
        factors, applied to both teams' batting (a property of the park).
        weather_factors: {"{stand}_{pull_tercile}": {...}, ...} for this
        game's weather bucket (see weather.py and simulate_half_inning) --
        a property of the game/day, applied identically to both teams
        batting, just like park_factors.
        home_bullpen/away_bullpen: optional {inning: pitcher_profile} overrides
        -- see _pitcher_for_inning. home_pitcher/away_pitcher remain the
        starter and the fallback default when no bullpen mapping is given.
        blowout_pitcher_profile: optional profile (see blowout.py) swapped in
        for a team's pitcher, for the REST of the game once that team is
        first found trailing by BLOWOUT_MARGIN+ runs in inning
        BLOWOUT_MIN_INNING or later -- overrides home_bullpen/away_bullpen
        and home_pitcher/away_pitcher alike once triggered, matching real
        bullpen-conservation behavior (a team doesn't burn a real reliever
        once the game is clearly decided).
        home_catcher_factor/away_catcher_factor: optional {"strikeout": mult,
        "walk": mult} for each team's starting catcher (see
        catcher_framing.py) -- a single catcher for the whole game, same
        simplification level as the single default pitcher (a mid-game
        catcher substitution is rare enough to not model). Applied to
        whichever team is CATCHING that half-inning: home_catcher_factor when
        the away lineup bats (home team pitching/catching), and vice versa.
        home_sb_rates/away_sb_rates: optional {lineup_idx: {"attempt_rate":,
        "success_rate":}} per team (see baserunning.py) -- unlike
        catcher/park/weather, this is the BATTING team's own skill, so
        home_sb_rates is passed to the HOME lineup's own simulate_half_inning
        call (not the away one), and vice versa.
        hfa_factors: optional {outcome: factor} league-wide home-field-
        advantage multiplier (see park_factors.build_hfa_factors) -- passed
        ONLY to the home lineup's own simulate_half_inning call below, never
        the away lineup's (this is specifically the residual home edge, not
        a property of the park/weather/day that would apply to both sides).
        events: optional list, appended in-place with every PA from
        simulate_half_inning, additionally stamped with "inning", "side"
        ("home"/"away", the BATTING team), and "blowout" (whether the
        PITCHING team was in blowout_pitcher_profile mode for this PA) --
        see simulate_half_inning.

        Also owns resetting each lineup's times-through-the-order counter
        (see ttop.py, simulate_half_inning) whenever the OPPOSING pitcher
        actually changes -- detected via Python object identity (`id()`) on
        the pitcher profile dict, which works because every bullpen-plan
        builder in this project reuses the SAME profile object across all
        innings a given real pitcher is assigned, rather than rebuilding it
        fresh each inning.

        crn_game_pk/crn_trial: optional ints (see crn.py) -- when BOTH are
        given, every half-inning is simulated with a deterministic
        crn_keys=(crn_game_pk, crn_trial, half_inning_ordinal), for paired
        A/B testing (see simulate_half_inning's own docstring). Either
        omitted (the default) is byte-for-byte identical to before this
        parameter existed. Also gates whether the task #137 latent
        pitcher-appearance shock (see _draw_pitcher_shock) is CRN-keyed or
        drawn from the plain sequential rng -- same rule as every other
        stochastic decision in this class.

        home_hook_context/away_hook_context: optional (task #144/#145)
        {"hook_table": (inning_b,pa_count_b,runs_allowed_b,margin_b)->p_exit
        lookup, "cutoff": the pregame-assumed starter-exit inning that
        home_bullpen/away_bullpen was built around (see sample_bullpen_plan)}.
        When given for a side, that side's STARTER hook timing becomes
        state-conditioned (per-PA hazard against this trial's own developing
        game state) INSTEAD of a fixed pregame cutoff -- reliever IDENTITY
        and ORDER are completely unchanged (still exactly whatever
        home_bullpen/away_bullpen assigned), only the inning each one starts
        shifts to match the actual (possibly earlier or later than assumed)
        hook point. Either omitted (the default, None) is byte-for-byte
        identical to before this parameter existed: the starter pitches
        through the pregame-assumed cutoff exactly as home_bullpen/
        away_bullpen already encodes. See MODEL_DOCUMENTATION.md sec 11.18-
        11.20 for the full mechanism/validation story.

        home_tier_context/away_tier_context: optional (task #144 step 4)
        {"cutoff": pregame-assumed starter-exit inning (same value passed to
        the matching hook_context, if any), "tier_by_margin": P(tier|margin_b)
        dict, "closer_by_situation": P(closer|inning_b,save_situation) dict,
        "roster_weights": {pid: walk-forward usage+rest weight} (see
        bullpen.build_team_bullpen_roster), "profile_lookup": pid->profile
        callable, "closer_id": this team's identified closer or None,
        "fallback_profile": profile used once the roster is exhausted}. When
        given for a side, EVERY post-starter reliever selection for that side
        (not just the first) is drawn LIVE, conditioned on the CURRENT
        trial's own margin/save-situation at that exact inning, from the
        WITHOUT-REPLACEMENT roster pool -- reweighting WITHIN the pregame
        availability-weighted roster, never overriding rest-day availability
        (see tier_selection.sample_tier_conditioned_reliever). Combines with
        home_hook_context/away_hook_context if BOTH are given for a side: the
        starter's exit timing stays state-conditioned (hook), and the
        reliever who takes over -- and every one after -- is tier-drawn
        live instead of using hook_context's static pregame `next_reliever`
        or home_bullpen/away_bullpen at all (home_bullpen/away_bullpen and
        hook_context's own `next_reliever` become irrelevant once tier_context
        is given; the caller may pass home_bullpen=None in that case). If
        ONLY tier_context is given (no hook_context) for a side, the starter
        still exits at the pregame-assumed `cutoff` unchanged, but every
        reliever from there on is tier-drawn live. Also the point where the
        blowout/position-player-pitching unification (see blowout.py) lives:
        once a side enters blowout mode, a tier-conditioned draw (naturally
        favoring the "mopup" tier at extreme margins) is preferred over
        `blowout_pitcher_profile` for as long as this side's roster pool
        still has anyone available -- matching real bullpen-conservation
        behavior (a position player is used specifically to save the
        remaining bullpen for tomorrow, only once real arms are spent, not
        reached for the instant the blowout threshold is crossed). Either
        omitted (the default, None) is byte-for-byte identical to before
        this parameter existed.

        postseason: real MLB rule -- the automatic extra-innings runner (see
        auto_runner in simulate_half_inning) does NOT apply in the
        postseason; extra innings start bases-empty like any other inning.
        When True, `auto_runner` is forced off regardless of inning, and
        blowout/position-player-pitching substitution never triggers either
        (real playoff rosters/incentives mean a team essentially never
        concedes a game to save a reliever -- see blowout.py). Default False
        is byte-for-byte identical to before this parameter existed; the
        caller is expected to derive it from the schedule's own game_type
        field (game_type != "R" among the season's real games, excluding
        spring training/exhibition/all-star rows too -- see
        MODEL_DOCUMENTATION.md for the exact game_type values observed in
        this project's own schedule data).

        hook_result: optional dict, populated in-place (same convention as
        `events`) with {"home_hook_inning": ..., "away_hook_inning": ...} --
        the ACTUAL exit inning for each side once hook_context fires (None
        if that side has no hook_context, or hasn't been hooked by the time
        the game ends). Needed by any caller doing per-inning/per-pitcher
        prop attribution from its own pregame-built id_plan (see
        bullpen.sample_bullpen_plan): home_bullpen/away_bullpen is shifted
        internally to match the real hook timing (see
        _shift_bullpen_after_hook), but a caller's separately-tracked id_plan
        is not touched by this function and will silently misattribute
        innings to the wrong reliever after a hook unless the caller applies
        the identical shift itself using these values. Omitted (the default,
        None) is byte-for-byte identical to before this parameter existed."""
        crn_enabled = crn_game_pk is not None and crn_trial is not None
        home_score = away_score = 0
        home_idx = away_idx = 0
        home_in_blowout = away_in_blowout = False
        away_thruorder_counts: dict[int, int] = {}  # away lineup vs. the CURRENT home_pitcher_this_inning
        home_thruorder_counts: dict[int, int] = {}  # home lineup vs. the CURRENT away_pitcher_this_inning
        last_home_pitcher_id, last_away_pitcher_id = id(home_pitcher), id(away_pitcher)
        # Task #137: each side's CURRENT pitcher-appearance latent shock,
        # redrawn (a fresh independent draw, appearance_idx incremented)
        # every time the pitcher-object identity actually changes -- i.e.
        # once per real pitcher appearance, not once per inning or per PA.
        # Drawing the two starters' initial shocks here (appearance_idx=0)
        # mirrors last_home_pitcher_id/last_away_pitcher_id's own
        # before-the-loop initialization for the same reason: the starters
        # don't trigger an id()-change event, they're simply already assigned.
        home_appearance_idx = away_appearance_idx = 0
        home_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 0, home_appearance_idx)
        away_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 1, away_appearance_idx)

        # Task #144 step 4: mutable per-side tier-selection state, built ONLY
        # when a tier_context is given for that side. `available` is a fresh
        # per-trial COPY of the pregame roster_weights (mutated via pop() as
        # relievers are drawn, without-replacement -- the caller's own dict is
        # never touched). `tier_labels` derives each reliever's tier from
        # their OWN walk-forward roster weight (see tier_selection.py) --
        # computed once, stable for the whole game even as `available` depletes.
        def _build_tier_state(tier_context):
            if tier_context is None:
                return None
            from src.models.tier_selection import tier_label_from_roster_weights
            weights = tier_context["roster_weights"]
            return {"available": dict(weights), "tier_labels": tier_label_from_roster_weights(weights)}

        home_tier_state = _build_tier_state(home_tier_context)
        away_tier_state = _build_tier_state(away_tier_context)

        def _draw_tier_reliever(tier_context, tier_state, inning, margin, side_tag):
            def uniform_fn(sub_idx):
                if crn_enabled:
                    return crn_uniform(crn_game_pk, crn_trial, inning, side_tag, DECISION_TIER_SELECT, sub_idx)
                return self.rng.random()
            pid, profile = sample_tier_conditioned_reliever(
                tier_state["available"], tier_state["tier_labels"], tier_context["tier_by_margin"],
                tier_context["closer_by_situation"], tier_context["closer_id"], inning, margin,
                tier_context["profile_lookup"], uniform_fn,
            )
            return profile if profile is not None else tier_context["fallback_profile"]

        # Task #144/#145: mutable per-side hook state, built ONLY when a
        # hook_context is given for that side -- None (every existing caller)
        # leaves home_hook_state/away_hook_state as None, and every branch
        # below that reads them collapses to the exact pre-existing behavior.
        def _build_hook_state(hook_context, bullpen, default_pitcher, side_tag, tier_context, tier_state):
            if hook_context is None:
                return None
            cutoff = hook_context["cutoff"]
            tier_draw_fn = (lambda inn, mgn: _draw_tier_reliever(tier_context, tier_state, inn, mgn, side_tag)) \
                if tier_context is not None else None
            # When tier selection is combined, the pregame next_reliever is irrelevant
            # (the actual reliever is drawn live at hook-time -- see the hook-fire
            # block in simulate_half_inning) -- skip resolving it from home_bullpen/
            # away_bullpen, which the caller may legitimately pass as None in that case.
            next_reliever = None if tier_draw_fn is not None else self._pitcher_for_inning(default_pitcher, bullpen, cutoff + 1)
            z = (crn_normal(0.0, 1.0, crn_game_pk, crn_trial, side_tag, DECISION_HOOK_FRAILTY)
                 if crn_enabled else self.rng.normal(0.0, 1.0))
            return {"hook_table": hook_context["hook_table"], "z": z, "next_reliever": next_reliever,
                    "hooked": False, "hook_inning": None, "pa_count": 0, "cum_runs_allowed": 0,
                    "tier_draw_fn": tier_draw_fn}

        home_hook_state = _build_hook_state(home_hook_context, home_bullpen, home_pitcher, 0, home_tier_context, home_tier_state)
        away_hook_state = _build_hook_state(away_hook_context, away_bullpen, away_pitcher, 1, away_tier_context, away_tier_state)

        for inning in range(1, innings + 1):
            # Postseason: position-player/mop-up substitution essentially never
            # happens in October -- rosters/incentives are different (no team
            # concedes a playoff game to save a reliever) -- so blowout mode can
            # never trigger for a postseason game, matching real behavior.
            if not postseason and blowout_pitcher_profile is not None and not home_in_blowout \
                    and inning >= self.BLOWOUT_MIN_INNING and (home_score - away_score) <= -self.BLOWOUT_MARGIN:
                home_in_blowout = True
            if home_in_blowout:
                # Task #144 step 4 blowout/tier unification: prefer a real (tier-drawn,
                # naturally mopup-heavy at this margin) reliever over the position-player
                # profile for as long as this side's own roster pool has anyone left --
                # a position player is used to CONSERVE the bullpen, not the instant the
                # blowout threshold is crossed. Falls to blowout_pitcher_profile once the
                # pool is genuinely exhausted, or unconditionally when tier_context is None.
                if home_tier_context is not None and home_tier_state["available"]:
                    home_pitcher_this_inning = _draw_tier_reliever(
                        home_tier_context, home_tier_state, inning, home_score - away_score, 0)
                else:
                    home_pitcher_this_inning = blowout_pitcher_profile
            elif home_hook_state is not None and not home_hook_state["hooked"]:
                home_pitcher_this_inning = home_pitcher  # starter, state-conditioned exit -- see simulate_half_inning
            elif home_tier_context is not None and home_hook_state is None and inning <= home_tier_context["cutoff"]:
                home_pitcher_this_inning = home_pitcher  # starter, FIXED pregame cutoff (tier-only, no hook_context)
            elif home_tier_context is not None:
                home_pitcher_this_inning = _draw_tier_reliever(
                    home_tier_context, home_tier_state, inning, home_score - away_score, 0)
            else:
                home_pitcher_this_inning = self._pitcher_for_inning(home_pitcher, home_bullpen, inning)
            if id(home_pitcher_this_inning) != last_home_pitcher_id:
                away_thruorder_counts = {}
                last_home_pitcher_id = id(home_pitcher_this_inning)
                home_appearance_idx += 1
                home_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 0, home_appearance_idx)
            start_len = len(events) if events is not None else 0
            away_crn_keys = (crn_game_pk, crn_trial, half_inning_ordinal(inning, is_bottom=False)) if crn_enabled else None
            away_runs, away_idx = self.simulate_half_inning(
                away_lineup, away_idx, home_pitcher_this_inning, park_factors=park_factors,
                weather_factors=weather_factors, thruorder_counts=away_thruorder_counts, events=events,
                catcher_factors=home_catcher_factor, sb_rates=away_sb_rates, auto_runner=inning >= 10 and not postseason,  # real MLB rule: no auto-runner in the postseason
                crn_keys=away_crn_keys, pitcher_shock_factors=home_pitcher_shock,
                inning=inning, hook_state=home_hook_state,
                pitching_score_entering=home_score, batting_score_entering=away_score,
            )
            if home_hook_state is not None and home_hook_state["hooked"] \
                    and last_home_pitcher_id != id(home_hook_state["next_reliever"]):
                # The starter was just hooked mid-call (thruorder_counts already
                # cleared inside simulate_half_inning at the exact swap point --
                # NOT redone here, to avoid double-clearing a reliever's own
                # legitimately-accumulated tally from the rest of that half-inning).
                last_home_pitcher_id = id(home_hook_state["next_reliever"])
                home_appearance_idx += 1
                home_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 0, home_appearance_idx)
                home_bullpen = _shift_bullpen_after_hook(home_bullpen, home_hook_context["cutoff"], home_hook_state["hook_inning"])
            if events is not None:
                for e in events[start_len:]:
                    e["inning"], e["side"], e["blowout"] = inning, "away", home_in_blowout
            away_score += away_runs
            if inning >= 9 and home_score > away_score:
                break  # would-be bottom half never happens: home already ahead
            walkoff_margin = (away_score - home_score) if inning >= 9 else None
            if not postseason and blowout_pitcher_profile is not None and not away_in_blowout \
                    and inning >= self.BLOWOUT_MIN_INNING and (away_score - home_score) <= -self.BLOWOUT_MARGIN:
                away_in_blowout = True
            if away_in_blowout:
                if away_tier_context is not None and away_tier_state["available"]:
                    away_pitcher_this_inning = _draw_tier_reliever(
                        away_tier_context, away_tier_state, inning, away_score - home_score, 1)
                else:
                    away_pitcher_this_inning = blowout_pitcher_profile
            elif away_hook_state is not None and not away_hook_state["hooked"]:
                away_pitcher_this_inning = away_pitcher  # starter, state-conditioned exit -- see simulate_half_inning
            elif away_tier_context is not None and away_hook_state is None and inning <= away_tier_context["cutoff"]:
                away_pitcher_this_inning = away_pitcher  # starter, FIXED pregame cutoff (tier-only, no hook_context)
            elif away_tier_context is not None:
                away_pitcher_this_inning = _draw_tier_reliever(
                    away_tier_context, away_tier_state, inning, away_score - home_score, 1)
            else:
                away_pitcher_this_inning = self._pitcher_for_inning(away_pitcher, away_bullpen, inning)
            if id(away_pitcher_this_inning) != last_away_pitcher_id:
                home_thruorder_counts = {}
                last_away_pitcher_id = id(away_pitcher_this_inning)
                away_appearance_idx += 1
                away_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 1, away_appearance_idx)
            start_len = len(events) if events is not None else 0
            home_crn_keys = (crn_game_pk, crn_trial, half_inning_ordinal(inning, is_bottom=True)) if crn_enabled else None
            home_runs, home_idx = self.simulate_half_inning(
                home_lineup, home_idx, away_pitcher_this_inning, walkoff_margin=walkoff_margin,
                park_factors=park_factors, weather_factors=weather_factors,
                thruorder_counts=home_thruorder_counts, events=events,
                catcher_factors=away_catcher_factor, sb_rates=home_sb_rates, auto_runner=inning >= 10 and not postseason,  # real MLB rule: no auto-runner in the postseason
                hfa_factors=hfa_factors, crn_keys=home_crn_keys, pitcher_shock_factors=away_pitcher_shock,
                inning=inning, hook_state=away_hook_state,
                pitching_score_entering=away_score, batting_score_entering=home_score,
            )
            if away_hook_state is not None and away_hook_state["hooked"] \
                    and last_away_pitcher_id != id(away_hook_state["next_reliever"]):
                last_away_pitcher_id = id(away_hook_state["next_reliever"])
                away_appearance_idx += 1
                away_pitcher_shock = self._draw_pitcher_shock(crn_game_pk, crn_trial, 1, away_appearance_idx)
                away_bullpen = _shift_bullpen_after_hook(away_bullpen, away_hook_context["cutoff"], away_hook_state["hook_inning"])
            if events is not None:
                for e in events[start_len:]:
                    e["inning"], e["side"], e["blowout"] = inning, "home", away_in_blowout
            home_score += home_runs
            if inning >= 9 and home_score != away_score:
                break  # walk-off, or a top-of-9th-or-later deficit that stands
        if hook_result is not None:
            # Exposes the actual (possibly hook-shifted) exit inning so a
            # caller doing PER-PITCHER prop attribution from a pregame-built
            # id_plan (see bullpen.sample_bullpen_plan) can re-derive which
            # inning each planned reliever ACTUALLY threw in, via the exact
            # same shift _shift_bullpen_after_hook already applied to the
            # bullpen dict itself -- home_bullpen/away_bullpen is byte-shifted
            # internally above, but a caller's own separately-built id_plan
            # is not, and silently mismatches without this.
            hook_result["home_hook_inning"] = home_hook_state["hook_inning"] if home_hook_state is not None else None
            hook_result["away_hook_inning"] = away_hook_state["hook_inning"] if away_hook_state is not None else None
        return home_score, away_score


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building wide batter pregame rates (15 outcomes)...", flush=True)
    batter_wide = build_wide_pregame_rates(pa, "batter")
    print("building wide pitcher pregame rates (15 outcomes)...", flush=True)
    pitcher_wide = build_wide_pregame_rates(pa, "pitcher")
    league_rates = build_league_rates_by_season(pa)
    print("done building rate tables.", flush=True)

    transitions = TransitionTable(pa)
    rng = np.random.default_rng(0)

    print("\n=== sanity: simulate 5 quick half-innings with league-average everyone (no platoon effect) ===")
    avg_rates = {o: league_rates[2025][o] for o in OUTCOMES}
    neutral_mult = {o: 1.0 for o in OUTCOMES}
    avg_profile = {"rates": avg_rates, "hand": "R", "same_mult": neutral_mult, "opp_mult": neutral_mult}
    sim = GameSimulator(transitions, league_rates[2025], rng)
    lineup = [avg_profile] * 9
    for _ in range(5):
        runs, _ = sim.simulate_half_inning(lineup, 0, avg_profile)
        print(f"  half-inning runs: {runs}")
