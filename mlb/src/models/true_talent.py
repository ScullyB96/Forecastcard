"""Walk-forward-safe true-talent rate estimation for batters and pitchers,
per outcome category (strikeout, walk, single, home_run, ...).

Design follows Marcel's published two-level structure (the one fully-open,
fully-documented public projection system -- see project research notes):
  1. PRESEASON prior: a recency-weighted average of the player's own rate
     over the prior 3 seasons (weights 5/4/3, Marcel's own weights), itself
     regressed toward the league-average rate based on how much prior data
     exists (reliability = PA / (PA + K), K = stabilization constant).
  2. IN-SEASON blend: as the current season accumulates real PAs, blend the
     preseason prior (treated as K pseudo-PAs of league/player-blended rate)
     with the season-to-date actual count -- a standard Bayesian update.

K (the stabilization constant, in PA) is set per outcome category from
published sabermetric stabilization-point research (Russell Carleton) --
K-rate and BB-rate stabilize fast, HR/ISO-type rates much slower. These are
REASONABLE STARTING VALUES, not yet empirically re-validated on our own data
-- that validation is the next step (see project task list), not assumed here.

Efficiency note: this is vectorized (groupby + cumsum + shift), not a
PA-by-PA Python loop -- deliberate, since MLB-scale PA tables (600K+ rows/
multi-season) would make a naive per-row loop noticeably slow, unlike the
sibling NFL project's team/week-level granularity where a Python loop was
fine.
"""

import numpy as np
import pandas as pd

# Stabilization constants (in PA/BF), sourced from Russell Carleton's
# published stabilization-point research (FanGraphs Sabermetrics Library:
# https://library.fangraphs.com/principles/sample-size/), which gives
# DIFFERENT thresholds for hitters vs. pitchers -- confirmed directly against
# his published numbers (2026-07-20 research pass) that a single shared dict
# was wrong: pitchers stabilize meaningfully SLOWER than hitters for several
# categories (e.g. home_run at 1320 BF for pitchers vs. 170 PA for hitters --
# using the hitter constant for pitcher HR-allowed, as this project did until
# now, was trusting a pitcher's HR-rate-allowed sample roughly 8x too fast).
# Carleton never published individual 2B/3B thresholds, only a combined
# extra-base-hit figure (1610 hitter / 1450 pitcher); both double and triple
# use that combined figure on each side rather than the arbitrary 500 this
# project used previously. Categories with no published Carleton figure at
# all (field_out and the rare fielding/situational outcomes) keep their
# prior reasonable-placeholder values, unchanged and still flagged as such.
STABILIZATION_PA_BATTER = {
    "strikeout": 60,
    "walk": 120,
    # split from walk (2026-07-20): confirmed real, stable batter-specific signal
    # (year-over-year rate correlation ~0.65 for 300+ PA hitters -- who a manager
    # chooses to pitch around is consistent enough to be a real, if modest, part
    # of a hitter's own profile, not just situational noise) -- but a raw IBB rate
    # never reaches Carleton's usual 0.70 split-half threshold within a single
    # season even for full-time players (plateaus ~0.57 at 300 PA/half), so this
    # constant is extrapolated via the standard reliability=N/(N+K) relationship
    # from that measured point rather than read directly off a stabilized curve.
    "intent_walk": 250,
    "hit_by_pitch": 240,
    "single": 290,
    "double": 1610,
    "triple": 1610,
    "home_run": 170,
    "field_out": 60,  # ~ the complement of contact-quality outcomes; fast to stabilize like K%
    "double_play": 500,  # checked via leakage-free preseason-only K-sweep (2026-07-21):
                          # correlation peaks right at 500-1000 (0.339 both), already close to optimal
    "sac_fly": 200,  # lowered from 500 (2026-07-21): leakage-free K-sweep showed correlation
                     # MONOTONICALLY DECREASING with more shrinkage (0.121 at K=100 -> 0.101 at
                     # K=16000) -- a real, fast-stabilizing batter skill (some hitters are
                     # genuinely better at productive outs with a runner on 3rd), previously
                     # over-shrunk. 200 balances the swept optimum (~100) against this session's
                     # established practice of a safety margin above a bare point estimate.
    "sac_bunt": 200,  # lowered from 500 (2026-07-21): same K-sweep pattern as sac_fly, even
                      # stronger (0.618 at K=100 -> 0.554 at K=16000) -- sac bunting is a real,
                      # deliberate batter skill/strategy (bunt specialists exist), previously
                      # over-shrunk more than any other category checked in this pass.
    "fielders_choice": 500,  # checked (2026-07-21): correlation plateaus 0.051-0.052 from
                             # K=1000-8000, current value already within the flat region
    "field_error": 500,  # checked (2026-07-21): correlation stays flat (0.069-0.074) across the
                         # ENTIRE K=100-16000 grid -- consistent with this being a fielder's
                         # mistake the batter has little real control over; no K value exploits
                         # more signal than any other, current value is as good as any
    "catcher_interf": 2000,  # raised from 1000 (2026-07-21): leakage-free K-sweep peaks right
                              # at K=2000 (0.389) for the batter side specifically
    "triple_play": 1000,  # checked (2026-07-21): correlation is ~0 (or slightly negative) across
                          # the ENTIRE K=100-16000 grid -- confirms (does not just suspect) this
                          # category is fundamentally unpredictable at our data volume (a handful
                          # of real events across 4 seasons), not a tuning problem; current value
                          # left unchanged since no K meaningfully differs from any other here
}
STABILIZATION_PA_PITCHER = {
    "strikeout": 70,
    "walk": 170,
    # split from walk (2026-07-20): pitcher-side IBB rate shows weak, inconsistent
    # year-over-year correlation (0.15-0.38 across two season pairs, vs. 0.65-0.66
    # on the batter side) -- who gets intentionally walked is overwhelmingly a
    # decision about the BATTER (and game situation), not a pitcher tendency, so
    # this is shrunk almost entirely to league average rather than given a
    # from-scratch stabilization estimate the data doesn't really support.
    "intent_walk": 2000,
    "hit_by_pitch": 640,
    "single": 670,
    "double": 1450,
    "triple": 1450,
    "home_run": 1320,
    "field_out": 70,
    "double_play": 500,  # checked (2026-07-21): correlation peaks at K=250 (0.270), current
                         # value already close, small further gain available but not pursued
    "sac_fly": 5000,  # raised from 500 (2026-07-21): leakage-free K-sweep shows the OPPOSITE
                      # pattern from the batter side -- correlation rises with more shrinkage
                      # (0.101 at K=100 -> 0.118 at K=16000, plateauing ~K=4000-8000) -- whether
                      # a sac fly happens is overwhelmingly a batter/situational decision, not a
                      # pitcher tendency, previously under-shrunk on the pitcher side.
    "sac_bunt": 10000,  # raised from 500 (2026-07-21): same asymmetric pattern as sac_fly, more
                        # pronounced -- correlation still rising at K=16000 (0.052), confirms
                        # this is almost entirely a batter/manager decision, not pitcher skill.
    "fielders_choice": 350,  # lowered from 500 (2026-07-21): leakage-free K-sweep peaks at
                             # K=250 (0.133) then declines with more shrinkage -- current value
                             # was already close but slightly on the high side
    "field_error": 500,  # checked (2026-07-21): flat/inconclusive across the K=100-16000 grid,
                         # same as the batter side -- left unchanged
    "catcher_interf": 10000,  # raised from 1000 (2026-07-21): leakage-free K-sweep is STILL
                              # RISING at K=16000 (0.065, hasn't peaked) -- confirms catcher
                              # interference is essentially not a pitcher skill at all (it's
                              # about the batter's swing path relative to the catcher's glove),
                              # needs far more shrinkage than the batter side's own K=2000.
    "triple_play": 1000,  # checked (2026-07-21): same as batter side, ~0 correlation regardless
                          # of K -- fundamentally unpredictable at our data volume, left unchanged
}
# Backward-compatible alias: game_simulator.py and park_factors.py only ever
# use this for `.keys()` (the canonical 16-category outcome list, identical
# on both sides), never the values -- safe to keep pointing at one dict.
STABILIZATION_PA = STABILIZATION_PA_BATTER

MARCEL_WEIGHTS = [5, 4, 3]  # most recent season first

# Marcel's own published age adjustment (Tango, tangotiger.net/archives/stud0346.shtml),
# added 2026-07-21 after a literature research pass found this was the one concrete,
# fully-specified component of real Marcel our reimplementation was missing entirely.
# A single global peak age, applied as a multiplicative % shift to a player's weighted
# historical rate BEFORE regression toward league average -- asymmetric: pre-peak
# improvement happens twice as fast as post-peak decline, per Tango's own formula.
AGE_PEAK = 29
AGE_ADJ_RATE_BELOW_PEAK = 0.006  # per year of age below peak
AGE_ADJ_RATE_ABOVE_PEAK = 0.003  # per year of age above peak


def _age_adjustment(age: float) -> float:
    if age < AGE_PEAK:
        return (AGE_PEAK - age) * AGE_ADJ_RATE_BELOW_PEAK
    return (AGE_PEAK - age) * AGE_ADJ_RATE_ABOVE_PEAK


def _project_age(pa: pd.DataFrame, player_col: str, target_season: int) -> pd.Series:
    """Each player's PROJECTED age in target_season: their most recent known
    age (Statcast's own age_bat/age_pit columns -- a fixed-reference-date
    convention confirmed on real data to increment by exactly 1 every season
    for every player, with zero exceptions) plus the season gap to
    target_season. Only defined for player_col in {"batter","pitcher"} --
    bullpen.py's pooled "team" bullpen rate has no single age."""
    if player_col not in ("batter", "pitcher"):
        return pd.Series(dtype=float)
    age_col = "age_bat" if player_col == "batter" else "age_pit"
    prior = pa[pa["season"] < target_season]
    if prior.empty:
        return pd.Series(dtype=float)
    by_season = prior.groupby([player_col, "season"])[age_col].first().reset_index()
    latest = by_season.sort_values("season").groupby(player_col).last()
    return latest[age_col] + (target_season - latest["season"])


# Age-adjustment sign/scale overrides by (player_col, outcome) -- confirmed via
# leakage-free preseason-only testing (2026-07-21) that Marcel's default
# convention (raw_rate * (1 + adj), same sign for every category) implicitly
# assumes "a higher rate means this player is improving" -- true for every
# category this project's true-talent rates are GOOD for that player role
# (batter hits/BB/HR, pitcher strikeout), but FALSE where a higher rate is BAD
# for that role. Applying the default sign to batter strikeout was confirmed to
# ACTIVELY HURT accuracy (mean corr 0.7532 -- worse than not adjusting at all,
# 0.7567), while the INVERTED sign was the best of the three tested (0.7575).
# batter field_out was also tested and found to have NO exploitable age signal
# in either direction (no-adjustment beat both the default and inverted sign,
# 0.6559 vs. 0.6527 vs. 0.6500) -- left at 0 rather than guess a sign with no
# supporting evidence. Pitcher-side "allowed" categories (walk, home_run) were
# tested too and showed no meaningful difference between any of the three
# options (differences within noise) -- left on the default rather than invert
# without real evidence; pitcher strikeout confirms the DEFAULT sign is
# correct there (0.5929, beats both no-adjustment 0.5902 and inverted 0.5851).
# The remaining situational/mechanical categories (double_play, sac_fly,
# sac_bunt, fielders_choice, field_error, catcher_interf, triple_play) were
# never individually tested for age-sign and already showed weak-to-no real
# skill signal in this session's STABILIZATION_PA leakage-free sweep -- left
# at 0 rather than apply an unvalidated, possibly-wrong-signed adjustment.
#
# Five pitcher "allowed" categories (intent_walk, hit_by_pitch, single,
# double, triple) were found MISSING from this dict entirely by a later
# audit (2026-07-21) -- meaning they silently defaulted to the untested +1.0
# sign, the exact same bug class already fixed for batter strikeout. Ran the
# same leakage-free sweep (default/inverted/no-adjustment, comparing each
# against a no-adjustment baseline to judge whether the CURRENT default was
# doing active harm, same standard the original strikeout fix used):
# hit_by_pitch (default 0.3647 underperforms no-adj 0.3660; inverted 0.3670
# best) and double (default 0.0965 underperforms no-adj 0.1009 by a real
# margin; inverted 0.1049 best) both confirmed the default was actively
# hurting -- set to inverted (-1.0). single (default 0.3336 clearly BEATS
# no-adj 0.3315 and inverted 0.3268) confirms the untested default was
# actually already correct -- kept at +1.0, now documented rather than
# accidental. intent_walk and triple showed only noise-level differences
# between all three options (<0.001 apart, and triple's overall correlation
# is very low, ~0.09, meaning age barely predicts it at all) -- left at 0
# per the same "no supporting evidence" standard as the situational
# categories above, rather than lock in a coin-flip sign.
AGE_ADJ_SIGN_OVERRIDE = {
    ("batter", "strikeout"): -1.0,
    ("batter", "field_out"): 0.0,
    ("batter", "double_play"): 0.0,
    ("batter", "sac_fly"): 0.0,
    ("batter", "sac_bunt"): 0.0,
    ("batter", "fielders_choice"): 0.0,
    ("batter", "field_error"): 0.0,
    ("batter", "catcher_interf"): 0.0,
    ("batter", "triple_play"): 0.0,
    ("pitcher", "field_out"): 0.0,
    ("pitcher", "double_play"): 0.0,
    ("pitcher", "sac_fly"): 0.0,
    ("pitcher", "sac_bunt"): 0.0,
    ("pitcher", "fielders_choice"): 0.0,
    ("pitcher", "field_error"): 0.0,
    ("pitcher", "catcher_interf"): 0.0,
    ("pitcher", "triple_play"): 0.0,
    ("pitcher", "intent_walk"): 0.0,
    ("pitcher", "hit_by_pitch"): -1.0,
    ("pitcher", "single"): 1.0,
    ("pitcher", "double"): -1.0,
    ("pitcher", "triple"): 0.0,
}


def _age_adj_sign(player_col: str, outcome: str) -> float:
    return AGE_ADJ_SIGN_OVERRIDE.get((player_col, outcome), 1.0)


def _stabilization_dict(player_col: str) -> dict:
    """Batter-side constants for player_col='batter'; pitcher-side constants
    otherwise -- covers both 'pitcher' and 'team' (bullpen.py's pooled
    team-aggregate bullpen rate is fundamentally a PITCHING quantity and
    should regress like one, not like a hitter)."""
    return STABILIZATION_PA_BATTER if player_col == "batter" else STABILIZATION_PA_PITCHER


def _park_neutral_events(pa: pd.DataFrame, is_outcome: pd.Series, outcome: str,
                          park_factors_df: pd.DataFrame | None) -> pd.Series:
    """Divide each observed event by that game's own park factor for this
    outcome (keyed on the game's home_team + season -- park effects apply
    identically to both teams playing there, whether we're neutralizing a
    batter's or a pitcher's rate) before it feeds Marcel rate estimation.
    Without this, a raw counting rate carries whatever mix of hitter-friendly/
    pitcher-friendly parks a player happened to play in baked in, and then
    game_simulator.py applies its own per-game park_factors multiplier ON TOP
    of that already-contaminated rate -- a double-count. `park_factors_df` is
    `park_factors.build_outcome_park_factors`'s own walk-forward-safe output
    (season, team, outcome, park_factor); missing lookups (e.g. the walk-
    forward table's own first/cold-start season, which has no prior data to
    build a factor from) default to a neutral 1.0 rather than dropping the PA.
    `park_factors_df=None` (the default) preserves the exact old, unadjusted
    behavior -- every existing caller that doesn't opt in is unaffected."""
    if park_factors_df is None:
        return is_outcome
    sub = park_factors_df[park_factors_df["outcome"] == outcome]
    pf_map = sub.set_index(["season", "team"])["park_factor"]
    keys = pd.Series(list(zip(pa["season"], pa["home_team"])), index=pa.index)
    pf = keys.map(pf_map).fillna(1.0)
    return is_outcome / pf


def _season_rates(pa: pd.DataFrame, player_col: str, outcome: str,
                   park_factors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (player, season): PA count and park-neutral event count/
    rate for this outcome (see _park_neutral_events)."""
    tmp = pa[[player_col, "season"]].copy()
    is_outcome = (pa["outcome"] == outcome).astype(int)
    tmp["is_outcome"] = is_outcome
    tmp["is_outcome_pn"] = _park_neutral_events(pa, is_outcome, outcome, park_factors_df)
    out = tmp.groupby([player_col, "season"]).agg(
        pa=("is_outcome", "size"), events=("is_outcome_pn", "sum")
    ).reset_index()
    out["rate"] = out["events"] / out["pa"]
    return out


def build_preseason_priors(pa: pd.DataFrame, player_col: str, outcome: str, target_season: int,
                            park_factors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The Marcel-style weighted, regressed prior for `target_season`, built
    only from seasons STRICTLY BEFORE it (no leakage)."""
    empty_cols = [player_col, "preseason_rate", "prior_weight_pa"]
    prior_seasons = pa[pa["season"] < target_season]

    if prior_seasons.empty:
        # true cold start (this is the earliest season in our cache) -- no
        # way to avoid some look-ahead here, since there is no prior data at
        # all; fall back to this season's own league rate and give every
        # player zero individual prior weight (in-season data does all the
        # work). Only affects the very first season in the dataset. Not
        # park-neutralized: a league-wide rate already averages evenly over
        # every park (every team's home PAs are exactly matched by another
        # team's road PAs at that same park), so it's already park-neutral
        # by construction -- no adjustment needed here, unlike an individual
        # player's rate, which can carry a lopsided mix of parks.
        league_rate = (pa.loc[pa["season"] == target_season, "outcome"] == outcome).mean()
        return pd.DataFrame(columns=empty_cols), league_rate

    league_rate = (prior_seasons["outcome"] == outcome).mean()
    season_rates = _season_rates(prior_seasons, player_col, outcome, park_factors_df)

    priors = []
    for player, g in season_rates.groupby(player_col):
        g = g.set_index("season")
        num, den, raw_pa = 0.0, 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                num += w * g.loc[s, "events"]
                den += w * g.loc[s, "pa"]
                raw_pa += g.loc[s, "pa"]
        if den == 0:
            continue
        priors.append({player_col: player, "prior_pa": den, "prior_events": num, "raw_prior_pa": raw_pa})
    priors_df = pd.DataFrame(priors)
    if priors_df.empty:
        return pd.DataFrame(columns=empty_cols), league_rate

    K = _stabilization_dict(player_col)[outcome]
    # RELIABILITY (and the pseudo-PA weight below) must use RAW, unweighted
    # cumulative PA -- NOT the MARCEL_WEIGHTS-weighted `prior_pa` (which sums
    # to 12, not 3, across a full 3-year history, inflating "PA" by ~4.5x for
    # any player with fewer than 3 full prior seasons). Confirmed a real bug
    # on our own data (2026-07-21): mean reliability computed against the
    # weighted sum was 0.784 vs. 0.574 using raw PA -- a large, systematic
    # OVER-statement of confidence. This matters because K itself comes from
    # Russell Carleton's published stabilization-point research, which is
    # calibrated via real split-half testing against RAW, unweighted PA --
    # plugging a ~4.5x-inflated denominator into a raw-PA-calibrated K
    # silently made the model trust a player's own history far more than
    # Carleton's own research would support, under-shrinking toward league
    # average across every outcome category for every player. The WEIGHTED
    # sum (`prior_pa`/`den`) is still exactly right for the RATE estimate
    # itself (that's real Marcel's actual recency-weighting) -- only the
    # reliability/confidence side of the formula was using the wrong units.
    priors_df["reliability"] = priors_df["raw_prior_pa"] / (priors_df["raw_prior_pa"] + K)
    raw_rate = priors_df["prior_events"] / priors_df["prior_pa"]

    # Age adjustment (see AGE_PEAK docstring above): applied to the weighted
    # historical rate BEFORE blending with league average, same stage real
    # Marcel applies it at. A player with no projectable age (e.g. player_col
    # == "team", or a genuine data gap) gets a neutral 0% adjustment.
    projected_age = _project_age(pa, player_col, target_season)
    age_adj = priors_df[player_col].map(projected_age).apply(
        lambda a: _age_adjustment(a) if pd.notna(a) else 0.0
    ) * _age_adj_sign(player_col, outcome)
    age_adjusted_rate = raw_rate * (1 + age_adj)

    priors_df["preseason_rate"] = (
        priors_df["reliability"] * age_adjusted_rate + (1 - priors_df["reliability"]) * league_rate
    )
    # pseudo-PA this prior is "worth" going into the in-season blend: a constant K
    # (same raw-PA units as season_pa_before in build_pregame_rates). Constant, not
    # min(raw_prior_pa, K) (2026-08-03 audit, finding M2): the old min() gave a player
    # with 1-59 raw prior PA a WEAKER prior anchor than a player with ZERO history
    # (whose missing row fillna's to K downstream) -- more information should never
    # produce a less sticky prior, and 67 real 2025 batters sat in that inverted band.
    # Bayesianly, a with-history player's preseason_rate is already a (league worth
    # K) + (own raw PA) blend, so its worth is >= K, and K is the floor that also
    # preserves the original cap's intent (a long track record still doesn't slow
    # in-season reactivity -- veterans' weight is unchanged at K by this fix).
    priors_df["prior_weight_pa"] = float(K)
    return priors_df[empty_cols], league_rate


EARLIEST_SEASON = 2023  # our own data's first season -- see build_debut_rate: a real MLB
                        # debut can't be distinguished from an established player in THIS
                        # specific season, since we have no earlier data to check against.


def build_debut_rate(pa: pd.DataFrame, player_col: str, outcome: str, target_season: int,
                      league_rate: float, park_factors_df: pd.DataFrame | None = None) -> float:
    """The walk-forward-safe empirical rate real MLB debut cohorts have
    posted in seasons STRICTLY BEFORE target_season, pooled across every
    such season and Bayesian-shrunk toward league_rate -- used as the
    true-cold-start prior for a player with literally zero PA in any prior
    season, replacing Marcel's own unqualified league-average fallback.

    A real rookie/debut performs below league average (that's close to the
    definition of replacement level); filling their prior with the plain
    league mean systematically OVERRATES every player with no MLB track
    record -- confirmed a real, structural gap in this project's own Marcel
    implementation (2026-07-22 critique). A player "debuts" in season s if
    they have a PA in s but none in any earlier season within our own data.
    EARLIEST_SEASON (2023, our data's own first season) can't identify real
    debuts at all -- every player looks like a debut that year regardless of
    actual MLB tenure -- so target_season <= EARLIEST_SEASON + 1 (i.e. no
    debut cohort has been fully observed yet) falls back to the plain
    league_rate, unchanged from the previous behavior."""
    prior_seasons = pa[pa["season"] < target_season]
    debut_cohort_seasons = sorted(s for s in prior_seasons["season"].unique() if s > EARLIEST_SEASON)
    if not debut_cohort_seasons:
        return league_rate  # not enough history yet to observe a real debut cohort

    first_season_by_player = prior_seasons.groupby(player_col)["season"].min()
    debutant_ids = first_season_by_player[first_season_by_player.isin(debut_cohort_seasons)].index
    debut_pa = prior_seasons[prior_seasons[player_col].isin(debutant_ids)]
    if debut_pa.empty:
        return league_rate

    is_outcome = (debut_pa["outcome"] == outcome).astype(int)
    weighted = _park_neutral_events(debut_pa, is_outcome, outcome, park_factors_df)
    n = len(debut_pa)
    K = _stabilization_dict(player_col)[outcome]
    reliability = n / (n + K)
    return reliability * (weighted.sum() / n) + (1 - reliability) * league_rate


def build_pregame_rates(pa: pd.DataFrame, player_col: str, outcome: str,
                         park_factors_df: pd.DataFrame | None = None,
                         use_debut_prior: bool = False) -> pd.DataFrame:
    """For every PA, the walk-forward-safe (no leakage) estimated rate for
    that player/outcome as of just before that PA -- preseason prior blended
    with in-season-to-date actuals.

    `park_factors_df` (see _park_neutral_events): when supplied, both the
    preseason prior AND the in-season-to-date blend are park-neutralized, so
    the resulting pregame_rate reflects the player's true talent independent
    of which parks their PAs happened to occur in -- game_simulator.py's own
    per-game park_factors multiplier remains the ONLY place park effects
    re-enter, instead of being double-counted on top of a park-contaminated
    rate. Defaults to None (unadjusted, old behavior) for backward compat.

    `use_debut_prior`: whether a player with zero prior-season PA falls back
    to build_debut_rate's empirical debut-cohort rate (True) or the plain
    league_rate (False, the default/old behavior) -- unlike park-neutralization
    and HFA (both correctness fixes for an existing double-count), this is a
    genuinely new mechanism this project hasn't yet full-stack A/B tested, so
    it defaults OFF until that test runs (see task #119) rather than silently
    changing every live caller's behavior."""
    # game_date (not game_pk) is the reliable chronological anchor -- game_pk
    # can reflect a game's originally-scheduled slot rather than when it was
    # actually played if postponed/made up later. game_pk still breaks ties
    # for same-day doubleheaders, where game_date alone can't order games.
    pa = pa.sort_values([player_col, "season", "game_date", "game_pk", "at_bat_number"]).copy()
    is_outcome = (pa["outcome"] == outcome).astype(int)
    pa["is_outcome"] = is_outcome
    pa["is_outcome_pn"] = _park_neutral_events(pa, is_outcome, outcome, park_factors_df)

    K = _stabilization_dict(player_col)[outcome]
    out_frames = []
    for season, sdf in pa.groupby("season"):
        priors_df, league_rate = build_preseason_priors(pa, player_col, outcome, season, park_factors_df)
        cold_start_rate = (
            build_debut_rate(pa, player_col, outcome, season, league_rate, park_factors_df)
            if use_debut_prior else league_rate
        )
        sdf = sdf.merge(priors_df, on=player_col, how="left")
        # a player with NO prior-season row at all (never merged) gets
        # cold_start_rate -- the empirical debut-cohort rate when
        # use_debut_prior=True (see build_debut_rate), else the plain league
        # average (old behavior). Established players' own regression-
        # toward-league-average blend (inside build_preseason_priors, for
        # players WITH some prior history) is untouched either way.
        sdf["preseason_rate"] = sdf["preseason_rate"].fillna(cold_start_rate)
        sdf["prior_weight_pa"] = sdf["prior_weight_pa"].fillna(K)

        grp = sdf.groupby(player_col)
        sdf["season_events_before"] = grp["is_outcome_pn"].cumsum() - sdf["is_outcome_pn"]
        sdf["season_pa_before"] = grp.cumcount()

        num = sdf["prior_weight_pa"] * sdf["preseason_rate"] + sdf["season_events_before"]
        den = sdf["prior_weight_pa"] + sdf["season_pa_before"]
        sdf["pregame_rate"] = num / den
        # the Bayesian-blend denominator IS the effective pseudo+real sample size backing
        # pregame_rate -- i.e. pregame_rate is exactly the mean of a Beta(alpha, beta)
        # posterior with alpha=pregame_rate*effective_n, beta=(1-pregame_rate)*effective_n.
        # Exposed for task #127 (posterior-sampled rates): a caller wanting genuine
        # parameter uncertainty (not just outcome-sampling noise) can draw ONE Beta sample
        # per trial from this instead of using the point estimate every trial -- see
        # sample_posterior_rate below. Purely additive (an extra output column), no
        # existing caller's behavior changes by this alone.
        sdf["effective_n"] = den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result.drop(columns=["is_outcome", "is_outcome_pn", "season_events_before", "season_pa_before"])


def sample_posterior_rate(pregame_rate: float, effective_n: float, rng: np.random.Generator) -> float:
    """Draw ONE random variate from the Beta(alpha, beta) posterior implied by
    a walk-forward pregame_rate/effective_n pair (see build_pregame_rates),
    instead of using the point-estimate mean directly. This is the
    correctness fix for the diagnosed total-runs under-dispersion (§11.8
    claim 5 / MODEL_DOCUMENTATION.md): every existing trial uses the SAME
    fixed point-estimate rate, so the simulator propagates outcome
    randomness but zero estimation/parameter uncertainty -- real totals are
    more spread out than the model's own Monte Carlo band because of this.
    The shrinkage formula this project already uses everywhere
    ((observed + PRIOR*prior_rate) / (n + PRIOR)) IS a Beta posterior mean by
    construction; this just actually samples from that posterior instead of
    only ever using its mean. `effective_n` <= 0 (a genuine edge case, e.g. a
    fully cold-started player with zero weight) returns pregame_rate
    unchanged rather than dividing by zero."""
    if effective_n <= 0:
        return pregame_rate
    alpha = max(pregame_rate * effective_n, 1e-6)
    beta = max((1 - pregame_rate) * effective_n, 1e-6)
    return float(rng.beta(alpha, beta))


def widen_rate(pregame_rate: float, league_rate: float, w: float) -> float:
    """Task #134 (Phase 1 pivot, 2026-07-22): stretch a walk-forward
    pregame_rate away from league average by factor `w`, restoring
    across-player TALENT spread that Marcel shrinkage compresses relative to
    real season-long outcome variance.

    Root cause this fixes: Phase 1.1's refined (opponent-adjusted)
    overdispersion test found NO detectable day-level latent effect on
    either side (pitcher std(z)=0.987, team-offense std(z)=0.979 -- both
    already AT the target ~1.0once matched against real season-long-rate
    matchups) -- ruling out a missing within-game correlation mechanism as
    the cause of the simulator's own under-dispersion. But real season-long
    rates are 1.4-1.9x MORE spread across players than this project's own
    walk-forward pregame rates (confirmed empirically per outcome category:
    home_run 1.72x, walk 1.58x, strikeout 1.42x, single 1.92x) -- shrinkage,
    tuned to maximize PREDICTIVE correlation on held-out data, compresses
    the across-player spread more than real talent differences justify. This
    directly explains the under-dispersion: every simulated matchup is
    closer to a league-average matchup than a real one actually is, so
    simulated outcomes cluster tighter than reality.

    This is deliberately NOT random (unlike sample_posterior_rate) -- it's a
    fixed transform of the point estimate itself, applied once per profile
    like the rate it replaces, not once per trial. w=1.0 is a no-op
    (identical to today's behavior). Preserves the population mean across
    players (E[widen_rate] = league_rate when E[pregame_rate] = league_rate,
    since the stretch is centered on league_rate), so this is a SPREAD
    correction, not a bias shift for any individual player."""
    return float(np.clip(league_rate + w * (pregame_rate - league_rate), 0.0, 1.0))


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")

    print("=== batter strikeout-rate pregame estimates: validation ===")
    result = build_pregame_rates(pa, "batter", "strikeout")
    test = result[result["season"].isin([2024, 2025])].copy()
    test["actual"] = (test["outcome"] == "strikeout").astype(int)
    mae = (test["pregame_rate"] - test["actual"]).abs().mean()
    naive_mae = (test["actual"].mean() - test["actual"]).abs().mean()
    corr_input = test.groupby("batter").agg(pred=("pregame_rate", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    corr_input = corr_input[corr_input["n"] >= 50]
    corr = corr_input["pred"].corr(corr_input["actual"])
    print(f"PA-level MAE: {mae:.4f}  naive(league rate) MAE: {naive_mae:.4f}")
    print(f"player-level correlation (pred pregame rate vs realized rate, batters w/ 50+ PA): {corr:.3f}  n={len(corr_input)}")
