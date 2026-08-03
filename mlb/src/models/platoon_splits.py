"""Platoon-split (vs same-handed / opposite-handed pitching) adjustment,
applied as a multiplier on top of a player's existing overall true-talent
rate (from true_talent.py) -- not a fully separate true-talent system.

Design follows Tango/Lichtman/Dolphin's *The Book* (Ch. 6) finding directly:
individual platoon-split samples are almost always dominated by noise --
it takes roughly 2,200 PA of a SPECIFIC player's own platoon data before
you'd even weight it 50/50 against the league-average platoon effect. So:
  1. Compute the LEAGUE-AVERAGE platoon effect per outcome (large-sample,
     stable, walk-forward/prior-seasons-only) as an odds-ratio multiplier:
     how much a rate changes facing opposite-handed vs. same-handed pitching.
  2. For a given player, the default projection is that league-average
     effect applied to their own overall rate -- sensible even with zero
     individual split data.
  3. Blend that default with the player's own observed split-specific
     multiplier (relative to their OWN combined rate, so it's a pure
     "shape" adjustment, not re-deriving their overall level), weighted by
     reliability = player_split_pa / (player_split_pa + 2200).

"same-handed" is always `stand == p_throws` -- a property of the matchup,
the same regardless of whether we're adjusting the batter's or the
pitcher's rate.

Confirmed the underlying platoon effect is real and large in our own data
before building this: league-wide walk rate is 0.0749 vs same-handed
pitching, 0.0932 vs opposite-handed (a ~24% relative difference) -- not
noise, a well-established, textbook-matching effect.
"""

import json

import pandas as pd

from src.models.matchup import odds, prob_from_odds
from src.models.true_talent import MARCEL_WEIGHTS
from src.utils.paths import DATA_PROCESSED

MEASURED_SHARED_TERM_PATH = DATA_PROCESSED / "platoon_shared_term.json"
_measured_shared_term_cache = None  # module-level cache -- loaded once, not once per outcome/season


def _load_measured_shared_term() -> dict:
    """{season_str: {outcome: {"LL"/"LR"/"RL"/"RR": ratio}}}, produced by
    measure_platoon_shared_term.py (see that module's docstring for the
    estimand and why it's a separate, periodically-refreshed script rather
    than something platoon_splits.py computes inline -- circular import).
    Returns {} (falls back to the exponent-based shared term everywhere) if
    the file hasn't been generated yet."""
    global _measured_shared_term_cache
    if _measured_shared_term_cache is None:
        if MEASURED_SHARED_TERM_PATH.exists():
            with open(MEASURED_SHARED_TERM_PATH) as f:
                _measured_shared_term_cache = json.load(f)
        else:
            _measured_shared_term_cache = {}
    return _measured_shared_term_cache


PLATOON_STABILIZATION_PA = 2200  # Tango/Lichtman/Dolphin's own published figure

# Structural double-count fix, part 1 (2026-08-03, external review) -- see the full
# derivation in build_platoon_multipliers below. 0.25: each side's shared population-
# effect term carries a QUARTER of the population log-odds effect, so two sides combine
# to the HALF-effect the data actually supports. SUPERSEDED as the primary shared-term
# source by the 2026-08-04 measured estimand (see _measured_shared_term_for) -- an
# exponent sweep found no single fixed value fit both home_run and strikeout (each
# outcome's real redundancy scales differently), and a "no exponent at all" attempt
# (using the like-currency reference at full strength) badly overcorrected. This
# exponent-based value is now ONLY the fallback for any (season, outcome, stand,
# p_throws) cell the measured table doesn't cover (new/unmeasured outcomes, or the
# true cold-start season) -- kept as a named constant, not removed, so a future A/B
# script can still monkeypatch it for a comparison arm, same convention as
# validate_game_simulator.py's disable_bat_speed/disable_pulled_air flags.
PLATOON_LEAGUE_DEFAULT_EXPONENT = 0.25

SPLIT_RATE_PRIOR_PA = 20  # small Bayesian prior for the observed same/opp rate
                          # itself, BEFORE the reliability-weighted blend below --
                          # confirmed a real, severe bug without this: a thin
                          # same/opp-hand sample (e.g. 1-for-1) hits exactly 0% or
                          # 100%, and odds() clips that to ~1e-6/1e6, producing a
                          # same_hand_mult/opp_hand_mult up to ~146,000x. The
                          # PLATOON_STABILIZATION_PA reliability weight alone does
                          # NOT bound this: reliability correctly gives a thin
                          # sample a tiny BLEND weight, but a tiny weight times an
                          # unbounded raw value can still be enormous (e.g.
                          # 0.00136 x 20,000,000 = 27,200) -- the raw own_same_mult/
                          # own_opp_mult need their own bounded prior, independent
                          # of the blend weighting.


def league_platoon_multiplier(pa: pd.DataFrame, outcome: str, target_season: int) -> float:
    """odds(opposite-hand rate) / odds(same-hand rate), league-wide, using
    only seasons strictly before target_season (no leakage)."""
    prior = pa[pa["season"] < target_season]
    ref = prior if len(prior) else pa[pa["season"] == target_season]
    same = ref[ref["stand"] == ref["p_throws"]]
    opp = ref[ref["stand"] != ref["p_throws"]]
    same_rate = (same["outcome"] == outcome).mean()
    opp_rate = (opp["outcome"] == outcome).mean()
    return odds(opp_rate) / odds(same_rate)


def _season_split_rates(pa: pd.DataFrame, player_col: str, outcome: str) -> pd.DataFrame:
    """One row per (player, season): same-hand and opposite-hand PA count +
    event count that season. Vectorized (groupby+agg), not a Python loop --
    same reasoning as true_talent.py: a per-row loop over 600K+ PAs across
    ~2000 players would be slow at MLB scale (confirmed: the original
    per-player Python loop here took several minutes; this is the fix)."""
    tmp = pa[[player_col, "season"]].copy()
    tmp["is_same_hand"] = pa["stand"] == pa["p_throws"]
    tmp["is_outcome"] = (pa["outcome"] == outcome).astype(int)
    tmp["same_pa"] = tmp["is_same_hand"].astype(int)
    tmp["same_events"] = tmp["is_same_hand"] * tmp["is_outcome"]
    tmp["opp_pa"] = (~tmp["is_same_hand"]).astype(int)
    tmp["opp_events"] = (~tmp["is_same_hand"]) * tmp["is_outcome"]
    return tmp.groupby([player_col, "season"]).agg(
        same_pa=("same_pa", "sum"), same_events=("same_events", "sum"),
        opp_pa=("opp_pa", "sum"), opp_events=("opp_events", "sum"),
    ).reset_index()


def _player_split_history(season_splits: pd.DataFrame, player_col: str, target_season: int) -> pd.DataFrame:
    """Marcel-weighted (5/4/3) sum of the 3 prior seasons' same/opposite-hand
    PA and event counts, per player, for `target_season` -- vectorized via a
    pivot + weighted sum rather than a per-player Python loop."""
    prior_seasons = [target_season - 1 - i for i in range(len(MARCEL_WEIGHTS))]
    sub = season_splits[season_splits["season"].isin(prior_seasons)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[player_col, "same_pa", "same_events", "opp_pa", "opp_events"])

    weight_map = dict(zip(prior_seasons, MARCEL_WEIGHTS))
    sub["w"] = sub["season"].map(weight_map)
    for col in ("same_pa", "same_events", "opp_pa", "opp_events"):
        sub[col] = sub[col] * sub["w"]

    hist = sub.groupby(player_col).agg(
        same_pa=("same_pa", "sum"), same_events=("same_events", "sum"),
        opp_pa=("opp_pa", "sum"), opp_events=("opp_events", "sum"),
    ).reset_index()
    return hist[(hist["same_pa"] > 0) | (hist["opp_pa"] > 0)]


def _season_league_split_totals(pa: pd.DataFrame, player_col: str, outcome: str) -> pd.DataFrame:
    """One row per (season, player_hand): same/opposite-hand PA + event
    counts pooled across every PA where this side's own handedness (stand
    for batter, p_throws for pitcher) matches player_hand. Used to build a
    HANDEDNESS-CONDITIONED league-level reference (see
    _league_split_reference) -- conditioning on the player's own hand
    matters (2026-08-04 follow-up): "same-hand" is a MAJORITY-exposure
    regime for a RHB (facing the ~70% of starters who are RHP) but a
    MINORITY-exposure regime for a LHB (facing the ~30% who are LHP), so
    pooling both together poisons the same-hand reference toward whichever
    handedness dominates the league's roster mix -- confirmed empirically:
    an unconditioned version of this reference specifically broke down on
    the same-hand side for home_run/walk, while opposite-hand (where the
    pooling distortion is smaller) behaved fine."""
    hand_col = "stand" if player_col == "batter" else "p_throws"
    is_same_hand = (pa["stand"] == pa["p_throws"]).to_numpy()
    is_outcome = (pa["outcome"] == outcome).astype(int).to_numpy()
    tmp = pd.DataFrame({
        "season": pa["season"].to_numpy(),
        "player_hand": pa[hand_col].to_numpy(),
        "same_pa": is_same_hand.astype(int),
        "same_events": is_same_hand.astype(int) * is_outcome,
        "opp_pa": (~is_same_hand).astype(int),
        "opp_events": (~is_same_hand).astype(int) * is_outcome,
    })
    return tmp.groupby(["season", "player_hand"]).agg(
        same_pa=("same_pa", "sum"), same_events=("same_events", "sum"),
        opp_pa=("opp_pa", "sum"), opp_events=("opp_events", "sum"),
    ).reset_index()


def _league_split_reference(season_league_totals: pd.DataFrame, target_season: int, player_hand: str) -> dict | None:
    """Marcel-weighted (5/4/3), walk-forward, HANDEDNESS-CONDITIONED league-
    level same/opp-vs-combined odds ratio for target_season -- built via the
    IDENTICAL 3-prior-season weighting window and odds-ratio-of-conditional-
    vs-combined-rate formula as an individual player's own_same_mult/
    own_opp_mult (see build_platoon_multipliers), restricted to players of
    the SAME predominant hand. This is what an exactly-average, fully-
    reliable player OF THIS HAND's own ratio would equal by construction --
    dividing a real player's own_same_mult/own_opp_mult by this reference
    therefore cancels the redundant population-level component exactly,
    rather than approximating its size with an assumed exponent (2026-08-04,
    follow-up to the 2026-08-03 fallback-only structural fix -- see that
    fix's own docstring below for why the fallback alone wasn't sufficient:
    it left the OBSERVED leg (own_same_mult/own_opp_mult) as a raw,
    unnormalized ratio that silently re-embeds the full shared effect once
    a player's reliability is high enough for the blend to be dominated by
    it). Returns None only in the same true-cold-start case
    _player_split_history already returns empty for (no season has any
    prior data yet) -- not expected to trigger in practice since league-
    pooled totals reach positive same_pa/opp_pa far before any individual
    player's own history does."""
    prior_seasons = [target_season - 1 - i for i in range(len(MARCEL_WEIGHTS))]
    sub = season_league_totals[
        season_league_totals["season"].isin(prior_seasons) & (season_league_totals["player_hand"] == player_hand)
    ]
    if sub.empty:
        return None
    weight_map = dict(zip(prior_seasons, MARCEL_WEIGHTS))
    w = sub["season"].map(weight_map)
    same_pa = (sub["same_pa"] * w).sum()
    same_events = (sub["same_events"] * w).sum()
    opp_pa = (sub["opp_pa"] * w).sum()
    opp_events = (sub["opp_events"] * w).sum()
    if same_pa <= 0 or opp_pa <= 0:
        return None
    combined_rate = (same_events + opp_events) / (same_pa + opp_pa)
    same_rate = same_events / same_pa
    opp_rate = opp_events / opp_pa
    return {
        "same_mult": odds(same_rate) / odds(combined_rate),
        "opp_mult": odds(opp_rate) / odds(combined_rate),
    }


def _predominant_hand(pa: pd.DataFrame, player_col: str) -> pd.Series:
    """Each player's modal stand (batter) / p_throws (pitcher) -- a minimal,
    local equivalent of validate_game_simulator.predominant_hand (not
    imported directly: that module imports build_platoon_multipliers FROM
    this one, so the reverse import would be circular). Real switch-hitters
    exist (~8.7% of batters), but their same-hand PAs are rare by
    construction (they switch specifically to face the pitcher from their
    off side), so picking a single modal side to select which measured
    shared-term cell applies to their (small) same-hand weight is accurate
    for the vast majority and low-consequence for the rest."""
    hand_col = "stand" if player_col == "batter" else "p_throws"
    return pa.groupby(player_col)[hand_col].agg(lambda s: s.mode().iloc[0])


def _measured_shared_term_for(season: int, outcome: str, stand: str, p_throws: str) -> float | None:
    """The measured total (both-sides-combined) shared displacement for this
    (season, outcome, batter-hand, pitcher-hand) cell -- see
    measure_platoon_shared_term.py. Returns this side's leg (sqrt of the
    total, split symmetrically -- no evidence favoring attributing more of
    the SHARED component to one side over the other), or None if this
    exact cell wasn't measured (missing season/outcome, or too few real PAs
    that season -- see that script's own n>=200 floor), so the caller can
    fall back to the exponent-based value."""
    table = _load_measured_shared_term()
    cell = table.get(str(season), {}).get(outcome, {}).get(f"{stand}{p_throws}")
    if cell is None or cell != cell:  # NaN check without importing math/numpy just for this
        return None
    return cell ** 0.5


def _shared_term_leg(player_col: str, player_hand: str, is_same_hand: bool, outcome: str, season: int,
                      exponent_fallback: float) -> float:
    """Maps (this side's own hand, same-or-opp) to the measured (stand,
    p_throws) cell, per the mechanical correspondence: a same-hand matchup
    for a LHB/LHP is the (L,L) cell; for a RHB/RHP it's the (R,R) cell. An
    opp-hand matchup for a LHB or a RHP is the (L,R) cell (LHB facing RHP,
    or RHP facing LHB -- the same cell, from either side's perspective);
    for a RHB or a LHP it's the (R,L) cell. Falls back to the exponent-
    based leg for any cell the measured table doesn't cover."""
    if is_same_hand:
        stand = p_throws = player_hand
    elif (player_col == "batter" and player_hand == "L") or (player_col == "pitcher" and player_hand == "R"):
        stand, p_throws = "L", "R"
    else:
        stand, p_throws = "R", "L"
    measured = _measured_shared_term_for(season, outcome, stand, p_throws)
    return measured if measured is not None else exponent_fallback


def build_platoon_multipliers(pa: pd.DataFrame, player_col: str, outcome: str) -> pd.DataFrame:
    """One row per (player, season): a same-hand and an opposite-hand
    multiplier to apply on top of that player's overall true-talent rate
    for `outcome` (from true_talent.py), walk-forward safe.

    Structural double-count fix, in three parts (2026-08-03 fallback fix,
    2026-08-04 observed-leg + measured-shared-term follow-ups -- all from
    the same external review thread). `league_mult` (from
    league_platoon_multiplier) is the FULL population-level same-vs-
    opposite-hand effect -- it already reflects BOTH the batter side's and
    the pitcher side's contribution jointly (a raw PA-level rate ratio
    can't separate "how much of this is batters" from "how much is
    pitchers"). This function is called ONCE per side (batter, then
    separately pitcher), and BOTH sides' multipliers are later multiplied
    together in game_simulator.combine_matchup_distribution for every real
    matchup -- so the shared population component must be applied exactly
    ONCE across the two sides combined, or it gets double-counted.

    Part 1 (shared term, always applied, per-cell measured): `same_shared`/
    `opp_shared` below carry this side's leg of the MEASURED total
    displacement for this specific (season, outcome, batter-hand, pitcher-
    hand) cell -- see measure_platoon_shared_term.py. This replaced an
    earlier attempt at a single fixed exponent (league_mult**-0.25 /
    **+0.25, split symmetrically to sum to the ~1.09-1.22x a decomposed
    regression found real data supports, vs. the ~2.0x the pre-2026-08-03
    code implicitly applied): an exponent sweep found no single value fit
    both home_run and strikeout, and the measured cells confirmed why --
    e.g. strikeout's LHB-vs-LHP cell shows an real +11.6% displacement
    while its RHB-vs-RHP cell shows barely +0.7%, plausibly a real
    selection effect (same-handed relief matchups are more heavily
    curated than the everyday RHB-vs-RHP case) that a symmetric exponent
    could never represent. Falls back to the exponent-based value (Part 1
    of the ORIGINAL 2026-08-03 fix) for any cell the measured table doesn't
    cover -- a new/unmeasured outcome, or the true cold-start season.

    Part 2 (deviation, reliability-shrunk toward 1.0): the initial
    2026-08-03 fix left `own_same_mult`/`own_opp_mult` (each player's own
    measured ratio, relative to their own combined rate) as a raw,
    unnormalized value -- documented at the time as "a real, non-redundant
    per-player signal, not part of the shared/double-counted population
    term." That was wrong: own_same_mult mechanically contains the SAME
    shared population effect (since a player's own same-hand PAs are drawn
    from a population that already reflects the league-wide platoon
    swing), so a fully-reliable player's raw own_same_mult still asserts
    close to the FULL effect a second time, on top of the always-applied
    shared term above -- confirmed by a held-out 2025 bucketed check that
    showed the residual concentrated in HIGH-reliability matchups (the
    opposite of where a leftover default-only bug would live). A first
    attempt at fixing this by pooling ALL players' own history into one
    league_ref regardless of handedness also broke down specifically on
    the same-hand side (own_same_mult for a RHB, whose same-hand PAs are
    a MAJORITY-exposure regime facing the ~70% of starters who are RHP, is
    a mechanically different quantity than for a LHB, whose same-hand PAs
    are a MINORITY-exposure regime) -- fixed below by conditioning
    `_league_split_reference` on the player's own predominant hand.

    Fix: own_same_mult/own_opp_mult are divided by a HANDEDNESS-CONDITIONED
    `_league_split_reference` -- a league-pooled ratio, built via the
    IDENTICAL Marcel-weighted 3-season window and odds-ratio-of-
    conditional-vs-combined-rate formula as the player-level calculation,
    computed separately for each predominant hand -- turning them into a
    PURE deviation from an exactly-average player OF THE SAME HAND (~1.0
    for a league-average same-handed player at ANY reliability, not just
    zero). That deviation is then shrunk toward 1.0 (neutral -- "no
    different from a league-average player of this hand"), NOT toward
    same_shared/opp_shared, since the shared term is already applied
    unconditionally above; blending the deviation toward it here too would
    reintroduce a smaller version of the same double-count."""
    season_splits = _season_split_rates(pa, player_col, outcome)  # computed once, reused per target season
    season_league_totals = _season_league_split_totals(pa, player_col, outcome)  # ditto
    player_hand = _predominant_hand(pa, player_col)
    out_frames = []
    for season in sorted(pa["season"].unique()):
        league_mult = league_platoon_multiplier(pa, outcome, season)
        same_exponent_fallback = 1.0 / (league_mult ** PLATOON_LEAGUE_DEFAULT_EXPONENT)
        opp_exponent_fallback = league_mult ** PLATOON_LEAGUE_DEFAULT_EXPONENT

        hist = _player_split_history(season_splits, player_col, season)
        if hist.empty:
            out_frames.append(pd.DataFrame(columns=[player_col, "season", "same_hand_mult", "opp_hand_mult"]))
            continue

        hist["same_reliability"] = hist["same_pa"] / (hist["same_pa"] + PLATOON_STABILIZATION_PA)
        hist["opp_reliability"] = hist["opp_pa"] / (hist["opp_pa"] + PLATOON_STABILIZATION_PA)

        # player's own observed multiplier, relative to their OWN combined rate
        # (a pure "shape" adjustment -- their overall level is already handled
        # by true_talent.py, this only captures how much MORE/LESS platoon-split
        # they show than a league-average player, not their overall talent).
        # own_same_rate/own_opp_rate are regularized (SPLIT_RATE_PRIOR_PA pseudo-PA
        # of the player's own combined rate) BEFORE taking odds -- prevents a thin
        # sample's exact 0%/100% observed rate from producing an unbounded ratio
        # via odds()'s clipping (see SPLIT_RATE_PRIOR_PA's docstring above).
        own_combined_rate = (hist["same_events"] + hist["opp_events"]) / (hist["same_pa"] + hist["opp_pa"])
        own_same_rate = (
            (hist["same_events"] + SPLIT_RATE_PRIOR_PA * own_combined_rate) / (hist["same_pa"] + SPLIT_RATE_PRIOR_PA)
        )
        own_opp_rate = (
            (hist["opp_events"] + SPLIT_RATE_PRIOR_PA * own_combined_rate) / (hist["opp_pa"] + SPLIT_RATE_PRIOR_PA)
        )
        own_same_mult = odds(own_same_rate) / odds(own_combined_rate)
        own_opp_mult = odds(own_opp_rate) / odds(own_combined_rate)

        # handedness-conditioned deviation: look up each player's own
        # predominant hand, then divide by the league reference built ONLY
        # from players of that same hand (see _league_split_reference).
        hist_hand = hist[player_col].map(player_hand)
        ref_cache = {
            h: _league_split_reference(season_league_totals, season, h) for h in hist_hand.dropna().unique()
        }

        def _ref_lookup(h, key):
            r = ref_cache.get(h)
            return r[key] if r is not None else float("nan")

        league_same_ref = hist_hand.map(lambda h: _ref_lookup(h, "same_mult"))
        league_opp_ref = hist_hand.map(lambda h: _ref_lookup(h, "opp_mult"))
        own_same_dev = own_same_mult / league_same_ref
        own_opp_dev = own_opp_mult / league_opp_ref

        hist["same_hand_dev"] = (
            hist["same_reliability"] * own_same_dev.fillna(1.0) + (1 - hist["same_reliability"]) * 1.0
        )
        hist["opp_hand_dev"] = (
            hist["opp_reliability"] * own_opp_dev.fillna(1.0) + (1 - hist["opp_reliability"]) * 1.0
        )

        # measured shared term (falls back to the exponent-based value for
        # any cell the measurement script hasn't covered -- see
        # _shared_term_leg/measure_platoon_shared_term.py).
        same_shared = hist_hand.map(
            lambda h: _shared_term_leg(player_col, h, True, outcome, season, same_exponent_fallback)
            if pd.notna(h) else same_exponent_fallback
        )
        opp_shared = hist_hand.map(
            lambda h: _shared_term_leg(player_col, h, False, outcome, season, opp_exponent_fallback)
            if pd.notna(h) else opp_exponent_fallback
        )

        hist["same_hand_mult"] = same_shared * hist["same_hand_dev"]
        hist["opp_hand_mult"] = opp_shared * hist["opp_hand_dev"]
        hist["season"] = season
        out_frames.append(hist[[player_col, "season", "same_hand_mult", "opp_hand_mult"]])
    return pd.concat(out_frames, ignore_index=True)


def apply_platoon_adjustment(overall_rate: float, is_same_hand: bool, same_hand_mult: float, opp_hand_mult: float) -> float:
    mult = same_hand_mult if is_same_hand else opp_hand_mult
    return prob_from_odds(odds(overall_rate) * mult)


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")

    print("=== league-wide platoon multipliers (as of 2025, prior-seasons only) ===")
    for outcome in ["strikeout", "home_run", "walk"]:
        mult = league_platoon_multiplier(pa[pa["season"] < 2025], outcome, 2025)
        print(f"{outcome}: opposite/same-hand odds ratio = {mult:.4f}")

    print("\n=== batter strikeout platoon multipliers: sample players ===")
    result = build_platoon_multipliers(pa, "batter", "strikeout")
    print(f"n player-seasons with usable multipliers: {len(result)}")
    print(result.dropna().sample(8, random_state=0).to_string(index=False))
