"""Walk-forward-safe true-talent rate estimation for PITCH-LEVEL sub-decisions,
conditioned on a count-leverage bucket -- mirrors true_talent.py's Marcel-style
structure (recency-weighted 3-season prior, regressed to league average via a
stabilization constant) one granularity level down: per PITCH within a count
bucket, rather than per PA.

One full pitch decomposes into a small decision tree over the 6 PITCH_RESULT
categories (see build_pitch_table.py):
  1. swing?           (population: all pitches, bucketed by swing_count_group
                        -- this decision specifically needs the 3-0 count split
                        out from the other "hitter ahead" counts; see
                        pitch_groups.py's docstring for the real swing-rate gap)
     if NOT swing:
       2. hit_by_pitch? (population: takes, bucketed by count_group)
          if not hbp:
       3. called_strike? (population: takes minus HBP, bucketed by count_group)
     if swing:
       4. whiff?        (population: swings, bucketed by count_group)
          if not whiff:
       5. hit_into_play? (population: swings minus whiffs -- i.e. contact,
                          bucketed by count_group) -- the complement is foul.

Each of these 5 binary decisions gets its OWN independently-estimated,
Bayesian-shrunk rate per (player, bucket), same modularity as true_talent.py
estimating each outcome category independently.

Deliberately simplified relative to true_talent.py for this first phase: NO
age adjustment (true_talent.py's age-adjustment sign overrides took this
entire project many sessions of leakage-free testing to get right per
outcome category -- adding that now, before even confirming the pitch-level
approach is worth pursuing at all, would be solving a problem before knowing
if it matters). Stabilization constants below are starting points validated
via this file's own leakage-free K-sweep (see __main__), not carried over
from STABILIZATION_PA_BATTER/_PITCHER, since pitch-count denominators and
these specific binary decisions aren't the same quantities true_talent.py's
constants were fit for.
"""

import numpy as np
import pandas as pd

from src.models.matchup import odds
from src.models.pitch_groups import count_group, swing_count_group

DECISIONS = ("swing", "whiff", "inplay", "called_strike", "hbp")
CLIP_MIN, CLIP_MAX = 1e-4, 1 - 1e-4

MARCEL_WEIGHTS = [5, 4, 3]  # most recent season first, same as true_talent.py

# Stabilization constants (in PITCHES, the relevant population for that
# decision -- e.g. "swing" is per all-pitches, "whiff" is per-swing).
# STARTING VALUES -- see __main__'s leakage-free K-sweep for validation.
STABILIZATION_PITCHES_BATTER = {
    "swing": 200,
    "whiff": 200,
    "inplay": 200,
    "called_strike": 200,
    "hbp": 2000,  # rare event, expect heavy shrinkage
}
STABILIZATION_PITCHES_PITCHER = {
    "swing": 250,
    "whiff": 250,
    "inplay": 250,
    "called_strike": 250,
    "hbp": 3000,
}

# Which count-bucket function each decision uses, and which subset of pitches
# forms its population + what its binary target column is.
DECISION_BUCKET_FN = {
    "swing": swing_count_group,
    "whiff": count_group,
    "inplay": count_group,
    "called_strike": count_group,
    "hbp": count_group,
}


def _stabilization_dict(player_col: str) -> dict:
    return STABILIZATION_PITCHES_BATTER if player_col == "batter" else STABILIZATION_PITCHES_PITCHER


def _clip(p):
    return min(max(p, CLIP_MIN), CLIP_MAX)


def resolve_terminal_probs(bucket_rates: dict) -> dict:
    """bucket_rates: {decision: {bucket: rate}}, already filled in with a
    league-average fallback for any bucket the input rates lack. Returns
    EXACT {"strikeout", "walk", "hit_by_pitch", "hit_into_play"} absorption
    probabilities for one simulated PA starting at (0, 0), via forward
    dynamic programming over the 12 real (balls, strikes) states -- not
    Monte Carlo simulation, since this state space is small enough to solve
    exactly. `bucket_rates` can be either one player's own rates (Phase 1's
    batter-only validation, where combining against an exactly-average
    "opponent" collapses to just that player's own rate) or a
    batter x pitcher x league ODDS-RATIO-COMBINED rate per bucket (Phase 2's
    real matchup usage, via matchup.py's combine_odds_ratio) -- this function
    has no batter/pitcher-specific logic of its own, it just consumes
    whatever bucket rate dict it's handed.

    The one real subtlety: a foul ball at 2 strikes is a genuine self-loop
    (count doesn't advance). Handled by renormalizing the OTHER 5 outcomes'
    probabilities to exclude the foul-no-op mass at that specific state,
    rather than modeling the self-loop directly -- the eventual distribution
    over "what happens once the count actually advances" is identical either
    way, and this keeps the state graph a proper DAG."""
    state_prob = {(0, 0): 1.0}
    terminal = {"strikeout": 0.0, "walk": 0.0, "hit_by_pitch": 0.0, "hit_into_play": 0.0}
    states_order = sorted([(b, s) for b in range(4) for s in range(3)], key=lambda x: x[0] + x[1])

    for (balls, strikes) in states_order:
        p_mass = state_prob.get((balls, strikes), 0.0)
        if p_mass <= 0:
            continue
        sb, cb = swing_count_group(balls, strikes), count_group(balls, strikes)
        p_swing = _clip(bucket_rates["swing"].get(sb, 0.5))
        p_hbp_t = _clip(bucket_rates["hbp"].get(cb, 0.003))
        p_cs_t = _clip(bucket_rates["called_strike"].get(cb, 0.5))
        p_whiff_s = _clip(bucket_rates["whiff"].get(cb, 0.2))
        p_inplay_s = _clip(bucket_rates["inplay"].get(cb, 0.5))

        p_take = 1 - p_swing
        p_ball = p_take * (1 - p_hbp_t) * (1 - p_cs_t)
        p_called_strike = p_take * (1 - p_hbp_t) * p_cs_t
        p_hbp = p_take * p_hbp_t
        p_whiff = p_swing * p_whiff_s
        p_inplay = p_swing * (1 - p_whiff_s) * p_inplay_s
        p_foul = p_swing * (1 - p_whiff_s) * (1 - p_inplay_s)

        if strikes == 2:
            denom = max(1 - p_foul, 1e-9)
            p_ball, p_called_strike, p_hbp, p_whiff, p_inplay = (
                p_ball / denom, p_called_strike / denom, p_hbp / denom, p_whiff / denom, p_inplay / denom
            )
            p_foul = 0.0

        terminal["hit_by_pitch"] += p_mass * p_hbp
        terminal["hit_into_play"] += p_mass * p_inplay

        nb = balls + 1
        if nb == 4:
            terminal["walk"] += p_mass * p_ball
        else:
            state_prob[(nb, strikes)] = state_prob.get((nb, strikes), 0.0) + p_mass * p_ball

        p_strike_total = p_called_strike + p_whiff
        ns = strikes + 1
        if ns == 3:
            terminal["strikeout"] += p_mass * p_strike_total
        else:
            state_prob[(balls, ns)] = state_prob.get((balls, ns), 0.0) + p_mass * p_strike_total

        if strikes < 2 and p_foul > 0:
            state_prob[(balls, strikes + 1)] = state_prob.get((balls, strikes + 1), 0.0) + p_mass * p_foul

    return terminal


def _decision_population(pitches: pd.DataFrame, decision: str) -> pd.DataFrame:
    """Filters `pitches` to the population relevant for `decision`, and adds
    an `is_target`/`bucket` column pair ready for rate estimation."""
    df = pitches
    if decision == "swing":
        pop = df.copy()
        pop["is_target"] = pop["is_swing"]
    elif decision == "whiff":
        pop = df[df["is_swing"]].copy()
        pop["is_target"] = pop["result"] == "swinging_strike"
    elif decision == "inplay":
        pop = df[df["is_swing"] & (df["result"] != "swinging_strike")].copy()  # contact only
        pop["is_target"] = pop["result"] == "hit_into_play"
    elif decision == "called_strike":
        pop = df[(~df["is_swing"]) & (df["result"] != "hit_by_pitch")].copy()
        pop["is_target"] = pop["result"] == "called_strike"
    elif decision == "hbp":
        pop = df[~df["is_swing"]].copy()
        pop["is_target"] = pop["result"] == "hit_by_pitch"
    else:
        raise ValueError(f"unknown decision: {decision}")

    bucket_fn = DECISION_BUCKET_FN[decision]
    pop["bucket"] = [bucket_fn(b, s) for b, s in zip(pop["balls"], pop["strikes"])]
    return pop


def _season_bucket_rates(pop: pd.DataFrame, player_col: str) -> pd.DataFrame:
    """One row per (player, season, bucket): pitch count and target rate."""
    tmp = pop[[player_col, "season", "bucket", "is_target"]].copy()
    tmp["is_target"] = tmp["is_target"].astype(int)
    out = tmp.groupby([player_col, "season", "bucket"]).agg(
        n=("is_target", "size"), events=("is_target", "sum")
    ).reset_index()
    out["rate"] = out["events"] / out["n"]
    return out


def build_preseason_bucket_rates(pitches: pd.DataFrame, player_col: str, decision: str,
                                  target_season: int) -> tuple[pd.DataFrame, dict]:
    """The Marcel-style weighted, regressed prior for `target_season`, PER
    COUNT BUCKET, built only from seasons strictly before it (no leakage).
    Mirrors true_talent.py's build_preseason_priors, generalized across
    buckets and a generic binary target instead of a fixed outcome column.

    Returns (df with columns [player_col, 'bucket', 'preseason_rate',
    'prior_weight_pitches'], {bucket: league_rate})."""
    pop = _decision_population(pitches, decision)
    prior_seasons = pop[pop["season"] < target_season]
    empty_cols = [player_col, "bucket", "preseason_rate", "prior_weight_pitches"]

    if prior_seasons.empty:
        # true cold start -- fall back to this season's own league rate per
        # bucket, zero individual prior weight (same convention as
        # true_talent.py's own cold-start handling).
        cur = pop[pop["season"] == target_season]
        league_rates = cur.groupby("bucket")["is_target"].mean().to_dict()
        return pd.DataFrame(columns=empty_cols), league_rates

    league_rates = prior_seasons.groupby("bucket")["is_target"].mean().to_dict()
    season_rates = _season_bucket_rates(prior_seasons, player_col)
    K = _stabilization_dict(player_col)[decision]

    priors = []
    for (player, bucket), g in season_rates.groupby([player_col, "bucket"]):
        g = g.set_index("season")
        num, den, raw_n = 0.0, 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                num += w * g.loc[s, "events"]
                den += w * g.loc[s, "n"]
                raw_n += g.loc[s, "n"]
        if den == 0:
            continue
        league_rate = league_rates.get(bucket, np.nan)
        raw_rate = num / den
        reliability = raw_n / (raw_n + K)  # raw (unweighted) pitch count -- see true_talent.py's
                                             # task-#53 fix docstring for why this must be raw, not
                                             # the MARCEL_WEIGHTS-weighted sum
        preseason_rate = reliability * raw_rate + (1 - reliability) * league_rate
        priors.append({
            player_col: player, "bucket": bucket, "preseason_rate": preseason_rate,
            "prior_weight_pitches": min(raw_n, K),
        })
    priors_df = pd.DataFrame(priors)
    if priors_df.empty:
        return pd.DataFrame(columns=empty_cols), league_rates
    return priors_df[empty_cols], league_rates


def build_bucket_rate_snapshot(pitches: pd.DataFrame, player_col: str, target_season: int) -> tuple[dict, dict]:
    """Season-level snapshot (preseason-only, walk-forward-safe -- same
    convention as sprint_speed.py's/defense_factor.py's own season snapshots,
    not the finer per-pitch in-season blend build_pregame_bucket_rates
    offers) across all 5 pitch-level decisions at once. Returns
    (player_rates, league_rates):
      player_rates: {player_id: {decision: {bucket: rate}}}
      league_rates: {decision: {bucket: rate}}

    Built for the Phase 2 wholesale strikeout/walk/hit_by_pitch mechanism
    swap in game_simulator.py's GameSimulator._pa_outcome -- that mechanism
    was reverted (2026-07-22) after a full-stack A/B regression (SU
    59.3%->55.6%), traced to the composed rate being under-dispersed
    relative to true_talent.py's own estimate (see build_composed_rates'
    docstring for the diagnostic and the regression-blend approach that
    replaced it). Currently unused, kept for a possible future revisit of
    the wholesale-swap approach with properly tuned stabilization constants."""
    player_rates: dict = {}
    league_rates: dict = {}
    for decision in DECISIONS:
        priors_df, lg = build_preseason_bucket_rates(pitches, player_col, decision, target_season)
        league_rates[decision] = lg
        for r in priors_df.itertuples():
            player = getattr(r, player_col)
            player_rates.setdefault(player, {}).setdefault(decision, {})[r.bucket] = r.preseason_rate
    return player_rates, league_rates


def build_composed_rates(pitches: pd.DataFrame, player_col: str, target_season: int) -> dict:
    """Each player's OWN composed strikeout/walk/hit_by_pitch probability
    (via resolve_terminal_probs, batter-only-vs-league-average-pitcher or
    pitcher-only-vs-league-average-batter -- an exactly-average "opponent"
    collapses the odds-ratio combine to just this player's own rate, so no
    matchup combine is actually needed here), preseason-only, walk-forward-
    safe. Returns {player_id: {"strikeout": p, "walk": p, "hit_by_pitch": p}}.

    This is the regression-blend INPUT signal for pitch_blend_multiplier
    below -- diagnosed (2026-07-22, after the wholesale mechanism-swap
    version of this factor failed its full-stack A/B) as under-dispersed
    relative to true_talent.py's own rate for the SAME real matchups
    (strikeout composed std 0.035 vs true_talent's 0.051 vs real 0.063,
    across 3000 real batter-pitcher pairs) -- a real, calibratable signal
    compressed toward the mean, not noise. Confirmed a simple linear
    rescale brings its own MAE to near-parity with true_talent.py's (e.g.
    strikeout: raw MAE 0.0338 -> rescaled 0.0329 vs true_talent's 0.0324).
    Rather than a standalone rescale (this project's isotonic-calibration
    revert is a cautionary precedent for "fix the marginal distribution in
    isolation" approaches breaking joint coherence), this signal is blended
    with true_talent.py's existing rate via a fitted multivariate regression
    -- the SAME architecture as hr_share_multiplier/groundball_single_
    multiplier in expected_stats.py (predicted = intercept + coef_existing*
    existing + coef_new*new_signal), the pattern every factor that's
    actually stuck in this project uses, rather than the wholesale-swap
    approach that just failed."""
    decision_data = {}
    for decision in DECISIONS:
        priors_df, league_rates = build_preseason_bucket_rates(pitches, player_col, decision, target_season)
        decision_data[decision] = (priors_df, league_rates)

    players = set()
    for decision in DECISIONS:
        players |= set(decision_data[decision][0][player_col].unique())

    out = {}
    for player in players:
        bucket_rates = {}
        for decision in DECISIONS:
            priors_df, league_rates = decision_data[decision]
            player_rows = priors_df[priors_df[player_col] == player]
            rates = dict(zip(player_rows["bucket"], player_rows["preseason_rate"]))
            for bucket, lr in league_rates.items():
                rates.setdefault(bucket, lr)
            bucket_rates[decision] = rates
        terminal = resolve_terminal_probs(bucket_rates)
        out[player] = {"strikeout": terminal["strikeout"], "walk": terminal["walk"],
                        "hit_by_pitch": terminal["hit_by_pitch"]}
    return out


# Regression-blend coefficients (2026-07-22): leakage-free fit of next-season
# real outcome rate ~ intercept + coef_existing*this-season existing
# true_talent rate + coef_composed*this-season composed pitch-level rate
# (both preseason-only, n=670-687 combined player-season pairs per side).
# Batter and pitcher get SEPARATE coefficients (same convention as
# true_talent.py's own separate batter/pitcher stabilization constants).
#
# ONLY walk gets this treatment. The SAME leakage-free incremental-R^2 test
# run for all three no-contact outcomes, both sides, found:
#   batter strikeout:    R^2 0.5117 -> 0.5120 (existing-only -> +composed)
#   batter hit_by_pitch: R^2 0.2282 -> 0.2287
#   pitcher strikeout:   R^2 0.2843 -> 0.2844
#   pitcher hit_by_pitch: R^2 0.0927 -> 0.0936
# -- essentially ZERO incremental value (<0.001 R^2) in all 4 cases once
# true_talent's existing rate is already in the model. Only walk showed a
# real, meaningful jump:
#   batter walk:  R^2 0.3151 -> 0.3613 (+0.046)
#   pitcher walk: R^2 0.1441 -> 0.1498 (+0.0057, smaller but real)
# On the batter side specifically, the composed signal's own coefficient
# (0.843) dwarfs the existing rate's (0.055) -- the pitch-level swing/
# discipline signal predicts a batter's FUTURE walk rate far better than
# their own past (noisy, needs-4-real-balls) observed walk rate does.
# Matches this project's "no supporting evidence, don't guess" standard
# (e.g. AGE_ADJ_SIGN_OVERRIDE's untested categories) -- strikeout/hit_by_pitch
# are NOT given this treatment.
#
# DEPLOYED AND REVERTED (2026-07-22): despite sitting fully fit and unused
# for a full session, this was finally wired into validate_game_simulator.py's
# build_profile (applied to BOTH batter and pitcher rates["walk"]) and given
# a proper full-stack A/B (n=597, baseline at the time: total MAE 3.409,
# margin MAE 3.445, SU 60.5%). Result: total MAE 3.409->3.377 (better),
# margin MAE 3.445->3.474 (worse), **SU 60.5%->59.1% (-1.4pp)** -- a clear
# net regression on the primary decision metric, essentially the same shape
# as whiff-rate-on-swings' original result (-1.3pp SU) despite this being a
# WALK signal, not a strikeout signal, and despite the real, meaningful
# leakage-free incremental R^2 gain above. REVERTED cleanly from
# validate_game_simulator.py (import, build_profile's pregame_pitch_walk_
# composed/player_col params, the season-level snapshot building, both
# batter_profile/pitcher_profile call sites) -- confirmed via source
# inspection matching the pre-deployment state. This function and its
# coefficients remain here as a documented-but-unused artifact, same
# treatment as every other investigated-and-rejected idea this session --
# this closes out the "designed/fit but never deployed" status from earlier
# in the session with an actual, honest full-stack answer.
PITCH_WALK_INTERCEPT = {"batter": -0.01326, "pitcher": 0.01458}
PITCH_WALK_COEF_EXISTING = {"batter": 0.05458, "pitcher": 0.45357}
PITCH_WALK_COEF_COMPOSED = {"batter": 0.84306, "pitcher": 0.28712}

# Real walk-rate range confirmed on 2025 data: existing 0.033-0.170 (batter),
# 0.033-0.147 (pitcher); composed 0.049-0.194 (batter), 0.070-0.152
# (pitcher). [0.02, 0.25] is comfortably wide for normal use, tight enough to
# catch a cold-start extreme -- same "clip both sides before the odds-ratio"
# discipline as hr_share_multiplier/groundball_single_multiplier in
# expected_stats.py, this project's repeated fix for the same failure mode
# (platoon 146,611x, weather 73x, HR-share 13.8x).
PITCH_WALK_CLIP_MIN, PITCH_WALK_CLIP_MAX = 0.02, 0.25


def pitch_walk_multiplier(existing_walk_rate: float, composed_walk_rate: float, player_col: str) -> float:
    """The odds-ratio multiplier to apply to "walk" ONLY, blending this
    player's existing true-talent walk rate with their composed pitch-level
    walk-rate signal (see build_composed_rates) via the fitted regression
    above -- same architecture as hr_share_multiplier/groundball_single_
    multiplier in expected_stats.py (the pattern every factor that's stuck
    in this project uses), not a wholesale replacement of the existing rate
    (the approach that just failed its full-stack A/B, see
    build_bucket_rate_snapshot's docstring)."""
    existing = np.clip(existing_walk_rate, PITCH_WALK_CLIP_MIN, PITCH_WALK_CLIP_MAX)
    predicted = (
        PITCH_WALK_INTERCEPT[player_col]
        + PITCH_WALK_COEF_EXISTING[player_col] * existing
        + PITCH_WALK_COEF_COMPOSED[player_col] * composed_walk_rate
    )
    predicted = np.clip(predicted, PITCH_WALK_CLIP_MIN, PITCH_WALK_CLIP_MAX)
    return odds(predicted) / odds(existing)


def build_pregame_bucket_rates(pitches: pd.DataFrame, player_col: str, decision: str) -> pd.DataFrame:
    """In-season-blended analog of build_preseason_bucket_rates -- mirrors
    true_talent.py's build_pregame_rates exactly (preseason prior, capped at
    K pseudo-pitches, Bayesian-blended with season-to-date actual counts),
    generalized across count buckets the same way build_preseason_bucket_rates
    already is. One row per pitch in `decision`'s population: the
    walk-forward-safe (no leakage) pregame rate for that player's bucket as of
    just before that specific pitch."""
    pop = _decision_population(pitches, decision)
    pop = pop.sort_values([player_col, "season", "game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()
    pop["is_target"] = pop["is_target"].astype(int)
    K = _stabilization_dict(player_col)[decision]

    out_frames = []
    for season, sdf in pop.groupby("season"):
        priors_df, league_rates = build_preseason_bucket_rates(pitches, player_col, decision, season)
        sdf = sdf.merge(priors_df, on=[player_col, "bucket"], how="left")
        sdf["preseason_rate"] = sdf["preseason_rate"].fillna(sdf["bucket"].map(league_rates))
        sdf["prior_weight_pitches"] = sdf["prior_weight_pitches"].fillna(K)

        grp = sdf.groupby([player_col, "bucket"])
        sdf["season_events_before"] = grp["is_target"].cumsum() - sdf["is_target"]
        sdf["season_pitches_before"] = grp.cumcount()

        num = sdf["prior_weight_pitches"] * sdf["preseason_rate"] + sdf["season_events_before"]
        den = sdf["prior_weight_pitches"] + sdf["season_pitches_before"]
        sdf["pregame_rate"] = num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result.drop(columns=["season_events_before", "season_pitches_before"])


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pitches = pd.read_parquet(DATA_PROCESSED / "pitch_table_2023_2026.parquet")

    print("=== leakage-free preseason-only YoY correlation, per decision/player/bucket ===")
    print("(preseason prior built from season < target vs. real target-season bucket rate)\n")
    seasons_available = sorted(pitches["season"].unique())
    target_seasons = [s for s in seasons_available if s - 1 in seasons_available]

    for player_col in ("batter", "pitcher"):
        for decision in ("swing", "whiff", "inplay", "called_strike", "hbp"):
            pop = _decision_population(pitches, decision)
            for target_season in target_seasons:
                priors_df, _ = build_preseason_bucket_rates(pitches, player_col, decision, target_season)
                if priors_df.empty:
                    continue
                actual = pop[pop["season"] == target_season]
                actual_rates = _season_bucket_rates(actual, player_col)[[player_col, "bucket", "n", "rate"]]
                merged = priors_df.merge(actual_rates, on=[player_col, "bucket"])
                for bucket, g in merged.groupby("bucket"):
                    g = g[g["n"] >= 30]
                    if len(g) < 20:
                        continue
                    corr = g["preseason_rate"].corr(g["rate"])
                    print(f"{player_col:8s} {decision:14s} {target_season} bucket={bucket:12s} "
                          f"n_players={len(g):4d}  corr={corr:.3f}")

    print("\n=== in-season-blended pregame_rate: pitch-level MAE vs. naive league rate ===")
    for decision in ("swing", "whiff", "inplay", "called_strike", "hbp"):
        result = build_pregame_bucket_rates(pitches, "batter", decision)
        test = result[result["season"].isin(target_seasons)]
        mae = (test["pregame_rate"] - test["is_target"]).abs().mean()
        naive_mae = (test["is_target"].mean() - test["is_target"]).abs().mean()
        print(f"batter {decision:14s} pitch-level MAE: {mae:.4f}  naive(pop rate) MAE: {naive_mae:.4f}")
