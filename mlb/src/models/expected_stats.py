"""Statcast expected-stats (xBA on batted balls) as a batter-side "contact
quality" correction -- validated first, per this project's established
discipline, before building anything.

The specific, decisive finding (2026-07-21, real 2023-2025 data, batters with
50+ balls in play/season): a batter's REAL batting-average-on-contact
("BACON" = hits / balls in play) correlates only 0.25-0.30 with their OWN
next-season real BACON, but their `estimated_ba_using_speedangle`
("xBACON" -- Statcast's per-batted-ball expected-hit-probability from exit
velocity + launch angle, averaged over the season) correlates 0.39 with next
season's real BACON -- clearly better, in both year-pairs tested. A
multivariate check (next-season BACON regressed on both this-season real
BACON and this-season xBACON together) confirmed this isn't just "xBACON is
slightly better" -- real BACON's own regression coefficient is ~0 once
xBACON is already in the model (R^2 barely moves, 0.152->0.153 and
0.149->0.151, adding real BACON on top of xBACON). Real outcome history adds
essentially no information beyond what contact quality already captures --
exactly the published sabermetric finding motivating expected stats in the
first place (a batted ball's actual outcome depends heavily on luck/defense
positioning; exit velocity + launch angle strip that out).

Scope, deliberately narrow and matching exactly what was validated: this
adjusts ONLY the aggregate hit-vs-out-on-contact rate (a single multiplier
applied equally to single/double/triple/home_run, i.e. "does this player's
contact turn into more or fewer hits than their recent outcome history alone
implies" -- NOT which specific hit type, which wasn't separately validated).
Batters only -- an earlier research pass this session found the analogous
pitcher-side expected-stat signal (CSW%) does NOT clearly beat outcome-based
K% (Baseball Prospectus found no significant FIP edge either), so this is not
extended to pitcher-allowed contact quality.

Applied directly to a batter's own `rates` dict at PROFILE-BUILD time (not
threaded through combine_matchup_distribution as a new factor-table param,
unlike park/weather/catcher factors) -- contact quality is a fixed, opponent-
independent property of the batter (like the Marcel-shrunk true-talent rate
itself), not something that varies by which specific pitcher/park/game this
PA happens in, so it belongs with the OTHER true-talent-level corrections
that already get baked into `rates` before matchup combination.

SECOND LAYER (2026-07-21), pushed as far as the data actually supports, no
further: within a batter's hits, does contact quality also predict WHICH
TYPE of hit better than outcome history alone? Tested three splits leakage-
free (next-season real share regressed on this-season real share AND a
candidate Statcast metric together):
  - home_run share among hits vs. BARREL RATE (launch_speed_angle==6, Statcast's
    own "well-struck" classification): barrel rate's regression coefficient
    (0.51-0.55) clearly exceeds real outcome share's own (0.31-0.42), R^2
    improves adding it (0.40->0.43, 0.41->0.44) -- a real, worthwhile signal,
    consistent with barrel rate being specifically THE Statcast metric
    designed to capture home-run-quality contact. BUILT (see
    build_pregame_barrel_rate/hr_share_multiplier below).
  - double+triple share among hits vs. xSLG (estimated_slg_using_speedangle):
    R^2 is barely above zero for EITHER predictor (0.01-0.03) -- neither real
    outcome history nor xSLG meaningfully predicts this split at the
    individual batter-season level. Matches real-world intuition (which of a
    double/triple/single a ball becomes depends heavily on park dimensions,
    outfielder positioning/arm, and batter running speed -- none of which
    exit velocity + launch angle capture) and this project's own existing
    design choice (true_talent.py already gives double/triple a SHARED
    Carleton stabilization constant rather than separate ones, for the same
    underlying reason). NOT built -- left to the existing, already-validated
    outcome-based Marcel shrinkage untouched.

THIRD LAYER (2026-07-21): sprint speed for infield/groundball hits. Tested
leakage-free (same methodology): a batter's real hit rate ON GROUNDBALLS
specifically (launch_angle<10) vs. Baseball Savant's Sprint Speed leaderboard
(a real, season-aggregate biometric measurement, not derivable from our own
per-pitch cache -- see fetch_sprint_speed_seasons in fetch.py). Multivariate
R^2 improves substantially adding sprint_speed to real-groundball-BACON-only:
0.088->0.141 (2023->2024, +60% relative) and 0.024->0.043 (2024->2025, +80%
relative), both coefficients meaningfully positive. Tried the SAME test
against OVERALL (non-groundball-conditioned) single rate first and found
essentially nothing (corr 0.03-0.05, R^2 flat) -- confirms the signal is
real but genuinely narrow to groundball-specific contact, diluted to
invisibility by non-groundball singles (line drives, soft liners) that
aren't speed-driven the same way. Applied to the "single" category ONLY,
SCALED by the batter's own walk-forward groundball rate (a new tracked
signal, see build_groundball_rate_by_season, adapting spray.py's exact
walk-forward pull-rate pattern) -- a fly-ball-heavy hitter gets almost no
adjustment, a groundball-heavy hitter gets close to the full effect.

FOURTH LAYER (2026-07-21): whiff rate on swings, for strikeout. This started
as an investigation into genuine pitch-level "arsenal matchup" modeling --
does a batter's real performance specifically against pitchers who throw a
lot of breaking balls differ from their performance against fastball-heavy
pitchers, beyond what their aggregate K rate already predicts? Tested
leakage-free (batter's real K rate specifically vs. pitcher-arsenal terciles,
Bayesian-shrunk odds-ratio multiplier, same discipline as platoon_splits.py)
across THREE outcomes (strikeout, home_run, single) -- all three came back
NEGATIVE (the matchup-conditional adjustment made predictions slightly WORSE
than the batter's aggregate rate alone in every case), matching this
session's earlier CSW% finding: once a real outcome-based rate exists for a
player, decomposing by a coarse pitcher-arsenal category doesn't add
incremental value on our data (a pitcher's SEASON-aggregate pitch mix is a
blunt proxy for what a batter actually sees in any one specific PA).

Before abandoning pitch-level data entirely, tested a DIFFERENT framing: not
"batter's rate vs. pitcher's arsenal" (matchup-conditional), but a pure
batter-side skill signal -- real whiff-rate-on-swings, analogous to how
barrel rate feeds hr_share_multiplier above. This DID show real incremental
signal in isolation: real whiff rate (any pitch type) predicting NEXT-season
K rate beat this-season's own K rate at predicting itself in a multivariate
sense -- incremental R^2 of +0.0134 (n=720 combined batter-season pairs).
Built as a whiff_rate_multiplier applied to "strikeout" ONLY (architecturally
identical to hr_share_multiplier), then run through the same full-stack A/B
test every other factor this session has been held to. Result: MIXED-TO-
NEGATIVE -- total MAE improved slightly (3.405->3.379) but margin MAE
(3.462->3.482) and straight-up accuracy (59.3%->58.0%, -1.3pp) both got
WORSE. Reverted, same as isotonic calibration earlier this session: a real,
statistically legitimate component-level signal does not always survive
contact with the full-stack simulator, and the full-stack A/B is the bar
that decides, not the component-level regression fit.
"""

import numpy as np
import pandas as pd

from src.models.spray import compute_pull_angle, PULL_ANGLE_THRESHOLD_DEG
from src.models.true_talent import MARCEL_WEIGHTS
from src.utils.paths import DATA_RAW

HIT_EVENTS = ["single", "double", "triple", "home_run"]
# every outcome that requires bat-on-ball contact (i.e. everything except the
# no-contact categories strikeout/walk/intent_walk/hit_by_pitch) -- the
# implicit "out on contact" side of the existing true-talent rates that
# XBACON_PRIOR_BIP corrects against.
BIP_OUT_EVENTS = ["field_out", "double_play", "fielders_choice", "field_error",
                  "sac_fly", "sac_bunt", "triple_play", "catcher_interf"]

XBACON_PRIOR_BIP = 300  # Bayesian-shrinkage prior, pseudo-balls-in-play. A method-of-
                         # moments fit on real data (decomposing observed batter-season
                         # xBACON variance into true-skill variance + per-batted-ball
                         # noise/n) gave ~138 pseudo-BIP; rounded up for a safety margin,
                         # same practice as catcher_framing.py's CATCHER_FRAMING_PRIOR_PITCHES.


def _season_bip_sums(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (batter, season): balls-in-play count and sum of xBA over
    those BIP -- the raw material for the Marcel-weighted preseason prior."""
    tmp = pa[["batter", "season"]].copy()
    tmp["is_bip"] = pa["estimated_ba_using_speedangle"].notna().astype(int)
    tmp["xba_sum"] = pa["estimated_ba_using_speedangle"].fillna(0.0)
    return tmp.groupby(["batter", "season"]).agg(
        bip=("is_bip", "sum"), xba_sum=("xba_sum", "sum")
    ).reset_index()


def _preseason_xbacon_priors(pa: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, float]:
    """Marcel-weighted (5/4/3), regressed-toward-league xBACON prior for
    `target_season`, built only from seasons STRICTLY BEFORE it -- same
    structure as true_talent.py's build_preseason_priors, generalized from a
    binary outcome count to a continuous per-BIP value."""
    empty_cols = ["batter", "preseason_xbacon", "prior_weight_bip"]
    prior_seasons = pa[pa["season"] < target_season]

    if prior_seasons.empty:
        cur = pa[pa["season"] == target_season]
        league_xbacon = cur["estimated_ba_using_speedangle"].mean()
        return pd.DataFrame(columns=empty_cols), league_xbacon

    league_xbacon = prior_seasons["estimated_ba_using_speedangle"].mean()
    season_sums = _season_bip_sums(prior_seasons)

    priors = []
    for batter, g in season_sums.groupby("batter"):
        g = g.set_index("season")
        num, den = 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                num += w * g.loc[s, "xba_sum"]
                den += w * g.loc[s, "bip"]
        if den == 0:
            continue
        priors.append({"batter": batter, "prior_bip": den, "prior_xba_sum": num})
    priors_df = pd.DataFrame(priors)
    if priors_df.empty:
        return pd.DataFrame(columns=empty_cols), league_xbacon

    priors_df["reliability"] = priors_df["prior_bip"] / (priors_df["prior_bip"] + XBACON_PRIOR_BIP)
    raw_xbacon = priors_df["prior_xba_sum"] / priors_df["prior_bip"]
    priors_df["preseason_xbacon"] = (
        priors_df["reliability"] * raw_xbacon + (1 - priors_df["reliability"]) * league_xbacon
    )
    priors_df["prior_weight_bip"] = np.minimum(priors_df["prior_bip"], XBACON_PRIOR_BIP)
    return priors_df[empty_cols], league_xbacon


def build_pregame_xbacon(pa: pd.DataFrame) -> pd.DataFrame:
    """For every PA (BIP or not -- a batter's contact-quality estimate is
    looked up regardless of what THIS specific PA's own outcome is, same as
    every true-talent rate), the walk-forward-safe (no leakage) xBACON
    estimate as of just before that PA: preseason Marcel prior blended with
    in-season-to-date actual BIP outcomes."""
    pa = pa.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"]).copy()
    # pandas' nullable "Float64" extension dtype (what parquet round-trips
    # Statcast's estimated_ba_using_speedangle as, since it's NA-heavy) behaves
    # oddly through cumsum/merge chains -- cast to plain numpy float64 up front.
    pa["estimated_ba_using_speedangle"] = pa["estimated_ba_using_speedangle"].astype("float64")
    pa["is_bip"] = pa["estimated_ba_using_speedangle"].notna().astype(int)
    pa["xba_value"] = pa["estimated_ba_using_speedangle"].fillna(0.0)

    out_frames = []
    for season, sdf in pa.groupby("season"):
        priors_df, league_xbacon = _preseason_xbacon_priors(pa, season)
        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_xbacon"] = sdf["preseason_xbacon"].fillna(league_xbacon)
        sdf["prior_weight_bip"] = sdf["prior_weight_bip"].fillna(XBACON_PRIOR_BIP)

        grp = sdf.groupby("batter")
        sdf["season_xba_before"] = grp["xba_value"].cumsum() - sdf["xba_value"]
        sdf["season_bip_before"] = grp["is_bip"].cumsum() - sdf["is_bip"]

        num = sdf["prior_weight_bip"] * sdf["preseason_xbacon"] + sdf["season_xba_before"]
        den = sdf["prior_weight_bip"] + sdf["season_bip_before"]
        # the earliest season (no prior data at all) merges against an EMPTY
        # priors_df, whose columns default to object dtype with no data to
        # infer from -- explicitly cast back to float64 so pd.concat across
        # seasons below doesn't upcast the whole column to object.
        sdf["pregame_xbacon"] = (num / den).astype("float64")
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "pregame_xbacon"]]


def player_game_xbacon_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter): pregame_xbacon as of that batter's
    FIRST PA in that game -- same "fixed pregame snapshot" convention as
    validate_game_simulator.py's player_game_snapshot, so this can be looked
    up the same way (including the same nearest_prior_pitcher_snapshot
    fallback for a future/live game_pk with no exact row)."""
    wide = build_pregame_xbacon(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)


BARREL_PRIOR_BIP = 300  # Bayesian-shrinkage prior, pseudo-balls-in-play. A method-of-
                         # moments fit here was numerically unstable (observed variance
                         # barely exceeded the estimated binomial-noise term, producing an
                         # implausibly large prior_n) -- used the same value as
                         # XBACON_PRIOR_BIP instead, a reasonable, consistent default
                         # rather than trusting an unstable derivation.


def _season_barrel_sums(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (batter, season): balls-in-play count and count of barrels
    (Statcast's own launch_speed_angle==6 "well-struck" classification)."""
    tmp = pa[["batter", "season"]].copy()
    tmp["is_bip"] = pa["launch_speed_angle"].notna().astype(int)
    tmp["is_barrel"] = (pa["launch_speed_angle"] == 6).astype(int)
    return tmp.groupby(["batter", "season"]).agg(
        bip=("is_bip", "sum"), barrels=("is_barrel", "sum")
    ).reset_index()


def _preseason_barrel_priors(pa: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, float]:
    """Marcel-weighted (5/4/3), regressed-toward-league barrel-rate prior for
    `target_season`, seasons STRICTLY BEFORE it only -- same structure as
    _preseason_xbacon_priors, generalized to a different per-BIP quantity."""
    empty_cols = ["batter", "preseason_barrel_rate", "prior_weight_bip"]
    prior_seasons = pa[pa["season"] < target_season]

    if prior_seasons.empty:
        cur = pa[pa["season"] == target_season]
        league_rate = (cur["launch_speed_angle"] == 6).mean()
        return pd.DataFrame(columns=empty_cols), league_rate

    league_rate = (prior_seasons["launch_speed_angle"] == 6).mean()
    season_sums = _season_barrel_sums(prior_seasons)

    priors = []
    for batter, g in season_sums.groupby("batter"):
        g = g.set_index("season")
        num, den = 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                num += w * g.loc[s, "barrels"]
                den += w * g.loc[s, "bip"]
        if den == 0:
            continue
        priors.append({"batter": batter, "prior_bip": den, "prior_barrels": num})
    priors_df = pd.DataFrame(priors)
    if priors_df.empty:
        return pd.DataFrame(columns=empty_cols), league_rate

    priors_df["reliability"] = priors_df["prior_bip"] / (priors_df["prior_bip"] + BARREL_PRIOR_BIP)
    raw_rate = priors_df["prior_barrels"] / priors_df["prior_bip"]
    priors_df["preseason_barrel_rate"] = (
        priors_df["reliability"] * raw_rate + (1 - priors_df["reliability"]) * league_rate
    )
    priors_df["prior_weight_bip"] = np.minimum(priors_df["prior_bip"], BARREL_PRIOR_BIP)
    return priors_df[empty_cols], league_rate


def build_pregame_barrel_rate(pa: pd.DataFrame) -> pd.DataFrame:
    """For every PA, the walk-forward-safe barrel-rate estimate as of just
    before that PA -- same structure/leakage-safety as build_pregame_xbacon."""
    pa = pa.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"]).copy()
    # same nullable-extension-dtype fix as build_pregame_xbacon -- pandas'
    # nullable Int64 propagates NA through `== 6` as NA (not False), which
    # then can't cast to plain int; casting to numpy float64 first makes
    # `NaN == 6` correctly evaluate to False.
    pa["launch_speed_angle"] = pa["launch_speed_angle"].astype("float64")
    pa["is_bip"] = pa["launch_speed_angle"].notna().astype(int)
    pa["is_barrel"] = (pa["launch_speed_angle"] == 6).astype(int)

    out_frames = []
    for season, sdf in pa.groupby("season"):
        priors_df, league_rate = _preseason_barrel_priors(pa, season)
        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_barrel_rate"] = sdf["preseason_barrel_rate"].fillna(league_rate)
        sdf["prior_weight_bip"] = sdf["prior_weight_bip"].fillna(BARREL_PRIOR_BIP)

        grp = sdf.groupby("batter")
        sdf["season_barrels_before"] = grp["is_barrel"].cumsum() - sdf["is_barrel"]
        sdf["season_bip_before"] = grp["is_bip"].cumsum() - sdf["is_bip"]

        num = sdf["prior_weight_bip"] * sdf["preseason_barrel_rate"] + sdf["season_barrels_before"]
        den = sdf["prior_weight_bip"] + sdf["season_bip_before"]
        sdf["pregame_barrel_rate"] = (num / den).astype("float64")
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "pregame_barrel_rate"]]


def player_game_barrel_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter): pregame_barrel_rate as of that batter's
    FIRST PA in that game -- same convention as player_game_xbacon_snapshot."""
    wide = build_pregame_barrel_rate(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)


# SEVENTH LAYER (2026-07-22): pulled-air-ball rate (direction, not power) as a
# further extension to hr_share_multiplier. Motivated by a research pass
# citing real Statcast batted-ball splits: pulled fly balls convert to home
# runs at a much higher rate than balls hit elsewhere (confirmed directly on
# this project's own 2023-2025 data below -- 17.8% HR rate on pulled air vs.
# 3.96% on non-pulled air, and pulled-air balls account for 67.5% of all real
# home runs, both closely matching the externally-cited figures). This is a
# genuinely different AXIS from barrel rate (which captures raw contact
# quality/power, not direction) and from the already-rejected
# HR-share-with-bat-speed extension (a different physical mechanism), so it
# isn't just a retry of a previously failed idea.
#
# "Pulled air" here = compute_pull_angle (see spray.py) > PULL_ANGLE_THRESHOLD_DEG
# AND launch_angle > GROUNDBALL_LAUNCH_ANGLE_MAX (i.e. not a groundball) --
# the complement of this file's own groundball cutoff, since Statcast's raw
# per-pitch feed used here has no separate bb_type (line_drive/fly_ball/popup)
# classification to match external sources' narrower "fly ball" definition
# exactly.
#
# Leakage-free incremental-R^2 test (2 season-pairs, n=387/385 batters with
# 50+ BIP/season): does a batter's own pulled-air rate add value to
# predicting NEXT season's real hr-share-among-hits BEYOND the current
# production signal (existing hr-share + barrel rate)? R^2 0.4259->0.4371
# (2023->2024) and 0.3876->0.4105 (2024->2025) -- both positive, comparable
# in magnitude to barrel rate's own original incremental gain over plain
# outcome history.
#
# Real-data limit test (2025, same 486 real batters as the barrel-only
# multiplier): raw pulled_air_rate pushed the worst-case multiplier to 5.9x
# vs. the CURRENT LIVE hr_share_multiplier's own already-existing 5.1x
# worst case on the SAME batters -- i.e. this is a modest widening of an
# already-accepted tail (same root cause: a nonzero regression intercept
# can't predict a real existing_hr_share of exactly 0), not a new order-of-
# magnitude blowup class like the original unclipped 13.8x bug. Bayesian-
# shrinking pulled_air_rate itself (PULLED_AIR_PRIOR_BIP, same mechanism as
# BARREL_PRIOR_BIP) toward the league-average rate (~17% every season
# 2023-2026, quite stable) modestly IMPROVES both the R^2 gain (+0.0226 vs.
# +0.0171 unshrunk, averaged) and the tail (max 5.64x vs. 5.90x) -- used here.
PULLED_AIR_PRIOR_BIP = 300  # same value as BARREL_PRIOR_BIP -- a reasonable, consistent
                             # default; league pulled-air rate is stable enough
                             # (~0.168-0.179 across 2023-2026) that shrinkage strength
                             # isn't especially sensitive to the exact prior weight.


def _clean_pulled_air_inputs(pa: pd.DataFrame) -> pd.DataFrame:
    """Cast hc_x/hc_y/launch_angle to plain float64 before any comparison --
    same nullable-extension-dtype fix as build_pregame_barrel_rate's own
    launch_speed_angle cast. Pandas' nullable dtypes propagate NA through
    comparisons as pd.NA (not False/NaN), which then can't `.astype(int)`;
    casting to numpy float64 first makes `NaN > threshold` correctly
    evaluate to plain False."""
    pa = pa.copy()
    for col in ("hc_x", "hc_y", "launch_angle"):
        pa[col] = pa[col].astype("float64")
    return pa


def _season_pulled_air_sums(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (batter, season): balls-in-play count and count of pulled
    air balls (pulled per spray.py's own angle convention AND not a
    groundball) -- same shape as _season_barrel_sums."""
    pa = _clean_pulled_air_inputs(pa)
    tmp = pa[["batter", "season"]].copy()
    tmp["is_bip"] = (pa["hc_x"].notna() & pa["hc_y"].notna() & pa["launch_angle"].notna()).astype(int)
    pull_angle = compute_pull_angle(pa)
    tmp["is_pulled_air"] = (
        (pull_angle > PULL_ANGLE_THRESHOLD_DEG) & (pa["launch_angle"] > GROUNDBALL_LAUNCH_ANGLE_MAX)
    ).astype(int)
    return tmp.groupby(["batter", "season"]).agg(
        bip=("is_bip", "sum"), pulled_air=("is_pulled_air", "sum")
    ).reset_index()


def _preseason_pulled_air_priors(pa: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, float]:
    """Marcel-weighted (5/4/3), regressed-toward-league pulled-air-rate prior
    for `target_season`, seasons STRICTLY BEFORE it only -- same structure as
    _preseason_barrel_priors, generalized to a different per-BIP quantity."""
    empty_cols = ["batter", "preseason_pulled_air_rate", "prior_weight_bip"]
    pa = _clean_pulled_air_inputs(pa)
    prior_seasons = pa[pa["season"] < target_season]

    if prior_seasons.empty:
        cur = pa[pa["season"] == target_season]
        cur_pull_angle = compute_pull_angle(cur)
        league_rate = ((cur_pull_angle > PULL_ANGLE_THRESHOLD_DEG) & (cur["launch_angle"] > GROUNDBALL_LAUNCH_ANGLE_MAX)).mean()
        return pd.DataFrame(columns=empty_cols), league_rate

    prior_pull_angle = compute_pull_angle(prior_seasons)
    league_rate = ((prior_pull_angle > PULL_ANGLE_THRESHOLD_DEG) & (prior_seasons["launch_angle"] > GROUNDBALL_LAUNCH_ANGLE_MAX)).mean()
    season_sums = _season_pulled_air_sums(prior_seasons)

    priors = []
    for batter, g in season_sums.groupby("batter"):
        g = g.set_index("season")
        num, den = 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                num += w * g.loc[s, "pulled_air"]
                den += w * g.loc[s, "bip"]
        if den == 0:
            continue
        priors.append({"batter": batter, "prior_bip": den, "prior_pulled_air": num})
    priors_df = pd.DataFrame(priors)
    if priors_df.empty:
        return pd.DataFrame(columns=empty_cols), league_rate

    priors_df["reliability"] = priors_df["prior_bip"] / (priors_df["prior_bip"] + PULLED_AIR_PRIOR_BIP)
    raw_rate = priors_df["prior_pulled_air"] / priors_df["prior_bip"]
    priors_df["preseason_pulled_air_rate"] = (
        priors_df["reliability"] * raw_rate + (1 - priors_df["reliability"]) * league_rate
    )
    priors_df["prior_weight_bip"] = np.minimum(priors_df["prior_bip"], PULLED_AIR_PRIOR_BIP)
    return priors_df[empty_cols], league_rate


def build_pregame_pulled_air_rate(pa: pd.DataFrame) -> pd.DataFrame:
    """For every PA, the walk-forward-safe pulled-air-rate estimate as of just
    before that PA -- same structure/leakage-safety as build_pregame_barrel_rate."""
    pa = pa.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"]).copy()
    pa = _clean_pulled_air_inputs(pa)
    pull_angle = compute_pull_angle(pa)
    pa["is_bip"] = (pa["launch_angle"].notna() & pa["hc_x"].notna() & pa["hc_y"].notna()).astype(int)
    pa["is_pulled_air"] = ((pull_angle > PULL_ANGLE_THRESHOLD_DEG) & (pa["launch_angle"] > GROUNDBALL_LAUNCH_ANGLE_MAX)).astype(int)

    out_frames = []
    for season, sdf in pa.groupby("season"):
        priors_df, league_rate = _preseason_pulled_air_priors(pa, season)
        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_pulled_air_rate"] = sdf["preseason_pulled_air_rate"].fillna(league_rate)
        sdf["prior_weight_bip"] = sdf["prior_weight_bip"].fillna(PULLED_AIR_PRIOR_BIP)

        grp = sdf.groupby("batter")
        sdf["season_pulled_air_before"] = grp["is_pulled_air"].cumsum() - sdf["is_pulled_air"]
        sdf["season_bip_before"] = grp["is_bip"].cumsum() - sdf["is_bip"]

        num = sdf["prior_weight_bip"] * sdf["preseason_pulled_air_rate"] + sdf["season_pulled_air_before"]
        den = sdf["prior_weight_bip"] + sdf["season_bip_before"]
        sdf["pregame_pulled_air_rate"] = (num / den).astype("float64")
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "pregame_pulled_air_rate"]]


def player_game_pulled_air_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter): pregame_pulled_air_rate as of that
    batter's FIRST PA in that game -- same convention as
    player_game_barrel_snapshot."""
    wide = build_pregame_pulled_air_rate(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)


def odds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


# real range confirmed on our own data (2026-07-21, added after an audit
# found this was the one multiplier in this file with no explicit clip,
# unlike hr_share_multiplier's HR_SHARE_CLIP_MIN/MAX and
# groundball_single_multiplier's GROUNDBALL_BACON_CLIP_MIN/MAX): real
# pregame_xbacon ranges 0.234-0.500 (663,415 real batter-PA snapshots), and
# existing_bip_hit_share's 0.1st-99.9th percentile band is 0.228-0.457 --
# both comfortably inside this clip. Its own extreme tails (as low as 0.06,
# as high as 0.88, both rare small-sample cold-start artifacts) are exactly
# the kind of input that produced this project's other rare odds-ratio
# blowups (13.8x HR-share, 146,611x platoon, 73x weather) before each was
# fixed the same way -- clipping both sides before the ratio.
CONTACT_QUALITY_CLIP_MIN, CONTACT_QUALITY_CLIP_MAX = 0.10, 0.60

# FIFTH LAYER (2026-07-22): bat-tracking data (bat speed specifically --
# Statcast's bat-tracking system, ~93-97% coverage within swing rows from
# 2024 on, ~65% of batters usably covered even in 2023's mid-season rollout
# with a 100+ real-swing minimum). Genuinely new data this project hadn't
# touched -- not another way of reslicing existing PA-outcome history.
# Leakage-free (2 season-pairs, 2023->2024 and 2024->2025, n=756 combined
# batter-season pairs): does bat speed add INCREMENTAL value beyond what's
# ALREADY in production (xBACON alone, this function's current mechanism)?
# R^2 0.139 (xBACON-only) -> 0.183 (+bat_speed), a real ~32% relative gain.
# Real-data limit test (n=557-575 real batters) came back well-behaved --
# multiplier range 0.51-1.86, comparable to this function's own existing
# range (0.71-1.79) -- so this IS built, blended via a genuine fitted
# regression (like hr_share_multiplier), replacing the previous pure
# xBACON-substitution when pregame_bat_speed is available.
#
# The SAME test for home_run-share-among-hits (hr_share_multiplier below)
# ALSO showed real incremental value (R^2 0.379->0.405, existing+barrel ->
# +bat_speed) but failed its own real-data limit test: multipliers up to
# 5.5x for batters with zero real home runs in their sample, the exact same
# "regression intercept prevents predicting near-zero" failure mode this
# file's own HR_SHARE_CLIP_MIN/MAX docstring already documents for the
# ORIGINAL 2-predictor version -- a tighter clip floor (0.10) tames it, but
# that would clip real, weak-power hitters far more aggressively than the
# genuine data range (1st pctile ~0.03) warrants. NOT built -- a real,
# validated incremental signal that isn't safely deployable with this
# specific regression without more work than this session's scope allows,
# same "found real signal but didn't clear the full bar" treatment as CSW%.
CONTACT_QUALITY_BATSPEED_INTERCEPT = -0.02000
CONTACT_QUALITY_BATSPEED_COEF_XBACON = 0.33275
CONTACT_QUALITY_BATSPEED_COEF_BATSPEED = 0.00330


def contact_quality_multiplier(rates: dict[str, float], pregame_xbacon: float,
                                pregame_bat_speed: float | None = None) -> float:
    """The single odds-ratio multiplier to apply to every HIT_EVENTS category
    in a batter's `rates` dict: a contact-quality-based hit probability vs.
    the rate this batter's existing Marcel-shrunk true-talent rates already
    imply for hit-on-contact. When pregame_bat_speed is given, the predicted
    share blends xBACON with bat speed via the fitted regression above
    (real incremental value beyond xBACON alone -- see module docstring);
    when None (no usable bat-tracking history, e.g. a rookie or a season
    predating 2023's rollout), falls back to xBACON used directly, same as
    before this signal existed. Returns 1.0 (neutral) if this batter has too
    little contact at all to form either side of the ratio (e.g. an extreme
    rookie-sample edge case) -- true-talent rates are always well-defined by
    construction so this only guards the pathological case of the whole
    numerator+denominator being ~0."""
    hit_rate = sum(rates[o] for o in HIT_EVENTS)
    bip_out_rate = sum(rates[o] for o in BIP_OUT_EVENTS)
    denom = hit_rate + bip_out_rate
    if denom <= 1e-9:
        return 1.0
    existing_bip_hit_share = hit_rate / denom
    xbacon_clipped = np.clip(pregame_xbacon, CONTACT_QUALITY_CLIP_MIN, CONTACT_QUALITY_CLIP_MAX)
    existing_clipped = np.clip(existing_bip_hit_share, CONTACT_QUALITY_CLIP_MIN, CONTACT_QUALITY_CLIP_MAX)
    if pregame_bat_speed is not None and not pd.isna(pregame_bat_speed):
        predicted = (
            CONTACT_QUALITY_BATSPEED_INTERCEPT
            + CONTACT_QUALITY_BATSPEED_COEF_XBACON * xbacon_clipped
            + CONTACT_QUALITY_BATSPEED_COEF_BATSPEED * pregame_bat_speed
        )
        predicted = np.clip(predicted, CONTACT_QUALITY_CLIP_MIN, CONTACT_QUALITY_CLIP_MAX)
        return odds(predicted) / odds(existing_clipped)
    return odds(xbacon_clipped) / odds(existing_clipped)


# empirically-fit (2026-07-21, see module docstring), averaged across the
# 2023->2024 and 2024->2025 leakage-free regressions: next-season real
# hr-share-among-hits regressed on this-season real hr-share-among-hits AND
# this-season barrel rate together. Both coefficients meaningfully positive
# (barrel rate weighted higher, matching it being the stronger single
# predictor) -- a genuine BLEND, unlike xBACON's near-total substitution.
HR_SHARE_INTERCEPT = 0.0442
HR_SHARE_COEF_EXISTING = 0.3616
HR_SHARE_COEF_BARREL = 0.5300

# Real batter-level hr-share-among-hits ranges from ~0.03 (1st pctile) to
# ~0.29 (99th pctile), confirmed on real data. Clipping both the existing
# share AND the regression-predicted share to this band before taking their
# odds-ratio is NECESSARY, not cosmetic -- confirmed a real bug without it:
# a linear regression's non-zero INTERCEPT (0.0442) means it never predicts
# a share anywhere near 0, so a genuinely weak-power batter with a
# near-zero existing_hr_share (e.g. 0.004, a real Marcel-shrunk value for a
# real slap-hitter) produces a tiny odds() denominator against a
# comparatively much larger predicted-share numerator -- observed directly:
# multipliers up to 13.8x across 989 real batters before this fix, the exact
# same "odds-ratio explodes when one side approaches zero" failure mode as
# this session's platoon_splits.py and weather.py bugs, just triggered by a
# regression intercept instead of a literal 0%/100% thin-sample observation.
HR_SHARE_CLIP_MIN = 0.02
HR_SHARE_CLIP_MAX = 0.35

# SEVENTH LAYER (see build_pregame_pulled_air_rate above for the full
# rationale/validation writeup): empirically-fit 2026-07-22, averaged across
# the 2023->2024 and 2024->2025 leakage-free regressions of next-season real
# hr-share-among-hits on this-season real hr-share, this-season barrel rate,
# AND this-season (PRIOR-shrunk) pulled-air rate together. Used only when
# pregame_pulled_air_rate is actually available (falls back to the 2-term
# HR_SHARE_* blend above otherwise), same "additive layer, not a silent
# fallback" convention as contact_quality_multiplier's bat-speed extension.
HR_SHARE_PULLEDAIR_INTERCEPT = -0.03469
HR_SHARE_PULLEDAIR_COEF_EXISTING = 0.21480
HR_SHARE_PULLEDAIR_COEF_BARREL = 0.64574
HR_SHARE_PULLEDAIR_COEF_PULLED_AIR = 0.50780


def hr_share_multiplier(rates: dict[str, float], pregame_barrel_rate: float,
                         pregame_pulled_air_rate: float | None = None) -> float:
    """The odds-ratio multiplier to apply to home_run ONLY (not the other
    HIT_EVENTS categories -- see apply_contact_quality, which renormalizes
    the whole vector downstream in combine_matchup_distribution so shifting
    just this one category correctly reallocates share within hits): the
    regression-blended predicted home_run share among this batter's hits vs.
    the share their existing true-talent rates alone already imply. When
    pregame_pulled_air_rate is given, blends in the batter's own pulled-air
    tendency (a direction-based signal, not power -- see module docstring)
    via the fitted 3-term regression above; when None (e.g. missing hc_x/hc_y
    coverage), falls back to the original 2-term existing+barrel blend."""
    hit_rate = sum(rates[o] for o in HIT_EVENTS)
    if hit_rate <= 1e-9:
        return 1.0
    existing_hr_share = np.clip(rates["home_run"] / hit_rate, HR_SHARE_CLIP_MIN, HR_SHARE_CLIP_MAX)
    if pregame_pulled_air_rate is not None and not pd.isna(pregame_pulled_air_rate):
        predicted_share = (
            HR_SHARE_PULLEDAIR_INTERCEPT
            + HR_SHARE_PULLEDAIR_COEF_EXISTING * existing_hr_share
            + HR_SHARE_PULLEDAIR_COEF_BARREL * pregame_barrel_rate
            + HR_SHARE_PULLEDAIR_COEF_PULLED_AIR * pregame_pulled_air_rate
        )
    else:
        predicted_share = (
            HR_SHARE_INTERCEPT + HR_SHARE_COEF_EXISTING * existing_hr_share + HR_SHARE_COEF_BARREL * pregame_barrel_rate
        )
    predicted_share = np.clip(predicted_share, HR_SHARE_CLIP_MIN, HR_SHARE_CLIP_MAX)
    return odds(predicted_share) / odds(existing_hr_share)


def apply_contact_quality(rates: dict[str, float], pregame_xbacon: float | None,
                           pregame_barrel_rate: float | None = None,
                           pregame_bacon_gb: float | None = None,
                           pregame_sprint_speed: float | None = None,
                           pregame_gb_rate: float | None = None,
                           pregame_bat_speed: float | None = None,
                           pregame_pulled_air_rate: float | None = None,
                           pregame_gb_rate_pitcher: float | None = None,
                           pregame_fb_rate_pitcher: float | None = None,
                           pitcher_fullmix: bool = False) -> dict[str, float]:
    """Returns a NEW rates dict with HIT_EVENTS categories scaled by the
    aggregate contact-quality multiplier (xBACON, optionally blended with bat
    speed -- see contact_quality_multiplier, hit-vs-out), then home_run
    specifically further adjusted by the barrel-rate-based HR-share
    multiplier (optionally further blended with pulled-air rate -- see
    hr_share_multiplier), and "single" further adjusted by the sprint-speed-
    based groundball-hit multiplier -- other categories (BB/K/BIP-outs/etc.)
    untouched. Safe to call with any subset of args as None (e.g. no BIP
    history yet, or no sprint-speed measurement) -- a missing signal is a
    no-op for its own layer, not a fallback to another. The groundball layer
    needs ALL THREE of its own args (bacon_gb, sprint_speed, gb_rate) to
    apply -- any one missing skips just that layer.

    pregame_gb_rate_pitcher/pregame_fb_rate_pitcher: the PITCHER-side GB%/FB%
    layer (task #120/#126) -- callers building a BATTER profile never pass
    these (leaving them None, a no-op), exactly the same "pitcher-profile
    callers just don't pass batter-only args" convention
    pregame_barrel_rate/pregame_bat_speed already use in reverse. HR-allowed
    (pitcher_hr_allowed_multiplier) is CONFIRMED and KEPT (real full-stack
    win, SU +1.53pp at n=7237, CI excludes zero). pitcher_fullmix (default
    False, matching this project's "new mechanism defaults off until A/B
    tested" convention) additionally applies the double-allowed and
    double-play-allowed extensions to the SAME GB%/FB% inputs (task #126) --
    set True to test the fuller mix against the already-confirmed HR-only
    baseline."""
    adjusted = dict(rates)
    if pregame_xbacon is not None and not pd.isna(pregame_xbacon):
        mult = contact_quality_multiplier(adjusted, pregame_xbacon, pregame_bat_speed)
        for o in HIT_EVENTS:
            adjusted[o] = adjusted[o] * mult
    if pregame_barrel_rate is not None and not pd.isna(pregame_barrel_rate):
        adjusted["home_run"] = adjusted["home_run"] * hr_share_multiplier(
            adjusted, pregame_barrel_rate, pregame_pulled_air_rate
        )
    if (pregame_bacon_gb is not None and not pd.isna(pregame_bacon_gb)
            and pregame_sprint_speed is not None and not pd.isna(pregame_sprint_speed)
            and pregame_gb_rate is not None and not pd.isna(pregame_gb_rate)):
        adjusted["single"] = adjusted["single"] * groundball_single_multiplier(
            pregame_bacon_gb, pregame_sprint_speed, pregame_gb_rate
        )
    if (pregame_gb_rate_pitcher is not None and not pd.isna(pregame_gb_rate_pitcher)
            and pregame_fb_rate_pitcher is not None and not pd.isna(pregame_fb_rate_pitcher)):
        adjusted["home_run"] = adjusted["home_run"] * pitcher_hr_allowed_multiplier(
            adjusted, pregame_gb_rate_pitcher, pregame_fb_rate_pitcher
        )
        if pitcher_fullmix:
            adjusted["double"] = adjusted["double"] * pitcher_double_allowed_multiplier(
                adjusted, pregame_gb_rate_pitcher, pregame_fb_rate_pitcher
            )
            adjusted["double_play"] = adjusted["double_play"] * pitcher_double_play_multiplier(
                adjusted, pregame_gb_rate_pitcher, pregame_fb_rate_pitcher
            )
    return adjusted


GROUNDBALL_LAUNCH_ANGLE_MAX = 10  # standard sabermetric groundball cutoff
GROUNDBALL_STABILIZATION_BATTED_BALLS = 100  # same order of magnitude as spray.py's
                                               # PULL_STABILIZATION_BATTED_BALLS -- both
                                               # are batted-ball-conditioned rates of a
                                               # similar real-world stability


def build_groundball_rate_by_season(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe estimated groundball rate for every batted ball, as
    of just before it -- EXACT same structure as spray.py's
    build_pull_rate_by_season (preseason Marcel prior + in-season blend),
    just tracking launch_angle<10 instead of pull direction."""
    bb = pa.dropna(subset=["launch_angle"]).copy()
    bb["is_gb"] = (bb["launch_angle"] < GROUNDBALL_LAUNCH_ANGLE_MAX).astype(int)
    bb = bb.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"])

    K = GROUNDBALL_STABILIZATION_BATTED_BALLS
    out_frames = []
    for season, sdf in bb.groupby("season"):
        prior = bb[bb["season"] < season]
        if prior.empty:
            league_rate = bb.loc[bb["season"] == season, "is_gb"].mean()
            priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])
        else:
            league_rate = prior["is_gb"].mean()
            season_agg = prior.groupby(["batter", "season"]).agg(
                bb=("is_gb", "size"), events=("is_gb", "sum")
            ).reset_index()
            priors = []
            for batter, g in season_agg.groupby("batter"):
                g = g.set_index("season")
                num, den = 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num += w * g.loc[s, "events"]
                        den += w * g.loc[s, "bb"]
                if den == 0:
                    continue
                priors.append({"batter": batter, "prior_bb": den, "prior_events": num})
            priors_df = pd.DataFrame(priors, columns=["batter", "prior_bb", "prior_events"])
            if not priors_df.empty:
                priors_df["reliability"] = priors_df["prior_bb"] / (priors_df["prior_bb"] + K)
                raw_rate = priors_df["prior_events"] / priors_df["prior_bb"]
                priors_df["preseason_rate"] = (
                    priors_df["reliability"] * raw_rate + (1 - priors_df["reliability"]) * league_rate
                )
                priors_df["prior_weight_bb"] = np.minimum(priors_df["prior_bb"], K)
                priors_df = priors_df[["batter", "preseason_rate", "prior_weight_bb"]]
            else:
                priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])

        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_rate"] = sdf["preseason_rate"].fillna(league_rate)
        sdf["prior_weight_bb"] = sdf["prior_weight_bb"].fillna(K)

        grp = sdf.groupby("batter")
        sdf["season_events_before"] = grp["is_gb"].cumsum() - sdf["is_gb"]
        sdf["season_bb_before"] = grp.cumcount()

        num = sdf["prior_weight_bb"] * sdf["preseason_rate"] + sdf["season_events_before"]
        den = sdf["prior_weight_bb"] + sdf["season_bb_before"]
        sdf["pregame_gb_rate"] = num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "game_date", "pregame_gb_rate"]]


def build_pregame_bacon_gb(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe real hit-rate-on-groundballs estimate, same
    preseason-prior + in-season-blend structure as build_pregame_xbacon,
    conditioned on launch_angle<10 batted balls only."""
    pa = pa.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"]).copy()
    # same nullable-extension-dtype fix as build_pregame_xbacon/build_pregame_barrel_rate --
    # cast to plain numpy float64 BEFORE comparing, so a missing launch_angle (not a
    # batted ball) correctly evaluates the comparison to False, not NA.
    pa["launch_angle"] = pa["launch_angle"].astype("float64")
    pa["is_gb"] = (pa["launch_angle"] < GROUNDBALL_LAUNCH_ANGLE_MAX).astype(float)
    pa.loc[pa["launch_angle"].isna(), "is_gb"] = 0.0
    pa["is_gb_hit"] = ((pa["is_gb"] == 1) & (pa["outcome"].isin(HIT_EVENTS))).astype(float)

    K = XBACON_PRIOR_BIP  # same order-of-magnitude shrinkage as the other contact-quality rates
    out_frames = []
    for season, sdf in pa.groupby("season"):
        prior_seasons = pa[pa["season"] < season]
        if prior_seasons.empty:
            cur = pa[pa["season"] == season]
            league_rate = cur.loc[cur["is_gb"] == 1, "is_gb_hit"].mean() if (cur["is_gb"] == 1).any() else 0.25
            priors_df = pd.DataFrame(columns=["batter", "preseason_bacon_gb", "prior_weight_gb"])
        else:
            gb_prior = prior_seasons[prior_seasons["is_gb"] == 1]
            league_rate = gb_prior["is_gb_hit"].mean()
            season_agg = gb_prior.groupby(["batter", "season"]).agg(
                n_gb=("is_gb_hit", "size"), gb_hits=("is_gb_hit", "sum")
            ).reset_index()
            rows = []
            for batter, g in season_agg.groupby("batter"):
                g = g.set_index("season")
                num, den = 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num += w * g.loc[s, "gb_hits"]
                        den += w * g.loc[s, "n_gb"]
                if den == 0:
                    continue
                rows.append({"batter": batter, "prior_gb": den, "prior_gb_hits": num})
            priors_df = pd.DataFrame(rows, columns=["batter", "prior_gb", "prior_gb_hits"])
            if not priors_df.empty:
                reliability = priors_df["prior_gb"] / (priors_df["prior_gb"] + K)
                raw_rate = priors_df["prior_gb_hits"] / priors_df["prior_gb"]
                priors_df["preseason_bacon_gb"] = reliability * raw_rate + (1 - reliability) * league_rate
                priors_df["prior_weight_gb"] = np.minimum(priors_df["prior_gb"], K)
                priors_df = priors_df[["batter", "preseason_bacon_gb", "prior_weight_gb"]]
            else:
                priors_df = pd.DataFrame(columns=["batter", "preseason_bacon_gb", "prior_weight_gb"])

        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_bacon_gb"] = sdf["preseason_bacon_gb"].fillna(league_rate)
        sdf["prior_weight_gb"] = sdf["prior_weight_gb"].fillna(K)

        grp = sdf.groupby("batter")
        sdf["season_gb_hits_before"] = grp["is_gb_hit"].cumsum() - sdf["is_gb_hit"]
        sdf["season_gb_before"] = grp["is_gb"].cumsum() - sdf["is_gb"]

        num = sdf["prior_weight_gb"] * sdf["preseason_bacon_gb"] + sdf["season_gb_hits_before"]
        den = sdf["prior_weight_gb"] + sdf["season_gb_before"]
        sdf["pregame_bacon_gb"] = (num / den).astype("float64")
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "pregame_bacon_gb"]]


def player_game_bacon_gb_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    wide = build_pregame_bacon_gb(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)


def player_game_groundball_rate_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    wide = build_groundball_rate_by_season(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)[
        ["game_pk", "batter", "season", "pregame_gb_rate"]
    ]


# EIGHTH LAYER -- pitcher-side GB%/FB% -> HR-allowed-share (task #120, critique
# claim 4, 2026-07-22). DIPS theory says pitchers don't meaningfully control
# contact-quality OUTCOME on balls in play (matches every one of this
# project's own 5-for-5 failed pitcher-side signals: arsenal-tercile,
# whiff-rate, pitcher-stuff, xBACON/barrel-rate-allowed -- see
# MODEL_DOCUMENTATION.md's ledger) -- but "DIPS 2.0" explicitly carves out
# batted-ball TYPE as something pitchers DO meaningfully control (GB%/FB%
# stabilize fast, ~70 BF, per published research), distinct from what happens
# to that batted ball once put in play. Confirmed on real data (leakage-free,
# 3 walk-forward season pairs): a pitcher's own GB%/FB% (launch_angle<10 /
# >=25 among their own balls in play) adds real incremental signal predicting
# NEXT-season HR-allowed-share-among-BIP beyond their existing HR-share alone
# -- R^2 roughly DOUBLES in every pair tested (0.051->0.123, 0.039->0.112,
# 0.045->0.096), the largest single pitcher-side component gain found this
# entire session. Structurally identical to hr_share_multiplier above, just
# on the PITCHER's home_run category and using GB/FB rate (not barrel rate)
# as the extra predictor.
FLYBALL_LAUNCH_ANGLE_MIN = 25  # standard sabermetric flyball floor (line drives sit 10-25)
GB_FB_STABILIZATION_BIP = 200  # same order of magnitude as GROUNDBALL_STABILIZATION_BATTED_BALLS;
                               # pitcher GB%/FB% is well-established to stabilize fast (~70 BF per
                               # Carleton's published research), a modest prior is enough

HR_SHARE_PITCHER_CLIP_MIN, HR_SHARE_PITCHER_CLIP_MAX = 0.01, 0.20  # pitcher HR-among-BIP runs much
                                                                     # lower than batter HR-among-hits
                                                                     # (BIP includes all the extra outs
                                                                     # HIT_EVENTS excludes) -- confirmed
                                                                     # on real data: p1=0.010, p99=0.103

# Fit 2026-07-22: leakage-free walk-forward regression of next-season real
# pitcher HR-share-among-BIP on this-season existing HR-share-among-BIP, GB
# rate, and FB rate together, averaged across the 2023->2024, 2024->2025, and
# 2025->2026 season pairs (see scratchpad/fit_pitcher_hr_share.py) -- same
# "average across pairs" convention as HR_SHARE_COEF_BARREL above.
HR_SHARE_PITCHER_INTERCEPT = 0.04506
HR_SHARE_PITCHER_COEF_EXISTING = 0.09815
HR_SHARE_PITCHER_COEF_GB = -0.03611
HR_SHARE_PITCHER_COEF_FB = 0.03635


def build_pitcher_gb_fb_rate_by_season(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe (preseason Marcel prior + in-season blend, same
    structure as build_groundball_rate_by_season) pitcher-side GB%/FB% among
    their own balls in play, as of just before each PA they throw. Computes
    BOTH rates in one pass (not two separate calls) since the multiplier
    below needs both together."""
    bb = pa.dropna(subset=["launch_angle"]).copy()
    bb["is_gb"] = (bb["launch_angle"] < GROUNDBALL_LAUNCH_ANGLE_MAX).astype(int)
    bb["is_fb"] = (bb["launch_angle"] >= FLYBALL_LAUNCH_ANGLE_MIN).astype(int)
    bb = bb.sort_values(["pitcher", "season", "game_date", "game_pk", "at_bat_number"])

    K = GB_FB_STABILIZATION_BIP
    out_frames = []
    for season, sdf in bb.groupby("season"):
        prior = bb[bb["season"] < season]
        if prior.empty:
            league_gb = bb.loc[bb["season"] == season, "is_gb"].mean()
            league_fb = bb.loc[bb["season"] == season, "is_fb"].mean()
            priors_df = pd.DataFrame(columns=["pitcher", "preseason_gb_rate", "preseason_fb_rate", "prior_weight_bip"])
        else:
            league_gb, league_fb = prior["is_gb"].mean(), prior["is_fb"].mean()
            season_agg = prior.groupby(["pitcher", "season"]).agg(
                bip=("is_gb", "size"), gb=("is_gb", "sum"), fb=("is_fb", "sum")
            ).reset_index()
            priors = []
            for pitcher, g in season_agg.groupby("pitcher"):
                g = g.set_index("season")
                num_gb, num_fb, den = 0.0, 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num_gb += w * g.loc[s, "gb"]
                        num_fb += w * g.loc[s, "fb"]
                        den += w * g.loc[s, "bip"]
                if den == 0:
                    continue
                priors.append({"pitcher": pitcher, "prior_bip": den, "prior_gb": num_gb, "prior_fb": num_fb})
            priors_df = pd.DataFrame(priors, columns=["pitcher", "prior_bip", "prior_gb", "prior_fb"])
            if not priors_df.empty:
                reliability = priors_df["prior_bip"] / (priors_df["prior_bip"] + K)
                priors_df["preseason_gb_rate"] = (
                    reliability * (priors_df["prior_gb"] / priors_df["prior_bip"]) + (1 - reliability) * league_gb
                )
                priors_df["preseason_fb_rate"] = (
                    reliability * (priors_df["prior_fb"] / priors_df["prior_bip"]) + (1 - reliability) * league_fb
                )
                priors_df["prior_weight_bip"] = np.minimum(priors_df["prior_bip"], K)
                priors_df = priors_df[["pitcher", "preseason_gb_rate", "preseason_fb_rate", "prior_weight_bip"]]
            else:
                priors_df = pd.DataFrame(columns=["pitcher", "preseason_gb_rate", "preseason_fb_rate", "prior_weight_bip"])

        sdf = sdf.merge(priors_df, on="pitcher", how="left")
        sdf["preseason_gb_rate"] = sdf["preseason_gb_rate"].fillna(league_gb)
        sdf["preseason_fb_rate"] = sdf["preseason_fb_rate"].fillna(league_fb)
        sdf["prior_weight_bip"] = sdf["prior_weight_bip"].fillna(K)

        grp = sdf.groupby("pitcher")
        sdf["season_gb_before"] = grp["is_gb"].cumsum() - sdf["is_gb"]
        sdf["season_fb_before"] = grp["is_fb"].cumsum() - sdf["is_fb"]
        sdf["season_bip_before"] = grp.cumcount()

        gb_num = sdf["prior_weight_bip"] * sdf["preseason_gb_rate"] + sdf["season_gb_before"]
        fb_num = sdf["prior_weight_bip"] * sdf["preseason_fb_rate"] + sdf["season_fb_before"]
        den = sdf["prior_weight_bip"] + sdf["season_bip_before"]
        sdf["pregame_gb_rate_pitcher"] = gb_num / den
        sdf["pregame_fb_rate_pitcher"] = fb_num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "pitcher", "season", "game_date",
                    "pregame_gb_rate_pitcher", "pregame_fb_rate_pitcher"]]


def player_game_gb_fb_rate_snapshot(pa: pd.DataFrame) -> pd.DataFrame:
    wide = build_pitcher_gb_fb_rate_by_season(pa)
    return wide.sort_values("at_bat_number").groupby(["game_pk", "pitcher"], as_index=False).nth(0)[
        ["game_pk", "pitcher", "season", "pregame_gb_rate_pitcher", "pregame_fb_rate_pitcher"]
    ]


def pitcher_hr_allowed_multiplier(rates: dict[str, float], pregame_gb_rate: float, pregame_fb_rate: float) -> float:
    """The odds-ratio multiplier to apply to the PITCHER's home_run rate ONLY
    -- direct pitcher-side analog of hr_share_multiplier above. existing_share
    is HR-allowed as a share of ALL balls in play the pitcher allows (not just
    hits, unlike the batter-side version -- HR-allowed is fundamentally about
    batted-ball type/outcome, the DIPS-2.0 framing this signal is built on,
    not a within-hits reallocation)."""
    bip_events = HIT_EVENTS + BIP_OUT_EVENTS
    bip_rate = sum(rates[o] for o in bip_events)
    if bip_rate <= 1e-9:
        return 1.0
    existing_share = np.clip(rates["home_run"] / bip_rate, HR_SHARE_PITCHER_CLIP_MIN, HR_SHARE_PITCHER_CLIP_MAX)
    predicted_share = (
        HR_SHARE_PITCHER_INTERCEPT
        + HR_SHARE_PITCHER_COEF_EXISTING * existing_share
        + HR_SHARE_PITCHER_COEF_GB * pregame_gb_rate
        + HR_SHARE_PITCHER_COEF_FB * pregame_fb_rate
    )
    predicted_share = np.clip(predicted_share, HR_SHARE_PITCHER_CLIP_MIN, HR_SHARE_PITCHER_CLIP_MAX)
    return odds(predicted_share) / odds(existing_share)


# NINTH LAYER -- GB/FB full-mix extension (2026-07-22, per an external
# reviewer's suggestion following the confirmed HR-allowed win above): the
# SAME stable pitcher input predicts the whole in-play outcome mix a pitcher
# allows, not just home runs -- groundball pitchers allow more double plays
# and fewer doubles/triples; flyball pitchers the reverse. Checked via the
# identical leakage-free walk-forward methodology (see
# scratchpad/fit_pitcher_gbfb_fullmix.py) across all 3 season pairs:
#   - double_play-share-among-BIP: R^2 GAIN EVEN LARGER than the HR result
#     (+0.04 to +0.11 across pairs) -- built below.
#   - double-share-among-BIP: real, consistent, smaller gain (+0.009 to
#     +0.011 across all 3 pairs) -- built below.
#   - triple-share-among-BIP: R^2 gain small AND inconsistent (+0.0003 in
#     one pair, near the noise floor given the tiny base R^2 for triples at
#     all) -- NOT built, same "no supporting evidence, don't guess" standard
#     used throughout this project for weak/inconsistent signals.
DOUBLE_SHARE_PITCHER_CLIP_MIN, DOUBLE_SHARE_PITCHER_CLIP_MAX = 0.02, 0.13
DOUBLE_PLAY_SHARE_PITCHER_CLIP_MIN, DOUBLE_PLAY_SHARE_PITCHER_CLIP_MAX = 0.005, 0.08

# Fit 2026-07-22, averaged across the 3 season pairs (same convention as HR above).
DOUBLE_SHARE_PITCHER_INTERCEPT = 0.09101
DOUBLE_SHARE_PITCHER_COEF_EXISTING = 0.05155
DOUBLE_SHARE_PITCHER_COEF_GB = -0.04795
DOUBLE_SHARE_PITCHER_COEF_FB = -0.03310

DOUBLE_PLAY_SHARE_PITCHER_INTERCEPT = 0.02556
DOUBLE_PLAY_SHARE_PITCHER_COEF_EXISTING = 0.06724
DOUBLE_PLAY_SHARE_PITCHER_COEF_GB = 0.02312
DOUBLE_PLAY_SHARE_PITCHER_COEF_FB = -0.02945
# NOTE on coefficient stability: unlike HR-allowed's coefficients (consistent
# sign across all 3 pairs), the double_play GB coefficient flips sign in one
# of the three individual pairs (+0.019, +0.067, -0.017) even though R^2
# improves in all three -- the AVERAGED coefficient used here is positive and
# matches baseball intuition (more groundballs -> more double-play chances),
# but this instability is flagged honestly rather than hidden.


def _pitcher_share_multiplier(rates: dict[str, float], outcome: str, pregame_gb_rate: float, pregame_fb_rate: float,
                               clip_min: float, clip_max: float,
                               intercept: float, coef_existing: float, coef_gb: float, coef_fb: float) -> float:
    """Shared mechanism for every pitcher-side GB/FB share-among-BIP
    multiplier (HR/double/double_play) -- odds-ratio of a predicted share
    (existing share + GB%/FB%, regression-blended) vs. the share this
    pitcher's existing Marcel-shrunk rates already imply, applied to
    `outcome` only. See pitcher_hr_allowed_multiplier for the full rationale."""
    bip_events = HIT_EVENTS + BIP_OUT_EVENTS
    bip_rate = sum(rates[o] for o in bip_events)
    if bip_rate <= 1e-9:
        return 1.0
    existing_share = np.clip(rates[outcome] / bip_rate, clip_min, clip_max)
    predicted_share = intercept + coef_existing * existing_share + coef_gb * pregame_gb_rate + coef_fb * pregame_fb_rate
    predicted_share = np.clip(predicted_share, clip_min, clip_max)
    return odds(predicted_share) / odds(existing_share)


def pitcher_double_allowed_multiplier(rates: dict[str, float], pregame_gb_rate: float, pregame_fb_rate: float) -> float:
    return _pitcher_share_multiplier(
        rates, "double", pregame_gb_rate, pregame_fb_rate,
        DOUBLE_SHARE_PITCHER_CLIP_MIN, DOUBLE_SHARE_PITCHER_CLIP_MAX,
        DOUBLE_SHARE_PITCHER_INTERCEPT, DOUBLE_SHARE_PITCHER_COEF_EXISTING,
        DOUBLE_SHARE_PITCHER_COEF_GB, DOUBLE_SHARE_PITCHER_COEF_FB,
    )


def pitcher_double_play_multiplier(rates: dict[str, float], pregame_gb_rate: float, pregame_fb_rate: float) -> float:
    return _pitcher_share_multiplier(
        rates, "double_play", pregame_gb_rate, pregame_fb_rate,
        DOUBLE_PLAY_SHARE_PITCHER_CLIP_MIN, DOUBLE_PLAY_SHARE_PITCHER_CLIP_MAX,
        DOUBLE_PLAY_SHARE_PITCHER_INTERCEPT, DOUBLE_PLAY_SHARE_PITCHER_COEF_EXISTING,
        DOUBLE_PLAY_SHARE_PITCHER_COEF_GB, DOUBLE_PLAY_SHARE_PITCHER_COEF_FB,
    )


def load_sprint_speed_by_season() -> pd.DataFrame:
    """Baseball Savant's Sprint Speed leaderboard (season-level, one row per
    qualifying player-season) -- see fetch_sprint_speed_seasons in fetch.py.
    Returns an empty frame if not yet fetched for any season, rather than
    raising -- callers already have a neutral-default fallback for missing
    speed data (see apply_contact_quality)."""
    frames = []
    for path in sorted(DATA_RAW.glob("sprint_speed_*.parquet")):
        season = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path, columns=["player_id", "sprint_speed"])
        df["season"] = season
        frames.append(df.rename(columns={"player_id": "batter"}))
    if not frames:
        return pd.DataFrame(columns=["batter", "season", "sprint_speed"])
    return pd.concat(frames, ignore_index=True)


def build_pregame_sprint_speed(target_season: int) -> pd.Series:
    """Each batter's walk-forward sprint speed for target_season: their most
    recently MEASURED prior season's value (a real season-aggregate
    biometric measurement, not something needing Marcel-style multi-season
    weighting or heavy shrinkage the way our own derived Statcast rates do).
    Returns a Series indexed by batter -- missing entries (no prior-season
    measurement, e.g. a rookie) should fall back to league average by the
    caller."""
    speed = load_sprint_speed_by_season()
    prior = speed[speed["season"] < target_season]
    if prior.empty:
        return pd.Series(dtype=float)
    latest = prior.sort_values("season").groupby("batter").last()
    return latest["sprint_speed"]


# Swing-tracking descriptions (any real swing -- contact or not) that carry
# bat_speed/swing_length/attack_angle: same set build_pitch_table.py's
# SWINGING_STRIKE_DESC/FOUL_DESC/HIT_INTO_PLAY_DESC union to, kept as a
# local literal here to avoid a cross-module import for one constant.
BAT_TRACKING_SWING_DESC = {"foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
                           "foul_tip", "foul_bunt", "missed_bunt", "bunt_foul_tip"}
BAT_TRACKING_MIN_SWINGS = 100  # confirmed on real data: 2023's mid-season bat-tracking
                                # rollout still leaves 427/655 batters with 100+ real swings
                                # that season -- comfortably usable, not just 2024-2025.


def load_bat_speed_by_season(seasons: list[int]) -> pd.DataFrame:
    """Each batter's real season-aggregate mean bat speed (mph, Statcast's
    bat-tracking system) -- unlike sprint speed, this comes from the raw
    per-pitch cache directly (no separate leaderboard file), same pattern as
    catcher_framing.py's compute_pitch_borderline_calls. Batters with fewer
    than BAT_TRACKING_MIN_SWINGS real (non-null) swings that season are
    dropped -- a small-sample season aggregate isn't worth trusting, same
    "check real sample size before including" discipline as elsewhere in
    this project."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["description", "bat_speed", "batter", "game_type"])
        df = df[(df["game_type"] == "R") & df["description"].isin(BAT_TRACKING_SWING_DESC)]
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["batter", "season", "mean_bat_speed"])
    swings = pd.concat(frames, ignore_index=True)
    agg = swings.groupby(["batter", "season"]).agg(
        mean_bat_speed=("bat_speed", "mean"), n_swings=("bat_speed", lambda s: s.notna().sum())
    ).reset_index()
    return agg[agg["n_swings"] >= BAT_TRACKING_MIN_SWINGS][["batter", "season", "mean_bat_speed"]]


def build_pregame_bat_speed(bat_speed_by_season: pd.DataFrame, target_season: int) -> pd.Series:
    """Each batter's walk-forward mean bat speed for target_season: their
    most recently MEASURED prior season's value -- same "season snapshot,
    no in-season blend" convention as build_pregame_sprint_speed (a real
    season-aggregate biomechanical measurement, not something needing
    Marcel-style multi-season weighting)."""
    prior = bat_speed_by_season[bat_speed_by_season["season"] < target_season]
    if prior.empty:
        return pd.Series(dtype=float)
    latest = prior.sort_values("season").groupby("batter").last()
    return latest["mean_bat_speed"]


# empirically-fit (2026-07-21, see module docstring): next-season real
# groundball-BACON regressed on this-season real groundball-BACON AND
# this-season sprint_speed together, across 725 combined batter-season pairs.
GROUNDBALL_SINGLE_INTERCEPT = 0.0066
GROUNDBALL_SINGLE_COEF_REAL = 0.1755
GROUNDBALL_SINGLE_COEF_SPEED = 0.00759

GROUNDBALL_BACON_CLIP_MIN = 0.05
GROUNDBALL_BACON_CLIP_MAX = 0.55


def groundball_single_multiplier(pregame_bacon_gb: float, pregame_sprint_speed: float,
                                  pregame_gb_rate: float) -> float:
    """The odds-ratio multiplier to apply to the "single" category, SCALED by
    this batter's own groundball rate (a fly-ball-heavy hitter gets almost no
    adjustment; a groundball-heavy hitter gets close to the full effect) --
    see apply_contact_quality."""
    existing = np.clip(pregame_bacon_gb, GROUNDBALL_BACON_CLIP_MIN, GROUNDBALL_BACON_CLIP_MAX)
    predicted = (
        GROUNDBALL_SINGLE_INTERCEPT + GROUNDBALL_SINGLE_COEF_REAL * existing
        + GROUNDBALL_SINGLE_COEF_SPEED * pregame_sprint_speed
    )
    predicted = np.clip(predicted, GROUNDBALL_BACON_CLIP_MIN, GROUNDBALL_BACON_CLIP_MAX)
    full_mult = odds(predicted) / odds(existing)
    gb_rate = np.clip(pregame_gb_rate, 0.0, 1.0)
    return 1.0 + gb_rate * (full_mult - 1.0)


# SIXTH LAYER (2026-07-22): pitcher "stuff" -- raw fastball velocity and
# spin rate (Statcast pitch-flight physics, ~99.2-99.6% coverage, far better
# than bat-tracking's rollout). This REVISITS pitcher-side contact/K quality
# after this project's earlier CSW% investigation (task #27) found that
# outcome-DERIVED pitcher signal (called-strike+whiff%) doesn't beat
# outcome-based K% -- but raw physical stuff (velocity/spin) is a genuinely
# DIFFERENT kind of signal, same "physical data, not another way of
# reslicing outcomes" distinction that made bat speed work for batters.
# Leakage-free (2 season-pairs, 2023->2024/2024->2025, n=776 combined
# pitcher-season pairs), tested against the ACTUAL production signal
# (true_talent.py's Marcel-shrunk, age-adjusted preseason K-rate, not a raw
# single-season rate): R^2 0.396 (existing alone) -> 0.409 (+fastball velo)
# -> 0.419 (+velo+spin), a real ~6% relative incremental gain, comparable in
# magnitude to bat speed's own HR-share gain. Real-data limit test (n=642
# real pitchers) came back clean on the first try -- multiplier range
# 0.81-1.23, mean 0.98, no blowup (unlike the HR-share-with-bat-speed
# attempt above, which needed a clip-floor compromise this session declined
# to make).
FASTBALL_TYPES_FOR_STUFF = {"FF", "SI", "FC", "FA"}  # same set as pitch_groups.py's FASTBALL_TYPES

PITCHER_STUFF_K_INTERCEPT = -0.240952
PITCHER_STUFF_K_COEF_EXISTING = 0.699641
PITCHER_STUFF_K_COEF_VELO = 0.00233966
PITCHER_STUFF_K_COEF_SPIN = 0.0000380032

# Real fastball velocity range 62.6-101.8 mph (1st/99th pctile 87.0-99.5),
# spin rate 1579-2791 rpm (1st/99th 1938-2645) -- both comfortably inside
# normal MLB ranges, no extreme small-sample artifacts found. K-rate clip
# matches this project's real observed pitcher K-rate range.
PITCHER_STUFF_K_CLIP_MIN, PITCHER_STUFF_K_CLIP_MAX = 0.05, 0.45


def load_fastball_stuff_by_season(seasons: list[int]) -> pd.DataFrame:
    """Each pitcher's real season-aggregate mean fastball velocity and spin
    rate (Statcast pitch-flight physics, straight from the raw per-pitch
    cache -- same pattern as load_bat_speed_by_season). Fastballs classified
    the same way catcher_framing.py's/pitch_groups.py's FASTBALL_TYPES
    already does, duplicated here as a plain literal to avoid importing
    pitch_groups.py's swing/count helpers for one set."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["release_speed", "release_spin_rate", "pitch_type", "pitcher", "game_type"])
        df = df[(df["game_type"] == "R") & df["pitch_type"].isin(FASTBALL_TYPES_FOR_STUFF)]
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["pitcher", "season", "mean_fb_velo", "mean_fb_spin"])
    fb = pd.concat(frames, ignore_index=True)
    agg = fb.groupby(["pitcher", "season"]).agg(
        mean_fb_velo=("release_speed", "mean"),
        mean_fb_spin=("release_spin_rate", "mean"),
        n=("release_speed", lambda s: s.notna().sum()),
    ).reset_index()
    return agg[agg["n"] >= 100][["pitcher", "season", "mean_fb_velo", "mean_fb_spin"]]


def build_pregame_fastball_stuff(stuff_by_season: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Each pitcher's walk-forward fastball velocity/spin for target_season:
    their most recently MEASURED prior season's values -- same "season
    snapshot, no in-season blend" convention as build_pregame_sprint_speed/
    build_pregame_bat_speed. Returns a DataFrame indexed by pitcher (needs
    both columns together, unlike the single-Series batter snapshots)."""
    prior = stuff_by_season[stuff_by_season["season"] < target_season]
    if prior.empty:
        return pd.DataFrame(columns=["mean_fb_velo", "mean_fb_spin"])
    latest = prior.sort_values("season").groupby("pitcher").last()
    return latest[["mean_fb_velo", "mean_fb_spin"]]


def pitcher_stuff_k_multiplier(existing_k_rate: float, pregame_fb_velo: float, pregame_fb_spin: float) -> float:
    """The odds-ratio multiplier to apply to a PITCHER's own "strikeout"
    rate: fastball velocity/spin blended with this pitcher's existing
    Marcel-shrunk K-rate via the fitted regression above -- same
    architecture as contact_quality_multiplier/hr_share_multiplier, just on
    the pitcher side (a genuinely new application -- every prior signal in
    this file has been explicitly batter-only)."""
    existing = np.clip(existing_k_rate, PITCHER_STUFF_K_CLIP_MIN, PITCHER_STUFF_K_CLIP_MAX)
    predicted = (
        PITCHER_STUFF_K_INTERCEPT
        + PITCHER_STUFF_K_COEF_EXISTING * existing
        + PITCHER_STUFF_K_COEF_VELO * pregame_fb_velo
        + PITCHER_STUFF_K_COEF_SPIN * pregame_fb_spin
    )
    predicted = np.clip(predicted, PITCHER_STUFF_K_CLIP_MIN, PITCHER_STUFF_K_CLIP_MAX)
    return odds(predicted) / odds(existing)


def apply_pitcher_stuff(rates: dict[str, float], pregame_fb_velo: float | None,
                         pregame_fb_spin: float | None) -> dict[str, float]:
    """Returns a NEW rates dict with "strikeout" scaled by the fastball-
    stuff-based multiplier -- other categories untouched. Both args needed
    to apply (a missing measurement is a no-op, not a guess); the caller
    passes pitcher-side signals here, never batter-side (mirrors
    apply_contact_quality's own batter-only scope, just for the other
    player role)."""
    adjusted = dict(rates)
    if (pregame_fb_velo is not None and not pd.isna(pregame_fb_velo)
            and pregame_fb_spin is not None and not pd.isna(pregame_fb_spin)):
        adjusted["strikeout"] = adjusted["strikeout"] * pitcher_stuff_k_multiplier(
            adjusted["strikeout"], pregame_fb_velo, pregame_fb_spin
        )
    return adjusted


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building walk-forward xBACON estimates...", flush=True)
    result = build_pregame_xbacon(pa)

    print("\n=== xBACON pregame-estimate distribution, most recent season ===")
    latest_season = result["season"].max()
    latest = result[result["season"] == latest_season]
    print(latest["pregame_xbacon"].describe())

    print("\n=== leakage-free validation: does pregame_xbacon predict this batter's FUTURE real BACON? ===")
    bip = pa[pa["estimated_ba_using_speedangle"].notna()].copy()
    bip["is_hit"] = bip["outcome"].isin(["single", "double", "triple", "home_run"]).astype(float)
    merged = result.merge(bip[["game_pk", "at_bat_number", "batter", "is_hit"]], on=["game_pk", "at_bat_number", "batter"])
    test = merged[merged["season"].isin([2024, 2025])]
    mae = (test["pregame_xbacon"] - test["is_hit"]).abs().mean()
    naive_mae = (test["is_hit"].mean() - test["is_hit"]).abs().mean()
    print(f"BIP-level MAE: pregame_xbacon={mae:.4f}  naive(league rate)={naive_mae:.4f}")

    print("\nbuilding walk-forward barrel-rate estimates...", flush=True)
    barrel_result = build_pregame_barrel_rate(pa)
    print("\n=== barrel-rate pregame-estimate distribution, most recent season ===")
    latest_barrel = barrel_result[barrel_result["season"] == barrel_result["season"].max()]
    print(latest_barrel["pregame_barrel_rate"].describe())

    print("\n=== limit test: hr_share_multiplier value distribution across real batters ===")
    from src.models.game_simulator import OUTCOMES, build_wide_pregame_rates
    from src.models.validate_game_simulator import player_game_snapshot

    batter_wide = build_wide_pregame_rates(pa, "batter")
    batter_snap = player_game_snapshot(batter_wide, "batter")
    latest_rates = batter_snap.sort_values("game_pk").groupby("batter").last()
    latest_barrel_by_batter = barrel_result.sort_values("game_pk").groupby("batter").last()["pregame_barrel_rate"]

    mults = []
    for pid, row in latest_rates.iterrows():
        if pid not in latest_barrel_by_batter.index:
            continue
        rates = {o: row[f"pregame_rate_{o}"] for o in OUTCOMES}
        mults.append(hr_share_multiplier(rates, latest_barrel_by_batter.loc[pid]))
    mults = pd.Series(mults)
    print(f"n batters: {len(mults)}")
    print(mults.describe())
