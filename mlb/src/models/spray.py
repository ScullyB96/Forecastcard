"""Individual batter pull tendency (spray angle), and how it modulates the
wind-direction effect beyond what stand (L/R) alone captures.

Confirmed real and worth building before starting (2026-07-20): among LEFTY
batters specifically, tercile-splitting by pull rate shows low-pull hitters
with essentially no home_run-rate gap between wind blowing to their pull side
(out_RF) vs. their opposite field (out_LF) (-0.13pp), while high-pull lefties
show a real, meaningfully larger gap (+0.46pp) -- pull-heavy hitters benefit
specifically MORE from wind blowing to their own pull field than balanced/
oppo hitters do, a real signal the stand-only weather factor can't express.
Same pattern (smaller magnitude) on the righty side. Pull rate itself is a
real, stable individual trait, not noise -- confirmed year-over-year
correlation 0.67-0.69 across two season pairs, comparable to this project's
other trusted rate metrics.

Spray angle computed from Statcast's own hit-ball coordinates (hc_x, hc_y,
only populated on a ball in play -- see build_pa_table.py) via the standard
formula anchored at home plate's approximate Gameday-coordinate position
(125.42, 198.27): raw_angle = atan2(hc_x-125.42, 198.27-hc_y), positive =
toward the 1B/RF side, negative = toward the 3B/LF side. pull_angle flips
sign for R-handed batters so positive ALWAYS means "pulled toward this
batter's own natural power side" regardless of hand -- a R-handed batter
pulls toward LF (the raw-negative side), a L-handed batter pulls toward RF
(the raw-positive side).

Walk-forward projection mirrors true_talent.py's Marcel two-level structure
(same MARCEL_WEIGHTS recency weights, same preseason-prior + in-season-blend
design) but on the BATTED-BALL subset specifically (denominator = batted
balls, not all PAs, since pull tendency is a property of contact, not of
every plate appearance) -- true_talent.py's build_pregame_rates couldn't be
reused directly since it assumes the outcome category lives in `pa["outcome"]`
with a denominator of ALL PAs.

PULL_STABILIZATION_BATTED_BALLS estimated the same way intent_walk's constant
was: split-half reliability was still rising with sample size within a single
season (never reaching Carleton's usual 0.70 threshold), so the constant is
extrapolated via reliability=N/(N+K) from the largest/least-noisy point
measured (N=100 batted balls -> r=0.60 -> K~=67, rounded up to 100 for the
same conservatism reason as intent_walk -- the trend was still rising).

PULL_TERCILE_CUTOFFS are FIXED (not re-fit per season) -- derived once from
the full 2023-2025 population's within-stand pull-rate distribution, the same
"baked-in validated constant" pattern as REST_AVAILABILITY_WEIGHT elsewhere in
this project -- re-fitting percentile boundaries per season using that
season's own peer population would risk a subtle look-ahead (this season's
league-wide pull-rate distribution isn't knowable in advance the way a fixed,
already-observed historical boundary is)."""

import numpy as np
import pandas as pd

from src.models.bullpen import nearest_prior_pitcher_snapshot
from src.models.park_factors import PULLED_CONTACT_QUALITY_CODES, PULLED_CONTACT_SPRAY_THRESHOLD_DEG
from src.models.true_talent import MARCEL_WEIGHTS

PULL_ANGLE_THRESHOLD_DEG = 15  # standard sabermetric "pulled" cutoff
PULL_STABILIZATION_BATTED_BALLS = 100

PULL_TERCILE_CUTOFFS = {
    "L": (0.447, 0.486),
    "R": (0.401, 0.449),
}


def compute_pull_angle(pa: pd.DataFrame) -> pd.Series:
    """pull_angle in degrees for each row with real hc_x/hc_y -- NaN for any
    PA that wasn't a ball in play. Positive = pulled toward this batter's own
    natural power side (R-handed pulls toward LF, L-handed toward RF)."""
    raw_angle = np.degrees(np.arctan2(pa["hc_x"] - 125.42, 198.27 - pa["hc_y"]))
    return np.where(pa["stand"] == "R", -raw_angle, raw_angle)


def pull_tercile(pull_rate: float, stand: str) -> str:
    """"low_pull"/"mid_pull"/"high_pull" for a given shrunk pull rate and
    batter stand -- switch-hitters ("S") use the R-side cutoffs as a
    reasonable default (no real basis to pick one side over the other for a
    switch-hitter's overall pull tendency across both stands combined)."""
    lo, hi = PULL_TERCILE_CUTOFFS.get(stand, PULL_TERCILE_CUTOFFS["R"])
    if pull_rate < lo:
        return "low_pull"
    if pull_rate < hi:
        return "mid_pull"
    return "high_pull"


def build_pull_rate_by_season(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe (no leakage) estimated pull rate for every BATTED
    BALL, as of just before it -- preseason Marcel-style prior blended with
    in-season-to-date actuals, same structure as
    true_talent.build_pregame_rates. One row per (game_pk, at_bat_number,
    batter): season, game_date, pregame_pull_rate. Only covers PAs that were
    actually a ball in play -- see attach_pull_tercile to carry this forward
    onto every PA, including non-batted-ball ones."""
    bb = pa.dropna(subset=["hc_x", "hc_y"]).copy()
    bb["pull_angle"] = compute_pull_angle(bb)
    bb["is_pulled"] = (bb["pull_angle"] > PULL_ANGLE_THRESHOLD_DEG).astype(int)
    bb = bb.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"])

    K = PULL_STABILIZATION_BATTED_BALLS
    out_frames = []
    for season, sdf in bb.groupby("season"):
        prior = bb[bb["season"] < season]
        if prior.empty:
            league_rate = bb.loc[bb["season"] == season, "is_pulled"].mean()
            priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])
        else:
            league_rate = prior["is_pulled"].mean()
            season_agg = prior.groupby(["batter", "season"]).agg(
                bb=("is_pulled", "size"), events=("is_pulled", "sum")
            ).reset_index()
            priors = []
            for batter, g in season_agg.groupby("batter"):
                g = g.set_index("season")
                num, den, raw_bb = 0.0, 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num += w * g.loc[s, "events"]
                        den += w * g.loc[s, "bb"]
                        raw_bb += g.loc[s, "bb"]
                if den == 0:
                    continue
                priors.append({"batter": batter, "prior_bb": den, "prior_events": num, "raw_prior_bb": raw_bb})
            priors_df = pd.DataFrame(priors, columns=["batter", "prior_bb", "prior_events", "raw_prior_bb"])
            if not priors_df.empty:
                # RELIABILITY must use RAW (unweighted) prior_bb -- same
                # units-mismatch bug true_talent.py's reliability formula
                # was fixed for (task #53), copy-pasted into this function
                # unfixed until now (task #160 correctness audit, 2026-07-26).
                # The rate estimate itself keeps using the weighted sum.
                priors_df["reliability"] = priors_df["raw_prior_bb"] / (priors_df["raw_prior_bb"] + K)
                raw_rate = priors_df["prior_events"] / priors_df["prior_bb"]
                priors_df["preseason_rate"] = (
                    priors_df["reliability"] * raw_rate + (1 - priors_df["reliability"]) * league_rate
                )
                priors_df["prior_weight_bb"] = np.minimum(priors_df["raw_prior_bb"], K)
                priors_df = priors_df[["batter", "preseason_rate", "prior_weight_bb"]]
            else:
                priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])

        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_rate"] = sdf["preseason_rate"].fillna(league_rate)
        sdf["prior_weight_bb"] = sdf["prior_weight_bb"].fillna(K)

        grp = sdf.groupby("batter")
        sdf["season_events_before"] = grp["is_pulled"].cumsum() - sdf["is_pulled"]
        sdf["season_bb_before"] = grp.cumcount()

        num = sdf["prior_weight_bb"] * sdf["preseason_rate"] + sdf["season_events_before"]
        den = sdf["prior_weight_bb"] + sdf["season_bb_before"]
        sdf["pregame_pull_rate"] = num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "game_date", "pregame_pull_rate"]]


def attach_pull_rate(pa: pd.DataFrame, pull_rate_pa: pd.DataFrame) -> pd.DataFrame:
    """Carries each batter's walk-forward pull-rate estimate (only directly
    computed at batted-ball PAs) forward onto EVERY PA, including strikeouts/
    walks/etc. that have no hc_x/hc_y of their own -- a batter's underlying
    pull tendency doesn't reset between plate appearances, so using the most
    recently KNOWN value (never a future one) for a non-batted-ball PA is
    walk-forward-safe, not a leak. Batters with no pull-rate history yet
    (true rookies, or before their first career batted ball) get NaN --
    callers should treat that as "use the league-average tercile" rather than
    guessing a side."""
    pa = pa.sort_values(["batter", "game_date", "game_pk", "at_bat_number"]).copy()
    pa["_pa_seq"] = pa.groupby("batter").cumcount()
    rate_with_seq = pa[["batter", "game_pk", "at_bat_number", "_pa_seq"]].merge(
        pull_rate_pa[["batter", "game_pk", "at_bat_number", "pregame_pull_rate"]],
        on=["batter", "game_pk", "at_bat_number"], how="inner",
    )
    merged = pd.merge_asof(
        pa.sort_values("_pa_seq"),
        rate_with_seq[["batter", "_pa_seq", "pregame_pull_rate"]].sort_values("_pa_seq"),
        on="_pa_seq", by="batter", direction="backward",
    )
    return merged.drop(columns=["_pa_seq"])


def attach_pull_tercile_column(pa: pd.DataFrame, pull_rate_pa: pd.DataFrame) -> pd.DataFrame:
    """attach_pull_rate, then collapse pregame_pull_rate (+ stand) into a
    ready-to-use `pull_tercile` column -- "mid_pull" (the neutral default)
    for any PA with no pull-rate history yet (a true rookie, or before a
    batter's first career batted ball), rather than guessing a side."""
    result = attach_pull_rate(pa, pull_rate_pa)
    result["pull_tercile"] = [
        pull_tercile(rate, stand) if pd.notna(rate) else "mid_pull"
        for rate, stand in zip(result["pregame_pull_rate"], result["stand"])
    ]
    return result.drop(columns=["pregame_pull_rate"])


def resolve_batter_pull_tercile(snapshot_with_dates: pd.DataFrame, batter_id, stand: str, game_date, game_pk: int) -> str:
    """This batter's pull tercile as of game_pk/game_date -- exact match if
    this batter had a batted ball in this specific game, else the most
    recent known prior estimate (see nearest_prior_pitcher_snapshot, which
    auto-detects the "batter" id column), else "mid_pull" (the neutral
    default for a batter with no pull-rate history yet)."""
    row = nearest_prior_pitcher_snapshot(snapshot_with_dates, batter_id, game_date, game_pk)
    if row is None:
        return "mid_pull"
    return pull_tercile(row["pregame_pull_rate"], stand)


def build_pull_rate_snapshot(pa: pd.DataFrame, pull_rate_pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter): pregame_pull_rate as of the start of
    that game (the most recent known estimate strictly before or at this
    game's first batted ball) -- mirrors validate_game_simulator's
    player_game_snapshot pattern. Games where this batter had zero batted
    balls (all Ks/BBs/HBP that game) simply have no row here -- callers
    already have a generic as-of fallback for exactly this shape
    (nearest_prior_pitcher_snapshot in bullpen.py, which auto-detects a
    "batter" column) rather than needing a bespoke one here."""
    with_rate = attach_pull_rate(pa, pull_rate_pa)
    first = with_rate.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)
    return first[["game_pk", "batter", "season", "game_date", "pregame_pull_rate"]].dropna(subset=["pregame_pull_rate"])


# EIGHTH addition (2026-08-04): a batter's own walk-forward rate of producing
# PULLED, QUALITY-CONTACT batted balls specifically -- not "pulled" alone
# (pregame_pull_rate above) or "barrel" alone (expected_stats.build_pregame_
# barrel_rate), but the JOINT event, using the exact same definition as
# park_factors.build_pulled_barrel_park_factor_by_stand's own target subset
# (PULLED_CONTACT_SPRAY_THRESHOLD_DEG + PULLED_CONTACT_QUALITY_CODES, imported
# from there rather than re-derived, so the batter-level rate and the park
# factor it's meant to be blended against are measuring literally the same
# thing). This is the natural blend weight for that park factor: a batter
# whose own contact profile rarely looks like this subset shouldn't get much
# of a park's PULLED-QUALITY-CONTACT-specific number, regardless of his stand.
#
# STILL PROVISIONAL (2026-08-04, same status as the park factor itself): not
# wired into build_profile/game_simulator.py yet, pending the held-out
# calibration + full-stack CRN backtest scoped alongside this whole mechanism.
PULLED_QUALITY_STABILIZATION_BATTED_BALLS = 100  # reuses PULL_STABILIZATION_
# BATTED_BALLS' own K=100 as a starting default (same reasoning: this project's
# usual reliability-curve-fit approach needs real split-half data this new,
# rarer joint event doesn't have yet) -- NOT independently validated.


def _is_pulled_quality_contact(pa: pd.DataFrame) -> pd.Series:
    """Boolean Series, True where a batted ball is BOTH quality contact
    (Statcast launch_speed_angle in PULLED_CONTACT_QUALITY_CODES) AND pulled
    toward this batter's own natural power side (raw spray angle past
    PULLED_CONTACT_SPRAY_THRESHOLD_DEG, sign per stand) -- mirrors park_
    factors._pulled_quality_contact's own filter logic exactly, applied here
    to a batted-ball-only frame that already has hc_x/hc_y."""
    raw_angle = np.degrees(np.arctan2(pa["hc_x"] - 125.42, 198.27 - pa["hc_y"]))
    pulled = np.where(
        pa["stand"] == "L",
        raw_angle > PULLED_CONTACT_SPRAY_THRESHOLD_DEG,
        raw_angle < -PULLED_CONTACT_SPRAY_THRESHOLD_DEG,
    )
    quality = pa["launch_speed_angle"].astype("float64").isin(PULLED_CONTACT_QUALITY_CODES)
    return quality.to_numpy() & pulled


def build_pulled_quality_contact_rate(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe (no leakage) estimated pulled-quality-contact rate
    for every BATTED BALL, as of just before it -- same Marcel two-level
    (preseason prior + in-season blend) structure as build_pull_rate_by_season,
    swapping in _is_pulled_quality_contact as the event instead of plain
    "pulled." Denominator is batted balls (a property of contact), matching
    build_pull_rate_by_season's own convention, not build_pregame_barrel_
    rate's balls-in-play-via-launch_speed_angle.notna() convention -- both are
    "did this PA produce a real hc_x/hc_y," just named differently per file;
    kept consistent with THIS file's own existing convention."""
    bb = pa.dropna(subset=["hc_x", "hc_y"]).copy()
    bb["is_event"] = _is_pulled_quality_contact(bb).astype(int)
    bb = bb.sort_values(["batter", "season", "game_date", "game_pk", "at_bat_number"])

    K = PULLED_QUALITY_STABILIZATION_BATTED_BALLS
    out_frames = []
    for season, sdf in bb.groupby("season"):
        prior = bb[bb["season"] < season]
        if prior.empty:
            league_rate = bb.loc[bb["season"] == season, "is_event"].mean()
            priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])
        else:
            league_rate = prior["is_event"].mean()
            season_agg = prior.groupby(["batter", "season"]).agg(
                bb=("is_event", "size"), events=("is_event", "sum")
            ).reset_index()
            priors = []
            for batter, g in season_agg.groupby("batter"):
                g = g.set_index("season")
                num, den, raw_bb = 0.0, 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num += w * g.loc[s, "events"]
                        den += w * g.loc[s, "bb"]
                        raw_bb += g.loc[s, "bb"]
                if den == 0:
                    continue
                priors.append({"batter": batter, "prior_bb": den, "prior_events": num, "raw_prior_bb": raw_bb})
            priors_df = pd.DataFrame(priors, columns=["batter", "prior_bb", "prior_events", "raw_prior_bb"])
            if not priors_df.empty:
                priors_df["reliability"] = priors_df["raw_prior_bb"] / (priors_df["raw_prior_bb"] + K)
                raw_rate = priors_df["prior_events"] / priors_df["prior_bb"]
                priors_df["preseason_rate"] = (
                    priors_df["reliability"] * raw_rate + (1 - priors_df["reliability"]) * league_rate
                )
                priors_df["prior_weight_bb"] = np.minimum(priors_df["raw_prior_bb"], K)
                priors_df = priors_df[["batter", "preseason_rate", "prior_weight_bb"]]
            else:
                priors_df = pd.DataFrame(columns=["batter", "preseason_rate", "prior_weight_bb"])

        sdf = sdf.merge(priors_df, on="batter", how="left")
        sdf["preseason_rate"] = sdf["preseason_rate"].fillna(league_rate)
        sdf["prior_weight_bb"] = sdf["prior_weight_bb"].fillna(K)

        grp = sdf.groupby("batter")
        sdf["season_events_before"] = grp["is_event"].cumsum() - sdf["is_event"]
        sdf["season_bb_before"] = grp.cumcount()

        num = sdf["prior_weight_bb"] * sdf["preseason_rate"] + sdf["season_events_before"]
        den = sdf["prior_weight_bb"] + sdf["season_bb_before"]
        sdf["pregame_pulled_quality_rate"] = num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "at_bat_number", "batter", "season", "game_date", "pregame_pulled_quality_rate"]]


def pulled_power_weight(pregame_pulled_quality_rate: float, league_rate: float) -> float:
    """Blend weight `w` in [0, 1] for how much of a batter's own contact
    profile matches the pulled-quality-contact subset build_pulled_barrel_
    park_factor_by_stand is defined on -- the blend weight for
    effective_park_hr_factor = pooled^(1-w) * pulled_barrel_specific^w
    (2026-08-04 scoping). Simple, transparent, linear-in-relative-rate
    design: a batter at exactly league-average gets w=0.5 (an even blend,
    reflecting genuine uncertainty about how much his profile resembles the
    target subset); 2x league average or more saturates at w=1.0 (fully the
    park's pulled-specific number); at or below 0 gets w=0.0 (fully the
    plain pooled park factor). NOT the only reasonable design -- a real
    tercile-cutoff approach (matching pull_tercile's own convention) was
    considered but rejected for now: this metric is too new to have a real
    fitted population distribution the way PULL_TERCILE_CUTOFFS does (that
    dict's own docstring: "derived once from the full population's...
    distribution") -- revisit once this mechanism has real validation data
    to fit cutoffs against, same as PULL_TERCILE_CUTOFFS itself once did."""
    if league_rate <= 0 or pd.isna(pregame_pulled_quality_rate):
        return 0.5
    ratio = pregame_pulled_quality_rate / league_rate
    return float(np.clip(ratio / 2.0, 0.0, 1.0))


def attach_pulled_quality_rate(pa: pd.DataFrame, rate_pa: pd.DataFrame) -> pd.DataFrame:
    """Same carry-forward-to-every-PA logic as attach_pull_rate (a batter's
    underlying pulled-quality-contact tendency doesn't reset between PAs, so
    the most recently KNOWN value carries forward onto non-batted-ball PAs
    too), for pregame_pulled_quality_rate instead of pregame_pull_rate."""
    pa = pa.sort_values(["batter", "game_date", "game_pk", "at_bat_number"]).copy()
    pa["_pa_seq"] = pa.groupby("batter").cumcount()
    rate_with_seq = pa[["batter", "game_pk", "at_bat_number", "_pa_seq"]].merge(
        rate_pa[["batter", "game_pk", "at_bat_number", "pregame_pulled_quality_rate"]],
        on=["batter", "game_pk", "at_bat_number"], how="inner",
    )
    merged = pd.merge_asof(
        pa.sort_values("_pa_seq"),
        rate_with_seq[["batter", "_pa_seq", "pregame_pulled_quality_rate"]].sort_values("_pa_seq"),
        on="_pa_seq", by="batter", direction="backward",
    )
    return merged.drop(columns=["_pa_seq"])


def build_pulled_quality_contact_snapshot(pa: pd.DataFrame, rate_pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter): pregame_pulled_quality_rate as of the
    start of that game -- mirrors build_pull_rate_snapshot's own pattern
    (carry-forward via attach_pulled_quality_rate first, then take each
    game's first PA, so a batter whose first PA that game wasn't itself a
    batted ball still gets the correct as-of estimate)."""
    with_rate = attach_pulled_quality_rate(pa, rate_pa)
    first = with_rate.sort_values("at_bat_number").groupby(["game_pk", "batter"], as_index=False).nth(0)
    return first[["game_pk", "batter", "pregame_pulled_quality_rate"]].dropna(subset=["pregame_pulled_quality_rate"])
