"""Real physical park geometry (fence distance + height, by direction) as a
batted-ball-level, physics-adjacent alternative to every outcome-statistics-
based park factor elsewhere in this project (park_factors.py).

2026-08-04: built after a deep-research pass confirmed (a) Baseball Savant's
OWN official Park Factors leaderboard never isolates a wall-height component
at all -- even MLB's own headline park-factor number is purely outcome-based
(Temperature/Elevation/Roof/Environment, see Tom Tango's own methodology
writeup) -- and (b) Statcast's separate "Adjusted Expected Home Runs"
(adj_xhr) tool solves this differently: for every real batted ball, replay
its actual trajectory against each of the 30 parks' REAL fence geometry to
decide home-run-or-not, rather than inferring a park's effect from aggregate
outcome frequency. This module is our own from-scratch version of that
second approach, built from data/raw/park_dimensions_2026.csv (real,
current, official Statcast dimensions data, scraped 2026-08-04 via the
Baseball Savant Park Factors leaderboard's own "Dimensions" toggle -- see
that CSV's own commit for provenance) plus the hc_x/hc_y/launch_angle
columns already in the PA table.

Real, concrete confirmation this matters: Fenway Park's own fence height is
NOT one number -- LF Line=37ft (the actual Green Monster), LF Gap=37ft,
CF=17ft, RF Gap=4ft, RF Line=4ft. Yankee Stadium, by contrast, has a
perfectly uniform 8ft wall everywhere -- its short right-field "porch" is
PURELY a distance effect (RF Line 313ft vs LF Line 318ft), with zero height
interaction, unlike Fenway. This module's 5-point-by-direction fence model
is the first place in this codebase that can tell these two apart.

Phase 1 (this module): a DISTANCE-only fence-clearance classifier -- did
this real batted ball's own empirically-measured hit distance exceed the
real fence distance at its own spray angle, at a given park. Height is
folded in only as a coarse, launch-angle-conditioned correction (see
GREEN_MONSTER_LAUNCH_ANGLE_BAND below), not a full trajectory-vs-height
physics integration (that would require the kind of drag+Magnus model Alan
Nathan's trajectory calculator implements -- a much larger undertaking,
flagged as a real Phase 2 if this Phase 1 version proves valuable)."""

import numpy as np
import pandas as pd

from src.models.park_factors import PULLED_CONTACT_QUALITY_CODES
from src.utils.paths import DATA_RAW

# Savant's own team display names (as scraped) -> this project's own
# abbreviation convention (see schedule_*.parquet's home_team column).
# "Field of Dreams" and "Journey Bank Ballpark" (one-off exhibition-game
# venues, not a real team's home park) are deliberately left unmapped --
# every real regular-season game maps to one of the 30 teams below.
SAVANT_TEAM_TO_ABBREV = {
    "Rockies": "COL", "D-backs": "AZ", "Royals": "KC", "Cardinals": "STL",
    "Tigers": "DET", "Rangers": "TEX", "Angels": "LAA", "Marlins": "MIA",
    "Rays": "TB", "Braves": "ATL", "Pirates": "PIT", "Athletics": "ATH",
    "Brewers": "MIL", "Giants": "SF", "Nationals": "WSH", "Padres": "SD",
    "Orioles": "BAL", "Twins": "MIN", "Mets": "NYM", "Dodgers": "LAD",
    "Yankees": "NYY", "White Sox": "CWS", "Cubs": "CHC", "Mariners": "SEA",
    "Guardians": "CLE", "Blue Jays": "TOR", "Phillies": "PHI", "Reds": "CIN",
    "Astros": "HOU", "Red Sox": "BOS",
}

# The 5 sampled points' real angle-from-center, using this project's own
# spray_angle sign convention (see park_factors._spray_angle_deg / spray.
# compute_pull_angle): positive = toward the 1B/RF side, negative = toward
# the 3B/LF side, 0 = straight to center. EQUALLY SPACED across the 90-degree
# fair-territory arc (foul line to foul line) -- a real, flagged simplifying
# assumption (Baseball Savant's own dimensions table doesn't publish the
# exact angle of its own "Gap" points; equal spacing is the standard
# convention this kind of stadium-diagram compilation has used elsewhere,
# e.g. the FanGraphs polar-coordinate fence-equation project surfaced by the
# 2026-08-04 research pass) -- NOT independently verified per-park.
DIRECTION_ANGLES_DEG = {
    "lf_line": -45.0, "lf_gap": -22.5, "cf": 0.0, "rf_gap": 22.5, "rf_line": 45.0,
}

# Green Monster / tall-wall launch-angle correction (2026-08-04): confirmed
# directly on our own data (Fenway righty-pulled solid-contact HR rate:
# 13.6% at 15-25deg launch angle vs 26.9% elsewhere -- roughly HALVED) that a
# tall wall specifically suppresses HR conversion for LOW-ARC line drives
# that would clear a normal fence but not a tall one, while HIGHER launch
# angles are unaffected or even helped by the park's short distance. This is
# a coarse proxy for "does this ball's arc clear the wall's height," not a
# real trajectory integration.
GREEN_MONSTER_LAUNCH_ANGLE_BAND = (15, 25)  # degrees; matches the empirical band above
TALL_WALL_HEIGHT_FT = 15.0  # a wall taller than this is "tall enough to matter" for the
                            # low-launch-angle suppression correction below -- provisional,
                            # not independently tuned; picked as clearly above the ~6-13ft
                            # range most parks' walls fall in, per park_dimensions_2026.csv.
TALL_WALL_SUPPRESSION_MULT = 0.5  # matches the observed ~halving effect directly


def load_park_dimensions() -> pd.DataFrame:
    """Real, current (2026) fence distance + height by direction for all 30
    parks, indexed by this project's own team abbreviation. See this
    module's docstring for provenance (scraped from Baseball Savant's own
    Park Factors leaderboard 'Dimensions' toggle, 2026-08-04)."""
    df = pd.read_csv(DATA_RAW / "park_dimensions_2026.csv")
    df = df[df["team"].isin(SAVANT_TEAM_TO_ABBREV)].copy()
    df["abbrev"] = df["team"].map(SAVANT_TEAM_TO_ABBREV)
    return df.set_index("abbrev")


def _interp_at_angle(dims_row: pd.Series, spray_angle: float, suffix: str) -> float:
    """Piecewise-linear interpolation of a park's real 5-point fence
    distance or height (suffix='dist_ft' or 'height_ft') at an arbitrary
    spray angle, clipped to the [-45, 45] range the 5 real sampled points
    cover (a more extreme angle would already be foul territory)."""
    angles = [DIRECTION_ANGLES_DEG[k] for k in ("lf_line", "lf_gap", "cf", "rf_gap", "rf_line")]
    values = [dims_row[f"{k}_{suffix}"] for k in ("lf_line", "lf_gap", "cf", "rf_gap", "rf_line")]
    clipped = np.clip(spray_angle, angles[0], angles[-1])
    return float(np.interp(clipped, angles, values))


def fence_distance_at_angle(dims: pd.DataFrame, team: str, spray_angle: float) -> float | None:
    """This park's real fence distance (feet) at an arbitrary spray angle,
    interpolated between the 5 real sampled points. None for an unmapped
    team (e.g. a spring-training/exhibition venue)."""
    if team not in dims.index:
        return None
    return _interp_at_angle(dims.loc[team], spray_angle, "dist_ft")


def fence_height_at_angle(dims: pd.DataFrame, team: str, spray_angle: float) -> float | None:
    """This park's real fence height (feet) at an arbitrary spray angle,
    interpolated between the 5 real sampled points. None for an unmapped team."""
    if team not in dims.index:
        return None
    return _interp_at_angle(dims.loc[team], spray_angle, "height_ft")


def estimate_hit_distance(pa: pd.DataFrame, scale_ft_per_unit: float) -> pd.Series:
    """Real, empirical hit distance (feet) from Statcast's own hc_x/hc_y
    landing-coordinate fields, via the standard sabermetric transform (origin
    at home plate, same one park_factors._spray_angle_deg uses for angle) --
    NaN for any PA without real hc_x/hc_y (not a batted ball). scale_ft_per_
    unit converts the raw Gameday-coordinate pixel distance to feet -- see
    calibrate_scale_factor for how this project's own value was fit (NOT
    blindly reused from a public blog's commonly-cited ~2.5 figure)."""
    x = pa["hc_x"] - 125.42
    y = 198.27 - pa["hc_y"]
    raw = np.sqrt(x ** 2 + y ** 2)
    return raw * scale_ft_per_unit


REAL_AVG_HR_DISTANCE_FT = 400.0  # well-documented real MLB average home-run
# distance in the current Statcast era (widely reported ~400-410 ft) -- the
# grounded physical anchor calibrate_scale_factor fits against.


def calibrate_scale_factor(pa: pd.DataFrame, dims: pd.DataFrame) -> dict:
    """Fits scale_ft_per_unit so the mean estimated distance of REAL confirmed
    home runs matches REAL_AVG_HR_DISTANCE_FT -- a physically grounded
    anchor, not a fit against our own classifier's output. (An earlier
    version of this function instead maximized "fraction of real HRs
    classified as clearing their own park's fence" directly -- a degenerate
    objective, since that fraction increases monotonically with scale with
    no natural peak: you can always inflate distance further to make more
    things clear, right up to absurd, physically-impossible values. Anchoring
    to a real, independently-known physical quantity avoids this.)

    Returns {'scale': float, 'hr_clear_rate': float} -- hr_clear_rate is
    reported as a VALIDATION check (what fraction of real HRs correctly
    clear their own park's fence at this physically-grounded scale), not
    the thing being optimized. A high rate (~95%+) here is a real, useful
    confirmation the whole pipeline (hc_x/hc_y transform, 5-point fence
    interpolation, scale factor) hangs together; a low rate would mean
    something upstream is wrong."""
    hr = pa[(pa["outcome"] == "home_run") & pa["hc_x"].notna() & pa["hc_y"].notna()].copy()
    hr = hr[hr["home_team"].isin(dims.index)]
    x = hr["hc_x"] - 125.42
    y = 198.27 - hr["hc_y"]
    raw_dist = np.sqrt(x ** 2 + y ** 2)
    spray = np.degrees(np.arctan2(x, y))
    scale = REAL_AVG_HR_DISTANCE_FT / raw_dist.mean()
    est_dist = raw_dist * scale
    fence_dist = np.array([fence_distance_at_angle(dims, t, a) for t, a in zip(hr["home_team"], spray)])
    clear_rate = float((est_dist > fence_dist).mean())
    return {"scale": float(scale), "hr_clear_rate": clear_rate, "n_hr": len(hr)}


def would_clear_fence(hit_distance_ft: float, spray_angle: float, launch_angle: float | None,
                       dims: pd.DataFrame, team: str) -> bool | None:
    """The core Phase-1 classifier: would a batted ball with this real hit
    distance and spray angle have been a home run at `team`'s park? None if
    `team` isn't mapped. Applies the Green-Monster-style tall-wall, low-
    launch-angle suppression correction (see module docstring) when the
    fence at this angle is tall AND the ball's launch angle falls in the
    empirically-confirmed suppression band -- a coarse, launch-angle-proxy
    stand-in for real trajectory-vs-height physics, not a full integration."""
    fence_dist = fence_distance_at_angle(dims, team, spray_angle)
    if fence_dist is None:
        return None
    clears_distance = hit_distance_ft > fence_dist
    if not clears_distance:
        return False
    fence_height = fence_height_at_angle(dims, team, spray_angle)
    lo, hi = GREEN_MONSTER_LAUNCH_ANGLE_BAND
    if (fence_height is not None and fence_height >= TALL_WALL_HEIGHT_FT
            and launch_angle is not None and lo <= launch_angle < hi):
        # coarse suppression: treat as a probabilistic outcome rather than a
        # hard reclassification, since we have no real height-vs-trajectory
        # calculation to fall back on -- caller decides how to use this.
        return "suppressed"  # not True/False -- signals "clears distance, but
                              # tall-wall-at-low-launch-angle uncertainty applies"
    return True


# ---------------------------------------------------------------------------
# Batter-level, walk-forward-safe geometry HR factor (2026-08-04, EXPERIMENTAL
# /opt-in -- the wiring counterpart to this module's validated classifier).
#
# Unlike park_factors.py's every park-factor variant (all population-level:
# one number per park, or one per park+stand, blended into a batter's rate
# via a generic weight), this replays a SPECIFIC batter's OWN real prior-
# season batted-ball trajectories (spray angle + hit distance) against a
# target park's real fence vs. a league-average ("neutral") fence, producing
# a genuinely per-batter, per-park ratio -- directly answering "how much does
# THIS park help THIS batter's own established spray/power profile," not an
# average hitter's.
#
# Walk-forward scope, deliberately simplified relative to this codebase's
# usual in-season-blend convention (build_pregame_barrel_rate etc.): uses
# ONLY strictly-prior-SEASONS' batted balls, not a growing in-season replay
# sample -- replaying a batter's full raw trajectory list per-PA against an
# arbitrary target park is a heavier computation than any other walk-forward
# rate in this project, and a batter's spray/power profile is reasonably
# stable within a season (same "true talent, not day-to-day noise"
# assumption pull_tercile/barrel_rate already make) -- a real, flagged
# simplification, not an oversight. A true rookie (no prior-season batted
# balls) gets None -- callers already treat that as neutral (1.0), same
# convention as every other cold-start case in this codebase.
# ---------------------------------------------------------------------------

GEOMETRY_HR_PRIOR_BB = 200  # shrinkage prior, in pseudo-batted-balls -- scaled
# to the REAL observed distribution (2026-08-04 check): median 147, mean 179
# batted balls per batter-season in our own data. 200 sits just above a full
# season's typical sample, so a batter with a partial season's data is still
# meaningfully shrunk toward neutral, while a multi-season veteran's own
# signal dominates -- not independently tuned/validated beyond this
# real-distribution sanity check.


def build_batter_trajectory_history(pa: pd.DataFrame, scale_ft_per_unit: float) -> pd.DataFrame:
    """Every real batted ball's (batter, season, spray_angle, hit_distance_ft,
    launch_angle) -- the raw material batter_geometry_hr_factor replays
    against an arbitrary target park's real fence. One row per real batted
    ball (hc_x/hc_y present); NaN launch_angle rows are kept (the tall-wall
    suppression check in would_clear_fence already treats a missing launch
    angle as "no suppression info," not an error)."""
    bb = pa.dropna(subset=["hc_x", "hc_y"]).copy()
    x = bb["hc_x"] - 125.42
    y = 198.27 - bb["hc_y"]
    bb["spray_angle"] = np.degrees(np.arctan2(x, y))
    bb["hit_distance_ft"] = np.sqrt(x ** 2 + y ** 2) * scale_ft_per_unit
    return bb[["batter", "season", "spray_angle", "hit_distance_ft", "launch_angle"]]


def neutral_fence_distance_at_angle(dims: pd.DataFrame, spray_angle: float) -> float:
    """League-average fence distance (feet) at a given spray angle -- the
    across-all-30-parks mean at each of the 5 sampled directions, then
    interpolated the same way a single park's own fence is. This is the
    "neutral" baseline batter_geometry_hr_factor compares a target park
    against, so the resulting ratio reads as "how much better/worse than an
    average park is this, for this batter's own real spray/power profile,"
    matching every other park-factor's own mean-1.0 framing in this project."""
    avg_row = dims[[f"{k}_dist_ft" for k in ("lf_line", "lf_gap", "cf", "rf_gap", "rf_line")]].mean()
    return _interp_at_angle(avg_row, spray_angle, "dist_ft")


def batter_geometry_hr_factor(history: pd.DataFrame, batter_id, target_season: int,
                               target_team: str, dims: pd.DataFrame) -> float | None:
    """The core experimental signal: this batter's own prior-season batted
    balls replayed against target_team's real fence vs. a neutral fence,
    shrunk toward 1.0 by GEOMETRY_HR_PRIOR_BB pseudo-batted-balls (same
    Bayesian-shrinkage pattern as every other factor in this project).
    Returns None if this batter has no prior-season batted-ball history yet
    (true rookie) -- callers should treat that as neutral (1.0), not a guess."""
    prior = history[(history["batter"] == batter_id) & (history["season"] < target_season)]
    if prior.empty or target_team not in dims.index:
        return None
    target_fence = np.array([fence_distance_at_angle(dims, target_team, a) for a in prior["spray_angle"]])
    neutral_fence = np.array([neutral_fence_distance_at_angle(dims, a) for a in prior["spray_angle"]])
    target_clears = int((prior["hit_distance_ft"].to_numpy() > target_fence).sum())
    neutral_clears = int((prior["hit_distance_ft"].to_numpy() > neutral_fence).sum())
    n = len(prior)
    # Bayesian-shrunk RATE for each side (not the raw ratio directly) -- same
    # "shrink each rate toward a prior before dividing" pattern as every
    # other park-factor variant in this file's sibling module, so a thin
    # sample shrinks toward the neutral-fence RATE on both sides (giving a
    # factor of 1.0), not toward some other arbitrary anchor.
    prior_rate = neutral_clears / n
    reliability = n / (n + GEOMETRY_HR_PRIOR_BB)
    target_rate_shrunk = reliability * (target_clears / n) + (1 - reliability) * prior_rate
    neutral_rate_shrunk = reliability * (neutral_clears / n) + (1 - reliability) * prior_rate
    if neutral_rate_shrunk <= 0:
        return None
    return float(target_rate_shrunk / neutral_rate_shrunk)


# ---------------------------------------------------------------------------
# Doubles/triples geometry factor (2026-08-04, EXPERIMENTAL/opt-in) -- a real
# extension of the HR-clearance idea beyond the binary clear/no-clear case.
#
# Investigated first, on real pooled data (all parks, quality contact only,
# grouped by hit-distance-to-real-fence-distance ratio): a genuinely clean,
# monotonic relationship exists (short ratio -> single, near-1.0 ratio -> a
# real mix of double/triple/cheap-HR, well-past-1.0 -> almost pure HR).
# BUT -- unlike HR, where nothing can catch a ball that clears the fence --
# the 0.8-1.0 ratio range (where most real extra-base-hit action lives) is
# STILL MOSTLY OUTS (65-69% caught), not automatically a hit: distance alone
# can't distinguish a warning-track catch from a ball that drops for a
# double, since that also depends on hang time / outfielder positioning,
# which this model doesn't have. So this can't be a binary reclassifier the
# way HR is -- it has to compare a real, EMPIRICAL P(double)/P(triple) at a
# given ratio (which already averages over whatever real catch/no-catch mix
# exists at that ratio, pooled across the whole league) between a target
# park's fence and a neutral one -- the "component method" applied to a
# 5-category outcome instead of a binary one.
# ---------------------------------------------------------------------------

RATIO_BINS = (0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 5.0)  # validated
# informally on real pooled data (2026-08-04) -- a clean, sensible outcome
# gradient across these bins (mostly-single -> mostly-out/mixed-XBH ->
# mostly-HR); not independently tuned/optimized beyond that sanity check.
MIN_RATIO_BIN_N = 200  # a bin with fewer than this many real prior-season
# batted balls is treated as "no real info" (None), rather than trusting a
# noisy empirical rate -- conservative floor, not independently validated.


def build_ratio_outcome_table(pa: pd.DataFrame, dims: pd.DataFrame, scale_ft_per_unit: float) -> dict:
    """Walk-forward-safe (prior-seasons-only, with the same true-cold-start
    exception as every other rate anchor in this project -- see task #131's
    audit), per-season empirical P(double)/P(triple) as a function of
    hit-distance-to-fence ratio, pooled across ALL batters and parks
    (component-method style, not per-batter -- this IS the population-level
    model the per-batter factor below compares a specific batter's own
    trajectories against). Restricted to quality contact (Statcast
    launch_speed_angle 5 or 6) -- a weakly-hit ball is essentially never a
    double/triple regardless of distance, so including it would just dilute
    every bin's rate with irrelevant near-zero-probability rows."""
    bb = pa.dropna(subset=["hc_x", "hc_y"]).copy()
    bb = bb[bb["launch_speed_angle"].astype("float64").isin(PULLED_CONTACT_QUALITY_CODES)]
    bb = bb[bb["home_team"].isin(dims.index)].copy()
    x = bb["hc_x"] - 125.42
    y = 198.27 - bb["hc_y"]
    bb["spray_angle"] = np.degrees(np.arctan2(x, y))
    bb["hit_distance_ft"] = np.sqrt(x ** 2 + y ** 2) * scale_ft_per_unit

    out = {}
    for season in sorted(bb["season"].unique()):
        prior = bb[bb["season"] < season]
        if prior.empty:
            prior = bb[bb["season"] == season]  # true cold start only, see build_league_rates_by_season
        fence_d = np.array([fence_distance_at_angle(dims, t, a) for t, a in zip(prior["home_team"], prior["spray_angle"])])
        ratio = prior["hit_distance_ft"].to_numpy() / fence_d
        tmp = pd.DataFrame({
            "bin": pd.cut(ratio, RATIO_BINS),
            "is_double": (prior["outcome"] == "double").to_numpy(dtype=float),
            "is_triple": (prior["outcome"] == "triple").to_numpy(dtype=float),
        })
        rates = tmp.groupby("bin", observed=True).agg(
            p_double=("is_double", "mean"), p_triple=("is_triple", "mean"), n=("is_double", "size")
        )
        rates = rates.where(rates["n"] >= MIN_RATIO_BIN_N)  # thin bins -> NaN, treated as "no info"
        out[season] = rates
    return out


def batter_geometry_xbh_factor(history: pd.DataFrame, batter_id, target_season: int, target_team: str,
                                dims: pd.DataFrame, ratio_table: dict, outcome: str) -> float | None:
    """Per-batter, per-park geometry factor for 'double' or 'triple' --
    same overall design as batter_geometry_hr_factor (replay this batter's
    own real prior-season batted balls against target vs. neutral fence,
    shrink toward 1.0), but looks up the walk-forward ratio->outcome
    empirical table (build_ratio_outcome_table) instead of a binary
    clear/no-clear threshold -- see this section's own module-level note for
    why HR's binary check doesn't generalize directly. None if this batter
    has no prior-season history, target_team is unmapped, or target_season
    has no ratio table (cold-start season) -- callers treat as neutral 1.0."""
    prior = history[(history["batter"] == batter_id) & (history["season"] < target_season)]
    if prior.empty or target_team not in dims.index or target_season not in ratio_table:
        return None
    target_fence = np.array([fence_distance_at_angle(dims, target_team, a) for a in prior["spray_angle"]])
    neutral_fence = np.array([neutral_fence_distance_at_angle(dims, a) for a in prior["spray_angle"]])
    target_ratio = prior["hit_distance_ft"].to_numpy() / target_fence
    neutral_ratio = prior["hit_distance_ft"].to_numpy() / neutral_fence

    rates = ratio_table[target_season]
    col = f"p_{outcome}"
    target_p = rates[col].reindex(pd.cut(target_ratio, RATIO_BINS)).to_numpy()
    neutral_p = rates[col].reindex(pd.cut(neutral_ratio, RATIO_BINS)).to_numpy()
    valid = ~np.isnan(target_p) & ~np.isnan(neutral_p)
    n = int(valid.sum())
    if n == 0:
        return None
    target_exp = float(target_p[valid].mean())
    neutral_exp = float(neutral_p[valid].mean())
    if neutral_exp <= 0:
        return None
    # shrink the TARGET side toward the neutral anchor (matching batter_
    # geometry_hr_factor's "shrink each rate toward a prior before dividing"
    # pattern) -- a thin sample of valid bins gives a factor of 1.0.
    reliability = n / (n + GEOMETRY_HR_PRIOR_BB)
    target_shrunk = reliability * target_exp + (1 - reliability) * neutral_exp
    return float(target_shrunk / neutral_exp)
