"""Catcher framing: a validated, real, moderately-persistent skill (this
project's own year-over-year check on 2023-2025 data: log-odds correlation
0.33-0.48 for a catcher's own called-strike rate on BORDERLINE pitches,
n=56-66 catchers with 150+ such pitches/season) -- computed from PITCH-LEVEL
Statcast data (plate_x/plate_z vs. the real rulebook zone edges), not the PA
table.

Pitch granularity is required, not optional: a coarser measure (this
project's first attempt used Statcast's own `zone` column, buckets 11-14 --
gave weak 0.05/0.20 YoY stability, inconclusive) or a PA-level measure gets
swamped by which specific pitching staff a catcher happens to catch for.
Confirmed directly: a within-team-season "this catcher's own K/BB rate vs.
his team-season's OTHER catchers" comparison (the same home/road-style device
park_factors.py uses to isolate park effects from team quality) shows ~ZERO
year-over-year stability for strikeout rate specifically (0.02-0.10) even
though the walk-rate version of the same comparison (0.17-0.31) and the
pitch-level borderline metric (0.33-0.48) both show real signal --
personal-catcher pairings (a team's strikeout-heavy ace often throwing to one
specific catcher) correlate with K rate far more than with BB rate, and
swamp the framing signal in any measure coarser than "same exact pitch
location, different catcher."

Translating the validated borderline-call-rate skill into a PA-outcome
multiplier (combine_matchup_distribution in game_simulator.py only operates
on PA-level rates, not pitch-by-pitch count state -- building a full
pitch-by-pitch simulator to use this natively is out of scope) uses an
empirically-fit slope, not a guess: regressing each catcher-season's
within-team-season K/BB log-odds effect (the noisy comparison above -- noisy,
but still an unbiased DEPENDENT variable; classical measurement error in Y
does not bias an OLS slope estimate, only adds noise to it) against that same
catcher-season's borderline-call-rate delta, across catcher-team-seasons with
200+ team PA and 100+ borderline pitches.

CONDITIONED on pitch-type x count (2026-07-21 refinement): the borderline
rate is computed relative to each (pitch-group x count-group) cell's own
mean, not a single pooled average across all borderline pitches -- catchers
don't face a random mix of pitch types/counts (a staff's specific pitch mix,
and how often that staff pitches from ahead/behind, both vary by team), and
this coarser pooling was a real, confirmed confound. Conditioning improved
YoY stability in 2 of 3 year-pairs tested (2023->2024: 0.33->0.36,
2024->2025: 0.38->0.42, 2023->2025: 0.48->0.48 flat) AND improved the
downstream K/BB slope-fit correlation too (K: 0.15->0.23, BB: -0.17->-0.22) --
confirming it's a genuinely cleaner signal, not just noise reshuffling.
Pitch types are grouped into 3 buckets (fastball/breaking/offspeed) and
counts into 3 leverage buckets (neutral/2-strike/hitter-ahead) rather than
the full type x count cross (12 raw pitch_type values x 12 count states would
leave most cells too thin to estimate a stable mean from).
  log(K effect)  =  0.7865 * excess_rate   (corr +0.23)
  log(BB effect) = -0.9857 * excess_rate   (corr -0.22)
Both directionally exactly as expected (more borderline calls go a catcher's
way -> more strikeouts, fewer walks) and modest in magnitude, consistent with
published catcher-framing research finding real but small effects (on the
order of a handful of runs per season even for the best/worst full-time
framers). intent_walk is deliberately excluded from the walk category here --
ball/strike calls are mechanically irrelevant to a non-competitive
intentional walk (see build_pa_table.py's OUTCOME_CATEGORIES split).
"""

import numpy as np
import pandas as pd

from src.models.pitch_groups import count_group as _count_group
from src.models.pitch_groups import pitch_group as _pitch_group
from src.utils.paths import DATA_RAW

STRIKE_ZONE_HALF_WIDTH = 0.83  # ft, rulebook plate half-width (8.5in) + ball radius allowance
BORDERLINE_BAND = 0.17  # ft (~2in) either side of the true edge -- pitches genuinely up for grabs

CATCHER_FRAMING_PRIOR_PITCHES = 300  # Bayesian-shrinkage prior, pseudo-pitches, toward a
                                       # mean-zero excess rate (see build_catcher_framing_factors_by_season)
                                       # -- kept at the same size as the original pooled-rate version;
                                       # the method-of-moments fit was already unstable there (see git
                                       # history), and conditioning changes the metric's SCALE, not the
                                       # amount of real per-catcher sample needed to trust it.
K_EFFECT_SLOPE = 0.7865   # empirically-fit, see module docstring
BB_EFFECT_SLOPE = -0.9857  # empirically-fit, see module docstring


def compute_pitch_borderline_calls(seasons: list[int]) -> pd.DataFrame:
    """One row per TAKEN (not swung at) pitch within BORDERLINE_BAND of the
    real rulebook strike-zone edge, across `seasons` -- the raw material for
    per-catcher framing rate. Needs full pitch-level Statcast columns
    (plate_x/plate_z/sz_top/sz_bot/pitch_type/balls/strikes), not the PA
    table (one row per PA)."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["description", "plate_x", "plate_z", "sz_top", "sz_bot",
                                              "fielder_2", "pitch_type", "balls", "strikes"])
        df["season"] = season
        frames.append(df)
    pitches = pd.concat(frames, ignore_index=True)

    taken = pitches[pitches["description"].isin(["called_strike", "ball"])].dropna(
        subset=["plate_x", "plate_z", "sz_top", "sz_bot", "fielder_2"]
    ).copy()
    dx_out = taken["plate_x"].abs() - STRIKE_ZONE_HALF_WIDTH
    dz_out = np.maximum(taken["sz_bot"] - taken["plate_z"], taken["plate_z"] - taken["sz_top"])
    taken["edge_dist"] = np.maximum(dx_out, dz_out)  # >0 outside the zone, <0 inside
    border = taken[taken["edge_dist"].abs() <= BORDERLINE_BAND].copy()
    border["catcher"] = border["fielder_2"].astype(int)
    border["is_strike"] = (border["description"] == "called_strike").astype(float)
    border["cell"] = border["pitch_type"].apply(_pitch_group) + "_" + border.apply(
        lambda r: _count_group(r["balls"], r["strikes"]), axis=1
    )
    return border[["season", "catcher", "is_strike", "cell"]]


def build_catcher_framing_factors_by_season(seasons: list[int]) -> dict[int, dict[int, dict[str, float]]]:
    """Walk-forward-safe (prior-seasons-only, cumulative -- same "prior if
    available else current season" convention as matchup.py's league_rate and
    weather.py): {season: {catcher_id: {"strikeout": factor, "walk": factor}}}.
    Every other outcome category gets an implicit 1.0 in
    resolve_catcher_factor -- framing has no plausible mechanism to affect them.

    Each pitch's "excess" is its called-strike indicator MINUS that (cell,
    season) group's own mean -- computed within `ref` (prior seasons only,
    walk-forward-safe) so a cell's mean never leaks the target season's own
    data. A catcher's raw mean excess is, by construction, centered on 0
    league-wide; shrunk toward 0 (not toward a rate) via the same Bayesian
    pseudo-pitch-count mechanism used everywhere else in this project."""
    border = compute_pitch_borderline_calls(seasons)
    out = {}
    for season in sorted(border["season"].unique()):
        prior = border[border["season"] < season]
        ref = prior if len(prior) else border[border["season"] == season]

        cell_means = ref.groupby("cell")["is_strike"].transform("mean")
        excess = ref["is_strike"] - cell_means

        by_catcher = pd.DataFrame({"catcher": ref["catcher"], "excess": excess}).groupby("catcher").agg(
            n=("excess", "size"), excess_sum=("excess", "sum")
        )
        shrunk_excess = by_catcher["excess_sum"] / (by_catcher["n"] + CATCHER_FRAMING_PRIOR_PITCHES)

        factors = {
            int(catcher_id): {
                "strikeout": float(np.exp(K_EFFECT_SLOPE * d)),
                "walk": float(np.exp(BB_EFFECT_SLOPE * d)),
            }
            for catcher_id, d in shrunk_excess.items()
        }
        out[season] = factors
    return out


CATCHER_WINDOW_DAYS = 30  # a starting catcher plays ~4-5x/week (rested ~1x/week), so 30
                           # trailing days gives ~15-20 games -- plenty to separate a
                           # primary catcher from a backup, same reasoning as
                           # bullpen.py's ROSTER_WINDOW_DAYS/CLOSER_WINDOW_DAYS.


def build_catcher_appearance_log(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, catcher, game_pk, game_date): whichever catcher
    caught the MOST PAs for `team` in that specific game (handles the rare
    in-game catcher substitution the same way team_home_road_runs handles
    primary_venue -- take the mode, not every row) -- the raw material for
    identify_starting_catcher, the catcher-specific analogue of
    bullpen.py's identify_closer/build_relief_appearance_log."""
    tmp = pa.dropna(subset=["catcher"]).copy()
    tmp["catcher"] = tmp["catcher"].astype(int)
    tmp["team"] = np.where(tmp["inning_topbot"] == "Top", tmp["home_team"], tmp["away_team"])
    counts = tmp.groupby(["team", "game_pk", "game_date", "catcher"]).size().rename("n_pa").reset_index()
    return counts.sort_values("n_pa").drop_duplicates(["team", "game_pk"], keep="last")


def identify_starting_catcher(catcher_log: pd.DataFrame, team: str, target_date, window_days: int = CATCHER_WINDOW_DAYS):
    """The catcher who caught the most games for `team` in the trailing
    `window_days` strictly before target_date -- walk-forward-safe. Returns
    None if there's no usable history yet (e.g. very early in a season)."""
    target_ts = pd.Timestamp(target_date)
    window_start = target_ts - pd.Timedelta(days=window_days)
    dates = pd.to_datetime(catcher_log["game_date"])
    sub = catcher_log[(catcher_log["team"] == team) & (dates < target_ts) & (dates >= window_start)]
    if sub.empty:
        return None
    return sub["catcher"].value_counts().idxmax()


NEUTRAL_CATCHER_FACTOR = {"strikeout": 1.0, "walk": 1.0}


def resolve_catcher_factor(factors_by_season: dict[int, dict[int, dict[str, float]]],
                            season: int, catcher_id) -> dict[str, float]:
    """{"strikeout": mult, "walk": mult} for one catcher in one season --
    neutral (1.0/1.0) if this catcher has no framing data yet (e.g. an MLB
    debut with zero prior-season borderline-pitch history)."""
    if catcher_id is None or pd.isna(catcher_id):
        return NEUTRAL_CATCHER_FACTOR
    season_factors = factors_by_season.get(season, {})
    return season_factors.get(int(catcher_id), NEUTRAL_CATCHER_FACTOR)


if __name__ == "__main__":
    seasons = list(range(2023, 2027))
    factors = build_catcher_framing_factors_by_season(seasons)
    latest_season = max(factors.keys())
    latest = factors[latest_season]
    print(f"=== catcher framing factors, {latest_season} (walk-forward, prior seasons only) ===")
    print(f"n catchers with a factor: {len(latest)}")
    k_vals = [v["strikeout"] for v in latest.values()]
    bb_vals = [v["walk"] for v in latest.values()]
    print(f"strikeout factor: min={min(k_vals):.4f} max={max(k_vals):.4f} mean={np.mean(k_vals):.4f}")
    print(f"walk factor:      min={min(bb_vals):.4f} max={max(bb_vals):.4f} mean={np.mean(bb_vals):.4f}")
