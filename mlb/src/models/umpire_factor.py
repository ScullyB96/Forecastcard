"""Home-plate umpire tendency: a real, persistent skill/signal, confirmed on
this project's own data using the EXACT pitch-level precision method that
already worked for catcher framing (see catcher_framing.py) -- and the exact
lesson repeats: a crude proxy finds nothing, precise pitch-location data finds
a real, strong effect.

This project's first umpire check (2026-07-21), a 400-game sample using each
game's overall K%/BB% (a PA-level, not pitch-level, proxy), found essentially
no signal (split-half correlation 0.13, not distinguishable from noise).
Rather than conclude umpires don't matter, the data-source gap was closed
(MLB Stats API's live-feed `officials` field, backfilled for ALL 7,277 real
completed regular-season games 2023-2025 via `fetch_umpires_for_seasons` in
fetch.py) and the SAME precise method built for catcher framing was applied:
each umpire's called-strike rate on pitches within BORDERLINE_BAND of the true
rulebook zone edge. Result: real, STRONG year-over-year stability -- 0.44
(2023->2024), 0.52 (2024->2025), 0.31 (2023->2025) -- comparable to or
stronger than catcher framing's own validated numbers (0.33-0.48).

Unlike catcher framing, NO within-team-season control is needed here: a
catcher is tied to one team's pitching staff for a whole season (creating the
"personal-catcher-pairing" confound that swamped a coarser K-rate comparison
there), but an umpire's assigned games are a representative cross-section of
the WHOLE league across many different team/pitcher pairings, so a simple
"this umpire's own K/BB rate vs. league average, this season" comparison is
already clean. The empirically-fit K/BB translation slopes (below) have
CORRELATIONS EVEN STRONGER than catcher framing's own (K: 0.25 vs. 0.15; BB:
-0.37 vs. -0.22) -- umpire tendency translates unusually cleanly to K/BB
PA-outcome effects.

LIVE-PATH RESOLUTION (2026-07-21): unlike a future lineup or a future
starting pitcher, MLB does NOT require modeling a crew-rotation schedule to
get a real umpire assignment ahead of time -- confirmed directly against the
live-feed API that `officials` (including the home-plate umpire) is already
populated once a game's status reaches "Pre-Game", same-day, a few hours
before first pitch -- the SAME lead time real posted lineups get (checked:
querying 1-5 days ahead returns an empty officials list; only same-day,
post-"Pre-Game" games have it). So no crew-rotation model is needed:
`resolve_live_umpire_factor` just fetches the real assignment directly via
`fetch_home_plate_umpire` (fetch.py), falling back to neutral 1.0/1.0 only
when queried too far ahead (e.g. a night-before run, mirroring the
"PROJECTED lineup" fallback in lineup_projection.py) -- the same honest
"no signal available yet" convention used elsewhere in this project (e.g.
resolve_catcher_factor's neutral default for a debut player with zero
history).
"""

import numpy as np
import pandas as pd

from src.ingest.fetch import fetch_home_plate_umpire
from src.utils.paths import DATA_RAW

STRIKE_ZONE_HALF_WIDTH = 0.83  # ft, rulebook plate half-width (8.5in) + ball radius allowance
BORDERLINE_BAND = 0.17  # ft (~2in) either side of the true edge -- pitches genuinely up for grabs

UMPIRE_PRIOR_PITCHES = 300  # Bayesian-shrinkage prior, pseudo-pitches. A method-of-moments fit
                             # on real data gave ~116 pseudo-pitches; rounded up for a safety
                             # margin, same practice as CATCHER_FRAMING_PRIOR_PITCHES.

# empirically-fit (2026-07-21, see module docstring): each umpire-season's own
# K/BB log-odds effect (vs. league average that season) regressed against that
# same umpire-season's borderline-call-rate delta from league average, across
# 273 umpire-seasons with 200+ PA and 100+ borderline pitches.
K_EFFECT_SLOPE = 0.3996   # corr +0.25
BB_EFFECT_SLOPE = -0.8076  # corr -0.37


def load_umpire_game_log() -> pd.DataFrame:
    """One row per (game_pk, ump_id, season) -- the real home-plate umpire for
    every backfilled game, from fetch_umpires_for_seasons' cache. Call this
    ONCE and reuse the result for per-game lookups (e.g. via .set_index
    ("game_pk")) -- re-reading/re-merging from disk per game in a loop over
    hundreds of games is wasteful; see real_umpire_for_game for the
    convenience (but inefficient-in-bulk) single-game version."""
    path = DATA_RAW / "umpires.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["game_pk", "ump_id", "season"])
    umps = pd.read_parquet(path)
    seasons = sorted(int(s.stem.split("_")[-1]) for s in DATA_RAW.glob("schedule_*.parquet"))
    sched = pd.concat(
        [pd.read_parquet(DATA_RAW / f"schedule_{s}.parquet")[["game_pk"]].assign(season=s) for s in seasons],
        ignore_index=True,
    ).drop_duplicates("game_pk")
    return umps.merge(sched, on="game_pk", how="inner")


def compute_pitch_borderline_calls(seasons: list[int]) -> pd.DataFrame:
    """One row per TAKEN (not swung at) pitch within BORDERLINE_BAND of the
    real rulebook strike-zone edge, joined to that game's real home-plate
    umpire -- the same geometry as catcher_framing.py's identically-named
    function, just joined to umpire instead of catcher."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["game_pk", "description", "plate_x", "plate_z", "sz_top", "sz_bot"])
        df["season"] = season
        frames.append(df)
    pitches = pd.concat(frames, ignore_index=True)

    taken = pitches[pitches["description"].isin(["called_strike", "ball"])].dropna(
        subset=["plate_x", "plate_z", "sz_top", "sz_bot"]
    ).copy()
    dx_out = taken["plate_x"].abs() - STRIKE_ZONE_HALF_WIDTH
    dz_out = np.maximum(taken["sz_bot"] - taken["plate_z"], taken["plate_z"] - taken["sz_top"])
    taken["edge_dist"] = np.maximum(dx_out, dz_out)  # >0 outside the zone, <0 inside
    border = taken[taken["edge_dist"].abs() <= BORDERLINE_BAND].copy()
    border["is_strike"] = (border["description"] == "called_strike").astype(float)

    ump_log = load_umpire_game_log()
    border = border.merge(ump_log[["game_pk", "ump_id"]], on="game_pk", how="inner")
    return border[["season", "ump_id", "is_strike"]]


def build_umpire_factors_by_season(seasons: list[int]) -> dict[int, dict[int, dict[str, float]]]:
    """Walk-forward-safe (prior-seasons-only, cumulative -- same convention as
    catcher_framing.py): {season: {ump_id: {"strikeout": factor, "walk": factor}}}.
    Every other outcome category gets an implicit 1.0 in resolve_umpire_factor."""
    border = compute_pitch_borderline_calls(seasons)
    out = {}
    for season in sorted(border["season"].unique()):
        prior = border[border["season"] < season]
        ref = prior if len(prior) else border[border["season"] == season]
        league_rate = ref["is_strike"].mean()

        by_ump = ref.groupby("ump_id").agg(n=("is_strike", "size"), k=("is_strike", "sum"))
        shrunk_rate = (by_ump["k"] + UMPIRE_PRIOR_PITCHES * league_rate) / (by_ump["n"] + UMPIRE_PRIOR_PITCHES)
        delta = shrunk_rate - league_rate

        factors = {
            int(ump_id): {
                "strikeout": float(np.exp(K_EFFECT_SLOPE * d)),
                "walk": float(np.exp(BB_EFFECT_SLOPE * d)),
            }
            for ump_id, d in delta.items()
        }
        out[season] = factors
    return out


NEUTRAL_UMPIRE_FACTOR = {"strikeout": 1.0, "walk": 1.0}


def resolve_umpire_factor(factors_by_season: dict[int, dict[int, dict[str, float]]],
                           season: int, ump_id) -> dict[str, float]:
    """{"strikeout": mult, "walk": mult} for one umpire in one season --
    neutral (1.0/1.0) if this umpire has no factor data yet, OR (the live/
    predictive path today) if the umpire simply isn't known in advance -- see
    module docstring's KNOWN LIMITATION."""
    if ump_id is None or pd.isna(ump_id):
        return NEUTRAL_UMPIRE_FACTOR
    season_factors = factors_by_season.get(season, {})
    return season_factors.get(int(ump_id), NEUTRAL_UMPIRE_FACTOR)


def real_umpire_for_game(game_pk: int) -> int | None:
    """The REAL home-plate umpire for an already-played game -- only usable
    for backtesting (validate_game_simulator.py's oracle path), since this
    looks up the actual historical assignment, not a prediction."""
    ump_log = load_umpire_game_log()
    row = ump_log[ump_log["game_pk"] == game_pk]
    return int(row["ump_id"].iloc[0]) if len(row) else None


def resolve_live_umpire_factor(factors_by_season: dict[int, dict[int, dict[str, float]]],
                                season: int, game_pk: int) -> dict[str, float]:
    """The live/predictive-path (props.py) equivalent of real_umpire_for_game
    + resolve_umpire_factor combined: fetches this SPECIFIC game's real
    umpire crew directly from the MLB Stats API (see module docstring's
    LIVE-PATH RESOLUTION note -- no crew-rotation model needed, since MLB
    posts the real assignment same-day). Falls back to neutral 1.0/1.0 if
    queried before the crew is posted (e.g. a night-before run)."""
    row = fetch_home_plate_umpire(game_pk)
    if row is None:
        return NEUTRAL_UMPIRE_FACTOR
    return resolve_umpire_factor(factors_by_season, season, row["ump_id"])


if __name__ == "__main__":
    seasons = list(range(2023, 2027))
    factors = build_umpire_factors_by_season(seasons)
    latest_season = max(factors.keys())
    latest = factors[latest_season]
    print(f"=== umpire factors, {latest_season} (walk-forward, prior seasons only) ===")
    print(f"n umpires with a factor: {len(latest)}")
    k_vals = [v["strikeout"] for v in latest.values()]
    bb_vals = [v["walk"] for v in latest.values()]
    print(f"strikeout factor: min={min(k_vals):.4f} max={max(k_vals):.4f} mean={np.mean(k_vals):.4f}")
    print(f"walk factor:      min={min(bb_vals):.4f} max={max(bb_vals):.4f} mean={np.mean(bb_vals):.4f}")
