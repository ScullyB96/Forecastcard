"""Defense/fielding quality: a real skill dimension this project had zero
coverage of until now (2026-07-21) -- every other factor eventually built
this session (catcher framing, umpire tendency, park, weather) touches K/BB/
HR, but nothing adjusted for whether tonight's infield/outfield is actually
good at converting batted balls into outs.

Uses Baseball Savant's real Outs Above Average (OAA) leaderboard (same
CSV-export pattern already proven for sprint_speed -- one row per player per
season, tagged with their primary position), NOT a proxy derived from our own
outcome rates -- confirmed directly (2026-07-21) that a crude team-level
composite (all 7 non-battery positions pooled, checked against ALL balls in
play) shows ~zero signal (pooled corr 0.01, inconsistent sign across the two
season-pairs tested) -- the SAME "crude proxy finds nothing" pattern already
seen twice this session for catcher framing and umpire tendency. Splitting by
batted-ball type before checking again found real, correctly-signed,
consistent-sign signal in both cases:
  - INFIELD OAA (1B/2B/3B/SS) vs. groundball (launch_angle<10) BABIP-against:
    pooled corr -0.29 (higher OAA -> fewer groundball hits allowed), R^2=0.082
    on a walk-forward (prior-season OAA predicting this-season's real rate)
    team-season regression, n=60 (30 teams x 2 season-pairs).
  - OUTFIELD OAA (LF/CF/RF) vs. extra-base-hit rate on AIR balls
    (launch_angle>=10): pooled corr -0.107, R^2=0.012, same design.
Both are real but WEAK signals (R^2 an order of magnitude smaller than most
other validated factors this session) -- built and full-stack A/B tested
honestly rather than assumed to help; see validate_game_simulator.py's
isolated A/B run for the actual verdict.

No prediction/identification step is needed for who's playing which position
(unlike catcher's identify_starting_catcher or the closer/bullpen-role
guessing) -- lineups.parquet already carries the real (or, for a future game,
platoon-aware-projected) position_code per player per game, the same data
already used to build the batting lineup. Defense is just "the other 8 slots
of that same lineup," not a separate prediction problem.
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_RAW

IF_POSITIONS = {3, 4, 5, 6}  # 1B, 2B, 3B, SS -- MLB position_code, matches fielder_3..fielder_6
OF_POSITIONS = {7, 8, 9}  # LF, CF, RF -- matches fielder_7..fielder_9


def load_oaa_by_season() -> pd.DataFrame:
    """Baseball Savant's Outs Above Average leaderboard (season-level, one row
    per qualifying player-season) -- see fetch_oaa_seasons in fetch.py. Returns
    an empty frame if not yet fetched, rather than raising (callers already
    have a neutral-default fallback for missing defensive data)."""
    frames = []
    for path in sorted(DATA_RAW.glob("oaa_*.parquet")):
        season = int(path.stem.split("_")[-1])
        df = pd.read_parquet(path, columns=["player_id", "outs_above_average"])
        df["season"] = season
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["player_id", "season", "outs_above_average"])
    return pd.concat(frames, ignore_index=True)


def build_pregame_oaa(target_season: int) -> pd.Series:
    """Each fielder's walk-forward OAA for target_season: their most recently
    MEASURED prior season's value (a real season-aggregate metric, same
    "most recent prior measurement, no in-season leakage" convention as
    build_pregame_sprint_speed in expected_stats.py). Returns a Series indexed
    by player_id -- missing entries (no prior-season measurement, e.g. a
    rookie) should resolve to a neutral (no-adjustment) composite."""
    oaa = load_oaa_by_season()
    prior = oaa[oaa["season"] < target_season]
    if prior.empty:
        return pd.Series(dtype=float)
    latest = prior.sort_values("season").groupby("player_id").last()
    return latest["outs_above_average"]


def _position_composite(sub_lineup: pd.DataFrame, positions: set) -> float:
    vals = sub_lineup.loc[sub_lineup["position_code"].isin(positions), "oaa"]
    return float(vals.mean()) if len(vals) else float("nan")


def team_game_defense_snapshot(lineups: pd.DataFrame, schedule: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (team, game_pk, game_date): infield_oaa/outfield_oaa
    composite built from that SPECIFIC game's real lineup (lineups.parquet
    already has position_code per player, whether the lineup is real-posted
    or platoon-aware-projected -- no separate defensive-alignment guess
    needed), using target `season`'s walk-forward (prior-season-only) OAA."""
    sched = schedule[["game_pk", "home_team", "away_team"]].drop_duplicates()
    lu = lineups.merge(sched, on="game_pk", how="inner")
    lu["team"] = np.where(lu["team_side"] == "home", lu["home_team"], lu["away_team"])
    lu["position_code"] = lu["position_code"].astype(int)  # raw lineup data stores this as a string
    pregame_oaa = build_pregame_oaa(season)
    lu["oaa"] = lu["player_id"].map(pregame_oaa)

    rows = []
    for (game_pk, team), g in lu.groupby(["game_pk", "team"]):
        rows.append({
            "game_pk": game_pk, "team": team, "game_date": g["date"].iloc[0],
            "infield_oaa": _position_composite(g, IF_POSITIONS),
            "outfield_oaa": _position_composite(g, OF_POSITIONS),
        })
    return pd.DataFrame(rows)


# empirically-fit (2026-07-21, see module docstring): walk-forward team-season
# regressions, n=60 (30 teams x {2023->2024, 2024->2025}).
GB_BABIP_LEAGUE_AVG = 0.2734
GB_BABIP_INTERCEPT = 0.27496
GB_BABIP_COEF_INFIELD_OAA = -0.0008754
GB_BABIP_CLIP_MIN, GB_BABIP_CLIP_MAX = 0.20, 0.35

XBH_AIR_LEAGUE_AVG = 0.1129
XBH_AIR_INTERCEPT = 0.11098
XBH_AIR_COEF_OUTFIELD_OAA = -0.0003469
XBH_AIR_CLIP_MIN, XBH_AIR_CLIP_MAX = 0.06, 0.18


def _odds(p: float) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p / (1 - p)


def infield_defense_multiplier(infield_oaa: float | None) -> float:
    """Odds-ratio multiplier for the "single" outcome, from this game's real
    infield's own walk-forward OAA composite. Returns 1.0 (neutral) if no
    composite is available (e.g. a lineup with unrecognized position data)."""
    if infield_oaa is None or pd.isna(infield_oaa):
        return 1.0
    predicted = np.clip(
        GB_BABIP_INTERCEPT + GB_BABIP_COEF_INFIELD_OAA * infield_oaa, GB_BABIP_CLIP_MIN, GB_BABIP_CLIP_MAX
    )
    return _odds(predicted) / _odds(GB_BABIP_LEAGUE_AVG)


def outfield_defense_multiplier(outfield_oaa: float | None) -> float:
    """Same as infield_defense_multiplier, but for "double"/"triple" (extra-base
    hits on air balls), from the outfield's own composite."""
    if outfield_oaa is None or pd.isna(outfield_oaa):
        return 1.0
    predicted = np.clip(
        XBH_AIR_INTERCEPT + XBH_AIR_COEF_OUTFIELD_OAA * outfield_oaa, XBH_AIR_CLIP_MIN, XBH_AIR_CLIP_MAX
    )
    return _odds(predicted) / _odds(XBH_AIR_LEAGUE_AVG)


NEUTRAL_DEFENSE_FACTOR = {"single": 1.0, "double": 1.0, "triple": 1.0}


def resolve_defense_factor(infield_oaa: float | None, outfield_oaa: float | None) -> dict[str, float]:
    """{"single": mult, "double": mult, "triple": mult} for one team's real
    defensive alignment in one game -- neutral for any piece with no
    composite available. Combines into the SAME slot as catcher/umpire
    factors (disjoint outcome keys: catcher/umpire touch strikeout/walk,
    this touches single/double/triple) -- no new game_simulator.py plumbing
    needed, just a dict merge at the call site."""
    return {
        "single": infield_defense_multiplier(infield_oaa),
        "double": outfield_defense_multiplier(outfield_oaa),
        "triple": outfield_defense_multiplier(outfield_oaa),
    }


if __name__ == "__main__":
    import sys
    from src.utils.paths import DATA_PROCESSED

    pa_path = sorted(DATA_PROCESSED.glob("pa_table_*.parquet"))[-1]
    seasons = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [2023, 2024, 2025, 2026]
    latest_season = max(seasons)
    lineups = pd.concat(
        [pd.read_parquet(DATA_RAW / f"lineups_{s}.parquet") for s in seasons if (DATA_RAW / f"lineups_{s}.parquet").exists()],
        ignore_index=True,
    )
    schedule = pd.concat(
        [pd.read_parquet(DATA_RAW / f"schedule_{s}.parquet") for s in seasons if (DATA_RAW / f"schedule_{s}.parquet").exists()],
        ignore_index=True,
    )
    snap = team_game_defense_snapshot(lineups, schedule, latest_season)
    print(f"=== defense snapshot sanity check, season={latest_season} ===")
    print(f"n rows: {len(snap)}")
    print(snap[["infield_oaa", "outfield_oaa"]].describe())
    print("\nsample multipliers:")
    for io in [-6, -2, 0, 2, 6]:
        print(f"  infield_oaa={io:+.0f} -> single mult {infield_defense_multiplier(io):.4f}")
    for oo in [-4, 0, 4, 10]:
        print(f"  outfield_oaa={oo:+.0f} -> double/triple mult {outfield_defense_multiplier(oo):.4f}")
