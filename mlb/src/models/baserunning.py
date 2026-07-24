"""Stolen bases: a real, individually-attributed skill this project had zero
coverage of until now (2026-07-21) -- the second "harder/deeper" frontier
tackled after defense/fielding quality, per explicit user request to invest
in the bigger rearchitecture rather than a team-aggregate simplification.

Real MLB stolen-base/caught-stealing counts aren't a clean Statcast `events`
value -- they show up only as incidental free text in a pitch's `des` field
(e.g. "Bryson Stott strikes out swinging. Max Kepler steals (1) 2nd base.")
on whatever OTHER pitch happened to be thrown around the same time, not tied
to a specific pitch or reliably to the PA structure this project's PA table
is built around. MLB's own live-feed boxscore (the SAME trusted endpoint
already used for umpires/officials) has real, clean, structured per-player
stolenBases/caughtStealing counts per game instead -- see
fetch_stolen_base_stats_for_seasons in fetch.py.

The OPPORTUNITY denominator (how many times a player was actually in a
position to attempt a steal) has to be derived from our own PA table's
on_1b/on_2b/on_3b columns -- these are real player IDs (not just occupancy
booleans), confirmed still present in the final PA table (build_pa_table.py
never drops them, only additionally derives the pre_bases bitmask from
them) -- scanned across EVERY PA in the game (not just this player's own
batting PAs), since being a baserunner during someone ELSE's plate
appearance is exactly what creates a steal opportunity. A player on 1st
with 2nd open, or on 2nd with 3rd open, counts as one opportunity; 3rd-base
steals are rare enough to fold into the same combined rate rather than
model separately (real per-game data doesn't distinguish which base a given
game's stolen bases came from anyway).

Walk-forward design deliberately mirrors sprint_speed.py/defense_factor.py
(a SEASON-LEVEL snapshot using the most recent measured PRIOR season, no
in-season-to-date blending within the target season) rather than
true_talent.py's full PA-by-PA in-season blend -- consistent with the two
other real, season-aggregate signals already in this project, and avoids
needing PA-level opportunity tracking (a bigger, separate undertaking) just
to blend in a signal that changes slowly within a single season anyway.
"""

import numpy as np
import pandas as pd

from src.utils.paths import DATA_RAW

MARCEL_WEIGHTS = [5, 4, 3]  # most recent season first, same recency weights as true_talent.py

# empirically-fit (2026-07-21) via a leakage-free K-sweep (preseason-only
# prediction vs. real full-season rate, target seasons 2024-2025): attempt_rate
# is a remarkably PERSISTENT, decision-driven trait -- correlation peaks at a
# tiny K=15 (0.725) and is already declining by K=30, an order of magnitude
# less shrinkage than nearly every other rate in this project (matches the
# real sabermetric finding that WHETHER a player runs is a much more stable
# signal than most outcome rates). success_rate, by contrast, showed almost
# no year-over-year predictability regardless of K (correlation flat at
# 0.07-0.08 across the whole 5-150 range tested) -- a real, weak signal, so
# leaned toward heavier shrinkage (bigger K) rather than trusting a player's
# own history much, consistent with this project's "weak signal -> shrink
# harder" practice elsewhere.
STABILIZATION_OPPORTUNITIES = 15  # for attempt_rate
STABILIZATION_ATTEMPTS = 80  # for success_rate


def build_sb_opportunities(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season): how many times this player was a real
    baserunner in a stolen-base-eligible situation (on 1st with 2nd open, OR
    on 2nd with 3rd open)."""
    opp1 = pa[pa["on_1b"].notna() & pa["on_2b"].isna()].groupby(["on_1b", "season"]).size()
    opp2 = pa[pa["on_2b"].notna() & pa["on_3b"].isna()].groupby(["on_2b", "season"]).size()
    total = opp1.rename_axis(["player", "season"]).add(opp2.rename_axis(["player", "season"]), fill_value=0)
    return total.rename("opportunities").reset_index().astype({"player": int, "opportunities": int})


def load_stolen_base_stats() -> pd.DataFrame:
    """Real per-player-per-game SB/CS counts from MLB's own boxscore -- see
    fetch_stolen_base_stats_for_seasons in fetch.py. Returns an empty frame
    if not yet fetched, rather than raising."""
    path = DATA_RAW / "stolen_bases.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["game_pk", "player_id", "stolen_bases", "caught_stealing"])
    return pd.read_parquet(path)


def build_season_sb_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season): opportunities, attempts (SB+CS),
    successes (stolen_bases) -- merges the opportunity denominator (derived
    from our own PA table's runner-identity columns) with the real per-game
    SB/CS counts."""
    opp = build_sb_opportunities(pa)
    sb_stats = load_stolen_base_stats()
    game_season = pa[["game_pk", "season"]].drop_duplicates()
    sb_stats = sb_stats.merge(game_season, on="game_pk", how="inner")
    season_totals = sb_stats.groupby(["player_id", "season"]).agg(
        stolen_bases=("stolen_bases", "sum"), caught_stealing=("caught_stealing", "sum")
    ).reset_index().rename(columns={"player_id": "player"})
    merged = opp.merge(season_totals, on=["player", "season"], how="left")
    merged[["stolen_bases", "caught_stealing"]] = merged[["stolen_bases", "caught_stealing"]].fillna(0)
    merged["attempts"] = merged["stolen_bases"] + merged["caught_stealing"]
    return merged


def build_pregame_sb_rates(season_stats: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Walk-forward (prior-seasons-only) attempt_rate/success_rate per
    player for target_season: Marcel-weighted (5/4/3) recency-weighted rate
    across the last 3 prior seasons, regressed toward league average via a
    stabilization constant -- same shrinkage principle as true_talent.py,
    applied to this bespoke "opportunities" denominator instead of PA."""
    prior = season_stats[season_stats["season"] < target_season]
    if prior.empty:
        return pd.DataFrame(columns=["player", "attempt_rate", "success_rate"])

    league_attempt_rate = prior["attempts"].sum() / prior["opportunities"].sum()
    league_success_rate = prior["stolen_bases"].sum() / prior["attempts"].sum()

    rows = []
    for player, g in prior.groupby("player"):
        g = g.set_index("season")
        opp_num, opp_den = 0.0, 0.0
        att_num, att_den = 0.0, 0.0
        for i, w in enumerate(MARCEL_WEIGHTS):
            s = target_season - 1 - i
            if s in g.index:
                opp_num += w * g.loc[s, "attempts"]
                opp_den += w * g.loc[s, "opportunities"]
                att_num += w * g.loc[s, "stolen_bases"]
                att_den += w * g.loc[s, "attempts"]
        if opp_den == 0:
            continue
        raw_opportunities = g["opportunities"].sum()
        raw_attempts = g["attempts"].sum()
        rows.append({
            "player": player, "raw_opportunities": raw_opportunities, "raw_attempts": raw_attempts,
            "weighted_attempt_rate": opp_num / opp_den,
            "weighted_success_rate": (att_num / att_den) if att_den > 0 else league_success_rate,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["player", "attempt_rate", "success_rate"])

    reliability_attempt = out["raw_opportunities"] / (out["raw_opportunities"] + STABILIZATION_OPPORTUNITIES)
    out["attempt_rate"] = reliability_attempt * out["weighted_attempt_rate"] + (1 - reliability_attempt) * league_attempt_rate
    reliability_success = out["raw_attempts"] / (out["raw_attempts"] + STABILIZATION_ATTEMPTS)
    out["success_rate"] = reliability_success * out["weighted_success_rate"] + (1 - reliability_success) * league_success_rate
    return out[["player", "attempt_rate", "success_rate"]]


def resolve_sb_rates(pregame_rates: pd.DataFrame, lineup_ids: list[int]) -> dict[int, dict[str, float]]:
    """{lineup_idx: {"attempt_rate":, "success_rate":}} for a real 9-batter
    lineup, keyed by LINEUP INDEX (0-8, matching thruorder_counts/sb_rates'
    convention in game_simulator.py) rather than player_id, since that's what
    GameSimulator.simulate_half_inning actually tracks runners by. A player
    with no usable prior-season history (e.g. an MLB debut) is simply
    omitted -- game_simulator.py's `if rates is None: continue` already
    treats a missing lineup_idx as "no attempt," not a guess."""
    rates_by_player = pregame_rates.set_index("player")[["attempt_rate", "success_rate"]]
    out = {}
    for idx, pid in enumerate(lineup_ids):
        if pid in rates_by_player.index:
            row = rates_by_player.loc[pid]
            out[idx] = {"attempt_rate": float(row["attempt_rate"]), "success_rate": float(row["success_rate"])}
    return out


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    season_stats = build_season_sb_stats(pa)
    latest_season = int(season_stats["season"].max())
    rates = build_pregame_sb_rates(season_stats, latest_season)
    print(f"=== stolen-base rate sanity check, season={latest_season} ===")
    print(f"n players with a walk-forward rate: {len(rates)}")
    print(rates[["attempt_rate", "success_rate"]].describe())
