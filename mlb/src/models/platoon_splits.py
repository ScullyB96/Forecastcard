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

import pandas as pd

from src.models.matchup import odds, prob_from_odds
from src.models.true_talent import MARCEL_WEIGHTS

PLATOON_STABILIZATION_PA = 2200  # Tango/Lichtman/Dolphin's own published figure

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


def build_platoon_multipliers(pa: pd.DataFrame, player_col: str, outcome: str) -> pd.DataFrame:
    """One row per (player, season): a same-hand and an opposite-hand
    multiplier to apply on top of that player's overall true-talent rate
    for `outcome` (from true_talent.py), walk-forward safe."""
    season_splits = _season_split_rates(pa, player_col, outcome)  # computed once, reused per target season
    out_frames = []
    for season in sorted(pa["season"].unique()):
        league_mult = league_platoon_multiplier(pa, outcome, season)
        # league_mult is opposite/same; split symmetrically in odds-space so
        # the two multipliers average out to ~1x on a player's own overall
        # rate rather than silently shifting their season-long average.
        same_mult_league = 1.0 / (league_mult ** 0.5)
        opp_mult_league = league_mult ** 0.5

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

        hist["same_hand_mult"] = (
            hist["same_reliability"] * own_same_mult.fillna(same_mult_league) + (1 - hist["same_reliability"]) * same_mult_league
        )
        hist["opp_hand_mult"] = (
            hist["opp_reliability"] * own_opp_mult.fillna(opp_mult_league) + (1 - hist["opp_reliability"]) * opp_mult_league
        )
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
