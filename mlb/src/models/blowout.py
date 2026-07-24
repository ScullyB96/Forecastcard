"""Position-player-pitching (mop-up) detection and modeling. MLB's real rule
allows a position player to pitch when trailing (or, rarely, leading) by 8+
runs -- confirmed directly in our own data: pitchers with a tiny total career
sample (<=15 PAs across all of 2023-2026) show up ENTIRELY (100%) in games
where their team trails by 8+ runs, overwhelmingly in innings 8-9 (87% of
these appearances) -- exactly matching the real rule and how teams actually
use it (wait until the game is clearly decided, not the instant the
threshold is hit). Their outcome rates are dramatically different from a
real pitcher, even in the SAME blowout situations: strikeout rate 3.4% vs
19.7% for real pitchers in the same >=6-run-deficit situations (22.4% league
average overall), home_run rate allowed 5.4% vs 3.4%, single rate 19.3% vs
14.2% -- position players simply can't get outs or miss bats the way even a
gassed legitimate reliever can.

Currently unmodeled entirely: GameSimulator has zero awareness of the
in-game score state when picking a pitcher -- a bullpen plan (oracle or
predictive) is built once pregame and never adapts to a blowout actually
happening mid-simulation. This module adds that: a fixed, pooled (not
walk-forward-by-season -- see below) outcome-rate profile, dynamically
substituted in-simulation once a team's pitching side is down 8+ runs in the
8th inning or later.

Deliberately NOT walk-forward-by-season like every other rate in this
project: the effect here is a categorical talent gap (a position player just
isn't a pitcher), not a skill/luck signal that could leak forward-looking
information the way a real player's own rate could -- splitting ~225
PAs/season four ways would only add noise without a corresponding benefit.

Known simplification: the separate MLB rule allowing a position player to
pitch in ANY extra inning regardless of score margin is not modeled here --
only the validated 8-run/8th-inning-or-later blowout case is.
"""

import pandas as pd

from src.models.game_simulator import OUTCOMES

BLOWOUT_MARGIN = 8
BLOWOUT_MIN_INNING = 8
MIN_PA_FOR_SUSPECT = 15
MIN_EXTREME_SHARE = 0.5


def _score_margin_for_pitcher(pa: pd.DataFrame) -> pd.Series:
    """The PITCHER's own team's score minus the batting team's score at the
    start of each PA -- negative means the pitcher's team is losing. Rebuilt
    from bat_score by forward-filling each side's own running score within
    each game (the PA table only keeps the BATTING team's score per row)."""
    pa = pa.sort_values(["game_pk", "inning", "at_bat_number"])
    away_running = pa["bat_score"].where(pa["inning_topbot"] == "Top")
    home_running = pa["bat_score"].where(pa["inning_topbot"] == "Bot")
    running = pd.DataFrame({"away": away_running.values, "home": home_running.values}, index=pa.index)
    running[["away", "home"]] = running.groupby(pa["game_pk"])[["away", "home"]].ffill().fillna(0)
    fld_score = running["home"].where(pa["inning_topbot"] == "Top", running["away"])
    return fld_score - pa["bat_score"]


def identify_position_player_appearances(pa: pd.DataFrame) -> pd.Index:
    """Pitcher IDs whose entire career sample in our data is tiny AND
    concentrated in extreme-deficit situations -- see module docstring for
    the validated 906-PA signature this heuristic recovers."""
    tmp = pa.copy()
    tmp["margin"] = _score_margin_for_pitcher(tmp)
    stats = tmp.groupby("pitcher").agg(
        total_pa=("outcome", "size"),
        pct_extreme=("margin", lambda s: (s <= -6).mean()),
    )
    return stats[(stats["total_pa"] <= MIN_PA_FOR_SUSPECT) & (stats["pct_extreme"] > MIN_EXTREME_SHARE)].index


def build_position_player_profile(pa: pd.DataFrame) -> dict:
    """A single pooled (not walk-forward) profile dict -- see module
    docstring for why pooling across all seasons is the right call here."""
    suspect_ids = identify_position_player_appearances(pa)
    suspect_pa = pa[pa["pitcher"].isin(suspect_ids)]
    rates = suspect_pa["outcome"].value_counts(normalize=True)
    neutral = {o: 1.0 for o in OUTCOMES}
    return {
        "rates": {o: rates.get(o, 0.0) for o in OUTCOMES},
        "hand": "R", "same_mult": neutral, "opp_mult": neutral,
    }
