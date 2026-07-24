"""Reduce pitch-level Statcast to one row per plate appearance (PA) with the
pre-PA base-out state, the outcome, runs scored, and the post-PA state.

Why build this ourselves rather than use a pre-built PA-level dataset: no
such free dataset exists granular enough (batter/pitcher IDs, exact base-out
state, handedness) to build a matchup-aware Markov simulation from -- same
"derive it ourselves from trusted raw data" principle used for the weekly
player stats in the sibling NFL project.

Key mechanics, confirmed directly against real data:
  - on_1b/on_2b/on_3b and outs_when_up on a PA's FIRST pitch give the PRE-PA
    base-out state (these columns reflect the state as of that pitch, not
    the state resulting from it).
  - `events` is non-null only on the LAST pitch of a PA -- that's the outcome.
  - bat_score (first pitch) vs. post_bat_score (last pitch) gives runs scored
    on that specific PA.
  - The POST-PA state is just the PRE-PA state of the *next* PA in the same
    half-inning (game_pk, inning, inning_topbot) -- if there is no next PA in
    that group, the half-inning ended right there (whether from a 3rd out or
    a walk-off; we don't need to know which, we only need "terminal", plus
    the runs already captured above).
"""

import pandas as pd

from src.utils.paths import DATA_RAW, DATA_PROCESSED

# Every real `events` value observed in 2023-2026 data, partitioned into
# outcome categories. `truncated_pa` is dropped -- not a real outcome (an
# incomplete PA record, e.g. game stoppage mid-at-bat).
OUTCOME_CATEGORIES = {
    "strikeout": {"strikeout", "strikeout_double_play"},
    "walk": {"walk"},
    "intent_walk": {"intent_walk"},
    "hit_by_pitch": {"hit_by_pitch"},
    "single": {"single"},
    "double": {"double"},
    "triple": {"triple"},
    "home_run": {"home_run"},
    "sac_fly": {"sac_fly", "sac_fly_double_play"},
    "sac_bunt": {"sac_bunt"},
    "field_error": {"field_error"},
    "fielders_choice": {"fielders_choice", "fielders_choice_out"},
    "double_play": {"grounded_into_double_play", "double_play"},
    "triple_play": {"triple_play"},
    "field_out": {"field_out", "force_out"},
    "catcher_interf": {"catcher_interf"},
}
EVENT_TO_CATEGORY = {ev: cat for cat, evs in OUTCOME_CATEGORIES.items() for ev in evs}
DROPPED_EVENTS = {"truncated_pa"}

PA_COLUMNS = [
    "game_pk", "game_year", "game_date", "at_bat_number", "pitch_number",
    "inning", "inning_topbot", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "bat_score", "post_bat_score", "events", "batter", "pitcher", "stand", "p_throws",
    "home_team", "away_team", "n_thruorder_pitcher", "hc_x", "hc_y", "fielder_2",
    "estimated_ba_using_speedangle", "launch_speed_angle", "age_bat", "age_pit", "launch_angle",
    "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
]


def build_pa_table(statcast: pd.DataFrame) -> pd.DataFrame:
    df = statcast[statcast["game_type"] == "R"][PA_COLUMNS].copy()

    # NOTE: groupby().first() is NOT "the literal first row" -- it independently
    # takes the first non-null value PER COLUMN, which silently Frankensteins
    # together base-state values from different pitches whenever a stolen
    # base/caught-stealing/pickoff changes on_1b/on_2b/on_3b mid-at-bat (a real,
    # common occurrence: confirmed on 3.13% of 2024 PAs). .nth(0) takes the
    # actual first row and must be used here instead.
    first = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).groupby(
        ["game_pk", "at_bat_number"], as_index=False
    ).nth(0)
    last_outcome = (
        df[df["events"].notna()]
        .sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .last()[["game_pk", "at_bat_number", "events", "post_bat_score", "hc_x", "hc_y",
                  "estimated_ba_using_speedangle", "launch_speed_angle", "launch_angle"]]
    )

    # hc_x/hc_y/estimated_ba_using_speedangle/launch_speed_angle/launch_angle (all
    # only populated on a ball in play) come from the LAST pitch of the PA, same
    # as events -- NOT from `first` (the first pitch, which is what everything
    # else in this table uses), since a batted ball is definitionally the
    # final pitch of its PA.
    pa = first.drop(columns=["events", "post_bat_score", "hc_x", "hc_y",
                              "estimated_ba_using_speedangle", "launch_speed_angle", "launch_angle"]).merge(
        last_outcome, on=["game_pk", "at_bat_number"], how="inner"
    )
    pa = pa[~pa["events"].isin(DROPPED_EVENTS)].copy()
    pa["outcome"] = pa["events"].map(EVENT_TO_CATEGORY)
    unmapped = pa["outcome"].isna().sum()
    if unmapped:
        print(f"WARNING: {unmapped} PAs have an unrecognized events value not in OUTCOME_CATEGORIES: "
              f"{pa.loc[pa['outcome'].isna(), 'events'].unique().tolist()}", flush=True)
        pa = pa[pa["outcome"].notna()]

    pa["runs_scored"] = pa["post_bat_score"] - pa["bat_score"]
    pa["pre_bases"] = (
        pa["on_1b"].notna().astype(int) * 1
        + pa["on_2b"].notna().astype(int) * 2
        + pa["on_3b"].notna().astype(int) * 4
    )  # 0-7, a bitmask: 1=runner on 1st, 2=on 2nd, 4=on 3rd
    pa["pre_state"] = pa["pre_bases"] * 10 + pa["outs_when_up"]  # e.g. 5 outs_when_up=1, runners on 1st+3rd -> 51
    pa = pa.rename(columns={"fielder_2": "catcher"})  # Statcast's raw name for the catcher's player id

    pa = pa.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number"]).reset_index(drop=True)
    half_inning = ["game_pk", "inning", "inning_topbot"]
    pa["next_pre_bases"] = pa.groupby(half_inning)["pre_bases"].shift(-1)
    pa["next_outs_when_up"] = pa.groupby(half_inning)["outs_when_up"].shift(-1)
    pa["terminal"] = pa["next_pre_bases"].isna()  # no next PA in this half-inning -> it ended here
    pa["post_state"] = pa["next_pre_bases"] * 10 + pa["next_outs_when_up"]

    return pa.drop(columns=["next_pre_bases", "next_outs_when_up"])


def build_and_save_pa_table(seasons: list[int]) -> "pd.DataFrame | None":
    """Rebuild the derived PA table from whatever's currently cached in
    statcast_{season}.parquet for each season, and save it to
    pa_table_{min}_{max}.parquet. Callable directly (not just via
    __main__) so the live daily pipeline can rebuild this AFTER refreshing
    raw Statcast data -- refreshing the raw cache alone does nothing for
    predictions until this derived table is rebuilt from it too."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            print(f"skipping {season}: no cached statcast data", flush=True)
            continue
        statcast = pd.read_parquet(path)
        pa = build_pa_table(statcast)
        pa["season"] = season
        print(f"{season}: {len(pa)} PAs built", flush=True)
        frames.append(pa)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_PROCESSED / f"pa_table_{min(seasons)}_{max(seasons)}.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"saved {out_path.name}: {len(combined)} PAs total", flush=True)
    return combined


if __name__ == "__main__":
    import sys

    seasons = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [2023, 2024, 2025, 2026]
    combined = build_and_save_pa_table(seasons)
    if combined is not None:
        print("\n=== league-wide outcome rate sanity check ===", flush=True)
        print((combined["outcome"].value_counts(normalize=True) * 100).round(2), flush=True)
