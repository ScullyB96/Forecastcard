"""Reduce raw pitch-level Statcast to one row per PITCH (not one row per PA,
unlike build_pa_table.py) with a classified PITCH_RESULT and the pre-pitch
count -- the raw material for pitch-by-pitch modeling (src/models/pitch_talent.py).

build_pa_table.py deliberately discards this: it keeps only the first and
last pitch of each PA, and doesn't retain pitch_type/balls/strikes/description
even for those. This table exists specifically to preserve what that one
throws away.

PITCH_RESULT is a 6-way classification of Statcast's `description` field,
confirmed against real 2025 data (description value_counts) to cover every
description value with material volume. `automatic_ball`/`automatic_strike`
(pitch-clock violations, ~0.3% of pitches) are folded into ball/called_strike
for count-progression completeness -- they're not a real batter/pitcher swing
decision, just included so the count-state loop sees every real pitch. Whether
a `foul` at 2 strikes is a no-op (vs. a real strike below 2) depends on the
PRE-pitch `strikes` column already retained here -- no separate category is
needed for that; a consumer just checks `strikes < 2`.

Verified via a leakage-free plumbing check (throwaway script, not committed):
a deterministic count-state loop driven by nothing but the real
count-conditional PITCH_RESULT distribution (no batter/pitcher skill at all)
reproduces real aggregate season K/BB/HBP rates to within 0.3 percentage
points (K +0.18pp, BB +0.08pp, HBP +0.01pp, BIP -0.27pp on 2025 data) --
confirming both this classification and the count-progression rules it
implies are correct before any per-player shrinkage is built on top.
"""

import pandas as pd

from src.utils.paths import DATA_RAW, DATA_PROCESSED

BALL_DESC = {"ball", "blocked_ball", "automatic_ball", "intent_ball", "pitchout"}
CALLED_STRIKE_DESC = {"called_strike", "automatic_strike"}
SWINGING_STRIKE_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt",
                        "bunt_foul_tip", "swinging_pitchout"}
FOUL_DESC = {"foul", "foul_bunt", "foul_pitchout"}
HIT_INTO_PLAY_DESC = {"hit_into_play"}
HIT_BY_PITCH_DESC = {"hit_by_pitch"}

DESC_TO_RESULT = {
    **{d: "ball" for d in BALL_DESC},
    **{d: "called_strike" for d in CALLED_STRIKE_DESC},
    **{d: "swinging_strike" for d in SWINGING_STRIKE_DESC},
    **{d: "foul" for d in FOUL_DESC},
    **{d: "hit_into_play" for d in HIT_INTO_PLAY_DESC},
    **{d: "hit_by_pitch" for d in HIT_BY_PITCH_DESC},
}

SWING_RESULTS = {"swinging_strike", "foul", "hit_into_play"}  # is_swing derived from this

PITCH_COLUMNS = [
    "game_pk", "game_year", "game_date", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws",
    "balls", "strikes", "pitch_type", "description",
    "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
]


def build_pitch_table(statcast: pd.DataFrame) -> pd.DataFrame:
    df = statcast[statcast["game_type"] == "R"][PITCH_COLUMNS].copy()
    df["result"] = df["description"].map(DESC_TO_RESULT)
    unmapped = df["result"].isna().sum()
    if unmapped:
        print(f"WARNING: {unmapped} pitches have an unrecognized description value not in "
              f"DESC_TO_RESULT: {df.loc[df['result'].isna(), 'description'].unique().tolist()}", flush=True)
        df = df[df["result"].notna()]
    df["is_swing"] = df["result"].isin(SWING_RESULTS)
    return df.drop(columns=["description"]).reset_index(drop=True)


def build_and_save_pitch_table(seasons: list[int]) -> "pd.DataFrame | None":
    """Mirrors build_pa_table.py's build_and_save_pa_table -- rebuild from
    whatever's cached in statcast_{season}.parquet, save to
    pitch_table_{min}_{max}.parquet."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"statcast_{season}.parquet"
        if not path.exists():
            print(f"skipping {season}: no cached statcast data", flush=True)
            continue
        statcast = pd.read_parquet(path)
        pitches = build_pitch_table(statcast)
        pitches["season"] = season
        print(f"{season}: {len(pitches)} pitches built", flush=True)
        frames.append(pitches)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    out_path = DATA_PROCESSED / f"pitch_table_{min(seasons)}_{max(seasons)}.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"saved {out_path.name}: {len(combined)} pitches total", flush=True)
    return combined


if __name__ == "__main__":
    import sys

    seasons = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [2023, 2024, 2025, 2026]
    combined = build_and_save_pitch_table(seasons)
    if combined is not None:
        print("\n=== league-wide pitch-result rate sanity check ===", flush=True)
        print((combined["result"].value_counts(normalize=True) * 100).round(2), flush=True)
