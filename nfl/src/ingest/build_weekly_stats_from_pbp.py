"""Derive weekly player stats directly from our own play-by-play data.

Why this exists: nfl_data_py's import_weekly_data() (the source
weekly_player_stats_2016_2024.parquet came from) has not published a 2025
release, and third-party Kaggle re-exports of the same underlying source
introduce a trust surface we don't control. But nflverse's PBP data -- which
we already have through 2025 and have used all session -- contains everything
needed to compute the same weekly aggregates ourselves. import_weekly_data()
is itself just this kind of aggregation under the hood.

Output schema matches what player_usage.py expects, so this is a drop-in
extension of weekly_player_stats_2016_2024.parquet, not a new pipeline.
"""

import pandas as pd

from src.utils.paths import DATA_RAW


def build_weekly_stats(pbp: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    pbp = pbp[pbp["season_type"] == "REG"]

    pass_plays = pbp[(pbp["play_type"] == "pass") & pbp["passer_player_id"].notna()]
    passing = (
        pass_plays.groupby(["season", "week", "posteam", "passer_player_id"])
        .agg(
            attempts=("passer_player_id", "size"),
            passing_epa=("epa", "sum"),
            completions=("complete_pass", "sum"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("pass_touchdown", "sum"),
            interceptions=("interception", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "recent_team", "passer_player_id": "player_id"})
    )

    rec_plays = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()].copy()
    rec_plays["air_yards"] = rec_plays["air_yards"].fillna(0.0)
    receiving = (
        rec_plays.groupby(["season", "week", "posteam", "receiver_player_id"])
        .agg(
            targets=("receiver_player_id", "size"),
            receptions=("complete_pass", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("pass_touchdown", "sum"),
            air_yards=("air_yards", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "recent_team", "receiver_player_id": "player_id"})
    )

    rush_plays = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()]
    rushing = (
        rush_plays.groupby(["season", "week", "posteam", "rusher_player_id"])
        .agg(
            carries=("rusher_player_id", "size"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rush_touchdown", "sum"),
        )
        .reset_index()
        .rename(columns={"posteam": "recent_team", "rusher_player_id": "player_id"})
    )

    merged = receiving.merge(
        rushing, on=["season", "week", "recent_team", "player_id"], how="outer"
    ).merge(passing, on=["season", "week", "recent_team", "player_id"], how="outer")
    for c in ["targets", "receptions", "receiving_yards", "receiving_tds", "air_yards", "carries", "rushing_yards",
              "rushing_tds", "attempts", "passing_epa", "completions", "passing_yards", "passing_tds",
              "interceptions"]:
        merged[c] = merged[c].fillna(0)

    # position + display name from weekly_rosters (already covers through 2025)
    ros = rosters[["season", "week", "team", "player_id", "position", "player_name"]].rename(
        columns={"team": "recent_team", "player_name": "player_display_name"}
    )
    merged = merged.merge(ros, on=["season", "week", "recent_team", "player_id"], how="left")
    merged["season_type"] = "REG"
    return merged


if __name__ == "__main__":
    pbp = pd.read_parquet(
        DATA_RAW / "pbp_2016_2025.parquet",
        columns=[
            "season", "week", "season_type", "posteam", "play_type", "receiver_player_id",
            "rusher_player_id", "passer_player_id", "epa", "complete_pass", "receiving_yards",
            "rushing_yards", "pass_touchdown", "rush_touchdown", "passing_yards", "interception",
        ],
    )
    pbp_2025 = pbp[pbp["season"] == 2025]
    rosters = pd.read_parquet(DATA_RAW / "weekly_rosters_2016_2025.parquet")

    weekly_2025 = build_weekly_stats(pbp_2025, rosters)
    print(f"built {len(weekly_2025)} player-week rows for 2025")
    print(weekly_2025[weekly_2025["targets"] >= 5].sort_values("receiving_yards", ascending=False).head(10))

    existing = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2024.parquet")
    combined_cols = [c for c in existing.columns if c in weekly_2025.columns]
    combined = pd.concat([existing[combined_cols], weekly_2025[combined_cols]], ignore_index=True)
    combined.to_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet", index=False)
    print(f"\nsaved weekly_player_stats_2016_2025.parquet ({len(combined)} rows, seasons {sorted(combined['season'].unique())})")
