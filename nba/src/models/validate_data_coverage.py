"""Phase 0's acceptance gate: is the raw cache trustworthy enough to build
on? No model-quality metric applies here -- this just reports, per season,
whether each raw source (schedule / advanced box score / play-by-play /
rotation) covers every game the schedule says should exist, and prints an
explicit missing-games report rather than silently building on a gap.

Run as `python -m src.models.validate_data_coverage`.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, current_nba_season, season_str
from src.utils.paths import DATA_RAW


def _season_ids(path_pattern: str, start_year: int, id_col: str) -> set:
    path = DATA_RAW / path_pattern.format(season=season_str(start_year))
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=[id_col])
    return set(df[id_col].unique())


def check_season(start_year: int) -> dict:
    schedule_path = DATA_RAW / f"schedule_{season_str(start_year)}.parquet"
    if not schedule_path.exists():
        return {"season": season_str(start_year), "status": "NO SCHEDULE"}

    schedule = pd.read_parquet(schedule_path)
    expected = set(schedule["gameId"].unique())

    have_box = _season_ids("boxscore_adv_team_{season}.parquet", start_year, "gameId")
    have_pbp = _season_ids("playbyplay_{season}.parquet", start_year, "gameId")
    have_rot = _season_ids("rotation_{season}.parquet", start_year, "GAME_ID")

    return {
        "season": season_str(start_year),
        "n_expected": len(expected),
        "missing_box": sorted(expected - have_box),
        "missing_pbp": sorted(expected - have_pbp),
        "missing_rot": sorted(expected - have_rot),
    }


if __name__ == "__main__":
    current = current_nba_season()
    print(f"data coverage report: {season_str(FIRST_DEV_SEASON)}..{season_str(current)}\n", flush=True)

    any_gap = False
    for y in range(FIRST_DEV_SEASON, current + 1):
        r = check_season(y)
        if r.get("status") == "NO SCHEDULE":
            print(f"{r['season']}: NO SCHEDULE CACHED -- run fetch_schedule.py first", flush=True)
            any_gap = True
            continue

        n = r["n_expected"]
        gaps = []
        for label, key in (("box", "missing_box"), ("pbp", "missing_pbp"), ("rot", "missing_rot")):
            missing = r[key]
            if missing:
                gaps.append(f"{label}={len(missing)}/{n} missing")
        status = "PASS" if not gaps else "INCOMPLETE (" + ", ".join(gaps) + ")"
        if gaps:
            any_gap = True
        print(f"{r['season']}: {n} games -- {status}", flush=True)
        for label, key in (("box", "missing_box"), ("pbp", "missing_pbp"), ("rot", "missing_rot")):
            missing = r[key]
            if missing and len(missing) <= 5:
                print(f"    {label} missing: {missing}", flush=True)

    print(flush=True)
    print("OVERALL: " + ("FAIL -- gaps present, see above" if any_gap else "PASS -- all sources fully cover the schedule"), flush=True)
