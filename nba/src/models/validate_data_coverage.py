"""Phase 0's acceptance gate: is the raw cache trustworthy enough to build
on? No model-quality metric applies here -- this just reports, per season,
whether each raw source (schedule / advanced box score / play-by-play /
traditional box score) covers every game the schedule says should exist,
and prints an explicit missing-games report rather than silently building
on a gap.

REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): this used to
check `rotation_{season}.parquet` (`GameRotation`) instead of
`boxscore_trad_player_{season}.parquet` -- but Sec5 of
MODEL_DOCUMENTATION.md documents `GameRotation` as ABANDONED ENTIRELY
(replaced by a substitution-event reconstruction from
`BoxScoreTraditionalV3` + `PlayByPlayV3`), and only
`data/raw/rotation_2015-16.parquet` exists on disk at all -- no
`rotation_*.parquet` for 2016-17 onward. Running this validator therefore
reported `missing_rot` as the ENTIRE expected game count for every season
except a partial 2015-16, printing a permanent, misleading `OVERALL: FAIL`
even though the source `build_stints.py` actually depends on
(`boxscore_trad_player_{season}.parquet`) was fully backfilled -- and this
validator never checked that source at all, so it neither confirmed real
completeness nor would have caught a genuine gap in it. Fixed to check the
traditional box score instead of the abandoned rotation feed.

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
    have_trad = _season_ids("boxscore_trad_player_{season}.parquet", start_year, "gameId")

    return {
        "season": season_str(start_year),
        "n_expected": len(expected),
        "missing_box": sorted(expected - have_box),
        "missing_pbp": sorted(expected - have_pbp),
        "missing_trad": sorted(expected - have_trad),
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
        for label, key in (("box", "missing_box"), ("pbp", "missing_pbp"), ("trad", "missing_trad")):
            missing = r[key]
            if missing:
                gaps.append(f"{label}={len(missing)}/{n} missing")
        status = "PASS" if not gaps else "INCOMPLETE (" + ", ".join(gaps) + ")"
        if gaps:
            any_gap = True
        print(f"{r['season']}: {n} games -- {status}", flush=True)
        for label, key in (("box", "missing_box"), ("pbp", "missing_pbp"), ("trad", "missing_trad")):
            missing = r[key]
            if missing and len(missing) <= 5:
                print(f"    {label} missing: {missing}", flush=True)

    print(flush=True)
    print("OVERALL: " + ("FAIL -- gaps present, see above" if any_gap else "PASS -- all sources fully cover the schedule"), flush=True)
