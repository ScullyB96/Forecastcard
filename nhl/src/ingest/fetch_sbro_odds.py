"""Pull historical NHL closing-line odds from Wayback Machine snapshots of
the ORIGINAL SportsbookReviewsOnline per-season .xlsx files -- see
MODEL_DOCUMENTATION.md Sec6.1 for the full investigation. The CURRENT live
site is a dead end (every season's page, including seasons finished years
ago, is truncated to the same ~November 27 cutoff -- confirmed the entire
archive was frozen/imported once around December 2022 and never updated).
The original site's per-season xlsx files are still recoverable via the
Wayback Machine and are genuinely complete FOR SEASONS THAT HAD ALREADY
FINISHED before that December 2022 freeze -- confirmed directly on three
real files (2022-23: truncated mid-season, matching the freeze theory;
2018-19 and 2015-16: both a real complete regular season + playoffs,
Oct-through-June row ranges).

KNOWN GAP, not silently worked around: seasons 2022-23 onward (including
this project's entire 2024-25/2025-26 holdout) are NOT recoverable complete
through this source -- the live copies are frozen mid-season and no later
Wayback crawl exists. This script only pulls the seasons confirmed
recoverable; callers should not assume market data exists for recent
seasons without checking.
"""

import io
import time

import pandas as pd
import requests

from src.utils.paths import DATA_RAW

REQUEST_DELAY_SECONDS = 1.0  # politeness delay between archive.org requests

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Seasons confirmed (or expected, by the same freeze-timing logic) to have
# finished before the ~Dec 2022 Wayback crawl -- 2022-23 explicitly excluded
# since it was confirmed truncated mid-season at that exact crawl.
RECOVERABLE_SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2007, 2022)]  # 2007-08 .. 2021-22

WAYBACK_AVAILABLE_API = "http://archive.org/wayback/available"
ORIGINAL_FILE_URL_TMPL = ("https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/"
                          "nhl%20odds%20{season}.xlsx")


def _find_snapshot_url(season: str) -> str | None:
    target_url = f"www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%20{season}.xlsx"
    resp = requests.get(WAYBACK_AVAILABLE_API, params={"url": target_url, "timestamp": "20230601"}, timeout=20)
    resp.raise_for_status()
    snap = resp.json().get("archived_snapshots", {}).get("closest")
    return snap["url"] if snap else None


def _get_with_retry(url: str, **kwargs) -> requests.Response:
    """archive.org is occasionally slow/flaky (confirmed real: a transient
    ConnectTimeout hit mid-run on 2026-07-23) -- a couple of retries with
    backoff is more robust than a bare single attempt, without masking a
    genuine, persistent failure."""
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(5 * (attempt + 1))
    raise last_exc


def fetch_season_raw(season: str) -> pd.DataFrame | None:
    """Returns the raw per-row (team-game side) table for one season, or
    None if no Wayback snapshot exists at all for that season's file."""
    snapshot_url = _find_snapshot_url(season)
    if snapshot_url is None:
        print(f"  {season}: no Wayback snapshot found, skipping")
        return None
    resp = _get_with_retry(snapshot_url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_DELAY_SECONDS)
    df = pd.read_excel(io.BytesIO(resp.content))
    df["season_label"] = season
    return df


def fetch_all_recoverable_seasons() -> pd.DataFrame:
    frames = []
    for season in RECOVERABLE_SEASONS:
        print(f"fetching {season} ...")
        df = fetch_season_raw(season)
        if df is not None:
            n_games = len(df) // 2
            print(f"  {season}: {len(df)} rows (~{n_games} games)")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = fetch_all_recoverable_seasons()
    print(f"\ntotal: {len(df)} rows across {df['season_label'].nunique()} seasons")

    # Real data-quality quirk, confirmed (2026-07-23): some odds cells contain the literal
    # string "NL" ("No Line" -- the total wasn't posted for that game), mixed into otherwise-
    # numeric columns. Coerce to numeric (NaN for any non-numeric placeholder) rather than
    # letting parquet's type inference fail outright -- these are genuine missing values,
    # not a parsing bug, so NaN is the correct representation, not a guessed number.
    numeric_cols = ["1st", "2nd", "3rd", "Final", "Open", "Close", "PuckLine", "OpenOU", "CloseOU"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop the "Unnamed: N" ghost columns from merged header cells in the original xlsx
    # formatting (confirmed real, varies by season's exact file) -- not real data.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    out_path = DATA_RAW / "sbro_odds_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
