"""Download MoneyPuck's free team-game-situation CSV and cache it locally.

Confirmed live (2026-07-23, see MODEL_DOCUMENTATION.md Sec1.2): plain HTTPS
GET, no login/paywall, no bot-block for a generic user agent. One row per
team-game-situation (5on5/5on4/4on5/all/other); covers the 2008 season
through the just-finished 2025-26 season (game dates confirmed through
2026-06-14). File is refreshed in place during the season, so re-running
this script later re-pulls the current version rather than assuming the
2026-07-23 snapshot stays valid forever.
"""

import pandas as pd
import requests

from src.utils.paths import DATA_RAW

ALL_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_all_teams_gamelog() -> pd.DataFrame:
    resp = requests.get(ALL_TEAMS_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


if __name__ == "__main__":
    df = fetch_all_teams_gamelog()
    print(f"fetched {len(df)} rows, seasons {sorted(df['season'].unique())[:3]}"
          f"...{sorted(df['season'].unique())[-3:]}")
    print(df["situation"].value_counts())
    print("distinct team codes:", sorted(df["team"].unique()))
    out_path = DATA_RAW / "moneypuck_all_teams.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
