"""Historical NBA odds/scores from SportsbookReviewsOnline -- calibration/
benchmark data only (NOT a training feature; see MODEL_DOCUMENTATION.md's
data-sources section for why market odds are deliberately kept out of the
model itself). Cached under `data/external/` per this project's convention
for manually-acquired reference data.

Confirmed live (2026-07-24): despite looking like a client-rendered SPA when
navigated interactively, `/scoresoddsarchives/nba-odds-<season>/` is a plain
server-rendered WordPress page -- a bare `requests.get` with a browser
User-Agent returns the full HTML directly, no browser automation needed.
Coverage is 2007-08 through 2022-23 ONLY (the page's own text says "will
not be updated") -- this does NOT cover 2023-24 onward, so it cannot
benchmark the most recent dev seasons or any holdout season. Treated as a
nice-to-have partial calibration check, not a requirement.

Table format (2 rows per game, `Rot` numbers pair consecutively, first row
= away/V, second = home/H): `Date` (MMDD, no year -- reconstructed from the
season's Oct-Jun boundary), quarter-by-quarter score columns, `Final`,
`Open`/`Close` (one row of the pair carries the TOTAL -- a value >= 100,
since NBA totals run ~180-240 -- and the OTHER carries the POINT SPREAD, a
small number; this is SBRO's well-known format quirk across sports, not
specific to this project), `ML` (moneyline), `2H` (second-half line, same
total-vs-spread ambiguity, not parsed here -- not needed for a full-game
benchmark)."""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.ingest.team_codes import ABBREV_TO_TEAM_ID
from src.utils.paths import DATA_EXTERNAL

BASE_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}/"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
FIRST_AVAILABLE_SEASON = 2007
LAST_AVAILABLE_SEASON = 2022  # confirmed live: page's own text says this archive "will not be updated"
TOTAL_THRESHOLD = 100.0  # a value at/above this in Open/Close is a total, not a spread

# SBRO concatenates city+nickname with no spaces/punctuation; only current-era (2015+) franchises
# are covered here since fetch_odds_sbro is only ever called for FIRST_DEV_SEASON (2015) onward.
_SBRO_NAME_TO_ABBREV = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA", "Chicago": "CHI",
    "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "GoldenState": "GSW",
    "Houston": "HOU", "Indiana": "IND", "LAClippers": "LAC", "LALakers": "LAL", "Memphis": "MEM",
    "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN", "NewOrleans": "NOP", "NewYork": "NYK",
    "OklahomaCity": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX",
    "Portland": "POR", "Sacramento": "SAC", "SanAntonio": "SAS", "Toronto": "TOR", "Utah": "UTA",
    "Washington": "WAS",
}


def _fetch_raw_table(season_start_year: int) -> list[list[str]]:
    season_str = f"{season_start_year}-{str((season_start_year + 1) % 100).zfill(2)}"
    resp = requests.get(BASE_URL.format(season=season_str), headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")
    header = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    return header, [[c.get_text(strip=True) for c in r.find_all(["th", "td"])] for r in rows[1:]]


def _reconstruct_date(mmdd: str, season_start_year: int) -> str:
    """MMDD with no year -- season runs Oct (start_year) through Jun
    (start_year+1); month <= 7 belongs to the following calendar year."""
    month = int(mmdd[:-2]) if len(mmdd) > 2 else int(mmdd[0])
    day = int(mmdd[-2:])
    year = season_start_year + 1 if month <= 7 else season_start_year
    return f"{year:04d}-{month:02d}-{day:02d}"


def _to_float(v: str) -> float | None:
    if v in ("", "NL"):
        return None
    if v.lower() == "pk":
        return 0.0
    try:
        return float(v)
    except ValueError:
        return None


def _split_spread_and_total(open_val: str, close_val: str) -> tuple:
    """Classifies OPEN and CLOSE INDEPENDENTLY (not as a single row-level
    total-vs-spread decision) -- confirmed necessary on real data: some
    rows genuinely mix conventions within the same row (e.g. a real
    2015-16 game's away-side row read Open=2 [a spread value] but
    Close=191 [a total value]). Returns (spread_val_or_None,
    total_val_or_None) for open, then the same pair for close."""
    o, c = _to_float(open_val), _to_float(close_val)
    o_is_total = o is not None and o >= TOTAL_THRESHOLD
    c_is_total = c is not None and c >= TOTAL_THRESHOLD
    return (
        (None if o_is_total else o), (o if o_is_total else None),
        (None if c_is_total else c), (c if c_is_total else None),
    )


def parse_season(season_start_year: int) -> pd.DataFrame:
    header, raw_rows = _fetch_raw_table(season_start_year)
    idx = {name: i for i, name in enumerate(header)}
    games = []

    for i in range(0, len(raw_rows) - 1, 2):
        r1, r2 = raw_rows[i], raw_rows[i + 1]
        if len(r1) < len(header) or len(r2) < len(header):
            continue
        vh1, vh2 = r1[idx["VH"]], r2[idx["VH"]]
        if not (vh1 == "V" and vh2 == "H"):
            continue  # unexpected pairing, skip rather than guess

        away_row, home_row = r1, r2
        away_name = _SBRO_NAME_TO_ABBREV.get(away_row[idx["Team"]])
        home_name = _SBRO_NAME_TO_ABBREV.get(home_row[idx["Team"]])
        if away_name is None or home_name is None:
            continue

        a_so, a_to_o, a_sc, a_to_c = _split_spread_and_total(away_row[idx["Open"]], away_row[idx["Close"]])
        h_so, h_to_o, h_sc, h_to_c = _split_spread_and_total(home_row[idx["Open"]], home_row[idx["Close"]])
        total_open = a_to_o if a_to_o is not None else h_to_o
        total_close = a_to_c if a_to_c is not None else h_to_c
        # the spread value is from the perspective of the team it's printed on (a positive number
        # = that team is the underdog getting points); store signed from HOME perspective. Prefer
        # the home row's own spread reading; fall back to negating the away row's if the home
        # row's close happened to be classified as the total instead (the real per-row mixing
        # confirmed above).
        home_spread_close = h_sc if h_sc is not None else (-a_sc if a_sc is not None else None)
        home_spread_open = h_so if h_so is not None else (-a_so if a_so is not None else None)

        try:
            games.append({
                "gameDate": _reconstruct_date(away_row[idx["Date"]], season_start_year),
                "awayTeamAbbrev": away_name, "homeTeamAbbrev": home_name,
                "awayTeamId": ABBREV_TO_TEAM_ID[away_name], "homeTeamId": ABBREV_TO_TEAM_ID[home_name],
                "awayScore": int(away_row[idx["Final"]]), "homeScore": int(home_row[idx["Final"]]),
                "totalOpen": total_open, "totalClose": total_close,
                "homeSpreadOpen": home_spread_open, "homeSpreadClose": home_spread_close,
                "awayMoneyline": away_row[idx["ML"]], "homeMoneyline": home_row[idx["ML"]],
            })
        except (ValueError, KeyError):
            continue

    return pd.DataFrame(games)


def fetch_odds_season(start_year: int, force: bool = False) -> pd.DataFrame:
    season_str = f"{start_year}-{str((start_year + 1) % 100).zfill(2)}"
    path = DATA_EXTERNAL / f"odds_sbro_{season_str}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    if not (FIRST_AVAILABLE_SEASON <= start_year <= LAST_AVAILABLE_SEASON):
        print(f"{season_str}: outside SBRO's available range ({FIRST_AVAILABLE_SEASON}-{LAST_AVAILABLE_SEASON}), skipping", flush=True)
        return pd.DataFrame()

    df = parse_season(start_year)
    df.to_parquet(path, index=False)
    print(f"{season_str}: parsed {len(df)} games, saved {path}", flush=True)
    return df


if __name__ == "__main__":
    import sys

    from src.ingest.fetch_schedule import FIRST_DEV_SEASON

    end = min(LAST_AVAILABLE_SEASON, int(sys.argv[1])) if len(sys.argv) > 1 else LAST_AVAILABLE_SEASON
    for y in range(FIRST_DEV_SEASON, end + 1):
        fetch_odds_season(y)
