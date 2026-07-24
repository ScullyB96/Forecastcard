"""Parses the raw SportsbookReviewsOnline rows (src/ingest/fetch_sbro_odds.py)
into one row per real game, joined to this project's own schedule.

Three real parsing quirks handled here, each confirmed against the actual
data rather than assumed:
1. `Date` is an ambiguous MMDD integer with no year -- the year flips mid
   file (October-December rows belong to the season's start year,
   January-June rows belong to its end year). Resolved via the season
   label already attached by the fetch step.
2. Games are two consecutive rows (away then home; neutral-site games are
   two consecutive "N" rows, away then home in the same order) -- paired
   by position, not by any explicit game-id column (none exists in this
   data).
3. Team names are CITY-only in most seasons but include the nickname in a
   few ("SeattleKraken", "WinnipegJets"), and use different old-franchise
   names in older seasons ("Phoenix" pre-2014, "Atlanta" pre-2011) --
   confirmed the exact set of 37 distinct raw names in the real data
   before building this map, not guessed from general knowledge.
"""

import pandas as pd

from src.utils.paths import DATA_RAW

# SBRO raw team name -> the abbreviation the NHL API itself used IN THAT ERA
# (matches this project's own schedule data directly, which already carries
# era-accurate abbreviations straight from the NHL API -- e.g. PHX not ARI
# for pre-2014 games, ATL not WPG for pre-2011 games).
SBRO_TEAM_TO_ABBREV = {
    "Anaheim": "ANA", "Arizona": "ARI", "Arizonas": "ARI", "Atlanta": "ATL",
    "Boston": "BOS", "Buffalo": "BUF", "Calgary": "CGY", "Carolina": "CAR",
    "Chicago": "CHI", "Colorado": "COL", "Columbus": "CBJ", "Dallas": "DAL",
    "Detroit": "DET", "Edmonton": "EDM", "Florida": "FLA", "LosAngeles": "LAK",
    "Minnesota": "MIN", "Montreal": "MTL", "NYIslanders": "NYI", "NYRangers": "NYR",
    "Nashville": "NSH", "NewJersey": "NJD", "Ottawa": "OTT", "Philadelphia": "PHI",
    "Phoenix": "PHX", "Pittsburgh": "PIT", "SanJose": "SJS", "Seattle": "SEA",
    "SeattleKraken": "SEA", "St.Louis": "STL", "Tampa": "TBL", "TampaBay": "TBL",
    "Toronto": "TOR", "Vancouver": "VAN", "Vegas": "VGK", "Washington": "WSH",
    "Winnipeg": "WPG", "WinnipegJets": "WPG",
}


def _resolve_year(date_val: int, season_label: str) -> int:
    start_year = int(season_label[:4])
    month = int(date_val) // 100
    return start_year if month >= 7 else start_year + 1


def parse_games(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.reset_index(drop=True)
    if len(raw) % 2 != 0:
        raise ValueError(f"expected an even number of rows (two per game), got {len(raw)}")

    away = raw.iloc[0::2].reset_index(drop=True)
    home = raw.iloc[1::2].reset_index(drop=True)

    mismatched_dates = away["Date"] != home["Date"]
    if mismatched_dates.any():
        raise ValueError(f"{mismatched_dates.sum()} paired rows have mismatched Date values -- "
                          f"the away/home row-pairing assumption is broken for these rows")

    games = pd.DataFrame({
        "season_label": away["season_label"],
        "date_raw": away["Date"],
        "away_team_raw": away["Team"], "home_team_raw": home["Team"],
        "away_final": away["Final"], "home_final": home["Final"],
        "away_open_ml": away["Open"], "home_open_ml": home["Open"],
        "away_close_ml": away["Close"], "home_close_ml": home["Close"],
        "open_total": away["OpenOU"], "close_total": away["CloseOU"],
    })

    games["year"] = [
        _resolve_year(d, s) for d, s in zip(games["date_raw"], games["season_label"])
    ]
    month = games["date_raw"].astype(int) // 100
    day = games["date_raw"].astype(int) % 100
    games["gameDate"] = pd.to_datetime(dict(year=games["year"], month=month, day=day)).dt.strftime("%Y-%m-%d")

    games["away_team"] = games["away_team_raw"].map(SBRO_TEAM_TO_ABBREV)
    games["home_team"] = games["home_team_raw"].map(SBRO_TEAM_TO_ABBREV)
    unmapped = games[games["away_team"].isna() | games["home_team"].isna()]
    if len(unmapped):
        bad_names = pd.concat([unmapped["away_team_raw"], unmapped["home_team_raw"]]).unique()
        raise ValueError(f"{len(unmapped)} games have an unmapped team name: {bad_names}")

    return games


def american_odds_to_prob(odds: pd.Series) -> pd.Series:
    """Standard American-odds -> implied probability conversion (still
    includes the vig -- no-vig normalization happens separately, once both
    sides' raw implied probabilities are available).

    BUG FOUND AND FIXED (2026-07-23) before this was ever used for real
    numbers: the original chained-`.where()` version produced a NEGATIVE
    probability for negative odds (tested: -150 -> -0.6, not 0.6) -- caught
    by a direct unit check against known values before trusting it on the
    real data, not discovered downstream."""
    is_negative = odds < 0
    return pd.Series(
        data=(odds.abs() / (odds.abs() + 100)).where(is_negative, 100 / (odds + 100)),
        index=odds.index,
    )


def add_novig_probabilities(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    for tag, home_col, away_col in [("open", "home_open_ml", "away_open_ml"), ("close", "home_close_ml", "away_close_ml")]:
        p_home_raw = american_odds_to_prob(games[home_col])
        p_away_raw = american_odds_to_prob(games[away_col])
        total = p_home_raw + p_away_raw
        games[f"{tag}_home_prob_novig"] = p_home_raw / total
    return games


def join_to_schedule(games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    sched = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")][
        ["gameId", "gameDate", "homeTeamAbbrev", "awayTeamAbbrev", "homeScore", "awayScore"]
    ].rename(columns={"homeTeamAbbrev": "home_team", "awayTeamAbbrev": "away_team"})
    merged = games.merge(sched, on=["gameDate", "home_team", "away_team"], how="left")

    score_mismatch = merged.dropna(subset=["gameId"])
    score_mismatch = score_mismatch[
        (score_mismatch["home_final"] != score_mismatch["homeScore"])
        | (score_mismatch["away_final"] != score_mismatch["awayScore"])
    ]
    return merged, score_mismatch


if __name__ == "__main__":
    raw = pd.read_parquet(DATA_RAW / "sbro_odds_raw.parquet")
    games = parse_games(raw)
    games = add_novig_probabilities(games)
    print(f"parsed {len(games)} games from {games['season_label'].nunique()} seasons")

    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    merged, score_mismatch = join_to_schedule(games, schedule)
    n_matched = merged["gameId"].notna().sum()
    print(f"matched {n_matched}/{len(merged)} games to the real schedule by (date, home, away)")
    print(f"of those, {len(score_mismatch)} have a FINAL SCORE mismatch (a real cross-check, "
          f"not just a join success check)")
    if len(score_mismatch):
        print(score_mismatch[["gameDate", "home_team", "away_team", "home_final", "homeScore",
                               "away_final", "awayScore"]].head(10))

    out_path = DATA_RAW / "sbro_odds_games.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
