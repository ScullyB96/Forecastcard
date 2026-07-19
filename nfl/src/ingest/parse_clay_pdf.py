"""Parse Mike Clay's 2026 ESPN Fantasy Projection Guide PDF into structured
data: the real 2026 schedule with ESPN's implied score/win-probability per
game, and player season-total projections.

This PDF is the only source we have for the actual 2026 schedule (nflverse
has no 2026 data yet -- the season hasn't been played) and for offseason
intelligence (trades, coaching changes, rookie evaluations, depth charts) that
a historical-EPA-only model has no way to see. ESPN's implied scores are used
here as an external benchmark/prior, not blindly trusted -- see predict_2026.py
for how it's combined with our own model.
"""

import re

import pandas as pd
import pdfplumber

from src.utils.paths import DATA_RAW

PDF_PATH = "/Users/brettscully/Downloads/NFLDK2026_CS_ClayProjections2026.pdf"
TEAM_PAGES = range(1, 33)  # 0-indexed pages 1..32 == printed pages 2..33

# the PDF's 3-letter codes differ from nflverse's standard codes for 5 teams
OPP_CODE_FIXES = {"CLV": "CLE", "BLT": "BAL", "ARZ": "ARI", "LAR": "LA", "HST": "HOU"}

# full team name (as extracted from each page title) -> nflverse standard code
TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

SCHEDULE_ROW = re.compile(r"(?<!\d)(\d{1,2})\s+([A-Z]{2,3})\s+([HV])\s+([\d.]+)\s+([\d.]+)\s+(\d+)%")

PLAYER_ROW = re.compile(
    r"^(QB|RB|WR|TE)\s+(?!Total)([A-Za-z.\'\-]+(?:\s[A-Za-z.\'\-]+){0,3}?)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\b"
)


def extract_team_name(text: str) -> str:
    m = re.search(r"2026 (.+?) Projections", text)
    return m.group(1)


def extract_schedule(text: str, team_name: str) -> pd.DataFrame:
    rows = SCHEDULE_ROW.findall(text)
    df = pd.DataFrame(rows, columns=["week", "opp", "loc", "team_pts", "opp_pts", "win_prob_pct"])
    df["team"] = TEAM_NAME_TO_CODE[team_name]
    df["opp"] = df["opp"].replace(OPP_CODE_FIXES)
    df["week"] = df["week"].astype(int)
    df["team_pts"] = df["team_pts"].astype(float)
    df["opp_pts"] = df["opp_pts"].astype(float)
    df["win_prob"] = df["win_prob_pct"].astype(float) / 100.0
    df["is_home"] = df["loc"] == "H"
    return df[["team", "week", "opp", "is_home", "team_pts", "opp_pts", "win_prob"]]


def extract_players(text: str, team_name: str) -> pd.DataFrame:
    records = []
    for line in text.split("\n"):
        m = PLAYER_ROW.match(line.strip())
        if not m:
            continue
        pos, name, gm, att, comp, yds, td, interc, sk, r_att, r_yds, r_td, tgt, rec, r_yd, r_td2, pts, rk = m.groups()
        records.append(
            {
                "team": TEAM_NAME_TO_CODE[team_name],
                "position": pos,
                "player": name.strip(),
                "games": int(gm),
                "pass_att": int(att),
                "pass_comp": int(comp),
                "pass_yds": int(yds),
                "pass_td": int(td),
                "interceptions": int(interc),
                "sacked": int(sk),
                "rush_att": int(r_att),
                "rush_yds": int(r_yds),
                "rush_td": int(r_td),
                "targets": int(tgt),
                "receptions": int(rec),
                "rec_yds": int(r_yd),
                "rec_td": int(r_td2),
                "ppr_pts": int(pts),
                "ppr_rank": int(rk),
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    schedules = []
    players = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for i in TEAM_PAGES:
            text = pdf.pages[i].extract_text() or ""
            team_name = extract_team_name(text)
            sched = extract_schedule(text, team_name)
            plyrs = extract_players(text, team_name)
            schedules.append(sched)
            players.append(plyrs)
            print(f"page {i+1}: {team_name} -- {len(sched)} schedule rows, {len(plyrs)} player rows")

    schedule_df = pd.concat(schedules, ignore_index=True)
    player_df = pd.concat(players, ignore_index=True)

    # sanity check: every (team, week, opp) should have a reciprocal (opp, week, team) row
    # with matching (swapped) implied scores -- confirms the extraction is internally consistent
    check = schedule_df.merge(
        schedule_df, left_on=["team", "week", "opp"], right_on=["opp", "week", "team"], suffixes=("", "_r")
    )
    mismatches = check[
        (check["team_pts"] - check["opp_pts_r"]).abs() > 0.15
    ]
    print(f"\nreciprocal consistency check: {len(check)} paired rows, {len(mismatches)} mismatches > 0.15 pts")
    if len(mismatches):
        print(mismatches[["team", "opp", "week", "team_pts", "opp_pts_r"]].head(10).to_string(index=False))

    schedule_df.to_parquet(DATA_RAW / "clay_2026_schedule.parquet", index=False)
    player_df.to_parquet(DATA_RAW / "clay_2026_player_projections.parquet", index=False)
    print(f"\nsaved {len(schedule_df)} schedule rows, {len(player_df)} player rows")
    print("\nsample schedule:")
    print(schedule_df.head(10).to_string(index=False))
    print("\nsample players:")
    print(player_df.head(10).to_string(index=False))
