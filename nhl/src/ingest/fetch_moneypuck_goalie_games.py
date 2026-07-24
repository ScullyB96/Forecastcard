"""Build a per-goalie-per-game table (shots faced, xG faced, goals allowed)
from MoneyPuck's shot-level data, since MoneyPuck does NOT publish a bulk
goalie game-log file the way it does for teams (`all_teams.csv`) -- checked
directly (2026-07-23): moneypuck/playerData/careers/gameByGame/ only has
all_teams.csv; probing for all_goalies.csv/goalies.csv (guessing the naming
convention) both 404. The shot-level files DO have everything needed to
build this ourselves: `goalieIdForShot`, `xGoal`, `goal`, `shotWasOnGoal`,
`game_id` -- confirmed via a real sample (shots_2010.csv) that
shotWasOnGoal==1 rows are exactly SHOT (saved) + GOAL (scored) events,
i.e. exactly "shots the goalie actually faced" (MISS/blocked attempts,
which a goalie never has to react to, are shotWasOnGoal==0 and excluded).

The goalie's OWN team isn't given directly -- `team`/`teamCode` are the
SHOOTING team; confirmed by inspection that the goalie's team is always the
OTHER one of (homeTeamCode, awayTeamCode).

Files are large (326MB compressed for the 2007-2024 bundle, ~20MB more for
the current season) -- read in chunks and aggregate immediately down to a
goalie-game table (expected ~tens of thousands of rows, not millions), so
peak memory stays bounded regardless of the raw file's size.

BUG FOUND AND FIXED #2 (2026-07-23): the first corrected run produced a
suspicious systematic bias -- primary-goalie GSAx averaged -0.51 to -0.56
across the WHOLE dataset, when it should hover near 0 (xG is calibrated
against real goals in aggregate, so a goalie facing average shots should
have average GSAx). Investigated by checking `shotOnEmptyNet`: MoneyPuck
still populates `goalieIdForShot` for empty-net shots (confirmed real,
2026-07-23 -- contrary to what a stray dictionary comment implies) -- and
those shots are ~100% likely to score (confirmed: 485/485 empty-net
on-goal shots in a real sample season were actual goals) while `xGoal`
for them only averages ~0.61, not ~1.0. So every empty-net goal an
opponent scores counts as a near-certain "goal allowed" against a
~0.39-xG shot for whichever goalie is nominally "of record" -- even though
pulling the goalie is a coaching/team decision made BECAUSE the team is
trailing, not a reflection of that goalie's own shot-stopping skill. Fixed
by excluding `shotOnEmptyNet==1` rows before aggregating -- standard
practice in public goaltending analytics for exactly this reason.

BUG FOUND AND FIXED #1 (2026-07-23): the shots file's own `game_id` is NOT the
same ID scheme as `all_teams.csv`'s `gameId` (which matches the NHL API's
full 10-digit id directly, confirmed earlier in team_strength_xg.py) --
it's short (e.g. 20001) and RESETS EVERY SEASON, confirmed directly:
aggregating without a season qualifier collapsed 18 different seasons'
opening games onto one row (1,461 distinct "games" out of an expected
~20,000+). Confirmed by testing a reconstruction against the real schedule:
the shots file's `game_id` is the NHL id's own TYPE+SEQUENCE suffix with
the leading zero stripped by integer parsing (e.g. "020001" -> 20001);
concatenating the shots file's own `season` column back on top
(`int(f"{season}{game_id:06d})")` reproduces the exact real gameId --
verified against a real game (TOR/MTL, 2010-10-07, gameId 2010020001,
confirmed present in the real schedule with matching teams). A reminder
that even WITHIN one data provider, different files can use different ID
schemes for supposedly the same entity -- checked here, not assumed from
the team-level file's behavior.
"""

import zipfile
from io import BytesIO

import pandas as pd
import requests

from src.utils.paths import DATA_RAW

SHOTS_URLS = [
    "https://peter-tanner.com/moneypuck/downloads/shots_2007-2024.zip",
    "https://peter-tanner.com/moneypuck/downloads/shots_2025.zip",
]
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
USECOLS = ["game_id", "season", "homeTeamCode", "awayTeamCode", "isHomeTeam", "goalieIdForShot",
           "goalieNameForShot", "shotWasOnGoal", "shotOnEmptyNet", "goal", "xGoal"]
CHUNKSIZE = 500_000
GROUP_KEYS = ["gameId", "goalieIdForShot", "goalieNameForShot", "goalie_team"]


def _aggregate_goalie_games(csv_bytes: BytesIO) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(csv_bytes, usecols=USECOLS, chunksize=CHUNKSIZE, low_memory=False):
        chunk = chunk[(chunk["shotWasOnGoal"] == 1) & (chunk["shotOnEmptyNet"] == 0)].copy()
        chunk = chunk.dropna(subset=["goalieIdForShot"])
        # BUG FOUND AND FIXED (2026-07-23): originally compared the `team`
        # column against `homeTeamCode` to infer the shooting side -- but
        # `team`'s real value is the literal string "HOME"/"AWAY" (confirmed
        # via MoneyPuck's own data dictionary), never a team abbreviation, so
        # that comparison was unconditionally True and every goalie got
        # mislabeled as the home team. Confirmed directly on a real game
        # (2023020001, TBL home/NSH away): BOTH Juuse Saros (NSH's actual
        # goalie) and Jonas Johansson (TBL's) came out tagged "goalie_team"
        # ==TBL, which then collided/overwrote each other in the
        # (gameId, team) dedup downstream -- exactly explaining an observed
        # ~50% team-side match-rate against the schedule (every away goalie
        # silently lost). `isHomeTeam` (a real 0/1 flag, not a string) is the
        # correct column: the shooting team is home when 1, so the GOALIE is
        # the away team, and vice versa.
        chunk["goalie_team"] = chunk["awayTeamCode"].where(chunk["isHomeTeam"] == 1, chunk["homeTeamCode"])
        # Reconstruct the real NHL gameId -- see module docstring's bug writeup:
        # this file's own `game_id` resets every season and needs `season`
        # concatenated back on to become the same id scheme all_teams.csv uses.
        chunk["gameId"] = (chunk["season"].astype(str) + chunk["game_id"].astype(int).map(lambda g: f"{g:06d}")
                            ).astype(int)
        agg = chunk.groupby(GROUP_KEYS, as_index=False).agg(
            shots_faced=("goal", "size"), goals_allowed=("goal", "sum"), xg_faced=("xGoal", "sum"))
        chunks.append(agg)
    combined = pd.concat(chunks, ignore_index=True)
    # A goalie can appear in multiple chunks for the same game (chunk boundaries
    # don't respect game_id) -- re-aggregate across chunks before returning.
    return combined.groupby(GROUP_KEYS, as_index=False).agg(
        shots_faced=("shots_faced", "sum"), goals_allowed=("goals_allowed", "sum"), xg_faced=("xg_faced", "sum"))


def fetch_and_build_goalie_game_log() -> pd.DataFrame:
    all_frames = []
    for url in SHOTS_URLS:
        print(f"downloading {url} ...")
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300)
        resp.raise_for_status()
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                all_frames.append(_aggregate_goalie_games(f))
        print(f"  aggregated {len(all_frames[-1])} goalie-game rows from {csv_name}")
    combined = pd.concat(all_frames, ignore_index=True)
    # Two source files could in principle overlap at a season boundary -- dedupe defensively.
    combined = combined.drop_duplicates(subset=["gameId", "goalieIdForShot"])
    return combined.sort_values(["gameId", "goalieIdForShot"]).reset_index(drop=True)


if __name__ == "__main__":
    df = fetch_and_build_goalie_game_log()
    print(f"total: {len(df)} goalie-game rows, {df['goalieIdForShot'].nunique()} distinct goalies, "
          f"{df['gameId'].nunique()} distinct games")
    out_path = DATA_RAW / "moneypuck_goalie_games.parquet"
    df.to_parquet(out_path, index=False)
    print(f"saved to {out_path}")
