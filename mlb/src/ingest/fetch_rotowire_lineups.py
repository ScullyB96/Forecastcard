"""Scrape RotoWire's daily MLB starting-lineups page as a higher-fidelity
fallback for lineup projection than this project's own "most recent complete
lineup" heuristic (see lineup_projection.py). Real, editorially-curated
confirmed/expected lineups account for rest days, roster moves, platoon
decisions, and injury news that a pure recency-based projection can't --
confirmed directly on a real case (2026-07-22): this project's own
project_lineup_platoon_aware included Randal Grichuk in the CWS @ TEX
prediction; he appears zero times anywhere on RotoWire's page for that game
(the real Expected Rangers lineup that day was Antonacci/Murakami/Vargas/
Montgomery/... instead).

Static HTML -- confirmed via a plain `requests.get` with no JS execution:
34 confirmed/expected lineup markers came through cleanly. robots.txt allows
this path for a general user-agent, and RotoWire's own llms.txt explicitly
lists the daily-lineups page as a public informational resource for LLMs.

The one real integration gap: RotoWire's player links use ITS OWN internal
player-ID scheme, not the MLB Stats API player_id this whole project is
keyed on. Names are matched instead (RotoWire gives full, unabbreviated
names via each player link's `title` attribute), scoped to that specific
team's own recent roster (see match_rotowire_names_to_player_ids) to keep
matching unambiguous -- a player newly confirmed for team X has, in the
overwhelming majority of cases, already appeared in one of team X's own
real lineups this season; the rare exception (a brand-new trade/debut)
falls back to unmatched rather than guessing.

Also handles the one confirmed team-abbreviation mismatch between RotoWire
and the MLB Stats API (RotoWire uses "ARI", this project's own data uses
"AZ" for Arizona) via ROTOWIRE_TEAM_ABBR_FIXUP -- extend this dict if
another mismatch is ever found.
"""

import re
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# RotoWire team abbreviation -> this project's own (MLB Stats API-derived) abbreviation.
# Confirmed mismatch (2026-07-22): RotoWire uses "ARI" for Arizona; this project's
# schedule/lineup data uses "AZ". Extend this dict if another mismatch is found.
ROTOWIRE_TEAM_ABBR_FIXUP = {"ARI": "AZ"}


def fetch_rotowire_page(date: str = "today") -> str:
    """date: "today" or "tomorrow" -- confirmed real, both work
    (2026-07-22): `?date=tomorrow` returns a genuinely different, smaller
    slate (fewer games have confirmed/expected lineups posted that far
    ahead). Any other date value isn't supported by RotoWire's own page --
    callers should not pass one."""
    params = {} if date == "today" else {"date": date}
    resp = requests.get(ROTOWIRE_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    return resp.text


def parse_rotowire_lineups(html: str) -> pd.DataFrame:
    """One row per (game, side, batting-order slot): away_team, home_team
    (both already remapped through ROTOWIRE_TEAM_ABBR_FIXUP), side
    ("away"/"home"), team, opp_team, status ("confirmed"/"expected"),
    batting_order (1-9), position, player_name (full name, e.g. "Miguel
    Vargas"), bats (L/R/S), pitcher_name, pitcher_throws (that side's
    starter, repeated on every row of that side for convenience)."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for box in soup.select("div.lineup__box"):
        abbrs = [d.get_text(strip=True) for d in box.select(".lineup__abbr")]
        if len(abbrs) != 2:
            continue
        away_team = ROTOWIRE_TEAM_ABBR_FIXUP.get(abbrs[0], abbrs[0])
        home_team = ROTOWIRE_TEAM_ABBR_FIXUP.get(abbrs[1], abbrs[1])
        for side, team, other_team, list_cls in [
            ("away", away_team, home_team, "is-visit"),
            ("home", home_team, away_team, "is-home"),
        ]:
            lineup_list = box.select_one(f"ul.lineup__list.{list_cls}")
            if lineup_list is None:
                continue
            status_el = lineup_list.select_one(".lineup__status")
            status = "confirmed" if status_el and "is-confirmed" in status_el.get("class", []) else "expected"
            pitcher_name = pitcher_throws = None
            highlight = lineup_list.select_one(".lineup__player-highlight-name")
            if highlight is not None:
                a = highlight.find("a")
                pitcher_name = a.get_text(strip=True) if a else None
                throws_el = highlight.select_one(".lineup__throws")
                pitcher_throws = throws_el.get_text(strip=True) if throws_el else None

            order = 0
            for li in lineup_list.select("li.lineup__player"):
                a = li.find("a")
                if a is None:
                    continue
                order += 1
                pos_el = li.select_one(".lineup__pos")
                bats_el = li.select_one(".lineup__bats")
                rows.append({
                    "away_team": away_team, "home_team": home_team,
                    "side": side, "team": team, "opp_team": other_team,
                    "status": status, "batting_order": order,
                    "position": pos_el.get_text(strip=True) if pos_el else None,
                    "player_name": a.get("title") or a.get_text(strip=True),
                    "bats": bats_el.get_text(strip=True) if bats_el else None,
                    "pitcher_name": pitcher_name, "pitcher_throws": pitcher_throws,
                })
    return pd.DataFrame(rows)


def _normalize_name(name: str) -> str:
    """Strip accents, punctuation, AND generational suffixes for matching --
    RotoWire and MLB Stats API don't always agree on e.g. "Jose" vs "José",
    diacritic spellings, or a trailing "Jr."/"II". The suffix strip was
    claimed by this docstring but never implemented until the 2026-08-03
    audit (finding M12): RotoWire "Luis Garcia" vs roster "Luis Garcia Jr."
    failed to match, and because rotowire_lineup_for_team requires ALL 9
    names to resolve, one suffix mismatch silently discarded the entire
    team's published RotoWire lineup down to the lower-fidelity projection
    tier (observed live for NYY on 2026-08-03). Collision check on real
    2026 roster data before shipping: zero (team, stripped-name) pairs map
    to more than one player_id."""
    if not isinstance(name, str):
        return ""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = stripped.replace(".", "").replace("'", "")
    stripped = " ".join(stripped.lower().split())
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", stripped)


def build_team_roster_name_map(lineups: pd.DataFrame, schedule: pd.DataFrame) -> dict[tuple[str, str], int]:
    """{(team, normalized_name): player_id} from this project's OWN cached
    lineup history (lineups_*.parquet joined to schedule for team codes) --
    the roster RotoWire's confirmed/expected names get matched against.
    Only a player who has appeared in a REAL lineup for that team already
    this season is matchable; a brand-new trade/debut is a rare, honest
    miss (falls back to the existing historical-projection tier), not a
    guessed match."""
    home = schedule[["game_pk", "home_team"]].rename(columns={"home_team": "team"})
    home["team_side"] = "home"
    away = schedule[["game_pk", "away_team"]].rename(columns={"away_team": "team"})
    away["team_side"] = "away"
    game_teams = pd.concat([home, away], ignore_index=True)

    merged = lineups.merge(game_teams, on=["game_pk", "team_side"], how="inner")
    merged["norm_name"] = merged["player_name"].apply(_normalize_name)
    merged = merged.sort_values("date").drop_duplicates(["team", "norm_name"], keep="last")
    return {(r.team, r.norm_name): int(r.player_id) for r in merged.itertuples()}


def match_rotowire_names_to_player_ids(rotowire_df: pd.DataFrame, roster_name_map: dict) -> pd.DataFrame:
    """Adds a `player_id` column (nullable Int64 -- NaN where no match was
    found, e.g. a brand-new trade/debut not yet in this project's own
    cached lineup history) to a parsed RotoWire lineup frame."""
    rotowire_df = rotowire_df.copy()
    rotowire_df["norm_name"] = rotowire_df["player_name"].apply(_normalize_name)
    rotowire_df["player_id"] = rotowire_df.apply(
        lambda r: roster_name_map.get((r["team"], r["norm_name"])), axis=1
    ).astype("Int64")
    return rotowire_df


def rotowire_lineup_for_team(rotowire_df: pd.DataFrame, team: str, probable_pitcher_name: str | None
                              ) -> tuple[list[int] | None, str | None]:
    """Finds the RotoWire lineup box for `team`. For the ORDINARY case (one
    game today), team alone is unambiguous -- used directly, even if
    RotoWire's own listed starter differs from this project's cached
    `probable_pitcher_name` (confirmed real, 2026-07-22: RotoWire had Cal
    Quantrill/Spencer Miles as TEX/TOR's starters where this project's own
    schedule snapshot still said Tyler Alexander/Braydon Fisher -- a real
    pitching change RotoWire's more-recent fetch caught; the BATTING
    lineup is still the correct one to use regardless of which pitcher
    ends up starting). Only for a DOUBLEHEADER (2+ distinct starting
    pitchers listed for the same team today -- confirmed real the same day
    for PIT@NYY and BAL@BOS) is `probable_pitcher_name` required to
    disambiguate which of the two games' lineups this is. Returns
    (player_ids_in_batting_order, status) or (None, None) if no confident
    match -- a genuine miss falls back to this project's own historical-
    projection tier, never a guess."""
    candidates = rotowire_df[rotowire_df["team"] == team]
    if candidates.empty:
        return None, None
    distinct_pitchers = [p for p in candidates["pitcher_name"].dropna().unique()]
    if len(distinct_pitchers) <= 1:
        grp = candidates
    else:
        if pd.isna(probable_pitcher_name):
            return None, None
        target = _normalize_name(probable_pitcher_name)
        matched_names = [p for p in distinct_pitchers
                         if (n := _normalize_name(p)) and (n == target or n in target or target in n)]
        if len(matched_names) != 1:
            return None, None  # zero or ambiguous (>1) matches -- don't guess
        grp = candidates[candidates["pitcher_name"] == matched_names[0]]
    grp = grp.sort_values("batting_order")
    if len(grp) != 9 or grp["player_id"].isna().any():
        return None, None  # incomplete lineup or an unresolved name -- don't guess
    return grp["player_id"].astype(int).tolist(), grp["status"].iloc[0]


_PYBASEBALL_PITCHER_ID_CACHE: dict[str, int | None] = {}


def resolve_rotowire_pitcher_id(pitcher_name: str | None) -> int | None:
    """Resolves a RotoWire-listed starter's name to an MLBAM player_id via
    pybaseball's Chadwick register (a comprehensive name<->id crosswalk
    across MLB's full player universe, not scoped to this project's own
    cached history) -- a pitcher, unlike a batter, doesn't need to have
    appeared in this exact team's own recent LINEUP history to be matchable
    this way, which is why this is a separate resolution path from
    match_rotowire_names_to_player_ids above rather than a reuse of the same
    roster-scoped map. Memoized per-run (module-level cache) since the same
    name may be looked up more than once in one daily run. Best-effort:
    returns None on any lookup failure (network hiccup, zero/ambiguous
    match) rather than raising -- callers already treat a None probable
    pitcher as "skip this game," the existing, safe fallback behavior."""
    if not pitcher_name or pd.isna(pitcher_name):
        return None
    if pitcher_name in _PYBASEBALL_PITCHER_ID_CACHE:
        return _PYBASEBALL_PITCHER_ID_CACHE[pitcher_name]
    parts = _normalize_name(pitcher_name).split()
    result = None
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        try:
            import pybaseball as pb
            match = pb.playerid_lookup(last, first)
            if len(match) == 1:
                result = int(match.iloc[0]["key_mlbam"])
            elif len(match) > 1:
                # ambiguous (two real MLB players sharing this name) -- prefer
                # whoever was active most recently, matching this project's
                # own "don't guess" discipline elsewhere (see e.g.
                # rotowire_lineup_for_team's own doubleheader disambiguation).
                match = match.sort_values("mlb_played_last", ascending=False)
                result = int(match.iloc[0]["key_mlbam"])
        except Exception:
            result = None
    _PYBASEBALL_PITCHER_ID_CACHE[pitcher_name] = result
    return result


def rotowire_pitcher_for_team(rotowire_df: pd.DataFrame, team: str) -> tuple[int | None, str | None, str | None]:
    """This team's RotoWire-listed starting pitcher (confirmed or expected),
    resolved to an MLBAM player_id -- the pitcher-side analog of
    rotowire_lineup_for_team, used specifically as a fallback when MLB's own
    probable_pitcher_id is still unannounced (real lineups/starters post
    2-4 hours before each game's OWN first pitch, not on a fixed clock, so a
    single once-a-day check can miss a starter that's only confirmed a few
    hours later -- see the intraday light-refresh cadence this exists for).
    Only meaningful when a team has exactly ONE distinct starter listed
    today; a doubleheader has 2 and no probable-pitcher name to disambiguate
    by (unlike rotowire_lineup_for_team's batting-lineup case, which always
    has MLB's own probable_pitcher_name to fall back on) -- returns
    (None, None, None) rather than guessing which start this is. Returns
    (player_id, pitcher_name, status) or (None, None, None) if no confident
    resolution."""
    candidates = rotowire_df[rotowire_df["team"] == team]
    if candidates.empty:
        return None, None, None
    distinct = candidates[["pitcher_name"]].dropna().drop_duplicates()
    if len(distinct) != 1:
        return None, None, None  # 0 (no starter listed yet) or 2+ (doubleheader, ambiguous) -- don't guess
    pitcher_name = distinct.iloc[0]["pitcher_name"]
    player_id = resolve_rotowire_pitcher_id(pitcher_name)
    if player_id is None:
        return None, None, None
    status = candidates["status"].iloc[0]
    return player_id, pitcher_name, status


def build_todays_rotowire_lineups(lineups: pd.DataFrame, schedule: pd.DataFrame, date: str = "today") -> pd.DataFrame:
    """One-call convenience: fetch + parse + name-match a RotoWire lineups
    page (date="today" or "tomorrow") against this project's own roster
    history."""
    html = fetch_rotowire_page(date)
    parsed = parse_rotowire_lineups(html)
    roster_map = build_team_roster_name_map(lineups, schedule)
    return match_rotowire_names_to_player_ids(parsed, roster_map)


if __name__ == "__main__":
    from src.utils.paths import DATA_RAW

    html = fetch_rotowire_page()
    parsed = parse_rotowire_lineups(html)
    print(f"parsed {parsed['side'].map(str).nunique() and len(parsed.groupby(['away_team','home_team']))} games, "
          f"{len(parsed)} batter rows")
    print(parsed[parsed["status"] == "expected"]["status"].count(), "expected,",
          parsed[parsed["status"] == "confirmed"]["status"].count(), "confirmed rows")

    schedule = pd.read_parquet(DATA_RAW / "schedule_2026.parquet")
    lineups = pd.read_parquet(DATA_RAW / "lineups_2026.parquet")
    matched = build_todays_rotowire_lineups(lineups, schedule)
    n_matched = matched["player_id"].notna().sum()
    print(f"matched {n_matched}/{len(matched)} players to MLB player_id via own roster history")
    print("\nunmatched names (first 20):")
    print(matched.loc[matched["player_id"].isna(), ["team", "player_name"]].drop_duplicates().head(20))
