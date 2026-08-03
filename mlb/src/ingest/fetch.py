"""Pull raw MLB data and cache it locally as parquet.

Design difference from a simple "refetch everything" pipeline: Statcast's
pitch-level history is far too large to wholesale-refetch every day (unlike a
small league's play-by-play). Instead, each season is either COMPLETE (a
season strictly before the current one -- fetched once, cached permanently)
or the CURRENT, in-progress season (fetched INCREMENTALLY: only the date
range since the last successful pull, appended to that season's cache).
This is what actually makes a daily-hosted, auto-updating pipeline practical.

Also carries forward a real lesson from the sibling NFL project: a
multi-request fetch can silently return fewer rows than it should (a
transient hiccup, not a hard error) with nothing raised. Every fetch here
logs row counts and date coverage so a silent gap is at least visible, even
though pybaseball's per-day chunking makes a *total* silent season-drop (like
the NFL one) much less likely than a single giant multi-season call would be.
"""

import datetime as _dt
import time
from pathlib import Path

import pandas as pd
import pybaseball as pb
import requests

from src.utils.paths import DATA_RAW

pb.cache.enable()

MAX_FETCH_RETRIES = 2
SEASON_START_MMDD = "02-01"  # covers spring training through the full regular season
SEASON_END_MMDD = "11-30"  # covers the World Series with margin
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def current_mlb_season() -> int:
    """MLB seasons run within a single calendar year (unlike NFL's Sep-Feb
    span), so "current season" is just the current year."""
    return _dt.date.today().year


def _statcast_with_retry(start_dt: str, end_dt: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES + 1):
        try:
            df = pb.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
            print(f"  statcast {start_dt}..{end_dt}: {len(df)} rows"
                  + (f", game_date range {df['game_date'].min()}..{df['game_date'].max()}" if len(df) else ""),
                  flush=True)
            return df
        except Exception as e:
            last_err = e
            if attempt < MAX_FETCH_RETRIES:
                print(f"  statcast fetch failed (attempt {attempt + 1}/{MAX_FETCH_RETRIES + 1}): {e}, retrying...", flush=True)
                time.sleep(3)
    raise RuntimeError(f"statcast fetch failed after {MAX_FETCH_RETRIES + 1} attempts: {last_err}")


def fetch_statcast_season(season: int, force: bool = False) -> Path:
    """Complete (past) seasons: fetch once, cache permanently. The current,
    in-progress season: fetch incrementally from the last cached date."""
    path = DATA_RAW / f"statcast_{season}.parquet"
    today = _dt.date.today()
    is_current = season == current_mlb_season()

    if path.exists() and not force and not is_current:
        return path

    season_start = _dt.date.fromisoformat(f"{season}-{SEASON_START_MMDD}")
    season_end = _dt.date.fromisoformat(f"{season}-{SEASON_END_MMDD}")
    end_bound = min(season_end, today)

    if not is_current or not path.exists() or force:
        # full (re)fetch of this season's whole window
        print(f"fetching full season {season} ({season_start}..{end_bound})...", flush=True)
        df = _statcast_with_retry(str(season_start), str(end_bound))
        df.to_parquet(path, index=False)
        print(f"  saved {path.name}: {len(df)} rows", flush=True)
        return path

    # incremental: current season, already have a cache -- fetch only new dates
    existing = pd.read_parquet(path)
    if existing.empty:
        last_cached = season_start - _dt.timedelta(days=1)
    else:
        last_cached = pd.to_datetime(existing["game_date"]).max().date()

    # Completeness guard (2026-08-03 audit, finding C3): the cursor used to
    # assume max(cached game_date) was a COMPLETE date -- but the evening
    # full-run cron (~9pm ET) fetches mid-slate, and on a mixed day/night
    # slate Savant returns the finished afternoon games' rows while the
    # night games are still playing. The cursor then advanced past the date
    # and the night games were permanently absent (the backfill couldn't
    # rescue them either -- it only looked at "Final" schedule rows, which
    # finding C2's frozen statuses never became). Fix: never advance past
    # the latest date for which the schedule says every regular-season game
    # is in a terminal status; re-fetching a partially-cached date overlaps
    # existing rows, which the keep-last dedup below already handles.
    # Requires the schedule to be refreshed first -- refresh_all_data() runs
    # fetch_schedule_season (with its own C2 stale-date healing) before this.
    fetch_start = last_cached + _dt.timedelta(days=1)
    sched_path = DATA_RAW / f"schedule_{season}.parquet"
    if sched_path.exists():
        sched = pd.read_parquet(sched_path)
        reg = sched[sched["game_type"] == "R"]
        if not reg.empty:
            terminal = {"Final", "Completed Early", "Postponed", "Cancelled"}
            incomplete = reg[~reg["status"].isin(terminal)]
            if not incomplete.empty:
                first_incomplete = pd.to_datetime(incomplete["date"]).min().date()
                if first_incomplete <= last_cached:
                    fetch_start = first_incomplete
                    print(f"  completeness guard: re-fetching from {fetch_start} "
                          f"(cached through {last_cached}, but that range has non-terminal games)", flush=True)

    if fetch_start > end_bound:
        print(f"season {season}: cache already current through {last_cached}, nothing new to fetch", flush=True)
        return path

    print(f"incremental fetch for season {season}: {fetch_start}..{end_bound} (cache was current through {last_cached})...", flush=True)
    new_df = _statcast_with_retry(str(fetch_start), str(end_bound))
    if new_df.empty:
        print(f"  no new rows (no games played {fetch_start}..{end_bound}, e.g. All-Star break or off day)", flush=True)
        return path

    combined = pd.concat([existing, new_df], ignore_index=True)
    # de-dupe defensively in case a date range overlap ever occurs
    dedupe_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in combined.columns]
    if dedupe_cols:
        before = len(combined)
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
        if len(combined) != before:
            print(f"  de-duped {before - len(combined)} overlapping rows", flush=True)
    combined.to_parquet(path, index=False)
    print(f"  appended {len(new_df)} new rows -> {len(combined)} total, saved {path.name}", flush=True)
    return path


def fetch_statcast_seasons(seasons: list[int], force: bool = False) -> dict[int, Path]:
    return {s: fetch_statcast_season(s, force=force) for s in seasons}


def verify_and_backfill_statcast(season: int) -> int:
    """Cross-check Statcast's game_pk coverage against the schedule (a
    DIFFERENT source) for the same season, and backfill any gap found.

    This closes a real gap: fetch_statcast_season only retries on a raised
    EXCEPTION, but pybaseball's per-day chunking can silently return fewer
    rows than requested with nothing raised (confirmed directly this
    project -- an 11-season pull once silently dropped two whole seasons).
    A single-date-range request has the same failure mode at smaller scale.
    Manually cross-checking Statcast's game_pks against the schedule's
    "Final" regular-season game_pks is exactly the check that caught the
    original bug -- this makes that check automatic instead of relying on
    remembering to run it by hand. Returns the number of games backfilled.
    """
    statcast_path = DATA_RAW / f"statcast_{season}.parquet"
    schedule_path = DATA_RAW / f"schedule_{season}.parquet"
    if not statcast_path.exists() or not schedule_path.exists():
        return 0

    statcast = pd.read_parquet(statcast_path, columns=["game_pk", "game_date"])
    schedule = pd.read_parquet(schedule_path)
    # "Completed Early" widened in (2026-08-03 audit, finding C3): rain-
    # shortened games ARE official and DO have Statcast data, but were
    # invisible to this safety net (and to the umpire/SB backfills, fixed
    # separately). Postponed/Cancelled correctly stay excluded (no pitches).
    expected = schedule[(schedule["game_type"] == "R") & (schedule["status"].isin(["Final", "Completed Early"]))]
    missing = expected[~expected["game_pk"].isin(set(statcast["game_pk"]))]

    if missing.empty:
        return 0

    missing_dates = sorted(pd.to_datetime(missing["date"]).dt.date.unique())
    print(f"  verify_and_backfill_statcast({season}): {len(missing)} games missing from Statcast "
          f"on {len(missing_dates)} date(s), backfilling...", flush=True)

    frames = [pd.read_parquet(statcast_path)]
    # pybaseball caches each per-day Savant CSV response for 7 days keyed on
    # the date-URL (2026-08-03 audit, finding C3/M22): a partial same-day
    # response would otherwise be silently re-served to every backfill
    # attempt for a week, making the "backfill" a no-op exactly when it
    # matters. Disable the cache for these targeted re-fetches only.
    pb.cache.disable()
    try:
        for d in missing_dates:
            backfill = _statcast_with_retry(str(d), str(d))
            if not backfill.empty:
                frames.append(backfill)
    finally:
        pb.cache.enable()
    combined = pd.concat(frames, ignore_index=True)
    dedupe_cols = [c for c in ("game_pk", "at_bat_number", "pitch_number") if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    combined.to_parquet(statcast_path, index=False)

    still_missing = expected[~expected["game_pk"].isin(set(combined["game_pk"]))]
    if len(still_missing):
        print(f"  WARNING: {len(still_missing)} games still missing from Statcast after backfill attempt "
              f"for season {season} -- may be games with no trackable pitch data, or a persistent source issue", flush=True)
    else:
        print(f"  backfill complete: all {len(expected)} expected games now present for season {season}", flush=True)
    return len(missing) - len(still_missing)


# --- MLB Stats API: schedule, results, weather, probable pitchers, lineups ---
# Free, unauthenticated, unofficial (undocumented) API that actually powers
# MLB.com. Confirmed directly: historical weather (condition/temp/wind,
# already expressed relative to the park -- e.g. "Out To CF" -- so we don't
# need to derive each park's compass orientation ourselves) and the actual
# lineups used both persist for past games, not just live/upcoming ones.

def _schedule_day(date: str) -> list[dict]:
    r = requests.get(f"{MLB_API_BASE}/schedule", params={
        "sportId": 1, "date": date, "hydrate": "probablePitcher,lineups,team,venue,weather",
    }, timeout=30)
    r.raise_for_status()
    dates = r.json().get("dates", [])
    return dates[0]["games"] if dates else []


def _schedule_day_with_retry(date: str) -> list[dict]:
    last_err = None
    for attempt in range(MAX_FETCH_RETRIES + 1):
        try:
            return _schedule_day(date)
        except Exception as e:
            last_err = e
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(2)
    raise RuntimeError(f"schedule fetch failed for {date} after {MAX_FETCH_RETRIES + 1} attempts: {last_err}")


def _parse_games(games: list[dict], date: str) -> tuple[list[dict], list[dict]]:
    game_rows, lineup_rows = [], []
    for g in games:
        home, away = g["teams"]["home"], g["teams"]["away"]
        weather = g.get("weather", {}) or {}
        venue = g.get("venue", {}) or {}
        game_rows.append({
            "game_pk": g["gamePk"], "date": date, "season": int(g.get("season", date[:4])),
            "game_type": g.get("gameType"), "status": g.get("status", {}).get("detailedState"),
            "home_team_id": home["team"]["id"], "home_team": home["team"].get("abbreviation"),
            "away_team_id": away["team"]["id"], "away_team": away["team"].get("abbreviation"),
            "home_score": home.get("score"), "away_score": away.get("score"),
            "venue_id": venue.get("id"), "venue_name": venue.get("name"),
            "weather_condition": weather.get("condition"), "weather_temp": weather.get("temp"),
            "weather_wind": weather.get("wind"),
            "home_probable_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
            "home_probable_pitcher_name": (home.get("probablePitcher") or {}).get("fullName"),
            "away_probable_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
            "away_probable_pitcher_name": (away.get("probablePitcher") or {}).get("fullName"),
        })
        lineups = g.get("lineups", {}) or {}
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for order, p in enumerate(lineups.get(key, []) or [], start=1):
                lineup_rows.append({
                    "game_pk": g["gamePk"], "date": date, "team_side": side,
                    "batting_order": order, "player_id": p["id"], "player_name": p.get("fullName"),
                    "position_code": (p.get("primaryPosition") or {}).get("code"),
                    "position_name": (p.get("primaryPosition") or {}).get("abbreviation"),
                })
    return game_rows, lineup_rows


def fetch_schedule_season(season: int, force: bool = False) -> tuple[Path, Path]:
    """Same complete-vs-incremental design as fetch_statcast_season, but hits
    the MLB Stats API's schedule endpoint (games + weather + probable
    pitchers + lineups) rather than Statcast."""
    sched_path = DATA_RAW / f"schedule_{season}.parquet"
    lineup_path = DATA_RAW / f"lineups_{season}.parquet"
    today = _dt.date.today()
    is_current = season == current_mlb_season()

    if sched_path.exists() and lineup_path.exists() and not force and not is_current:
        return sched_path, lineup_path

    season_start = _dt.date.fromisoformat(f"{season}-{SEASON_START_MMDD}")
    season_end = _dt.date.fromisoformat(f"{season}-{SEASON_END_MMDD}")
    end_bound = min(season_end, today)

    stale_dates: list[_dt.date] = []
    if not is_current or not sched_path.exists() or force:
        fetch_from = season_start
        existing_games, existing_lineups = pd.DataFrame(), pd.DataFrame()
    else:
        existing_games = pd.read_parquet(sched_path)
        existing_lineups = pd.read_parquet(lineup_path) if lineup_path.exists() else pd.DataFrame()
        last_cached = (
            pd.to_datetime(existing_games["date"]).max().date() if not existing_games.empty else season_start - _dt.timedelta(days=1)
        )
        fetch_from = last_cached + _dt.timedelta(days=1)

        # Stale-past-date healing (2026-08-03 audit, finding C2): the
        # incremental cursor above starts at max(cached date)+1, and
        # fetch_schedule_day(tomorrow) -- called by every evening full run --
        # pushes that max to TOMORROW, so yesterday's/today's rows were
        # structurally never revisited after the last same-day light run
        # (16:00 ET, before most games end). Evening games froze forever at
        # Pre-Game/In Progress with NULL weather and no lineups (76 real
        # rows found), silently degrading lineup-projection history, weather
        # climatology, and every schedule-status-gated backfill (Statcast/
        # umpire/stolen-base -- see verify_and_backfill_statcast). Fix:
        # explicitly re-fetch any PAST date whose cached rows contain a
        # non-terminal status. "Suspended" is deliberately non-terminal (the
        # game resumes later); a rescheduled game heals via the game_pk
        # keep-last dedup below (same pk re-fetched under its new date wins).
        if not existing_games.empty:
            terminal = {"Final", "Completed Early", "Postponed", "Cancelled"}
            past = existing_games[
                (pd.to_datetime(existing_games["date"]).dt.date < today)
                & (~existing_games["status"].isin(terminal))
            ]
            stale_dates = sorted(pd.to_datetime(past["date"]).dt.date.unique())
            if stale_dates:
                print(f"schedule {season}: re-fetching {len(stale_dates)} past date(s) with "
                      f"non-terminal cached rows: {stale_dates[0]}..{stale_dates[-1]}", flush=True)

        if fetch_from > end_bound and not stale_dates:
            print(f"schedule {season}: cache already current through {last_cached}, nothing new to fetch", flush=True)
            return sched_path, lineup_path

    dates_to_fetch = list(stale_dates)
    d = fetch_from
    while d <= end_bound:
        if d not in stale_dates:
            dates_to_fetch.append(d)
        d += _dt.timedelta(days=1)
    dates_to_fetch.sort()

    print(f"fetching schedule/lineups for season {season}: {len(dates_to_fetch)} date(s)...", flush=True)
    all_game_rows, all_lineup_rows = [], []
    for d in dates_to_fetch:
        games = _schedule_day_with_retry(str(d))
        g_rows, l_rows = _parse_games(games, str(d))
        all_game_rows.extend(g_rows)
        all_lineup_rows.extend(l_rows)
    print(f"  fetched {len(all_game_rows)} games, {len(all_lineup_rows)} lineup slots", flush=True)

    new_games = pd.DataFrame(all_game_rows)
    new_lineups = pd.DataFrame(all_lineup_rows)
    combined_games = pd.concat([existing_games, new_games], ignore_index=True) if len(new_games) else existing_games
    combined_lineups = pd.concat([existing_lineups, new_lineups], ignore_index=True) if len(new_lineups) else existing_lineups
    if len(combined_games) and "game_pk" in combined_games.columns:
        combined_games = combined_games.drop_duplicates(subset=["game_pk"], keep="last")
    if len(combined_lineups) and "game_pk" in combined_lineups.columns:
        combined_lineups = combined_lineups.drop_duplicates(subset=["game_pk", "team_side", "batting_order"], keep="last")

    combined_games.to_parquet(sched_path, index=False)
    combined_lineups.to_parquet(lineup_path, index=False)
    print(f"  saved {sched_path.name} ({len(combined_games)} games), {lineup_path.name} ({len(combined_lineups)} lineup slots)", flush=True)
    return sched_path, lineup_path


def fetch_schedule_seasons(seasons: list[int], force: bool = False) -> dict[int, tuple[Path, Path]]:
    return {s: fetch_schedule_season(s, force=force) for s in seasons}


def fetch_schedule_day(date: str) -> tuple[Path, Path]:
    """Fetch and merge in a SINGLE arbitrary date's games/lineups -- including
    a FUTURE date, unlike fetch_schedule_season, which structurally caps its
    incremental fetch at `min(season_end, today)` and so can never reach
    tomorrow (that cap makes sense for Statcast, which literally can't exist
    for a future game, but the schedule endpoint's probable-pitcher data DOES
    exist ahead of time and fetch_schedule_season was never built to ask for
    it). Confirmed directly against the live API: a future date's games come
    back with real probable pitchers populated, but empty lineups and
    weather (both are only posted a few hours before first pitch, and stay
    empty even fetched the night before) -- see lineup_projection.py for the
    fallback this makes necessary."""
    season = int(date[:4])
    sched_path = DATA_RAW / f"schedule_{season}.parquet"
    lineup_path = DATA_RAW / f"lineups_{season}.parquet"
    existing_games = pd.read_parquet(sched_path) if sched_path.exists() else pd.DataFrame()
    existing_lineups = pd.read_parquet(lineup_path) if lineup_path.exists() else pd.DataFrame()

    games = _schedule_day_with_retry(date)
    g_rows, l_rows = _parse_games(games, date)
    print(f"  fetched {len(g_rows)} games, {len(l_rows)} lineup slots for {date}", flush=True)

    new_games, new_lineups = pd.DataFrame(g_rows), pd.DataFrame(l_rows)
    combined_games = pd.concat([existing_games, new_games], ignore_index=True) if len(new_games) else existing_games
    combined_lineups = pd.concat([existing_lineups, new_lineups], ignore_index=True) if len(new_lineups) else existing_lineups
    if len(combined_games) and "game_pk" in combined_games.columns:
        combined_games = combined_games.drop_duplicates(subset=["game_pk"], keep="last")
    if len(combined_lineups) and "game_pk" in combined_lineups.columns:
        combined_lineups = combined_lineups.drop_duplicates(subset=["game_pk", "team_side", "batting_order"], keep="last")

    combined_games.to_parquet(sched_path, index=False)
    combined_lineups.to_parquet(lineup_path, index=False)
    return sched_path, lineup_path


def fetch_transactions(start_date: str, end_date: str) -> pd.DataFrame:
    """Real trade transactions from the MLB Stats API's free `/transactions`
    endpoint (trade-deadline hardening) -- feeds bullpen.build_traded_pitcher_
    overrides, which seeds a just-traded reliever's usage weight/closer-
    candidacy from his PRIOR team's role instead of starting him cold on his
    new team (see bullpen.py's module docstring for why this matters: a
    reliever's team affiliation, unlike a starter's or batter's, has NO live
    signal anywhere else in this project -- it's inferred entirely from
    historical PA rows, which lag real trades by however long ingestion + the
    player's first appearance for his new team takes).

    Returns one row per (transaction, player) -- typeCode=="TR" (trade) only,
    "person" field required (the API also emits TEAM-level duplicate records
    for the same trade with NO player attached, e.g. a bare fromTeam/toTeam
    pair -- filtered out here). Confirmed directly against the live API:
    multi-player trades reuse the SAME transaction id across each distinct
    player (e.g. one 3-for-2 trade produced 3 rows all sharing one id) -- so
    dedup keys on (transaction_id, pitcher_id), NOT transaction_id alone
    (an earlier draft of this function deduped by id only and would have
    silently dropped 2 of 3 players in that exact real trade)."""
    r = requests.get(f"{MLB_API_BASE}/transactions",
                      params={"startDate": start_date, "endDate": end_date}, timeout=30)
    r.raise_for_status()
    txns = r.json().get("transactions", [])
    rows = []
    seen = set()
    for t in txns:
        if t.get("typeCode") != "TR" or "person" not in t:
            continue
        from_team, to_team = t.get("fromTeam"), t.get("toTeam")
        if not from_team or not to_team:
            continue
        key = (t["id"], t["person"]["id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "transaction_id": t["id"], "pitcher_id": t["person"]["id"], "pitcher_name": t["person"]["fullName"],
            "from_team_id": from_team["id"], "to_team_id": to_team["id"],
            "date": t.get("effectiveDate", t.get("date")),
        })
    return pd.DataFrame(rows, columns=[
        "transaction_id", "pitcher_id", "pitcher_name", "from_team_id", "to_team_id", "date",
    ])


SAVANT_SPRINT_SPEED_URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"


def fetch_sprint_speed_season(season: int, force: bool = False) -> Path:
    """Baseball Savant's Sprint Speed leaderboard, season-level (one row per
    qualifying player). Unlike the park-factors leaderboard tried earlier in
    this project (its csv=true parameter returns the JS-rendered HTML page,
    not real data), this leaderboard's CSV export genuinely works -- confirmed
    directly via curl before building this. A cheap, real, season-aggregate
    speed measurement, not derivable from our own per-pitch Statcast cache."""
    out_path = DATA_RAW / f"sprint_speed_{season}.parquet"
    if out_path.exists() and not force:
        return out_path
    r = requests.get(SAVANT_SPRINT_SPEED_URL, params={"year": season, "position": "", "team": "", "min": 10, "csv": "true"},
                      timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.to_parquet(out_path, index=False)
    print(f"sprint speed {season}: {len(df)} qualifying players", flush=True)
    return out_path


def fetch_sprint_speed_seasons(seasons: list[int], force: bool = False) -> dict[int, Path]:
    return {s: fetch_sprint_speed_season(s, force=force) for s in seasons}


SAVANT_OAA_URL = "https://baseballsavant.mlb.com/leaderboard/outs_above_average"


def fetch_oaa_season(season: int, force: bool = False) -> Path:
    """Baseball Savant's Outs Above Average leaderboard, season-level (one row
    per qualifying fielder, tagged with their primary position) -- same proven
    CSV-export pattern as fetch_sprint_speed_season. Real, Statcast-derived
    defensive-skill measurement (already controls for play difficulty via a
    catch-probability model), not derivable from our own PA-level cache."""
    out_path = DATA_RAW / f"oaa_{season}.parquet"
    if out_path.exists() and not force:
        return out_path
    r = requests.get(SAVANT_OAA_URL, params={"year": season, "type": "Fielder", "csv": "true"}, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.to_parquet(out_path, index=False)
    print(f"OAA {season}: {len(df)} qualifying fielders", flush=True)
    return out_path


def fetch_oaa_seasons(seasons: list[int], force: bool = False) -> dict[int, Path]:
    return {s: fetch_oaa_season(s, force=force) for s in seasons}


MLB_API_BASE_V11 = "https://statsapi.mlb.com/api/v1.1"


def fetch_home_plate_umpire(game_pk: int) -> dict | None:
    """One game's home-plate umpire (id + name) from the live-feed endpoint's
    boxscore officials -- the same trusted, already-used MLB Stats API this
    project's schedule/lineup fetches use, no new provider. Works for a
    completed historical game AND a genuinely future game once MLB posts its
    umpire crew (confirmed same-day, once the game's status reaches
    "Pre-Game" -- typically a few hours before first pitch, the same lead
    time real lineups get; see umpire_factor.resolve_live_umpire_factor).
    Returns None on a genuine miss (crew not posted yet, e.g. querying more
    than a day ahead)."""
    try:
        r = requests.get(f"{MLB_API_BASE_V11}/game/{game_pk}/feed/live", timeout=10)
        r.raise_for_status()
        officials = r.json()["liveData"]["boxscore"]["officials"]
        hp = next((o["official"] for o in officials if o["officialType"] == "Home Plate"), None)
        if hp is None:
            return None
        return {"game_pk": game_pk, "ump_id": hp["id"], "ump_name": hp["fullName"]}
    except (requests.RequestException, KeyError, StopIteration):
        return None


def fetch_umpires_for_seasons(seasons: list[int], checkpoint_every: int = 100) -> Path:
    """Backfills home-plate umpire assignments for every real, completed
    regular-season game across `seasons`, resumable (re-running only fetches
    game_pks not already cached) -- the full-history version of the 400-game
    sample used earlier this session to validate the data source works.
    Saved to a single combined data/raw/umpires.parquet (not per-season,
    since the whole point is one persistent, growing cache)."""
    out_path = DATA_RAW / "umpires.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["game_pk", "ump_id", "ump_name"])
    already_have = set(existing["game_pk"])

    frames = []
    for season in seasons:
        sched_path = DATA_RAW / f"schedule_{season}.parquet"
        if not sched_path.exists():
            continue
        sched = pd.read_parquet(sched_path)
        reg = sched[(sched["game_type"] == "R") & (sched["status"] == "Final")]
        frames.append(reg[["game_pk"]])
    all_games = pd.concat(frames, ignore_index=True)["game_pk"].unique() if frames else []
    todo = [g for g in all_games if g not in already_have]
    print(f"umpire backfill: {len(already_have)} already cached, {len(todo)} to fetch", flush=True)

    new_rows = []
    for i, game_pk in enumerate(todo):
        row = fetch_home_plate_umpire(int(game_pk))
        if row is not None:
            new_rows.append(row)
        if (i + 1) % checkpoint_every == 0:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"  {i + 1}/{len(todo)} fetched, {len(new_rows)} with umpire data this run "
                  f"(checkpoint saved, {len(combined)} total)", flush=True)

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    combined.to_parquet(out_path, index=False)
    print(f"umpire backfill done: {len(combined)} total games with umpire data", flush=True)
    return out_path


def _fetch_game_stolen_base_stats(game_pk: int) -> list[dict]:
    """One game's real per-player stolenBases/caughtStealing counts, from the
    SAME live-feed boxscore endpoint already used for umpires (MLB's own
    official per-player batting box score, not something derivable from our
    own Statcast cache -- stolen bases show up there only as incidental free
    text in a pitch's `des` field, not as a structured `events` value, and
    don't reliably tie to any specific pitch). Returns one row per player
    with any plate appearance, [] on a genuine fetch miss."""
    try:
        r = requests.get(f"{MLB_API_BASE_V11}/game/{game_pk}/feed/live", timeout=10)
        r.raise_for_status()
        box = r.json()["liveData"]["boxscore"]
        rows = []
        for side in ("home", "away"):
            for pid, pdata in box["teams"][side]["players"].items():
                batting = pdata.get("stats", {}).get("batting", {})
                if not batting or batting.get("plateAppearances", 0) == 0:
                    continue
                rows.append({
                    "game_pk": game_pk, "player_id": pdata["person"]["id"],
                    "stolen_bases": batting.get("stolenBases", 0),
                    "caught_stealing": batting.get("caughtStealing", 0),
                })
        return rows
    except (requests.RequestException, KeyError):
        return []


def fetch_stolen_base_stats_for_seasons(seasons: list[int], checkpoint_every: int = 100) -> Path:
    """Backfills real per-player-per-game stolen-base/caught-stealing counts
    for every real, completed regular-season game across `seasons` --
    resumable, same checkpointed pattern as fetch_umpires_for_seasons. Saved
    to a single combined data/raw/stolen_bases.parquet."""
    out_path = DATA_RAW / "stolen_bases.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(
        columns=["game_pk", "player_id", "stolen_bases", "caught_stealing"]
    )
    already_have = set(existing["game_pk"])

    frames = []
    for season in seasons:
        sched_path = DATA_RAW / f"schedule_{season}.parquet"
        if not sched_path.exists():
            continue
        sched = pd.read_parquet(sched_path)
        reg = sched[(sched["game_type"] == "R") & (sched["status"] == "Final")]
        frames.append(reg[["game_pk"]])
    all_games = pd.concat(frames, ignore_index=True)["game_pk"].unique() if frames else []
    todo = [g for g in all_games if g not in already_have]
    print(f"stolen-base backfill: {len(already_have)} games already cached, {len(todo)} to fetch", flush=True)

    new_rows = []
    for i, game_pk in enumerate(todo):
        new_rows.extend(_fetch_game_stolen_base_stats(int(game_pk)))
        if (i + 1) % checkpoint_every == 0:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"  {i + 1}/{len(todo)} games fetched (checkpoint saved, {len(combined)} player-game rows total)",
                  flush=True)

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_pk", "player_id"], keep="last")
    combined.to_parquet(out_path, index=False)
    print(f"stolen-base backfill done: {len(combined)} total player-game rows", flush=True)
    return out_path


if __name__ == "__main__":
    import sys

    start = int(sys.argv[1]) if len(sys.argv) > 1 else current_mlb_season()
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start
    seasons = list(range(start, end + 1))
    fetch_schedule_seasons(seasons)
    fetch_statcast_seasons(seasons)
