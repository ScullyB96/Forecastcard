"""The live entry point: generate props for every real game on a given date
(default tomorrow), using whatever's actually available from MLB right now
and falling back sensibly where it isn't.

Confirmed directly against the live API before building this: even fetched
the night before, a future game has a real probable pitcher but EMPTY
lineups and weather (both are only posted a few hours before first pitch).
Two different responses to that gap, not the same one:
  - Lineups: THREE tiers, in priority order. (1) The real posted MLB lineup,
    once available. (2) RotoWire's own editorially-curated confirmed/
    expected lineup (see fetch_rotowire_lineups.py) -- added 2026-07-22 after
    a real, confirmed miss: this project's own historical-recency projection
    included Randal Grichuk in a CWS@TEX game where RotoWire's real Expected
    Rangers lineup did not have him at all (a rest day / roster decision our
    own "most recent complete lineup" heuristic has no way to see coming).
    (3) Only when NEITHER real source has the game: projected from each
    team's most recent complete lineup, with a platoon-aware adjustment on
    top (see lineup_projection.py) -- swaps in a platoon partner when a real
    signal exists in this team's recent history (a different player who's
    clearly the preferred starter at that exact position against today's
    opposing pitcher's hand) and the baseline happens to have the wrong-side
    player in from a recent game against the other hand. Validated on real
    2025 data both for the plain baseline (76% roster overlap game-to-game)
    and the platoon swap specifically (a real Athletics DH platoon pair,
    confirmed to swap correctly).
  - Weather: a real temperature forecast (free, no-key Open-Meteo API)
    narrows this park's own climatological wind/bucket distribution for this
    calendar month, sampled fresh per Monte Carlo trial -- see
    weather_forecast.py for why wind DIRECTION specifically isn't
    forecast-driven (would need each park's compass orientation, which
    couldn't be reliably sourced).

Uses fetch_schedule_day (not fetch_schedule_season, which structurally caps
at today and can never reach a future date) to pull the target date's real
schedule/probable-pitchers into the local cache first.

A real bug found during an audit of this pipeline (2026-07-20): this script
used to read pa_table_2023_2026.parquet directly without ever refreshing it
first -- refresh_all_data() (which pulls YESTERDAY's actual completed games)
was never called here, only fetch_schedule_day for the target (future) date.
Confirmed directly: the cached PA table was stale by 2 real days when this
was caught. Fixed by calling refresh_all_data() (raw Statcast/schedule) AND
build_and_save_pa_table() (the derived PA table, which does NOT update
itself just because the raw cache did) at the start of every run.
"""

import datetime as _dt
import sys

import numpy as np
import pandas as pd

from src.ingest.build_pa_table import build_and_save_pa_table
from src.ingest.fetch import fetch_schedule_day, fetch_transactions
from src.ingest.fetch_rotowire_lineups import build_todays_rotowire_lineups, rotowire_lineup_for_team
from src.models.bullpen import build_traded_pitcher_overrides, TRADE_OVERRIDE_LOOKBACK_DAYS
from src.models.lineup_projection import project_lineup_platoon_aware
from src.models.props import build_pregame_context, generate_game_props
from src.models.validate_game_simulator import predominant_hand
from src.pipeline.daily_update import refresh_all_data
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.tz import eastern_today, default_slate_date


POSTSEASON_GAME_TYPES = {"F", "D", "L", "W"}  # wild card, division series, league
                                                # championship, world series -- confirmed
                                                # against this project's own schedule data
                                                # (see MODEL_DOCUMENTATION.md sec 11.24).
                                                # Deliberately excludes "S"(spring training)/
                                                # "E"(exhibition)/"A"(all-star) -- real games
                                                # this project has never modeled or intends to.
GAME_TYPES_TO_PREDICT = {"R"} | POSTSEASON_GAME_TYPES


def games_for_date(target_date: str) -> pd.DataFrame:
    season = int(target_date[:4])
    sched = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
    return sched[(sched["date"] == target_date) & (sched["game_type"].isin(GAME_TYPES_TO_PREDICT))]


def lineup_for_game(schedule: pd.DataFrame, lineups: pd.DataFrame, pitcher_hand: pd.Series, game_pk: int,
                     team: str, side: str, target_date: str, opposing_pitcher_id,
                     rotowire_df: pd.DataFrame | None = None,
                     own_probable_pitcher_name: str | None = None) -> tuple[list[int] | None, str]:
    """Real posted lineup if available (9 batters for this game_pk/side) --
    highest priority, since it's simply true. Otherwise a real, editorially-
    curated RotoWire confirmed/expected lineup if `rotowire_df` is given and
    a confident match exists (see fetch_rotowire_lineups.py) -- accounts for
    rest days, roster moves, and platoon/injury news that this project's own
    historical-recency projection has no way to know about (confirmed real,
    2026-07-22: this project's own projection included Randal Grichuk in a
    CWS@TEX game where RotoWire's real Expected Rangers lineup did not have
    him at all). Falls back to this project's own historical-recency +
    platoon-aware projection only when NEITHER real source has this game.
    Returns (player_ids, source) where source is one of "real"/"rotowire"/
    "projected" (player_ids is None only if even the projection fails)."""
    real = lineups[(lineups["game_pk"] == game_pk) & (lineups["team_side"] == side)].sort_values("batting_order")
    if len(real) == 9:
        return real["player_id"].tolist(), "real"
    if rotowire_df is not None:
        ids, _status = rotowire_lineup_for_team(rotowire_df, team, own_probable_pitcher_name)
        if ids is not None:
            return ids, "rotowire"
    projected = project_lineup_platoon_aware(schedule, lineups, pitcher_hand, team, target_date, opposing_pitcher_id)
    return projected, "projected"


if __name__ == "__main__":
    target_date = sys.argv[1] if len(sys.argv) > 1 else str(default_slate_date())
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    season = int(target_date[:4])
    print(f"=== generating props for {target_date} ===", flush=True)

    print("refreshing raw Statcast/schedule data (yesterday's real games)...", flush=True)
    status = refresh_all_data()
    if status["errors"]:
        print(f"  {len(status['errors'])} error(s) during refresh: {status['errors']}", flush=True)
    print("rebuilding derived PA table from refreshed raw data...", flush=True)
    build_and_save_pa_table(status["seasons_covered"])

    print("fetching this date's schedule/probable pitchers...", flush=True)
    fetch_schedule_day(target_date)

    games = games_for_date(target_date)
    if games.empty:
        print(f"no games found for {target_date} (schedule may not be posted yet, or it's an off day).", flush=True)
        sys.exit(0)
    print(f"found {len(games)} games", flush=True)

    print("loading PA history and building pregame context (this is the slow part)...", flush=True)
    pa_path = DATA_PROCESSED / f"pa_table_{min(status['seasons_covered'])}_{max(status['seasons_covered'])}.parquet"
    pa = pd.read_parquet(pa_path)
    ctx = build_pregame_context(pa)
    pitcher_hand = predominant_hand(pa, "pitcher")

    schedule = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
    lineups = pd.read_parquet(DATA_RAW / f"lineups_{season}.parquet")

    # Trade-deadline hardening: a reliever's team affiliation (unlike a starter's
    # or batter's) has NO live signal anywhere else in this pipeline -- it's
    # inferred entirely from historical PA rows, which lag a real trade by
    # however long it takes him to actually pitch (and get ingested) for his
    # new team. Real trade data from the free MLB Stats API's /transactions
    # endpoint fixes this by seeding a just-traded reliever's usage weight/
    # closer-candidacy from his established PRIOR role (see bullpen.py's
    # module docstring). Same resilience convention as the RotoWire fetch
    # above: a network hiccup here must not break the whole daily run.
    print("fetching recent trades (trade-deadline hardening)...", flush=True)
    try:
        lookback_start = str(eastern_today() - _dt.timedelta(days=TRADE_OVERRIDE_LOOKBACK_DAYS))
        transactions = fetch_transactions(lookback_start, target_date)
        team_id_to_abbrev = pd.concat([
            schedule[["home_team_id", "home_team"]].rename(columns={"home_team_id": "id", "home_team": "abbrev"}),
            schedule[["away_team_id", "away_team"]].rename(columns={"away_team_id": "id", "away_team": "abbrev"}),
        ]).drop_duplicates(subset="id").set_index("id")["abbrev"].to_dict()
        traded_overrides = build_traded_pitcher_overrides(transactions, team_id_to_abbrev, target_date)
        print(f"  {len(traded_overrides)} pitcher(s) with an active trade override", flush=True)
    except Exception as e:
        print(f"  trade fetch failed ({e}) -- proceeding without overrides for today", flush=True)
        traded_overrides = {}
    ctx["traded_overrides"] = traded_overrides

    # RotoWire confirmed/expected lineups -- a real, editorially-curated
    # middle tier between "real posted MLB lineup" and this project's own
    # historical-recency projection (see fetch_rotowire_lineups.py). Only
    # supports "today"/"tomorrow"; any other target_date skips it entirely
    # (falls straight through to the existing projection, unchanged
    # behavior). A network hiccup or a RotoWire page-structure change must
    # NOT break the whole daily run -- caught and treated as "unavailable
    # today," same resilience convention as weather_forecast.py's
    # climatological fallback.
    today_str = str(eastern_today())
    tomorrow_str = str(eastern_today() + _dt.timedelta(days=1))
    rotowire_df = None
    if target_date in (today_str, tomorrow_str):
        rw_date = "today" if target_date == today_str else "tomorrow"
        print(f"fetching RotoWire {rw_date} lineups...", flush=True)
        try:
            rotowire_df = build_todays_rotowire_lineups(lineups, schedule, date=rw_date)
            print(f"  parsed {len(rotowire_df)} batter rows, "
                  f"{rotowire_df['player_id'].notna().sum()} matched to a known player_id", flush=True)
        except Exception as e:
            print(f"  RotoWire fetch/parse failed ({e}) -- continuing without it", flush=True)

    rng = np.random.default_rng()
    results = []
    batter_prop_rows = []
    pitcher_prop_rows = []
    for row in games.itertuples():
        home_pid, away_pid = row.home_probable_pitcher_id, row.away_probable_pitcher_id
        if pd.isna(home_pid) or pd.isna(away_pid):
            print(f"  skipping {row.away_team} @ {row.home_team}: probable pitcher not yet announced", flush=True)
            continue

        # each team's opponent for platoon purposes is the OTHER team's probable starter
        home_ids, home_source = lineup_for_game(schedule, lineups, pitcher_hand, row.game_pk,
                                                 row.home_team, "home", target_date, away_pid,
                                                 rotowire_df=rotowire_df,
                                                 own_probable_pitcher_name=row.home_probable_pitcher_name)
        away_ids, away_source = lineup_for_game(schedule, lineups, pitcher_hand, row.game_pk,
                                                 row.away_team, "away", target_date, home_pid,
                                                 rotowire_df=rotowire_df,
                                                 own_probable_pitcher_name=row.away_probable_pitcher_name)
        if home_ids is None or away_ids is None:
            print(f"  skipping {row.away_team} @ {row.home_team}: no lineup history to project from", flush=True)
            continue

        has_real_weather = pd.notna(row.weather_condition)
        flags = []

        def _side_flag(label, source):
            return f"{label} {source}" if source != "real" else None

        lineup_flags = [f for f in [_side_flag("home", home_source), _side_flag("away", away_source)] if f]
        if lineup_flags:
            flags.append(", ".join(lineup_flags))
        flags.append("real weather" if has_real_weather else "forecast/climatological weather")
        flag_str = f"  [{', '.join(flags)}]"

        print(f"  {row.away_team} @ {row.home_team}: {row.away_probable_pitcher_name} vs "
              f"{row.home_probable_pitcher_name}{flag_str}", flush=True)

        is_postseason = row.game_type in POSTSEASON_GAME_TYPES
        try:
            props = generate_game_props(
                ctx, season, row.game_pk, row.home_team, row.away_team, target_date,
                home_ids, away_ids, int(home_pid), int(away_pid),
                venue_name=row.venue_name, n_trials=n_trials, rng=rng,
                postseason=is_postseason,
            )
        except ValueError as e:
            print(f"    skipped: {e}", flush=True)
            continue
        if props["debut_fallback_pids"]:
            # task #149: a true debut with zero cached history anywhere used the
            # generic (below-league-average) debut profile instead of crashing
            # this game -- flagged, not hidden, same transparency convention as
            # the lineup-source flags above.
            print(f"    note: player id(s) {props['debut_fallback_pids']} had no cached history "
                  f"(true MLB debut) -- used generic debut-cohort profile", flush=True)
        matchup = f"{row.away_team} @ {row.home_team}"
        results.append({
            "game_pk": row.game_pk, "matchup": matchup,
            "home_lineup_source": home_source, "away_lineup_source": away_source,
            "real_weather": has_real_weather, "postseason": is_postseason,
            "used_debut_fallback": bool(props["debut_fallback_pids"]), **props["game_props"],
        })
        # task #163 (combined-site export prerequisite, 2026-08): generate_game_props
        # already computes batter_props/pitcher_props per game -- previously discarded,
        # only game_props was ever persisted. Tag each with game_pk/matchup (not part of
        # the per-player DataFrame itself) and collect across games, same convention as
        # `results` above, so the site's export script has real per-player prop data to
        # read, matching NBA's own daily_props file pattern.
        bp = props["batter_props"].reset_index()  # index is already named "batter_id"
        bp.insert(0, "matchup", matchup)
        bp.insert(0, "game_pk", row.game_pk)
        batter_prop_rows.append(bp)
        pp = props["pitcher_props"].reset_index()  # index is already named "pitcher_id"
        # pitcher_id mixes real int ids with the "TEAM_POSITION_PLAYER" blowout
        # placeholder (props.py:619) -- parquet needs one type per column, so
        # cast to str for the persisted output (in-memory model code is unaffected).
        pp["pitcher_id"] = pp["pitcher_id"].astype(str)
        pp.insert(0, "matchup", matchup)
        pp.insert(0, "game_pk", row.game_pk)
        pitcher_prop_rows.append(pp)

    summary = pd.DataFrame(results)
    out_path = DATA_PROCESSED / f"daily_props_{target_date}.parquet"
    summary.to_parquet(out_path, index=False)
    print(f"\n=== summary: {len(summary)} games, saved to {out_path.name} ===", flush=True)

    batter_props_out = pd.concat(batter_prop_rows, ignore_index=True) if batter_prop_rows else pd.DataFrame()
    pitcher_props_out = pd.concat(pitcher_prop_rows, ignore_index=True) if pitcher_prop_rows else pd.DataFrame()
    batter_props_path = DATA_PROCESSED / f"batter_props_{target_date}.parquet"
    pitcher_props_path = DATA_PROCESSED / f"pitcher_props_{target_date}.parquet"
    batter_props_out.to_parquet(batter_props_path, index=False)
    pitcher_props_out.to_parquet(pitcher_props_path, index=False)
    print(f"=== {len(batter_props_out)} batter-prop rows saved to {batter_props_path.name} ===", flush=True)
    print(f"=== {len(pitcher_props_out)} pitcher-prop rows saved to {pitcher_props_path.name} ===", flush=True)
    for row in summary.itertuples():
        matchup = row.matchup
        away_team, home_team = matchup.split(" @ ")
        winner = home_team if row.home_win_prob > row.away_win_prob else away_team
        win_pct = max(row.home_win_prob, row.away_win_prob)
        flags = []
        if row.home_lineup_source != "real" or row.away_lineup_source != "real":
            sources = {row.home_lineup_source, row.away_lineup_source} - {"real"}
            flags.append("+".join(sorted(sources)) + " lineup")
        if not row.real_weather:
            flags.append("forecast weather")
        flag_str = f"  ({', '.join(flags)})" if flags else ""
        print(f"  {away_team} {row.mean_away_score:.1f} @ {home_team} {row.mean_home_score:.1f}"
              f"  -- {winner} favored {win_pct:.0%}{flag_str}", flush=True)
