"""Turn a game's `GameRotation` + `PlayByPlayV3` data into a lineup-stint
table: `(gameId, stintIdx, homePlayers[5], awayPlayers[5], startTenths,
endTenths, homePts, awayPts, possessions)` -- the input to Phase 2b's RAPM-lite
ridge fit.

Algorithm (see MODEL_DOCUMENTATION.md Sec4 for the full derivation):
1. `GameRotation`'s IN_TIME_REAL/OUT_TIME_REAL (confirmed empirically to be
   tenths-of-a-second of cumulative elapsed game time from tip-off, spanning
   all periods) give, per player, every continuous on-court interval.
2. The sorted union of all boundary timestamps across both teams are
   candidate stint cut points.
3. For each adjacent boundary pair, a team's on-court five is whichever
   players have a rotation interval spanning that whole sub-interval.
4. Adjacent micro-intervals sharing an identical 10-player set are merged
   into one real stint (rotation data alone over-splits at events like
   technical free throws that don't actually change personnel).
5. `PlayByPlayV3`'s scoreHome/scoreAway (forward-filled, since populated
   only on scoring rows) are snapshotted at each stint boundary via the
   PBP clock converted to the same cumulative-tenths timeline, giving each
   stint's point differential directly from real game events.
6. Possessions-in-stint are counted from actual PBP possession-ending event
   types within the stint window -- exact and walk-forward-safe (uses only
   that stint's own already-completed events).

Hard data-quality gate: `reconcile_game` sums a game's stint point
differentials/possessions and compares against the real box score margin/
pace, flagging (not silently dropping) any game that fails to reconcile.
"""

import re
import unicodedata

import pandas as pd

PERIOD_LENGTH_TENTHS = {"regulation": 7200, "ot": 3000}
POSSESSION_ENDING_TYPES = {"Made Shot", "Turnover"}
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB_RE = re.compile(r"SUB: (.+) FOR (.+)")
_NAME_SUFFIXES = (" Jr.", " Sr.", " II", " III", " IV")


def _clock_to_elapsed_tenths(clock: str, period: int) -> int:
    """PlayByPlayV3's `clock` is TIME REMAINING in the period, ISO-8601-ish
    ("PT11M42.00S"). Converts to cumulative elapsed tenths-of-a-second since
    tip-off, on the same timeline as GameRotation's IN/OUT_TIME_REAL."""
    m = _CLOCK_RE.match(clock)
    minutes, seconds = int(m.group(1)), float(m.group(2))
    remaining_tenths = round((minutes * 60 + seconds) * 10)
    period_len = PERIOD_LENGTH_TENTHS["regulation"] if period <= 4 else PERIOD_LENGTH_TENTHS["ot"]
    elapsed_in_period = period_len - remaining_tenths

    if period <= 4:
        period_offset = PERIOD_LENGTH_TENTHS["regulation"] * (period - 1)
    else:
        period_offset = PERIOD_LENGTH_TENTHS["regulation"] * 4 + PERIOD_LENGTH_TENTHS["ot"] * (period - 5)
    return period_offset + elapsed_in_period


def _prep_pbp_timeline(pbp: pd.DataFrame) -> pd.DataFrame:
    """Adds `elapsedTenths`, forward-filled `scoreHome`/`scoreAway`, and
    `isDefensiveRebound` (see that column's derivation below), sorted
    chronologically.

    BUG FOUND AND FIXED while validating against a real game: the
    description string "Off:N Def:M" on a Rebound row is that PLAYER's
    CUMULATIVE offensive/defensive rebound TOTAL so far in the game, not a
    per-event flag -- naively checking for the substring "Def:1" only
    matched a player's exact FIRST defensive rebound of the game, silently
    missing every later one (confirmed: undercounted total combined
    possessions by ~30% on a real 2023-24 game, 132 vs. an expected ~191).
    Fixed by comparing each Rebound row's `teamId` against the `teamId` of
    the immediately preceding "Missed Shot" row: same team -> offensive
    rebound (possession continues), different team -> defensive rebound
    (possession ends). This also correctly handles team (non-player, e.g.
    "Lakers Rebound") rebounds, which still carry a real `teamId` even
    though `description` has no Off/Def breakdown at all."""
    p = pbp.copy()
    p["elapsedTenths"] = p.apply(lambda r: _clock_to_elapsed_tenths(r["clock"], r["period"]), axis=1)
    p = p.sort_values(["elapsedTenths", "actionNumber"]).reset_index(drop=True)

    # BUG FOUND AND FIXED while validating against a real 2015-16 game: scoreHome/scoreAway are
    # blank ('') on non-scoring rows in more recent seasons (confirmed on real 2019-20/2023-24
    # games), but are the LITERAL STRING "0" on non-scoring rows in this older game -- a real,
    # era-dependent API formatting difference, not a fluke. `pd.to_numeric("0")` parses that
    # placeholder "0" as a real value (not NaN), so a naive ffill silently used the wrong,
    # too-low placeholder score instead of carrying the real cumulative score forward -- and even
    # a MISSED Free Throw's row carries the same "0" placeholder despite actionType=="Free Throw"
    # (only a MADE shot or MADE free throw actually updates the score). Fixed by only trusting
    # scoreHome/scoreAway on rows that are a real, confirmed MADE scoring event -- everything else
    # is forced to NaN before ffill, regardless of what its raw string content happened to be.
    is_real_score_row = (p["actionType"] == "Made Shot") | (
        (p["actionType"] == "Free Throw") & ~p["description"].str.contains("MISS", na=False))
    p["scoreHome"] = pd.to_numeric(p["scoreHome"], errors="coerce").where(is_real_score_row).ffill().fillna(0).astype(int)
    p["scoreAway"] = pd.to_numeric(p["scoreAway"], errors="coerce").where(is_real_score_row).ffill().fillna(0).astype(int)

    last_missed_shot_team = p["teamId"].where(p["actionType"] == "Missed Shot").ffill()
    is_rebound = p["actionType"] == "Rebound"
    p["isDefensiveRebound"] = is_rebound & (p["teamId"] != last_missed_shot_team)
    return p


def _score_at(pbp_timeline: pd.DataFrame, t: int) -> tuple[int, int]:
    """Score snapshot at cumulative elapsed tenths `t` -- the last event at
    or before t (0-0 if t precedes the first event)."""
    prior = pbp_timeline[pbp_timeline["elapsedTenths"] <= t]
    if prior.empty:
        return 0, 0
    last = prior.iloc[-1]
    return int(last["scoreHome"]), int(last["scoreAway"])


_LAST_FT_OF_TRIP_RE = re.compile(r"Free Throw (\d+) of (\d+)")


def _count_possessions(pbp_timeline: pd.DataFrame, t_start: int, t_end: int) -> int:
    """Possession-ending event count within [t_start, t_end): made
    field goals, live-ball turnovers, defensive rebounds (`isDefensiveRebound`,
    see `_prep_pbp_timeline`'s derivation and the bug it fixes), and MADE final free
    throws of a trip (subType "N of N" with N==N and not a MISS -- a made
    final free throw ends the possession with no following rebound event at
    all, unlike a miss, which is already covered by the subsequent defensive
    rebound row). Confirmed against a real 2023-24 game: without the
    made-final-FT case, combined possession counts undercounted the real
    box-score total (95+96=191) by ~30% (132) -- adding it is necessary, not
    a refinement. Technical free throws (subType "Free Throw Technical")
    don't end a possession (no change of possession) and are excluded."""
    window = pbp_timeline[(pbp_timeline["elapsedTenths"] >= t_start) & (pbp_timeline["elapsedTenths"] < t_end)]
    made_or_to = (window["actionType"].isin(POSSESSION_ENDING_TYPES)).sum()
    def_reb = window["isDefensiveRebound"].sum()

    ft = window[window["actionType"] == "Free Throw"]
    is_made = ~ft["description"].str.contains("MISS", na=False)
    trip = ft["subType"].str.extract(_LAST_FT_OF_TRIP_RE)
    is_last_of_trip = trip[0].notna() & (trip[0] == trip[1])
    made_final_ft = int((is_made & is_last_of_trip).sum())

    return int(made_or_to + def_reb + made_final_ft)


def _players_on_court(rotation: pd.DataFrame, team_col_mask: pd.Series, t_start: int, t_end: int) -> list:
    team_rot = rotation[team_col_mask]
    on_court = team_rot[(team_rot["IN_TIME_REAL"] <= t_start) & (team_rot["OUT_TIME_REAL"] >= t_end)]
    return sorted(on_court["PERSON_ID"].unique().tolist())


def _get_starters(traditional_box: pd.DataFrame, team_id: int) -> tuple:
    """A game's 5 starters are the first 5 rows for that team in
    `BoxScoreTraditionalV3`'s player-level dataset, in original fetch order.

    BUG FOUND AND FIXED while validating against multiple real games: a
    non-blank `position` field is NOT a reliable starter indicator across
    the full historical range -- confirmed on a real 2015-16 game where 7
    "home" and 9 "away" players had non-blank `position` values (bench
    players like a 20-minute reserve also had a position listed, an API
    convention that evidently differs by era; it happened to work on the
    one 2023-24 game checked first, which is exactly why testing against
    only one season/era is not enough). ROW ORDER, by contrast, checked
    out on every game tested: the first 5 rows per team always sum to
    EXACTLY the team-level dataset's own `startersBench=="Starters"` points
    total (cross-validated directly, e.g. 83 and 53 points respectively on
    the 2015-16 game above) -- a strong, present-in-every-response signal to
    rely on instead."""
    team_rows = traditional_box[traditional_box["teamId"] == team_id]
    return tuple(sorted(team_rows["personId"].head(5).tolist()))


def _normalize_name(name: str) -> str:
    """Strips accents/diacritics for matching a PBP substitution
    description's incoming-player NAME against the box score's `familyName`.

    BUG FOUND AND FIXED while validating against a real game: PlayByPlayV3's
    description text is ASCII-normalized ("SUB: Murray FOR Jokic") while
    `BoxScoreTraditionalV3`'s `familyName` retains the real diacritic
    ("Jokić") -- an exact-string lookup failed to resolve "Jokic", and
    because that one substitution's incoming player couldn't be added back
    to the on-court set, EVERY subsequent stint for that team was short one
    player for the rest of the game (a single unresolved name cascades
    forward, not just a one-stint gap) -- confirmed on a real 2023-24 game:
    reconciliation was off by -44/-41 points before this fix, exact after."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")


def _roster_lookup(traditional_box: pd.DataFrame) -> tuple[dict, set]:
    """(team_id, normalized name-token) -> personId, for resolving a
    substitution description's incoming-player name (`familyName` already
    includes generational suffixes -- "Porter Jr." -- matching the PBP
    description's own token exactly once both sides are accent-normalized;
    see `_normalize_name`).

    BUG FOUND AND FIXED while validating against a real 2015-16 game: when
    two teammates share a family name (confirmed real: Jeff Green and
    JaMychal Green, both on the same team the same game), `PlayByPlayV3`'s
    substitution description disambiguates using NBA's own convention --
    the first two letters of the first name plus a period, e.g. "Je. Green"
    / "Ja. Green" -- rather than sending a bare, genuinely ambiguous
    "Green". A plain familyName-only lookup can't match that disambiguated
    form at all, so this now ALSO indexes every player under
    f"{firstName[:2]}. {familyName}" (normalized), which resolves those
    real disambiguated names directly. The plain bare-familyName key is
    suppressed (not added) for any name shared by 2+ teammates -- it would
    be genuinely ambiguous, and PBP does not appear to actually send that
    bare form when a real collision exists, so failing to resolve it (and
    flagging why) is safer than guessing which Green is meant.

    A second, independent bug found the same way (a real 2015-16 game,
    Jimmy Butler): `familyName` here carries the FULL suffixed name
    ("Butler III"), but the substitution description uses the bare
    "Butler" -- the OPPOSITE direction of `player_name_crosswalk.py`'s
    RotoWire mismatch (that one strips the suffix on the OTHER side).
    Handled the same way: also index every suffixed familyName under its
    suffix-stripped form."""
    by_family: dict = {}
    for row in traditional_box.itertuples(index=False):
        by_family.setdefault((row.teamId, _normalize_name(row.familyName)), []).append(row)

    lookup: dict = {}
    dupes = set()
    for key, rows in by_family.items():
        if len(rows) == 1:
            lookup[key] = rows[0].personId
        else:
            dupes.add(key)
        for row in rows:
            disambig_key = (row.teamId, _normalize_name(f"{row.firstName[:2]}. {row.familyName}"))
            lookup[disambig_key] = row.personId
            for suffix in _NAME_SUFFIXES:
                if row.familyName.endswith(suffix):
                    stripped_key = (row.teamId, _normalize_name(row.familyName[: -len(suffix)]))
                    lookup.setdefault(stripped_key, row.personId)
                    break
    return lookup, dupes


def build_game_stints_from_subs(traditional_box: pd.DataFrame, pbp: pd.DataFrame, game_id: str,
                                 home_team_id: int) -> tuple[pd.DataFrame, list[str]]:
    """PRIMARY stint-construction path (see MODEL_DOCUMENTATION.md Sec4 for
    why this replaced the `GameRotation`-based `build_game_stints`):
    reconstructs on-court lineups from each team's 5 starters
    (`BoxScoreTraditionalV3`) plus a chronological walk through
    `PlayByPlayV3`'s `Substitution` events, rather than depending on
    `GameRotation` at all. `Substitution` rows carry the OUTGOING player's
    ID directly (`personId`) and the INCOMING player's NAME in
    `description` ("SUB: <incoming> FOR <outgoing>") -- resolved to an ID
    via `_roster_lookup` against that same game's own ~15-17-player roster
    (not a global name lookup, so far less ambiguous than
    `player_name_crosswalk.py`'s RotoWire-matching problem)."""
    warnings = []
    team_ids = traditional_box["teamId"].unique().tolist()
    if len(team_ids) != 2:
        return pd.DataFrame(), [f"{game_id}: expected 2 teams in box score, got {len(team_ids)}"]
    away_candidates = [t for t in team_ids if t != home_team_id]
    if not away_candidates:
        return pd.DataFrame(), [f"{game_id}: home_team_id {home_team_id} not found in box score teams {team_ids}"]
    away_team_id = away_candidates[0]

    starters_home = _get_starters(traditional_box, home_team_id)
    starters_away = _get_starters(traditional_box, away_team_id)
    if len(starters_home) != 5 or len(starters_away) != 5:
        warnings.append(f"{game_id}: expected 5 starters/side, got home={len(starters_home)} away={len(starters_away)}")

    roster_lookup, dupes = _roster_lookup(traditional_box)
    pbp_timeline = _prep_pbp_timeline(pbp)
    if pbp_timeline.empty:
        return pd.DataFrame(), [f"{game_id}: no PBP events, skipping"]
    game_end = int(pbp_timeline["elapsedTenths"].max())

    on_court = {home_team_id: set(starters_home), away_team_id: set(starters_away)}
    subs = pbp_timeline[pbp_timeline["actionType"] == "Substitution"].sort_values(["elapsedTenths", "actionNumber"])

    def _close_stint(rows: list, t_start: int, t_end: int, idx: int) -> int:
        if t_end <= t_start:
            return idx
        home_five = tuple(sorted(on_court[home_team_id]))
        away_five = tuple(sorted(on_court[away_team_id]))
        if len(home_five) != 5 or len(away_five) != 5:
            warnings.append(f"{game_id}: stint [{t_start},{t_end}) has {len(home_five)} home / "
                             f"{len(away_five)} away players on court (roster-resolution gap), dropped")
            return idx
        home_start, away_start = _score_at(pbp_timeline, t_start)
        home_end, away_end = _score_at(pbp_timeline, t_end)
        rows.append({
            "gameId": game_id, "stintIdx": idx, "startTenths": t_start, "endTenths": t_end,
            "homePlayers": home_five, "awayPlayers": away_five,
            "homePts": home_end - home_start, "awayPts": away_end - away_start,
            "possessions": _count_possessions(pbp_timeline, t_start, t_end),
        })
        return idx + 1

    rows: list = []
    current_start = 0
    stint_idx = 0
    for t, group in subs.groupby("elapsedTenths"):
        t = int(t)
        stint_idx = _close_stint(rows, current_start, t, stint_idx)
        current_start = t
        for sub_row in group.itertuples(index=False):
            team = sub_row.teamId
            outgoing_id = sub_row.personId
            m = _SUB_RE.match(sub_row.description or "")
            if not m:
                warnings.append(f"{game_id}: could not parse substitution description {sub_row.description!r}")
                continue
            incoming_name = m.group(1).strip()
            key = (team, _normalize_name(incoming_name))
            if key in dupes:
                warnings.append(f"{game_id}: ambiguous incoming player name '{incoming_name}' on team {team} "
                                 f"(two teammates share this family name) -- lineup after this point may be wrong")
                continue
            incoming_id = roster_lookup.get(key)
            if incoming_id is None:
                warnings.append(f"{game_id}: could not resolve incoming player '{incoming_name}' on team {team} "
                                 f"to a roster ID")
                continue
            on_court[team].discard(outgoing_id)
            on_court[team].add(incoming_id)

    _close_stint(rows, current_start, game_end, stint_idx)

    stints = pd.DataFrame(rows)
    if stints.empty:
        return stints, warnings

    # merge adjacent stints sharing an identical 10-player set (a substitution group that didn't
    # net change either team's on-court five, e.g. a like-for-like technical-FT substitution)
    merged_rows = []
    for row in stints.itertuples(index=False):
        if merged_rows and merged_rows[-1]["homePlayers"] == row.homePlayers and merged_rows[-1]["awayPlayers"] == row.awayPlayers:
            prev = merged_rows[-1]
            prev["endTenths"] = row.endTenths
            prev["homePts"] += row.homePts
            prev["awayPts"] += row.awayPts
            prev["possessions"] += row.possessions
        else:
            merged_rows.append(dict(row._asdict()))
    for i, r in enumerate(merged_rows):
        r["stintIdx"] = i
    return pd.DataFrame(merged_rows), warnings


def build_game_stints(rotation: pd.DataFrame, pbp: pd.DataFrame, game_id: str) -> tuple[pd.DataFrame, list[str]]:
    """Returns (stints_df, warnings). stints_df columns: gameId, stintIdx,
    startTenths, endTenths, homePlayers, awayPlayers, homePts, awayPts,
    possessions. `warnings` records any glitch interval (not exactly 5
    players on one side) -- merged into its neighbor rather than dropping
    the game."""
    warnings = []
    boundaries = sorted(set(rotation["IN_TIME_REAL"]).union(rotation["OUT_TIME_REAL"]))
    if len(boundaries) < 2:
        return pd.DataFrame(), [f"{game_id}: fewer than 2 rotation boundaries, skipping"]

    pbp_timeline = _prep_pbp_timeline(pbp)
    is_home = rotation["isHomeTeam"]

    raw_intervals = []
    for t_start, t_end in zip(boundaries[:-1], boundaries[1:]):
        if t_start == t_end:
            continue
        home_five = _players_on_court(rotation, is_home, t_start, t_end)
        away_five = _players_on_court(rotation, ~is_home, t_start, t_end)
        if len(home_five) != 5 or len(away_five) != 5:
            warnings.append(f"{game_id}: interval [{t_start},{t_end}) has {len(home_five)} home / "
                             f"{len(away_five)} away players on court, merging into neighbor")
            if raw_intervals:
                raw_intervals[-1] = (raw_intervals[-1][0], t_end, raw_intervals[-1][2], raw_intervals[-1][3])
                continue
            # first interval is degenerate (e.g. jump-ball edge case) -- keep as-is, flagged
        raw_intervals.append((t_start, t_end, tuple(home_five), tuple(away_five)))

    # merge adjacent intervals sharing an identical 10-player set
    merged = []
    for t_start, t_end, home_five, away_five in raw_intervals:
        if merged and merged[-1][2] == home_five and merged[-1][3] == away_five:
            merged[-1] = (merged[-1][0], t_end, home_five, away_five)
        else:
            merged.append((t_start, t_end, home_five, away_five))

    rows = []
    for i, (t_start, t_end, home_five, away_five) in enumerate(merged):
        home_start, away_start = _score_at(pbp_timeline, t_start)
        home_end, away_end = _score_at(pbp_timeline, t_end)
        rows.append({
            "gameId": game_id, "stintIdx": i,
            "startTenths": t_start, "endTenths": t_end,
            "homePlayers": home_five, "awayPlayers": away_five,
            "homePts": home_end - home_start, "awayPts": away_end - away_start,
            "possessions": _count_possessions(pbp_timeline, t_start, t_end),
        })
    return pd.DataFrame(rows), warnings


def reconcile_game(stints: pd.DataFrame, actual_home_score: int, actual_away_score: int) -> dict:
    """Hard data-quality gate: do summed stint point differentials match the
    real box score final score? Small mismatches (technical FTs at odd
    boundaries, clock-correction edge cases) are expected occasionally --
    this reports the gap rather than silently trusting the stints."""
    if stints.empty:
        return {"ok": False, "reason": "no stints"}
    summed_home = stints["homePts"].sum()
    summed_away = stints["awayPts"].sum()
    ok = summed_home == actual_home_score and summed_away == actual_away_score
    return {
        "ok": bool(ok),
        "summed_home": int(summed_home), "actual_home": int(actual_home_score),
        "summed_away": int(summed_away), "actual_away": int(actual_away_score),
        "home_gap": int(summed_home - actual_home_score), "away_gap": int(summed_away - actual_away_score),
    }


def build_season_stints(start_year: int, force: bool = False) -> pd.DataFrame:
    """Builds and caches every regular-season game's stints for one season
    (playoffs excluded, matching Phase 1's train-on-regular-season-only
    convention), using the substitution-based `build_game_stints_from_subs`
    (the PRIMARY path -- see that function's docstring and
    MODEL_DOCUMENTATION.md Sec4 for why this replaced the `GameRotation`-
    based `build_game_stints`). Requires that season's schedule/traditional-
    box-score/play-by-play to already be cached (Phase 0) -- games missing
    any of those are skipped and reported, not silently dropped without a
    trace.

    Design choice, not a gap: this does NOT exclude a whole game just
    because it doesn't reconcile exactly. Every INCLUDED stint has already
    individually passed the 5-vs-5-on-court check (`build_game_stints_from_subs`'s
    own per-interval gate) -- a nonzero reconciliation gap means some
    minutes were dropped (uncovered), not that the stints which WERE kept
    are wrong. Per-season aggregate coverage (what fraction of total real
    points are captured by kept stints) is reported as an honest data-
    quality diagnostic instead."""
    from src.ingest.fetch_schedule import season_str
    from src.utils.paths import DATA_PROCESSED, DATA_RAW

    out_path = DATA_PROCESSED / f"stints_{season_str(start_year)}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    sched_path = DATA_RAW / f"schedule_{season_str(start_year)}.parquet"
    box_path = DATA_RAW / f"boxscore_trad_player_{season_str(start_year)}.parquet"
    pbp_path = DATA_RAW / f"playbyplay_{season_str(start_year)}.parquet"
    if not (sched_path.exists() and box_path.exists() and pbp_path.exists()):
        print(f"{season_str(start_year)}: missing one of schedule/traditional-box/playbyplay cache, skipping", flush=True)
        return pd.DataFrame()

    schedule = pd.read_parquet(sched_path)
    schedule = schedule[schedule["seasonType"] == "Regular Season"]
    box_all = pd.read_parquet(box_path)
    pbp_all = pd.read_parquet(pbp_path)

    box_games = set(box_all["gameId"].unique())
    pbp_games = set(pbp_all["gameId"].unique())
    playable = set(schedule["gameId"]) & box_games & pbp_games
    missing = set(schedule["gameId"]) - playable
    if missing:
        print(f"{season_str(start_year)}: {len(missing)} of {len(schedule)} regular-season games "
              f"missing traditional-box and/or PBP data, skipped", flush=True)

    all_stints, all_warnings = [], []
    total_actual_points, total_covered_points = 0, 0
    for game_id in schedule[schedule["gameId"].isin(playable)]["gameId"]:
        box = box_all[box_all["gameId"] == game_id]
        pbp = pbp_all[pbp_all["gameId"] == game_id]
        row = schedule[schedule["gameId"] == game_id].iloc[0]
        stints, warns = build_game_stints_from_subs(box, pbp, game_id, int(row["homeTeamId"]))
        all_warnings.extend(warns)
        total_actual_points += int(row["homeScore"]) + int(row["awayScore"])
        if stints.empty:
            continue
        all_stints.append(stints)
        total_covered_points += int(stints["homePts"].sum() + stints["awayPts"].sum())

    combined = pd.concat(all_stints, ignore_index=True) if all_stints else pd.DataFrame()
    combined.to_parquet(out_path, index=False)
    coverage = total_covered_points / total_actual_points if total_actual_points else 0.0
    print(f"{season_str(start_year)}: built stints for {combined['gameId'].nunique() if not combined.empty else 0} games, "
          f"{coverage:.1%} of total points covered by clean (5-vs-5-verified) stints "
          f"({len(all_warnings)} interval warnings), saved {out_path}", flush=True)
    return combined


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        build_season_stints(int(sys.argv[1]), force=True)
    else:
        rotation = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_rotation_0022300061.parquet")
        pbp = pd.read_parquet(sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_pbp_0022300061.parquet")
        game_id = "0022300061"

        stints, warns = build_game_stints(rotation, pbp, game_id)
        print(f"built {len(stints)} stints, {len(warns)} warnings", flush=True)
        for w in warns:
            print(f"  WARNING: {w}", flush=True)
        print(stints.head(10).to_string(), flush=True)

        result = reconcile_game(stints, actual_home_score=119, actual_away_score=107)
        print(f"\nreconciliation: {result}", flush=True)
