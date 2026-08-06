"""Shared active-lineup-resolution helpers, extracted from
`generate_predictions.py` so both live entry points (`generate_predictions.py`
and `generate_props.py`) call the EXACT same logic for "who's actually
playing tonight and how many minutes will they get" -- a single source of
truth that can't silently drift between the two pipelines.

**Active-lineup resolution is 2-tier for v1, not a 3-tier design** --
documented honestly rather than silently shipped as if a third tier
existed: (1) RotoWire's Out/Doubtful signal (`fetch_rotowire_lineups.py`,
matched to a player ID via `player_name_crosswalk.py`) is the only real
injury signal actually wired up (the NBA official PDF injury report was
deliberately not built for v1, see MODEL_DOCUMENTATION.md); (2) a
last-resort trailing-minutes projection for every player NOT flagged
Out/Doubtful, renormalized across the team's recent rotation. Every output
row is tagged with which tier produced it.
"""

import pandas as pd
from nba_api.stats.endpoints import scoreboardv3

from src.ingest.fetch_schedule import season_str
from src.ingest.player_name_crosswalk import player_id_for_name
from src.ingest.team_codes import ABBREV_TO_TEAM_ID
from src.utils.paths import DATA_RAW

MINUTES_LOOKBACK_GAMES = 10


# gameId's 3rd character encodes the game type in every NBA Stats API endpoint that returns one
# (confirmed live 2026-08-02 via ScoreboardV3 on both a real regular-season date, "0022400561",
# and a real playoff date, "0042400141", the latter also carrying a populated `gameLabel`/
# `poRoundDesc` -- e.g. "West First Round" -- where the regular-season row's equivalent fields are
# blank). Regular season is by far the common case; anything else just needs a clear flag, not an
# exhaustive enum, since this project's own training data (`build_team_game_log`,
# `build_team_stat_game_log`) is regular-season-only throughout -- the model has never seen a
# playoff game's rotation/pace/HCA dynamics and has no validated basis for a playoff-specific
# adjustment (see MODEL_DOCUMENTATION.md for the open item).
_REGULAR_SEASON_GAME_ID_PREFIX = "0022"


def _game_type_from_id(game_id: str) -> str:
    return "regular_season" if game_id.startswith(_REGULAR_SEASON_GAME_ID_PREFIX) else "non_regular_season"


def games_on_date(game_date: str) -> pd.DataFrame:
    """ScoreboardV3's team-rows dataset doesn't carry an explicit home/away
    column, but `gameCode` (e.g. "20260115/MEMORL") reliably encodes it --
    confirmed live (2026-07-24) against a real slate: the 6 characters after
    the "/" are the AWAY team's 3-letter tricode followed by the HOME team's,
    matching each game's actual real-world home team. Parsed from
    `gameCode` directly rather than trusting team-row order (unconfirmed and
    not worth relying on).

    Adds `gameType` (`_game_type_from_id`) so callers can flag/warn when
    asked to predict a non-regular-season game -- this model's team ratings,
    pace, and home-court multiplier are all fit exclusively on regular-season
    data (see `team_strength.build_team_game_log`'s own docstring), so a
    playoff or play-in prediction is a genuinely different, unvalidated
    regime, not just "the same model, later in the calendar"."""
    sb = scoreboardv3.ScoreboardV3(game_date=game_date, timeout=30)
    dfs = sb.get_data_frames()
    game_meta, team_rows = dfs[1], dfs[2]

    games = []
    for row in game_meta.itertuples(index=False):
        code = row.gameCode.split("/")[1]
        away_abbrev, home_abbrev = code[:3], code[3:6]
        away_id = ABBREV_TO_TEAM_ID.get(away_abbrev)
        home_id = ABBREV_TO_TEAM_ID.get(home_abbrev)
        games.append({"gameId": row.gameId, "homeTeamId": home_id, "awayTeamId": away_id,
                      "homeAbbrev": home_abbrev, "awayAbbrev": away_abbrev,
                      "gameType": _game_type_from_id(row.gameId)})
    return pd.DataFrame(games)


def build_team_history(team_log: pd.DataFrame) -> tuple[dict, dict]:
    """From the long (one row per team per game) team-game log: for each
    team, its own gameIds ordered by date (all strictly prior to today,
    since `team_log` only ever contains already-completed, box-scored
    games), plus a per-team {gameId: 'home'/'away'} side lookup -- what
    `lineup_rating.team_recent_roster_rapm` and `resolve_active_lineup`
    both need for their lookback."""
    team_log = team_log.sort_values("gameDate")
    team_history: dict[int, list] = {}
    team_side: dict[int, dict] = {}
    for row in team_log.itertuples(index=False):
        team_history.setdefault(row.team, []).append(row.gameId)
        team_side.setdefault(row.team, {})[row.gameId] = "home" if row.is_home else "away"
    return team_history, team_side


def load_current_roster_player_ids(season: int) -> dict[int, set]:
    """teamId -> set of playerIds currently on that team's roster, from
    `CommonTeamRoster`'s cached snapshot for `season`. Same WALL-CLOCK-ONLY
    limitation as RotoWire's injury report (see
    `fetch_rotowire_lineups.warn_if_stale_for_backtest`): `CommonTeamRoster`
    returns "as of whenever last fetched", not a point-in-time archive, so
    this is only trustworthy for a genuinely live "tonight" call -- not a
    historical backtest re-query, which will see today's roster applied to
    a past date (the same class of contamination the injury-report warning
    already covers, generalized to also mention this source)."""
    path = DATA_RAW / f"team_rosters_{season_str(season)}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {int(team_id): set(g["PLAYER_ID"]) for team_id, g in df.groupby("TeamID")}


def resolve_active_lineup(team_abbrev: str, team_prior_game_ids: list, player_minutes: pd.DataFrame,
                           team_side_lookup: dict, injury_report: list[dict],
                           current_roster_ids: set | None = None) -> tuple[pd.Series, str]:
    """Returns (minutes_shares renormalized to 1.0, source_tag). Players
    RotoWire flags Out/Doubtful are excluded by player ID (via
    `player_name_crosswalk.player_id_for_name`) before minutes shares are
    computed -- a name with no crosswalk match (recent rookie/two-way player
    not yet in `nba_api`'s static list) can't be excluded by ID and is
    logged, not silently ignored.

    REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the trailing-
    minutes lookback had no check against actual CURRENT roster membership
    at all -- Sec22 fixed the cross-SEASON version of this (a team's "last
    N games" reaching into a PRIOR season's roster), but a player traded or
    waived MID-season kept contributing his real trailing minutes (and a
    nonzero share of tonight's projection) for up to
    `MINUTES_LOOKBACK_GAMES` games after he left, diluting every
    actually-rostered player's share and generating a full phantom prop row
    for someone who isn't on the team and won't play. `current_roster_ids`
    (from `load_current_roster_player_ids`, default `None` -- preserving
    the exact original behavior when not supplied) additionally excludes
    any player NOT on that set, same as the existing RotoWire exclusion."""
    out_names = [r["player"] for r in injury_report if r["team"] == team_abbrev and r["likely_out"]]
    out_ids = set()
    for name in out_names:
        pid = player_id_for_name(name)
        if pid is not None:
            out_ids.add(pid)
        else:
            print(f"    WARNING: RotoWire flagged '{name}' ({team_abbrev}) as Out/Doubtful but no "
                  f"player-ID crosswalk match was found -- this player CANNOT be excluded from "
                  f"tonight's projected minutes", flush=True)

    recent = team_prior_game_ids[-MINUTES_LOOKBACK_GAMES:]
    rows = []
    for gid in recent:
        side = team_side_lookup.get(gid)
        if side is None:
            continue
        g = player_minutes[(player_minutes["gameId"] == gid) & (player_minutes["team_side"] == side)]
        rows.append(g)
    if not rows:
        return pd.Series(dtype=float), "no recent rotation history"

    combined = pd.concat(rows, ignore_index=True)
    avg_minutes = combined.groupby("playerId")["minutes"].mean()
    avg_minutes = avg_minutes.drop(index=[pid for pid in out_ids if pid in avg_minutes.index])

    departed_count = 0
    if current_roster_ids is not None:
        departed = [pid for pid in avg_minutes.index if pid not in current_roster_ids]
        if departed:
            avg_minutes = avg_minutes.drop(index=departed)
            departed_count = len(departed)

    total = avg_minutes.sum()
    if total <= 0:
        return pd.Series(dtype=float), "no positive minutes in lookback window after exclusions"
    tag = (f"predictive (trailing {MINUTES_LOOKBACK_GAMES}-game minutes, {len(out_ids)} player(s) "
           f"excluded via RotoWire, {departed_count} excluded as no longer on the roster)")
    return avg_minutes / total, tag
