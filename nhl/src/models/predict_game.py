"""Live single-game prediction: the first live prediction entry point this
project has ever had (Sec41 of MODEL_DOCUMENTATION.md). Predicts a
hypothetical, not-yet-played matchup using the FROZEN production
configuration (`walk_forward_tie_ratio_poisson`, Cycle 26/Sec26, Sec38),
reusing every existing scoring primitive rather than re-deriving them.

Architecture (confirmed feasible before writing this, Sec41): every global
walk-forward constant in the production chain -- `away_b2b_adj`, the OT
logistic `(a,b)`, `ot_split`, `home_ice_multiplier` (Sec40), the reg-ratio
EWMA, the tie-mass calibration ratio -- is either a dev-fit-frozen constant
or a single pooled EWMA/expanding-mean value, so "today's" value is simply
the latest point in an already-existing series. `fit_deltas_per_game`'s
tie-mass machinery only needs the PROJECTED joint distribution (computable
for any hypothetical lambda pair, real outcome not required), so the full
tie-mass transfer genuinely applies to an unplayed game.

The one thing that can't be reused directly: `run_treated()`'s own output
only exposes already-COMBINED per-game lambdas (one specific historical
matchup already resolved) -- predicting a NEW matchup needs each team's
own per-channel attack/defense state BEFORE combination, which lives one
layer down (`team_strength_situational.py`'s per-team-game log). This
module rebuilds that one layer for "today," reusing every downstream
primitive (`predict_situational_lambda`, `two_sided_transfer`,
`predict_ot_logistic_prob`, `_resolve_joint`) unchanged.
"""

import numpy as np
import pandas as pd

from src.models.final_holdout_check import DEV_MAX_SEASON
from src.models.ot_logistic import fit_ot_so_split, predict_ot_logistic_prob
from src.models.overtime_shootout import add_regulation_score, add_walk_forward_reg_ratio
from src.models.rest_schedule import add_rest_days, fit_away_b2b_adjustment, symmetric_b2b_bias_credit
from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength_goalie import (
    GOALIE_ADJUSTMENT_FLOOR, add_walk_forward_goalie_strength, build_primary_goalie_per_team_game,
)
from src.models.team_strength_situational import (
    add_walk_forward_situational_strength, build_team_game_situational_log, predict_situational_lambda,
)
from src.models.two_sided_diagonal_transfer import (
    _indep_joint, above_mass, below_mass, diagonal_mass, fit_deltas_per_game, two_sided_transfer,
)
from src.models.validate_baseline import fit_home_ice_multiplier
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_drift import HALFLIFE_GAMES
from src.models.validate_tie_mass_ratio import MAX_GOALS, MIN_DEV_SEASON, _fit_dev_only_ot_logistic, _resolve_joint
from src.models.walk_forward_tie_ratio import add_walk_forward_ot_calibration_ratio
from src.utils.paths import DATA_RAW

PRIOR_GAMES_TOI = 10  # matches validate_situational_toi.py exactly
HEURISTIC_WINDOW_GAMES = 10  # matches validate_goalie_info_gap.py exactly

SIT_LOG_COLS = [
    "ev_attack_rate_per60", "ev_defense_rate_per60", "ev_league_avg_for_per60", "ev_league_avg_against_per60",
    "pp_attack_rate_per60", "pk_defense_rate_per60", "pp_league_avg_for_per60", "pk_league_avg_against_per60",
    "pk_league_avg_for_per60", "other_attack_rate_per60", "other_defense_rate_per60",
    "other_league_avg_for_per60", "other_league_avg_against_per60",
    "pp_toi_shrunk_mean", "pk_toi_shrunk_mean", "league_avg_ev_toi_min", "league_avg_other_toi_min",
]


def _load_schedule():
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    return schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]


def build_rated_situational_log(schedule: pd.DataFrame) -> pd.DataFrame:
    """Full history, one row per team per real played game, with every
    walk-forward rate column production actually uses -- IDENTICAL
    construction to `validate_situational_toi.run_validation`'s own body,
    up to (not including) the historical-matchup merge step, since a live
    prediction needs each team's OWN latest row, not a merged pair."""
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")
    log = build_team_game_situational_log(schedule, moneypuck)
    log = add_walk_forward_situational_strength(
        log, league_avg_halflife_games=HALFLIFE_GAMES, cross_season_weight=CROSS_SEASON_WEIGHT,
        prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER)
    for col in ("ev_league_avg_for_per60", "ev_league_avg_against_per60",
                "pp_league_avg_for_per60", "pk_league_avg_against_per60", "pk_league_avg_for_per60",
                "other_league_avg_for_per60", "other_league_avg_against_per60"):
        log[col] = log[col].bfill()
    log["league_avg_ev_toi_min"] = log["ev_toi_min"].shift(1).expanding().mean().bfill()
    log["league_avg_other_toi_min"] = log["other_toi_min"].shift(1).expanding().mean().bfill()
    log = add_walk_forward_mean(log, "pp_toi_min", PRIOR_GAMES_TOI, "pp_toi")
    log = add_walk_forward_mean(log, "pk_toi_min", PRIOR_GAMES_TOI, "pk_toi")
    log["pp_toi_shrunk_mean"] = log["pp_toi_shrunk_mean"].bfill()
    log["pk_toi_shrunk_mean"] = log["pk_toi_shrunk_mean"].bfill()
    return log


def latest_team_state(sit_log: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Each team's own most recent row STRICTLY BEFORE `as_of_date` -- the
    walk-forward state it carries INTO its next game."""
    prior = sit_log[pd.to_datetime(sit_log["gameDate"]) < as_of_date]
    return prior.sort_values("gameDate").groupby("team").tail(1).set_index("team")


def build_rated_goalie_log(schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_appearances, rated_log). `raw_appearances`
    (`build_primary_goalie_per_team_game`'s own output, pre-walk-forward)
    is what `goalie_relative_as_of` needs to correctly extend the real
    walk-forward series by one step for a live prediction. `rated_log`
    (full history, one row per (team, real played game), with each
    appearance's own goalie's walk-forward-shrunk `goalie_relative` --
    IDENTICAL construction to `validate_goalie.run_validation`'s goalie
    step, production leaves `goalie_league_avg_halflife_games=None`, its
    own un-passed default) is what `predict_starter` needs for the
    recent-workhorse heuristic."""
    raw = build_primary_goalie_per_team_game(schedule)
    rated = add_walk_forward_goalie_strength(raw)
    rated["goalie_relative"] = (rated["goalie_shrunk_mean"] - rated["goalie_league_avg"]).fillna(0.0)
    return raw, rated


def _normalize_goalie_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch.isspace())


def predict_starter(goalie_log: pd.DataFrame, team: str, as_of_date: pd.Timestamp,
                     window_games: int = HEURISTIC_WINDOW_GAMES,
                     injury_report: list[dict] | None = None, team_abbrev: str | None = None) -> str | None:
    """Recent-workhorse heuristic (matches `validate_goalie_info_gap.py`'s
    dev-only backtest version): whichever goalie had the most starts among
    this team's own trailing `window_games` real appearances strictly
    before `as_of_date` -- **now WITH real RotoWire injury exclusion for
    live prediction** (Sec39.1/Sec42.2), which the dev-only backtest
    version never needed (no historical injury archive exists to backtest
    against) but a genuine live prediction should use, matching the
    NHL/NBA/MLB siblings' shared "recent-workhorse heuristic + real-time
    injury exclusion" design.

    `injury_report`: raw rows from `fetch_rotowire_injuries.
    fetch_current_injury_report()`; `team_abbrev` must be RotoWire's own
    3-letter code for `team` (this project's own NHL API abbreviation --
    confirmed matching directly in `fetch_rotowire_injuries.py`'s own
    docstring sample, no crosswalk needed unless a future mismatch is
    found). Matches by NORMALIZED NAME against `goalie_log`'s own
    `goalieNameForShot` -- a name with no match is logged, not silently
    ignored (same convention as `fetch_rotowire_injuries.likely_out_
    player_names`'s own callers)."""
    team_hist = goalie_log[(goalie_log["team"] == team)
                            & (pd.to_datetime(goalie_log["gameDate"]) < as_of_date)].sort_values("gameDate")
    recent = team_hist[["goalieIdForShot", "goalieNameForShot"]].tail(window_games)
    if recent.empty:
        return None

    excluded_ids = set()
    if injury_report is not None and team_abbrev is not None:
        from src.ingest.fetch_rotowire_injuries import likely_out_player_names
        # Filter to goalies (position "G") BEFORE checking -- the raw injury report covers every
        # position, and checking skater names against a goalie-only history would print a
        # misleading "cannot exclude" warning for every injured skater, not just goalies.
        goalie_injury_rows = [r for r in injury_report if r.get("position") == "G"]
        out_names = {_normalize_goalie_name(n) for n in likely_out_player_names(goalie_injury_rows, team_abbrev)}
        # BUG FOUND AND FIXED (2026-07-25, caught by the live orchestration test, not silently
        # shipped): `.set_index(team_hist[...])` used the FULL, non-deduplicated `team_hist`
        # column as the new index while `.drop_duplicates(...)` had already shrunk the frame
        # itself -- a length mismatch (confirmed real: "Expected 29 rows, received array of
        # length 1397" on every single game in the first live test run). Both operations must
        # use the SAME already-deduplicated subset.
        deduped = team_hist.drop_duplicates("goalieIdForShot")
        id_by_name = deduped.set_index(deduped["goalieNameForShot"].map(_normalize_goalie_name))["goalieIdForShot"]
        for norm_name in out_names:
            if norm_name in id_by_name.index:
                excluded_ids.add(id_by_name.loc[norm_name])
            else:
                print(f"predict_starter: RotoWire flagged '{norm_name}' ({team_abbrev}) as Out/IR but no "
                      f"matching goalie found in this team's own recent appearance history -- cannot exclude", flush=True)

    candidates = recent[~recent["goalieIdForShot"].isin(excluded_ids)]
    if candidates.empty:
        candidates = recent  # every recent starter is flagged out -- fall back rather than predict no starter at all
    vals, counts = np.unique(candidates["goalieIdForShot"].values, return_counts=True)
    return vals[np.argmax(counts)]


def goalie_relative_as_of(raw_goalie_appearances: pd.DataFrame, goalie_id, as_of_date: pd.Timestamp,
                           team: str) -> float:
    """A specific goalie's own trailing `goalie_relative`, as of just
    before `as_of_date`.

    BUG FOUND AND FIXED while validating against a real historical game
    (2026-07-25, a genuinely bigger discrepancy than the same-day-ordering
    fix above -- ~0.07 in `lambda_away` for one real game): a goalie's own
    historical appearance ROW stores the value ENTERING that appearance,
    not the value after incorporating its own result. Returning "the last
    prior appearance's own stored value" is therefore off by exactly one
    walk-forward step -- confirmed real: for goalie 8473503.0's real
    2026-03-19 appearance, its own stored `goalie_relative` was -0.189206,
    but its ACTUAL 2026-03-23 appearance (4 days later, same walk-forward
    series) shows -0.215250 -- a real, non-trivial difference caused
    entirely by the 2026-03-19 game's own result being folded into the
    cumulative sum by the time of the NEXT appearance.

    Fixed the safe way -- reusing the real walk-forward function rather
    than hand-deriving the one-step recurrence, which risks a subtly wrong
    shrinkage-formula re-implementation: appends ONE placeholder appearance
    for this specific goalie at `as_of_date` (raw value irrelevant, never
    read) to the REAL, already-strictly-prior appearance history, then
    re-runs `add_walk_forward_goalie_strength` -- identical to how this
    project's own established "append + rerun the real function" pattern
    already works elsewhere. `raw_goalie_appearances` must be the RAW
    (pre-walk-forward) appearance table from `build_primary_goalie_per_team_game`,
    already filtered to strictly-prior real appearances only (so no future
    appearance of ANY goalie leaks into the pooled league-average)."""
    if goalie_id is None:
        return 0.0
    prior = raw_goalie_appearances[pd.to_datetime(raw_goalie_appearances["gameDate"]) < as_of_date]
    if not (prior["goalieIdForShot"] == goalie_id).any():
        return 0.0
    placeholder = pd.DataFrame([{
        "gameId": -1, "goalieIdForShot": goalie_id, "team": team, "gsax": 0.0,
        # gameDate must stay a plain "YYYY-MM-DD" string, matching every real row -- a
        # pd.Timestamp here would make the column mixed-type, breaking the walk-forward
        # function's own date sort (confirmed real: this silently produced NaN for the
        # placeholder row instead of raising, the first time this was written).
        "gameDate": as_of_date.strftime("%Y-%m-%d"), "season": prior["season"].max(),
    }])
    augmented = pd.concat([prior, placeholder], ignore_index=True)
    rated = add_walk_forward_goalie_strength(augmented)
    rated["goalie_relative"] = (rated["goalie_shrunk_mean"] - rated["goalie_league_avg"]).fillna(0.0)
    return float(rated.loc[rated["gameId"] == -1, "goalie_relative"].iloc[0])


def _global_constants(schedule: pd.DataFrame):
    """Every dev-fit-frozen constant and the LATEST value of every pooled
    walk-forward series, needed by any live prediction regardless of
    matchup. Computed once per process, not once per game."""
    schedule_ot = add_regulation_score(schedule)
    sit_log = build_rated_situational_log(schedule)

    home_ice_multiplier = fit_home_ice_multiplier(sit_log, max_season_exclusive=DEV_MAX_SEASON)

    raw_goalie_appearances, goalie_log = build_rated_goalie_log(schedule)

    # away_b2b_adj needs post-goalie-overlay lambda_away on dev-only games -- reconstruct the
    # full historical per-game combine exactly as validate_situational_toi.py + validate_goalie.py
    # do, to get a real dev-only lambda_away/actual_away series to fit against.
    games = sit_log[sit_log["is_home"]].copy().rename(columns={"goals_for": "actual_home"})
    away_side = sit_log[~sit_log["is_home"]][["gameId", "goals_for"] + SIT_LOG_COLS].rename(
        columns={"goals_for": "actual_away"})
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    lam_home_list, lam_away_list = [], []
    for row in games.itertuples():
        home_pp_toi = (row.pp_toi_shrunk_mean_home + row.pk_toi_shrunk_mean_away) / 2
        away_pp_toi = (row.pp_toi_shrunk_mean_away + row.pk_toi_shrunk_mean_home) / 2
        sh_kwargs = {"league_avg_sh_rate_per60": row.pk_league_avg_for_per60_home,
                     "home_pk_toi_min": away_pp_toi, "away_pk_toi_min": home_pp_toi}
        lh, la = predict_situational_lambda(
            home_ev_attack=row.ev_attack_rate_per60_home, home_ev_league_avg_for=row.ev_league_avg_for_per60_home,
            home_ev_defense=row.ev_defense_rate_per60_home, home_ev_league_avg_against=row.ev_league_avg_against_per60_home,
            away_ev_attack=row.ev_attack_rate_per60_away, away_ev_league_avg_for=row.ev_league_avg_for_per60_away,
            away_ev_defense=row.ev_defense_rate_per60_away, away_ev_league_avg_against=row.ev_league_avg_against_per60_away,
            home_pp_attack=row.pp_attack_rate_per60_home, home_pp_league_avg_for=row.pp_league_avg_for_per60_home,
            away_pp_attack=row.pp_attack_rate_per60_away, away_pp_league_avg_for=row.pp_league_avg_for_per60_away,
            home_pk_defense=row.pk_defense_rate_per60_home, home_pk_league_avg_against=row.pk_league_avg_against_per60_home,
            away_pk_defense=row.pk_defense_rate_per60_away, away_pk_league_avg_against=row.pk_league_avg_against_per60_away,
            home_other_attack=row.other_attack_rate_per60_home, home_other_league_avg_for=row.other_league_avg_for_per60_home,
            home_other_defense=row.other_defense_rate_per60_home, home_other_league_avg_against=row.other_league_avg_against_per60_home,
            away_other_attack=row.other_attack_rate_per60_away, away_other_league_avg_for=row.other_league_avg_for_per60_away,
            away_other_defense=row.other_defense_rate_per60_away, away_other_league_avg_against=row.other_league_avg_against_per60_away,
            league_avg_ev_toi_min=row.league_avg_ev_toi_min_home,
            home_pp_toi_min=home_pp_toi, away_pp_toi_min=away_pp_toi,
            league_avg_other_toi_min=row.league_avg_other_toi_min_home, **sh_kwargs,
        )
        lam_home_list.append(lh * home_ice_multiplier)
        lam_away_list.append(la / home_ice_multiplier)
    games["lambda_home"], games["lambda_away"] = lam_home_list, lam_away_list

    sched_teams = schedule[["gameId", "homeTeamAbbrev", "awayTeamAbbrev"]].drop_duplicates()
    games = games.merge(sched_teams, on="gameId", how="left")
    games = games.merge(goalie_log[["gameId", "team", "goalie_relative"]].rename(
        columns={"team": "homeTeamAbbrev", "goalie_relative": "home_goalie"}), on=["gameId", "homeTeamAbbrev"], how="left")
    games = games.merge(goalie_log[["gameId", "team", "goalie_relative"]].rename(
        columns={"team": "awayTeamAbbrev", "goalie_relative": "away_goalie"}), on=["gameId", "awayTeamAbbrev"], how="left")
    games["home_goalie"] = games["home_goalie"].fillna(0.0)
    games["away_goalie"] = games["away_goalie"].fillna(0.0)
    games["lambda_home"] = (games["lambda_home"] - games["away_goalie"]).clip(lower=GOALIE_ADJUSTMENT_FLOOR)
    games["lambda_away"] = (games["lambda_away"] - games["home_goalie"]).clip(lower=GOALIE_ADJUSTMENT_FLOOR)

    dev_only = games[(games["season"] >= MIN_DEV_SEASON) & (games["season"] < DEV_MAX_SEASON)]
    away_b2b_adj = fit_away_b2b_adjustment(schedule, dev_only["actual_away"], dev_only["lambda_away"], dev_only["gameId"])

    rest = add_rest_days(schedule)
    away_rest = rest[~rest["is_home"]][["gameId", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    games = games.merge(away_rest, on="gameId", how="left")
    from src.models.rest_schedule import add_walk_forward_b2b_incidence
    b2b_incidence = add_walk_forward_b2b_incidence(schedule)
    games = games.merge(b2b_incidence, on="gameId", how="left")
    credit = symmetric_b2b_bias_credit(games["p_b2b_walk_forward"], away_b2b_adj)
    games["lambda_home"] = games["lambda_home"] + credit
    games["lambda_away"] = games["lambda_away"] + credit + (games["away_rest"] == 0) * away_b2b_adj

    reg_ratios = add_walk_forward_reg_ratio(schedule_ot, HALFLIFE_GAMES)
    games = games.merge(reg_ratios, on="gameId", how="left")
    games["lambda_home_reg"] = games["lambda_home"] * games["home_reg_ratio_wf"]
    games["lambda_away_reg"] = games["lambda_away"] * games["away_reg_ratio_wf"]

    ratio_df = add_walk_forward_ot_calibration_ratio(games, schedule_ot, HALFLIFE_GAMES, MAX_GOALS)
    games = games.merge(ratio_df, on="gameId", how="left")

    ot_split = fit_ot_so_split(schedule_ot, max_season_exclusive=DEV_MAX_SEASON)
    a, b = _fit_dev_only_ot_logistic(games, schedule_ot)

    # BUG FOUND AND FIXED while validating against a real historical game (2026-07-25): these
    # walk-forward series update PER GAME, not per calendar day -- on a date with multiple games,
    # each game gets a slightly different reg_ratio_wf/ot_calibration_ratio value depending on its
    # exact position in the (gameDate, gameId) sort. Pre-extracting a single "latest" scalar here
    # (once per process) picked an ARBITRARY one of that day's games via a gameDate-only sort's
    # stable tie-break -- confirmed real: this put ot_calibration_ratio off by 1.8% relative to
    # run_treated()'s own value for a specific real game on a 4-game day. FIXED: return the full
    # per-row series (sorted gameDate+gameId, matching every walk-forward function in this
    # project); `predict_game` now filters to STRICTLY BEFORE its own specific `game_date` and
    # takes the last row from that filtered series -- correct regardless of how many other games
    # share that same calendar date, since none of THEM should inform tonight's prediction either.
    games_sorted = games.sort_values(["gameDate", "gameId"]).reset_index(drop=True)

    return {
        "sit_log": sit_log, "goalie_log": goalie_log, "raw_goalie_appearances": raw_goalie_appearances,
        "games_sorted": games_sorted,
        "home_ice_multiplier": home_ice_multiplier, "away_b2b_adj": away_b2b_adj,
        "ot_split": ot_split, "a": a, "b": b,
    }


def _latest_before(games_sorted: pd.DataFrame, as_of: pd.Timestamp, col: str) -> float:
    prior = games_sorted[pd.to_datetime(games_sorted["gameDate"]) < as_of]
    return float(prior[col].iloc[-1])


def predict_game(home_team: str, away_team: str, game_date: str, state: dict = None,
                  injury_report: list[dict] | None = None) -> dict:
    """Predicts a hypothetical matchup on `game_date` (YYYY-MM-DD, or any
    date -- including a REAL PAST date, for validation against
    `run_treated()`'s own already-computed output on that exact game).
    `state`: pass a pre-built `_global_constants(...)` dict to avoid
    rebuilding it per call (expensive); None rebuilds it fresh.
    `injury_report`: raw rows from `fetch_rotowire_injuries.
    fetch_current_injury_report()` -- None (the default, and always the
    case for a real past date, since no historical injury archive exists)
    means the recent-workhorse heuristic runs with no injury exclusion at
    all (Sec37.2's "realistic" comparator already validated this alone
    recovers nearly all of the real-starter information value)."""
    as_of = pd.Timestamp(game_date)
    schedule = _load_schedule()
    if state is None:
        state = _global_constants(schedule)

    latest = latest_team_state(state["sit_log"], as_of)
    if home_team not in latest.index or away_team not in latest.index:
        raise ValueError(f"no prior-game history for {home_team!r} or {away_team!r} before {game_date}")
    h, a = latest.loc[home_team], latest.loc[away_team]

    home_pp_toi = (h["pp_toi_shrunk_mean"] + a["pk_toi_shrunk_mean"]) / 2
    away_pp_toi = (a["pp_toi_shrunk_mean"] + h["pk_toi_shrunk_mean"]) / 2
    sh_kwargs = {"league_avg_sh_rate_per60": h["pk_league_avg_for_per60"],
                 "home_pk_toi_min": away_pp_toi, "away_pk_toi_min": home_pp_toi}
    lam_home_raw, lam_away_raw = predict_situational_lambda(
        home_ev_attack=h["ev_attack_rate_per60"], home_ev_league_avg_for=h["ev_league_avg_for_per60"],
        home_ev_defense=h["ev_defense_rate_per60"], home_ev_league_avg_against=h["ev_league_avg_against_per60"],
        away_ev_attack=a["ev_attack_rate_per60"], away_ev_league_avg_for=a["ev_league_avg_for_per60"],
        away_ev_defense=a["ev_defense_rate_per60"], away_ev_league_avg_against=a["ev_league_avg_against_per60"],
        home_pp_attack=h["pp_attack_rate_per60"], home_pp_league_avg_for=h["pp_league_avg_for_per60"],
        away_pp_attack=a["pp_attack_rate_per60"], away_pp_league_avg_for=a["pp_league_avg_for_per60"],
        home_pk_defense=h["pk_defense_rate_per60"], home_pk_league_avg_against=h["pk_league_avg_against_per60"],
        away_pk_defense=a["pk_defense_rate_per60"], away_pk_league_avg_against=a["pk_league_avg_against_per60"],
        home_other_attack=h["other_attack_rate_per60"], home_other_league_avg_for=h["other_league_avg_for_per60"],
        home_other_defense=h["other_defense_rate_per60"], home_other_league_avg_against=h["other_league_avg_against_per60"],
        away_other_attack=a["other_attack_rate_per60"], away_other_league_avg_for=a["other_league_avg_for_per60"],
        away_other_defense=a["other_defense_rate_per60"], away_other_league_avg_against=a["other_league_avg_against_per60"],
        league_avg_ev_toi_min=h["league_avg_ev_toi_min"], home_pp_toi_min=home_pp_toi, away_pp_toi_min=away_pp_toi,
        league_avg_other_toi_min=h["league_avg_other_toi_min"], **sh_kwargs,
    )
    lam_home = lam_home_raw * state["home_ice_multiplier"]
    lam_away = lam_away_raw / state["home_ice_multiplier"]

    goalie_log = state["goalie_log"]
    raw_goalie_appearances = state["raw_goalie_appearances"]
    home_starter = predict_starter(goalie_log, home_team, as_of, injury_report=injury_report, team_abbrev=home_team)
    away_starter = predict_starter(goalie_log, away_team, as_of, injury_report=injury_report, team_abbrev=away_team)
    home_goalie = goalie_relative_as_of(raw_goalie_appearances, home_starter, as_of, home_team)
    away_goalie = goalie_relative_as_of(raw_goalie_appearances, away_starter, as_of, away_team)
    lam_home = max(lam_home - away_goalie, GOALIE_ADJUSTMENT_FLOOR)
    lam_away = max(lam_away - home_goalie, GOALIE_ADJUSTMENT_FLOOR)

    games_sorted = state["games_sorted"]
    latest_p_b2b = _latest_before(games_sorted, as_of, "p_b2b_walk_forward")
    latest_reg_ratio_home = _latest_before(games_sorted, as_of, "home_reg_ratio_wf")
    latest_reg_ratio_away = _latest_before(games_sorted, as_of, "away_reg_ratio_wf")
    latest_ot_calibration_ratio = _latest_before(games_sorted, as_of, "ot_calibration_ratio")

    credit = -0.5 * latest_p_b2b * state["away_b2b_adj"]
    lam_home += credit
    lam_away += credit
    rest_hist = add_rest_days(schedule)
    away_prior = rest_hist[(~rest_hist["is_home"]) & (rest_hist["team"] == away_team)
                            & (pd.to_datetime(rest_hist["gameDate_dt"]) < as_of)].sort_values("gameDate_dt")
    away_on_b2b = False
    if not away_prior.empty:
        last_away_game_date = pd.to_datetime(away_prior["gameDate_dt"].iloc[-1])
        away_on_b2b = (as_of - last_away_game_date).days == 1
    if away_on_b2b:
        lam_away += state["away_b2b_adj"]

    lam_home_reg = lam_home * latest_reg_ratio_home
    lam_away_reg = lam_away * latest_reg_ratio_away

    reg_joint = _indep_joint(lam_home_reg, lam_away_reg, MAX_GOALS)
    diag, above, below = diagonal_mass(reg_joint), above_mass(reg_joint), below_mass(reg_joint)
    target_diag = latest_ot_calibration_ratio * diag
    delta_above, delta_below = fit_deltas_per_game(
        np.array([diag]), np.array([above]), np.array([below]), np.array([target_diag]))
    reg_joint = two_sided_transfer(reg_joint, float(delta_above[0]), float(delta_below[0]))

    lambda_diff_reg = lam_home_reg - lam_away_reg
    p_ot = predict_ot_logistic_prob(state["a"], state["b"], lambda_diff_reg)
    ot_split = state["ot_split"]
    p_home_wins_tie = ot_split["p_decided_in_ot"] * p_ot + ot_split["p_needs_so"] * ot_split["p_home_wins_so_flat"]
    final_joint = _resolve_joint(reg_joint, p_home_wins_tie, MAX_GOALS)

    n = final_joint.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return {
        "home_team": home_team, "away_team": away_team, "game_date": game_date,
        "home_starter_goalie_id": home_starter, "away_starter_goalie_id": away_starter,
        "lambda_home": float((final_joint * xs).sum()), "lambda_away": float((final_joint * ys).sum()),
        "home_win_prob_full": float(final_joint[xs > ys].sum()),
    }


if __name__ == "__main__":
    import sys
    home, away, gdate = sys.argv[1], sys.argv[2], sys.argv[3]
    result = predict_game(home, away, gdate)
    print(result)
