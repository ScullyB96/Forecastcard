"""Bullpen modeling for LIVE/FUTURE games, where (unlike a historical
backtest) we don't know who will actually pitch relief. Two walk-forward-safe
pieces:

  1. Each team's own aggregate bullpen rate: pool ALL relief (non-starter) PAs
     by that team's pitchers and reuse true_talent.py's generic Marcel
     machinery keyed by "team" instead of by individual pitcher. Pooling every
     reliever into one team-level unit is a deliberate simplification --
     predicting exactly WHICH reliever pitches which specific inning of a
     future game is a much harder roster-construction problem than this
     project takes on yet; this instead answers the more tractable "how good
     is this team's bullpen as a whole" question.
  2. A walk-forward projection of how many innings a probable starter is
     expected to go before the bullpen takes over, using the same Marcel
     two-level structure (recency-weighted prior-season average regressed
     toward league average, blended with in-season starts-to-date) applied to
     a continuous per-start scalar (innings) instead of a binary outcome rate.

This is the PREDICTIVE counterpart to validate_game_simulator.py's oracle
backtest, which replays each real historical game's ACTUAL pitcher-per-inning
sequence -- that's the right tool for asking "does using per-inning pitcher
identity help at all", but it can't be used on a game that hasn't been played
yet. See validate_predictive_bullpen.py for the live-style validation.

TRADE-DEADLINE GAP (fixed below): unlike a starter or a batter -- both of
which get a genuinely live, same-day team-affiliation signal from the MLB
Stats API's probable-pitcher/lineup fields -- a RELIEVER's team affiliation
here is inferred entirely from historical PA rows (mark_pitching_team, using
that specific game's own recorded home/away team, which is per-game-accurate
once a game is actually ingested). A pitcher traded 5 days ago has zero (or
a handful of) TeamB-tagged relief_log/closer_log rows, so
build_team_bullpen_roster/identify_closer will either miss him entirely or
badly understate his real role, while a newly-traded closer specifically
will lose to his TeamB predecessor's still-decaying appearance count for
WEEKS. `build_traded_pitcher_overrides` (below, fed by
src.ingest.fetch.fetch_transactions's real trade data) fixes this by
relabeling a just-traded pitcher's ENTIRE relief/closer history to his new
team once he's past his trade's effective_date -- seeding his usage weight
and closer-candidacy from his established PRIOR role rather than starting
him cold. See MODEL_DOCUMENTATION.md for the fuller story.
"""

import numpy as np
import pandas as pd

from src.models.true_talent import MARCEL_WEIGHTS, build_pregame_rates
from src.models.game_simulator import OUTCOMES
from src.models.crn import crn_bool, crn_choice, DECISION_BULLPEN_PICK, DECISION_CLOSER_USAGE

K_STARTS = 8  # stabilization constant in START-count units for expected innings/start --
              # a reasonable starting guess (not yet empirically re-validated), same
              # spirit as true_talent.py's STABILIZATION_PA constants.


def mark_pitching_team(pa: pd.DataFrame) -> pd.Series:
    """The team whose pitcher is on the mound for this PA (defense) -- the
    home team pitches in the Top half, the away team pitches in the Bottom half."""
    return pa["home_team"].where(pa["inning_topbot"] == "Top", pa["away_team"])


def _attach_starter_id(pa: pd.DataFrame) -> pd.DataFrame:
    tmp = pa.copy()
    tmp["pitching_team"] = mark_pitching_team(tmp)
    first_pa = tmp.sort_values(["game_pk", "pitching_team", "inning", "at_bat_number"]).groupby(
        ["game_pk", "pitching_team"], as_index=False
    ).nth(0)
    starters = first_pa[["game_pk", "pitching_team", "pitcher"]].rename(columns={"pitcher": "starter_id"})
    return tmp.merge(starters, on=["game_pk", "pitching_team"], how="left")


def build_starts_table(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, team) start: which pitcher started, and how many
    innings he pitched -- approximated as the LAST inning number he appears
    in (e.g. pulled after 2 batters in the 6th still counts as 6 innings, a
    known rounding-up simplification; exact fractional innings pitched isn't
    directly recoverable from PA-level Statcast without out-by-out sequencing)."""
    tmp = _attach_starter_id(pa)
    starter_rows = tmp[tmp["pitcher"] == tmp["starter_id"]]
    innings = starter_rows.groupby(["game_pk", "pitching_team", "starter_id", "season", "game_date"])["inning"].max()
    return innings.reset_index().rename(columns={"starter_id": "pitcher", "inning": "innings_pitched"})


def build_relief_pa(pa: pd.DataFrame) -> pd.DataFrame:
    """All PA rows thrown by a RELIEVER (not that game's starter), with a
    `team` column identifying the pitching team -- feeds true_talent.py's
    generic build_pregame_rates keyed by team instead of individual pitcher,
    giving each team's own walk-forward aggregate bullpen rate."""
    tmp = _attach_starter_id(pa)
    relief = tmp[tmp["pitcher"] != tmp["starter_id"]].copy()
    relief["team"] = relief["pitching_team"]
    return relief


def build_wide_bullpen_rates(pa: pd.DataFrame, park_factors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (team, game_pk, at_bat_number): pregame_rate_{outcome} for
    that team's AGGREGATE bullpen (all relievers pooled), walk-forward-safe.
    Only has rows for games where this team's bullpen actually threw at least
    one pitch -- see build_bullpen_snapshot for the as-of lookup that covers
    games where it didn't.

    park_factors_df: same park-neutralization as true_talent.build_pregame_rates
    -- a team's pooled bullpen rate is built via the identical Marcel machinery,
    so it carries the same park-contamination risk as an individual pitcher."""
    relief_pa = build_relief_pa(pa)
    base_cols = ["game_pk", "at_bat_number", "season"]
    result = None
    for outcome in OUTCOMES:
        r = build_pregame_rates(relief_pa, "team", outcome, park_factors_df)[base_cols + ["team", "pregame_rate"]].rename(
            columns={"pregame_rate": f"pregame_rate_{outcome}"}
        )
        result = r if result is None else result.merge(r, on=base_cols + ["team"], how="inner")
    return result


def build_bullpen_snapshot(pa: pd.DataFrame, park_factors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (team, game_pk): that team's aggregate bullpen rate as of
    just before that game (first relief appearance's pregame_rate, mirroring
    validate_game_simulator.player_game_snapshot's per-game pattern), plus
    game_date for the as-of lookup a candidate game (which may have used ZERO
    real relief that day) needs -- see nearest_prior_bullpen."""
    wide = build_wide_bullpen_rates(pa, park_factors_df)
    first = wide.sort_values("at_bat_number").groupby(["game_pk", "team"], as_index=False).nth(0)
    game_dates = pa[["game_pk", "game_date"]].drop_duplicates()
    first = first.merge(game_dates, on="game_pk")
    return first.sort_values(["team", "game_date"]).reset_index(drop=True)


def nearest_prior_bullpen(snapshot: pd.DataFrame, team: str, game_date, game_pk: int) -> pd.Series | None:
    """The most recent bullpen snapshot for `team` with game_date <= this
    game's date (inclusive of this exact game_pk if it itself had relief
    appearances -- its pregame_rate already only reflects data strictly
    before this game, so that's not a leak). Returns None if this team has no
    bullpen history yet (e.g. the very first games of the earliest season)."""
    sub = snapshot[(snapshot["team"] == team) & (snapshot["game_date"] <= game_date)]
    if sub.empty:
        return None
    exact = sub[sub["game_pk"] == game_pk]
    if not exact.empty:
        return exact.iloc[0]
    return sub.sort_values("game_date").iloc[-1]


def nearest_prior_pitcher_snapshot(snapshot_with_dates: pd.DataFrame, pitcher_id, game_date, game_pk: int) -> pd.Series | None:
    """Same as nearest_prior_bullpen, but for an individual pitcher rather
    than a team -- needed because a sampled roster reliever (see
    build_team_bullpen_roster) may not have actually pitched in THIS specific
    historical game_pk, so a plain game_pk-filtered snapshot (like
    validate_game_simulator.player_game_snapshot's per-game lookup) won't
    have a row for them. `snapshot_with_dates` must have a `game_date` column
    (player_game_snapshot's output doesn't include one by default -- merge it
    in from the PA table first)."""
    id_col = "pitcher" if "pitcher" in snapshot_with_dates.columns else "batter"
    sub = snapshot_with_dates[(snapshot_with_dates[id_col] == pitcher_id) & (snapshot_with_dates["game_date"] <= game_date)]
    if sub.empty:
        return None
    exact = sub[sub["game_pk"] == game_pk]
    if not exact.empty:
        return exact.iloc[0]
    return sub.sort_values("game_date").iloc[-1]


def build_expected_starter_innings(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe projection of how many innings a probable starter is
    expected to go, per (pitcher, team, game_pk) -- same Marcel two-level
    structure as true_talent.py (recency-weighted 5/4/3 prior-season average,
    regressed toward league average via a starts-based stabilization constant
    K_STARTS, blended with in-season starts-to-date), applied to a continuous
    per-start scalar (innings) instead of a binary per-PA outcome rate."""
    starts = build_starts_table(pa).sort_values(["pitcher", "season", "game_date", "game_pk"]).copy()

    out_frames = []
    for season, sdf in starts.groupby("season"):
        prior = starts[starts["season"] < season]
        if prior.empty:
            league_avg = starts.loc[starts["season"] == season, "innings_pitched"].mean()
            priors_df = pd.DataFrame(columns=["pitcher", "preseason_innings", "prior_weight_starts"])
        else:
            league_avg = prior["innings_pitched"].mean()
            season_agg = prior.groupby(["pitcher", "season"]).agg(
                starts=("innings_pitched", "size"), total_innings=("innings_pitched", "sum")
            ).reset_index()
            priors = []
            for pitcher, g in season_agg.groupby("pitcher"):
                g = g.set_index("season")
                num, den, raw_starts = 0.0, 0.0, 0.0
                for i, w in enumerate(MARCEL_WEIGHTS):
                    s = season - 1 - i
                    if s in g.index:
                        num += w * g.loc[s, "total_innings"]
                        den += w * g.loc[s, "starts"]
                        raw_starts += g.loc[s, "starts"]
                if den == 0:
                    continue
                priors.append({"pitcher": pitcher, "prior_starts": den, "prior_innings": num, "raw_prior_starts": raw_starts})
            priors_df = pd.DataFrame(priors, columns=["pitcher", "prior_starts", "prior_innings", "raw_prior_starts"])
            if not priors_df.empty:
                # RELIABILITY (and prior_weight_starts below) must use RAW,
                # unweighted starts -- not the MARCEL_WEIGHTS-weighted
                # `prior_starts`, which sums to 12 (not 3) across a full
                # 3-year history. Same bug class true_talent.py's own
                # reliability formula was fixed for (task #53): plugging a
                # ~4-5x-inflated denominator into K_STARTS silently
                # overstates confidence for any pitcher without a full,
                # uninterrupted 3-season track record (rookies, midseason
                # call-ups, converted relievers) -- verified on real 2025
                # data (task #160 correctness audit): median reliability
                # 0.852 (buggy, weighted) vs. 0.579 (correct, raw). The
                # WEIGHTED sum (prior_innings/prior_starts) is still exactly
                # right for the RATE estimate itself (real Marcel recency-
                # weighting) -- only the confidence side was wrong units.
                raw_avg = priors_df["prior_innings"] / priors_df["prior_starts"]
                priors_df["reliability"] = priors_df["raw_prior_starts"] / (priors_df["raw_prior_starts"] + K_STARTS)
                priors_df["preseason_innings"] = (
                    priors_df["reliability"] * raw_avg + (1 - priors_df["reliability"]) * league_avg
                )
                priors_df["prior_weight_starts"] = np.minimum(priors_df["raw_prior_starts"], K_STARTS)
                priors_df = priors_df[["pitcher", "preseason_innings", "prior_weight_starts"]]
            else:
                priors_df = pd.DataFrame(columns=["pitcher", "preseason_innings", "prior_weight_starts"])

        sdf = sdf.merge(priors_df, on="pitcher", how="left")
        sdf["preseason_innings"] = sdf["preseason_innings"].fillna(league_avg)
        sdf["prior_weight_starts"] = sdf["prior_weight_starts"].fillna(K_STARTS)

        grp = sdf.groupby("pitcher")
        sdf["season_innings_before"] = grp["innings_pitched"].cumsum() - sdf["innings_pitched"]
        sdf["season_starts_before"] = grp.cumcount()

        num = sdf["prior_weight_starts"] * sdf["preseason_innings"] + sdf["season_innings_before"]
        den = sdf["prior_weight_starts"] + sdf["season_starts_before"]
        sdf["expected_innings"] = num / den
        out_frames.append(sdf)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["pitcher", "pitching_team", "game_pk", "season", "game_date", "expected_innings", "prior_weight_starts", "preseason_innings", "season_innings_before", "season_starts_before", "innings_pitched"]]


def build_live_expected_innings(pa: pd.DataFrame) -> pd.Series:
    """Per-pitcher expected starter innings AS OF THEIR NEXT START -- the
    live-game companion to build_expected_starter_innings (2026-08-03
    audit, finding C4). The per-start table above is keyed by
    (pitcher, game_pk) from HISTORICAL PA rows, so a live/future game_pk
    can never hit it and props.py's lookup silently returned the 5.4
    fallback for EVERY starter, every game -- the whole per-pitcher
    innings projection was dead code on the live path while backtests
    (historical game_pks) exercised it, i.e. the validated configuration
    was not the deployed one.

    Value = each pitcher's blend INCLUDING his most recent start's
    innings (one Bayesian update past his last per-start row), taken from
    his latest available season. A pitcher returning after a season with
    zero starts gets his end-of-prior-season value (close to, not
    identical to, the re-weighted Marcel preseason prior -- documented
    approximation, not a gap). A pitcher with no starts anywhere (true
    opener/debut) is absent -- callers keep their explicit fallback."""
    per_start = build_expected_starter_innings(pa)
    if per_start.empty:
        return pd.Series(dtype=float)
    per_start = per_start.sort_values(["pitcher", "season", "game_date", "game_pk"])
    last = per_start.groupby(["pitcher", "season"]).tail(1).copy()
    next_num = (last["prior_weight_starts"] * last["preseason_innings"]
                + last["season_innings_before"] + last["innings_pitched"])
    next_den = last["prior_weight_starts"] + last["season_starts_before"] + 1
    last["expected_innings_next"] = next_num / next_den
    latest = last.sort_values("season").groupby("pitcher").tail(1)
    return latest.set_index("pitcher")["expected_innings_next"]


def build_predictive_bullpen_plan(starter_profile: dict, expected_innings: float,
                                   team_bullpen_profile: dict, innings: int = 9) -> dict:
    """{inning: profile} for a PREDICTIVE (no-foresight) game: the starter
    pitches through his walk-forward-projected expected innings (rounded to
    the nearest whole inning, minimum 1), then the team's own aggregate
    bullpen profile takes over for the rest. Simplified vs. reality in two
    ways, both deliberate first-version choices: a single hard cutoff rather
    than a probabilistic pull decision, and one pooled bullpen unit rather
    than assigning specific relievers to specific innings.

    Superseded by sample_bullpen_plan below for anywhere it's wired in --
    kept here since it's still a reasonable, simpler fallback when a team has
    no rest/usage history to build a roster from (e.g. earliest games of a
    season)."""
    cutoff = max(1, round(expected_innings))
    return {inning: (starter_profile if inning <= cutoff else team_bullpen_profile) for inning in range(1, innings + 1)}


# Relative to a fully-rested baseline of 1.0 (2+ days rest) -- derived
# directly from our own data, not guessed: across 2023-2026, a relief
# appearance happens on 0 days rest 15.1% as often as on 2+ days rest, and on
# 1 day rest 25.3% as often (0.1509/0.5616 and 0.2534/0.5616 respectively).
# This captures real bullpen-management behavior (an arm that just pitched is
# much less likely to be used again immediately) WITHOUT assuming a
# performance penalty -- checked directly and confirmed there isn't one:
# outcome rates (K%, BB%, HR%) are flat across rest-day buckets in our data,
# meaning managers already gate WHO pitches by rest, so a pitcher who DOES
# appear performs at his normal rate regardless of days rest.
REST_AVAILABILITY_WEIGHT = {0: 0.269, 1: 0.451}
DEFAULT_AVAILABILITY_WEIGHT = 1.0  # 2+ days rest, or no recent appearance on record
ROSTER_WINDOW_DAYS = 45


def build_relief_appearance_log(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, pitcher, game_pk, game_date) relief appearance --
    the roster + usage-frequency half of availability weighting (see
    build_team_bullpen_roster)."""
    relief = build_relief_pa(pa)
    return relief[["team", "pitcher", "game_pk", "game_date", "season"]].drop_duplicates(subset=["team", "pitcher", "game_pk"])


CLOSER_WINDOW_DAYS = 90  # longer than ROSTER_WINDOW_DAYS -- save situations are rarer
                          # than general relief appearances (confirmed: a team's top
                          # 9th-inning arm accumulates a median of just 22 such
                          # appearances in a trailing 90-day window), so a shorter
                          # window risks too little signal to identify the role reliably.
CLOSER_INNING9_RATE = 0.56  # measured directly from real 2023-2025 data: of every real
                            # 9th-inning-or-later relief appearance where the pitching
                            # team led by 1-3 runs (a save situation), 55.9% went to that
                            # team's own identified closer (the pitcher with the most
                            # 9th+-inning relief appearances in the same trailing window)
                            # vs. essentially never (1.6% of appearances) for any other
                            # reliever -- a signal build_team_bullpen_roster's general
                            # usage+rest weighting doesn't capture, since a real closer
                            # can easily have FEWER total relief innings than a multi-
                            # inning middle reliever despite being overwhelmingly the
                            # preferred arm specifically for the highest-leverage inning.


def build_closer_appearance_log(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, pitcher, game_pk, game_date) where `pitcher` was the
    FIRST RELIEVER (not that game's starter) to pitch in an inning >= 9 for
    `team` that game -- the role-identifying signal for identify_closer
    (deliberately separate from build_relief_appearance_log's general usage
    log: general relief-innings count is actually a poor proxy for "closer",
    since multi-inning middle relievers routinely rack up MORE total relief
    innings per season than a team's actual 9th-inning arm -- confirmed
    directly before trusting this, see project memory).

    Filters through build_relief_pa (the same starter-exclusion used
    elsewhere in this file) -- an EARLIER version of this function used the
    raw PA table directly and didn't exclude the starter, so a complete-game
    bid (starter still pitching in the 9th) would get logged as if that
    starter were the team's closer for that game, contaminating
    identify_closer's count. Caught in a later code-review pass; see project
    memory."""
    relief = build_relief_pa(pa)
    late = relief[relief["inning"] >= 9].sort_values(["game_pk", "team", "inning", "at_bat_number"])
    first_late = late.groupby(["game_pk", "team"], as_index=False).nth(0)
    return first_late[["team", "pitcher", "game_pk", "game_date", "season"]].drop_duplicates(
        subset=["team", "pitcher", "game_pk"]
    )


TRADE_OVERRIDE_LOOKBACK_DAYS = 60  # comfortably longer than ROSTER_WINDOW_DAYS=45 (the
                                    # longest trailing window an override needs to survive
                                    # for) -- see build_traded_pitcher_overrides.


def _apply_traded_overrides(log: pd.DataFrame, traded_overrides: dict | None, target_date) -> pd.DataFrame:
    """Relabels a just-traded pitcher's ENTIRE `team` history in `log` (both
    pre- and post-trade rows -- his real established role, not just what
    little he's done for the new team so far) to his new team, for any
    pitcher in `traded_overrides` whose trade is already in effect as of
    target_date. No-op (returns `log` unchanged, same object) when
    traded_overrides is None/empty -- every existing caller is unaffected.

    Deliberately does NOT expire the relabeling after some fixed horizon:
    once past effective_date, both his pre-trade (relabeled) and real
    post-trade appearances count toward his new team, and the CALLER's own
    existing trailing-window filter (ROSTER_WINDOW_DAYS/CLOSER_WINDOW_DAYS)
    already limits how far back any of this can matter -- well past that
    window, his real new-team appearances (if he's actually still pitching
    there) dominate on their own, no separate expiry needed."""
    if not traded_overrides:
        return log
    target_ts = pd.Timestamp(target_date)
    log = log.copy()
    for pid, info in traded_overrides.items():
        if pd.Timestamp(info["effective_date"]) <= target_ts:
            log.loc[log["pitcher"] == pid, "team"] = info["new_team"]
    return log


def build_traded_pitcher_overrides(transactions: pd.DataFrame, team_id_to_abbrev: dict,
                                    target_date, lookback_days: int = TRADE_OVERRIDE_LOOKBACK_DAYS) -> dict:
    """{pitcher_id: {"new_team": abbrev, "effective_date": Timestamp}} for
    real trades in the trailing `lookback_days` before target_date -- see
    src.ingest.fetch.fetch_transactions for the source data and this
    module's own docstring for why this exists. Keeps only the MOST RECENT
    trade per pitcher (a pitcher traded twice in one window keeps his latest
    team). Trades to a team_id not in `team_id_to_abbrev` (e.g. a Mexican/
    winter-league team appearing in the same feed) are silently skipped --
    not one of this project's tracked MLB teams."""
    if transactions.empty:
        return {}
    target_ts = pd.Timestamp(target_date)
    window_start = target_ts - pd.Timedelta(days=lookback_days)
    dates = pd.to_datetime(transactions["date"])
    sub = transactions[(dates >= window_start) & (dates <= target_ts)].sort_values("date")
    overrides = {}
    for _, row in sub.iterrows():
        new_team = team_id_to_abbrev.get(row["to_team_id"])
        if new_team is None:
            continue
        overrides[row["pitcher_id"]] = {"new_team": new_team, "effective_date": pd.Timestamp(row["date"])}
    return overrides


def identify_closer(closer_log: pd.DataFrame, team: str, target_date, window_days: int = CLOSER_WINDOW_DAYS,
                     traded_overrides: dict | None = None):
    """The pitcher with the most 9th+-inning relief appearances for `team` in
    the trailing `window_days` strictly before target_date -- walk-forward-safe.
    Returns None if there's no usable history yet (e.g. very early in a season).

    traded_overrides: optional (see build_traded_pitcher_overrides) -- a
    newly-traded pitcher's closer_log history is relabeled to `team` BEFORE
    filtering, so a just-acquired real closer is recognized immediately
    instead of losing to his new team's still-decaying incumbent for weeks.
    None (every existing caller) is byte-for-byte identical to before this
    parameter existed."""
    closer_log = _apply_traded_overrides(closer_log, traded_overrides, target_date)
    target_ts = pd.Timestamp(target_date)
    window_start = target_ts - pd.Timedelta(days=window_days)
    dates = pd.to_datetime(closer_log["game_date"])
    sub = closer_log[(closer_log["team"] == team) & (dates < target_ts) & (dates >= window_start)]
    if sub.empty:
        return None
    return sub["pitcher"].value_counts().idxmax()


def build_all_appearance_log(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, game_pk, game_date) for ANY appearance (start OR
    relief) -- the rest-day half of availability weighting: a pitcher who
    started (and threw 90+ pitches) yesterday is just as unavailable today as
    a reliever who threw yesterday, so rest lookups use every appearance, not
    just relief ones."""
    tmp = pa.copy()
    tmp["pitching_team"] = mark_pitching_team(tmp)
    return tmp[["pitcher", "game_pk", "game_date", "season"]].drop_duplicates(subset=["pitcher", "game_pk"])


def build_team_bullpen_roster(relief_log: pd.DataFrame, all_appearance_log: pd.DataFrame,
                               team: str, target_date: str, window_days: int = ROSTER_WINDOW_DAYS,
                               traded_overrides: dict | None = None) -> dict:
    """{pitcher_id: weight} for `team` as of target_date -- walk-forward-safe
    (only appearances STRICTLY BEFORE target_date). weight = how many times
    this pitcher appeared in relief for this team in the trailing
    `window_days` (a recency-weighted role/trust proxy -- a team's most-used
    relievers are, on average, the ones the manager trusts most) times a
    rest-day availability discount specific to target_date (see
    REST_AVAILABILITY_WEIGHT). A trailing window rather than full-season
    pooling deliberately tracks roster churn (call-ups, trades, injuries)
    that a full-season blend would smear over.

    traded_overrides: optional (see build_traded_pitcher_overrides) -- a
    newly-traded pitcher's relief_log history is relabeled to `team` BEFORE
    filtering, so his usage weight reflects his real established role
    instead of starting cold (or being invisible entirely) on his new team.
    None (every existing caller) is byte-for-byte identical to before this
    parameter existed."""
    relief_log = _apply_traded_overrides(relief_log, traded_overrides, target_date)
    target_ts = pd.Timestamp(target_date)
    window_start = target_ts - pd.Timedelta(days=window_days)
    dates = pd.to_datetime(relief_log["game_date"])
    sub = relief_log[(relief_log["team"] == team) & (dates < target_ts) & (dates >= window_start)]
    if sub.empty:
        return {}

    usage_counts = sub.groupby("pitcher").size()
    all_dates = pd.to_datetime(all_appearance_log["game_date"])
    weights = {}
    for pid, count in usage_counts.items():
        prior = all_appearance_log[(all_appearance_log["pitcher"] == pid) & (all_dates < target_ts)]
        if prior.empty:
            rest_weight = DEFAULT_AVAILABILITY_WEIGHT
        else:
            last_date = pd.to_datetime(prior["game_date"]).max()
            days_rest = (target_ts - last_date).days - 1
            rest_weight = REST_AVAILABILITY_WEIGHT.get(days_rest, DEFAULT_AVAILABILITY_WEIGHT)
        weights[pid] = count * rest_weight
    return weights


def sample_bullpen_plan(rng: np.random.Generator, starter_profile: dict, expected_innings: float,
                         roster_weights: dict, profile_lookup, fallback_profile: dict | None,
                         innings: int = 9, closer_id=None,
                         crn_keys: tuple | None = None) -> tuple[dict, dict]:
    """The roster-aware counterpart to build_predictive_bullpen_plan: instead
    of one flat blended bullpen rate for every post-starter inning, actually
    SAMPLE a specific reliever (their own individual walk-forward rate, via
    profile_lookup(pitcher_id)) for each inning, weighted by roster_weights
    (see build_team_bullpen_roster) and WITHOUT replacement -- a given
    reliever pitches at most one inning per simulated game, matching typical
    real bullpen usage (most relief appearances are exactly 1 inning). Falls
    back to fallback_profile (the old pooled aggregate rate) once the roster
    is exhausted or if no roster/rest history exists at all (e.g. earliest
    games of a season).

    Relief innings are assigned in REVERSE order (last inning first, working
    backward toward the first post-starter inning) -- confirmed on real data
    that a team's most-used relievers pitch LATER on average (closers in the
    9th, low-usage arms mopping up earlier innings): correlation 0.16 between
    a reliever's season appearance count and the inning of their relief
    stints, and mean first-relief-inning rises from 6.96 (lowest-usage
    quartile) to 7.58 (highest-usage quartile). Sampling from the full,
    undepleted weighted pool for the LATEST inning first (rather than
    left-to-right) systematically biases higher-weighted (more-trusted) arms
    toward later innings without hard-coding a rigid "closer always pitches
    the 9th" rule -- an earlier left-to-right version was tried and
    validated WORSE (SU accuracy 58.5% -> 57.5% vs. the simpler flat pooled
    rate it was meant to improve on) precisely because it discarded this
    real structure; see project memory for that finding.

    closer_id: this team's identified 9th-inning arm (see identify_closer),
    or None if unavailable/unidentified. Inning 9 specifically gets an extra,
    separate preference for this pitcher (sampled with probability
    CLOSER_INNING9_RATE whenever they're still in the available pool, matching
    the real measured rate) ON TOP OF the general reverse-order usage bias
    above -- confirmed necessary because usage-frequency alone doesn't
    reliably surface the closer: a team's real 9th-inning arm can have FEWER
    total relief innings than a multi-inning middle reliever despite being
    overwhelmingly the preferred choice specifically for the 9th (73% of a
    real closer's own appearances are in inning 9, vs. no comparable
    concentration for other relievers -- see project memory). No other inning
    (including extra innings 10+) gets this special treatment: a save
    specifically requires the pitching team to be leading, which this
    plan-built-in-advance can't check (it doesn't yet know the simulated
    score), so inning 9 -- where the overwhelming majority of real save
    situations occur -- is used as a proxy instead of genuine score-aware logic.

    Returns (profile_plan, id_plan): profile_plan is the {inning: profile}
    dict GameSimulator consumes; id_plan is the parallel {inning: pitcher_id}
    (or a fallback pseudo-id) for prop-generation attribution.

    crn_keys: optional (game_pk, trial[, side_tag]) tuple (see crn.py) --
    when given, both stochastic decisions below (the closer-preference check
    and the roster-weighted pick) are drawn via crn_bool/crn_choice keyed by
    *crn_keys, inning, DECISION_{CLOSER_USAGE,BULLPEN_PICK} instead of
    consuming the shared sequential rng stream, so two CRN-paired arms that
    build an identical roster/weights for a given (game, trial, inning) draw
    the exact same pick -- the same "stay synchronized until the real point
    of divergence" property game_simulator.py's own crn_keys already gives
    per-PA decisions (task #160 audit: this was flagged as dormant/unwired
    infrastructure -- DECISION_BULLPEN_PICK/DECISION_CLOSER_USAGE were
    defined in crn.py but never referenced anywhere until this). Default
    None preserves the exact prior rng.random()/rng.choice() behavior,
    byte-for-byte, for every existing caller."""
    cutoff = max(1, round(expected_innings))
    profile_plan, id_plan = {}, {}
    available = list(roster_weights.items())
    for inning in range(innings, cutoff, -1):
        # NOTE: the original code short-circuits on `inning == 9` FIRST (via
        # `and`), so rng.random() is only ever called for inning 9 -- this
        # must stay an explicit inning==9 gate, not a computed-every-inning
        # flag, or the non-CRN rng stream desyncs from its pre-existing
        # sequence for every OTHER inning (a real bug caught while writing
        # this, not present in the shipped version).
        if inning == 9 and closer_id is not None and any(pid == closer_id for pid, _ in available):
            if crn_keys is not None:
                closer_fires = crn_bool(CLOSER_INNING9_RATE, *crn_keys, inning, DECISION_CLOSER_USAGE)
            else:
                closer_fires = rng.random() < CLOSER_INNING9_RATE
        else:
            closer_fires = False
        if inning == 9 and closer_fires:
            pid = closer_id
            profile_plan[inning] = profile_lookup(pid)
            id_plan[inning] = pid
            available = [(p, w) for p, w in available if p != pid]
        elif available:
            ids, weights = zip(*available)
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()
            if crn_keys is not None:
                pick = crn_choice(list(range(len(ids))), weights, *crn_keys, inning, DECISION_BULLPEN_PICK)
            else:
                pick = rng.choice(len(ids), p=weights)
            pid = ids[pick]
            profile_plan[inning] = profile_lookup(pid)
            id_plan[inning] = pid
            available.pop(pick)
        elif fallback_profile is not None:
            profile_plan[inning] = fallback_profile
            id_plan[inning] = "BULLPEN_FALLBACK"
        else:
            # degenerate case (no roster AND no pooled fallback): reuse the
            # starter's RATES but override any per-PA attribution stamp (see
            # props.py's _stamp_pid / finding M15) -- these innings were never
            # really his, and crediting them to his real pid would inflate
            # his simulated K/outs props exactly like the illegal-re-entry
            # bug finding M17 removed.
            profile_plan[inning] = {**starter_profile, "pid": "BULLPEN_FALLBACK"}
            id_plan[inning] = "BULLPEN_FALLBACK"
    for inning in range(1, cutoff + 1):
        profile_plan[inning] = starter_profile
    return profile_plan, id_plan
