"""Park factors, derived directly from our own schedule data rather than a
third-party scrape. Baseball Savant's own Statcast Park Factors leaderboard
was tried first, but its `csv=true` URL parameter returns the JS-rendered
HTML page, not a CSV -- the real data is loaded via some other, unfound
endpoint. Rather than keep fighting an undocumented API, this computes park
factors ourselves the standard way, directly from data we already have and
fully control (same "derive it ourselves from trusted data" principle used
throughout this project).

Standard methodology (matches Baseball-Reference/FanGraphs' own basic
approach): for a team's home park, compare their combined (runs scored +
runs allowed) per game AT HOME vs. ON THE ROAD. Since the team's own talent
level is identical in both samples, this isolates the park's own scoring
environment. A 3-year rolling window is used (industry convention, per
project research notes) since single-season park factors are noisy.
"""

import numpy as np
import pandas as pd

from src.models.true_talent import STABILIZATION_PA
from src.utils.paths import DATA_RAW, DATA_PROCESSED

ROLLING_YEARS = 3
OUTCOMES = list(STABILIZATION_PA.keys())

# Real physical-building relocations (2026-07-25, verified against real schedule
# data): the A's left Oakland Coliseum for Sutter Health Park in 2025 (plus a
# handful of 2026 games at Las Vegas Ballpark), and the Rays left Tropicana Field
# for George M. Steinbrenner Field for ONE season (2025, hurricane damage to the
# Trop) before returning to Tropicana Field in 2026. Every OTHER team that shows
# more than one `venue_name` across 2023-2026 is a same-building sponsorship
# rename, not a relocation (confirmed directly: Guaranteed Rate Field/Rate Field,
# Minute Maid Park/Daikin Park, Dodger Stadium/UNIQLO Field at Dodger Stadium all
# have the SAME team playing 79-81 games/season at what is, physically, the one
# building) -- those don't need canonicalizing below since venue_name never
# feeds the actual factor math, only the venue-CONTINUITY check this dict exists
# for. Mapping is old-name -> current-name; both keys and the mapped-to value
# are treated as equivalent by `_canonical_venue`.
VENUE_RENAME_ALIASES = {
    "Guaranteed Rate Field": "Rate Field",
    "Minute Maid Park": "Daikin Park",
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
}


def _canonical_venue(venue_name: str) -> str:
    return VENUE_RENAME_ALIASES.get(venue_name, venue_name)


def team_home_road_runs(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): runs/game at home vs. on the road."""
    reg = schedule[(schedule["game_type"] == "R") & (schedule["status"] == "Final")].copy()
    reg["total_runs"] = reg["home_score"] + reg["away_score"]

    home = reg.groupby(["season", "home_team"]).agg(
        home_runs_per_game=("total_runs", "mean"), home_games=("total_runs", "size"),
        primary_venue=("venue_name", lambda s: s.mode().iloc[0]),
    ).reset_index().rename(columns={"home_team": "team"})
    home["canonical_venue"] = home["primary_venue"].map(_canonical_venue)
    road = reg.groupby(["season", "away_team"]).agg(
        road_runs_per_game=("total_runs", "mean"), road_games=("total_runs", "size"),
    ).reset_index().rename(columns={"away_team": "team"})

    return home.merge(road, on=["season", "team"], how="inner")


def _same_venue_rolling_mean(g: pd.DataFrame, value_col: str) -> pd.Series:
    """For each row (a team's one season), average `value_col` over that
    team's own PRIOR seasons that were played at the SAME canonical venue as
    THIS row -- not just the most recent 3 calendar seasons regardless of
    ballpark. A plain `.rolling(3)` silently blends a relocated team's old
    park into its new one's factor (confirmed real for the A's: Oakland
    Coliseum bleeding into the Sutter Health Park factor) or dilutes a
    returning team's stable history with a one-off displaced season
    (confirmed real for the Rays: one 2025 season at Steinbrenner Field
    diluting the 2026 Tropicana Field factor they'd already established
    across 2023-2024). `g` must already be sorted by season."""
    out = []
    venues = g["canonical_venue"].to_numpy()
    values = g[value_col].to_numpy()
    for i in range(len(g)):
        same_venue_prior = [values[j] for j in range(i) if venues[j] == venues[i]]
        recent = same_venue_prior[-ROLLING_YEARS:]
        out.append(sum(recent) / len(recent) if recent else float("nan"))
    return pd.Series(out, index=g.index)


def build_park_factors(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = DATA_RAW / f"schedule_{season}.parquet"
        if not path.exists():
            continue
        sched = pd.read_parquet(path)
        frames.append(team_home_road_runs(sched))
    all_seasons = pd.concat(frames, ignore_index=True).sort_values(["team", "season"])

    # Same-venue rolling window over seasons STRICTLY BEFORE each one -- a
    # season's own data must never contribute to its own park factor, AND
    # (2026-07-25 fix) a DIFFERENT ballpark's history must never contribute
    # to a relocated/returning team's current-venue factor either.
    all_seasons["home_runs_sum"] = all_seasons.groupby("team", group_keys=False).apply(
        lambda g: _same_venue_rolling_mean(g, "home_runs_per_game"), include_groups=False
    )
    all_seasons["road_runs_sum"] = all_seasons.groupby("team", group_keys=False).apply(
        lambda g: _same_venue_rolling_mean(g, "road_runs_per_game"), include_groups=False
    )
    all_seasons["park_factor"] = all_seasons["home_runs_sum"] / all_seasons["road_runs_sum"]
    return all_seasons


def _team_venue_lookup(seasons: list[int]) -> pd.DataFrame:
    """(season, team, canonical_venue) for every season present -- reads the
    raw schedule files directly (same source `team_home_road_runs` uses),
    since the PA table itself carries no venue_name column. Needed so
    `build_outcome_park_factors` (which only receives `pa`, not schedule) can
    apply the same same-venue-only rolling window as `build_park_factors`."""
    frames = []
    for season in seasons:
        path = DATA_RAW / f"schedule_{season}.parquet"
        if not path.exists():
            continue
        sched = pd.read_parquet(path)
        reg = sched[(sched["game_type"] == "R") & (sched["status"] == "Final")]
        primary = reg.groupby(["season", "home_team"])["venue_name"].agg(
            lambda s: s.mode().iloc[0]
        ).reset_index().rename(columns={"home_team": "team", "venue_name": "primary_venue"})
        frames.append(primary)
    lookup = pd.concat(frames, ignore_index=True)
    lookup["canonical_venue"] = lookup["primary_venue"].map(_canonical_venue)
    return lookup[["season", "team", "canonical_venue"]]


def _same_venue_rolling_sum(g: pd.DataFrame, value_col: str) -> pd.Series:
    """Same logic as `_same_venue_rolling_mean` but summing (for PA-count/
    event-count columns, not per-game rates) -- see that function's docstring
    for why a plain calendar-year rolling window is wrong for a relocated or
    displaced-then-returned team. `g` must already be sorted by season and
    carry a `canonical_venue` column."""
    out = []
    venues = g["canonical_venue"].to_numpy()
    values = g[value_col].to_numpy()
    for i in range(len(g)):
        same_venue_prior = [values[j] for j in range(i) if venues[j] == venues[i]]
        recent = same_venue_prior[-ROLLING_YEARS:]
        out.append(sum(recent) if recent else 0.0)
    return pd.Series(out, index=g.index)


def team_home_road_outcome_rates(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season, outcome): that team's OWN batters' rate for
    `outcome` when playing at home vs. on the road -- the per-category
    extension of team_home_road_runs, needed to plug park effects into the
    same per-outcome multiplicative-factor mechanism as state_factors and
    platoon multipliers (see game_simulator.py), rather than a single
    aggregate "runs" scalar that can't be wired into per-PA sampling."""
    # pa_table.py already filters to regular season only, no game_type column to re-filter on
    reg = pa.copy()
    is_home_batting = reg["inning_topbot"] == "Bot"
    reg["team"] = reg["home_team"].where(is_home_batting, reg["away_team"])
    reg["at_home"] = is_home_batting

    rows = []
    for outcome in OUTCOMES:
        reg["is_outcome"] = (reg["outcome"] == outcome).astype(int)
        g = reg.groupby(["season", "team", "at_home"]).agg(
            pa_count=("is_outcome", "size"), events=("is_outcome", "sum")
        ).reset_index()
        home = g[g["at_home"]].rename(columns={"pa_count": "home_pa", "events": "home_events"})
        road = g[~g["at_home"]].rename(columns={"pa_count": "road_pa", "events": "road_events"})
        merged = home.merge(road, on=["season", "team"], how="inner")
        merged["outcome"] = outcome
        rows.append(merged[["season", "team", "outcome", "home_pa", "home_events", "road_pa", "road_events"]])
    return pd.concat(rows, ignore_index=True)


def team_home_road_outcome_rates_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """Same as team_home_road_outcome_rates, but split further by the
    batter's own `stand` (L/R) -- the input to a handedness-specific park
    factor (2026-08-04 scoping: a park's short porch can be far more
    HR-friendly for one hand than the other, and the plain, hand-pooled
    park_factor above cannot express that; only a batter-facing park like
    Yankee Stadium's real physical asymmetry motivates this at all -- see
    build_outcome_park_factors_by_stand's own docstring for the real sample-
    size tradeoff this split introduces)."""
    reg = pa.copy()
    is_home_batting = reg["inning_topbot"] == "Bot"
    reg["team"] = reg["home_team"].where(is_home_batting, reg["away_team"])
    reg["at_home"] = is_home_batting

    rows = []
    for outcome in OUTCOMES:
        reg["is_outcome"] = (reg["outcome"] == outcome).astype(int)
        g = reg.groupby(["season", "team", "at_home", "stand"]).agg(
            pa_count=("is_outcome", "size"), events=("is_outcome", "sum")
        ).reset_index()
        home = g[g["at_home"]].rename(columns={"pa_count": "home_pa", "events": "home_events"})
        road = g[~g["at_home"]].rename(columns={"pa_count": "road_pa", "events": "road_events"})
        merged = home.merge(road, on=["season", "team", "stand"], how="inner")
        merged["outcome"] = outcome
        rows.append(merged[["season", "team", "stand", "outcome", "home_pa", "home_events", "road_pa", "road_events"]])
    return pd.concat(rows, ignore_index=True)


PARK_FACTOR_PRIOR_PA = 5000  # pseudo-PA of Bayesian shrinkage toward the league
                              # rate, see below. Raised from 200 (2026-07-21) --
                              # found via a systematic "check every factor
                              # table's real value distribution for extremes"
                              # audit (prompted by a real user-flagged anomaly
                              # in weather.py, which had the identical bug) that
                              # 200 pseudo-PA still left up to 28x-extreme park
                              # factors for the same astronomically rare
                              # categories (triple_play, catcher_interf) --
                              # their true rates are so tiny that even 200
                              # pseudo-PA barely dents a real observation.
                              # Re-verified the COMMON, outcome-driving
                              # categories (HR, hits, etc.) are still sane and
                              # directionally correct at 5000 (Coors still
                              # highest, Seattle/SF still lowest for home_run).


def build_outcome_park_factors(pa: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward-safe (3-year rolling window over the seasons STRICTLY
    BEFORE each season -- shift(1) before rolling, never including the
    target season's own data), per-(team, season, outcome) park factor: rate
    at home / rate on the road for that specific outcome category.

    Rare categories (catcher_interf, triple_play, sac_bunt, ...) can have
    ZERO road occurrences for a team in a 3-year window -- confirmed on real
    data: catcher_interf produced a literal inf park factor (division by
    zero) for one team, which then poisoned an entire Monte Carlo simulation
    with NaN probabilities. Fixed with the same Bayesian-shrinkage principle
    used throughout this project (true_talent.py, platoon_splits.py): add
    PARK_FACTOR_PRIOR_PA pseudo-PA of the league-average rate to both the
    home and road rate estimates before taking their ratio, so the
    denominator can never be exactly zero.

    Second bug, found via full-simulator validation (park factors made total
    score MAE meaningfully WORSE, 3.532 -> 3.846, simulated totals drifting
    FURTHER from actual): a raw home/road ratio is a right-skewed estimator
    (bounded below at 0, unbounded above), so its mean across the 30 teams is
    systematically > 1.0 even though every team's road games are some other
    team's home games and the league-wide total is definitionally balanced --
    confirmed directly on real data, EVERY outcome category except strikeout
    averaged above 1.0 across all three seasons (e.g. walk ~1.05, home_run
    ~1.05, sac_bunt ~1.13-1.23, catcher_interf ~1.24-2.87, triple_play a wild
    19-57x), consistently shifting probability mass toward hitting outcomes
    and away from strikeouts every time it was applied. Real park-factor
    providers (FanGraphs, ESPN) explicitly index their tables to a league
    average of 1.00 for exactly this reason. Fixed by renormalizing every
    (season, outcome) group to a population mean of 1.0 -- this preserves each
    team's park effect RELATIVE to the league (Coors still up, Seattle still
    down) while removing the estimator's systematic upward bias."""
    rates = team_home_road_outcome_rates(pa)
    venue_lookup = _team_venue_lookup(sorted(pa["season"].unique().tolist()))
    rates = rates.merge(venue_lookup, on=["season", "team"], how="left")
    rates = rates.sort_values(["team", "outcome", "season"])
    grp = rates.groupby(["team", "outcome"], group_keys=False)
    # SAME-VENUE rolling window (2026-07-25 fix, see _same_venue_rolling_mean's
    # docstring) -- a season's own data must never contribute to its own park
    # factor, and (the actual fix) a DIFFERENT ballpark's history must never
    # contribute to a relocated/returning team's current-venue factor either.
    home_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "home_pa"), include_groups=False)
    home_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "home_events"), include_groups=False)
    road_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "road_pa"), include_groups=False)
    road_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "road_events"), include_groups=False)

    # WALK-FORWARD league-rate anchor (task #160, 2026-07-26 correctness audit
    # fix): previously `rates.groupby("outcome")` pooled ALL seasons (2023-2026)
    # into one anchor reused for every season, contradicting this function's
    # own "STRICTLY BEFORE" docstring claim -- a real leakage bug, since the
    # Bayesian-shrinkage anchor is what PARK_FACTOR_PRIOR_PA=5000 pseudo-PA
    # dominates for any sparse cell. Verified: up to 26.6% relative error on
    # rare categories (triple_play, sac_bunt) from including 2025-2026 data in
    # what should be a strictly-prior-seasons anchor for a 2024 factor. Fixed
    # by cumulative-summing home_events/home_pa per outcome across seasons,
    # shifted by 1 (matching every other walk-forward step in this file) --
    # the true first season (no prior data at all) correctly gets NaN here,
    # which downstream callers (validate_game_simulator.py's park_factors_wide
    # lookup) already treat as "no adjustment" (1.0), same as a missing key.
    season_totals = rates.groupby(["outcome", "season"], as_index=False).agg(
        se_ev=("home_events", "sum"), se_pa=("home_pa", "sum")
    ).sort_values(["outcome", "season"])
    season_totals["cum_ev"] = season_totals.groupby("outcome")["se_ev"].transform(lambda s: s.shift(1).cumsum())
    season_totals["cum_pa"] = season_totals.groupby("outcome")["se_pa"].transform(lambda s: s.shift(1).cumsum())
    season_totals["league_rate"] = season_totals["cum_ev"] / season_totals["cum_pa"]
    # looked up via a (outcome, season) map rather than a merge -- a merge
    # resets rates' index, which would desync it from home_roll_pa/home_roll_ev/
    # road_roll_pa/road_roll_ev (Series computed above, aligned to rates' PRE-
    # merge index labels via the group-apply calls).
    league_rate_lookup = season_totals.set_index(["outcome", "season"])["league_rate"]
    league_rate = pd.Series(
        pd.MultiIndex.from_frame(rates[["outcome", "season"]]).map(league_rate_lookup), index=rates.index
    )

    rates["home_rate_rolling"] = (home_roll_ev + PARK_FACTOR_PRIOR_PA * league_rate) / (home_roll_pa + PARK_FACTOR_PRIOR_PA)
    rates["road_rate_rolling"] = (road_roll_ev + PARK_FACTOR_PRIOR_PA * league_rate) / (road_roll_pa + PARK_FACTOR_PRIOR_PA)
    raw_factor = rates["home_rate_rolling"] / rates["road_rate_rolling"]
    # A venue-season with ZERO same-venue prior observations has raw_factor
    # exactly 1.0 by construction (the shrinkage prior fills both sides of
    # the ratio) -- that is NO information, not a measured neutral factor
    # (2026-08-03 audit, finding M9). The old normalize-everything step then
    # divided that placeholder by the group mean (1.0326 for HR in 2025), so
    # a brand-new park was simulated as a ~3% HR-SUPPRESSING park on zero
    # observations (Steinbrenner Field, the short-porch Yankee-Stadium
    # clone, all of 2025), and the placeholders dragged the normalizing mean
    # itself toward 1 for the other teams. Fix: exclude no-history rows from
    # the normalizing mean and pin their final factor to exactly neutral 1.0
    # (byte-equivalent downstream to the missing-key/NaN "no adjustment"
    # path those callers already have).
    no_history = (home_roll_pa.fillna(0) == 0) | (road_roll_pa.fillna(0) == 0)
    # normalize each (season, outcome) group to a population mean of 1.0 --
    # only over venues with real same-venue prior data
    group_mean = raw_factor.where(~no_history).groupby([rates["season"], rates["outcome"]]).transform("mean")
    rates["park_factor"] = (raw_factor / group_mean).where(~no_history, 1.0)
    return rates[["season", "team", "outcome", "park_factor"]]


PARK_FACTOR_STAND_PRIOR_PA = 5000  # 2026-08-04 scoping: same magnitude as the
# pooled PARK_FACTOR_PRIOR_PA, kept equal for now rather than guessed larger --
# splitting by stand roughly halves the PA feeding each cell (confirmed on real
# 2024 data: ~1,268 avg home PA/team-season for lefties, ~1,702 for righties,
# vs. ~2,970 pooled), so this same prior weight produces MORE shrinkage toward
# neutral per cell than the pooled version does, which is the conservative
# direction to err in for a first pass -- raise only if validation shows the
# split factors are still too noisy even after this.


def build_outcome_park_factors_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """Same walk-forward, same-venue-rolling, Bayesian-shrinkage, mean-1.0-
    renormalization methodology as build_outcome_park_factors, extended with
    an extra `stand` grouping dimension throughout -- see that function's own
    docstring for why each of those three mechanisms (shrinkage against a
    walk-forward league anchor, same-venue-only rolling, mean-1.0 renorm) is
    there; nothing about the reasoning changes by adding `stand`, only the
    grouping keys do.

    The league-rate anchor and the mean-1.0 renormalization are both computed
    PER STAND (i.e. lefties are shrunk toward and renormalized against the
    LEFTY league-wide rate, righties against the RIGHTY league-wide rate) --
    not pooled -- so a park's lefty and righty factors can each be read as
    "how much better/worse than an average park is this, for a batter of
    that hand," independently of any overall batted-ball-rate difference
    between lefties and righties leaguewide.

    2026-08-04 scoping note (real numbers, not estimated): splitting by stand
    roughly halves the PA in every team-season-outcome cell versus the pooled
    version (see PARK_FACTOR_STAND_PRIOR_PA's own comment) -- expect these
    factors to be shrunk harder toward neutral than the pooled ones, and to
    carry real, not-yet-validated uncertainty. Treat as an experimental
    output pending the sanity-check + full-stack A/B this scoping call
    onward, not as a drop-in replacement for the pooled park_factor."""
    rates = team_home_road_outcome_rates_by_stand(pa)
    venue_lookup = _team_venue_lookup(sorted(pa["season"].unique().tolist()))
    rates = rates.merge(venue_lookup, on=["season", "team"], how="left")
    rates = rates.sort_values(["team", "stand", "outcome", "season"])
    grp = rates.groupby(["team", "stand", "outcome"], group_keys=False)
    home_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "home_pa"), include_groups=False)
    home_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "home_events"), include_groups=False)
    road_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "road_pa"), include_groups=False)
    road_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "road_events"), include_groups=False)

    season_totals = rates.groupby(["outcome", "stand", "season"], as_index=False).agg(
        se_ev=("home_events", "sum"), se_pa=("home_pa", "sum")
    ).sort_values(["outcome", "stand", "season"])
    season_totals["cum_ev"] = season_totals.groupby(["outcome", "stand"])["se_ev"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["cum_pa"] = season_totals.groupby(["outcome", "stand"])["se_pa"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["league_rate"] = season_totals["cum_ev"] / season_totals["cum_pa"]
    league_rate_lookup = season_totals.set_index(["outcome", "stand", "season"])["league_rate"]
    league_rate = pd.Series(
        pd.MultiIndex.from_frame(rates[["outcome", "stand", "season"]]).map(league_rate_lookup),
        index=rates.index,
    )

    rates["home_rate_rolling"] = (
        home_roll_ev + PARK_FACTOR_STAND_PRIOR_PA * league_rate
    ) / (home_roll_pa + PARK_FACTOR_STAND_PRIOR_PA)
    rates["road_rate_rolling"] = (
        road_roll_ev + PARK_FACTOR_STAND_PRIOR_PA * league_rate
    ) / (road_roll_pa + PARK_FACTOR_STAND_PRIOR_PA)
    raw_factor = rates["home_rate_rolling"] / rates["road_rate_rolling"]
    no_history = (home_roll_pa.fillna(0) == 0) | (road_roll_pa.fillna(0) == 0)
    group_mean = raw_factor.where(~no_history).groupby(
        [rates["season"], rates["outcome"], rates["stand"]]
    ).transform("mean")
    rates["park_factor"] = (raw_factor / group_mean).where(~no_history, 1.0)
    return rates[["season", "team", "stand", "outcome", "park_factor"]]


def all_batters_at_park_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team=home park, stand, outcome): outcome
    events/PA from EVERY batter (both the home team's own hitters AND every
    visiting team's hitters) in games hosted at that park, split by stand.

    This is the real fix for team_home_road_outcome_rates_by_stand's failed
    sanity check (2026-08-04): that function only ever counts the home
    team's OWN batters (its `team` column is set to whoever is CURRENTLY
    BATTING, so grouping by team+at_home structurally excludes the visiting
    team's PA at that same park) -- so a handedness split of it is really
    measuring "does this one team's specific lefty/righty hitters do better
    at home," not "does this PARK favor lefties/righties as a class." With
    only ~13-15 hitters of a given hand on one roster, that's dominated by
    which specific sluggers happen to be on the team, not the park's actual
    geometry -- confirmed directly: Yankee Stadium's real, extreme, well-
    known short-porch lefty advantage came out BACKWARDS (righty factor >
    lefty) under that method, and the raw year-over-year home/road ratio for
    NYY lefties swung from 0.91 to 1.72 -- noise, not signal.

    Grouping by home_team here instead (every PA in a game hosted at team
    T's park, regardless of who's at bat) pools every OTHER team's batters
    who've ever visited too -- turning a ~13-15-hitter sample into a real,
    league-wide cross-section for that park, which is what a true park-
    geometry effect needs to be measurable at all."""
    reg = pa.copy()
    rows = []
    for outcome in OUTCOMES:
        reg["is_outcome"] = (reg["outcome"] == outcome).astype(int)
        g = reg.groupby(["season", "home_team", "stand"]).agg(
            at_pa=("is_outcome", "size"), at_events=("is_outcome", "sum")
        ).reset_index().rename(columns={"home_team": "team"})
        g["outcome"] = outcome
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def build_component_park_factors_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """Component-method handedness-split park factor: for each (team=park,
    stand, outcome), compares the outcome rate of ALL batters of that hand
    who played AT that park (home team's own hitters plus every visiting
    team's, see all_batters_at_park_by_stand) against the same-hand league-
    wide rate EVERYWHERE ELSE (every other park's games combined) -- not
    against that one team's own road games, unlike build_outcome_park_
    factors_by_stand's team-fixed-effect design. Pooling every visiting
    team's batters gives each park-hand cell a real, diverse, league-wide
    sample instead of one team's ~13-15-hitter roster, which is what that
    simpler version's failed Yankee-Stadium/Fenway sanity check showed was
    necessary, not just nice-to-have (2026-08-04).

    Two asymmetric walk-forward treatments, intentional:
    - AT-PARK side: 3-year SAME-VENUE rolling sum (_same_venue_rolling_sum,
      identical convention to every other park-factor function in this
      file) -- protects against a relocated team's old building bleeding
      into its new one's factor, same real bug class already fixed
      elsewhere here.
    - ELSEWHERE side: plain trailing-3-season walk-forward sum, NOT same-
      venue-restricted -- "elsewhere" has no single venue, it's the rest of
      the whole league in the same trailing window, so a venue restriction
      doesn't apply; using the same 3-season depth keeps both sides
      comparing the same scoring-environment era (ball changes, rule
      changes) rather than one side windowed and the other all-history.

    Still walk-forward-safe throughout (a season's own data never leaks
    into its own factor) and still uses the same Bayesian-shrinkage-then-
    mean-1.0-renormalization pattern as every other factor in this file --
    only the definition of "the comparison sample" changes."""
    at_park = all_batters_at_park_by_stand(pa)
    venue_lookup = _team_venue_lookup(sorted(pa["season"].unique().tolist()))
    at_park = at_park.merge(venue_lookup, on=["season", "team"], how="left")
    at_park = at_park.sort_values(["team", "stand", "outcome", "season"])

    # League-wide per-season total (every park, every team) -- summing
    # at_park's own at_pa/at_events across all 30 teams reconstructs the
    # whole league's PA for that (season, stand, outcome), since every PA
    # belongs to exactly one team's home games.
    league_totals = at_park.groupby(["season", "stand", "outcome"], as_index=False).agg(
        league_pa=("at_pa", "sum"), league_events=("at_events", "sum")
    )
    at_park = at_park.merge(league_totals, on=["season", "stand", "outcome"], how="left")
    at_park["elsewhere_pa"] = at_park["league_pa"] - at_park["at_pa"]
    at_park["elsewhere_events"] = at_park["league_events"] - at_park["at_events"]

    grp = at_park.groupby(["team", "stand", "outcome"], group_keys=False)
    at_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "at_pa"), include_groups=False)
    at_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "at_events"), include_groups=False)
    # plain trailing-3-season walk-forward sum (shift(1) before rolling(3)) --
    # not same-venue-restricted, see docstring.
    elsewhere_roll_pa = grp.apply(
        lambda g: g["elsewhere_pa"].shift(1).rolling(ROLLING_YEARS, min_periods=1).sum(), include_groups=False
    )
    elsewhere_roll_ev = grp.apply(
        lambda g: g["elsewhere_events"].shift(1).rolling(ROLLING_YEARS, min_periods=1).sum(), include_groups=False
    )

    season_totals = at_park.groupby(["outcome", "stand", "season"], as_index=False).agg(
        se_ev=("at_events", "sum"), se_pa=("at_pa", "sum")
    ).sort_values(["outcome", "stand", "season"])
    season_totals["cum_ev"] = season_totals.groupby(["outcome", "stand"])["se_ev"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["cum_pa"] = season_totals.groupby(["outcome", "stand"])["se_pa"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["league_rate"] = season_totals["cum_ev"] / season_totals["cum_pa"]
    league_rate_lookup = season_totals.set_index(["outcome", "stand", "season"])["league_rate"]
    league_rate = pd.Series(
        pd.MultiIndex.from_frame(at_park[["outcome", "stand", "season"]]).map(league_rate_lookup),
        index=at_park.index,
    )

    at_park["at_rate_rolling"] = (at_roll_ev + PARK_FACTOR_STAND_PRIOR_PA * league_rate) / (
        at_roll_pa + PARK_FACTOR_STAND_PRIOR_PA
    )
    at_park["elsewhere_rate_rolling"] = (elsewhere_roll_ev + PARK_FACTOR_STAND_PRIOR_PA * league_rate) / (
        elsewhere_roll_pa + PARK_FACTOR_STAND_PRIOR_PA
    )
    raw_factor = at_park["at_rate_rolling"] / at_park["elsewhere_rate_rolling"]
    no_history = (at_roll_pa.fillna(0) == 0) | (elsewhere_roll_pa.fillna(0) == 0)
    group_mean = raw_factor.where(~no_history).groupby(
        [at_park["season"], at_park["outcome"], at_park["stand"]]
    ).transform("mean")
    at_park["park_factor"] = (raw_factor / group_mean).where(~no_history, 1.0)
    return at_park[["season", "team", "stand", "outcome", "park_factor"]]


PULLED_CONTACT_SPRAY_THRESHOLD_DEG = 10  # matches the 2026-08-04 geometry-check
# investigation -- past this many degrees from straight up the middle counts as
# "pulled" toward the batter's own side. Sign convention (positive = right-field
# side, negative = left-field side, 0 = dead center) validated directly against
# real league-wide HR data: median spray angle for lefty HRs is +26 deg (pull to
# RF), for righty HRs -24 deg (pull to LF) -- confirms real, well-known pull
# tendencies, not an assumed sign.
PULLED_CONTACT_QUALITY_CODES = (5, 6)  # Statcast launch_speed_angle codes for
# "solid contact" and "barrel" -- the two highest-quality-contact buckets.
# Barrel-only (6) was the exact subset the 2026-08-04 geometry check used to
# confirm the real Yankee Stadium short-porch effect (NYY lefty pulled-barrel
# HR rate 73.96% at home vs 62.61% elsewhere, #2 of 30 parks); including solid
# contact (5) too roughly doubles usable sample per park-hand cell (NYY lefty:
# 67-122 raw events/season barrel-only vs 112-194 combined) -- a deliberate
# precision/power tradeoff, NOT yet validated against barrel-only at the
# full-stack level.
PULLED_BARREL_PRIOR_PA = 500  # 2026-08-04: provisional, NOT validated/tuned.
# Raw per-season, per-park, per-hand sample here (roughly 100-200 events) is
# an order of magnitude thinner than even PARK_FACTOR_STAND_PRIOR_PA's already-
# thin cells (see that constant's own comment) -- reusing 5000 here would
# shrink nearly every park to dead neutral, defeating the whole point. 500 is
# a first-pass guess scaled to the real observed counts; needs a real
# held-out calibration check before being trusted, same as every other prior
# constant in this file.


def _spray_angle_deg(pa: pd.DataFrame) -> pd.Series:
    """Batter-facing spray angle in degrees via the standard Statcast hc_x/hc_y
    hit-coordinate transform (origin ~home plate, y toward the outfield) --
    validated 2026-08-04 against real league-wide HR data: median spray angle
    for lefty HRs is +26 deg (pull to RF), for righty HRs -24 deg (pull to LF),
    matching real, well-known pull tendencies. Confirms the sign convention
    (positive = right-field side, negative = left-field side, 0 = straight to
    center) is correct, not merely assumed."""
    x = pa["hc_x"] - 125.42
    y = 198.27 - pa["hc_y"]
    return np.degrees(np.arctan2(x, y))


def _pulled_quality_contact(pa: pd.DataFrame) -> pd.DataFrame:
    """Rows of `pa` that are (a) solid-contact-or-better (Statcast launch_speed_
    angle in PULLED_CONTACT_QUALITY_CODES) and (b) hit toward the batter's OWN
    pull side (spray angle past PULLED_CONTACT_SPRAY_THRESHOLD_DEG, sign per
    stand) -- the exact subset the 2026-08-04 geometry check used to confirm a
    real short-porch HR-conversion effect at Yankee Stadium that the plain,
    all-contact-types build_outcome_park_factors_by_stand/build_component_
    park_factors_by_stand couldn't see (both average over EVERY batted-ball
    type, diluting a real effect that's concentrated in this narrow slice)."""
    reg = pa.copy()
    reg["spray_angle"] = _spray_angle_deg(reg)
    # launch_speed_angle is a nullable Int64 (NA for non-batted-ball PAs like
    # strikeouts/walks) -- .isin() on it yields a nullable boolean Series
    # (NA stays NA, doesn't collapse to False), which raises "boolean value
    # of NA is ambiguous" when combined with a plain numpy bool array via
    # `&`. fillna(False) first so both sides are plain bool.
    quality = (reg["launch_speed_angle"].isin(PULLED_CONTACT_QUALITY_CODES) & reg["spray_angle"].notna()).fillna(False)
    pulled = np.where(
        reg["stand"] == "L",
        reg["spray_angle"] > PULLED_CONTACT_SPRAY_THRESHOLD_DEG,
        reg["spray_angle"] < -PULLED_CONTACT_SPRAY_THRESHOLD_DEG,
    )
    return reg[quality.to_numpy() & pulled].copy()


def all_pulled_quality_contact_at_park_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """Same shape/role as all_batters_at_park_by_stand (every batter, home AND
    visiting, split by stand) but restricted to the pulled-quality-contact
    subset (_pulled_quality_contact) and to the home_run outcome specifically
    -- the target quantity the 2026-08-04 geometry check measured directly
    (HR-conversion rate among pulled quality contact), not a general
    per-outcome rate table."""
    subset = _pulled_quality_contact(pa)
    subset["is_hr"] = (subset["outcome"] == "home_run").astype(int)
    g = subset.groupby(["season", "home_team", "stand"]).agg(
        at_pa=("is_hr", "size"), at_events=("is_hr", "sum")
    ).reset_index().rename(columns={"home_team": "team"})
    g["outcome"] = "home_run"
    return g


def build_pulled_barrel_park_factor_by_stand(pa: pd.DataFrame) -> pd.DataFrame:
    """Component-method, walk-forward, Bayesian-shrunk, mean-1.0-renormalized
    park factor for HR-conversion among PULLED QUALITY CONTACT specifically
    (see all_pulled_quality_contact_at_park_by_stand) -- same skeleton as
    build_component_park_factors_by_stand (at-park vs. league-wide-elsewhere,
    same-venue 3yr rolling for at-park, plain trailing-3yr for elsewhere, same
    shrinkage-then-renorm pattern), swapping in the much smaller PULLED_
    BARREL_PRIOR_PA (see its own comment) since this subset's raw per-cell
    sample is roughly an order of magnitude thinner than the plain HR-outcome
    cells build_component_park_factors_by_stand works with.

    2026-08-04: built in direct response to a real, geometry-validated finding
    (NYY lefty pulled-quality-contact HR rate 73.96% at home vs 62.61%
    elsewhere, #2 of 30 parks) that neither build_outcome_park_factors_by_
    stand nor build_component_park_factors_by_stand could detect, because
    both average over EVERY contact type -- this isolates the specific slice
    where the real effect lives. STILL EXPERIMENTAL/PROVISIONAL: not yet
    wired into game_simulator.py, pending (1) a held-out walk-forward
    calibration check and (2) the full-stack CRN-paired backtest scoped
    alongside this build -- do not consume this output for live predictions
    yet."""
    at_park = all_pulled_quality_contact_at_park_by_stand(pa)
    venue_lookup = _team_venue_lookup(sorted(pa["season"].unique().tolist()))
    at_park = at_park.merge(venue_lookup, on=["season", "team"], how="left")
    at_park = at_park.sort_values(["team", "stand", "outcome", "season"])

    league_totals = at_park.groupby(["season", "stand", "outcome"], as_index=False).agg(
        league_pa=("at_pa", "sum"), league_events=("at_events", "sum")
    )
    at_park = at_park.merge(league_totals, on=["season", "stand", "outcome"], how="left")
    at_park["elsewhere_pa"] = at_park["league_pa"] - at_park["at_pa"]
    at_park["elsewhere_events"] = at_park["league_events"] - at_park["at_events"]

    grp = at_park.groupby(["team", "stand", "outcome"], group_keys=False)
    at_roll_pa = grp.apply(lambda g: _same_venue_rolling_sum(g, "at_pa"), include_groups=False)
    at_roll_ev = grp.apply(lambda g: _same_venue_rolling_sum(g, "at_events"), include_groups=False)
    # plain trailing-3-season walk-forward sum (shift(1) before rolling(3)) --
    # not same-venue-restricted, matches build_component_park_factors_by_
    # stand's own "elsewhere has no single venue" reasoning.
    elsewhere_roll_pa = grp.apply(
        lambda g: g["elsewhere_pa"].shift(1).rolling(ROLLING_YEARS, min_periods=1).sum(), include_groups=False
    )
    elsewhere_roll_ev = grp.apply(
        lambda g: g["elsewhere_events"].shift(1).rolling(ROLLING_YEARS, min_periods=1).sum(), include_groups=False
    )

    season_totals = at_park.groupby(["outcome", "stand", "season"], as_index=False).agg(
        se_ev=("at_events", "sum"), se_pa=("at_pa", "sum")
    ).sort_values(["outcome", "stand", "season"])
    season_totals["cum_ev"] = season_totals.groupby(["outcome", "stand"])["se_ev"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["cum_pa"] = season_totals.groupby(["outcome", "stand"])["se_pa"].transform(
        lambda s: s.shift(1).cumsum()
    )
    season_totals["league_rate"] = season_totals["cum_ev"] / season_totals["cum_pa"]
    league_rate_lookup = season_totals.set_index(["outcome", "stand", "season"])["league_rate"]
    league_rate = pd.Series(
        pd.MultiIndex.from_frame(at_park[["outcome", "stand", "season"]]).map(league_rate_lookup),
        index=at_park.index,
    )

    at_park["at_rate_rolling"] = (at_roll_ev + PULLED_BARREL_PRIOR_PA * league_rate) / (
        at_roll_pa + PULLED_BARREL_PRIOR_PA
    )
    at_park["elsewhere_rate_rolling"] = (elsewhere_roll_ev + PULLED_BARREL_PRIOR_PA * league_rate) / (
        elsewhere_roll_pa + PULLED_BARREL_PRIOR_PA
    )
    raw_factor = at_park["at_rate_rolling"] / at_park["elsewhere_rate_rolling"]
    no_history = (at_roll_pa.fillna(0) == 0) | (elsewhere_roll_pa.fillna(0) == 0)
    group_mean = raw_factor.where(~no_history).groupby(
        [at_park["season"], at_park["outcome"], at_park["stand"]]
    ).transform("mean")
    at_park["park_factor"] = (raw_factor / group_mean).where(~no_history, 1.0)
    return at_park[["season", "team", "stand", "outcome", "park_factor"]]


HFA_PRIOR_PA = 20000  # pooled across all 30 teams (not a single team's PA volume like
                       # PARK_FACTOR_PRIOR_PA), so needs to be much larger for a comparable
                       # amount of shrinkage -- real home-field advantage is a small,
                       # single-digit-percent effect, and this guards against one walk-forward
                       # season's pooled noise producing an implausibly large multiplier.
HFA_CLIP_MIN, HFA_CLIP_MAX = 0.7, 1.4  # same defensive-clip discipline as every other
                                        # multiplicative factor in this project (weather.py,
                                        # expected_stats.py) -- not expected to bind given the
                                        # large prior above, but cheap insurance against a
                                        # pathological rare-category blowup (see park_factor's
                                        # own catcher_interf/triple_play history).


def build_hfa_factors(pa: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Walk-forward-safe (3-year rolling window over the seasons STRICTLY
    BEFORE each one, same convention as build_outcome_park_factors), LEAGUE-
    WIDE (pooled across all 30 teams, not per-team) home-field-advantage
    multiplier per outcome category.

    This measures a genuinely different quantity than build_outcome_park_factors,
    not a duplicate of it: that function's own final step re-normalizes every
    (season, outcome) group of 30 team-level home/road ratios to a population
    mean of 1.0 (a real, necessary fix for a right-skewed-estimator bias --
    see its own docstring). But if a UNIVERSAL home edge (crowd noise, travel
    fatigue for the visiting team, umpire tendency, batting-last strategic
    advantage -- independent of any one stadium's own dimensions/altitude/
    mound) inflates every team's own home/road ratio by roughly the same
    multiplicative amount, that normalization step silently cancels it out
    too, indistinguishable from the bias it exists to fix. This function
    instead pools ALL teams' home and road PAs together into a single
    league-wide home/road ratio per outcome -- since it's one ratio, not an
    average of 30 per-team ratios, there's no Jensen's-inequality skew to
    correct for, so no renormalization is applied here, and any real
    universal home edge survives intact. Missing/first-season lookups
    default to 1.0 (no adjustment)."""
    rates = team_home_road_outcome_rates(pa)
    pooled = rates.groupby(["season", "outcome"]).agg(
        home_pa=("home_pa", "sum"), home_events=("home_events", "sum"),
        road_pa=("road_pa", "sum"), road_events=("road_events", "sum"),
    ).reset_index().sort_values(["outcome", "season"])

    grp = pooled.groupby("outcome")
    home_roll_pa = grp["home_pa"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    home_roll_ev = grp["home_events"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    road_roll_pa = grp["road_pa"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    road_roll_ev = grp["road_events"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())

    # WALK-FORWARD league-rate anchor (task #160, 2026-07-26 correctness audit
    # fix -- same bug and same fix as build_outcome_park_factors above): this
    # previously pooled ALL seasons into one anchor reused everywhere,
    # contradicting this function's own "STRICTLY BEFORE" docstring claim.
    # `.transform` (not a separate merge) keeps this correctly aligned to
    # `pooled`'s own index, which home_roll_pa/home_roll_ev/etc. below also rely on.
    league_cum_ev = grp["home_events"].transform(lambda s: s.shift(1).cumsum())
    league_cum_pa = grp["home_pa"].transform(lambda s: s.shift(1).cumsum())
    league_rate = league_cum_ev / league_cum_pa

    home_rate_rolling = (home_roll_ev + HFA_PRIOR_PA * league_rate) / (home_roll_pa + HFA_PRIOR_PA)
    road_rate_rolling = (road_roll_ev + HFA_PRIOR_PA * league_rate) / (road_roll_pa + HFA_PRIOR_PA)
    pooled["hfa_factor"] = (home_rate_rolling / road_rate_rolling).clip(HFA_CLIP_MIN, HFA_CLIP_MAX).fillna(1.0)

    out: dict[int, dict[str, float]] = {}
    for season, g in pooled.groupby("season"):
        out[int(season)] = dict(zip(g["outcome"], g["hfa_factor"]))
    return out


if __name__ == "__main__":
    pf = build_park_factors(list(range(2023, 2027)))
    pf.to_parquet(DATA_PROCESSED / "park_factors.parquet", index=False)
    print("=== overall park factors, most recent season available per team (3-yr rolling) ===")
    latest = pf.sort_values("season").groupby("team").last()
    print(latest[["primary_venue", "park_factor"]].sort_values("park_factor", ascending=False).round(3).to_string())

    print("\n=== per-outcome park factors: home_run, most recent season (3-yr rolling) ===")
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    outcome_pf = build_outcome_park_factors(pa)
    outcome_pf.to_parquet(DATA_PROCESSED / "outcome_park_factors.parquet", index=False)
    hr_pf = outcome_pf[outcome_pf["outcome"] == "home_run"].sort_values("season").groupby("team").last()
    print(hr_pf[["park_factor"]].sort_values("park_factor", ascending=False).round(3).to_string())

    print("\n=== league-wide home-field-advantage factor, most recent season (3-yr rolling) ===")
    hfa = build_hfa_factors(pa)
    latest_season = max(hfa)
    hfa_series = pd.Series(hfa[latest_season]).sort_values(ascending=False)
    print(hfa_series.round(4).to_string())
