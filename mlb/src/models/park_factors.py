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

import pandas as pd

from src.models.true_talent import STABILIZATION_PA
from src.utils.paths import DATA_RAW, DATA_PROCESSED

ROLLING_YEARS = 3
OUTCOMES = list(STABILIZATION_PA.keys())


def team_home_road_runs(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): runs/game at home vs. on the road."""
    reg = schedule[(schedule["game_type"] == "R") & (schedule["status"] == "Final")].copy()
    reg["total_runs"] = reg["home_score"] + reg["away_score"]

    home = reg.groupby(["season", "home_team"]).agg(
        home_runs_per_game=("total_runs", "mean"), home_games=("total_runs", "size"),
        primary_venue=("venue_name", lambda s: s.mode().iloc[0]),
    ).reset_index().rename(columns={"home_team": "team"})
    road = reg.groupby(["season", "away_team"]).agg(
        road_runs_per_game=("total_runs", "mean"), road_games=("total_runs", "size"),
    ).reset_index().rename(columns={"away_team": "team"})

    return home.merge(road, on=["season", "team"], how="inner")


def build_park_factors(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = DATA_RAW / f"schedule_{season}.parquet"
        if not path.exists():
            continue
        sched = pd.read_parquet(path)
        frames.append(team_home_road_runs(sched))
    all_seasons = pd.concat(frames, ignore_index=True).sort_values(["team", "season"])

    # 3-year rolling window over seasons STRICTLY BEFORE each one (shift(1)
    # before rolling) -- a season's own data must never contribute to its own
    # park factor.
    all_seasons["home_runs_sum"] = all_seasons.groupby("team")["home_runs_per_game"].transform(
        lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).mean()
    )
    all_seasons["road_runs_sum"] = all_seasons.groupby("team")["road_runs_per_game"].transform(
        lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).mean()
    )
    all_seasons["park_factor"] = all_seasons["home_runs_sum"] / all_seasons["road_runs_sum"]
    return all_seasons


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
    rates = rates.sort_values(["team", "outcome", "season"])
    grp = rates.groupby(["team", "outcome"])
    # shift(1) first: a season's own data must never contribute to its own park factor
    home_roll_pa = grp["home_pa"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    home_roll_ev = grp["home_events"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    road_roll_pa = grp["road_pa"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())
    road_roll_ev = grp["road_events"].transform(lambda s: s.shift(1).rolling(ROLLING_YEARS, min_periods=1).sum())

    league_rate_by_outcome = rates.groupby("outcome").apply(
        lambda d: d["home_events"].sum() / d["home_pa"].sum(), include_groups=False
    )
    league_rate = rates["outcome"].map(league_rate_by_outcome)

    rates["home_rate_rolling"] = (home_roll_ev + PARK_FACTOR_PRIOR_PA * league_rate) / (home_roll_pa + PARK_FACTOR_PRIOR_PA)
    rates["road_rate_rolling"] = (road_roll_ev + PARK_FACTOR_PRIOR_PA * league_rate) / (road_roll_pa + PARK_FACTOR_PRIOR_PA)
    raw_factor = rates["home_rate_rolling"] / rates["road_rate_rolling"]
    # normalize each (season, outcome) group to a population mean of 1.0 --
    # only over teams with a real (non-NaN, i.e. prior-data-available) factor
    group_mean = raw_factor.groupby([rates["season"], rates["outcome"]]).transform("mean")
    rates["park_factor"] = raw_factor / group_mean
    return rates[["season", "team", "outcome", "park_factor"]]


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

    league_rate_by_outcome = pooled.groupby("outcome").apply(
        lambda d: d["home_events"].sum() / d["home_pa"].sum(), include_groups=False
    )
    league_rate = pooled["outcome"].map(league_rate_by_outcome)

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
