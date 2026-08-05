"""The deliverable the whole project was built toward: not just a final
score, but a full per-at-bat Monte Carlo simulation of one specific game,
aggregated into per-batter, per-pitcher, per-inning, and per-game prop
probabilities -- the same "predict every inning, every player, every at bat"
system originally asked for.

Uses the PREDICTIVE (live-usable) path throughout, not the oracle backtest:
each team's real probable lineup/starter, park factors, this game's actual
weather, a roster-aware sampled bullpen (starter's own walk-forward projected
innings, then a SPECIFIC reliever sampled per inning -- weighted by recent
usage and rest-day availability for this exact date, resampled fresh every
trial -- see bullpen.py's sample_bullpen_plan), and dynamic
blowout/position-player-pitching substitution once a team is down 8+ runs in
the 8th inning or later (see blowout.py). This is the same machinery
validated in validate_predictive_bullpen.py and validate_game_simulator.py,
just run once per game at higher trial count and with full event-level
tracking turned on (see game_simulator.py's `events` parameter) instead of
only the final score.

RBI is approximated as runs scored on that specific PA (our transition
table's own runs_scored output) -- close to but not identical to official
MLB RBI rules in every edge case, a reasonable simplification given this
project's PA-level, not out-by-out, event granularity. Task #154 (2026-07-25
prop-rules audit) tightened this: `RBI_EXCLUDED_OUTCOMES` below now excludes
`double_play` (Official Rule 9.04(b)(1) -- no RBI on a grounded-into force
double play, unambiguous, no scorer's-judgment involved) and `field_error`
(Rule 9.04(b)(2) -- no RBI when the run's scoring is a direct result of the
same misplay, UNLESS the scorer judges the run would have scored regardless;
our data has no way to make that specific judgment call, so this defaults to
DENY -- a conservative approximation that may occasionally under-credit a
real RBI, never over-credit one). Quantified on real 2025 data before
fixing: double_play removed 68 of 21,594 season-total RBI-equivalent runs
(0.31%), field_error another 164 (1.07% combined) -- individually small
league-wide, but a real, systematic overstatement for any specific batter's
own RBI prop. `fielders_choice` and `sac_bunt`/`sac_fly` are deliberately
NOT excluded -- real official scoring generally DOES credit RBI on those
(a fielder's choice run is credited unless it was itself a double-play
situation, already carved out separately), so excluding them would make the
approximation worse, not better. Stolen bases, defensive plays, and pitcher
win/loss remain out of scope -- this only covers what the underlying
simulation actually models.
"""

import numpy as np
import pandas as pd

from src.models.base_out_transitions import TransitionTable
from src.models.blowout import build_position_player_profile
from src.models.bullpen import (
    build_all_appearance_log,
    build_bullpen_snapshot,
    build_closer_appearance_log,
    build_expected_starter_innings,
    build_live_expected_innings,
    build_relief_appearance_log,
    build_team_bullpen_roster,
    identify_closer,
    nearest_prior_bullpen,
    nearest_prior_pitcher_snapshot,
    sample_bullpen_plan,
)
from src.models.bullpen_usage_policy import attach_margin, build_starter_hook_events, fit_hook_policy
from src.models.game_simulator import (
    OUTCOMES,
    GameSimulator,
    build_league_rates_by_season,
    build_state_factors_by_season,
    build_wide_platoon_multipliers,
    build_wide_pregame_rates,
)
from src.models.catcher_framing import (
    build_catcher_appearance_log,
    build_catcher_framing_factors_by_season,
    identify_starting_catcher,
    resolve_catcher_factor,
)
from src.models.expected_stats import (
    build_pregame_bat_speed,
    build_pregame_sprint_speed,
    load_bat_speed_by_season,
    player_game_bacon_gb_snapshot,
    player_game_barrel_snapshot,
    player_game_gb_fb_rate_snapshot,
    player_game_groundball_rate_snapshot,
    player_game_pulled_air_snapshot,
    player_game_xbacon_snapshot,
)
from src.models.park_factors import build_outcome_park_factors, build_hfa_factors
from src.models.spray import build_pull_rate_by_season, build_pull_rate_snapshot, resolve_batter_pull_tercile
from src.models.spray import attach_pull_tercile_column
from src.models.ttop import build_ttop_factors_by_season
from src.models.umpire_factor import build_umpire_factors_by_season, resolve_live_umpire_factor
from src.models.defense_factor import resolve_defense_factor, team_game_defense_snapshot
from src.models.weather import (
    attach_weather_bucket,
    bucket_weather,
    build_venue_weather_norms,
    build_weather_factors_by_season,
    park_relative_weather_factors,
)
from src.models.weather_forecast import build_historical_game_buckets, resolve_weather_distribution, sample_weather_bucket
from src.models.true_talent import build_debut_rate
from src.models.validate_game_simulator import build_profile, player_game_snapshot, predominant_hand, SHOCK_SIGMA
from src.models.park_geometry import (
    load_park_dimensions, calibrate_scale_factor, build_batter_trajectory_history,
    batter_geometry_hr_factor, build_ratio_outcome_table, batter_geometry_xbh_factor,
)
from src.utils.paths import DATA_PROCESSED, DATA_RAW

HIT_OUTCOMES = {"single", "double", "triple", "home_run"}
TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
N_TRIALS = 1000

# Task #154 (2026-07-25 prop-rules audit) -- see module docstring for the
# exact rules and real-data quantification (0.31%/1.07% of season-total
# RBI-equivalent runs). Used both here (the live prop) and in
# validate_prop_calibration.py's "real" ground-truth RBI label, so any
# future calibration re-check stays apples-to-apples against the SAME
# definition of RBI, not a stale one on one side.
#
# "strikeout" added (task #160, 2026-07-26 correctness audit): a run scored
# during a strikeout PA can only come from a wild pitch, passed ball, or
# defensive indifference -- no batted ball occurred at all, so Official
# Rule 9.04 never credits an RBI here, unambiguously (no scorer's judgment
# involved, unlike the fielders_choice case this project already
# deliberately leaves alone). Quantified before fixing: 379 real strikeout
# PAs with runs_scored>0 across 2023-2026 (116/109/91/63 by season) --
# comparable magnitude to double_play's already-fixed 68-of-21594 (0.43% of
# 2025's 21594 season-total runs), the same bug class task #154 fixed,
# simply missed for this outcome at the time.
RBI_EXCLUDED_OUTCOMES = {"double_play", "field_error", "strikeout"}

HOOK_TABLE_FIT_SEASONS = {2023, 2024}  # matches task #145's own fit period exactly.


def build_hook_table(pa: pd.DataFrame) -> dict:
    """Task #144/#145 starter-hook-frailty mechanism, deployed PROPS-PATH
    ONLY (2026-07-24 starter-outs calibration check): a game-level SU/Brier
    factorial (sec 11.23) found this mechanism carries no measurable
    game-level win-probability value -- but that was never its claim. Its
    claim was always about the distribution of starter outs/innings pitched,
    the quantity props.py actually prices, and THAT check (real 2025 games,
    hook-enabled vs. deterministic-cutoff baseline, segmented by pregame
    run-value tercile) found a large, statistically real win: aggregate CRPS
    0.747 vs. 0.908 (baseline), CI excludes zero, and the win holds in EVERY
    run-value tercile including the shelled-tail (worst-pitchers) slice where
    it was designed to matter most (CRPS delta -0.227, the largest of the
    three). The three props-priced quantities confirm it: P(exits by the
    4th) error drops from 0.067 (baseline) to 0.011 (hook) in aggregate, and
    from 0.068 to 0.003 in the worst-pitchers tercile specifically -- the
    deterministic baseline has essentially ZERO early-exit tail by
    construction (a fixed cutoff can't produce a start that ends early), so
    the shelled-start scenario a prop bettor actually cares about was
    entirely unmodeled before this. Deployed here with the EXACT SHOCK_SIGMA
    precedent: a distribution-realism/variance-shape fix, with NO
    game-level win-probability claim (that lens stays a confirmed null, see
    sec 11.23) -- kept OFF in validate_game_simulator.py/
    validate_oracle_vs_predictive.py's own game-level oracle backtest, which
    stays byte-identical to before. Reliever TIER selection (the other half
    of task #144 step 4) is NOT deployed here -- its own specific claim
    (reliever-inning attribution/prop calibration) has not yet been checked
    on this same lens, so it stays dormant pending that separate test."""
    pa_m = attach_margin(pa)
    events = build_starter_hook_events(pa_m)
    fit_events = events[events["season"].isin(HOOK_TABLE_FIT_SEASONS)]
    hook_table_df = fit_hook_policy(fit_events)
    return hook_table_df.set_index(["inning_b", "pa_count_b", "runs_allowed_b", "margin_b"])["p_exit"].to_dict()


def build_generic_debut_profile(pa: pd.DataFrame, player_col: str, season: int,
                                 league_rates_this_season: dict[str, float]) -> dict:
    """Task #149: a LAST-RESORT synthetic profile for a player with ZERO
    cached history anywhere -- not even a debut-cohort snapshot row exists
    for them (a true, brand-new-to-our-data MLB debut). Confirmed as a REAL,
    not hypothetical, failure mode via task #150's postseason dress rehearsal:
    a September call-up outfielder started two 2025 Wild Card games with no
    prior MLB PA in our data at all, causing generate_game_props' own
    missing-snapshot ValueError.

    Reuses true_talent.build_debut_rate (task #119's already-validated,
    walk-forward-safe empirical debut-cohort rate per outcome -- real MLB
    debut cohorts perform BELOW league average, matching replacement-level
    expectations) rather than a guessed quartile cutoff. Note this is a
    narrower reuse than task #119's own rejected use_debut_prior flag: that
    flag would have applied the debut rate as the ONGOING cold-start prior
    for every player inside the regular Marcel blend (tested full-stack,
    found NULL, not deployed) -- this is a last-resort fallback ONLY for a
    player who would otherwise crash the whole game's prop generation,
    which was never tested and is a different, much narrower claim.

    effective_n=0 for every outcome (a real, already-handled edge case in
    true_talent.sample_posterior_rate: "a fully cold-started player with
    zero weight" -- there is nothing to estimate a confidence interval
    from). hand defaults 'R' (genuinely unknown) and platoon multipliers
    stay neutral -- no signal exists to differentiate either."""
    rates = {o: build_debut_rate(pa, player_col, o, season, league_rates_this_season[o]) for o in OUTCOMES}
    neutral = {o: 1.0 for o in OUTCOMES}
    effective_n = {o: 0.0 for o in OUTCOMES}
    return {"rates": rates, "hand": "R", "same_mult": neutral, "opp_mult": neutral,
            "pull_tercile": None, "effective_n": effective_n}


def build_pregame_context(pa: pd.DataFrame, geometry_hr_enabled: bool = False,
                           geometry_xbh_enabled: bool = False) -> dict:
    """Every walk-forward table the props generator needs, built once and
    reused across as many games as you want to generate props for -- the
    expensive part; generating props for one more game after this is cheap.

    geometry_hr_enabled/geometry_xbh_enabled (2026-08-05, EXPERIMENTAL/
    opt-in -- see park_geometry.py, MODEL_DOCUMENTATION.md sec 11.44): the
    batter-level park-geometry HR/doubles/triples factors were built and
    CRN-tested against the ORACLE backtest (validate_game_simulator.py) but
    never wired into this live/predictive props path at all -- this is
    that wiring, so validate_prop_calibration.py can check whether a
    real-but-full-stack-immaterial signal on SU/Brier is ALSO immaterial on
    the actual prop probabilities it's meant to move (HR/TB props). Both
    False (the default, every existing caller) is a byte-for-byte no-op --
    geometry_dims/geometry_history/geometry_ratio_table stay None below,
    generate_game_props's own geometry lookups all short-circuit to None."""
    geometry_dims, geometry_history, geometry_ratio_table = None, None, None
    if geometry_hr_enabled or geometry_xbh_enabled:
        geometry_dims = load_park_dimensions()
        geometry_scale = calibrate_scale_factor(pa, geometry_dims)["scale"]
        geometry_history = build_batter_trajectory_history(pa, geometry_scale)
        if geometry_xbh_enabled:
            geometry_ratio_table = build_ratio_outcome_table(pa, geometry_dims, geometry_scale)
    # park-neutralize batter/pitcher rates BEFORE this project's own per-game
    # park_factors multiplier acts on them below (see park_factors_wide) --
    # otherwise a park effect gets double-counted (true_talent.build_pregame_rates).
    park_factors_long = build_outcome_park_factors(pa)
    batter_wide = build_wide_pregame_rates(pa, "batter", park_factors_long)
    pitcher_wide = build_wide_pregame_rates(pa, "pitcher", park_factors_long)
    all_schedules = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA_RAW.glob("schedule_*.parquet"))], ignore_index=True
    )
    all_lineups = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA_RAW.glob("lineups_*.parquet"))], ignore_index=True
    )
    # defense composite is walk-forward PER SEASON (uses that season's own
    # prior-season OAA) -- team_game_defense_snapshot doesn't filter by season
    # itself, so each call is restricted to that season's own game_pks (via
    # the PA table's own season tagging) before building, else a game_pk from
    # a DIFFERENT season would silently get the wrong season's walk-forward
    # OAA. Concatenated afterward -- game_pk is globally unique across
    # seasons, so no collision risk.
    defense_snap_frames = []
    for season in sorted(pa["season"].unique()):
        season_game_pks = pa.loc[pa["season"] == season, "game_pk"].unique()
        season_lineups = all_lineups[all_lineups["game_pk"].isin(season_game_pks)]
        season_schedule = all_schedules[all_schedules["game_pk"].isin(season_game_pks)]
        defense_snap_frames.append(team_game_defense_snapshot(season_lineups, season_schedule, season))
    defense_snap = pd.concat(defense_snap_frames, ignore_index=True)
    pull_rate_pa = build_pull_rate_by_season(pa)
    pull_rate_snapshot = build_pull_rate_snapshot(pa, pull_rate_pa)
    pa_with_weather = attach_weather_bucket(pa, all_schedules)
    pa_with_weather = attach_pull_tercile_column(pa_with_weather, pull_rate_pa)
    weather_factors_by_season = build_weather_factors_by_season(pa_with_weather)
    historical_game_buckets = build_historical_game_buckets(all_schedules)
    pitcher_snap = player_game_snapshot(pitcher_wide, "pitcher")
    batter_snap = player_game_snapshot(batter_wide, "batter")
    xbacon_snap = player_game_xbacon_snapshot(pa)
    barrel_snap = player_game_barrel_snapshot(pa)
    pulled_air_snap = player_game_pulled_air_snapshot(pa)
    bacon_gb_snap = player_game_bacon_gb_snapshot(pa)
    gb_rate_snap = player_game_groundball_rate_snapshot(pa)
    gbfb_pitcher_snap = player_game_gb_fb_rate_snapshot(pa)  # task #120, GB/FB pitcher HR-share -- KEPT, real full-stack win
    # season-level (not per-game), so precompute for every season present rather
    # than per-game like the other expected-stats snapshots above.
    sprint_speed_by_season = {season: build_pregame_sprint_speed(season) for season in pa["season"].unique()}
    # season-level (not per-game), same convention as sprint_speed_by_season above.
    bat_speed_raw = load_bat_speed_by_season(sorted(pa["season"].unique()))
    bat_speed_by_season = {season: build_pregame_bat_speed(bat_speed_raw, season) for season in pa["season"].unique()}
    # season-level (not per-game), same convention as sprint_speed_by_season above.
    game_dates = pa[["game_pk", "game_date"]].drop_duplicates()
    # Task #149: precomputed once per (player_col, season) here -- cheap relative to
    # everything else in this function, and every game needing this fallback in a given
    # season shares the identical profile anyway (there's no per-player signal to give
    # a true zero-history debut beyond "which season is it").
    league_rates_by_season = build_league_rates_by_season(pa)
    generic_debut_batter_profile_by_season = {
        season: build_generic_debut_profile(pa, "batter", season, league_rates_by_season[season])
        for season in pa["season"].unique()
    }
    generic_debut_pitcher_profile_by_season = {
        season: build_generic_debut_profile(pa, "pitcher", season, league_rates_by_season[season])
        for season in pa["season"].unique()
    }
    return dict(
        league_rates=league_rates_by_season,
        generic_debut_batter_profile_by_season=generic_debut_batter_profile_by_season,
        generic_debut_pitcher_profile_by_season=generic_debut_pitcher_profile_by_season,
        state_factors=build_state_factors_by_season(pa),
        ttop_factors=build_ttop_factors_by_season(pa),
        pull_rate_snapshot=pull_rate_snapshot,
        batter_platoon=build_wide_platoon_multipliers(pa, "batter").set_index(["batter", "season"]),
        pitcher_platoon=build_wide_platoon_multipliers(pa, "pitcher").set_index(["pitcher", "season"]),
        batter_hand=predominant_hand(pa, "batter"),
        pitcher_hand=predominant_hand(pa, "pitcher"),
        park_factors_wide=park_factors_long.pivot(
            index=["team", "season"], columns="outcome", values="park_factor"
        ),
        hfa_factors_by_season=build_hfa_factors(pa),
        weather_factors_by_season=weather_factors_by_season,
        # per-venue climatological expected factors (finding M10) -- the
        # divisor that makes applied weather PARK-RELATIVE, so a park's
        # average climate stays only in its park factor
        venue_weather_norms=build_venue_weather_norms(historical_game_buckets, weather_factors_by_season),
        catcher_factors_by_season=build_catcher_framing_factors_by_season(sorted(pa["season"].unique())),
        catcher_log=build_catcher_appearance_log(pa),
        umpire_factors_by_season=build_umpire_factors_by_season(sorted(pa["season"].unique())),
        # venue_name included (finding M10): the weather-norm division needs
        # to know WHICH park each historical/live game_pk is in
        game_weather=all_schedules[["game_pk", "venue_name", "weather_condition", "weather_temp", "weather_wind"]]
        .drop_duplicates("game_pk").set_index("game_pk"),
        historical_game_buckets=historical_game_buckets,
        bullpen_snap=build_bullpen_snapshot(pa, park_factors_long),
        expected_innings=build_expected_starter_innings(pa).set_index(["pitcher", "game_pk"])["expected_innings"],
        # live-path companion (2026-08-03 audit, finding C4): the per-start
        # table above only ever hits for HISTORICAL game_pks -- a live game's
        # lookup always missed and fell back to a constant 5.4 for every
        # starter, silently disabling the whole per-pitcher innings model in
        # production while backtests exercised it.
        expected_innings_live=build_live_expected_innings(pa),
        hook_table=build_hook_table(pa),
        relief_log=build_relief_appearance_log(pa),
        all_appearance_log=build_all_appearance_log(pa),
        closer_log=build_closer_appearance_log(pa),
        blowout_profile=build_position_player_profile(pa),
        batter_snap=batter_snap,
        batter_snap_dates=batter_snap.merge(game_dates, on="game_pk"),
        pitcher_snap=pitcher_snap,
        pitcher_snap_dates=pitcher_snap.merge(game_dates, on="game_pk"),
        xbacon_snap=xbacon_snap,
        xbacon_snap_dates=xbacon_snap.merge(game_dates, on="game_pk"),
        barrel_snap=barrel_snap,
        barrel_snap_dates=barrel_snap.merge(game_dates, on="game_pk"),
        pulled_air_snap=pulled_air_snap,
        pulled_air_snap_dates=pulled_air_snap.merge(game_dates, on="game_pk"),
        bacon_gb_snap=bacon_gb_snap,
        bacon_gb_snap_dates=bacon_gb_snap.merge(game_dates, on="game_pk"),
        gb_rate_snap=gb_rate_snap,
        gb_rate_snap_dates=gb_rate_snap.merge(game_dates, on="game_pk"),
        gbfb_pitcher_snap=gbfb_pitcher_snap,
        gbfb_pitcher_snap_dates=gbfb_pitcher_snap.merge(game_dates, on="game_pk"),
        sprint_speed_by_season=sprint_speed_by_season,
        bat_speed_by_season=bat_speed_by_season,
        # defense_snap already has its own team/game_date columns (keyed by
        # (game_pk, team), not by an individual player) -- no separate
        # "_dates" merge needed, unlike the player-keyed snapshots above; it's
        # already in the exact shape nearest_prior_bullpen expects.
        defense_snap=defense_snap,
        transitions=TransitionTable(pa),
        geometry_dims=geometry_dims, geometry_history=geometry_history,
        geometry_ratio_table=geometry_ratio_table,
    )


# Outcome categories surfaced in the debug/factors panel -- a representative
# subset of the full 16-category OUTCOMES list (see true_talent.py), not all
# of them: these are the ones a reader actually wants to see ("how often does
# this lineup homer/strike out"), not every fielding-detail category.
_DEBUG_OUTCOMES = ("home_run", "single", "double", "triple", "walk", "strikeout")


def _lineup_summary(lineup: list[dict]) -> dict:
    """Average FINAL per-PA rate (post-platoon, post-contact-quality/bat-speed,
    post-age-adjustment -- whatever build_profile already baked in) across the
    9 starters, as a simple 'how strong is this lineup' proxy. Reads only
    values build_profile already computed; no new modeling here."""
    return {o: float(np.mean([p["rates"].get(o, 0.0) for p in lineup])) for o in _DEBUG_OUTCOMES}


def _pitcher_summary(pitcher: dict) -> dict:
    """Same idea as _lineup_summary but for a single starter's own final
    per-PA-allowed rates."""
    return {o: float(pitcher["rates"].get(o, 0.0)) for o in _DEBUG_OUTCOMES}


def _weather_headline(weather_factors: dict | None, outcome: str) -> float | None:
    """A single representative scalar for the debug panel. weather_factors
    is keyed by '{batter_stand}_{pull_tercile}' (each batter is looked up by
    their OWN bucket when the simulator applies it, see combine_matchup_
    distribution's caller) -- not a flat {outcome: factor} dict like park/
    hfa -- so there's no single 'the' weather factor for this game. Averages
    across every bucket as a representative magnitude to display; not itself
    a value the model ever computes or uses directly."""
    if not weather_factors:
        return None
    vals = [bucket.get(outcome) for bucket in weather_factors.values() if bucket]
    return float(np.mean(vals)) if vals else None


def generate_game_props(ctx: dict, season: int, game_pk: int, home_team: str, away_team: str, game_date: str,
                         home_ids: list, away_ids: list, home_pitcher_id: int, away_pitcher_id: int,
                         venue_name: str | None = None,
                         n_trials: int = N_TRIALS, rng: np.random.Generator | None = None,
                         postseason: bool = False) -> dict:
    """Run n_trials full-game Monte Carlo simulations (predictive bullpen,
    this game's real park + weather) and return aggregated prop
    probabilities. home_ids/away_ids: 9 batter IDs in real batting order.
    venue_name: needed only when this game's real weather isn't posted yet --
    used to resolve a forecast/climatological fallback (see weather_forecast.py).

    postseason: real MLB rule -- no automatic extra-innings runner and no
    position-player-pitching substitution in the postseason (see
    GameSimulator.simulate_game's own docstring). Default False (every
    existing caller) is byte-for-byte identical to before this parameter
    existed; the caller derives it from the schedule's own game_type field
    (see generate_daily_props.py)."""
    rng = rng or np.random.default_rng()
    batter_snap = ctx["batter_snap"][ctx["batter_snap"]["game_pk"] == game_pk].set_index("batter")
    pitcher_snap = ctx["pitcher_snap"][ctx["pitcher_snap"]["game_pk"] == game_pk].set_index("pitcher")
    xbacon_snap = ctx["xbacon_snap"][ctx["xbacon_snap"]["game_pk"] == game_pk].set_index("batter")
    barrel_snap = ctx["barrel_snap"][ctx["barrel_snap"]["game_pk"] == game_pk].set_index("batter")
    pulled_air_snap = ctx["pulled_air_snap"][ctx["pulled_air_snap"]["game_pk"] == game_pk].set_index("batter")
    bacon_gb_snap = ctx["bacon_gb_snap"][ctx["bacon_gb_snap"]["game_pk"] == game_pk].set_index("batter")
    gb_rate_snap = ctx["gb_rate_snap"][ctx["gb_rate_snap"]["game_pk"] == game_pk].set_index("batter")
    gbfb_pitcher_snap = ctx["gbfb_pitcher_snap"][ctx["gbfb_pitcher_snap"]["game_pk"] == game_pk].set_index("pitcher")
    speed_snap = ctx["sprint_speed_by_season"].get(season, pd.Series(dtype=float))
    bat_speed_snap = ctx["bat_speed_by_season"].get(season, pd.Series(dtype=float))
    # Task #149: pids that had ZERO cached history anywhere (not even a debut-cohort
    # snapshot) and fell back to build_generic_debut_profile -- surfaced in the
    # returned dict (props["debut_fallback_pids"]) so the caller can flag it, same
    # transparency spirit as home_lineup_source/away_lineup_source.
    debut_fallback_pids: list = []

    def batter_profile(pid):
        # exact game_pk match works for a BACKTEST (a historical game_pk
        # always has real PA data for its actual participants); a genuinely
        # future, never-played game_pk never will, so fall back to this
        # batter's own most recent pregame snapshot as of this date.
        if pid in batter_snap.index:
            row = batter_snap.loc[pid]
        else:
            row = nearest_prior_pitcher_snapshot(ctx["batter_snap_dates"], pid, game_date, game_pk)
            if row is None:
                # a TRUE debut with zero cached PA history anywhere -- see
                # build_generic_debut_profile's docstring (task #149, a real
                # failure mode confirmed via task #150's postseason dress
                # rehearsal, not hypothetical). Always non-None (build_debut_rate
                # itself falls back to the plain league rate when no debut
                # cohort has been observed yet), so this never re-raises.
                debut_fallback_pids.append(pid)
                return ctx["generic_debut_batter_profile_by_season"].get(season)
        key = (pid, season)
        prow = ctx["batter_platoon"].loc[key] if key in ctx["batter_platoon"].index else None
        hand = ctx["batter_hand"].get(pid, "R")
        tercile = resolve_batter_pull_tercile(ctx["pull_rate_snapshot"], pid, hand, game_date, game_pk)
        if pid in xbacon_snap.index:
            xbacon = xbacon_snap.loc[pid, "pregame_xbacon"]
        else:
            xrow = nearest_prior_pitcher_snapshot(ctx["xbacon_snap_dates"], pid, game_date, game_pk)
            xbacon = xrow["pregame_xbacon"] if xrow is not None else None
        if pid in barrel_snap.index:
            barrel = barrel_snap.loc[pid, "pregame_barrel_rate"]
        else:
            brow = nearest_prior_pitcher_snapshot(ctx["barrel_snap_dates"], pid, game_date, game_pk)
            barrel = brow["pregame_barrel_rate"] if brow is not None else None
        if pid in pulled_air_snap.index:
            pulled_air = pulled_air_snap.loc[pid, "pregame_pulled_air_rate"]
        else:
            parow = nearest_prior_pitcher_snapshot(ctx["pulled_air_snap_dates"], pid, game_date, game_pk)
            pulled_air = parow["pregame_pulled_air_rate"] if parow is not None else None
        if pid in bacon_gb_snap.index:
            bacon_gb = bacon_gb_snap.loc[pid, "pregame_bacon_gb"]
        else:
            bgrow = nearest_prior_pitcher_snapshot(ctx["bacon_gb_snap_dates"], pid, game_date, game_pk)
            bacon_gb = bgrow["pregame_bacon_gb"] if bgrow is not None else None
        if pid in gb_rate_snap.index:
            gb_rate = gb_rate_snap.loc[pid, "pregame_gb_rate"]
        else:
            gbrow = nearest_prior_pitcher_snapshot(ctx["gb_rate_snap_dates"], pid, game_date, game_pk)
            gb_rate = gbrow["pregame_gb_rate"] if gbrow is not None else None
        sprint_speed = speed_snap.loc[pid] if pid in speed_snap.index else None
        bat_speed = bat_speed_snap.loc[pid] if pid in bat_speed_snap.index else None
        # park-geometry factors (2026-08-05, EXPERIMENTAL/opt-in -- see
        # build_pregame_context's own docstring): None on both counts
        # (ctx["geometry_history"] absent, i.e. build_pregame_context's
        # flags were both False) is a byte-for-byte no-op, same as every
        # other optional signal in this function.
        geom_factor = None
        if ctx.get("geometry_history") is not None:
            geom_factor = batter_geometry_hr_factor(
                ctx["geometry_history"], pid, season, home_team, ctx["geometry_dims"])
        geom_2b, geom_3b = None, None
        if ctx.get("geometry_history") is not None and ctx.get("geometry_ratio_table") is not None:
            geom_2b = batter_geometry_xbh_factor(
                ctx["geometry_history"], pid, season, home_team, ctx["geometry_dims"],
                ctx["geometry_ratio_table"], "double")
            geom_3b = batter_geometry_xbh_factor(
                ctx["geometry_history"], pid, season, home_team, ctx["geometry_dims"],
                ctx["geometry_ratio_table"], "triple")
        return build_profile(row, prow, hand, pull_tercile=tercile, pregame_xbacon=xbacon, pregame_barrel_rate=barrel,
                              pregame_bacon_gb=bacon_gb, pregame_sprint_speed=sprint_speed, pregame_gb_rate=gb_rate,
                              geometry_hr_factor=geom_factor, geometry_double_factor=geom_2b,
                              geometry_triple_factor=geom_3b,
                              pregame_bat_speed=bat_speed, pregame_pulled_air_rate=pulled_air)

    def _pitcher_gb_fb(pid):
        if pid in gbfb_pitcher_snap.index:
            row = gbfb_pitcher_snap.loc[pid]
            return row["pregame_gb_rate_pitcher"], row["pregame_fb_rate_pitcher"]
        row = nearest_prior_pitcher_snapshot(ctx["gbfb_pitcher_snap_dates"], pid, game_date, game_pk)
        if row is None:
            return None, None
        return row["pregame_gb_rate_pitcher"], row["pregame_fb_rate_pitcher"]

    def pitcher_profile(pid):
        if pid in pitcher_snap.index:
            row = pitcher_snap.loc[pid]
        else:
            row = nearest_prior_pitcher_snapshot(ctx["pitcher_snap_dates"], pid, game_date, game_pk)
            if row is None:
                # true debut, zero cached history -- see batter_profile's identical
                # fallback above and build_generic_debut_profile's docstring.
                debut_fallback_pids.append(pid)
                return ctx["generic_debut_pitcher_profile_by_season"].get(season)
        key = (pid, season)
        prow = ctx["pitcher_platoon"].loc[key] if key in ctx["pitcher_platoon"].index else None
        gb_p, fb_p = _pitcher_gb_fb(pid)
        return build_profile(row, prow, ctx["pitcher_hand"].get(pid, "R"),
                              pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p)

    def roster_pitcher_profile(pid):
        # a sampled roster reliever may not have pitched in THIS exact game
        # (unlike the two starters), so fall back to their own most recent
        # pregame snapshot as of this date -- see nearest_prior_pitcher_snapshot.
        if pid in pitcher_snap.index:
            row = pitcher_snap.loc[pid]
        else:
            row = nearest_prior_pitcher_snapshot(ctx["pitcher_snap_dates"], pid, game_date, game_pk)
            if row is None:
                return None
        key = (pid, season)
        prow = ctx["pitcher_platoon"].loc[key] if key in ctx["pitcher_platoon"].index else None
        gb_p, fb_p = _pitcher_gb_fb(pid)
        return build_profile(row, prow, ctx["pitcher_hand"].get(pid, "R"),
                              pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p)

    def _stamp_pid(prof, pid, apply_ttop=True):
        # Per-PA pitcher attribution (2026-08-03 audit, finding M15): every
        # pitcher profile the simulator can ever use in this path carries its
        # own real id, so simulate_half_inning can stamp WHO threw each PA at
        # the moment it happens. Copies rather than mutates -- the generic
        # debut profile is a shared ctx object served to multiple pids.
        # apply_ttop=False for RELIEVERS (finding M4): the TTOP table is fit
        # within-pitcher on starter PAs; a reliever's rates are already
        # first-pass rates, so the simulator skips the factor for him.
        return None if prof is None else {**prof, "pid": pid, "apply_ttop": apply_ttop}

    home_lineup = [batter_profile(pid) for pid in home_ids]
    away_lineup = [batter_profile(pid) for pid in away_ids]
    home_pitcher = _stamp_pid(pitcher_profile(home_pitcher_id), home_pitcher_id)
    away_pitcher = _stamp_pid(pitcher_profile(away_pitcher_id), away_pitcher_id)

    # The explicit pre-PA stolen-base layer (task #74) is RETIRED (2026-08-03
    # audit, finding M5): build_pa_table defines post_state as the NEXT PA's
    # start state, so every mid-PA steal/CS/pickoff/WP is already embedded in
    # the resampled transitions -- the explicit layer added the same movement
    # AGAIN (~2x real steal volume, doubled CS outs; embedded movement
    # measured at 11-14% per runner-on-1st cell + explicit 6.3%x80% on top).
    # Decision by pre-registered CRN-paired A/B (n=697, K=300, 2023-24):
    # retiring costs no material point-metric loss (SU gap -1.15pp at K=300
    # but +1.15pp at K=100 -- sign flips inside noise; Brier gap +0.0009,
    # both under the 1.5pp / 0.002 bars), so volume correctness wins.
    # baserunning.py and the SB-stats fetch stay intact for a future
    # conditional-transition redesign (de-embed, then re-add identity).
    missing = [pid for pid, prof in zip(home_ids + away_ids, home_lineup + away_lineup) if prof is None]
    if home_pitcher is None:
        missing.append(home_pitcher_id)
    if away_pitcher is None:
        missing.append(away_pitcher_id)
    if missing:
        raise ValueError(f"no pregame snapshot for player id(s) {missing} as of {game_date} "
                          f"(likely an MLB debut with zero PA history in our cached data)")

    park_key = (home_team, season)
    park_factors = None
    if park_key in ctx["park_factors_wide"].index:
        row = ctx["park_factors_wide"].loc[park_key]
        park_factors = {o: row.get(o, 1.0) if pd.notna(row.get(o, 1.0)) else 1.0 for o in OUTCOMES}
    hfa_factors = ctx["hfa_factors_by_season"].get(season)

    # real posted weather is FIXED for every trial (we know it); missing
    # weather is instead RESOLVED ONCE into a distribution here (the one
    # part of this that costs a network call, if a forecast is available)
    # and then SAMPLED FRESH each trial below -- propagating real day-to-day
    # weather uncertainty into the simulated outcome distribution, same as
    # every other stochastic element in this project.
    weather_factors = None
    weather_dist = None
    game_venue = venue_name
    if game_pk in ctx["game_weather"].index:
        wx = ctx["game_weather"].loc[game_pk]
        if pd.notna(wx.get("venue_name")):
            game_venue = wx["venue_name"]
        bucket = bucket_weather(wx["weather_condition"], wx["weather_temp"], wx["weather_wind"])
        if bucket is not None:
            weather_factors = ctx["weather_factors_by_season"].get(season, {}).get(bucket)
        elif venue_name is not None:
            weather_dist = resolve_weather_distribution(ctx["historical_game_buckets"], venue_name, game_date)
    # PARK-RELATIVE weather (2026-08-03 audit, finding M10): the park factor
    # already embeds this venue's AVERAGE climate (it's measured from real
    # home/road outcome rates), so the applied weather factor is divided by
    # the venue's climatological expected factor -- weather contributes only
    # the deviation from this park's own typical conditions. Unknown venue /
    # cold-start season passes through unchanged (absolute, old behavior).
    venue_weather_norm = ctx["venue_weather_norms"].get(season, {}).get(game_venue)
    weather_factors = park_relative_weather_factors(weather_factors, venue_weather_norm)

    def fallback_profile(team):
        # the old flat pooled aggregate rate -- only used if a team has no
        # roster/rest history at all (e.g. earliest games of a season).
        row = nearest_prior_bullpen(ctx["bullpen_snap"], team, game_date, game_pk)
        if row is None:
            return None
        neutral = {o: 1.0 for o in OUTCOMES}
        return {"rates": {o: row[f"pregame_rate_{o}"] for o in OUTCOMES}, "hand": "R",
                "same_mult": neutral, "opp_mult": neutral, "pid": "BULLPEN_FALLBACK",
                "apply_ttop": False}

    # Task: trade-deadline hardening -- optional {pitcher_id: {"new_team":,
    # "effective_date":}} (see bullpen.build_traded_pitcher_overrides), set
    # once per day by the caller (generate_daily_props.py) into ctx. Absent
    # (ctx.get returns None) is byte-for-byte identical to before this existed.
    traded_overrides = ctx.get("traded_overrides")

    def build_roster_profiles(team, own_starter_id):
        raw = build_team_bullpen_roster(ctx["relief_log"], ctx["all_appearance_log"], team, game_date,
                                         traded_overrides=traded_overrides)
        # exclude the day's own probable starter (2026-08-03 audit, finding
        # M17): swingmen/bulk arms qualify for their own team's 45-day
        # relief pool in 13% of real starts, so the simulator could draw
        # the starter to "relieve" the very game he started -- an illegal
        # re-entry that also double-credited his outs/K props.
        raw.pop(own_starter_id, None)
        profiles, weights = {}, {}
        for pid, w in raw.items():
            prof = roster_pitcher_profile(pid)
            if prof is not None:
                profiles[pid] = _stamp_pid(prof, pid, apply_ttop=False)
                weights[pid] = w
        return profiles, weights

    home_roster_profiles, home_roster_weights = build_roster_profiles(home_team, home_pitcher_id)
    away_roster_profiles, away_roster_weights = build_roster_profiles(away_team, away_pitcher_id)
    # 3-tier lookup (2026-08-03 audit, finding C4): exact historical
    # (pitcher, game_pk) row for backtests -> the pitcher's live
    # as-of-next-start projection for real future games -> 5.4 only for a
    # pitcher with no starter history anywhere (true opener/debut).
    def _exp_ip(pitcher_id):
        v = ctx["expected_innings"].get((pitcher_id, game_pk))
        if v is None:
            v = ctx["expected_innings_live"].get(pitcher_id)
        return 5.4 if v is None else float(v)

    home_exp_ip = _exp_ip(home_pitcher_id)
    away_exp_ip = _exp_ip(away_pitcher_id)
    home_fallback = fallback_profile(home_team)
    away_fallback = fallback_profile(away_team)
    home_closer_id = identify_closer(ctx["closer_log"], home_team, game_date, traded_overrides=traded_overrides)
    away_closer_id = identify_closer(ctx["closer_log"], away_team, game_date, traded_overrides=traded_overrides)

    home_catcher_id = identify_starting_catcher(ctx["catcher_log"], home_team, game_date)
    away_catcher_id = identify_starting_catcher(ctx["catcher_log"], away_team, game_date)
    home_catcher_factor = resolve_catcher_factor(ctx["catcher_factors_by_season"], season, home_catcher_id)
    away_catcher_factor = resolve_catcher_factor(ctx["catcher_factors_by_season"], season, away_catcher_id)

    # real home-plate umpire for THIS specific game -- fetched live (works
    # same-day once MLB posts the crew; neutral fallback otherwise, see
    # umpire_factor.py's resolve_live_umpire_factor). Applies identically to
    # BOTH teams' batting (one umpire calls the whole game), so it's
    # multiplied into both catcher-factor dicts rather than threaded as a
    # separate simulator parameter, same as the oracle backtest does.
    umpire_factor = resolve_live_umpire_factor(ctx["umpire_factors_by_season"], season, game_pk)
    home_catcher_factor = {o: home_catcher_factor[o] * umpire_factor[o] for o in home_catcher_factor}
    away_catcher_factor = {o: away_catcher_factor[o] * umpire_factor[o] for o in away_catcher_factor}

    # real defensive alignment this game (see defense_factor.py) -- exact
    # game_pk match works once the real lineup is posted; falls back to this
    # team's own most recent real defensive alignment otherwise (same
    # nearest_prior_bullpen fallback the predictive bullpen roster uses).
    # Disjoint outcome keys (single/double/triple) from catcher/umpire's
    # strikeout/walk, so a plain dict merge is safe.
    home_defense_row = nearest_prior_bullpen(ctx["defense_snap"], home_team, game_date, game_pk)
    away_defense_row = nearest_prior_bullpen(ctx["defense_snap"], away_team, game_date, game_pk)
    home_defense_factor = resolve_defense_factor(
        home_defense_row["infield_oaa"] if home_defense_row is not None else None,
        home_defense_row["outfield_oaa"] if home_defense_row is not None else None,
    )
    away_defense_factor = resolve_defense_factor(
        away_defense_row["infield_oaa"] if away_defense_row is not None else None,
        away_defense_row["outfield_oaa"] if away_defense_row is not None else None,
    )
    home_catcher_factor = {**home_catcher_factor, **home_defense_factor}
    away_catcher_factor = {**away_catcher_factor, **away_defense_factor}

    # Task #167 (site "Factors" debug panel): a game-level snapshot of the
    # already-resolved factors above, captured here rather than recomputed --
    # every value below is a local variable that already exists at this point
    # in the function. Deliberately does NOT include per-trial-stochastic
    # elements (bullpen sequence, weather bucket when unposted, hook timing,
    # latent pitcher shock -- see game_simulator.py/bullpen.py docstrings)
    # since those vary trial-to-trial by design and have no single "the model
    # used X" value; this only surfaces the FIXED pregame inputs every trial
    # shares. home_catcher_factor/away_catcher_factor are already
    # catcher+umpire+defense multiplied together (see above) -- labeled as
    # such rather than re-split, since re-splitting would need recomputation.
    debug = {
        "park_factors": park_factors,
        "hfa_factors": hfa_factors,
        "weather_factors": weather_factors,
        "weather_headline_home_run": _weather_headline(weather_factors, "home_run"),
        "weather_is_forecast": weather_factors is None and weather_dist is not None,
        "umpire_factor": umpire_factor,
        "home": {
            "lineup_summary": _lineup_summary(home_lineup),
            "pitcher_summary": _pitcher_summary(home_pitcher),
            "catcher_id": home_catcher_id,
            "catcher_defense_umpire_factor": home_catcher_factor,
            "closer_id": home_closer_id,
            "expected_starter_innings": home_exp_ip,
        },
        "away": {
            "lineup_summary": _lineup_summary(away_lineup),
            "pitcher_summary": _pitcher_summary(away_pitcher),
            "catcher_id": away_catcher_id,
            "catcher_defense_umpire_factor": away_catcher_factor,
            "closer_id": away_closer_id,
            "expected_starter_innings": away_exp_ip,
        },
    }

    transitions, league_rates, state_factors = ctx["transitions"], ctx["league_rates"], ctx["state_factors"]
    sim = GameSimulator(transitions, league_rates[season], rng, state_factors=state_factors[season],
                        ttop_factors=ctx["ttop_factors"][season], shock_sigma=SHOCK_SIGMA)

    # Starter-hook frailty (task #144/#145), PROPS PATH ONLY -- see
    # build_hook_table's docstring for the 2026-07-24 starter-outs
    # calibration result that justifies this (distribution-realism fix, no
    # win-probability claim; game-level oracle backtest stays off this).
    # cutoff must match sample_bullpen_plan's own internal cutoff formula
    # exactly (max(1, round(expected_innings))) -- it's the pregame-assumed
    # exit inning the sampled home_bullpen/away_bullpen plan was built around.
    home_hook_context = {"hook_table": ctx["hook_table"], "cutoff": max(1, round(home_exp_ip))}
    away_hook_context = {"hook_table": ctx["hook_table"], "cutoff": max(1, round(away_exp_ip))}

    all_events, final_scores = [], []
    for trial in range(n_trials):
        # a specific reliever is SAMPLED per inning (roster-weighted by
        # recent usage + rest-day availability), resampled fresh every trial
        # since which reliever actually pitches is itself part of the
        # uncertainty a Monte Carlo simulation should propagate.
        home_bullpen, _ = sample_bullpen_plan(
            rng, home_pitcher, home_exp_ip, home_roster_weights,
            lambda pid: home_roster_profiles[pid], home_fallback, closer_id=home_closer_id,
        )
        away_bullpen, _ = sample_bullpen_plan(
            rng, away_pitcher, away_exp_ip, away_roster_weights,
            lambda pid: away_roster_profiles[pid], away_fallback, closer_id=away_closer_id,
        )
        trial_weather_factors = weather_factors
        if trial_weather_factors is None and weather_dist:
            sampled_bucket = sample_weather_bucket(weather_dist, rng)
            trial_weather_factors = park_relative_weather_factors(
                ctx["weather_factors_by_season"].get(season, {}).get(sampled_bucket),
                venue_weather_norm,  # same M10 park-relative division as the fixed-weather path
            )
        events = []
        h, a = sim.simulate_game(
            home_lineup, away_lineup, home_pitcher, away_pitcher, innings=18,
            park_factors=park_factors, weather_factors=trial_weather_factors,
            home_bullpen=home_bullpen, away_bullpen=away_bullpen,
            blowout_pitcher_profile=ctx["blowout_profile"], events=events,
            home_catcher_factor=home_catcher_factor, away_catcher_factor=away_catcher_factor,
            hfa_factors=hfa_factors, postseason=postseason,
            home_hook_context=home_hook_context, away_hook_context=away_hook_context,
        )
        final_scores.append((h, a))
        for e in events:
            e["trial"] = trial
            # Per-PA pitcher attribution (2026-08-03 audit, finding M15):
            # the simulator stamps WHO actually threw each PA on the event
            # itself ("pid", read from the profile at PA time -- every
            # profile in this path carries one via _stamp_pid above), which
            # replaces the old reconstruct-from-id_plan inning mapping. The
            # old rule was whole-inning granular, so every post-hook
            # reliever PA in the hook inning itself was silently credited
            # to the starter (~2-4 extra PAs per hooked start), inflating
            # starter K/outs props. A blowout PA is attributed to a generic
            # "TEAM_POSITION_PLAYER" pseudo-id for the pitching team (the
            # OTHER side's label), since blowout_pitcher_profile carries no
            # real identity.
            pid = e.pop("pid")
            if e["blowout"]:
                pitching_team = home_team if e["side"] == "away" else away_team
                e["pitcher_id"] = f"{pitching_team}_POSITION_PLAYER"
            elif pid is not None:
                e["pitcher_id"] = pid
            else:
                raise ValueError(
                    f"simulator event with no pitcher pid stamp (side={e['side']}, "
                    f"inning={e['inning']}, game_pk={game_pk}) -- a pitcher profile "
                    "was built without _stamp_pid"
                )
            e["batter_id"] = (away_ids if e["side"] == "away" else home_ids)[e["batter_idx"]]
        all_events.extend(events)

    events_df = pd.DataFrame(all_events)
    scores_df = pd.DataFrame(final_scores, columns=["home_score", "away_score"])
    return {
        "batter_props": _batter_props(events_df, n_trials),
        "pitcher_props": _pitcher_props(events_df, n_trials),
        "inning_props": _inning_props(events_df, n_trials),
        "game_props": _game_props(scores_df, n_trials),
        "debut_fallback_pids": sorted(set(debut_fallback_pids)),
        "debug": debug,
    }


# Empirically-fit post-hoc calibration for batter game-level "at least N"
# props -- see validate_prop_calibration.py, the first-ever check of whether this
# project's PROP-LEVEL probabilities (not just final-score accuracy) are genuinely
# calibrated against real outcomes. Found real, consistently-reproducible overconfidence:
# Monte Carlo trial variance expresses MORE night-to-night differentiation between
# batters than real single-game outcomes support (unsurprising at only ~4 PA/batter/
# game -- a much smaller single-game information budget than a pitcher's ~20-25 batters
# faced, which is why pitcher props needed no such correction, see p_6plus_k). A simple
# linear recalibration toward league average, validated via 5 INDEPENDENT random
# train/test splits (never trust one split -- see this project's own history with
# rare-outcome-category calibration, which failed exactly this test for several
# categories) is kept ONLY for props whose correction beats the raw/uncorrected
# model's Brier score in 4-5 of 5 splits -- anything weaker is left uncorrected
# rather than "fixed" on what amounts to a coin flip.
#
# REFIT 2026-07-23 (task #138), after task #137 (the latent pitcher-appearance shock)
# went live: the shock widens the simulated distribution, making raw prop probabilities
# LESS overconfident than before -- so the correction needed today is smaller, AND,
# critically, the 5-split stability test itself now excludes two props that used to
# pass: p_1plus_hit (was 4-5/5, now only 3/5 post-shock) and p_1plus_rbi (now only
# 2/5 -- the exact same instability signature that excluded p_1plus_hr originally).
# Both were left uncorrected at that point, same treatment as p_1plus_hr. Re-run the
# 5-split check (scratchpad script, or promote it into validate_prop_calibration.py)
# before ever re-adding either -- do not just re-plug new coefficients without
# re-checking stability, since that's exactly the shortcut that would have missed this.
#
# RE-CHECKED 2026-07-25 on a fresh 150-game/2024-2025 sample (row-level 5-split test,
# not bucketed): p_1plus_hit 2/5 (still unstable, stays uncorrected), p_1plus_hr 1/5
# (still unstable, third confirmation), p_1plus_rbi **5/5 -- passed, briefly deployed
# below with (0.1196, 0.5977)**.
#
# REVERTED, same day, after task #154's RBI_EXCLUDED_OUTCOMES fix (see module
# docstring): that "5/5, needs a 0.60 slope correction" result was fit against the
# OLD, rule-violating RBI ground truth (which over-credited runs on double_play/
# field_error outcomes). Once the ground truth itself was corrected, re-ran the SAME
# 150-game sample + 5-split check: slope moved to ~1.10 (near-perfect, was 0.60) and
# stability dropped to 1/5 (unstable -- a correction is no longer even fittable, let
# alone needed). The original miscalibration was a symptom of the RBI-attribution
# bug, not a real modeling flaw -- fixing the root cause fixed the calibration too.
# p_1plus_rbi now joins p_1plus_hit/p_1plus_hr as correctly uncorrected. See
# MODEL_DOCUMENTATION.md sec 8.3 for the full before/after numbers.
BATTER_PROP_CALIBRATION = {
    # REFIT 2026-08-03 on the post-audit stack (finding M19): fit against the
    # {prop}_raw bypass columns (see _apply_batter_prop_calibration -- the
    # old slopes were fit 2026-07-23/25, before the platoon/TTOP/weather/
    # shock fixes changed the distributions they correct), 150 fresh games x
    # 150 trials, 5-fold CV by game, affine kept only where >=4/5 held-out
    # folds improve Brier -- the SAME two props as the original fit pass,
    # the other three (1plus_hit 3/5, 1plus_hr 0/5 -- raw already
    # well-calibrated, slope 1.04 -- and 1plus_rbi 3/5) stay uncalibrated.
    "p_2plus_hits": (0.1012, 0.5317),  # 4/5 splits (was 0.1106, 0.4892)
    "p_1plus_bb": (0.0618, 0.6950),  # 4/5 splits (was 0.0804, 0.6442)
}


def _apply_batter_prop_calibration(df: pd.DataFrame) -> pd.DataFrame:
    for prop, (a, b) in BATTER_PROP_CALIBRATION.items():
        # RAW bypass (2026-08-03 audit, M19 prerequisite): keep the
        # pre-calibration Monte Carlo frequency alongside the calibrated
        # value -- any future refit of BATTER_PROP_CALIBRATION must fit
        # actuals against {prop}_raw, never against the already-calibrated
        # column (a slope fit on post-calibration output composes the old
        # slopes into the new ones -- refit circularity).
        df[f"{prop}_raw"] = df[prop]
        df[prop] = np.clip(a + b * df[prop], 0.001, 0.999)
    return df


def _batter_props(events_df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """Per batter_id: probability of 1+/2+ hits, HR, walk, strikeout, RBI,
    and the mean simulated hits/HR/BB/RBI/total-bases/strikeouts across
    trials. The mean_hr/mean_bb/mean_rbi columns (task: site prop-display
    redesign) read straight off per_trial's already-computed hr/bb/rbi
    columns below -- the probability columns aren't removed (still used
    for calibration), just no longer the only per-player numbers available."""
    events_df = events_df.copy()
    events_df["is_hit"] = events_df["outcome"].isin(HIT_OUTCOMES)
    events_df["is_hr"] = events_df["outcome"] == "home_run"
    events_df["is_bb"] = events_df["outcome"].isin({"walk", "intent_walk"})
    events_df["is_k"] = events_df["outcome"] == "strikeout"
    events_df["bases"] = events_df["outcome"].map(TOTAL_BASES).fillna(0)
    # Task #154: a run scored on a PA whose outcome is in RBI_EXCLUDED_OUTCOMES
    # (double_play, field_error) doesn't count toward this batter's RBI prop --
    # see module docstring for the exact rules and real-data quantification.
    # The TOTAL runs-scored simulation itself is untouched; only the RBI
    # attribution changes.
    events_df["rbi_eligible_runs"] = events_df["runs"].where(~events_df["outcome"].isin(RBI_EXCLUDED_OUTCOMES), 0)

    per_trial = events_df.groupby(["batter_id", "trial"]).agg(
        hits=("is_hit", "sum"), hr=("is_hr", "sum"), bb=("is_bb", "sum"),
        k=("is_k", "sum"), rbi=("rbi_eligible_runs", "sum"), bases=("bases", "sum"), pa=("outcome", "size"),
    ).reset_index()

    def summarize(g):
        n = len(g)
        return pd.Series({
            "pa_per_game": g["pa"].mean(),
            "p_1plus_hit": (g["hits"] >= 1).mean(),
            "p_2plus_hits": (g["hits"] >= 2).mean(),
            "p_1plus_hr": (g["hr"] >= 1).mean(),
            "p_1plus_bb": (g["bb"] >= 1).mean(),
            "p_1plus_rbi": (g["rbi"] >= 1).mean(),
            "mean_total_bases": g["bases"].mean(),
            "mean_hits": g["hits"].mean(),
            "mean_hr": g["hr"].mean(),
            "mean_bb": g["bb"].mean(),
            "mean_rbi": g["rbi"].mean(),
            "mean_k": g["k"].mean(),
        })

    result = per_trial.groupby("batter_id").apply(summarize, include_groups=False)
    return _apply_batter_prop_calibration(result)


def _pitcher_props(events_df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """Per pitcher_id: mean/median strikeouts thrown, walks, hits, TOTAL runs
    allowed, and outs recorded (an innings-pitched proxy) across trials.

    Task #154 audit note: `mean_runs_allowed` was previously mis-described
    here as "earned runs allowed" -- the underlying column was always
    correctly total runs (earned + unearned), just wrongly labeled in this
    docstring. Fixed the wording, not the computation. True earned-run
    tracking (excluding runs that scored only because of a defensive error,
    with the correct "would have ended the inning under average defense"
    counterfactual official scoring actually applies) would need reasoning
    about an error's downstream effect on the REST of the inning across
    multiple PAs, not just this one -- out of scope for this project's
    PA-level, not out-by-out, granularity. `mean_runs_allowed` should be
    read as runs allowed, full stop, not an ERA-consistent figure."""
    events_df = events_df.copy()
    events_df["is_k"] = events_df["outcome"] == "strikeout"
    events_df["is_bb"] = events_df["outcome"].isin({"walk", "intent_walk"})
    events_df["is_hit"] = events_df["outcome"].isin(HIT_OUTCOMES)

    per_trial = events_df.groupby(["pitcher_id", "trial"]).agg(
        k=("is_k", "sum"), bb=("is_bb", "sum"), hits=("is_hit", "sum"),
        runs_allowed=("runs", "sum"), batters_faced=("outcome", "size"),
    ).reset_index()

    def summarize(g):
        return pd.Series({
            "mean_k": g["k"].mean(), "mean_bb": g["bb"].mean(), "mean_hits_allowed": g["hits"].mean(),
            "mean_runs_allowed": g["runs_allowed"].mean(), "mean_batters_faced": g["batters_faced"].mean(),
            "p_6plus_k": (g["k"] >= 6).mean(),
            # 2026-08-03 audit, finding M22: per_trial only has rows for
            # trials where this pitcher actually appeared, so every mean
            # above is E[stat | appeared] -- ~1.0 for the two starters, but
            # far below 1 for any sampled reliever. This column makes that
            # conditioning explicit (P(appears) across all n_trials) so a
            # reliever's "1.6 K" can be read for what it is instead of
            # masquerading as an unconditional projection.
            "appearance_rate": len(g) / n_trials,
        })

    return per_trial.groupby("pitcher_id").apply(summarize, include_groups=False)


def _inning_props(events_df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    """P(1+ run scored) per team (side) per inning, across trials."""
    per_trial = events_df.groupby(["side", "inning", "trial"])["runs"].sum().reset_index()
    return per_trial.groupby(["side", "inning"])["runs"].apply(lambda s: (s >= 1).mean()).unstack("side")


def _game_props(scores_df: pd.DataFrame, n_trials: int) -> dict:
    total = scores_df["home_score"] + scores_df["away_score"]
    margin = scores_df["home_score"] - scores_df["away_score"]
    return {
        "home_win_prob": (margin > 0).mean(),
        "away_win_prob": (margin < 0).mean(),
        "mean_home_score": scores_df["home_score"].mean(),
        "mean_away_score": scores_df["away_score"].mean(),
        "mean_total": total.mean(),
        "p_over_8.5": (total > 8.5).mean(),
        "p_under_8.5": (total < 8.5).mean(),
        "home_covers_minus_1_5": (margin > 1.5).mean(),
        "away_covers_plus_1_5": (margin < 1.5).mean(),
    }


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building pregame context (all walk-forward tables)...", flush=True)
    ctx = build_pregame_context(pa)

    season = 2025
    lineups = pd.read_parquet(DATA_RAW / f"lineups_{season}.parquet")
    schedule = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
    complete = lineups.groupby(["game_pk", "team_side"]).size()
    complete_games = complete[complete == 9].reset_index()["game_pk"].unique()
    reg = schedule[(schedule["game_type"] == "R") & (schedule["status"] == "Final") & (schedule["game_pk"].isin(complete_games))]
    demo = reg.sample(1, random_state=7).iloc[0]
    game_pk = int(demo["game_pk"])
    game_lu = lineups[lineups["game_pk"] == game_pk]
    home_ids = game_lu[game_lu["team_side"] == "home"].sort_values("batting_order")["player_id"].tolist()
    away_ids = game_lu[game_lu["team_side"] == "away"].sort_values("batting_order")["player_id"].tolist()
    game_pa = pa[pa.game_pk == game_pk]
    home_pitcher_id = int(game_pa[game_pa.inning_topbot == "Top"].sort_values(["inning", "at_bat_number"])["pitcher"].iloc[0])
    away_pitcher_id = int(game_pa[game_pa.inning_topbot == "Bot"].sort_values(["inning", "at_bat_number"])["pitcher"].iloc[0])

    print(f"generating props for game {game_pk}: {demo['away_team']} @ {demo['home_team']} "
          f"({demo['date']}), {N_TRIALS} trials...", flush=True)
    rng = np.random.default_rng(0)
    props = generate_game_props(
        ctx, season, game_pk, demo["home_team"], demo["away_team"], demo["date"],
        home_ids, away_ids, home_pitcher_id, away_pitcher_id, n_trials=N_TRIALS, rng=rng,
    )

    print(f"\n=== actual result: {demo['away_team']} {demo['away_score']} @ {demo['home_team']} {demo['home_score']} ===")
    print("\n--- game props ---")
    for k, v in props["game_props"].items():
        print(f"  {k}: {v:.3f}")
    print("\n--- batter props (home lineup, batting order) ---")
    print(props["batter_props"].loc[[pid for pid in home_ids if pid in props["batter_props"].index]].round(3).to_string())
    print("\n--- pitcher props ---")
    print(props["pitcher_props"].round(2).to_string())
    print("\n--- inning-by-inning P(1+ run) ---")
    print(props["inning_props"].round(3).to_string())
