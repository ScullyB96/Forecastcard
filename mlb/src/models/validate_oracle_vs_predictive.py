"""Decomposes the oracle-vs-deployable accuracy gap into its independently
resolvable components (task from a reviewer, 2026-07-23): every accuracy
number this project has ever reported (SU ~58%, Brier ~0.24, std(z)~1.0) was
measured on the ORACLE path (validate_game_simulator.py) -- real confirmed
lineups, real per-inning bullpen sequence, real recorded weather, real
starting catchers. The actually-deployed path (props.py/generate_daily_props.py)
guesses or infers ALL FOUR of these when the real information isn't posted
yet (lineup: project_lineup_platoon_aware; bullpen: sample_bullpen_plan;
weather: climatological/forecast fallback; catcher: identify_starting_catcher).
This gap has never been measured end to end.

Method: for each candidate game, resolve each of the 4 dimensions via EITHER
its oracle (real) source or its predictive (guessed) source, independently
controlled per dimension via a `sources: dict` config, e.g.
{"lineup": "oracle", "bullpen": "predictive", "weather": "predictive",
"catcher": "predictive"}. Every configuration is CRN-paired against every
other (same crn_game_pk/crn_trial keys throughout), so wherever two
configurations agree on a dimension's source, that dimension's random draws
stay perfectly synchronized -- only the toggled dimension(s) can actually
differ between two runs. Running FULL_ORACLE, FULL_PREDICTIVE, and 4
"oracle-X-only" ablations (X = lineup/bullpen/weather/catcher, holding the
other 3 predictive) lets the gap between FULL_PREDICTIVE and each
oracle-X-only run directly attribute how much of the total oracle-vs-
predictive gap that one dimension's guess is responsible for.

WEATHER SCOPING NOTE, read before trusting the weather arm's magnitude:
resolve_weather_distribution's live forecast-narrowing tiers
(fetch_temperature_forecast/fetch_wind_forecast, weather_forecast.py) query
Open-Meteo's FUTURE forecast endpoint -- it cannot serve arbitrary past
dates, so a historical backtest structurally cannot exercise those tiers.
This decomposition's "predictive weather" arm therefore uses ONLY the
climatological (tier-3) distribution, resampled fresh every trial (matching
weather_forecast.py's own per-trial-uncertainty convention) -- this somewhat
OVERSTATES real deployed weather uncertainty, since a real live game usually
also gets the temperature-forecast narrowing on top of climatology. Flagged,
not hidden -- the weather arm's attributed cost is an upper bound on the
real deployed cost, not an exact measurement of it.

Reuses build_shared_tables (validate_game_simulator.py, everything the
ORACLE side needs) plus the exact predictive-path machinery already
validated in validate_predictive_bullpen.py (bullpen/catcher) and
lineup_projection.py/weather_forecast.py (lineup/weather) -- no re-derived
logic for anything that already has a validated implementation.

KNOWN SCOPE LIMITATION (lineup arm specifically): a PROJECTED lineup can
name a player who did not actually bat in this exact historical game_pk (a
real, near-miss projection is exactly the point). The core rate/platoon
signal correctly falls back to that player's nearest-prior snapshot (see
batter_snap_dates/nearest_prior_pitcher_snapshot below) -- but the
SECONDARY contact-quality signals (xbacon/barrel/pulled-air/bacon-gb/gb-rate/
bat-speed/sprint-speed snapshots) are still scoped to game_pk with no
equivalent nearest-prior fallback, so they silently no-op (already-existing
None-safe behavior, not a crash) for any player who isn't a real participant
of this specific game. This means the ORACLE_LINEUP/FULL_PREDICTIVE arms
UNDERSTATE the predictive lineup profile's precision slightly relative to
what generate_daily_props.py would actually compute for a projected player
live -- a conservative bias (makes the lineup dimension's measured cost a
slight overestimate, not an underestimate), left as a known limitation
rather than built out given these are established secondary-order signals.
"""
import numpy as np
import pandas as pd

from src.models.base_out_transitions import TransitionTable
from src.models.blowout import build_position_player_profile
from src.models.catcher_framing import (
    build_catcher_appearance_log,
    identify_starting_catcher,
    resolve_catcher_factor,
)
from src.models.crn import crn_uniform
from src.models.defense_factor import resolve_defense_factor
from src.models.baserunning import resolve_sb_rates
from src.models.bullpen import (
    build_all_appearance_log,
    build_bullpen_snapshot,
    build_expected_starter_innings,
    build_closer_appearance_log,
    build_relief_appearance_log,
    build_team_bullpen_roster,
    identify_closer,
    nearest_prior_bullpen,
    nearest_prior_pitcher_snapshot,
    sample_bullpen_plan,
)
from src.models.game_simulator import OUTCOMES, GameSimulator
from src.models.lineup_projection import project_lineup_platoon_aware
from src.models.umpire_factor import resolve_umpire_factor
from src.models.validate_game_simulator import build_profile, build_shared_tables, player_game_snapshot, SHOCK_SIGMA
from src.models.weather import bucket_weather
from src.models.weather_forecast import build_historical_game_buckets, climatological_bucket_distribution
from src.utils.paths import DATA_PROCESSED, DATA_RAW

DECISION_WEATHER_SAMPLE = 101  # ad-hoc CRN tag, local to this validator (not one of
                                # game_simulator.py's own DECISION_* constants, since
                                # weather-bucket resampling happens OUTSIDE GameSimulator,
                                # in this file's own per-trial loop)


def build_predictive_tables(pa: pd.DataFrame, shared: dict, all_schedules: pd.DataFrame) -> dict:
    """Every additional table the PREDICTIVE side of each of the 4 dimensions
    needs, beyond what build_shared_tables already provides for the oracle
    side. See module docstring for what each feeds."""
    print("[predictive] building bullpen snapshot + expected starter innings...", flush=True)
    bullpen_snap = build_bullpen_snapshot(pa, shared["park_factors_long"])
    expected_innings = build_expected_starter_innings(pa).set_index(["pitcher", "game_pk"])["expected_innings"]
    print("[predictive] building bullpen roster + rest-day availability logs...", flush=True)
    relief_log = build_relief_appearance_log(pa)
    all_appearance_log = build_all_appearance_log(pa)
    closer_log = build_closer_appearance_log(pa)
    print("[predictive] building catcher starting-catcher log...", flush=True)
    catcher_log = build_catcher_appearance_log(pa)
    print("[predictive] building climatological weather buckets...", flush=True)
    historical_game_buckets = build_historical_game_buckets(all_schedules)
    venue_by_team_season = all_schedules[["home_team", "season", "venue_name"]].dropna().drop_duplicates(
        subset=["home_team", "season"]
    ).set_index(["home_team", "season"])["venue_name"]
    pitcher_snap_dates = shared["pitcher_snap"].merge(pa[["game_pk", "game_date"]].drop_duplicates(), on="game_pk")
    # A PROJECTED lineup (lineup_source="predictive") can name a player who did
    # NOT actually bat in this specific historical game_pk (they played
    # recently, just not here) -- batter_snap only has rows for real
    # participants of that exact game_pk, so a plain game_pk-scoped lookup
    # would silently skip the vast majority of games. Same fix already used
    # for sampled bullpen relievers (nearest_prior_pitcher_snapshot, despite
    # its name, auto-detects a "batter" column too -- reused as-is below).
    batter_snap_dates = shared["batter_snap"].merge(pa[["game_pk", "game_date"]].drop_duplicates(), on="game_pk")
    return dict(
        bullpen_snap=bullpen_snap, expected_innings=expected_innings, relief_log=relief_log,
        all_appearance_log=all_appearance_log, closer_log=closer_log, catcher_log=catcher_log,
        historical_game_buckets=historical_game_buckets, venue_by_team_season=venue_by_team_season,
        pitcher_snap_dates=pitcher_snap_dates, batter_snap_dates=batter_snap_dates,
    )


def run_decomposition(shared: dict, predictive: dict, test_seasons: set[int], n_games: int, n_trials: int,
                       sources: dict, shock_sigma: float = SHOCK_SIGMA, seed: int = 42,
                       output_path=None, verbose: bool = True) -> pd.DataFrame:
    """sources: {"lineup": "oracle"|"predictive", "bullpen": ..., "weather": ..., "catcher": ...}.
    CRN-paired throughout (crn_game_pk=game_pk, crn_trial=trial for every
    simulate_game call) so any two configs that agree on a dimension's source
    stay synchronized on that dimension's own random draws."""
    pa = shared["pa"]
    park_factors_wide = shared["park_factors_wide"]
    hfa_factors_by_season = shared["hfa_factors_by_season"]
    league_rates = shared["league_rates"]
    state_factors = shared["state_factors"]
    ttop_factors = shared["ttop_factors"]
    batter_platoon = shared["batter_platoon"]
    pitcher_platoon = shared["pitcher_platoon"]
    batter_hand = shared["batter_hand"]
    pitcher_hand = shared["pitcher_hand"]
    blowout_profile = shared["blowout_profile"]
    pull_rate_snapshot = shared["pull_rate_snapshot"]
    weather_factors_by_season = shared["weather_factors_by_season"]
    game_weather = shared["game_weather"]
    catcher_factors_by_season = shared["catcher_factors_by_season"]
    umpire_factors_by_season = shared["umpire_factors_by_season"]
    umpire_log = shared["umpire_log"]
    xbacon_snap = shared["xbacon_snap"]
    barrel_snap = shared["barrel_snap"]
    pulled_air_snap = shared["pulled_air_snap"]
    bacon_gb_snap = shared["bacon_gb_snap"]
    gb_rate_snap = shared["gb_rate_snap"]
    gbfb_pitcher_snap = shared["gbfb_pitcher_snap"]
    sprint_speed_by_season = shared["sprint_speed_by_season"]
    bat_speed_raw = shared["bat_speed_raw"]
    bat_speed_by_season = shared["bat_speed_by_season"]
    batter_snap = shared["batter_snap"]
    pitcher_snap = shared["pitcher_snap"]
    transitions = shared["transitions"]
    schedules = shared["schedules"]
    lineups = shared["lineups"]
    defense_snap_by_season = shared["defense_snap_by_season"]
    sb_rates_by_season = shared["sb_rates_by_season"]

    bullpen_snap = predictive["bullpen_snap"]
    expected_innings = predictive["expected_innings"]
    relief_log = predictive["relief_log"]
    all_appearance_log = predictive["all_appearance_log"]
    closer_log = predictive["closer_log"]
    catcher_log = predictive["catcher_log"]
    historical_game_buckets = predictive["historical_game_buckets"]
    venue_by_team_season = predictive["venue_by_team_season"]
    pitcher_snap_dates = predictive["pitcher_snap_dates"]
    batter_snap_dates = predictive["batter_snap_dates"]

    rng = np.random.default_rng(seed)

    candidates = []
    for season in test_seasons:
        sched = schedules[season][(schedules[season]["game_type"] == "R") & (schedules[season]["status"] == "Final")]
        lu = lineups[season]
        lineup_counts = lu.groupby(["game_pk", "team_side"]).size()
        complete_games = lineup_counts[lineup_counts == 9].reset_index()["game_pk"].unique()
        sched = sched[sched["game_pk"].isin(complete_games)]
        for row in sched.sample(min(len(sched), n_games // len(test_seasons)), random_state=1).itertuples():
            candidates.append((season, row.game_pk, row.home_score, row.away_score))

    if verbose:
        print(f"[decomposition sources={sources}] validating on {len(candidates)} real games, "
              f"{n_trials} trials each...", flush=True)
    results = []
    for season, game_pk, actual_home, actual_away in candidates:
        lu = lineups[season]
        game_lu = lu[lu["game_pk"] == game_pk]
        real_home_ids = game_lu[game_lu["team_side"] == "home"].sort_values("batting_order")["player_id"].tolist()
        real_away_ids = game_lu[game_lu["team_side"] == "away"].sort_values("batting_order")["player_id"].tolist()
        if len(real_home_ids) != 9 or len(real_away_ids) != 9:
            continue

        game_pa = pa[pa.game_pk == game_pk]
        top_pa = game_pa[game_pa.inning_topbot == "Top"].sort_values(["inning", "at_bat_number"])
        bot_pa = game_pa[game_pa.inning_topbot == "Bot"].sort_values(["inning", "at_bat_number"])
        if top_pa.empty or bot_pa.empty:
            continue
        home_pitcher_id = top_pa["pitcher"].iloc[0]
        away_pitcher_id = bot_pa["pitcher"].iloc[0]

        home_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "home_team"].iloc[0]
        away_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "away_team"].iloc[0]
        game_date = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "date"].iloc[0]

        # --- LINEUP: oracle (real posted) vs predictive (platoon-aware projection
        # from each team's own recent history) ---
        if sources["lineup"] == "oracle":
            home_ids, away_ids = real_home_ids, real_away_ids
        else:
            proj_home = project_lineup_platoon_aware(schedules[season], lu, pitcher_hand, home_team, game_date, away_pitcher_id)
            proj_away = project_lineup_platoon_aware(schedules[season], lu, pitcher_hand, away_team, game_date, home_pitcher_id)
            if proj_home is None or proj_away is None or len(proj_home) != 9 or len(proj_away) != 9:
                continue  # no usable projection yet (e.g. earliest games) -- skip, matching the live pipeline's own behavior
            home_ids, away_ids = proj_home, proj_away

        bsnap = batter_snap[batter_snap["game_pk"] == game_pk].set_index("batter")
        psnap = pitcher_snap[pitcher_snap["game_pk"] == game_pk].set_index("pitcher")
        if home_pitcher_id not in psnap.index or away_pitcher_id not in psnap.index:
            continue

        xsnap = xbacon_snap[xbacon_snap["game_pk"] == game_pk].set_index("batter")
        barsnap = barrel_snap[barrel_snap["game_pk"] == game_pk].set_index("batter")
        pullairsnap = pulled_air_snap[pulled_air_snap["game_pk"] == game_pk].set_index("batter")
        bacsnap = bacon_gb_snap[bacon_gb_snap["game_pk"] == game_pk].set_index("batter")
        gbsnap = gb_rate_snap[gb_rate_snap["game_pk"] == game_pk].set_index("batter")
        gbfbsnap = gbfb_pitcher_snap[gbfb_pitcher_snap["game_pk"] == game_pk].set_index("pitcher")
        if season not in sprint_speed_by_season:
            from src.models.expected_stats import build_pregame_sprint_speed
            sprint_speed_by_season[season] = build_pregame_sprint_speed(season)
        speed_snap = sprint_speed_by_season[season]
        if season not in bat_speed_by_season:
            from src.models.expected_stats import build_pregame_bat_speed
            bat_speed_by_season[season] = build_pregame_bat_speed(bat_speed_raw, season)
        bat_speed_snap = bat_speed_by_season[season]

        def batter_profile(pid):
            if pid in bsnap.index:
                row = bsnap.loc[pid]
            else:
                # projected lineup named a player who didn't actually bat in
                # THIS historical game_pk -- fall back to their most recent
                # snapshot as of this date (see build_predictive_tables' note).
                row = nearest_prior_pitcher_snapshot(batter_snap_dates, pid, game_date, game_pk)
                if row is None:
                    return None
            key = (pid, season)
            prow = batter_platoon.loc[key] if key in batter_platoon.index else None
            hand = batter_hand.get(pid, "R")
            from src.models.spray import resolve_batter_pull_tercile
            tercile = resolve_batter_pull_tercile(pull_rate_snapshot, pid, hand, game_date, game_pk)
            xbacon = xsnap.loc[pid, "pregame_xbacon"] if pid in xsnap.index else None
            barrel = barsnap.loc[pid, "pregame_barrel_rate"] if pid in barsnap.index else None
            pulled_air = pullairsnap.loc[pid, "pregame_pulled_air_rate"] if pid in pullairsnap.index else None
            bacon_gb = bacsnap.loc[pid, "pregame_bacon_gb"] if pid in bacsnap.index else None
            sprint_speed = speed_snap.loc[pid] if pid in speed_snap.index else None
            gb_rate = gbsnap.loc[pid, "pregame_gb_rate"] if pid in gbsnap.index else None
            bat_speed = bat_speed_snap.loc[pid] if pid in bat_speed_snap.index else None
            return build_profile(row, prow, hand, pull_tercile=tercile, pregame_xbacon=xbacon,
                                  pregame_barrel_rate=barrel, pregame_bacon_gb=bacon_gb,
                                  pregame_sprint_speed=sprint_speed, pregame_gb_rate=gb_rate,
                                  pregame_bat_speed=bat_speed, pregame_pulled_air_rate=pulled_air)

        def pitcher_profile(pid):
            row = psnap.loc[pid]
            key = (pid, season)
            prow = pitcher_platoon.loc[key] if key in pitcher_platoon.index else None
            gb_p = gbfbsnap.loc[pid, "pregame_gb_rate_pitcher"] if pid in gbfbsnap.index else None
            fb_p = gbfbsnap.loc[pid, "pregame_fb_rate_pitcher"] if pid in gbfbsnap.index else None
            return build_profile(row, prow, pitcher_hand.get(pid, "R"),
                                  pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p)

        def roster_pitcher_profile(pid, game_date_):
            if pid in psnap.index:
                row = psnap.loc[pid]
            else:
                row = nearest_prior_pitcher_snapshot(pitcher_snap_dates, pid, game_date_, game_pk)
                if row is None:
                    return None
            key = (pid, season)
            prow = pitcher_platoon.loc[key] if key in pitcher_platoon.index else None
            gb_p = gbfbsnap.loc[pid, "pregame_gb_rate_pitcher"] if pid in gbfbsnap.index else None
            fb_p = gbfbsnap.loc[pid, "pregame_fb_rate_pitcher"] if pid in gbfbsnap.index else None
            return build_profile(row, prow, pitcher_hand.get(pid, "R"),
                                  pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p)

        home_lineup = [batter_profile(pid) for pid in home_ids]
        away_lineup = [batter_profile(pid) for pid in away_ids]
        if any(p is None for p in home_lineup + away_lineup):
            continue  # a projected player has zero usable history yet (e.g. early cold-start) -- skip, matching live pipeline behavior
        home_pitcher = pitcher_profile(home_pitcher_id)
        away_pitcher = pitcher_profile(away_pitcher_id)

        park_key = (home_team, season)
        if park_key in park_factors_wide.index:
            park_row = park_factors_wide.loc[park_key]
            park_factors_this_game = {o: park_row.get(o, 1.0) if pd.notna(park_row.get(o, 1.0)) else 1.0 for o in OUTCOMES}
        else:
            park_factors_this_game = None

        # --- WEATHER: oracle (real recorded bucket, fixed for the whole game)
        # vs predictive (climatological distribution, RESAMPLED every trial --
        # see module docstring's weather scoping note) ---
        weather_source = sources["weather"]
        weather_factors_fixed = None
        weather_dist = None
        if weather_source == "oracle":
            if game_pk in game_weather.index:
                wx = game_weather.loc[game_pk]
                bucket = bucket_weather(wx["weather_condition"], wx["weather_temp"], wx["weather_wind"])
                weather_factors_fixed = weather_factors_by_season.get(season, {}).get(bucket)
        else:
            venue_name = venue_by_team_season.get((home_team, season))
            if venue_name is not None:
                month = pd.Timestamp(game_date).month
                weather_dist = climatological_bucket_distribution(historical_game_buckets, venue_name, month)
            if not weather_dist:
                weather_factors_fixed = None  # no history for this venue -- neutral, matches oracle's own "no data" fallback

        # --- BULLPEN: oracle (real per-inning sequence) vs predictive (sampled
        # roster reliever per inning, resampled every trial) ---
        bullpen_source = sources["bullpen"]
        if bullpen_source == "oracle":
            home_pitcher_by_inning = top_pa.groupby("inning")["pitcher"].first()
            away_pitcher_by_inning = bot_pa.groupby("inning")["pitcher"].first()
            home_bullpen_fixed = {int(inn): pitcher_profile(pid) for inn, pid in home_pitcher_by_inning.items() if pid in psnap.index}
            away_bullpen_fixed = {int(inn): pitcher_profile(pid) for inn, pid in away_pitcher_by_inning.items() if pid in psnap.index}
        else:
            neutral = {o: 1.0 for o in OUTCOMES}

            def fallback_profile(team):
                row = nearest_prior_bullpen(bullpen_snap, team, game_date, game_pk)
                if row is None:
                    return None
                return {"rates": {o: row[f"pregame_rate_{o}"] for o in OUTCOMES}, "hand": "R",
                        "same_mult": neutral, "opp_mult": neutral}

            def build_roster_profiles(team):
                raw = build_team_bullpen_roster(relief_log, all_appearance_log, team, game_date)
                profiles, weights = {}, {}
                for pid, w in raw.items():
                    prof = roster_pitcher_profile(pid, game_date)
                    if prof is not None:
                        profiles[pid] = prof
                        weights[pid] = w
                return profiles, weights

            home_roster_profiles, home_roster_weights = build_roster_profiles(home_team)
            away_roster_profiles, away_roster_weights = build_roster_profiles(away_team)
            home_exp_ip = expected_innings.get((home_pitcher_id, game_pk), 5.4)
            away_exp_ip = expected_innings.get((away_pitcher_id, game_pk), 5.4)
            home_fallback = fallback_profile(home_team)
            away_fallback = fallback_profile(away_team)
            home_closer_id = identify_closer(closer_log, home_team, game_date)
            away_closer_id = identify_closer(closer_log, away_team, game_date)

        # --- CATCHER: oracle (real starting catcher) vs predictive (recency-inferred) ---
        catcher_source = sources["catcher"]
        if catcher_source == "oracle":
            home_catcher_id = top_pa["catcher"].iloc[0]
            away_catcher_id = bot_pa["catcher"].iloc[0]
        else:
            home_catcher_id = identify_starting_catcher(catcher_log, home_team, game_date)
            away_catcher_id = identify_starting_catcher(catcher_log, away_team, game_date)
        home_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, home_catcher_id)
        away_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, away_catcher_id)

        ump_id = umpire_log.get(game_pk)
        umpire_factor = resolve_umpire_factor(umpire_factors_by_season, season, ump_id)
        home_catcher_factor = {o: home_catcher_factor[o] * umpire_factor[o] for o in home_catcher_factor}
        away_catcher_factor = {o: away_catcher_factor[o] * umpire_factor[o] for o in away_catcher_factor}

        defense_snap = defense_snap_by_season[season]
        home_defense_row = defense_snap.loc[(game_pk, home_team)] if (game_pk, home_team) in defense_snap.index else None
        away_defense_row = defense_snap.loc[(game_pk, away_team)] if (game_pk, away_team) in defense_snap.index else None
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

        home_sb_rates = resolve_sb_rates(sb_rates_by_season[season], home_ids)
        away_sb_rates = resolve_sb_rates(sb_rates_by_season[season], away_ids)

        sim = GameSimulator(transitions, league_rates[season], rng, state_factors=state_factors[season],
                            ttop_factors=ttop_factors[season], shock_sigma=shock_sigma)
        sim_home, sim_away = [], []
        for trial in range(n_trials):
            if bullpen_source == "oracle":
                home_bullpen_plan, away_bullpen_plan = home_bullpen_fixed, away_bullpen_fixed
            else:
                home_bullpen_plan, _ = sample_bullpen_plan(
                    rng, home_pitcher, home_exp_ip, home_roster_weights,
                    lambda pid: home_roster_profiles[pid], home_fallback, closer_id=home_closer_id,
                )
                away_bullpen_plan, _ = sample_bullpen_plan(
                    rng, away_pitcher, away_exp_ip, away_roster_weights,
                    lambda pid: away_roster_profiles[pid], away_fallback, closer_id=away_closer_id,
                )

            if weather_source == "oracle":
                weather_factors_this_trial = weather_factors_fixed
            elif weather_dist:
                u = crn_uniform(game_pk, trial, DECISION_WEATHER_SAMPLE)
                buckets, probs = zip(*weather_dist.items())
                cum = np.cumsum(probs)
                idx = int(np.searchsorted(cum, u, side="right"))
                bucket = buckets[min(idx, len(buckets) - 1)]
                weather_factors_this_trial = weather_factors_by_season.get(season, {}).get(bucket)
            else:
                weather_factors_this_trial = None

            h, a = sim.simulate_game(
                home_lineup, away_lineup, home_pitcher, away_pitcher, innings=18,
                park_factors=park_factors_this_game,
                weather_factors=weather_factors_this_trial,
                home_bullpen=home_bullpen_plan, away_bullpen=away_bullpen_plan,
                blowout_pitcher_profile=blowout_profile,
                home_catcher_factor=home_catcher_factor, away_catcher_factor=away_catcher_factor,
                home_sb_rates=home_sb_rates, away_sb_rates=away_sb_rates,
                hfa_factors=hfa_factors_by_season.get(season),
                crn_game_pk=game_pk, crn_trial=trial,
            )
            sim_home.append(h)
            sim_away.append(a)

        sim_home_arr, sim_away_arr = np.array(sim_home), np.array(sim_away)
        results.append({
            "game_pk": game_pk, "actual_home": actual_home, "actual_away": actual_away,
            "sim_home_mean": np.mean(sim_home), "sim_away_mean": np.mean(sim_away),
            "sim_home_std": np.std(sim_home), "sim_away_std": np.std(sim_away),
            "sim_home_win_prob": (sim_home_arr > sim_away_arr).mean(),
        })

    r = pd.DataFrame(results)
    if output_path is not None:
        r.to_parquet(output_path, index=False)
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["sim_total_mean"] = r["sim_home_mean"] + r["sim_away_mean"]
    r["actual_margin"] = r["actual_home"] - r["actual_away"]
    r["sim_margin_mean"] = r["sim_home_mean"] - r["sim_away_mean"]
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    r["brier"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
    if verbose:
        su = ((r["sim_home_win_prob"] > 0.5).astype(int) == r["actual_home_win"]).mean()
        total_mae = (r["sim_total_mean"] - r["actual_total"]).abs().mean()
        print(f"  -> n={len(r)}  SU={su:.4f}  Brier={r['brier'].mean():.4f}  total_MAE={total_mae:.3f}", flush=True)
    return r


if __name__ == "__main__":
    TEST_SEASONS = {2023, 2024, 2025}
    N_GAMES = 25000  # effectively "every complete-lineup game" (~7237 total), matching the
                      # canonical protocol's own scale -- see validate_game_simulator.py's note
    N_TRIALS = 50  # matches the canonical (K=50) protocol for direct comparability

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    all_schedules = pd.concat([pd.read_parquet(p) for p in sorted(DATA_RAW.glob("schedule_*.parquet"))], ignore_index=True)
    print("building oracle-side shared tables...", flush=True)
    shared = build_shared_tables(pa, TEST_SEASONS)
    print("building predictive-side shared tables...", flush=True)
    predictive = build_predictive_tables(pa, shared, all_schedules)

    CONFIGS = {
        "FULL_ORACLE": {"lineup": "oracle", "bullpen": "oracle", "weather": "oracle", "catcher": "oracle"},
        "FULL_PREDICTIVE": {"lineup": "predictive", "bullpen": "predictive", "weather": "predictive", "catcher": "predictive"},
        "ORACLE_LINEUP": {"lineup": "oracle", "bullpen": "predictive", "weather": "predictive", "catcher": "predictive"},
        "ORACLE_BULLPEN": {"lineup": "predictive", "bullpen": "oracle", "weather": "predictive", "catcher": "predictive"},
        "ORACLE_WEATHER": {"lineup": "predictive", "bullpen": "predictive", "weather": "oracle", "catcher": "predictive"},
        "ORACLE_CATCHER": {"lineup": "predictive", "bullpen": "predictive", "weather": "predictive", "catcher": "oracle"},
    }

    for name, sources in CONFIGS.items():
        print(f"\n=== {name}: {sources} ===", flush=True)
        run_decomposition(shared, predictive, TEST_SEASONS, N_GAMES, N_TRIALS, sources,
                           output_path=DATA_PROCESSED / f"oracle_vs_predictive_{name}.parquet")
