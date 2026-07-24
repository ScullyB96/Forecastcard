"""The honest, deployable-relevant test: same 200 real historical games as
validate_game_simulator.py, same everything else, but the bullpen plan is
built PREDICTIVELY (starter pitches through his own walk-forward-projected
expected innings, then this team's own aggregate bullpen rate takes over) --
never looking at what actually happened in that specific game's real pitcher
sequence. This is the version relevant to a live/future game, unlike
validate_game_simulator.py's oracle backtest (which knows the real sequence
and can't exist for a game that hasn't been played).

Compares against both known baselines from the same 200-game/200-trial
protocol: no bullpen modeling at all (single starter for 9 innings, total MAE
3.728 / margin MAE 3.661 / SU 62.5%) and the oracle real-sequence bullpen
(total MAE 3.604 / margin MAE 3.445 / SU 67.5%) -- expected to land somewhere
between the two, since it has real signal (expected innings, team bullpen
quality) but not perfect foresight of which specific relievers pitch when.

Bullpen plan built via sample_bullpen_plan (see bullpen.py): rather than one
flat blended rate for every post-starter inning (which was found to erase
which specific reliever pitches, hurting straight-up accuracy), a specific
reliever is actually SAMPLED per inning -- weighted by recent usage
frequency and a rest-day availability discount for THIS game's date (see
build_team_bullpen_roster) -- and simulated with their own individual
walk-forward rate. Resampled fresh EVERY trial (not fixed once per game),
since which reliever a manager actually uses is itself part of the
uncertainty a Monte Carlo simulation should propagate, same as every other
stochastic element here.
"""

import numpy as np
import pandas as pd

from src.models.base_out_transitions import TransitionTable
from src.models.blowout import build_position_player_profile
from src.models.catcher_framing import (
    build_catcher_appearance_log,
    build_catcher_framing_factors_by_season,
    identify_starting_catcher,
    resolve_catcher_factor,
)
from src.models.defense_factor import resolve_defense_factor, team_game_defense_snapshot
from src.models.baserunning import build_season_sb_stats, build_pregame_sb_rates, resolve_sb_rates
from src.models.umpire_factor import build_umpire_factors_by_season, load_umpire_game_log, resolve_umpire_factor
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
from src.models.expected_stats import (
    apply_contact_quality,
    build_pregame_bat_speed,
    build_pregame_sprint_speed,
    load_bat_speed_by_season,
    player_game_bacon_gb_snapshot,
    player_game_barrel_snapshot,
    player_game_groundball_rate_snapshot,
    player_game_gb_fb_rate_snapshot,
    player_game_pulled_air_snapshot,
    player_game_xbacon_snapshot,
)
from src.models.game_simulator import (
    OUTCOMES,
    GameSimulator,
    build_league_rates_by_season,
    build_state_factors_by_season,
    build_wide_platoon_multipliers,
    build_wide_pregame_rates,
)
from src.models.park_factors import build_outcome_park_factors, build_hfa_factors
from src.models.spray import build_pull_rate_by_season, build_pull_rate_snapshot, resolve_batter_pull_tercile
from src.models.spray import attach_pull_tercile_column
from src.models.ttop import build_ttop_factors_by_season
from src.models.validate_game_simulator import build_profile, player_game_snapshot, predominant_hand, SHOCK_SIGMA
from src.models.weather import attach_weather_bucket, bucket_weather, build_weather_factors_by_season
from src.utils.paths import DATA_PROCESSED, DATA_RAW

N_TRIALS_PER_GAME = 50  # lowered from 200 (2026-07-22), matches validate_game_simulator.py's
                        # same tradeoff -- prioritize game count (between-game sample size,
                        # which gates resolving ~1pp-scale effects) over trials/game (which
                        # mainly reduces within-game Monte Carlo noise, a smaller contributor
                        # once thousands of games are aggregated).
N_GAMES_TO_VALIDATE = 25000  # effectively "every complete-lineup game", see
                             # validate_game_simulator.py's own docstring for this same constant
TEST_SEASONS = {2023, 2024, 2025}  # added 2023 (2026-07-22, critique claim 6), see
                                   # validate_game_simulator.py's own note on this same constant


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    print("building park factors...", flush=True)
    park_factors_long = build_outcome_park_factors(pa)
    park_factors_wide = park_factors_long.pivot(index=["team", "season"], columns="outcome", values="park_factor")
    hfa_factors_by_season = build_hfa_factors(pa)
    print("building wide pregame rate tables...", flush=True)
    # park-neutralized (see true_talent.build_pregame_rates) -- avoids double-
    # counting the park effect this project's own park_factors multiplier
    # applies below at simulate time.
    batter_wide = build_wide_pregame_rates(pa, "batter", park_factors_long)
    pitcher_wide = build_wide_pregame_rates(pa, "pitcher", park_factors_long)
    league_rates = build_league_rates_by_season(pa)
    print("building state factors...", flush=True)
    state_factors = build_state_factors_by_season(pa)
    print("building times-through-order-penalty factors...", flush=True)
    ttop_factors = build_ttop_factors_by_season(pa)
    print("building platoon multipliers...", flush=True)
    batter_platoon = build_wide_platoon_multipliers(pa, "batter").set_index(["batter", "season"])
    pitcher_platoon = build_wide_platoon_multipliers(pa, "pitcher").set_index(["pitcher", "season"])
    batter_hand = predominant_hand(pa, "batter")
    pitcher_hand = predominant_hand(pa, "pitcher")
    print("building batter pull-rate (spray angle) estimates...", flush=True)
    pull_rate_pa = build_pull_rate_by_season(pa)
    pull_rate_snapshot = build_pull_rate_snapshot(pa, pull_rate_pa)
    print("building weather factors...", flush=True)
    all_schedules = pd.concat(
        [pd.read_parquet(p) for p in sorted(DATA_RAW.glob("schedule_*.parquet"))], ignore_index=True
    )
    pa_with_weather = attach_weather_bucket(pa, all_schedules)
    pa_with_weather = attach_pull_tercile_column(pa_with_weather, pull_rate_pa)
    weather_factors_by_season = build_weather_factors_by_season(pa_with_weather)
    game_weather = all_schedules[["game_pk", "weather_condition", "weather_temp", "weather_wind"]].drop_duplicates(
        "game_pk"
    ).set_index("game_pk")
    print("building bullpen snapshot + expected starter innings...", flush=True)
    bullpen_snap = build_bullpen_snapshot(pa, park_factors_long)
    expected_innings = build_expected_starter_innings(pa).set_index(["pitcher", "game_pk"])["expected_innings"]
    print("building bullpen roster + rest-day availability logs...", flush=True)
    relief_log = build_relief_appearance_log(pa)
    all_appearance_log = build_all_appearance_log(pa)
    closer_log = build_closer_appearance_log(pa)
    print("building catcher framing factors + starting-catcher log...", flush=True)
    catcher_factors_by_season = build_catcher_framing_factors_by_season(sorted(pa["season"].unique()))
    catcher_log = build_catcher_appearance_log(pa)
    print("building umpire factors...", flush=True)
    umpire_factors_by_season = build_umpire_factors_by_season(sorted(pa["season"].unique()))
    umpire_log = load_umpire_game_log().set_index("game_pk")["ump_id"]
    print("building blowout/position-player-pitching profile...", flush=True)
    blowout_profile = build_position_player_profile(pa)

    print("building batter contact-quality (expected-stats) estimates...", flush=True)
    xbacon_snap = player_game_xbacon_snapshot(pa)
    barrel_snap = player_game_barrel_snapshot(pa)
    pulled_air_snap = player_game_pulled_air_snapshot(pa)
    bacon_gb_snap = player_game_bacon_gb_snapshot(pa)
    gb_rate_snap = player_game_groundball_rate_snapshot(pa)
    gbfb_pitcher_snap = player_game_gb_fb_rate_snapshot(pa)  # task #120, GB/FB pitcher HR-share -- KEPT, real full-stack win
    sprint_speed_by_season = {}  # memoized per season -- build_pregame_sprint_speed is season-level, not per-game

    print("building batter bat-tracking (bat speed) estimates...", flush=True)
    bat_speed_raw = load_bat_speed_by_season(sorted(pa["season"].unique()))
    bat_speed_by_season = {}  # memoized per season -- build_pregame_bat_speed is season-level, not per-game

    batter_snap = player_game_snapshot(batter_wide, "batter")
    pitcher_snap = player_game_snapshot(pitcher_wide, "pitcher")
    # nearest_prior_pitcher_snapshot needs game_date on every row -- a sampled
    # roster reliever (see build_team_bullpen_roster) may not have actually
    # pitched in this exact historical game_pk, unlike the two starters.
    pitcher_snap_dates = pitcher_snap.merge(pa[["game_pk", "game_date"]].drop_duplicates(), on="game_pk")

    transitions = TransitionTable(pa)
    rng = np.random.default_rng(42)

    schedules, lineups = {}, {}
    for season in TEST_SEASONS:
        schedules[season] = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
        lineups[season] = pd.read_parquet(DATA_RAW / f"lineups_{season}.parquet")

    print("building defense (OAA) factors...", flush=True)
    defense_snap_by_season = {
        season: team_game_defense_snapshot(lineups[season], schedules[season], season).set_index(["game_pk", "team"])
        for season in TEST_SEASONS
    }

    print("building stolen-base rates...", flush=True)
    season_sb_stats = build_season_sb_stats(pa)
    sb_rates_by_season = {season: build_pregame_sb_rates(season_sb_stats, season) for season in TEST_SEASONS}

    candidates = []
    for season in TEST_SEASONS:
        sched = schedules[season][(schedules[season]["game_type"] == "R") & (schedules[season]["status"] == "Final")]
        lu = lineups[season]
        lineup_counts = lu.groupby(["game_pk", "team_side"]).size()
        complete_games = lineup_counts[lineup_counts == 9].reset_index()["game_pk"].unique()
        sched = sched[sched["game_pk"].isin(complete_games)]
        for row in sched.sample(min(len(sched), N_GAMES_TO_VALIDATE // len(TEST_SEASONS)), random_state=1).itertuples():
            candidates.append((season, row.game_pk, row.home_score, row.away_score))

    print(f"validating on {len(candidates)} real games, {N_TRIALS_PER_GAME} simulated trials each...", flush=True)
    results = []
    for season, game_pk, actual_home, actual_away in candidates:
        lu = lineups[season]
        game_lu = lu[lu["game_pk"] == game_pk]
        home_ids = game_lu[game_lu["team_side"] == "home"].sort_values("batting_order")["player_id"].tolist()
        away_ids = game_lu[game_lu["team_side"] == "away"].sort_values("batting_order")["player_id"].tolist()
        if len(home_ids) != 9 or len(away_ids) != 9:
            continue

        bsnap = batter_snap[batter_snap["game_pk"] == game_pk].set_index("batter")
        psnap = pitcher_snap[pitcher_snap["game_pk"] == game_pk].set_index("pitcher")
        if not all(pid in bsnap.index for pid in home_ids + away_ids):
            continue

        xsnap = xbacon_snap[xbacon_snap["game_pk"] == game_pk].set_index("batter")
        barsnap = barrel_snap[barrel_snap["game_pk"] == game_pk].set_index("batter")
        pullairsnap = pulled_air_snap[pulled_air_snap["game_pk"] == game_pk].set_index("batter")
        bacsnap = bacon_gb_snap[bacon_gb_snap["game_pk"] == game_pk].set_index("batter")
        gbsnap = gb_rate_snap[gb_rate_snap["game_pk"] == game_pk].set_index("batter")
        gbfbsnap = gbfb_pitcher_snap[gbfb_pitcher_snap["game_pk"] == game_pk].set_index("pitcher")
        if season not in sprint_speed_by_season:
            sprint_speed_by_season[season] = build_pregame_sprint_speed(season)
        speed_snap = sprint_speed_by_season[season]
        if season not in bat_speed_by_season:
            bat_speed_by_season[season] = build_pregame_bat_speed(bat_speed_raw, season)
        bat_speed_snap = bat_speed_by_season[season]
        game_date = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "date"].iloc[0]

        def batter_profile(pid):
            row = bsnap.loc[pid]
            key = (pid, season)
            prow = batter_platoon.loc[key] if key in batter_platoon.index else None
            hand = batter_hand.get(pid, "R")
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
            # a sampled roster reliever may not have pitched in THIS exact
            # historical game_pk (unlike the two starters, whose row is
            # guaranteed since we only include games where they started).
            if pid in psnap.index:
                row = psnap.loc[pid]
            else:
                row = nearest_prior_pitcher_snapshot(pitcher_snap_dates, pid, game_date_, game_pk)
                if row is None:
                    return None
            key = (pid, season)
            prow = pitcher_platoon.loc[key] if key in pitcher_platoon.index else None
            # gb/fb snapshot is only keyed for THIS exact game_pk (unlike row
            # above, no nearest-prior fallback) -- a reliever missing here
            # just gets no-op (None), same graceful-degradation convention
            # as every other optional signal in this project.
            gb_p = gbfbsnap.loc[pid, "pregame_gb_rate_pitcher"] if pid in gbfbsnap.index else None
            fb_p = gbfbsnap.loc[pid, "pregame_fb_rate_pitcher"] if pid in gbfbsnap.index else None
            return build_profile(row, prow, pitcher_hand.get(pid, "R"),
                                  pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p)

        home_lineup = [batter_profile(pid) for pid in home_ids]
        away_lineup = [batter_profile(pid) for pid in away_ids]

        game_pa = pa[pa.game_pk == game_pk]
        top_pa = game_pa[game_pa.inning_topbot == "Top"].sort_values(["inning", "at_bat_number"])
        bot_pa = game_pa[game_pa.inning_topbot == "Bot"].sort_values(["inning", "at_bat_number"])
        home_pitcher_id = top_pa["pitcher"].iloc[0]
        away_pitcher_id = bot_pa["pitcher"].iloc[0]
        if home_pitcher_id not in psnap.index or away_pitcher_id not in psnap.index:
            continue
        home_pitcher = pitcher_profile(home_pitcher_id)
        away_pitcher = pitcher_profile(away_pitcher_id)

        home_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "home_team"].iloc[0]
        away_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "away_team"].iloc[0]

        park_key = (home_team, season)
        if park_key in park_factors_wide.index:
            park_row = park_factors_wide.loc[park_key]
            park_factors_this_game = {o: park_row.get(o, 1.0) if pd.notna(park_row.get(o, 1.0)) else 1.0 for o in OUTCOMES}
        else:
            park_factors_this_game = None

        # weather: this game's actual recorded conditions, bucketed the same
        # way the walk-forward factor table was built (see weather.py) --
        # real for a backtest, but also genuinely usable live since MLB
        # reports forecast/gametime conditions before first pitch, unlike
        # bullpen usage (see validate_game_simulator.py, same reasoning).
        weather_factors_this_game = None
        if game_pk in game_weather.index:
            wx = game_weather.loc[game_pk]
            bucket = bucket_weather(wx["weather_condition"], wx["weather_temp"], wx["weather_wind"])
            weather_factors_this_game = weather_factors_by_season.get(season, {}).get(bucket)

        # PREDICTIVE bullpen: starter's own walk-forward-projected expected
        # innings, then a SPECIFIC reliever sampled per inning -- weighted by
        # recent usage + rest-day availability for this exact date (see
        # build_team_bullpen_roster) -- never looking at what actually
        # happened in this specific game. Falls back to the old flat pooled
        # aggregate rate (see bullpen_snap) only if no roster history exists.
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

        # PREDICTIVE starting catcher: whoever caught most games for this team
        # recently as of game_date (see identify_starting_catcher) -- never
        # looking at who actually caught THIS specific game, same "no
        # foresight" discipline as the predictive bullpen above.
        home_catcher_id = identify_starting_catcher(catcher_log, home_team, game_date)
        away_catcher_id = identify_starting_catcher(catcher_log, away_team, game_date)
        home_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, home_catcher_id)
        away_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, away_catcher_id)

        # real home-plate umpire this game -- this is a backtest of an
        # actual played game (only the BULLPEN usage is predictive here, not
        # the umpire), so the real historical assignment is known and used
        # directly, same as validate_game_simulator.py's oracle path.
        ump_id = umpire_log.get(game_pk)
        umpire_factor = resolve_umpire_factor(umpire_factors_by_season, season, ump_id)
        home_catcher_factor = {o: home_catcher_factor[o] * umpire_factor[o] for o in home_catcher_factor}
        away_catcher_factor = {o: away_catcher_factor[o] * umpire_factor[o] for o in away_catcher_factor}

        # real defensive alignment this game (see defense_factor.py) --
        # disjoint outcome keys (single/double/triple) from catcher's
        # strikeout/walk, so a plain dict merge is safe.
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

        # real per-runner stolen-base attempt/success rates this game (see
        # baserunning.py) -- the BATTING team's own skill, keyed by lineup index.
        home_sb_rates = resolve_sb_rates(sb_rates_by_season[season], home_ids)
        away_sb_rates = resolve_sb_rates(sb_rates_by_season[season], away_ids)

        sim = GameSimulator(transitions, league_rates[season], rng, state_factors=state_factors[season],
                            ttop_factors=ttop_factors[season], shock_sigma=SHOCK_SIGMA)
        sim_home, sim_away = [], []
        for _ in range(N_TRIALS_PER_GAME):
            home_bullpen_plan, _ = sample_bullpen_plan(
                rng, home_pitcher, home_exp_ip, home_roster_weights,
                lambda pid: home_roster_profiles[pid], home_fallback, closer_id=home_closer_id,
            )
            away_bullpen_plan, _ = sample_bullpen_plan(
                rng, away_pitcher, away_exp_ip, away_roster_weights,
                lambda pid: away_roster_profiles[pid], away_fallback, closer_id=away_closer_id,
            )
            h, a = sim.simulate_game(
                home_lineup, away_lineup, home_pitcher, away_pitcher, innings=18,
                park_factors=park_factors_this_game,
                weather_factors=weather_factors_this_game,
                home_bullpen=home_bullpen_plan, away_bullpen=away_bullpen_plan,
                blowout_pitcher_profile=blowout_profile,
                home_catcher_factor=home_catcher_factor, away_catcher_factor=away_catcher_factor,
                home_sb_rates=home_sb_rates, away_sb_rates=away_sb_rates,
                hfa_factors=hfa_factors_by_season.get(season),
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
        print(f"  game {game_pk}: actual {actual_away}-{actual_home}  "
              f"sim {np.mean(sim_away):.1f}-{np.mean(sim_home):.1f} "
              f"(std {np.std(sim_away):.1f}/{np.std(sim_home):.1f})", flush=True)

    r = pd.DataFrame(results)
    r.to_parquet(DATA_PROCESSED / "predictive_bullpen_validation.parquet", index=False)
    print(f"\n=== validated on {len(r)} real games ===")
    print(f"home score MAE: {(r['sim_home_mean']-r['actual_home']).abs().mean():.3f}")
    print(f"away score MAE: {(r['sim_away_mean']-r['actual_away']).abs().mean():.3f}")
    r["actual_total"] = r["actual_home"] + r["actual_away"]
    r["sim_total_mean"] = r["sim_home_mean"] + r["sim_away_mean"]
    print(f"total score MAE: {(r['sim_total_mean']-r['actual_total']).abs().mean():.3f}")
    print(f"mean actual total: {r['actual_total'].mean():.2f}   mean simulated total: {r['sim_total_mean'].mean():.2f}")
    r["actual_margin"] = r["actual_home"] - r["actual_away"]
    r["sim_margin_mean"] = r["sim_home_mean"] - r["sim_away_mean"]
    print(f"margin MAE: {(r['sim_margin_mean']-r['actual_margin']).abs().mean():.3f}")
    r["actual_home_win"] = (r["actual_home"] > r["actual_away"]).astype(int)
    # SU + Brier/log-loss, matching validate_game_simulator.py's 2026-07-22 update --
    # trial-level home-win share (sim_home_win_prob, already computed above per game)
    # as the PRIMARY SU metric, not sign of the mean simulated margin (see that file's
    # own note re: walk-off truncation skewing the mean margin's distribution).
    print(f"straight-up accuracy (trial-level home-win share > 0.5, PRIMARY): "
          f"{((r['sim_home_win_prob']>0.5).astype(int)==r['actual_home_win']).mean():.3f}")
    print(f"straight-up accuracy (sign of mean simulated margin matches actual, legacy metric): "
          f"{(np.sign(r['sim_margin_mean'])==np.sign(r['actual_margin'])).mean():.3f}")
    r["brier"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
    p_clipped = r["sim_home_win_prob"].clip(1e-6, 1 - 1e-6)
    r["log_loss"] = -(r["actual_home_win"] * np.log(p_clipped) + (1 - r["actual_home_win"]) * np.log(1 - p_clipped))
    print(f"Brier score (P(home win) vs actual, lower=better): {r['brier'].mean():.4f}")
    print(f"log loss (P(home win) vs actual, lower=better): {r['log_loss'].mean():.4f}")
