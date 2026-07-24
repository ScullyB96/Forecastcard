"""The real test: simulate actual historical games (real lineups, real
starting pitchers, each player's walk-forward pregame true-talent snapshot
as of that game) many times each, and compare the simulated score
distribution to what actually happened. Everything upstream (PA table,
transition table, true-talent rates, matchup combination, platoon splits)
was validated piece by piece already -- this is the first test of the FULL
chain together.

First-version scope, clearly flagged: bullpen usage replays each real game's
ACTUAL sequence of which pitcher started each inning (own walk-forward
pregame snapshot per reliever) -- a faithful backtest of real usage, not a
predictive model of pitching-change decisions (mid-inning removals aren't
modeled, and a live/future-game version will need a genuine "expected bullpen
usage" model instead of a real sequence that doesn't exist yet). Weather uses
each game's ACTUAL recorded conditions (see weather.py) -- real for a
backtest, but also genuinely usable live since MLB reports forecast/gametime
conditions before first pitch, unlike bullpen usage. Only games with complete
9-batter lineup data on both sides are used. Switch-hitters are correctly
modeled as always batting opposite the pitcher's hand (see predominant_hand
below), not approximated by a fixed side.
"""

import numpy as np
import pandas as pd

from src.models.base_out_transitions import TransitionTable
from src.models.blowout import build_position_player_profile
from src.models.catcher_framing import build_catcher_framing_factors_by_season, resolve_catcher_factor
from src.models.expected_stats import (
    apply_contact_quality,
    build_pregame_bat_speed,
    build_pregame_sprint_speed,
    load_bat_speed_by_season,
    player_game_barrel_snapshot,
    player_game_bacon_gb_snapshot,
    player_game_gb_fb_rate_snapshot,
    player_game_groundball_rate_snapshot,
    player_game_pulled_air_snapshot,
    player_game_xbacon_snapshot,
)
from src.models.umpire_factor import build_umpire_factors_by_season, load_umpire_game_log, resolve_umpire_factor
from src.models.defense_factor import resolve_defense_factor, team_game_defense_snapshot
from src.models.baserunning import build_season_sb_stats, build_pregame_sb_rates, resolve_sb_rates
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
from src.models.true_talent import widen_rate
from src.models.ttop import build_ttop_factors_by_season
from src.models.weather import attach_weather_bucket, bucket_weather, build_weather_factors_by_season
from src.utils.paths import DATA_PROCESSED, DATA_RAW

N_TRIALS_PER_GAME = 50  # lowered from 200 (2026-07-22), see N_GAMES_TO_VALIDATE below --
                        # deliberate tradeoff: per-game trial count mainly reduces WITHIN-game
                        # Monte Carlo noise on one game's own sim_home_win_prob estimate (SE
                        # ~sqrt(0.25/50) =~ 7% per game), a smaller, largely-averaging-out
                        # contributor once thousands of independent games are aggregated --
                        # unlike BETWEEN-game sample size (game count), which is what actually
                        # gates resolving the ~1pp-scale effects §11.7 found unresolvable at
                        # n=597-995. Prioritizing game count over trials/game keeps this
                        # ~12x-larger protocol's runtime tractable (~7300 games * 50 trials is
                        # a similar total trial count to the OLD 600-game/200-trial protocol
                        # run ~1.2x, not 12x, more compute).
N_GAMES_TO_VALIDATE = 25000  # effectively "every complete-lineup game", not a cap --
                             # 2023+2024+2025 combined have ~7277 real complete-9-batter-lineup
                             # regular season games total (confirmed via direct count,
                             # 2026-07-22); this number just needs to exceed that so the
                             # min(len(sched), ...) below always resolves to len(sched) (use
                             # every available game that season, no subsampling at all).
WIDEN_W = 1.0  # task #134 (Phase 1 pivot): rate-spread-restoring stretch factor, see
               # true_talent.widen_rate. 1.0 = no-op -- this fix did NOT work (see
               # MODEL_DOCUMENTATION.md sec 11.13, a clean negative result) and was never
               # deployed. Scratchpad sweep scripts import build_profile directly and
               # override this per-run rather than editing it here.
SHOCK_SIGMA = 0.40  # task #137 (Phase 1, the latent per-pitcher-appearance shock): FROZEN
                    # production value, 2026-07-23. Fit in-sim on 2023-2024 (std(z)=1.0 crossing
                    # ~0.40-0.44), validated on 2025 held out (std(z) 1.154->1.033, tail coverage
                    # improved, SU/Brier/MAE within noise under CRN pairing), confirmed on the
                    # canonical n=7237 protocol (std(z) 1.136->1.015 -- the best dispersion
                    # result of the whole session) with a small SU/Brier/MAE dip that a
                    # follow-up K-scaling check (K=30/100/300) showed shrinking toward zero as
                    # K grows -- a K=50 estimator artifact, not a real model cost. See
                    # MODEL_DOCUMENTATION.md sec 11.14 for the full story before ever changing
                    # this constant -- in particular, this value is a SIMULATOR-INTERNAL
                    # calibration constant (attenuated by odds-space renormalization + game-flow
                    # damping), not a literal claim about real pitcher day-to-day variance, and
                    # it may be absorbing other missing variance (e.g. a batter-side day-effect,
                    # if ever added) -- refit down at that point, don't just stack a second
                    # mechanism on top of this one unchanged.
TEST_SEASONS = {2023, 2024, 2025}  # added 2023 (2026-07-22, critique claim 6) -- all under the
                                   # identical 2023+ rules regime (pitch clock, shift ban, big
                                   # bases, zombie runner), giving ~6500-7000+ games instead of
                                   # the previous ~600-1000-game sampled subset, specifically to
                                   # resolve marginal (~1pp-scale) effects §11.7 found were not
                                   # resolvable at the smaller sample sizes. NOTE: 2023 is this
                                   # project's own data's true cold-start season (true_talent.py
                                   # has zero prior-season history to build a Marcel prior from),
                                   # so EARLY-season 2023 games specifically get weaker
                                   # (league-average-only) preseason priors than 2024/2025 games
                                   # do -- an accepted, unavoidable limitation of the data window,
                                   # not a new bug; the in-season blend still improves as each
                                   # 2023 game's season progresses, same as any other season.


def player_game_snapshot(wide: pd.DataFrame, player_col: str) -> pd.DataFrame:
    """Each player's rate as of their FIRST PA in a given game -- the fixed
    pregame skill snapshot to use for a full from-scratch simulated replay
    of that game (not an evolving mid-game estimate). Also carries
    effective_n_{outcome} alongside each pregame_rate_{outcome} (task #127,
    posterior-sampled rates) -- see build_wide_pregame_rates."""
    rate_cols = [f"pregame_rate_{o}" for o in OUTCOMES]
    effective_n_cols = [f"effective_n_{o}" for o in OUTCOMES]
    first_pa = wide.sort_values("at_bat_number").groupby(["game_pk", player_col], as_index=False).nth(0)
    return first_pa[["game_pk", player_col, "season"] + rate_cols + effective_n_cols]


def predominant_hand(pa: pd.DataFrame, player_col: str) -> pd.Series:
    """Each player's batting/throwing side. Real switch-hitters (batters
    only -- pitchers don't throw ambidextrously in any meaningful numbers)
    are correctly detected from a genuinely mixed stand history (>20%
    minority side, confirmed matching the real ~8.7% MLB switch-hitter rate)
    and marked "S" rather than guessed at a fixed side -- a switch-hitter, by
    definition, ALWAYS bats opposite the pitcher's hand, so this is the
    simpler and more correct rule, not a simplification. An earlier version
    of this check had a bug (value_counts().min() on a single-category
    series trivially returns 1.0) that misclassified 98.6% of batters as
    switch-hitters -- caught before trusting it, see project memory."""
    hand_col = "stand" if player_col == "batter" else "p_throws"
    predominant = pa.groupby(player_col)[hand_col].agg(lambda s: s.mode().iloc[0])
    if player_col != "batter":
        return predominant

    def minority_share(s):
        vc = s.value_counts(normalize=True)
        return vc.min() if len(vc) > 1 else 0.0

    is_switch = pa.groupby(player_col)[hand_col].agg(minority_share) > 0.2
    predominant[is_switch[is_switch].index] = "S"
    return predominant


def build_profile(rate_row: pd.Series, platoon_row: pd.Series | None, hand: str, pull_tercile: str | None = None,
                   pregame_xbacon: float | None = None, pregame_barrel_rate: float | None = None,
                   pregame_bacon_gb: float | None = None, pregame_sprint_speed: float | None = None,
                   pregame_gb_rate: float | None = None, pregame_bat_speed: float | None = None,
                   pregame_pulled_air_rate: float | None = None,
                   pregame_gb_rate_pitcher: float | None = None,
                   pregame_fb_rate_pitcher: float | None = None,
                   pitcher_fullmix: bool = False,
                   league_rates_this_season: dict[str, float] | None = None,
                   widen_w: float = 1.0) -> dict:
    rates = {o: rate_row[f"pregame_rate_{o}"] for o in OUTCOMES}
    # effective_n backing each raw Marcel rate (task #127, posterior-sampled
    # rates) -- stored alongside the (possibly contact-quality-adjusted)
    # rates below so a caller can later draw a Beta-posterior sample instead
    # of always using the point estimate. Approximation, clearly flagged: the
    # adjustment multipliers applied below (contact quality, HR-share, GB/FB)
    # have their own uncertainty not separately tracked here -- effective_n
    # reflects confidence in the RAW Marcel rate only, applied as a proxy for
    # overall confidence to the final adjusted rate when resampling.
    effective_n = {o: rate_row[f"effective_n_{o}"] for o in OUTCOMES}
    # Task #134 (Phase 1 pivot, spread-restoring fix): stretch the raw
    # walk-forward Marcel rate away from league average BEFORE the
    # contact-quality layer runs, matching where the rate-spread compression
    # was actually measured (real season-long rates are 1.4-1.9x more spread
    # across players than these pregame rates -- see true_talent.widen_rate).
    # widen_w=1.0 (default) is a byte-for-byte no-op vs. today's behavior.
    if widen_w != 1.0 and league_rates_this_season is not None:
        rates = {o: widen_rate(rates[o], league_rates_this_season[o], widen_w) for o in OUTCOMES}
    # contact-quality (expected-stats) adjustment is a BATTER-only, opponent-
    # independent correction to the batter's own true-talent rates -- see
    # expected_stats.py. pitcher-profile callers never pass any of these,
    # leaving them None, which apply_contact_quality treats as a no-op for
    # each layer independently. pregame_gb_rate_pitcher/pregame_fb_rate_pitcher
    # are the mirror image (task #120): only PITCHER-profile callers pass
    # these, batter-profile callers leave them None. pitcher_fullmix (task
    # #126, default False): the confirmed HR-only layer stays live either
    # way; set True to additionally test the double/double_play extensions.
    rates = apply_contact_quality(rates, pregame_xbacon, pregame_barrel_rate,
                                   pregame_bacon_gb, pregame_sprint_speed, pregame_gb_rate,
                                   pregame_bat_speed=pregame_bat_speed,
                                   pregame_pulled_air_rate=pregame_pulled_air_rate,
                                   pregame_gb_rate_pitcher=pregame_gb_rate_pitcher,
                                   pregame_fb_rate_pitcher=pregame_fb_rate_pitcher,
                                   pitcher_fullmix=pitcher_fullmix)
    if platoon_row is not None:
        same_mult = {o: platoon_row[f"same_hand_mult_{o}"] for o in OUTCOMES}
        opp_mult = {o: platoon_row[f"opp_hand_mult_{o}"] for o in OUTCOMES}
    else:
        same_mult = opp_mult = {o: 1.0 for o in OUTCOMES}
    # pull_tercile is a BATTER-only concept (see spray.py) -- pitcher-profile
    # callers simply never pass it, leaving it None; game_simulator.py treats
    # a None/missing tercile as "mid_pull" (the neutral default) when looking
    # up the weather factor for the PITCHER side of a matchup, which never
    # actually needs it (only the batter's own tercile modulates the wind
    # effect on their batted balls).
    return {"rates": rates, "hand": hand, "same_mult": same_mult, "opp_mult": opp_mult,
            "pull_tercile": pull_tercile, "effective_n": effective_n}


def build_shared_tables(pa: pd.DataFrame, test_seasons: set[int]) -> dict:
    """Every walk-forward/history-dependent table this validator needs, built
    ONCE from the full PA table. Kept separate from run_validation (below) so
    a caller sweeping a single per-profile parameter (e.g. task #134's widen_w)
    never has to pay the cost of rebuilding history-dependent tables that
    parameter doesn't affect -- only the per-game profile-build + simulate
    step in run_validation needs to repeat per sweep point."""
    print("building park factors...", flush=True)
    park_factors_long = build_outcome_park_factors(pa)
    park_factors_wide = park_factors_long.pivot(index=["team", "season"], columns="outcome", values="park_factor")
    print("building home-field-advantage factors...", flush=True)
    hfa_factors_by_season = build_hfa_factors(pa)
    # park-neutralize batter/pitcher rates BEFORE game_simulator.py's own
    # per-game park_factors multiplier acts on them below -- otherwise a
    # park effect gets double-counted (see true_talent.build_pregame_rates).
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

    print("building blowout/position-player-pitching profile...", flush=True)
    blowout_profile = build_position_player_profile(pa)

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

    print("building catcher framing factors...", flush=True)
    catcher_factors_by_season = build_catcher_framing_factors_by_season(sorted(pa["season"].unique()))

    print("building umpire factors...", flush=True)
    umpire_factors_by_season = build_umpire_factors_by_season(sorted(pa["season"].unique()))
    umpire_log = load_umpire_game_log().set_index("game_pk")["ump_id"]

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

    transitions = TransitionTable(pa)

    schedules, lineups = {}, {}
    for season in test_seasons:
        schedules[season] = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
        lineups[season] = pd.read_parquet(DATA_RAW / f"lineups_{season}.parquet")

    print("building defense (OAA) factors...", flush=True)
    defense_snap_by_season = {
        season: team_game_defense_snapshot(lineups[season], schedules[season], season).set_index(["game_pk", "team"])
        for season in test_seasons
    }

    print("building stolen-base rates...", flush=True)
    season_sb_stats = build_season_sb_stats(pa)
    sb_rates_by_season = {season: build_pregame_sb_rates(season_sb_stats, season) for season in test_seasons}

    return dict(
        pa=pa, park_factors_long=park_factors_long, park_factors_wide=park_factors_wide,
        hfa_factors_by_season=hfa_factors_by_season, league_rates=league_rates,
        state_factors=state_factors, ttop_factors=ttop_factors,
        batter_platoon=batter_platoon, pitcher_platoon=pitcher_platoon,
        batter_hand=batter_hand, pitcher_hand=pitcher_hand, blowout_profile=blowout_profile,
        pull_rate_snapshot=pull_rate_snapshot, weather_factors_by_season=weather_factors_by_season,
        game_weather=game_weather, catcher_factors_by_season=catcher_factors_by_season,
        umpire_factors_by_season=umpire_factors_by_season, umpire_log=umpire_log,
        xbacon_snap=xbacon_snap, barrel_snap=barrel_snap, pulled_air_snap=pulled_air_snap,
        bacon_gb_snap=bacon_gb_snap, gb_rate_snap=gb_rate_snap, gbfb_pitcher_snap=gbfb_pitcher_snap,
        sprint_speed_by_season=sprint_speed_by_season, bat_speed_raw=bat_speed_raw,
        bat_speed_by_season=bat_speed_by_season, batter_snap=batter_snap, pitcher_snap=pitcher_snap,
        transitions=transitions, schedules=schedules, lineups=lineups,
        defense_snap_by_season=defense_snap_by_season, sb_rates_by_season=sb_rates_by_season,
    )


def run_validation(shared: dict, test_seasons: set[int], n_games: int, n_trials: int,
                    widen_w: float = 1.0, seed: int = 42, output_path=None,
                    write_ledger: bool = True, notes: str = "validate_game_simulator.py oracle backtest",
                    verbose: bool = True, shock_sigma: float = 0.0, crn_pairing: bool = False,
                    trial_capture: dict | None = None) -> pd.DataFrame:
    """Run the real-games oracle backtest against `shared` (from
    build_shared_tables). widen_w=1.0, shock_sigma=0.0, crn_pairing=False
    (all defaults) + seed=42 reproduces this file's own historical default
    run byte-for-byte; task #134's sweep script calls this repeatedly with
    different widen_w, task #137's sweep calls it with different shock_sigma.

    shock_sigma: passed straight to GameSimulator's own shock_sigma (task
    #137, the latent per-pitcher-appearance shock) -- 0.0 is an exact no-op.
    crn_pairing: when True, threads crn_game_pk=game_pk/crn_trial=<trial
    index> into every sim.simulate_game call, so a shock_sigma sweep compares
    CRN-paired arms (every decision besides the shock itself stays
    synchronized) instead of unpaired independent Monte Carlo runs -- see
    crn.py. False (the default) matches every existing caller's behavior
    exactly (no CRN keys passed, sequential rng stream throughout)."""
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
    rng = np.random.default_rng(seed)

    # sample real, complete games from the test seasons
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
        print(f"validating on {len(candidates)} real games, {n_trials} simulated trials each "
              f"(widen_w={widen_w})...", flush=True)
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
            continue  # a starter had zero PAs recorded this game (rare data edge case) -- skip

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
                                  pregame_bat_speed=bat_speed, pregame_pulled_air_rate=pulled_air,
                                  league_rates_this_season=league_rates[season], widen_w=widen_w)

        def pitcher_profile(pid):
            row = psnap.loc[pid]
            key = (pid, season)
            prow = pitcher_platoon.loc[key] if key in pitcher_platoon.index else None
            gb_p = gbfbsnap.loc[pid, "pregame_gb_rate_pitcher"] if pid in gbfbsnap.index else None
            fb_p = gbfbsnap.loc[pid, "pregame_fb_rate_pitcher"] if pid in gbfbsnap.index else None
            return build_profile(row, prow, pitcher_hand.get(pid, "R"),
                                  pregame_gb_rate_pitcher=gb_p, pregame_fb_rate_pitcher=fb_p,
                                  league_rates_this_season=league_rates[season], widen_w=widen_w)

        home_lineup = [batter_profile(pid) for pid in home_ids]
        away_lineup = [batter_profile(pid) for pid in away_ids]
        # starting pitcher = whoever pitched the 1st PA of each team's defense
        game_pa = pa[pa.game_pk == game_pk]
        top_pa = game_pa[game_pa.inning_topbot == "Top"].sort_values(["inning", "at_bat_number"])
        bot_pa = game_pa[game_pa.inning_topbot == "Bot"].sort_values(["inning", "at_bat_number"])
        home_pitcher_id = top_pa["pitcher"].iloc[0]
        away_pitcher_id = bot_pa["pitcher"].iloc[0]
        if home_pitcher_id not in psnap.index or away_pitcher_id not in psnap.index:
            continue
        home_pitcher = pitcher_profile(home_pitcher_id)
        away_pitcher = pitcher_profile(away_pitcher_id)

        # bullpen: the REAL historical sequence of who started each inning on
        # the mound for each team in this actual game -- a faithful backtest
        # of real bullpen usage (see GameSimulator._pitcher_for_inning), not a
        # predictive model of pitching-change decisions. Pitchers with no
        # usable pregame snapshot (e.g. a rookie's MLB debut with zero prior
        # PA history) are skipped -- the fallback then reuses the most recent
        # known pitcher, never a guess.
        home_pitcher_by_inning = top_pa.groupby("inning")["pitcher"].first()
        away_pitcher_by_inning = bot_pa.groupby("inning")["pitcher"].first()
        home_bullpen = {int(inn): pitcher_profile(pid) for inn, pid in home_pitcher_by_inning.items() if pid in psnap.index}
        away_bullpen = {int(inn): pitcher_profile(pid) for inn, pid in away_pitcher_by_inning.items() if pid in psnap.index}

        home_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "home_team"].iloc[0]
        away_team = schedules[season].loc[schedules[season]["game_pk"] == game_pk, "away_team"].iloc[0]
        park_key = (home_team, season)
        if park_key in park_factors_wide.index:
            park_row = park_factors_wide.loc[park_key]
            park_factors_this_game = {o: park_row.get(o, 1.0) if pd.notna(park_row.get(o, 1.0)) else 1.0 for o in OUTCOMES}
        else:
            park_factors_this_game = None  # no prior-season data yet (e.g. 2023) -- no adjustment, not a guess

        # weather: this game's actual recorded conditions, bucketed the same
        # way the walk-forward factor table was built (see weather.py).
        weather_factors_this_game = None
        if game_pk in game_weather.index:
            wx = game_weather.loc[game_pk]
            bucket = bucket_weather(wx["weather_condition"], wx["weather_temp"], wx["weather_wind"])
            weather_factors_this_game = weather_factors_by_season.get(season, {}).get(bucket)

        # real starting catchers this game (this is a backtest of an actual
        # played game, so the real catcher is known -- not a projection).
        home_catcher_id = top_pa["catcher"].iloc[0]
        away_catcher_id = bot_pa["catcher"].iloc[0]
        home_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, home_catcher_id)
        away_catcher_factor = resolve_catcher_factor(catcher_factors_by_season, season, away_catcher_id)

        # real home-plate umpire this game -- this oracle backtest looks up
        # the actual historical assignment directly (props.py's live path
        # uses resolve_live_umpire_factor instead; see umpire_factor.py's
        # module docstring). Applies identically to BOTH teams' batting (one
        # umpire calls the whole game), unlike catcher which is team-specific
        # -- multiplied into both catcher-factor dicts rather than threaded
        # as a separate simulator parameter, since it operates on the exact
        # same categories via the exact same mechanism.
        ump_id = umpire_log.get(game_pk)
        umpire_factor = resolve_umpire_factor(umpire_factors_by_season, season, ump_id)
        home_catcher_factor = {o: home_catcher_factor[o] * umpire_factor[o] for o in home_catcher_factor}
        away_catcher_factor = {o: away_catcher_factor[o] * umpire_factor[o] for o in away_catcher_factor}

        # real defensive alignment this game (see defense_factor.py) --
        # disjoint outcome keys (single/double/triple) from catcher/umpire's
        # strikeout/walk, so a plain dict merge is safe. home_catcher_factor
        # is applied when the HOME team is fielding (see game_simulator.py's
        # simulate_game), so it gets HOME's own defense composite, and vice versa.
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
        # baserunning.py) -- unlike catcher/defense/park, this is the BATTING
        # team's own skill, keyed by lineup index (game_simulator.py tracks
        # runner identity by lineup slot, not player_id).
        home_sb_rates = resolve_sb_rates(sb_rates_by_season[season], home_ids)
        away_sb_rates = resolve_sb_rates(sb_rates_by_season[season], away_ids)

        sim = GameSimulator(transitions, league_rates[season], rng, state_factors=state_factors[season],
                            ttop_factors=ttop_factors[season], shock_sigma=shock_sigma)
        sim_home, sim_away = [], []
        for trial in range(n_trials):
            crn_kwargs = {"crn_game_pk": game_pk, "crn_trial": trial} if crn_pairing else {}
            h, a = sim.simulate_game(
                home_lineup, away_lineup, home_pitcher, away_pitcher, innings=18,
                park_factors=park_factors_this_game,
                weather_factors=weather_factors_this_game,
                home_bullpen=home_bullpen, away_bullpen=away_bullpen,
                blowout_pitcher_profile=blowout_profile,
                home_catcher_factor=home_catcher_factor, away_catcher_factor=away_catcher_factor,
                home_sb_rates=home_sb_rates, away_sb_rates=away_sb_rates,
                hfa_factors=hfa_factors_by_season.get(season),
                **crn_kwargs,
            )
            sim_home.append(h)
            sim_away.append(a)

        sim_home_arr, sim_away_arr = np.array(sim_home), np.array(sim_away)
        if trial_capture is not None:
            # raw per-trial arrays (task #137's K-scaling check) -- since CRN trial
            # draws are a pure function of trial index, a caller can compute
            # K-subset statistics (first K of these arrays) post-hoc for several K
            # values from ONE run at the largest K, instead of rerunning per K.
            trial_capture[game_pk] = (sim_home_arr, sim_away_arr, actual_home, actual_away)
        results.append({
            "game_pk": game_pk, "actual_home": actual_home, "actual_away": actual_away,
            "sim_home_mean": np.mean(sim_home), "sim_away_mean": np.mean(sim_away),
            "sim_home_std": np.std(sim_home), "sim_away_std": np.std(sim_away),
            "sim_home_win_prob": (sim_home_arr > sim_away_arr).mean(),
        })
        if verbose:
            print(f"  game {game_pk}: actual {actual_away}-{actual_home}  "
                  f"sim {np.mean(sim_away):.1f}-{np.mean(sim_home):.1f} "
                  f"(std {np.std(sim_away):.1f}/{np.std(sim_home):.1f})", flush=True)

    r = pd.DataFrame(results)
    if output_path is not None:
        r.to_parquet(output_path, index=False)
    print(f"\n=== validated on {len(r)} real games (widen_w={widen_w}) ===")
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
    # PRIMARY SU metric (2026-07-22, critique claim 6): pick the side with
    # trial-level home-win SHARE > 0.5 (sim_home_win_prob, already computed
    # per game below for Brier score), not the sign of the mean simulated
    # margin. Walk-off truncation (a home win ends the instant they take the
    # lead, cutting off what would otherwise be a longer, higher-variance
    # right tail) skews the mean margin's distribution in a way the trial-level
    # win share doesn't share, since win share is a direct empirical estimate
    # of P(home wins) from the actual simulated outcomes, not a derived
    # statistic of the mean. Kept the OLD sign-of-mean-margin metric printed
    # alongside for direct comparison/transparency, not silently replaced.
    print(f"straight-up accuracy (trial-level home-win share > 0.5, PRIMARY): "
          f"{((r['sim_home_win_prob']>0.5).astype(int)==r['actual_home_win']).mean():.3f}")
    print(f"straight-up accuracy (sign of mean simulated margin matches actual, legacy metric): "
          f"{(np.sign(r['sim_margin_mean'])==np.sign(r['actual_margin'])).mean():.3f}")
    # Brier score / log-loss on P(home win) -- a CONTINUOUS metric using every
    # game's full simulated win-probability distribution, not just whether the
    # sign of the mean margin happened to match. Added 2026-07-22 after an
    # external methodology critique: straight-up accuracy at n~600 has a
    # binomial SE of ~2pp (95% CI +/-4pp on a single arm, wider still on an
    # A/B delta since the two arms are NOT actually paired -- see the note on
    # rng below), so a lot of this session's keep/revert calls were decided on
    # deltas smaller than the metric's own noise floor. Brier score uses the
    # full continuous sim_home_win_prob (already computed above, no extra
    # simulation cost) against the real binary outcome, giving a much tighter
    # comparison at the same sample size. (actual_home_win computed above.)
    r["brier"] = (r["sim_home_win_prob"] - r["actual_home_win"]) ** 2
    p_clipped = r["sim_home_win_prob"].clip(1e-6, 1 - 1e-6)
    r["log_loss"] = -(r["actual_home_win"] * np.log(p_clipped) + (1 - r["actual_home_win"]) * np.log(1 - p_clipped))
    print(f"Brier score (P(home win) vs actual, lower=better): {r['brier'].mean():.4f}")
    print(f"log loss (P(home win) vs actual, lower=better): {r['log_loss'].mean():.4f}")
    # NOTE on RNG pairing: `rng` (created once, above the per-game loop) is
    # reused sequentially across every game AND every trial in this run. Two
    # runs of this script that differ in only ONE tested factor will diverge
    # in their random draws from the first differing outcome onward -- i.e.
    # they are NOT a paired comparison of the same 597 games' random draws,
    # they are two independent Monte Carlo realizations that merely start
    # aligned. Any A/B delta's true uncertainty is closer to the UNPAIRED
    # combination of both arms' own noise than a naive "same seed" framing
    # suggests -- do not treat a small point-estimate delta as dispositive
    # without a bootstrap CI (see the paired-bootstrap comparison tool).

    # Home-field-advantage calibration check (Phase 0.2, 2026-07-22 roadmap):
    # does the simulator's own mean win probability match the real home win
    # rate, or is there still a gap (the original §11.8 diagnostic that
    # motivated the HFA fix in the first place)?
    print(f"mean sim_home_win_prob: {r['sim_home_win_prob'].mean():.4f}   "
          f"actual home win rate: {r['actual_home_win'].mean():.4f}")

    # PIT/z-score dispersion diagnostic (§11.8 claim 5 / §11.11) -- promoted
    # from a one-off scratch check into the standard summary output so every
    # future run reports it automatically instead of requiring a bespoke
    # script. z should be ~N(0,1) if the simulator's own trial-to-trial
    # spread matches reality; std(z) > 1 and tail-frequency inflation both
    # indicate under-dispersion.
    combined_std = np.sqrt(r["sim_home_std"] ** 2 + r["sim_away_std"] ** 2)
    z = (r["actual_total"] - r["sim_total_mean"]) / combined_std.replace(0, np.nan)
    print(f"PIT/z-score dispersion check: std(z)={z.std():.4f} (target ~1.0)  "
          f"frac|z|>1.96={( z.abs()>1.96).mean():.4f} (target ~0.05)  "
          f"frac|z|>2.58={(z.abs()>2.58).mean():.4f} (target ~0.01)  "
          f"13+ run games actual={(r['actual_total']>=13).mean():.4f}")

    # Persisted metrics ledger (Phase 0.1, 2026-07-22 roadmap) -- every run
    # durably logs its own config + result, so a future comparison never has
    # to guess (or get wrong, see §11.9) which codebase state a saved
    # *_validation.parquet came from. Sweep-style callers (task #134) pass
    # write_ledger=False -- dozens of small-sample sweep points would just be
    # noise in a ledger meant for genuine keep/revert decisions.
    if write_ledger:
        from src.models.metrics_ledger import append_run
        ledger_row = append_run(
            r, config_flags={
                "test_seasons": sorted(test_seasons), "n_games_requested": n_games,
                "hfa": True, "park_neutral": True, "gbfb_pitcher": True, "gbfb_fullmix": False,
                "rookie_prior": False, "posterior_sample": False, "crn_enabled": False,
                "widen_w": widen_w,
            },
            n_trials=n_trials, notes=notes,
        )
        print(f"logged to metrics ledger: {ledger_row['timestamp']} (git {ledger_row['git_hash'][:8]}"
              f"{'*' if ledger_row['git_dirty'] else ''})")
    return r


if __name__ == "__main__":
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    shared = build_shared_tables(pa, TEST_SEASONS)
    run_validation(shared, TEST_SEASONS, N_GAMES_TO_VALIDATE, N_TRIALS_PER_GAME, widen_w=WIDEN_W,
                   shock_sigma=SHOCK_SIGMA,
                   output_path=DATA_PROCESSED / "game_simulator_validation.parquet")
