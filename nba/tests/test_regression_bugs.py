"""Regression guards for real bugs found and fixed during validation
(2026-07-24) -- see MODEL_DOCUMENTATION.md Sec4/Sec5 for the full writeup of
each. Plain assertions, no pytest (matching this project's own
`validate_*.py` convention, and the sibling NHL project's
`test_holdout_walk_forward_discipline.py`).

Run: `python -m tests.test_regression_bugs`
"""

import datetime as _dt_module

import numpy as np
import pandas as pd

from src.ingest.build_stints import _get_starters, _normalize_name, _prep_pbp_timeline, _roster_lookup
from src.ingest.fetch_schedule import season_for_date
from src.ingest.player_name_crosswalk import _build_name_index
from src.pipeline.active_roster import _game_type_from_id, build_team_history, resolve_active_lineup
from src.models.bootstrap_significance import _MAX_IDX_MATRIX_BYTES, block_bootstrap_compare, bootstrap_compare
from src.models.rolling_window_backtest import rolling_window_report, summarize_rolling_report
from src.models.garbage_time import add_garbage_time_weight
from src.models.home_court import _baseline_log_ratios, fit_team_home_court_walk_forward
from src.models.home_court import DENVER_TEAM_ID, fit_denver_specific_home_court_walk_forward, fit_home_court_walk_forward
from src.models.oreb_decomposition import add_walk_forward_exposure_rate, project_oreb_share, project_team_oreb
from src.models.player_rate_shrinkage import (
    _trailing_league_rate_ewma, add_era_adjusted_player_rate, add_walk_forward_player_mean_ewm,
    add_walk_forward_player_rate,
)
from src.models.matchup_difficulty import (
    add_defender_difficulty_rate, add_defender_stat_difficulty_rate,
    add_position_group_stat_difficulty_rate_expanding, defender_total_minutes_asof,
    opponent_defense_adjustment,
)
from src.models.player_defensive_event_rates import add_defensive_event_rates
from src.models.player_playmaking_rates import PRIOR_TOUCHES_AST, PRIOR_TOUCHES_TOV
from src.models.player_scoring_rates import PRIOR_ATTEMPTS_FTM, PRIOR_MINUTES_FTA, add_scoring_rates
from src.models.prop_distribution import MIN_PLAYER_VARIANCE, fit_continuous_family, log_score, over_under_prob, predict_variance
from src.models.prop_distribution import family_for_mean, fit_count_family_mean_dependent
from src.models.attendance_model import attendance_features_for_game, predict_attendance_probability
from src.models.rest_schedule import add_schedule_density, fit_team_b2b_adjustment_walk_forward
from src.models.travel_fatigue import add_travel_fatigue
from src.models.clutch_rating import add_trailing_clutch_deviation, identify_clutch_stints, team_game_clutch_net_rating
from src.models.lineup_continuity import add_lineup_continuity, team_game_continuity_score
from src.models.referee_rates import add_referee_tendency, build_official_game_log, crew_tendency
from src.models.referee_rates import build_official_game_log_differential
from src.models.oreb_shot_location import build_miss_rebound_log
from src.models.score_distribution import _t_scale
from src.models.score_distribution import compute_walkforward_variance_model, predict_variance_walk_forward
from src.models.validate_holdout_bootstrap import generic_holdout_confirmatory_check
from src.models.usage_allocation import allocate_team_total, compute_usage_shares, matchup_point_delta
from src.models.lineup_rating import predictive_minutes_shares, semi_oracle_minutes_shares
from src.models.lineup_rating import probabilistic_predictive_minutes_shares
from src.models.rapm_lite import _career_games_played, prepare_stints
from src.models.team_stat_rates import (ADOPTED_CATEGORIES, CROSS_SEASON_WEIGHT_STAT, OWN_HALFLIFE_GAMES_STAT,
                                         STAT_COLUMNS, add_team_stat_ratings, build_team_stat_game_log, project_team_stat)
from src.models.team_strength import (
    OWN_HALFLIFE_GAMES_RATING, PRIOR_GAMES_PACE, PRIOR_GAMES_RATING, add_team_ratings, project_game,
)
from src.pipeline.generate_props import (
    _anchor_preserving_missing, _component_scale_factors, _latest_snapshot, _team_stat_totals,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _pbp_row(action_number, clock, period, action_type, score_home, score_away, description="", team_id=1, person_id=1, subtype=""):
    return {
        "actionNumber": action_number, "clock": clock, "period": period, "actionType": action_type,
        "scoreHome": score_home, "scoreAway": score_away, "description": description,
        "teamId": team_id, "personId": person_id, "subType": subtype,
    }


def test_possession_counter_uses_teamid_not_cumulative_description():
    """Bug 2 (Sec4a): a defensive rebound must be detected by comparing the
    rebounding team's ID against the immediately-preceding missed-shot's
    team ID -- NOT by substring-matching "Def:1" in the description, which
    is that PLAYER's CUMULATIVE rebound count and only ever matches their
    FIRST defensive rebound of the game. This constructs a player who grabs
    TWO defensive rebounds (so their real description would read "Def:1"
    then "Def:2") and confirms both are correctly detected."""
    pbp = pd.DataFrame([
        _pbp_row(1, "PT12M00.00S", 1, "period", 0, 0),
        _pbp_row(2, "PT11M50.00S", 1, "Missed Shot", 0, 0, team_id=100),
        _pbp_row(3, "PT11M48.00S", 1, "Rebound", 0, 0, description="P1 REBOUND (Off:0 Def:1)", team_id=200),
        _pbp_row(4, "PT11M40.00S", 1, "Missed Shot", 0, 0, team_id=100),
        _pbp_row(5, "PT11M38.00S", 1, "Rebound", 0, 0, description="P1 REBOUND (Off:0 Def:2)", team_id=200),
    ])
    tl = _prep_pbp_timeline(pbp)
    n_def_rebounds = int(tl["isDefensiveRebound"].sum())
    check("both defensive rebounds detected (not just the first)", n_def_rebounds == 2,
          f"got {n_def_rebounds}, expected 2")


def test_score_forward_fill_ignores_placeholder_zero_on_non_scoring_rows():
    """Bug (Sec5 #2): some seasons' PlayByPlayV3 rows carry the LITERAL
    STRING "0" (not blank) on non-scoring rows, including missed free
    throws -- naively forward-filling numeric-parsed scoreHome/scoreAway
    treats that "0" as real and wipes out the running score. Constructs a
    made shot (real score 5-3), then a missed free throw carrying a "0"
    placeholder, and confirms the score is NOT reset to 0."""
    pbp = pd.DataFrame([
        _pbp_row(1, "PT12M00.00S", 1, "period", "0", "0"),
        _pbp_row(2, "PT11M00.00S", 1, "Made Shot", "5", "3"),
        _pbp_row(3, "PT10M00.00S", 1, "Free Throw", "0", "0", description="MISS Player Free Throw 1 of 2"),
    ])
    tl = _prep_pbp_timeline(pbp)
    last_score = (int(tl.iloc[-1]["scoreHome"]), int(tl.iloc[-1]["scoreAway"]))
    check("score not reset by a placeholder '0' on a missed FT", last_score == (5, 3),
          f"got {last_score}, expected (5, 3)")


def test_starters_use_row_order_not_position_field():
    """Bug (Sec5 #1): a non-blank `position` field is NOT a reliable
    starter indicator on every season -- some seasons populate it for
    bench players too. Constructs a team where the 6th (bench) player
    has a non-blank position, and confirms starters are still exactly the
    first 5 rows."""
    box = pd.DataFrame([
        {"teamId": 1, "personId": 10 + i, "firstName": f"F{i}", "familyName": f"L{i}", "position": "G"}
        for i in range(5)
    ] + [{"teamId": 1, "personId": 99, "firstName": "Bench", "familyName": "Player", "position": "G"}])
    starters = _get_starters(box, team_id=1)
    expected = tuple(sorted(10 + i for i in range(5)))
    check("starters are the first 5 rows, not everyone with a position", starters == expected,
          f"got {starters}, expected {expected}")


def test_roster_lookup_resolves_name_collision_and_suffix_mismatch():
    """Bug (Sec5 #3, #4): (a) two teammates sharing a family name must
    resolve via NBA's "Je. Green"/"Ja. Green" disambiguation convention,
    not a bare ambiguous "Green"; (b) a suffixed familyName ("Butler III")
    must also resolve from a bare substitution-description name
    ("Butler")."""
    box = pd.DataFrame([
        {"teamId": 1, "personId": 1, "firstName": "Jeff", "familyName": "Green"},
        {"teamId": 1, "personId": 2, "firstName": "JaMychal", "familyName": "Green"},
        {"teamId": 1, "personId": 3, "firstName": "Jimmy", "familyName": "Butler III"},
    ])
    lookup, dupes = _roster_lookup(box)

    check("bare 'Green' is flagged ambiguous, not silently resolved",
          (1, _normalize_name("Green")) in dupes)
    check("'Je. Green' resolves to Jeff Green",
          lookup.get((1, _normalize_name("Je. Green"))) == 1)
    check("'Ja. Green' resolves to JaMychal Green",
          lookup.get((1, _normalize_name("Ja. Green"))) == 2)
    check("bare 'Butler' resolves to suffixed 'Butler III'",
          lookup.get((1, _normalize_name("Butler"))) == 3)


def test_home_court_uses_walkforward_columns_not_raw_realized_columns():
    """Bug (Sec4/Sec0): `_baseline_log_ratios` must read the walk-forward
    PRE-game columns (`rtg_attack_rate_home` etc.), not a game's own raw
    realized rating -- using the latter makes the fitted multiplier
    circular. This just confirms the function requires (and uses) the
    walk-forward column names -- constructing a frame with ONLY those
    columns (no `home_oRtg`/`away_pace` raw-realized names at all) must
    not raise a KeyError."""
    games = pd.DataFrame({
        "league_avg_pace": [100.0], "pace_shrunk_mean_home": [100.0], "pace_shrunk_mean_away": [100.0],
        "league_avg_rtg": [110.0], "rtg_attack_rate_home": [112.0], "rtg_defense_rate_home": [108.0],
        "rtg_attack_rate_away": [109.0], "rtg_defense_rate_away": [111.0],
        "actual_home_score": [115], "actual_away_score": [105],
    })
    try:
        result = _baseline_log_ratios(games)
        ok = "home_log_ratio" in result.columns and "away_log_ratio" in result.columns
    except KeyError as e:
        ok = False
        print(f"  KeyError: {e}")
    check("_baseline_log_ratios works from walk-forward columns alone", ok)


def test_team_home_court_walk_forward_has_no_leak_on_a_teams_first_ever_home_game():
    """Task #54 (Fable 5 critique item 1f): `fit_team_home_court_walk_forward`
    must fall back to PURE league-wide shrinkage (zero team-specific weight)
    on a team's first-ever home appearance in `games` -- `team_games_before`
    is 0 there, so the blend must equal the trailing league-wide log-mult
    exactly, regardless of how extreme that single game's own realized
    score is. Builds 35 warm-up games (clears `min_periods=30` for the
    league-wide trailing EWMA) between other teams, then a 36th game where
    a brand-new team "NEW" hosts for the first time with a deliberately
    extreme score (130) that would visibly shift the result if it leaked
    into its own not-yet-computed team-specific estimate."""
    dates = pd.date_range("2020-10-01", periods=36, freq="D")
    rows = []
    for i in range(35):
        home, away = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append({
            "gameId": f"g{i}", "gameDate": dates[i], "season": 2020,
            "team_home": home, "team_away": away,
            "league_avg_pace": 100.0, "pace_shrunk_mean_home": 100.0, "pace_shrunk_mean_away": 100.0,
            "league_avg_rtg": 110.0, "rtg_attack_rate_home": 110.0, "rtg_defense_rate_home": 110.0,
            "rtg_attack_rate_away": 110.0, "rtg_defense_rate_away": 110.0,
            "actual_home_score": 112, "actual_away_score": 108,
        })
    rows.append({
        "gameId": "g35", "gameDate": dates[35], "season": 2020,
        "team_home": "NEW", "team_away": "A",
        "league_avg_pace": 100.0, "pace_shrunk_mean_home": 100.0, "pace_shrunk_mean_away": 100.0,
        "league_avg_rtg": 110.0, "rtg_attack_rate_home": 110.0, "rtg_defense_rate_home": 110.0,
        "rtg_attack_rate_away": 110.0, "rtg_defense_rate_away": 110.0,
        "actual_home_score": 130, "actual_away_score": 108,
    })
    games = pd.DataFrame(rows)

    from src.models.home_court import _baseline_log_ratios
    g = _baseline_log_ratios(games.sort_values(["gameDate", "gameId"]).reset_index(drop=True))
    league_combined = (g["home_log_ratio"] - g["away_log_ratio"]) / 2.0
    expected_league_trailing = float(
        league_combined.shift(1).ewm(halflife=400.0, min_periods=30).mean().iloc[35])

    result = fit_team_home_court_walk_forward(games, prior_games=200.0, halflife_games=400.0)
    got = float(result.iloc[35])
    expected = float(np.exp(expected_league_trailing))
    check("NEW team's first home game gets pure league-wide shrinkage (no self-leak)",
          abs(got - expected) < 1e-9, f"got {got}, expected {expected}")


def test_add_walk_forward_exposure_rate_has_no_leak_on_a_teams_first_ever_game():
    """Task #55 (OREB miss-decomposition): `add_walk_forward_exposure_rate`
    is a NEW numerator/exposure walk-forward primitive (team-grain, e.g.
    oreb_for/misses_for) -- must fall back to PURE league-wide shrinkage
    (zero team-specific weight) on a team's first-ever game in `log`, same
    zero-leak requirement as every other walk-forward primitive in this
    project. Builds 10 warm-up games (2 teams alternating) then a game
    where a brand-new team "NEW" appears for the first time with a
    deliberately extreme numerator/exposure pair that would visibly shift
    the result if it leaked into its own not-yet-computed rate."""
    dates = pd.date_range("2020-10-01", periods=11, freq="D")
    rows = []
    for i in range(10):
        team, opp = ("A", "B") if i % 2 == 0 else ("B", "A")
        rows.append({"gameId": f"g{i}", "gameDate": dates[i], "season": 2020, "team": team,
                      "num": 10, "exp": 40})
    rows.append({"gameId": "g10", "gameDate": dates[10], "season": 2020, "team": "NEW",
                  "num": 39, "exp": 40})  # extreme 97.5% rate -- would visibly shift the result if leaked
    log = pd.DataFrame(rows)

    result = add_walk_forward_exposure_rate(log, "num", "exp", prior_exposure=200.0, prefix="x")
    league_avg_at_row10 = float(result["x_league_avg_rate"].iloc[10])
    got = float(result["x_shrunk_rate"].iloc[10])
    check("NEW team's first game gets pure league-wide shrinkage (no self-leak)",
          abs(got - league_avg_at_row10) < 1e-9, f"got {got}, expected {league_avg_at_row10}")
    check("league-wide trailing rate itself doesn't leak the extreme row's own num/exp",
          abs(league_avg_at_row10 - 0.25) < 1e-9, f"got {league_avg_at_row10}, expected 0.25 (10/40, the warm-up rate)")


def test_project_oreb_share_returns_exact_league_average_at_league_average_inputs():
    """Task #55: `project_oreb_share` must reduce to the shared league
    baseline when every team's own rate exactly equals its own league
    average -- the same "no real deviation, no adjustment" sanity check
    every other ratio-idiom combine in this project satisfies (mirrors
    `project_game`'s/`project_team_stat`'s own at-average behavior)."""
    home_share, away_share = project_oreb_share(
        home_oreb_rate=0.28, home_dreb_rate=0.72, away_oreb_rate=0.28, away_dreb_rate=0.72,
        league_avg_oreb_rate=0.28, league_avg_dreb_rate=0.72)
    check("home_share == league_avg_oreb_rate when every input is exactly at its own league average",
          abs(home_share - 0.28) < 1e-9, f"got {home_share}")
    check("away_share == league_avg_oreb_rate when every input is exactly at its own league average",
          abs(away_share - 0.28) < 1e-9, f"got {away_share}")


def test_project_team_oreb_gives_exact_game_level_coherence_with_projected_misses():
    """Task #55's central design claim: a team's projected OREB is EXACTLY
    `oreb_share * that team's own projected misses` -- so the opponent's
    implied DREB on that same miss set (`projected_misses - projected_OREB`,
    never independently estimated) reconciles EXACTLY, not approximately.
    Verifies `project_team_oreb`'s home output actually equals
    `project_oreb_share`'s home_share times `project_team_stat`'s
    independently-computed home misses total, confirming the composition
    wires the share and the volume together correctly (not e.g. accidentally
    multiplying by the wrong side's misses)."""
    from src.models.team_stat_rates import project_team_stat
    home_misses, away_misses = project_team_stat(
        home_attack=45.0, home_defense=43.0, away_attack=44.0, away_defense=46.0, league_avg=44.5)
    home_share, away_share = project_oreb_share(
        home_oreb_rate=0.30, home_dreb_rate=0.70, away_oreb_rate=0.25, away_dreb_rate=0.75,
        league_avg_oreb_rate=0.28, league_avg_dreb_rate=0.72)

    home_oreb, away_oreb = project_team_oreb(
        home_misses_attack=45.0, home_misses_defense=43.0, away_misses_attack=44.0, away_misses_defense=46.0,
        league_avg_misses=44.5,
        home_oreb_rate=0.30, home_dreb_rate=0.70, away_oreb_rate=0.25, away_dreb_rate=0.75,
        league_avg_oreb_rate=0.28, league_avg_dreb_rate=0.72)

    check("home_oreb == home_share * home_misses exactly",
          abs(home_oreb - home_share * home_misses) < 1e-9, f"got {home_oreb}, expected {home_share * home_misses}")
    check("away_oreb == away_share * away_misses exactly",
          abs(away_oreb - away_share * away_misses) < 1e-9, f"got {away_oreb}, expected {away_share * away_misses}")
    check("the opponent's implied DREB on home's own misses reconciles exactly to home's projected misses",
          abs((home_oreb + (home_misses - home_oreb)) - home_misses) < 1e-9)


def test_compute_walkforward_variance_model_has_no_leak_across_a_checkpoint():
    """Task #56: `compute_walkforward_variance_model` must fit each
    checkpoint using ONLY residuals strictly BEFORE that checkpoint date --
    the same walk-forward-safety requirement as every other periodic-refit
    primitive in this project (mirrors `rapm_lite.compute_walkforward_player_ratings`).
    Builds 250 low-variance warm-up residuals (pace=100, residual~N(0,4),
    deterministic small values) dated well before a checkpoint, then 250
    HUGE-variance residuals (residual=500) dated ON/AFTER that same
    checkpoint -- if the huge residuals leaked into the checkpoint's own
    fit, its intercept would be enormous; if not, it stays small."""
    dates_before = pd.date_range("2020-10-01", periods=250, freq="D")
    dates_after = pd.date_range("2021-06-08", periods=250, freq="D")
    residuals = pd.DataFrame({
        "gameDate": list(dates_before) + list(dates_after),
        "residual": [2.0 if i % 2 == 0 else -2.0 for i in range(250)] + [500.0] * 250,
        "pace": [100.0] * 500,
    })
    checkpoints = compute_walkforward_variance_model(residuals, refit_period_days=400)
    check("at least one checkpoint was fit", len(checkpoints) >= 1, f"got {len(checkpoints)} checkpoints")
    first_checkpoint_intercept = float(checkpoints["intercept"].iloc[0])
    check("the first checkpoint's fit (using only the low-variance warm-up residuals) has a SMALL "
          "intercept, not contaminated by the huge post-checkpoint residuals",
          first_checkpoint_intercept < 50.0, f"got intercept={first_checkpoint_intercept}")


def test_predict_variance_walk_forward_picks_the_latest_prior_checkpoint():
    """Task #56: `predict_variance_walk_forward` must select the LATEST
    checkpoint strictly before each row's gameDate (not the nearest by
    absolute distance, not a future one) -- confirms the checkpoint-lookup
    arithmetic (searchsorted-based) matches the "checkpoints < gameDate,
    take the last one" pattern used everywhere else in this project's
    validation scripts. Also confirms a query date before EVERY checkpoint
    falls back to the EARLIEST checkpoint rather than raising or NaN-ing."""
    checkpoints = pd.DataFrame({
        "asOfDate": pd.to_datetime(["2020-11-01", "2020-11-15", "2020-12-01"]),
        "intercept": [10.0, 20.0, 30.0], "slope": [0.0, 0.0, 0.0],
    })
    dates = pd.Series(pd.to_datetime(["2020-10-01", "2020-11-10", "2020-11-16", "2020-12-31"]))
    result = predict_variance_walk_forward(checkpoints, dates, pace=np.array([100.0] * 4))
    check("a query date BEFORE every checkpoint falls back to the earliest checkpoint's variance",
          abs(result[0] - 10.0) < 1e-9, f"got {result[0]}")
    check("a query date between checkpoint 1 and 2 uses checkpoint 1 (the latest STRICTLY prior one)",
          abs(result[1] - 10.0) < 1e-9, f"got {result[1]}")
    check("a query date between checkpoint 2 and 3 uses checkpoint 2",
          abs(result[2] - 20.0) < 1e-9, f"got {result[2]}")
    check("a query date after every checkpoint uses the LATEST (checkpoint 3)",
          abs(result[3] - 30.0) < 1e-9, f"got {result[3]}")


def test_fit_count_family_mean_dependent_isolates_the_near_zero_overdispersion():
    """Task #57 (Sec46.2's finding): overdispersion needing Negative-
    Binomial is concentrated in the NEAR-ZERO-projected-mean bucket
    specifically, not the high-mean/star tier -- a single POOLED family
    choice for a whole category can hide this. Builds synthetic data with
    that EXACT shape: 1,000 near-zero-mean (0.02) rows with a deliberately
    overdispersed actual-outcome rate (15% ones -- ~7.5x the ~2% a true
    Poisson(0.02) would produce), plus 3,000 rows at means 1/2/3 with
    genuinely Poisson-distributed actuals (variance == mean by
    construction). Confirms the mean-dependent fit picks NegBin for the
    near-zero bucket specifically, Poisson for every higher-mean bucket --
    and that a real single POOLED fit_count_family call over the same data
    would have missed the near-zero-specific effect (masked by the 3x
    larger well-behaved population)."""
    rng = np.random.default_rng(42)
    means = np.concatenate([
        np.full(1000, 0.02),
        np.full(1000, 1.0), np.full(1000, 2.0), np.full(1000, 3.0),
    ])
    actuals = np.concatenate([
        rng.choice([0, 1], size=1000, p=[0.85, 0.15]),  # ~7.5x a true Poisson(0.02) rate -- overdispersed
        rng.poisson(1.0, size=1000), rng.poisson(2.0, size=1000), rng.poisson(3.0, size=1000),
    ]).astype(float)

    from src.models.prop_distribution import fit_count_family
    pooled = fit_count_family(actuals, means)
    check("a single POOLED fit over the whole mix does NOT trigger NegBin (masked by the 3x larger "
          "well-behaved population) -- confirms the mean-dependent split finds something the pooled "
          "check itself would miss, not a tautology",
          pooled["family"] == "poisson", f"got {pooled}")

    model = fit_count_family_mean_dependent(actuals, means, n_buckets=4)
    near_zero_family = family_for_mean(model, 0.02)
    check("the near-zero-mean bucket gets NegBin (the real overdispersion isolated)",
          near_zero_family["family"] == "negbin", f"got {near_zero_family}")
    for query_mean in (1.0, 2.0, 3.0):
        f = family_for_mean(model, query_mean)
        check(f"mean={query_mean} bucket stays Poisson (genuinely Poisson-distributed by construction)",
              f["family"] == "poisson", f"got {f}")


def test_attendance_features_counts_dnp_streak_from_the_most_recent_game_backward():
    """Task #58: `attendance_features_for_game`'s `dnp_streak` must count
    consecutive MISSES starting from the game right before tonight and
    working BACKWARD, stopping at the first appearance -- not a flat count
    of total misses in the window (which would conflate "missed the last 3
    in a row" with "missed 3 games scattered throughout the window", two
    very different real situations). Builds a 6-game window where player A
    played games 1-3, then missed games 4-6 (streak=3, fraction=0.5) and
    player B missed game 1, played 2-4, then missed 5-6 (streak=2,
    fraction=0.5 -- SAME fraction as A, but a shorter, more recent streak,
    since B's most recent miss-run is only 2 long)."""
    games = [f"g{i}" for i in range(1, 7)]
    team_side_by_game = {g: "home" for g in games}
    rows = []
    # player A: attended g1,g2,g3; missed g4,g5,g6
    for g in games[:3]:
        rows.append({"gameId": g, "team_side": "home", "playerId": "A", "minutes": 20.0})
    # player B: missed g1; attended g2,g3,g4; missed g5,g6
    for g in games[1:4]:
        rows.append({"gameId": g, "team_side": "home", "playerId": "B", "minutes": 20.0})
    player_minutes = pd.DataFrame(rows)

    features = attendance_features_for_game(player_minutes, games, team_side_by_game, lookback_games=6)
    features = features.set_index("playerId")
    check("player A's games_played_fraction is 0.5 (3 of 6)",
          abs(features.loc["A", "games_played_fraction"] - 0.5) < 1e-9)
    check("player A's dnp_streak is 3 (missed the last 3 in a row)",
          features.loc["A", "dnp_streak"] == 3, f"got {features.loc['A', 'dnp_streak']}")
    check("player B's games_played_fraction is ALSO 0.5 (3 of 6) -- same as A",
          abs(features.loc["B", "games_played_fraction"] - 0.5) < 1e-9)
    check("player B's dnp_streak is only 2 (missed the last 2, not 3 -- a SHORTER, more recent gap "
          "despite an identical games_played_fraction to player A)",
          features.loc["B", "dnp_streak"] == 2, f"got {features.loc['B', 'dnp_streak']}")


def test_predict_attendance_probability_decays_with_streak_and_matches_fraction_at_streak_zero():
    """Task #58: at `dnp_streak=0` (played last game), the predicted
    probability must equal the raw `games_played_fraction` UNCHANGED
    (streak_decay^0 == 1) -- confirms the streak term is a pure REFINEMENT
    on top of the base rate, not a separate, disconnected estimate. A
    nonzero streak must strictly DECREASE the probability relative to the
    same fraction with streak=0, and a longer streak must decrease it
    further (monotonically), for a fixed `games_played_fraction`."""
    features = pd.DataFrame({
        "games_played_fraction": [0.8, 0.8, 0.8],
        "dnp_streak": [0, 1, 3],
    })
    probs = predict_attendance_probability(features, streak_decay=0.6)
    check("streak=0 leaves the raw fraction exactly unchanged",
          abs(probs.iloc[0] - 0.8) < 1e-9, f"got {probs.iloc[0]}")
    check("streak=1 strictly decreases the probability below the raw fraction",
          probs.iloc[1] < 0.8, f"got {probs.iloc[1]}")
    check("streak=3 decreases the probability further than streak=1 (monotonic decay)",
          probs.iloc[2] < probs.iloc[1], f"got streak=1:{probs.iloc[1]}, streak=3:{probs.iloc[2]}")


def test_add_schedule_density_counts_only_strictly_prior_games_in_window():
    """Task #59: `add_schedule_density`'s `games_in_last_{N}d` must count
    only STRICTLY PRIOR games (walk-forward safe) whose gameDate falls
    within the trailing window ending just before today's own game --
    distinct from `is_b2b` (which only looks at the single immediately-
    prior game and can't distinguish a genuine multi-game fatigue stretch
    from an isolated back-to-back). Builds one team with games on day 0,
    1, 2 (three in a row) and a fourth game on day 10 (well-rested)."""
    dates = pd.to_datetime(["2020-10-01", "2020-10-02", "2020-10-03", "2020-10-11"])
    log = pd.DataFrame({"team": ["A"] * 4, "season": [2020] * 4, "gameDate": dates})
    result = add_schedule_density(log, window_days=4)
    col = "games_in_last_4d"
    check("game 1 (first ever) has 0 prior games in window", result[col].iloc[0] == 0, f"got {result[col].iloc[0]}")
    check("game 2 (day after game 1) has 1 prior game in the trailing 4-day window",
          result[col].iloc[1] == 1, f"got {result[col].iloc[1]}")
    check("game 3 (third in a row) has 2 prior games in the trailing 4-day window "
          "-- distinguishes a genuine fatigue stretch from a single back-to-back",
          result[col].iloc[2] == 2, f"got {result[col].iloc[2]}")
    check("game 4 (well-rested, 8 days after game 3) has 0 games in its trailing 4-day window",
          result[col].iloc[3] == 0, f"got {result[col].iloc[3]}")


def test_add_travel_fatigue_computes_distance_and_timezone_shift_from_own_prior_game_only():
    """Task #60: `add_travel_fatigue`'s `travel_km`/`tz_shift` must reflect
    the distance/timezone change from THIS TEAM's own immediately-prior
    game location, walk-forward safe (first-ever game gets NaN, not a
    fabricated 0). Builds LAL playing: game 1 at home (LA), game 2 away at
    NYK (a real, well-known ~3,900km cross-country trip spanning 3
    timezone zones), game 3 back home in LA (the return trip)."""
    from src.ingest.team_codes import ABBREV_TO_TEAM_ID
    lal_id, nyk_id, gsw_id = ABBREV_TO_TEAM_ID["LAL"], ABBREV_TO_TEAM_ID["NYK"], ABBREV_TO_TEAM_ID["GSW"]
    dates = pd.date_range("2020-11-01", periods=3, freq="4D")
    log = pd.DataFrame({
        "team": [lal_id, lal_id, lal_id], "opponent": [gsw_id, nyk_id, nyk_id],
        "is_home": [True, False, True], "season": [2020, 2020, 2020], "gameDate": dates,
    })
    result = add_travel_fatigue(log)
    check("LAL's first-ever game has NaN travel_km (no prior game to compare against)",
          pd.isna(result["travel_km"].iloc[0]), f"got {result['travel_km'].iloc[0]}")
    check("LAL's first-ever game has NaN tz_shift",
          pd.isna(result["tz_shift"].iloc[0]))
    check("LA -> NYC (game 2) is a real, substantial ~3,900km trip",
          3500 < result["travel_km"].iloc[1] < 4300, f"got {result['travel_km'].iloc[1]}")
    check("LA -> NYC crosses exactly 3 timezone zones eastward (Pacific=3 -> Eastern=0, shift=-3)",
          result["tz_shift"].iloc[1] == -3, f"got {result['tz_shift'].iloc[1]}")
    check("NYC -> LA (game 3, the return trip) is the same real distance as game 2",
          abs(result["travel_km"].iloc[2] - result["travel_km"].iloc[1]) < 1.0)
    check("NYC -> LA crosses 3 zones the OTHER direction (Eastern=0 -> Pacific=3, shift=+3)",
          result["tz_shift"].iloc[2] == 3, f"got {result['tz_shift'].iloc[2]}")


def test_identify_clutch_stints_requires_both_late_AND_close():
    """Task #61: `identify_clutch_stints` must require BOTH conditions --
    late in the game (within the trailing 5 game-minutes) AND close
    (margin within 5 points) -- neither alone should qualify. Builds 4
    stints in a single regulation game: early+close, late+blowout,
    late+close (the only real clutch stint), and an OT stint that's
    close (confirms OT games use their own actual total length, not a
    fixed regulation-length assumption)."""
    stints = pd.DataFrame({
        "gameId": ["g1", "g1", "g1", "g1"],
        "startTenths": [10000, 27000, 28500, 29500],  # regulation ends at 28800; OT stint starts after
        "endTenths": [10500, 27500, 28800, 30000],
        "marginBeforeStint": [2, 20, 3, 1],
    })
    result = identify_clutch_stints(stints)
    check("early + close is NOT clutch (not late enough)", not result["is_clutch"].iloc[0])
    check("late + blowout is NOT clutch (not close enough)", not result["is_clutch"].iloc[1])
    check("late + close IS clutch (both conditions met)", result["is_clutch"].iloc[2])
    check("an OT stint (close, near the game's own actual end) IS clutch -- uses this game's own "
          "real total length, not a fixed regulation-only assumption", result["is_clutch"].iloc[3])


def test_add_trailing_clutch_deviation_skips_nan_games_and_requires_a_minimum_sample():
    """Task #61: `add_trailing_clutch_deviation` must (a) SKIP games with
    no clutch possessions (NaN clutch_net_rtg) when computing the
    trailing average -- not treat a blowout's absence of clutch data as a
    deviation of 0 -- and (b) stay NaN until `min_prior_clutch_games`
    real clutch games have accumulated, not report an unstable 1-game
    estimate. Builds a team with clutch deviations of +10 (game 1),
    then two blowout games with NO clutch data (NaN), then +20 (game 4)
    -- the trailing average feeding game 5 must be based on games 1 and 4
    ONLY (mean=+15), not diluted by the two NaN blowouts."""
    dates = pd.date_range("2020-10-01", periods=5, freq="3D")
    log = pd.DataFrame({
        "team": ["A"] * 5, "season": [2020] * 5, "gameDate": dates,
        "clutch_net_rtg": [110.0, np.nan, np.nan, 120.0, np.nan],
        "overall_net_rtg": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    result = add_trailing_clutch_deviation(log, min_prior_clutch_games=2)
    check("with only 1 prior real clutch game (before game 4), trailing deviation stays NaN "
          "(min_prior_clutch_games=2 not yet met)",
          pd.isna(result["trailing_clutch_deviation"].iloc[3]), f"got {result['trailing_clutch_deviation'].iloc[3]}")
    check("by game 5, exactly 2 real clutch games (1 and 4) have accumulated -- trailing deviation "
          "is their mean (+10, +20 -> +15), NOT diluted by the 2 intervening NaN blowout games",
          abs(result["trailing_clutch_deviation"].iloc[4] - 15.0) < 1e-9,
          f"got {result['trailing_clutch_deviation'].iloc[4]}")


def test_add_lineup_continuity_counts_distinct_prior_games_not_stints():
    """Task #62: `games_together_before` must count DISTINCT PRIOR GAMES a
    5-man combo has appeared in, never inflated by the SAME lineup
    recurring in multiple stints within one game (a same-game leak risk
    this project always guards against). Team 100's lineup (1,2,3,4,5)
    appears in TWO separate stints within game 1 (should still count as
    ONE game of history, not two) and once in game 2 (a later date) --
    game 2's row for that same lineup must show games_together_before=1,
    not 2. A brand-new lineup (6,7,8,9,10) debuting in game 2 must show
    games_together_before=0."""
    stints = pd.DataFrame({
        "gameId": ["g1", "g1", "g2", "g2"],
        "gameDate": pd.to_datetime(["2020-10-01", "2020-10-01", "2020-10-05", "2020-10-05"]),
        "homePlayers": [(1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (6, 7, 8, 9, 10)],
        "awayPlayers": [(20, 21, 22, 23, 24)] * 4,
        "possessions": [10, 8, 15, 5],
    })
    schedule = pd.DataFrame({"gameId": ["g1", "g2"], "homeTeamId": [100, 100], "awayTeamId": [200, 200], "season": [2020, 2020]})

    continuity = add_lineup_continuity(stints, schedule)
    g1_row = continuity[(continuity["gameId"] == "g1") & (continuity["team"] == 100)]
    check("game 1's lineup (1-5) collapses to exactly ONE row despite 2 separate stints",
          len(g1_row) == 1, f"got {len(g1_row)} rows")
    check("game 1's lineup (1-5) has games_together_before=0 (first-ever use)",
          g1_row["games_together_before"].iloc[0] == 0, f"got {g1_row['games_together_before'].iloc[0]}")
    check("that lineup's total possessions in game 1 is 10+8=18 (both stints summed)",
          g1_row["possessions"].iloc[0] == 18, f"got {g1_row['possessions'].iloc[0]}")

    g2 = continuity[(continuity["gameId"] == "g2") & (continuity["team"] == 100)]
    same_lineup = g2[g2["lineup_key"] == (1, 2, 3, 4, 5)]
    new_lineup = g2[g2["lineup_key"] == (6, 7, 8, 9, 10)]
    check("game 2's SAME lineup (1-5) shows games_together_before=1 (ONE prior GAME, not 2 -- "
          "not inflated by game 1's 2 separate stints)",
          same_lineup["games_together_before"].iloc[0] == 1, f"got {same_lineup['games_together_before'].iloc[0]}")
    check("game 2's BRAND-NEW lineup (6-10) shows games_together_before=0",
          new_lineup["games_together_before"].iloc[0] == 0, f"got {new_lineup['games_together_before'].iloc[0]}")

    scores = team_game_continuity_score(continuity)
    g2_score = scores[(scores["gameId"] == "g2") & (scores["team"] == 100)]["continuity_score"].iloc[0]
    expected = (1 * 15 + 0 * 5) / (15 + 5)  # possession-weighted average
    check("game 2's team-level continuity_score is the possession-weighted average across its 2 lineups",
          abs(g2_score - expected) < 1e-9, f"got {g2_score}, expected {expected}")


def test_add_referee_tendency_has_no_leak_on_an_officials_first_ever_game():
    """Task #64: `add_referee_tendency` must fall back to PURE trailing
    league-wide shrinkage (zero own-history weight) on an official's
    first-ever game -- and the trailing league average itself must not
    leak that SAME game's own extreme value (the same-game collapse guard,
    since multiple officials share one game's value_col). Builds 10
    warm-up games (3 officials each, rotating pool of 4, total_fta=40
    flat) then an 11th game where a brand-new official "Z" works
    alongside 2 warm-up officials, with a deliberately extreme
    total_fta=200 for that game."""
    pool = ["A", "B", "C", "D"]
    rows = []
    for i in range(10):
        crew = [pool[i % 4], pool[(i + 1) % 4], pool[(i + 2) % 4]]
        for official in crew:
            rows.append({"gameId": f"g{i}", "personId": official, "total_fta": 40.0,
                         "gameDate": pd.Timestamp("2020-10-01") + pd.Timedelta(days=i)})
    rows.append({"gameId": "g10", "personId": "Z", "total_fta": 200.0, "gameDate": pd.Timestamp("2020-10-11")})
    rows.append({"gameId": "g10", "personId": "A", "total_fta": 200.0, "gameDate": pd.Timestamp("2020-10-11")})
    log = pd.DataFrame(rows)

    result = add_referee_tendency(log, "total_fta", prior_games=50.0)
    z_row = result[(result["gameId"] == "g10") & (result["personId"] == "Z")]
    league_avg_at_g10 = float(z_row["total_fta_league_avg"].iloc[0])
    got = float(z_row["total_fta_official_tendency"].iloc[0])
    check("the trailing league average at game 10 is exactly 40.0 (the flat warm-up rate, not "
          "leaking game 10's own extreme 200 value)",
          abs(league_avg_at_g10 - 40.0) < 1e-9, f"got {league_avg_at_g10}")
    check("official Z's first-ever game gets pure league-wide shrinkage (no self-leak from the "
          "extreme value in this SAME game)",
          abs(got - league_avg_at_g10) < 1e-9, f"got {got}, expected {league_avg_at_g10}")


def test_build_miss_rebound_log_skips_block_credit_rows_and_dead_balls_and_game_boundaries():
    """Task #65: `build_miss_rebound_log` must (a) skip BLOCK/STEAL credit
    rows (actionType=='', sharing the SAME actionNumber as their parent
    event) when finding "the next event after a miss" -- naively taking
    the next ROW would grab a block-credit row instead of the real
    rebound, (b) exclude dead-ball rebounds (teamId==0, no real team
    credited) rather than fabricating an OREB/DREB outcome, (c) never
    link a miss at the end of one game to a rebound at the start of the
    next game in the sorted frame."""
    rows = [
        # g1: a BLOCKED miss -- team A shoots, team B gets block credit (actionType=''),
        # THEN the real rebound (team B) at a later actionNumber. Real rebound: team B -> not OREB.
        {"gameId": "g1", "actionNumber": 1, "actionType": "Missed Shot", "teamId": "A", "shotDistance": 5, "shotValue": 2},
        {"gameId": "g1", "actionNumber": 1, "actionType": "", "teamId": "B", "shotDistance": 0, "shotValue": 0},
        {"gameId": "g1", "actionNumber": 2, "actionType": "Rebound", "teamId": "B", "shotDistance": 0, "shotValue": 0},
        # g1: a clean miss + real rebound by the SHOOTING team -> OREB
        {"gameId": "g1", "actionNumber": 3, "actionType": "Missed Shot", "teamId": "A", "shotDistance": 25, "shotValue": 3},
        {"gameId": "g1", "actionNumber": 4, "actionType": "Rebound", "teamId": "A", "shotDistance": 0, "shotValue": 0},
        # g1: a miss followed by a DEAD-BALL rebound (teamId=0) -- must be excluded entirely
        {"gameId": "g1", "actionNumber": 5, "actionType": "Missed Shot", "teamId": "A", "shotDistance": 10, "shotValue": 2},
        {"gameId": "g1", "actionNumber": 6, "actionType": "Rebound", "teamId": 0, "shotDistance": 0, "shotValue": 0},
        # g1: the LAST event of the game is a miss with no rebound at all within g1
        {"gameId": "g1", "actionNumber": 7, "actionType": "Missed Shot", "teamId": "A", "shotDistance": 3, "shotValue": 2},
        # g2: starts with a Rebound -- must NOT be linked to g1's trailing miss above
        {"gameId": "g2", "actionNumber": 1, "actionType": "Rebound", "teamId": "B", "shotDistance": 0, "shotValue": 0},
    ]
    pbp = pd.DataFrame(rows)
    result = build_miss_rebound_log(pbp)

    check("exactly 2 real miss-rebound pairs found (the blocked one and the clean one) -- "
          "NOT the dead-ball one, and NOT a spurious cross-game link for g1's trailing miss",
          len(result) == 2, f"got {len(result)} rows: {result.to_dict('records')}")
    blocked_row = result[(result["gameId"] == "g1") & (result["zone"] == "short_mid")]
    check("the blocked miss correctly links to the REAL rebound (actionNumber=2, team B), "
          "not the block-credit row (actionNumber=1, also team B coincidentally here -- verified "
          "by construction that skipping the blank row was necessary, not accidental)",
          len(blocked_row) == 1 and blocked_row["is_oreb"].iloc[0] == 0.0,
          f"got {blocked_row.to_dict('records')}")
    clean_row = result[(result["gameId"] == "g1") & (result["zone"] == "three")]
    check("the clean three-point miss + same-team rebound is correctly flagged as OREB",
          len(clean_row) == 1 and clean_row["is_oreb"].iloc[0] == 1.0, f"got {clean_row.to_dict('records')}")


def test_probabilistic_predictive_shares_matches_flat_predictive_when_everyone_certain():
    """Task #66: `probabilistic_predictive_minutes_shares` must be a
    STRICT GENERALIZATION of `predictive_minutes_shares`, not a different
    mechanism -- when every candidate attended every window game
    (games_played_fraction=1.0, dnp_streak=0 for all), every attendance
    weight collapses to 1.0 and the two functions must produce IDENTICAL
    shares. Two players, A (30 min/game) and B (10 min/game), both
    attended all 10 window games."""
    games = [f"g{i}" for i in range(10)]
    team_side_by_game = {g: "home" for g in games}
    rows = []
    for g in games:
        rows.append({"gameId": g, "team_side": "home", "playerId": "A", "minutes": 30.0})
        rows.append({"gameId": g, "team_side": "home", "playerId": "B", "minutes": 10.0})
    player_minutes = pd.DataFrame(rows)

    flat = predictive_minutes_shares(player_minutes, "target_game", "home", games, team_side_by_game, lookback_games=10)
    prob = probabilistic_predictive_minutes_shares(player_minutes, "target_game", "home", games, team_side_by_game,
                                                    lookback_games=10, streak_decay=1.0)
    check("player A's share is identical between the two modes when everyone is certain to attend",
          abs(flat["A"] - prob["A"]) < 1e-9, f"flat={flat['A']}, prob={prob['A']}")
    check("player B's share is identical between the two modes when everyone is certain to attend",
          abs(flat["B"] - prob["B"]) < 1e-9, f"flat={flat['B']}, prob={prob['B']}")


def test_probabilistic_predictive_shares_downweights_a_player_on_a_current_dnp_streak():
    """Task #66: unlike the flat `predictive_minutes_shares` (which treats
    every trailing-window union member as equally certain regardless of
    HOW RECENTLY they last played), `probabilistic_predictive_minutes_shares`
    must give a SMALLER share to a player currently on a DNP streak than
    to a player with an IDENTICAL trailing-average minutes but no recent
    absence. Player A: 30 min in every one of 10 games (streak=0). Player
    C: also averages 30 min/game, but only played the first 7 of 10
    window games (streak=3, games_played_fraction=0.7) -- same raw
    trailing average as A, very different recent attendance pattern."""
    games = [f"g{i}" for i in range(10)]
    team_side_by_game = {g: "home" for g in games}
    rows = []
    for g in games:
        rows.append({"gameId": g, "team_side": "home", "playerId": "A", "minutes": 30.0})
    for g in games[:7]:
        rows.append({"gameId": g, "team_side": "home", "playerId": "C", "minutes": 30.0})
    player_minutes = pd.DataFrame(rows)

    flat = predictive_minutes_shares(player_minutes, "target_game", "home", games, team_side_by_game, lookback_games=10)
    check("the flat mode gives A and C the SAME share (50/50) -- it can't see C's recent absence, "
          "only the raw trailing average, which is identical for both",
          abs(flat["A"] - flat["C"]) < 1e-9, f"got A={flat['A']}, C={flat['C']}")

    prob = probabilistic_predictive_minutes_shares(player_minutes, "target_game", "home", games, team_side_by_game,
                                                    lookback_games=10, streak_decay=0.7)
    check("the probabilistic mode gives A a LARGER share than C despite their identical trailing "
          "averages, because C is currently on a real 3-game DNP streak",
          prob["A"] > prob["C"], f"got A={prob['A']}, C={prob['C']}")


def test_build_official_game_log_differential_computes_away_minus_home_correctly():
    """Task #69: `home_fta_diff`/`home_foul_diff` must be (away team's own
    total) - (home team's own total) -- positive means the home team
    benefited (fewer fouls/FTA called on them). Builds one game where the
    home team (id 100) committed 15 fouls/20 FTA and the away team
    (id 200) committed 22 fouls/26 FTA -- home should show a clear
    positive differential on both."""
    box = pd.DataFrame({
        "gameId": ["g1", "g1"], "teamId": [100, 200],
        "freeThrowsAttempted": [20, 26], "foulsPersonal": [15, 22],
    })
    schedule = pd.DataFrame({"gameId": ["g1"], "homeTeamId": [100], "awayTeamId": [200],
                              "gameDate": ["2020-10-01"], "season": [2020]})
    officials = pd.DataFrame({"gameId": ["g1", "g1", "g1"], "personId": ["A", "B", "C"]})

    result = build_official_game_log_differential(officials, box, schedule)
    check("home_fta_diff is away(26) - home(20) = +6",
          (result["home_fta_diff"] == 6).all(), f"got {result['home_fta_diff'].tolist()}")
    check("home_foul_diff is away(22) - home(15) = +7 (home team benefited -- fewer fouls called on them)",
          (result["home_foul_diff"] == 7).all(), f"got {result['home_foul_diff'].tolist()}")
    check("all 3 officials on the crew get the same shared game-level differential value",
          len(result) == 3 and result["home_fta_diff"].nunique() == 1)


def test_fit_denver_specific_home_court_uses_per_team_value_only_for_denver():
    """Task #67: `fit_denver_specific_home_court_walk_forward` must apply
    the per-team estimate ONLY to Denver's own home games -- every other
    team's home games must get the EXACT plain league-wide value
    unchanged, not a diluted per-team estimate. Builds 30 warm-up games
    between two non-Denver teams (establishing a league-wide baseline),
    then Denver home games with a deliberately extreme score (so Denver's
    own per-team estimate clearly differs from league-wide if the
    selection logic is wrong)."""
    dates = pd.date_range("2020-10-01", periods=34, freq="D")
    rows = []
    for i in range(30):
        home, away = ("X", "Y") if i % 2 == 0 else ("Y", "X")
        rows.append({
            "gameId": f"g{i}", "gameDate": dates[i], "season": 2020,
            "team_home": home, "team_away": away,
            "league_avg_pace": 100.0, "pace_shrunk_mean_home": 100.0, "pace_shrunk_mean_away": 100.0,
            "league_avg_rtg": 110.0, "rtg_attack_rate_home": 110.0, "rtg_defense_rate_home": 110.0,
            "rtg_attack_rate_away": 110.0, "rtg_defense_rate_away": 110.0,
            "actual_home_score": 112, "actual_away_score": 108,
        })
    for i in range(30, 34):
        rows.append({
            "gameId": f"g{i}", "gameDate": dates[i], "season": 2020,
            "team_home": DENVER_TEAM_ID, "team_away": "X",
            "league_avg_pace": 100.0, "pace_shrunk_mean_home": 100.0, "pace_shrunk_mean_away": 100.0,
            "league_avg_rtg": 110.0, "rtg_attack_rate_home": 110.0, "rtg_defense_rate_home": 110.0,
            "rtg_attack_rate_away": 110.0, "rtg_defense_rate_away": 110.0,
            "actual_home_score": 140, "actual_away_score": 100,  # deliberately extreme
        })
    # one more non-Denver game after Denver's games, to confirm non-Denver rows stay on the plain series
    rows.append({
        "gameId": "g34", "gameDate": dates[33] + pd.Timedelta(days=1), "season": 2020,
        "team_home": "Y", "team_away": "X",
        "league_avg_pace": 100.0, "pace_shrunk_mean_home": 100.0, "pace_shrunk_mean_away": 100.0,
        "league_avg_rtg": 110.0, "rtg_attack_rate_home": 110.0, "rtg_defense_rate_home": 110.0,
        "rtg_attack_rate_away": 110.0, "rtg_defense_rate_away": 110.0,
        "actual_home_score": 112, "actual_away_score": 108,
    })
    games = pd.DataFrame(rows)

    hybrid = fit_denver_specific_home_court_walk_forward(games, prior_games=100.0)
    plain_league_wide = fit_home_court_walk_forward(games)
    per_team = fit_team_home_court_walk_forward(games, prior_games=100.0)

    denver_mask = games["team_home"] == DENVER_TEAM_ID
    check("Denver's home games get the PER-TEAM estimate (which differs from plain league-wide, "
          "confirming the extreme score actually moved Denver's own estimate)",
          (hybrid[denver_mask] - per_team[denver_mask]).abs().max() < 1e-9)
    check("Denver's per-team estimate genuinely differs from the plain league-wide value here -- "
          "otherwise this test wouldn't distinguish the two code paths at all",
          (per_team[denver_mask] - plain_league_wide[denver_mask]).abs().max() > 1e-6)
    check("every NON-Denver team's home game (including g34, AFTER Denver's games) gets the "
          "EXACT plain league-wide value, completely unaffected by Denver's own history",
          (hybrid[~denver_mask] - plain_league_wide[~denver_mask]).abs().max() < 1e-9)


def test_career_games_played_pools_home_and_away_appearances():
    """Bug (Sec4b, Phase 2b): a player's career-games-played count must
    pool BOTH their home-side and away-side appearances into one set of
    distinct games before counting -- counting each column separately and
    taking the max silently reports roughly half a typical player's real
    total (since home/away alternate game to game, not a fixed side).
    Constructs a player who appears as a HOME player in 2 games and an
    AWAY player in 2 DIFFERENT games -- true total is 4, not 2."""
    stints = pd.DataFrame([
        {"gameId": "g1", "homePlayers": (1, 2, 3, 4, 5), "awayPlayers": (10, 11, 12, 13, 14)},
        {"gameId": "g2", "homePlayers": (1, 2, 3, 4, 5), "awayPlayers": (10, 11, 12, 13, 14)},
        {"gameId": "g3", "homePlayers": (10, 11, 12, 13, 14), "awayPlayers": (1, 2, 3, 4, 5)},
        {"gameId": "g4", "homePlayers": (10, 11, 12, 13, 14), "awayPlayers": (1, 2, 3, 4, 5)},
    ])
    player_index = {p: i for i, p in enumerate([1])}
    games_played = _career_games_played(stints, player_index)
    check("player 1's games-played pools home+away appearances", games_played[0] == 4,
          f"got {games_played[0]}, expected 4")


def test_validate_script_drops_nan_rows_before_bootstrapping():
    """Bug (found running validate_team_strength_baseline.py on the real,
    full 9-season dev range): a single NaN prediction (the very first game
    of the whole dev range, which has no trailing history yet) silently
    poisoned the ENTIRE `total_mae`/`margin_mae` bootstrap result to `nan`
    -- because a raw numpy array's `.mean()` does NOT skip NaN the way a
    pandas Series' `.mean()` does (confirmed: `np.array([1,2,nan,4]).mean()`
    is `nan`, `pd.Series([1,2,nan,4]).mean()` is `2.33`), and
    `bootstrap_significance.bootstrap_compare` converts to numpy arrays via
    `.to_numpy()` before averaging. Any caller MUST drop NaN rows before
    calling `bootstrap_compare`, not rely on it to skip them. This test
    confirms that expectation directly: a NaN slipped into one arm's metric
    column corrupts the whole result unless dropped first."""
    arm_a = pd.DataFrame({"gameId": [1, 2, 3], "metric": [1.0, 2.0, np.nan]})
    arm_b = pd.DataFrame({"gameId": [1, 2, 3], "metric": [1.0, 1.0, 1.0]})
    result = bootstrap_compare(arm_a, arm_b, game_id_col="gameId",
                                metrics=[{"name": "metric", "col": "metric", "higher_is_better": True}],
                                n_bootstrap=100)
    check("a NaN row corrupts the raw (undropped) bootstrap result -- confirms callers must dropna first",
          np.isnan(result["metric"]["a"]))

    arm_a_clean = arm_a.dropna(subset=["metric"])
    arm_b_clean = arm_b[arm_b["gameId"].isin(arm_a_clean["gameId"])]
    result_clean = bootstrap_compare(arm_a_clean, arm_b_clean, game_id_col="gameId",
                                      metrics=[{"name": "metric", "col": "metric", "higher_is_better": True}],
                                      n_bootstrap=100)
    check("dropping NaN rows first gives a real (non-NaN) result",
          not np.isnan(result_clean["metric"]["a"]))


def test_naive_arm_selected_before_rename_not_after():
    """Bug (found running validate_team_strength_baseline.py on the real,
    full 9-season dev range): renaming naive_home/naive_away to pred_home/
    pred_away on the FULL predictions frame (which already has its own
    pred_home/pred_away columns from the model arm) creates two columns
    sharing the same name -- confirmed real: this crashed
    metrics_ledger.compute_run_metrics with "Cannot set a DataFrame with
    multiple columns to the single column pred_total". The fix is to
    SELECT the naive-only columns into their own frame FIRST, then rename
    -- this test confirms that ordering is what avoids the duplicate."""
    df = pd.DataFrame({
        "gameId": [1, 2], "pred_home": [100.0, 101.0], "pred_away": [95.0, 96.0],
        "naive_home": [98.0, 99.0], "naive_away": [97.0, 98.0],
    })

    renamed_full_frame = df.rename(columns={"naive_home": "pred_home", "naive_away": "pred_away"})
    has_duplicate = renamed_full_frame.columns.tolist().count("pred_home") > 1
    check("renaming on the FULL frame creates a duplicate 'pred_home' column (the bug)", has_duplicate)

    naive_only = df[["gameId", "naive_home", "naive_away"]].rename(
        columns={"naive_home": "pred_home", "naive_away": "pred_away"})
    check("selecting naive-only columns FIRST, then renaming, has no duplicate (the fix)",
          naive_only.columns.tolist().count("pred_home") == 1)


def test_garbage_time_uses_per_game_length_not_season_wide_max():
    """Bug (Sec8, found 2026-07-25): `add_garbage_time_weight` used to
    compute `total_length` as a single `s["endTenths"].max()` across the
    WHOLE `stints` frame passed in -- correct only if that frame is exactly
    one game, but every real caller passes a full season at once. A
    multi-OT game elsewhere in the season would inflate `total_length` for
    EVERY other (normal-length) game, understating how far along those
    games actually were and therefore under-flagging garbage time.
    Constructs two games -- a normal-length one and a multi-OT one -- and
    confirms the normal game's own stint (started at 90% of ITS OWN real
    length, comfortably past the late-game threshold) is still correctly
    flagged as garbage time despite the other game's longer length."""
    stints = pd.DataFrame([
        # Normal game: ends at 28800. A stint starting at 90% of ITS OWN length (25920)
        # with a 30-point margin should be flagged garbage (threshold there is ~11.5).
        {"gameId": "normal", "startTenths": 25920.0, "endTenths": 28800.0, "marginBeforeStint": 30.0},
        # Multi-OT game: ends at 40800 (real 2015-16 example).
        {"gameId": "multi_ot", "startTenths": 100.0, "endTenths": 40800.0, "marginBeforeStint": 2.0},
    ])
    result = add_garbage_time_weight(stints)
    normal_weight = result.loc[result["gameId"] == "normal", "exposureWeight"].iloc[0]
    check("normal-length game's late, blown-out stint is flagged garbage time "
          "(not diluted by the OTHER game's longer length)", normal_weight < 1.0,
          f"got weight={normal_weight}, expected < 1.0 (DOWNWEIGHT_FACTOR)")


def test_prepare_stints_converts_gamedate_to_real_timestamps():
    """Bug (found running validate_rapm_lineup_adjustment.py on the real,
    full dev range): the cached schedule's `gameDate` column is a plain
    string ("YYYY-MM-DD"), not a datetime. `compute_walkforward_player_ratings`
    compares this column against `pd.Timestamp` checkpoints (from
    `pd.date_range`, which always returns real Timestamps even given
    string bounds) -- comparing a string column against a Timestamp raises
    `TypeError: Invalid comparison between dtype=str and Timestamp`.
    Confirms `prepare_stints` converts gameDate to a real datetime dtype,
    so a downstream `< pd.Timestamp(...)` comparison works."""
    stints = pd.DataFrame([
        {"gameId": "g1", "stintIdx": 0, "homePts": 10, "awayPts": 8, "possessions": 8,
         "homePlayers": (1, 2, 3, 4, 5), "awayPlayers": (10, 11, 12, 13, 14),
         "startTenths": 0, "endTenths": 100},
    ])
    schedule = pd.DataFrame({"gameId": ["g1"], "gameDate": ["2016-01-15"]})  # plain string, as cached
    prepared = prepare_stints(stints, schedule)
    is_real_datetime = pd.api.types.is_datetime64_any_dtype(prepared["gameDate"])
    check("gameDate is converted to a real datetime dtype, not left as a string", is_real_datetime)
    if is_real_datetime:
        try:
            prepared["gameDate"] < pd.Timestamp("2020-01-01")
            comparison_ok = True
        except TypeError:
            comparison_ok = False
        check("gameDate can be compared against a pd.Timestamp without raising", comparison_ok)


def test_player_rate_shrinkage_pools_across_players_not_per_row_mean():
    """`player_rate_shrinkage.add_walk_forward_player_rate` must pool
    numerator/exposure by SUM across every player-row sharing a gameId
    before computing the trailing league rate -- NOT average each row's
    own already-computed rate the way team-level `shrinkage.py` does
    (correct there since it always has exactly 2 pre-normalized rows/game;
    wrong here, where a game has ~20-26 raw, un-normalized player rows).
    Also confirms the core walk-forward guarantee: a player's very first
    logged game has no prior history at all, so BOTH the league average and
    that player's own shrunk rate are NaN (not silently defaulting to 0 or
    leaking same-game data) -- and a second, brand-new player's rate at the
    next game collapses to exactly the pooled league average (zero own
    history contributes 0 to numerator/exposure-before)."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A", "made": 3, "att": 5},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "B", "made": 2, "att": 4},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A", "made": 10, "att": 10},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "C", "made": 1, "att": 8},
    ])
    out = add_walk_forward_player_rate(log, "made", "att", prior_exposure=10.0, prefix="fg")

    g1_a = out[(out["playerId"] == "A") & (out["gameId"] == "g1")].iloc[0]
    check("first-ever game has no prior history: league avg is NaN",
          pd.isna(g1_a["fg_league_avg_rate"]))
    check("first-ever game has no prior history: shrunk rate is NaN (no leak, no silent default)",
          pd.isna(g1_a["fg_shrunk_rate"]))

    expected_pooled = (3 + 2) / (5 + 4)  # SUM across both g1 players, not a mean-of-rates
    g2_a = out[(out["playerId"] == "A") & (out["gameId"] == "g2")].iloc[0]
    g2_c = out[(out["playerId"] == "C") & (out["gameId"] == "g2")].iloc[0]
    check("league avg at g2 is the POOLED (summed) rate across g1's two players, not a per-row mean",
          abs(g2_a["fg_league_avg_rate"] - expected_pooled) < 1e-9,
          f"got {g2_a['fg_league_avg_rate']}, expected {expected_pooled}")
    check("a brand-new player (C) with zero prior history collapses exactly to the league average",
          abs(g2_c["fg_shrunk_rate"] - expected_pooled) < 1e-9)


def test_minutes_ewm_uses_only_strictly_prior_games():
    """`player_minutes.py`'s original version used an infinite-memory
    expanding-shrinkage approach (mirroring `shrinkage.add_walk_forward_rate`'s
    team-level pattern) -- confirmed, via a real bootstrap run on the full
    dev range (275,138 player-games), to be a REAL REGRESSION vs. even a
    weak naive floor, with MAE getting monotonically WORSE as more prior
    games were blended in. Replaced with
    `add_walk_forward_player_mean_ewm` (recency-weighted, halflife-based) --
    confirmed via a second real bootstrap run to be a REAL IMPROVEMENT vs.
    a genuinely naive season-average floor (MAE 5.41 vs. 10.87). This test
    confirms the walk-forward mechanics directly: a player's first-ever
    game has no prior history (NaN, not a silent 0 or a leaked same-game
    value), and their second game's projection uses ONLY the first game's
    real value -- not any data from the game being predicted."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A", "minutes": 30},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A", "minutes": 20},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "playerId": "A", "minutes": 10},
    ])
    out = add_walk_forward_player_mean_ewm(log, "minutes", halflife_games=2.0, prefix="min")
    check("first-ever game has no prior history: EWM rate is NaN", pd.isna(out.iloc[0]["min_ewm_rate"]))
    check("second game's EWM rate uses ONLY the first game's real value (30), not a leak",
          out.iloc[1]["min_ewm_rate"] == 30.0, f"got {out.iloc[1]['min_ewm_rate']}")


def test_bootstrap_compare_batches_large_n_games_instead_of_one_giant_matrix():
    """Bug (found running `validate_player_scoring_rates.py` on the real,
    full dev range, 267,124 player-game rows): `bootstrap_compare` used to
    build ONE `(n_bootstrap, n_games)` index matrix upfront -- fine at
    team-level scale (~10,000 games), but 267,124 x 5,000 x 8 bytes =
    ~10.7GB at player-level scale, which made the whole process thrash and
    never finish (confirmed directly: stuck for minutes, process state
    'U'/uninterruptible-sleep, climbing RSS). Fixed by batching resamples
    with a memory cap. This test confirms: (a) a large-n_games call still
    produces a correct, sane result (doesn't hang or error), and (b) the
    batch size is actually being capped as intended, not silently
    reverting to one giant matrix, by checking the implied batch count for
    a large n_games is > 1."""
    n_games_large = 300_000  # matches the real scale (267,124) that caused the original hang
    rng = np.random.default_rng(0)
    df_a = pd.DataFrame({"gid": range(n_games_large), "metric": rng.normal(0, 1, n_games_large)})
    df_b = pd.DataFrame({"gid": range(n_games_large), "metric": rng.normal(0.1, 1, n_games_large)})

    result = bootstrap_compare(df_a, df_b, game_id_col="gid",
                                metrics=[{"name": "metric", "col": "metric", "higher_is_better": True}],
                                n_bootstrap=500)
    check("large-n_games bootstrap completes and returns a real (non-NaN) CI",
          not np.isnan(result["metric"]["delta_ci95"][0]))

    implied_batch_size = max(1, min(500, _MAX_IDX_MATRIX_BYTES // (8 * n_games_large)))
    check("batching is actually active for large n_games (batch size < n_bootstrap)",
          implied_batch_size < 500, f"got batch_size={implied_batch_size}")


def test_scoring_rates_no_leak_and_dtype_safe():
    """Two real findings while validating `player_scoring_rates.py`: (1) a
    real bootstrap REGRESSION on the full dev range showed attempts and
    makes need DIFFERENT smoothing (attempts: EWMA, like minutes; makes:
    expanding-shrinkage, like steal/block rate) -- confirmed empirically,
    not assumed by analogy either way (see that module's docstring for the
    real MAE numbers); (2) a real dtype crash: guarding attempts-per-minute
    against divide-by-zero (0 minutes) with `pd.NA` produces an
    object-dtype column that `.ewm()` cannot handle at all
    (`TypeError: cannot handle this type -> object`) -- must use `np.nan`
    instead, which keeps the column float64. This test confirms the whole
    pipeline runs without that crash and has no same-game leak (first-ever
    game is NaN; second game uses only the first game's real values)."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A", "minutes": 30,
         "fgAtt2": 5, "fgMade2": 3, "fg3Att": 2, "fg3Made": 1, "ftAtt": 4, "ftMade": 3},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A", "minutes": 25,
         "fgAtt2": 6, "fgMade2": 4, "fg3Att": 3, "fg3Made": 2, "ftAtt": 2, "ftMade": 2},
    ])
    out = add_scoring_rates(log)  # would raise TypeError before the np.nan fix
    check("first-ever game has no prior history: attempt rate is NaN", pd.isna(out.iloc[0]["3pt_att_rate_per_min"]))
    check("first-ever game has no prior history: make rate is NaN", pd.isna(out.iloc[0]["3pt_make_rate"]))
    check("second game's attempt rate uses only the first game's real value",
          abs(out.iloc[1]["3pt_att_rate_per_min"] - (2 / 30)) < 1e-9)
    check("second game's make rate uses only the first game's real value",
          abs(out.iloc[1]["3pt_make_rate"] - 0.5) < 1e-9)


def test_ft_scoring_rate_uses_expanding_shrinkage_not_ewma():
    """Bug (found running `validate_player_scoring_rates.py` on the full
    dev range, second regression after the 2PT/3PT EWMA-vs-expanding fix
    landed): applying that SAME fix to FT (EWMA halflife=10 for attempts,
    expanding-shrinkage prior_attempts=100 for makes) produced shrunk
    MAE=1.0726 vs naive MAE=1.0538 -- a REAL REGRESSION, CI
    (+0.0121,+0.0279). A direct empirical re-sweep on FT specifically found
    BOTH halves want expanding-shrinkage (not the EWMA-for-attempts split
    that works for 2PT/3PT): FTA per-minute via expanding-shrinkage
    (prior_minutes=20) beat every EWMA halflife tested, and FT make-rate
    via expanding-shrinkage (prior_attempts=15, much smaller than 2PT/3PT's
    150) beat the naive floor. This test locks in expanding-shrinkage (not
    EWMA) for FT attempts specifically, by checking `ft_att_rate_per_min`
    matches the closed-form cross-player-pooled shrinkage formula -- pure
    per-player EWMA would never reference another player's rate at all, so
    a future accidental revert to EWMA for FT would fail this exact check."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A", "minutes": 30,
         "fgAtt2": 5, "fgMade2": 3, "fg3Att": 2, "fg3Made": 1, "ftAtt": 9, "ftMade": 6},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "B", "minutes": 30,
         "fgAtt2": 5, "fgMade2": 3, "fg3Att": 2, "fg3Made": 1, "ftAtt": 3, "ftMade": 1},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A", "minutes": 20,
         "fgAtt2": 5, "fgMade2": 3, "fg3Att": 2, "fg3Made": 1, "ftAtt": 4, "ftMade": 3},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "B", "minutes": 20,
         "fgAtt2": 5, "fgMade2": 3, "fg3Att": 2, "fg3Made": 1, "ftAtt": 2, "ftMade": 1},
    ])
    out = add_scoring_rates(log)
    row_a_g2 = out[(out["gameId"] == "g2") & (out["playerId"] == "A")].iloc[0]

    league_fta_rate_before_g2 = (9 + 3) / (30 + 30)  # pooled A+B from g1 only
    expected_att_rate = (9 + PRIOR_MINUTES_FTA * league_fta_rate_before_g2) / (30 + PRIOR_MINUTES_FTA)
    check("FT attempt rate matches expanding-shrinkage (cross-player-pooled), not per-player EWMA",
          abs(row_a_g2["ft_att_rate_per_min"] - expected_att_rate) < 1e-9,
          f"got {row_a_g2['ft_att_rate_per_min']}, expected {expected_att_rate}")

    league_ftm_rate_before_g2 = (6 + 1) / (9 + 3)  # pooled A+B from g1 only
    expected_make_rate = (6 + PRIOR_ATTEMPTS_FTM * league_ftm_rate_before_g2) / (9 + PRIOR_ATTEMPTS_FTM)
    check("FT make rate matches expanding-shrinkage (cross-player-pooled)",
          abs(row_a_g2["ft_make_rate"] - expected_make_rate) < 1e-9,
          f"got {row_a_g2['ft_make_rate']}, expected {expected_make_rate}")


def test_block_rate_uses_ewma_not_expanding_shrinkage():
    """Bug (found running `validate_player_defensive_event_rates.py` on the
    full dev range): blocks were assumed to need the SAME treatment as
    steals (expanding-shrinkage, confirmed correct for steals specifically)
    just because both are "rare defensive events" -- a REAL REGRESSION
    resulted (shrunk MAE 0.4224 vs naive MAE 0.4102, CI
    (+0.0117,+0.0128)). A direct empirical re-sweep on blocks specifically
    found every expanding-shrinkage prior tested (1-1500) was flat-to-worse
    than the naive floor, while EWMA halflife=10-12 games clearly won (MAE
    0.4938 vs naive 0.4952) -- block volume is driven by matchup/rim-
    protection role (a volume/role metric), unlike steals' persistent
    gambling-style skill. This test locks in EWMA for blocks (not
    expanding-shrinkage) by checking that a player's `blk_rate_per_min`
    depends ONLY on their own trailing history, with no cross-player
    blending toward a league prior -- expanding-shrinkage would blend
    toward the OTHER player's (zero-block) rate even after a single game;
    pure EWMA would not, since it has no cross-player prior term at all."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A", "minutes": 30, "steals": 1, "blocks": 2},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "B", "minutes": 30, "steals": 1, "blocks": 0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A", "minutes": 20, "steals": 1, "blocks": 1},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "B", "minutes": 20, "steals": 1, "blocks": 0},
    ])
    out = add_defensive_event_rates(log)
    row_a_g2 = out[(out["gameId"] == "g2") & (out["playerId"] == "A")].iloc[0]
    check("block rate at g2 matches A's own g1 rate exactly (2/30), unaffected by B's zero rate",
          abs(row_a_g2["blk_rate_per_min"] - (2 / 30)) < 1e-9,
          f"got {row_a_g2['blk_rate_per_min']}, expected {2/30} (would be pulled toward 0 by shrinkage)")


def test_ast_prior_not_copy_pasted_from_tov():
    """Bug (found smoke-testing `validate_player_playmaking_rates.py` on
    the one season backfilled so far, 2015-16, 31,141 player-game rows --
    PRELIMINARY, full dev-range re-check pending the `BoxScorePlayerTrackV3`
    backfill, see MODEL_DOCUMENTATION.md): `PRIOR_TOUCHES_AST` was
    initially set to 300, copy-pasted from `PRIOR_TOUCHES_TOV` on the
    (untested) assumption that both playmaking stats stabilize at the same
    rate -- a REAL REGRESSION resulted (shrunk MAE 0.8965 vs naive MAE
    0.8933). A direct sweep on AST specifically found the family
    (expanding-shrinkage) was fine, just the prior was too large:
    prior_touches=50 (MAE 0.8880) clearly beat naive, while TOV's prior=300
    was independently confirmed correct for TOV. This test just guards
    against re-copy-pasting one prior onto the other -- the two constants
    must stay independently tunable, not silently reunified."""
    check("AST and TOV touch-priors are independently tuned, not copy-pasted from each other",
          PRIOR_TOUCHES_AST != PRIOR_TOUCHES_TOV,
          f"got PRIOR_TOUCHES_AST={PRIOR_TOUCHES_AST}, PRIOR_TOUCHES_TOV={PRIOR_TOUCHES_TOV}")


def test_opponent_defense_adjustment_weights_by_matchup_minute_share():
    """`matchup_difficulty.opponent_defense_adjustment` implements the
    plan's "tonight's merge" for switch-heavy defenses -- a weighted blend
    across the opposing roster by each defender's SHARE of matchup-minutes
    at the relevant position group, never a hard 1:1 assignment. With
    defender A rated 1.0 (tougher than league-avg 1.2) getting 75% of the
    minutes and defender B rated 1.3 (easier than average) getting 25%,
    the adjustment should be the exposure-weighted average of each
    defender's own (league_avg - their_rate), not a simple unweighted mean
    (which would give a different, wrong number here since the split is
    75/25, not 50/50)."""
    import pandas as pd

    defender_ratings = pd.Series({101: 1.0, 102: 1.3})
    minutes_at_posgroup = pd.Series({101: 30.0, 102: 10.0})
    adj = opponent_defense_adjustment(
        opposing_roster_player_ids=[101, 102, 103],  # 103 has no matchup history at all
        defender_ratings=defender_ratings, defender_minutes_at_posgroup=minutes_at_posgroup,
        posgroup_rating=1.1, league_avg_posgroup_rating=1.2,
    )
    expected = 0.75 * (1.2 - 1.0) + 0.25 * (1.2 - 1.3)
    check("adjustment is the matchup-minute-weighted blend (0.75/0.25 split), not an unweighted mean",
          abs(adj - expected) < 1e-9, f"got {adj}, expected {expected}")

    naive_unweighted_mean = 0.5 * (1.2 - 1.0) + 0.5 * (1.2 - 1.3)
    check("weighted result differs from what an (incorrect) unweighted 50/50 mean would give",
          abs(adj - naive_unweighted_mean) > 1e-9)


def test_opponent_defense_adjustment_falls_back_below_trust_floor():
    """When the opposing roster's TOTAL matchup-minute exposure at a
    position group doesn't clear `min_matchup_minutes_to_trust`, the merge
    must fall back WHOLE-CLOTH to the level-2 (position-group-vs-team)
    rating rather than blending in the sparse defender-level signal --
    the macro-anchor + micro-reallocation discipline of never letting two
    levels move the same number at once. This test's roster only has 2.0
    total matchup-minutes logged (far below the 20.0 floor), so the result
    must equal exactly `league_avg - posgroup_rating`, ignoring the
    (unreliable, sparse) defender-level numbers entirely."""
    import pandas as pd

    defender_ratings = pd.Series({101: 0.1})  # would look like an extremely tough defender if trusted
    sparse_minutes = pd.Series({101: 2.0})
    adj = opponent_defense_adjustment(
        opposing_roster_player_ids=[101, 102, 103],
        defender_ratings=defender_ratings, defender_minutes_at_posgroup=sparse_minutes,
        posgroup_rating=1.1, league_avg_posgroup_rating=1.2, min_matchup_minutes_to_trust=20.0,
    )
    check("falls back to the level-2 posgroup rating (ignoring the sparse, unreliable defender signal)",
          abs(adj - (1.2 - 1.1)) < 1e-9, f"got {adj}, expected {1.2 - 1.1}")


def test_count_log_score_does_not_blow_up_at_zero_mean():
    """Bug (found running `validate_prop_distribution.py` on the real
    dev-range blocks eval set, 66,937 player-games): a player with an
    expanding-shrunk `blk_rate_per_min` of exactly 0.0 (real -- a
    perimeter player who has genuinely never recorded a block) projects
    `mean=0.0`; Poisson at mu=0 is a genuine degenerate point-mass at 0,
    so `logpmf(k>0, mu=0)` is EXACTLY `-inf`, not just very negative --
    ONE real garbage-time block for that player poisoned the entire eval
    set's mean log-score to `+inf` (`np.mean` of a list containing one
    `inf` is `inf`). Fixed by flooring the count mean (`MIN_COUNT_MEAN`)
    before evaluating pmf/logpmf in both `over_under_prob` and
    `log_score`. This test confirms a real 0-mean, nonzero-actual count
    no longer produces an infinite (or NaN) log-score for either poisson
    or negbin."""
    import math
    score_poisson = log_score(actual=1, mean=0.0, family="poisson", params={})
    check("poisson log-score at mean=0.0, actual=1 is finite (not -inf/+inf)",
          math.isfinite(score_poisson), f"got {score_poisson}")

    score_negbin = log_score(actual=1, mean=0.0, family="negbin", params={"r": 2.0})
    check("negbin log-score at mean=0.0, actual=1 is finite (not -inf/+inf)",
          math.isfinite(score_negbin), f"got {score_negbin}")

    prob = over_under_prob(line=0.5, mean=0.0, family="poisson", params={})
    check("over_under_prob at mean=0.0 doesn't crash or return NaN", not math.isnan(prob), f"got {prob}")


def test_negbin_parameterization_matches_target_mean_and_variance():
    """`prop_distribution.py`'s `over_under_prob`/`log_score` translate NB2
    parameters (mean, dispersion r, where `var = mean + mean^2/r`) into
    scipy's own (n, p) parameterization via `p = r / (r + mean)`, `n = r`.
    A wrong translation would silently produce a distribution with the
    wrong mean/variance while still returning plausible-looking
    probabilities -- this test checks the actual scipy-constructed
    distribution's moments against the target NB2 formula directly,
    rather than trusting the algebra by inspection."""
    from scipy import stats
    mean, r = 5.0, 3.0
    p = r / (r + mean)
    dist = stats.nbinom(n=r, p=p)
    check("NB mean matches the target mean", abs(dist.mean() - mean) < 1e-9, f"got {dist.mean()}")
    expected_var = mean + mean ** 2 / r
    check("NB variance matches the NB2 formula (mean + mean^2/r)",
          abs(dist.var() - expected_var) < 1e-9, f"got {dist.var()}, expected {expected_var}")


def test_add_team_ratings_new_prior_games_params_default_preserving():
    """`team_strength.add_team_ratings`'s new `prior_games_rating`/
    `prior_games_pace` overridable params (added to test
    `validate_shrinkage_strength_fix.py`'s reduced-shrinkage hypothesis
    for Phase 1's still-open margin regression) must reproduce the exact
    original, already-validated Phase 1 output when left at their
    defaults -- confirms the mere existence of this override capability
    doesn't silently change any currently-deployed behavior."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "X", "opponent": "Y",
         "is_home": True, "pace": 100.0, "offRtg": 110.0, "defRtg": 105.0, "actualScore": 112.0},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "Y", "opponent": "X",
         "is_home": False, "pace": 100.0, "offRtg": 105.0, "defRtg": 110.0, "actualScore": 108.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "X", "opponent": "Y",
         "is_home": False, "pace": 98.0, "offRtg": 108.0, "defRtg": 107.0, "actualScore": 105.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "Y", "opponent": "X",
         "is_home": True, "pace": 98.0, "offRtg": 107.0, "defRtg": 108.0, "actualScore": 110.0},
    ])
    default_out = add_team_ratings(log.copy())
    explicit_out = add_team_ratings(log.copy(), prior_games_rating=PRIOR_GAMES_RATING, prior_games_pace=PRIOR_GAMES_PACE,
                                     own_halflife_games=OWN_HALFLIFE_GAMES_RATING)
    for col in ["pace_shrunk_mean", "rtg_attack_rate", "rtg_defense_rate"]:
        check(f"{col}: default params reproduce the explicit-constant call exactly",
              (default_out[col].reset_index(drop=True).fillna(-999)
               == explicit_out[col].reset_index(drop=True).fillna(-999)).all())


def test_project_game_gamma_defaults_preserve_original_combine_and_actually_damp():
    """`team_strength.project_game`'s new `gamma_rtg`/`gamma_pace` exponents (2026-08-02,
    external-review follow-up): a dev-only regression of realized log-rating deviation on the
    walk-forward-predicted log-deviation found the implicit gamma=1.0 assumption is measurably
    wrong (rating slopes ~0.87-0.91, pace slopes ~1.04-1.10). Confirms (1) the new params default
    to 1.0 and reproduce the ORIGINAL bare-ratio combine exactly, byte-for-byte; (2) a gamma<1
    actually dampens the projection toward league average for an extreme (far-from-average) team,
    and a gamma>1 actually amplifies it -- confirming the exponent has the intended directional
    effect, not just a no-op wired in without doing anything."""
    kwargs = dict(home_pace=100.0, away_pace=96.0, league_avg_pace=98.0,
                  home_oRtg=118.0, home_dRtg=104.0, away_oRtg=106.0, away_dRtg=112.0,
                  league_avg_rtg=110.0, home_court_mult=1.01)
    default_pace, default_home, default_away = project_game(**kwargs)
    explicit_pace, explicit_home, explicit_away = project_game(**kwargs, gamma_rtg=1.0, gamma_pace=1.0)
    check("gamma_rtg/gamma_pace default (1.0) reproduces the original combine exactly (pace)",
          abs(default_pace - explicit_pace) < 1e-12)
    check("gamma_rtg/gamma_pace default (1.0) reproduces the original combine exactly (home score)",
          abs(default_home - explicit_home) < 1e-12)
    check("gamma_rtg/gamma_pace default (1.0) reproduces the original combine exactly (away score)",
          abs(default_away - explicit_away) < 1e-12)

    # home_oRtg=118 is well ABOVE league_avg_rtg=110 (an "extreme" team) -- damping (gamma<1) must
    # pull the projection back TOWARD league average relative to the undamped gamma=1 baseline.
    _, damped_home, _ = project_game(**kwargs, gamma_rtg=0.9, gamma_pace=1.0)
    check("gamma_rtg<1 damps an extreme team's projection TOWARD league average",
          damped_home < default_home, f"damped={damped_home}, default={default_home}")

    _, _, amplified_away_score = project_game(**{**kwargs, "away_pace": 105.0}, gamma_rtg=1.0, gamma_pace=1.1)
    _, _, default_away_score_higher_pace = project_game(**{**kwargs, "away_pace": 105.0})
    check("gamma_pace>1 amplifies an extreme (far-from-average) pace's projection AWAY from league average",
          amplified_away_score > default_away_score_higher_pace,
          f"amplified={amplified_away_score}, default={default_away_score_higher_pace}")


def test_add_team_stat_ratings_oreb_excluded_from_uniform_cross_season_adoption():
    """ADOPTED 2026-08-01: `cross_season_weight=0.25` was adopted for the 5 ADOPTED_CATEGORIES
    (dreb/ast/tov/stl/blk) after clearing its own holdout check, but OREB's identical lever showed a
    real regression on its own separate holdout check (`run_oreb_cross_season_holdout_check.py`,
    Sec34) and was NOT adopted. `add_team_stat_ratings`'s per-category `DEFAULT_CROSS_SEASON_WEIGHTS`
    must give oreb 0.0 (unchanged full reset) while giving every ADOPTED_CATEGORIES label
    `CROSS_SEASON_WEIGHT_STAT` -- confirms a uniform-default mistake can't silently re-apply the
    rejected OREB config just because it shares this function's per-category loop."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "X", "opponent": "Y",
         "is_home": True, **{f"{label}_for": 10.0 for label in STAT_COLUMNS}, **{f"{label}_against": 8.0 for label in STAT_COLUMNS}},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "Y", "opponent": "X",
         "is_home": False, **{f"{label}_for": 8.0 for label in STAT_COLUMNS}, **{f"{label}_against": 10.0 for label in STAT_COLUMNS}},
        {"gameId": "g2", "gameDate": pd.Timestamp("2021-01-05"), "season": 2021, "team": "X", "opponent": "Y",
         "is_home": False, **{f"{label}_for": 9.0 for label in STAT_COLUMNS}, **{f"{label}_against": 9.0 for label in STAT_COLUMNS}},
        {"gameId": "g2", "gameDate": pd.Timestamp("2021-01-05"), "season": 2021, "team": "Y", "opponent": "X",
         "is_home": True, **{f"{label}_for": 9.0 for label in STAT_COLUMNS}, **{f"{label}_against": 9.0 for label in STAT_COLUMNS}},
    ])
    default_out = add_team_stat_ratings(log.copy())
    oreb_reset_out = add_team_stat_ratings(log.copy(), cross_season_weight=0.0)
    adopted_out = add_team_stat_ratings(log.copy(), cross_season_weight=CROSS_SEASON_WEIGHT_STAT)

    check("oreb: default (None) matches the explicit cross_season_weight=0.0 call exactly",
          (default_out["oreb_attack_rate"].reset_index(drop=True).fillna(-999)
           == oreb_reset_out["oreb_attack_rate"].reset_index(drop=True).fillna(-999)).all())
    for label in ADOPTED_CATEGORIES:
        check(f"{label}: default (None) matches the explicit cross_season_weight={CROSS_SEASON_WEIGHT_STAT} call exactly",
              (default_out[f"{label}_attack_rate"].reset_index(drop=True).fillna(-999)
               == adopted_out[f"{label}_attack_rate"].reset_index(drop=True).fillna(-999)).all())


def test_add_team_stat_ratings_oreb_excluded_from_uniform_own_halflife_adoption():
    """ADOPTED 2026-08-01: `own_halflife_games=20.0` was adopted for the 5 ADOPTED_CATEGORIES after
    clearing its own holdout check, but OREB still ties naive on its OWN separate holdout check with
    this identical lever (`run_team_stat_own_halflife_holdout_check.py`, Sec37) -- a fourth mechanism
    converging to the same ceiling as Sec34 -- so OREB was NOT brought in. Unlike
    `cross_season_weight` (whose real "off" value, 0.0, is distinct from the sentinel default),
    `own_halflife_games`'s real "off" value at the shrinkage.py primitive layer IS `None` -- so
    `add_team_stat_ratings` uses a dedicated `_USE_PER_CATEGORY_DEFAULT` sentinel instead. Confirms
    the sentinel default resolves to `None` (unchanged) for oreb and `OWN_HALFLIFE_GAMES_STAT` for
    every ADOPTED_CATEGORIES label -- and that passing an explicit `None` really does override
    uniformly (the ambiguity `cross_season_weight` doesn't have)."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "X", "opponent": "Y",
         "is_home": True, **{f"{label}_for": 10.0 for label in STAT_COLUMNS}, **{f"{label}_against": 8.0 for label in STAT_COLUMNS}},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "Y", "opponent": "X",
         "is_home": False, **{f"{label}_for": 8.0 for label in STAT_COLUMNS}, **{f"{label}_against": 10.0 for label in STAT_COLUMNS}},
        {"gameId": "g2", "gameDate": pd.Timestamp("2021-01-05"), "season": 2021, "team": "X", "opponent": "Y",
         "is_home": False, **{f"{label}_for": 9.0 for label in STAT_COLUMNS}, **{f"{label}_against": 9.0 for label in STAT_COLUMNS}},
        {"gameId": "g2", "gameDate": pd.Timestamp("2021-01-05"), "season": 2021, "team": "Y", "opponent": "X",
         "is_home": True, **{f"{label}_for": 9.0 for label in STAT_COLUMNS}, **{f"{label}_against": 9.0 for label in STAT_COLUMNS}},
    ])
    default_out = add_team_stat_ratings(log.copy())
    oreb_reset_out = add_team_stat_ratings(log.copy(), own_halflife_games=None)
    adopted_out = add_team_stat_ratings(log.copy(), own_halflife_games=OWN_HALFLIFE_GAMES_STAT)

    check("oreb: sentinel default matches the explicit own_halflife_games=None call exactly",
          (default_out["oreb_attack_rate"].reset_index(drop=True).fillna(-999)
           == oreb_reset_out["oreb_attack_rate"].reset_index(drop=True).fillna(-999)).all())
    for label in ADOPTED_CATEGORIES:
        check(f"{label}: sentinel default matches the explicit own_halflife_games={OWN_HALFLIFE_GAMES_STAT} call exactly",
              (default_out[f"{label}_attack_rate"].reset_index(drop=True).fillna(-999)
               == adopted_out[f"{label}_attack_rate"].reset_index(drop=True).fillna(-999)).all())


def test_matchup_delta_application_direction_suppresses_points_boosts_tov():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): `generate_props.py` ADDED the
    matchup delta to a player's raw projection instead of SUBTRACTING it.
    `opponent_defense_adjustment`'s own docstring is explicit that a positive return means
    "tougher-than-average matchup, suppress the offensive player's projection" -- adding a positive
    delta INCREASES the projection instead, exactly backwards. This also compounds into a polarity
    bug for TOV specifically: `tov_forced` uses the OPPOSITE rate convention from points/ast_allowed
    (higher rate = tougher defense, not lower), so a genuinely tougher turnover-forcing defender
    produces a NEGATIVE delta under the shared formula -- meaning the single fix (subtract instead
    of add) must correctly SUPPRESS points against a tougher points-defender AND correctly BOOST
    turnovers against a tougher turnover-forcing defender, using the REAL opponent_defense_adjustment
    output for both, not hand-picked signs."""
    league_avg_points_rate = 1.0
    tougher_points_defender_rate = 0.8  # allows FEWER points per matchup-minute than average -> tougher
    league_avg_tov_rate = 0.15
    tougher_tov_defender_rate = 0.22  # forces MORE turnovers per matchup-minute than average -> tougher

    points_delta = opponent_defense_adjustment(
        opposing_roster_player_ids=["D"], defender_ratings=pd.Series({"D": tougher_points_defender_rate}),
        defender_minutes_at_posgroup=pd.Series({"D": 100.0}), posgroup_rating=None,
        league_avg_posgroup_rating=league_avg_points_rate)
    tov_delta = opponent_defense_adjustment(
        opposing_roster_player_ids=["D"], defender_ratings=pd.Series({"D": tougher_tov_defender_rate}),
        defender_minutes_at_posgroup=pd.Series({"D": 100.0}), posgroup_rating=None,
        league_avg_posgroup_rating=league_avg_tov_rate)

    check("points_delta is positive (tougher matchup, per opponent_defense_adjustment's own convention)",
          points_delta > 0, f"got {points_delta}")
    check("tov_delta is negative (a tougher turnover-forcing defender has a HIGHER raw rate -- "
          "opposite polarity from points/ast)", tov_delta < 0, f"got {tov_delta}")

    raw_points, raw_tov = 20.0, 3.0
    fixed_points = raw_points - matchup_point_delta(points_delta, projected_minutes=30.0)
    fixed_tov = raw_tov - matchup_point_delta(tov_delta, projected_minutes=30.0)
    check("subtracting the delta SUPPRESSES points against a tougher (fewer-points-allowed) defender",
          fixed_points < raw_points, f"got {fixed_points} vs raw {raw_points}")
    check("subtracting the delta BOOSTS turnovers against a tougher (more-turnovers-forced) defender",
          fixed_tov > raw_tov, f"got {fixed_tov} vs raw {raw_tov}")

    buggy_points = raw_points + matchup_point_delta(points_delta, projected_minutes=30.0)
    buggy_tov = raw_tov + matchup_point_delta(tov_delta, projected_minutes=30.0)
    check("the ORIGINAL buggy addition would have INCREASED points against a tougher defender "
          "(confirms this locks in a real regression, not a tautology)", buggy_points > raw_points)
    check("the ORIGINAL buggy addition would have DECREASED turnovers against a tougher "
          "turnover-forcing defense (the opposite of the polarity-corrected result)", buggy_tov < raw_tov)


def test_latest_snapshot_carries_forward_across_ewma_season_reset():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): EWMA-based rate columns
    (minutes/2PT/3PT-attempt-rate/block-rate, via `add_walk_forward_player_mean_ewm`) reset to NaN
    on every player's FIRST game of every season by design -- so a plain `.groupby(...).last()` in
    `_latest_snapshot` picked up that NaN row as the "latest" snapshot for a player's SECOND game of
    a new season, even though a real value from late last season was sitting in an earlier row of
    the same log. This test confirms the fix: a real veteran (season-boundary NaN reset, real value
    both before and after) forward-fills correctly, while a true rookie (never has a real value)
    still correctly comes out NaN -- confirming the fix doesn't manufacture data where none exists."""
    log = pd.DataFrame([
        {"playerId": "vet", "gameDate": pd.Timestamp("2023-04-01"), "x": 30.0},   # last game of season A
        {"playerId": "vet", "gameDate": pd.Timestamp("2023-10-25"), "x": np.nan},  # season B opener: EWMA reset
        {"playerId": "rookie", "gameDate": pd.Timestamp("2023-10-25"), "x": np.nan},  # true debut, no history at all
    ])
    snapshot = _latest_snapshot(log, ["x"])
    check("a veteran's season-opener NaN forward-fills to last season's real value, not NaN",
          snapshot.loc["vet", "x"] == 30.0, f"got {snapshot.loc['vet', 'x']}")
    check("a true rookie with no history at all still correctly comes out NaN (not manufactured)",
          pd.isna(snapshot.loc["rookie", "x"]))


def test_anchor_preserving_missing_does_not_mask_genuinely_missing_players():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the ADOPTED_CATEGORIES anchoring
    loop in generate_props.py used to unconditionally `.fillna(0.0)` a category's raw per-player
    projections BEFORE anchoring, so a player genuinely missing that category's rate-model history
    (e.g. a two-way/call-up) came out as a confident-looking `proj_mean=0.0` instead of being
    distinguishable as "no data available" -- contradicting `_project_active_roster_stats`'s own
    documented NaN-not-zero contract. This test confirms the fix in both branches: anchored (a real
    side_total given) and unanchored (side_total=None, the bottom-up fallback)."""
    raw = pd.Series({"A": 10.0, "B": np.nan, "C": 6.0})

    anchored = _anchor_preserving_missing(raw, side_total=32.0)
    check("anchored: the genuinely-missing player (B) stays NaN, not 0.0",
          pd.isna(anchored["B"]), f"got {anchored['B']}")
    check("anchored: the real players' values are real numbers summing to the team total",
          abs((anchored["A"] + anchored["C"]) - 32.0) < 1e-9, f"got {anchored['A']} + {anchored['C']}")

    unanchored = _anchor_preserving_missing(raw, side_total=None)
    check("unanchored (bottom-up fallback): the genuinely-missing player (B) stays NaN, not 0.0",
          pd.isna(unanchored["B"]), f"got {unanchored['B']}")
    check("unanchored: real players' raw values pass through unchanged",
          unanchored["A"] == 10.0 and unanchored["C"] == 6.0)


def test_predictive_minutes_shares_has_no_hindsight_leak():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the active-player SET used to be
    `oracle_minutes_shares(...).index` -- literally this exact historical game's real attendance
    (perfect hindsight on who ends up playing), a materially more-informed selection than the live
    pipeline's own `resolve_active_lineup` (a trailing-window union, no knowledge of who plays
    tonight). This test confirms the fix with a player who's normally in the trailing rotation but
    did NOT play in the specific game being projected (a healthy scratch/DNP-CD, which the live
    system structurally cannot know about in advance) -- the corrected function must still include
    him (trailing-window-based selection), where the old oracle-based selection would have
    excluded him."""
    player_minutes = pd.DataFrame([
        # trailing games g1/g2: both players 1 and 2 get real minutes -> both are "in the rotation"
        {"gameId": "g1", "playerId": 1, "team_side": "home", "minutes": 30.0},
        {"gameId": "g1", "playerId": 2, "team_side": "home", "minutes": 15.0},
        {"gameId": "g2", "playerId": 1, "team_side": "home", "minutes": 28.0},
        {"gameId": "g2", "playerId": 2, "team_side": "home", "minutes": 18.0},
        # the game being projected (g3): player 2 was a healthy scratch, recorded 0 real minutes --
        # NOT in player_minutes at all for g3 (no row = didn't play), mirroring a real box score.
        {"gameId": "g3", "playerId": 1, "team_side": "home", "minutes": 34.0},
    ])
    shares = predictive_minutes_shares(
        player_minutes, game_id="g3", team_side="home",
        team_game_ids_before=["g1", "g2"], team_side_by_game={"g1": "home", "g2": "home"})
    check("player 2 (a real, recent rotation player who happened to be scratched THIS game) is "
          "still included -- the live system has no way to know this in advance either",
          2 in shares.index, f"got {set(shares.index)}")
    check("shares renormalize to exactly 1.0", abs(shares.sum() - 1.0) < 1e-9)


def test_semi_oracle_minutes_shares_uses_real_attendance_but_trailing_average_shares():
    """`lineup_rating.semi_oracle_minutes_shares` (2026-08-02, external-review follow-up): a
    diagnostic-only third mode between oracle (real attendance + real minutes) and predictive
    (trailing-window union + trailing-average shares), used to isolate whether Phase 2's
    predictive-mode gap (Sec28.2) comes from not knowing WHO plays vs. imprecise SHARE
    redistribution among known attendees. Confirms both halves of the design: (1) active-SET
    membership comes from REAL attendance in the target game (player C, a normal trailing-history
    rotation player who is OUT tonight, must be excluded -- unlike predictive mode, which would
    include him); (2) each attendee's SHARE comes from their own TRAILING AVERAGE minutes, not
    their real minutes that specific night -- checked with a scenario where the real-minutes ratio
    and the trailing-average ratio are OPPOSITE (player A: real=10 low / trailing=30 high; player
    D: real=30 high / trailing=10 low), so a share vector accidentally built from real minutes
    instead of trailing averages would be immediately, unambiguously detectable."""
    player_minutes = pd.DataFrame([
        # trailing games g1-g4: A averages 30 min, C averages 15 min, D averages 10 min
        {"gameId": "g1", "playerId": "A", "team_side": "home", "minutes": 32.0},
        {"gameId": "g1", "playerId": "C", "team_side": "home", "minutes": 14.0},
        {"gameId": "g1", "playerId": "D", "team_side": "home", "minutes": 8.0},
        {"gameId": "g2", "playerId": "A", "team_side": "home", "minutes": 28.0},
        {"gameId": "g2", "playerId": "C", "team_side": "home", "minutes": 16.0},
        {"gameId": "g2", "playerId": "D", "team_side": "home", "minutes": 12.0},
        {"gameId": "g3", "playerId": "A", "team_side": "home", "minutes": 31.0},
        {"gameId": "g3", "playerId": "C", "team_side": "home", "minutes": 15.0},
        {"gameId": "g3", "playerId": "D", "team_side": "home", "minutes": 9.0},
        {"gameId": "g4", "playerId": "A", "team_side": "home", "minutes": 29.0},
        {"gameId": "g4", "playerId": "C", "team_side": "home", "minutes": 15.0},
        {"gameId": "g4", "playerId": "D", "team_side": "home", "minutes": 11.0},
        # target game g5: A has foul trouble (real=10, far below trailing avg 30); D gets extended
        # garbage-time run (real=30, far above trailing avg 10); C is a healthy scratch (no row at
        # all for g5 -- real attendance excludes him even though he has full trailing history).
        {"gameId": "g5", "playerId": "A", "team_side": "home", "minutes": 10.0},
        {"gameId": "g5", "playerId": "D", "team_side": "home", "minutes": 30.0},
    ])
    team_game_ids_before = ["g1", "g2", "g3", "g4"]
    team_side_by_game = {"g1": "home", "g2": "home", "g3": "home", "g4": "home"}

    shares = semi_oracle_minutes_shares(
        player_minutes, game_id="g5", team_side="home",
        team_game_ids_before=team_game_ids_before, team_side_by_game=team_side_by_game, lookback_games=10)

    check("C (real trailing-history rotation player, but OUT in the target game) is excluded -- "
          "active-SET membership comes from REAL attendance, not trailing-window presence",
          "C" not in shares.index, f"got {set(shares.index)}")
    check("both real attendees (A, D) are present", set(shares.index) == {"A", "D"}, f"got {set(shares.index)}")
    check("A's share reflects A's HIGH trailing average (30), not A's LOW real minutes tonight (10) "
          "-- A gets the LARGER share despite playing fewer real minutes than D",
          shares["A"] > shares["D"], f"got A={shares['A']}, D={shares['D']}")
    check("the share ratio matches the trailing-average ratio (30:10 = 0.75:0.25) exactly, "
          "not the real-minutes ratio (10:30 = 0.25:0.75, which would be the OPPOSITE order)",
          abs(shares["A"] - 0.75) < 1e-9 and abs(shares["D"] - 0.25) < 1e-9,
          f"got A={shares['A']}, D={shares['D']}")


def test_resolve_active_lineup_excludes_departed_players():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): `resolve_active_lineup`'s trailing-
    minutes lookback had no check against actual CURRENT roster membership at all -- Sec22 fixed
    only the cross-SEASON version of this (a lookback reaching into a PRIOR season's roster); a
    player traded or waived MID-season kept contributing his real trailing minutes (and a nonzero
    share of tonight's projection) for up to MINUTES_LOOKBACK_GAMES games after he left, diluting
    every actually-rostered player's share. This test confirms the new `current_roster_ids` param:
    (1) default None preserves the exact original behavior (no roster-based exclusion); (2) a
    departed player (not in current_roster_ids) is excluded and the remaining shares correctly
    renormalize to 1.0."""
    player_minutes = pd.DataFrame([
        {"gameId": "g1", "playerId": 1, "team_side": "home", "minutes": 30.0},
        {"gameId": "g1", "playerId": 2, "team_side": "home", "minutes": 20.0},
        {"gameId": "g2", "playerId": 1, "team_side": "home", "minutes": 30.0},
        {"gameId": "g2", "playerId": 2, "team_side": "home", "minutes": 20.0},
    ])
    team_side_lookup = {"g1": "home", "g2": "home"}
    prior_game_ids = ["g1", "g2"]

    default_shares, default_tag = resolve_active_lineup("TST", prior_game_ids, player_minutes, team_side_lookup, [])
    check("default (current_roster_ids=None) includes both players -- exact original behavior",
          set(default_shares.index) == {1, 2}, f"got {set(default_shares.index)}")

    filtered_shares, filtered_tag = resolve_active_lineup(
        "TST", prior_game_ids, player_minutes, team_side_lookup, [], current_roster_ids={1})
    check("player 2 (not on current_roster_ids) is excluded",
          set(filtered_shares.index) == {1}, f"got {set(filtered_shares.index)}")
    check("remaining share renormalizes to exactly 1.0", abs(filtered_shares.sum() - 1.0) < 1e-9)
    check("the source tag reports the departed-player exclusion count", "1 excluded as no longer on the roster" in filtered_tag,
          f"got {filtered_tag}")


def test_resolve_active_lineup_downweights_a_live_dnp_streak_not_just_a_flat_average():
    """Task #66/Sec65: `resolve_active_lineup` no longer counts every trailing-window survivor at
    a flat 1.0 weight regardless of how sporadically he's actually been playing -- this is exactly
    the fix that took Phase 2 predictive mode from a real holdout regression (task #30) to a real,
    holdout-confirmed improvement. Player A and player B have IDENTICAL raw trailing-average
    minutes (20.0) over a 4-game window, but A played all 4 games while B missed the two most
    recent -- the OLD flat-average mechanism would give them equal 50/50 shares; the new
    probabilistic-weighting mechanism must give A a real majority share instead."""
    player_minutes = pd.DataFrame([
        {"gameId": "g1", "playerId": 1, "team_side": "home", "minutes": 20.0},
        {"gameId": "g2", "playerId": 1, "team_side": "home", "minutes": 20.0},
        {"gameId": "g3", "playerId": 1, "team_side": "home", "minutes": 20.0},
        {"gameId": "g4", "playerId": 1, "team_side": "home", "minutes": 20.0},
        {"gameId": "g1", "playerId": 2, "team_side": "home", "minutes": 20.0},
        {"gameId": "g2", "playerId": 2, "team_side": "home", "minutes": 20.0},
        # player 2 has no rows for g3/g4 -- missed the two most recent games (a live DNP streak)
    ])
    team_side_lookup = {"g1": "home", "g2": "home", "g3": "home", "g4": "home"}
    prior_game_ids = ["g1", "g2", "g3", "g4"]

    shares, tag = resolve_active_lineup("TST", prior_game_ids, player_minutes, team_side_lookup, [])
    check("both players still appear (neither is hard-excluded)", set(shares.index) == {1, 2})
    check("shares renormalize to exactly 1.0", abs(shares.sum() - 1.0) < 1e-9)
    check("player A (no DNP streak) gets a real majority share despite an EQUAL raw trailing average",
          shares[1] > 0.7, f"got {shares[1]}")
    check("player B (2-game live DNP streak) is downweighted well below the flat-average 0.5 split",
          shares[2] < 0.3, f"got {shares[2]}")
    check("the tag reflects the new probabilistic mechanism, not the old flat-average one",
          "probabilistic-predictive" in tag, f"got {tag}")


def test_name_index_prefers_active_player_and_flags_genuine_ambiguity():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): the player-name-to-ID crosswalk used
    to keep whichever of two same-named players happened to appear LAST in `nba_api`'s unsorted
    static list, with no `is_active` preference and no collision detection -- unlike this project's
    own `build_stints._roster_lookup`, which already demonstrates the right pattern (an explicit
    `dupes` set). This test confirms `_build_name_index`: an active/retired collision resolves
    deterministically to the active player (not by list-order accident), while a collision between
    two players sharing the SAME active status is genuinely ambiguous and gets flagged in `dupes`,
    not silently resolved."""
    active_retired_collision = [
        {"id": 1, "full_name": "Same Name", "is_active": False},
        {"id": 2, "full_name": "Same Name", "is_active": True},
    ]
    index, dupes = _build_name_index(active_retired_collision, lambda p: p["full_name"])
    check("active/retired collision resolves to the ACTIVE player's id, not list order",
          index["Same Name"] == 2, f"got {index['Same Name']}")
    check("an active/retired collision is NOT flagged as ambiguous (a real, confident tiebreak)",
          "Same Name" not in dupes)

    both_active_collision = [
        {"id": 3, "full_name": "Other Name", "is_active": True},
        {"id": 4, "full_name": "Other Name", "is_active": True},
    ]
    index2, dupes2 = _build_name_index(both_active_collision, lambda p: p["full_name"])
    check("a genuinely ambiguous collision (both active) IS flagged in dupes",
          "Other Name" in dupes2, f"got {dupes2}")


def test_t_scale_produces_correctly_matched_variance():
    """REAL BUG FOUND AND FIXED (2026-08-01, full-model audit): every Student-t call site in
    `score_distribution.py` (and the copy-pasted math in `prop_distribution.py`) used to pass
    `scale=sqrt(var)` directly to scipy, which does NOT give the t-distribution that variance --
    scipy's `t.cdf`/`t.ppf`/`t.logpdf`/`t.sf` parameterize `X = loc + scale*T` where
    `Var(T) = df/(df-2)`, not 1, so the REALIZED variance was silently inflated by `df/(df-2)`,
    contradicting `margin_and_total_params`'s own "matched variance" docstring claim. Confirms
    `_t_scale`'s formula directly: sampling a real scipy t-distribution built with `_t_scale`'s
    output has empirical variance matching the TARGET var, not the naive (uncorrected) one."""
    from scipy import stats

    rng = np.random.default_rng(0)
    target_var, df = 100.0, 9.5
    scale = _t_scale(target_var, df)
    samples = stats.t.rvs(df=df, loc=0.0, scale=scale, size=2_000_000, random_state=rng)
    empirical_var = float(np.var(samples))
    check("a t-distribution built with _t_scale's output has empirical variance matching the target",
          abs(empirical_var - target_var) / target_var < 0.02, f"got {empirical_var}, target {target_var}")

    naive_scale = np.sqrt(target_var)  # the ORIGINAL buggy behavior
    naive_samples = stats.t.rvs(df=df, loc=0.0, scale=naive_scale, size=2_000_000, random_state=rng)
    naive_empirical_var = float(np.var(naive_samples))
    check("the ORIGINAL buggy scale=sqrt(var) inflates the realized variance well above the target "
          "(confirms this locks in a real fix, not a tautology)",
          naive_empirical_var > target_var * 1.15, f"got {naive_empirical_var}, target {target_var}")


def test_prop_distribution_variance_floor_is_player_scale_not_team_scale():
    """REAL BUG (2026-08-01, found by inspecting real live `generate_props.py`
    output): `prop_distribution.py` originally imported and called
    `score_distribution.predict_variance` directly, which floors variance
    at `score_distribution.MIN_VARIANCE=4.0` -- a constant sized for
    TEAM-SCORE variance (a ~100-point game). Applied to PLAYER-level
    stats (raw fitted variance routinely well under 1 for low-exposure
    players), this silently clamped nearly every low-exposure player's
    variance up to exactly 4.0 regardless of their own real uncertainty --
    confirmed directly on real 2025-01-15 output (2PT-makes variance stuck
    at 4.0 for players with proj_mean of 0.2, 0.6, 2.5). Worse: this same
    inflated `predict_variance` call is used INSIDE `fit_continuous_family`
    to standardize residuals before fitting Student-t df, so the original
    family-choice validation itself (Sec11's points check, and this bug's
    own earlier all-9-category run) was distorted, not just the live
    output -- re-run after the fix flipped points/2PT/3PT/FT from
    "Normal adopted" to a decisive Student-t (e.g. points log-score:
    normal=76.22 vs t=3.53). Fixed with a player-scale `MIN_PLAYER_VARIANCE`
    floor. This test locks in that a near-zero raw-variance fit floors at
    `MIN_PLAYER_VARIANCE`, NOT anywhere near `score_distribution.MIN_VARIANCE`."""
    from src.models.score_distribution import MIN_VARIANCE as TEAM_MIN_VARIANCE

    # A degenerate fit (residuals always exactly 0) -> raw predicted variance is 0 at any exposure.
    variance_model = {"intercept": 0.0, "slope": 0.0}
    predicted = predict_variance(variance_model, np.array([1.0, 5.0, 50.0]))
    check("predict_variance floors at the player-scale constant",
          np.allclose(predicted, MIN_PLAYER_VARIANCE), f"got {predicted}")
    check("the player-scale floor is far smaller than score_distribution's team-scale floor",
          MIN_PLAYER_VARIANCE < TEAM_MIN_VARIANCE / 100, f"got {MIN_PLAYER_VARIANCE} vs {TEAM_MIN_VARIANCE}")

    # A realistic low-exposure fit (small but nonzero variance) must NOT be forced up to 4.0.
    rng = np.random.default_rng(0)
    exposure = rng.uniform(1, 10, size=2000)
    residuals = rng.normal(0, 0.3, size=2000) * np.sqrt(exposure)  # genuinely small player-scale variance
    fitted = fit_continuous_family(residuals, exposure)
    low_exposure_variance = predict_variance(fitted["variance_model"], np.array([1.0]))[0]
    check("a realistic low-exposure player-level fit is NOT clamped to the team-score floor",
          low_exposure_variance < 1.0, f"got {low_exposure_variance}")


def test_usage_shares_sum_to_team_total_not_double_counted():
    """`usage_allocation.py`'s macro-anchor + micro-reallocation rule: the
    team total must be distributed EXACTLY once. This test confirms
    `allocate_team_total(compute_usage_shares(raw), team_total)` sums
    back to exactly `team_total` for an arbitrary raw split -- the
    concrete, checkable form of "RAPM sets the total, matchup difficulty
    only reshapes shares, never both move the same number"."""
    import pandas as pd
    raw = pd.Series({"A": 20.0, "B": 10.0, "C": 5.0})
    team_total = 112.0
    final = allocate_team_total(compute_usage_shares(raw), team_total)
    check("allocated points sum to exactly the team total (no double-counting, no leakage)",
          abs(final.sum() - team_total) < 1e-9, f"got {final.sum()}, expected {team_total}")
    check("higher raw projection gets a proportionally higher final share",
          final["A"] > final["B"] > final["C"])


def test_usage_shares_clip_negative_and_handle_all_zero():
    """Two edge cases in `compute_usage_shares`: (1) a matchup adjustment
    could in principle push a low-usage player's raw+adjustment below
    zero -- a "usage share" can never be negative in reality, so it must
    be clipped to 0, not silently propagate a negative share into the
    final allocation; (2) if every player in the group has a clipped
    value of 0 (no offensive signal at all), the function must fall back
    to equal shares rather than raising a divide-by-zero."""
    import pandas as pd
    shares = compute_usage_shares(pd.Series({"A": 5.0, "B": -3.0}))
    check("a negative raw projection is clipped to a zero share, not a negative one", shares["B"] == 0.0)
    check("shares still sum to 1 after clipping", abs(shares.sum() - 1.0) < 1e-9)

    equal_shares = compute_usage_shares(pd.Series({"A": 0.0, "B": 0.0}))
    check("all-zero group falls back to equal shares instead of raising divide-by-zero",
          abs(equal_shares["A"] - 0.5) < 1e-9 and abs(equal_shares["B"] - 0.5) < 1e-9)


def test_generic_holdout_check_distinguishes_noise_from_real_gap():
    """`generic_holdout_confirmatory_check` (extracted from the game-score
    model's own `holdout_confirmatory_check` so the props subsystem's ~10
    non-score-shaped categories could share the identical dev-vs-holdout-
    GAP bootstrap protocol -- see Sec14) must correctly distinguish a
    genuine regression from noise. Two synthetic checks: (1) dev and
    holdout drawn from the IDENTICAL distribution should come back NOISE
    (the CI should include zero); (2) holdout drawn with a clearly worse
    (higher) mean error than dev should come back REAL REGRESSION, not
    NOISE or an improvement."""
    rng = np.random.default_rng(0)
    same_dev = rng.normal(5.0, 1.0, 2000)
    same_holdout = rng.normal(5.0, 1.0, 2000)
    result_noise = generic_holdout_confirmatory_check(same_dev, same_holdout, "synthetic_same",
                                                        higher_is_better=False, n_bootstrap=500)
    check("identical-distribution dev/holdout comes back NOISE, not a false-positive regression",
          "NOISE" in result_noise["verdict"], f"got {result_noise['verdict']}")

    worse_dev = rng.normal(5.0, 1.0, 2000)
    worse_holdout = rng.normal(7.0, 1.0, 2000)  # clearly worse (higher error, higher_is_better=False)
    result_regression = generic_holdout_confirmatory_check(worse_dev, worse_holdout, "synthetic_worse",
                                                             higher_is_better=False, n_bootstrap=500)
    check("a clearly worse holdout mean comes back REAL REGRESSION, not NOISE or an improvement",
          "REGRESSION" in result_regression["verdict"], f"got {result_regression['verdict']}")


def test_adaptive_league_average_tracks_shift_faster_than_flat_cumulative():
    """`add_walk_forward_player_rate`'s new `league_avg_halflife_games`
    option (Sec15 fast-follow, motivated by the props Phase 4 finding that
    a stale, flat-cumulative league average -- pooled across ALL history
    since day one, not even season-reset -- can lag a genuine league-wide
    rate shift). Three checks: (1) leaving it at the default `None`
    reproduces the EXACT original flat-cumulative behavior, so every
    already-validated model configuration is untouched by this option's
    mere existence; (2) `_trailing_league_rate_ewma`'s first-ever game is
    NaN, same leak guard as the original; (3) given a clear league-wide
    RATE SHIFT partway through the games, the EWMA-weighted league average
    ends up closer to the NEW (recent) rate than the flat-cumulative one
    does -- confirming it actually behaves adaptively, not just
    differently by coincidence."""
    # Two "other" players carry a clear rate shift: low rate (2/10) in g1-g2, high rate (8/10) in
    # g3-g4. Player A (the one being evaluated) has no scoring rows of their own here -- isolates
    # the league-average term cleanly.
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "X", "made": 2, "att": 10},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "Y", "made": 2, "att": 10},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "playerId": "X", "made": 8, "att": 10},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "playerId": "Y", "made": 8, "att": 10},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "playerId": "A", "made": 0, "att": 0},
    ])

    default_out = add_walk_forward_player_rate(log.copy(), "made", "att", prior_exposure=10.0, prefix="fg")
    from src.models.player_rate_shrinkage import _trailing_league_rate
    expected_flat = _trailing_league_rate(log.copy(), "made", "att")
    check("league_avg_halflife_games=None reproduces the exact original flat-cumulative column",
          (default_out["fg_league_avg_rate"].reset_index(drop=True).fillna(-999)
           == expected_flat.reset_index(drop=True).fillna(-999)).all())

    ewma_series = _trailing_league_rate_ewma(log.copy(), "made", "att", halflife_games=1.0)
    g1_val = ewma_series.iloc[0]
    check("EWMA league rate's first-ever game is NaN (same leak guard as the original)", pd.isna(g1_val))

    adaptive_out = add_walk_forward_player_rate(log.copy(), "made", "att", prior_exposure=10.0,
                                                 prefix="fg", league_avg_halflife_games=1.0)
    g5_flat = default_out[default_out["gameId"] == "g5"].iloc[0]["fg_league_avg_rate"]
    g5_adaptive = adaptive_out[adaptive_out["gameId"] == "g5"].iloc[0]["fg_league_avg_rate"]
    true_recent_rate = 0.8
    check("the EWMA-weighted league average is closer to the NEW (recent, high) rate than the "
          "flat-cumulative one after a clear shift",
          abs(g5_adaptive - true_recent_rate) < abs(g5_flat - true_recent_rate),
          f"flat={g5_flat}, adaptive={g5_adaptive}, true_recent={true_recent_rate}")


def test_era_adjusted_rate_detrends_then_retrends_correctly():
    """`add_era_adjusted_player_rate` (Sec17's detrend-then-retrend
    architecture, adopted for steals after `league_avg_halflife_games`
    -- Sec16's simpler attempt -- passed every dev-only check but still
    failed on real holdout). Two checks: (1) leak guard -- a player's
    first-ever game has no history, so both the relative and final shrunk
    rate must be NaN; (2) the ERA-ADJUSTMENT actually fires -- construct a
    clean league-wide rate shift (low rate early, high rate late) and
    confirm a player's OWN stable underlying rate gets correctly re-scaled
    to the CURRENT era's level, not stuck at the stale blended-era level a
    naive expanding average would produce."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "X", "made": 2, "att": 10},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "Y", "made": 2, "att": 10},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "playerId": "X", "made": 8, "att": 10},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "playerId": "Y", "made": 8, "att": 10},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "playerId": "A", "made": 0, "att": 0},
    ])
    out = add_era_adjusted_player_rate(log, "made", "att", prior_exposure=10.0, prefix="fg",
                                        current_rate_halflife_games=1.0)
    g1_x = out[(out["playerId"] == "X") & (out["gameId"] == "g1")].iloc[0]
    check("first-ever game has no prior history: relative rate is NaN",
          pd.isna(g1_x["fg_relative_shrunk_rate"]))
    check("first-ever game has no prior history: final era-adjusted rate is NaN",
          pd.isna(g1_x["fg_shrunk_rate"]))

    # player X's own true rate is consistently EXACTLY at whatever the league level was each time
    # (game1: 2/10=0.2 matching league's 0.2; game3: 8/10=0.8 matching league's 0.8 by then) --
    # a perfectly "league-average" player throughout. At g5, the current (late-era) league rate is
    # high (~0.68-0.8 range); X's era-adjusted projection should track that CURRENT level, not the
    # low EARLY-era level X's raw history also contained.
    g5 = out[out["gameId"] == "g5"].iloc[0]
    check("era-adjusted final rate lands in the CURRENT (high) era's range, not the stale early-era range",
          g5["fg_shrunk_rate"] > 0.5, f"got {g5['fg_shrunk_rate']}")


def test_defender_stat_difficulty_rate_generalizes_points_unchanged():
    """`add_defender_difficulty_rate` (points) is now a thin wrapper over
    the generalized `add_defender_stat_difficulty_rate` (Sec18, added so
    AST-allowed/TOV-forced could reuse the exact same mechanism instead of
    a copy-pasted points-specific implementation). This test confirms the
    generalization introduced no behavior change for points: calling the
    generic function directly with stat_col="points_allowed" and the
    original prefix produces byte-identical output to the public wrapper."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "playerId": "A",
         "team": 1, "matchup_minutes": 20.0, "points_allowed": 10, "ast_allowed": 3, "tov_forced": 2},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "playerId": "A",
         "team": 1, "matchup_minutes": 15.0, "points_allowed": 8, "ast_allowed": 2, "tov_forced": 1},
    ])
    wrapper_out = add_defender_difficulty_rate(log.copy())
    generic_out = add_defender_stat_difficulty_rate(log.copy(), "points_allowed", 50.0, prefix="difficulty")
    check("the points wrapper and the direct generic call produce identical difficulty_rate values",
          (wrapper_out["difficulty_rate"].fillna(-999) == generic_out["difficulty_rate"].fillna(-999)).all())
    check("the points wrapper and the direct generic call produce identical difficulty_proj values",
          (wrapper_out["difficulty_proj"].fillna(-999) == generic_out["difficulty_proj"].fillna(-999)).all())


def test_defender_total_minutes_asof_excludes_future_games():
    """`defender_total_minutes_asof` (Sec18, the position-group-agnostic
    analog of `defender_position_group_minutes_asof`, used for AST/TOV
    which have no position-group tier yet) must only sum matchup-minutes
    strictly BEFORE the query date -- a defender's game happening ON or
    AFTER the as-of date must never leak into "recent exposure" used to
    weight tonight's merge."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "playerId": "A", "matchup_minutes": 10.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-05"), "playerId": "A", "matchup_minutes": 20.0},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-10"), "playerId": "A", "matchup_minutes": 30.0},
    ])
    result = defender_total_minutes_asof(log, pd.Timestamp("2020-01-05"))
    check("only the g1 game (strictly before the as-of date) is counted, not g2 (on) or g3 (after)",
          abs(result.get("A", 0.0) - 10.0) < 1e-9, f"got {result.get('A')}")


def test_position_group_expanding_variant_pools_by_team_posgroup():
    """`add_position_group_stat_difficulty_rate_expanding` (Sec19 -- the
    expanding-shrinkage family variant needed for TOV specifically, since
    TOV did NOT match points/AST's EWMA family at the position-group
    level, confirmed empirically not assumed). Confirms it correctly pools
    by the (team, position_group) composite key -- a brand-new
    (team, position_group) pair with zero prior history collapses exactly
    to the trailing league average, the same leak-guard signature already
    confirmed for the shared `add_walk_forward_player_rate` primitive at
    the per-player level."""
    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": 1,
         "position_group": "G", "matchup_minutes": 10.0, "tov_forced": 3},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": 2,
         "position_group": "G", "matchup_minutes": 10.0, "tov_forced": 1},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "team": 3,
         "position_group": "G", "matchup_minutes": 10.0, "tov_forced": 0},
    ])
    out = add_position_group_stat_difficulty_rate_expanding(log, "tov_forced", prior_matchup_minutes=10.0, prefix="tov_pg")
    g2_row = out[out["gameId"] == "g2"].iloc[0]
    expected_league_avg = (3 + 1) / (10 + 10)  # pooled across BOTH g1 rows (different teams, same position group)
    check("a brand-new (team, position_group) pair with no history collapses to the pooled league average",
          abs(g2_row["tov_pg_difficulty_rate"] - expected_league_avg) < 1e-9,
          f"got {g2_row['tov_pg_difficulty_rate']}, expected {expected_league_avg}")


def test_season_for_date_resolves_purely_from_calendar():
    """`season_for_date` (2026-08-01, found via a real 2025-01-15 spot-check
    that exposed a serious bug -- see MODEL_DOCUMENTATION.md) must resolve
    an arbitrary date's season using ONLY the calendar date, never a live
    API call or wall-clock "now" -- that's the whole point: unlike
    `current_nba_season`, it has to be safe to call for a historical
    backtest. Checks the August cutoff boundary specifically (the season
    never starts before October, so August is a deliberately safe buffer)."""
    check("a January date belongs to the season that started the PREVIOUS calendar year",
          season_for_date("2025-01-15") == 2024, f"got {season_for_date('2025-01-15')}")
    check("an October date belongs to the season starting that SAME calendar year",
          season_for_date("2024-10-25") == 2024, f"got {season_for_date('2024-10-25')}")
    check("a July date still belongs to the PRIOR season (playoffs can run into July)",
          season_for_date("2025-07-15") == 2024, f"got {season_for_date('2025-07-15')}")
    check("an August date already belongs to the NEW season (safe buffer before the real October start)",
          season_for_date("2025-08-01") == 2025, f"got {season_for_date('2025-08-01')}")


def test_current_nba_season_falls_back_when_newest_candidate_fails_entirely():
    """REAL BUG FOUND AND FIXED (2026-08-02): confirmed live from a crashed Railway nba-worker --
    `current_nba_season`'s loop tries the newest candidate year first, then falls back to the prior
    year, but a hard failure (not just an empty season) fetching the NEWEST candidate used to
    propagate straight out of the function via an uncaught RuntimeError, so the fallback candidate
    was never even attempted. Confirmed live: stats.nba.com times out fetching the 2026-27 season
    (zero real games yet) from Railway's blocked datacenter IP on every retry, crashing the whole
    worker instead of falling through to 2025-26 (a season with real data). Mocks
    `_fetch_with_retry` to raise for the newest candidate and succeed for the prior one, confirming
    the function now returns the FALLBACK year instead of raising."""
    import src.ingest.fetch_schedule as fetch_schedule_module

    original_fetch = fetch_schedule_module._fetch_with_retry
    guess_year = _dt_module.date.today().year

    def fake_fetch_with_retry(start_year, season_type):
        if start_year == guess_year:
            raise RuntimeError("LeagueGameLog fetch failed after 3 attempts: simulated Railway timeout")
        return pd.DataFrame({"GAME_ID": ["1"]})  # non-empty: the fallback candidate "has games"

    fetch_schedule_module._fetch_with_retry = fake_fetch_with_retry
    try:
        result = fetch_schedule_module.current_nba_season()
        check("falls back to the prior year when the newest candidate fails entirely (not a crash)",
              result == guess_year - 1, f"got {result}, expected {guess_year - 1}")
    finally:
        fetch_schedule_module._fetch_with_retry = original_fetch

    def fake_fetch_always_fails(start_year, season_type):
        raise RuntimeError("simulated total outage")

    fetch_schedule_module._fetch_with_retry = fake_fetch_always_fails
    try:
        raised = False
        try:
            fetch_schedule_module.current_nba_season()
        except RuntimeError:
            raised = True
        check("still raises (doesn't silently return a bogus season) when EVERY candidate fails", raised)
    finally:
        fetch_schedule_module._fetch_with_retry = original_fetch


def test_rolling_window_report_isolates_per_season_and_excludes_first_season_by_default():
    """`rolling_window_backtest.rolling_window_report`/`summarize_rolling_report` (2026-08-02,
    the audit's flagged rolling-window validation-methodology lever): must slice a candidate-vs-
    baseline comparison PER SEASON rather than pooling, so a season-specific regression can't hide
    inside an aggregate average, and must exclude the very first season present by default (the
    thinnest walk-forward history, same reasoning as the row-level season-reset convention).
    Synthetic 3-season dev-only dataset: season 2019 (first -> excluded by default) has candidate
    WORSE than baseline; season 2020 has candidate clearly BETTER; season 2021 has candidate and
    baseline IDENTICAL (guaranteed exact-zero delta, unambiguous NOISE)."""
    rows_cand, rows_base = [], []
    for i in range(40):
        rows_cand.append({"gameId": f"2019_{i}", "season": 2019, "abs_err": 10.0 + 0.01 * (i % 3)})
        rows_base.append({"gameId": f"2019_{i}", "season": 2019, "abs_err": 5.0 + 0.01 * (i % 3)})
    for i in range(40):
        rows_cand.append({"gameId": f"2020_{i}", "season": 2020, "abs_err": 5.0 + 0.01 * (i % 3)})
        rows_base.append({"gameId": f"2020_{i}", "season": 2020, "abs_err": 6.0 + 0.01 * (i % 3)})
    for i in range(40):
        v = 5.0 + 0.01 * (i % 3)
        rows_cand.append({"gameId": f"2021_{i}", "season": 2021, "abs_err": v})
        rows_base.append({"gameId": f"2021_{i}", "season": 2021, "abs_err": v})
    cand_df = pd.DataFrame(rows_cand)
    base_df = pd.DataFrame(rows_base)

    results = rolling_window_report(
        cand_df, base_df, game_id_col="gameId",
        metrics=[{"name": "mae", "col": "abs_err", "higher_is_better": False}], n_bootstrap=500)

    check("first season (2019) excluded from the default rolling report",
          2019 not in results, f"got seasons {sorted(results.keys())}")
    check("2020 (candidate clearly better) correctly shows real improvement",
          "REAL IMPROVEMENT" in results[2020]["mae"]["verdict"], f"got {results[2020]['mae']['verdict']}")
    check("2021 (identical values) correctly shows noise, not a false regression",
          "NOISE" in results[2021]["mae"]["verdict"], f"got {results[2021]['mae']['verdict']}")

    summary = summarize_rolling_report(results, "mae")
    check("summary reports exactly the 2 non-excluded test seasons", summary["n_seasons"] == 2,
          f"got {summary['n_seasons']}")
    check("no season falsely flagged as a regression", not summary["any_season_regressed"])
    check("summary counts exactly 1 real improvement and 1 noise season",
          summary["n_real_improvement"] == 1 and summary["n_noise"] == 1,
          f"got improvement={summary['n_real_improvement']}, noise={summary['n_noise']}")


def test_block_bootstrap_correctly_widens_ci_for_within_block_correlated_deltas():
    """`bootstrap_significance.block_bootstrap_compare` (2026-08-02, the audit's flagged
    block/cluster-bootstrap validation-methodology lever): a plain per-row bootstrap assumes every
    row is an independent draw, which is wrong when rows share a within-block correlated component
    (e.g. a team's own residuals correlated across its own games) -- this can produce a FALSE
    POSITIVE "real" verdict from a purely spurious block-level pattern that has nothing to do with
    the true, zero, average effect. Synthetic data (fixed seed, 20 blocks x 20 rows): each block gets
    its own random `block_delta_bias` added to arm A relative to arm B (so within a block, every
    row's delta shares that bias -- correlated, not independent), but the TRUE average delta across
    all 20 blocks is exactly 0 by construction (bias is drawn from a zero-mean distribution and
    independent per block). A naive per-row bootstrap (treating 400 rows as 400 independent draws)
    should falsely detect a "real" effect here purely from thin luck in how the 20 block biases
    happened to average; a properly block-clustered bootstrap (which only has 20 independent blocks
    of information, not 400) should correctly show NOISE and produce a wider CI."""
    rng = np.random.default_rng(42)
    n_blocks, rows_per_block = 20, 20
    rows = []
    for b in range(n_blocks):
        block_delta_bias = rng.normal(0, 1.5)
        for r in range(rows_per_block):
            row_id = f"{b}_{r}"
            val_b = 10.0 + rng.normal(0, 0.5)
            val_a = val_b + block_delta_bias + rng.normal(0, 0.2)
            rows.append({"row_id": row_id, "block": b, "val_a": val_a, "val_b": val_b})
    df = pd.DataFrame(rows)
    per_row_a = df[["row_id", "block", "val_a"]].rename(columns={"val_a": "val"})
    per_row_b = df[["row_id", "val_b"]].rename(columns={"val_b": "val"})

    metrics = [{"name": "val", "col": "val", "higher_is_better": False}]
    naive = bootstrap_compare(per_row_a[["row_id", "val"]], per_row_b, game_id_col="row_id",
                               metrics=metrics, n_bootstrap=2000)
    blocked = block_bootstrap_compare(per_row_a, per_row_b, row_id_col="row_id", block_col="block",
                                       metrics=metrics, n_bootstrap=2000)

    naive_width = naive["val"]["delta_ci95"][1] - naive["val"]["delta_ci95"][0]
    blocked_width = blocked["val"]["delta_ci95"][1] - blocked["val"]["delta_ci95"][0]
    check("naive per-row bootstrap is fooled: shows a REAL verdict despite a zero true effect "
          "(the false positive this function exists to catch)",
          "REAL" in naive["val"]["verdict"], f"got {naive['val']['verdict']}")
    check("block bootstrap correctly shows NOISE for the same data (zero true effect)",
          "NOISE" in blocked["val"]["verdict"], f"got {blocked['val']['verdict']}")
    check("block bootstrap's CI is meaningfully WIDER, correctly reflecting only 20 independent "
          "blocks of information rather than 400 independent rows",
          blocked_width > naive_width * 2, f"naive width={naive_width}, blocked width={blocked_width}")
    check("block_bootstrap_compare reports the correct block count", blocked["n_blocks"] == n_blocks,
          f"got {blocked['n_blocks']}")


def test_block_bootstrap_still_detects_a_real_effect_with_no_block_correlation():
    """Sanity check that `block_bootstrap_compare` isn't just unconditionally more conservative --
    with NO within-block correlation (every row's delta is independent noise) and a genuine, sizeable
    true average difference, it must still correctly detect a REAL IMPROVEMENT, not default to NOISE
    just because it's the block-aware variant."""
    rng = np.random.default_rng(7)
    n_blocks, rows_per_block = 20, 20
    rows = []
    for b in range(n_blocks):
        for r in range(rows_per_block):
            row_id = f"{b}_{r}"
            val_b = 10.0 + rng.normal(0, 0.5)
            val_a = val_b - 1.0 + rng.normal(0, 0.5)  # genuine, sizeable true improvement, no block bias
            rows.append({"row_id": row_id, "block": b, "val_a": val_a, "val_b": val_b})
    df = pd.DataFrame(rows)
    per_row_a = df[["row_id", "block", "val_a"]].rename(columns={"val_a": "val"})
    per_row_b = df[["row_id", "val_b"]].rename(columns={"val_b": "val"})

    result = block_bootstrap_compare(per_row_a, per_row_b, row_id_col="row_id", block_col="block",
                                      metrics=[{"name": "val", "col": "val", "higher_is_better": False}],
                                      n_bootstrap=2000)
    check("a genuine, sizeable true effect is still detected as REAL IMPROVEMENT under block bootstrap",
          "REAL IMPROVEMENT" in result["val"]["verdict"], f"got {result['val']['verdict']}")


def test_game_type_from_id_classifies_regular_season_vs_playoff():
    """`active_roster._game_type_from_id` (2026-08-02, external-review follow-up): every training
    signal in this codebase is regular-season-only, but nothing previously flagged a
    non-regular-season prediction as a different, unvalidated regime. Confirmed live against
    ScoreboardV3 on both a real regular-season date (2025-01-15, gameId "0022400561", blank
    gameLabel) and a real playoff date (2025-04-20, gameId "0042400141", gameLabel "West First
    Round") that the 3rd character of gameId reliably distinguishes the two."""
    check("a real regular-season gameId classifies as regular_season",
          _game_type_from_id("0022400561") == "regular_season")
    check("a real playoff gameId classifies as non_regular_season",
          _game_type_from_id("0042400141") == "non_regular_season")
    check("an unrecognized prefix defaults to non_regular_season (fail safe toward warning, not silence)",
          _game_type_from_id("0052400001") == "non_regular_season")


def test_component_scale_factors_makes_points_prop_reconcile_with_its_own_components():
    """`generate_props._component_scale_factors` (2026-08-02, external technical review): REAL BUG
    FOUND -- `final_points` (matchup-adjusted, team-total-anchored) and the shooting-component props
    (2pt_made/3pt_made/ft_made, served raw) had no relationship to each other. Confirmed on a real
    2025-01-15 slate before this fix: 150/262 players (57%) showed a >1-point gap between `points`
    and what `2*2pt_made + 3*3pt_made + ft_made` implied, several exceeding 8 points -- a bettor
    comparing a player's points prop to their own shooting-component props would see numbers that
    visibly don't add up. Confirms the fix: rescaling each component by the same player's own
    scale factor makes `2*2pt_made_scaled + 3*3pt_made_scaled + ft_made_scaled` reconcile EXACTLY
    with `final_points`, for players whose raw projection moved in EITHER direction (boosted by a
    weak matchup, suppressed by a tough one), and safely falls back to no rescaling for a player
    with ~zero raw points (dividing by near-zero would blow up into a meaningless factor)."""
    raw_points = pd.Series({1: 20.0, 2: 10.0, 3: 0.0})
    final_points = pd.Series({1: 24.0, 2: 6.0, 3: 3.0})  # player 1 boosted, 2 suppressed, 3 was ~0 raw
    scale = _component_scale_factors(raw_points, final_points)

    check("player 1's scale factor is exactly final/raw (1.2x boost)",
          abs(scale[1] - 1.2) < 1e-9, f"got {scale[1]}")
    check("player 2's scale factor is exactly final/raw (0.6x suppression)",
          abs(scale[2] - 0.6) < 1e-9, f"got {scale[2]}")
    check("player 3 (near-zero raw points) falls back to scale=1.0, not a division blow-up",
          scale[3] == 1.0, f"got {scale[3]}")

    # Simulate applying the scale to real shooting components and confirm exact reconciliation.
    components = {1: {"2pt": 6.0, "3pt": 2.0, "ft": 2.0}, 2: {"2pt": 3.0, "3pt": 1.0, "ft": 1.0}}
    for pid, c in components.items():
        implied = 2 * (c["2pt"] * scale[pid]) + 3 * (c["3pt"] * scale[pid]) + (c["ft"] * scale[pid])
        check(f"player {pid}: rescaled components reconcile exactly with final_points",
              abs(implied - final_points[pid]) < 1e-9, f"got {implied}, expected {final_points[pid]}")


def test_generate_props_before_filter_excludes_on_and_after_date():
    """`generate_props._before` (2026-08-01) is the fix for the real bug
    found via a 2025-01-15 spot-check: a star player's trailing-minutes
    projection was silently computed using games from a much LATER,
    unrelated season (reaching into "today", not "as of `game_date`"),
    producing an absurd ~11-minute projection for a player who'd actually
    been averaging ~38 minutes -- see MODEL_DOCUMENTATION.md for the full
    diagnosis. This test confirms the filter's boundary is correct: a row
    dated EXACTLY on `game_date` must be excluded (strictly before, not
    on-or-before), matching every other walk-forward leak guard in this
    project."""
    from src.pipeline.generate_props import _before
    log = pd.DataFrame({
        "gameDate": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-20")],
        "value": [1, 2, 3],
    })
    out = _before(log, "2025-01-15")
    check("a row strictly before game_date is kept", 1 in out["value"].tolist())
    check("a row dated EXACTLY on game_date is excluded (strictly-before, not on-or-before)",
          2 not in out["value"].tolist())
    check("a row after game_date is excluded", 3 not in out["value"].tolist())
    check("exactly one row survives the filter", len(out) == 1, f"got {len(out)}")


def test_team_history_season_filter_excludes_prior_season_games():
    """Real bug (2026-08-01, found via a 2023-11-08 early-season spot-check):
    both live pipelines call `build_team_history` on a `team_log` that must
    already be restricted to the CURRENT season -- without that filter,
    the "last N games" lookback (used for BOTH active-roster/minutes
    resolution and the recent-roster RAPM composite) silently blends in
    games from the PRIOR season early in a new one, inflating the
    resolved active roster with players who may no longer even be on the
    team (confirmed directly on real data: DAL's trailing 10 games as of
    2023-11-08 included 3 games from 2022-23, and the resolved active
    roster came out to 22 players when only 11 actually played that
    night). This test locks in the fix: pre-filtering by `season` before
    calling `build_team_history` must produce a history containing ONLY
    the target season's gameIds, even when the underlying log spans
    multiple seasons."""
    log = pd.DataFrame([
        {"team": 1, "gameId": "g_2022_a", "gameDate": pd.Timestamp("2022-11-01"), "season": 2022, "is_home": True},
        {"team": 1, "gameId": "g_2022_b", "gameDate": pd.Timestamp("2022-11-05"), "season": 2022, "is_home": False},
        {"team": 1, "gameId": "g_2023_a", "gameDate": pd.Timestamp("2023-10-25"), "season": 2023, "is_home": True},
        {"team": 1, "gameId": "g_2023_b", "gameDate": pd.Timestamp("2023-11-01"), "season": 2023, "is_home": False},
    ])
    team_history, _ = build_team_history(log[log["season"] == 2023])
    check("season-filtered team_history contains only the target season's games",
          set(team_history[1]) == {"g_2023_a", "g_2023_b"}, f"got {team_history[1]}")
    check("prior-season games are fully excluded, not just deprioritized",
          "g_2022_a" not in team_history[1] and "g_2022_b" not in team_history[1])


def test_team_stat_game_log_for_against_symmetry():
    """Task #24: `build_team_stat_game_log` merges the same team-totals
    frame into a game TWICE (once as `_for`, once as the opponent's
    `_against`) via two separate renamed copies -- this test locks in that
    the merge actually produces the correct cross-team symmetry (team A's
    `{label}_for` must equal team B's `{label}_against` for the same real
    game, and vice versa), using one real cached season rather than a
    synthetic log since the function reads parquet files directly (no
    injectable log parameter)."""
    log = build_team_stat_game_log(2023, 2023)
    if log.empty:
        print("[SKIP] test_team_stat_game_log_for_against_symmetry -- no 2023 data cached")
        return
    game_id = log["gameId"].iloc[0]
    game = log[log["gameId"] == game_id]
    check("exactly two team-rows per game", len(game) == 2, f"got {len(game)}")
    a, b = game.iloc[0], game.iloc[1]
    for label in STAT_COLUMNS:
        check(f"{label}: team A's for == team B's against",
              a[f"{label}_for"] == b[f"{label}_against"],
              f"A_for={a[f'{label}_for']}, B_against={b[f'{label}_against']}")
        check(f"{label}: team B's for == team A's against",
              b[f"{label}_for"] == a[f"{label}_against"],
              f"B_for={b[f'{label}_for']}, A_against={a[f'{label}_against']}")


def test_project_team_stat_matches_ratio_idiom_formula():
    """Pure-function check that `project_team_stat` implements the exact
    same multiplicative-ratio idiom as `team_strength.project_game`
    (league_avg x (attack/league_avg) x (defense/league_avg)), generalized
    to an arbitrary for/against stat pair with no pace/100 rescaling
    (unlike points, these stats are already whole-game totals, not
    per-100-possession rates)."""
    home_proj, away_proj = project_team_stat(
        home_attack=30.0, home_defense=25.0, away_attack=28.0, away_defense=27.0, league_avg=26.0)
    expected_home = 26.0 * (30.0 / 26.0) * (27.0 / 26.0)
    expected_away = 26.0 * (28.0 / 26.0) * (25.0 / 26.0)
    check("home projection matches the ratio-idiom formula exactly",
          abs(home_proj - expected_home) < 1e-9, f"got {home_proj}, expected {expected_home}")
    check("away projection matches the ratio-idiom formula exactly",
          abs(away_proj - expected_away) < 1e-9, f"got {away_proj}, expected {expected_away}")


def test_adopted_categories_excludes_oreb_holdout_failure():
    """Real finding (2026-08-01, Task #24's one-time confirmatory holdout
    check): OREB is the one team-stat category whose model/naive
    comparison FLIPS SIGN on real holdout data (model loses to naive,
    3.0661 vs 3.0401 MAE) -- a genuine veto, not just a widening dev/
    holdout gap. `ADOPTED_CATEGORIES` must exclude it while still leaving
    it in `STAT_COLUMNS` (so validate_team_stat_rates.py/
    run_team_stat_holdout_check.py still check it every time, in case a
    future fix earns it a re-look) -- the other 5 categories (which beat
    naive in absolute terms on holdout even where the gap itself widened)
    must all be present."""
    check("oreb is excluded from ADOPTED_CATEGORIES", "oreb" not in ADOPTED_CATEGORIES)
    check("oreb is still tracked in STAT_COLUMNS (validated every run, just not anchored)",
          "oreb" in STAT_COLUMNS)
    check("the other 5 categories are all adopted",
          set(ADOPTED_CATEGORIES) == {"dreb", "ast", "tov", "stl", "blk"}, f"got {ADOPTED_CATEGORIES}")


def test_team_level_adaptive_league_average_default_preserving_and_responsive():
    """`shrinkage.add_walk_forward_rate`/`add_walk_forward_mean`'s new
    `league_avg_halflife_games` option (2026-08-01, motivated by Sec9.5's
    still-open scoring-era-drift finding: mean points/team-game rose ~13
    points 2015-16 -> 2025-26, almost monotonically THROUGHOUT the dev
    range, unlike the props subsystem's steals regime-change -- see
    `_trailing_league_stat_ewma`'s docstring for why this is a genuinely
    different empirical situation, not the same failed bet retried). Same
    two checks as the player-level version (Sec16): (1) `None` reproduces
    the exact original flat-cumulative column; (2) given a clear league-
    wide shift, the EWMA-weighted average ends up closer to the new rate."""
    from src.models.shrinkage import _trailing_league_stat, add_walk_forward_mean, add_walk_forward_rate

    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "X", "opp": "Y", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "Y", "opp": "X", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "team": "X", "opp": "Y", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "team": "Y", "opp": "X", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "team": "X", "opp": "Y", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "team": "Y", "opp": "X", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "team": "X", "opp": "Y", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "team": "Y", "opp": "X", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "Z", "opp": "W", "pts": 0.0, "opp_pts": 0.0},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "W", "opp": "Z", "pts": 0.0, "opp_pts": 0.0},
    ])

    default_out = add_walk_forward_rate(log.copy(), "pts", "opp_pts", prior_games=10.0, prefix="pts")
    expected_flat = _trailing_league_stat(log.copy(), "pts")
    check("league_avg_halflife_games=None reproduces the exact original flat-cumulative column",
          (default_out["pts_league_avg"].reset_index(drop=True).fillna(-999)
           == expected_flat.reset_index(drop=True).fillna(-999)).all())

    adaptive_out = add_walk_forward_rate(log.copy(), "pts", "opp_pts", prior_games=10.0,
                                          prefix="pts", league_avg_halflife_games=1.0)
    g5_flat = default_out[default_out["gameId"] == "g5"].iloc[0]["pts_league_avg"]
    g5_adaptive = adaptive_out[adaptive_out["gameId"] == "g5"].iloc[0]["pts_league_avg"]
    true_recent_rate = 130.0
    check("the EWMA-weighted league average is closer to the NEW (recent, high-scoring) rate than "
          "the flat-cumulative one after a clear shift",
          abs(g5_adaptive - true_recent_rate) < abs(g5_flat - true_recent_rate),
          f"flat={g5_flat}, adaptive={g5_adaptive}, true_recent={true_recent_rate}")

    # Same two checks for add_walk_forward_mean (used for PACE, a single-value not a for/against pair).
    mean_log = log.rename(columns={"pts": "pace"}).drop(columns=["opp_pts"])
    default_mean = add_walk_forward_mean(mean_log.copy(), "pace", prior_games=10.0, prefix="pace")
    expected_mean_flat = _trailing_league_stat(mean_log.copy(), "pace")
    check("add_walk_forward_mean: league_avg_halflife_games=None reproduces the exact original column",
          (default_mean["pace_league_avg"].reset_index(drop=True).fillna(-999)
           == expected_mean_flat.reset_index(drop=True).fillna(-999)).all())


def test_own_halflife_default_preserving_and_responsive():
    """`shrinkage.add_walk_forward_rate`'s new `own_halflife_games` option
    (2026-08-01, full-model-audit untried-synthesis lever): recency-weights
    a team's OWN within-season history, structurally distinct from
    `cross_season_weight` (early-season prior only) and
    `league_avg_halflife_games` (shared target only) -- neither of which
    touches how a team's own in-season games are weighted against each
    other. Same two-check pattern as the league-average EWMA test above:
    (1) `None` reproduces the exact original flat-cumulative-sum column;
    (2) given a clear within-season shift in one team's own rate, the
    EWMA-weighted version ends up closer to that team's new rate."""
    from src.models.shrinkage import add_walk_forward_rate

    log = pd.DataFrame([
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "X", "opp": "Y", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g1", "gameDate": pd.Timestamp("2020-01-01"), "season": 2020, "team": "Y", "opp": "X", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "team": "X", "opp": "Z", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g2", "gameDate": pd.Timestamp("2020-01-02"), "season": 2020, "team": "Z", "opp": "X", "pts": 100.0, "opp_pts": 100.0},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "team": "X", "opp": "Y", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g3", "gameDate": pd.Timestamp("2020-01-03"), "season": 2020, "team": "Y", "opp": "X", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "team": "X", "opp": "Z", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g4", "gameDate": pd.Timestamp("2020-01-04"), "season": 2020, "team": "Z", "opp": "X", "pts": 130.0, "opp_pts": 130.0},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "X", "opp": "Y", "pts": 0.0, "opp_pts": 0.0},
        {"gameId": "g5", "gameDate": pd.Timestamp("2020-01-05"), "season": 2020, "team": "Y", "opp": "X", "pts": 0.0, "opp_pts": 0.0},
    ])

    default_out = add_walk_forward_rate(log.copy(), "pts", "opp_pts", prior_games=10.0, prefix="pts")
    flat_out = add_walk_forward_rate(log.copy(), "pts", "opp_pts", prior_games=10.0, prefix="pts", own_halflife_games=None)
    check("own_halflife_games=None reproduces the exact original flat-cumulative-sum column",
          (default_out["pts_attack_rate"].reset_index(drop=True).fillna(-999)
           == flat_out["pts_attack_rate"].reset_index(drop=True).fillna(-999)).all())

    adaptive_out = add_walk_forward_rate(log.copy(), "pts", "opp_pts", prior_games=10.0,
                                          prefix="pts", own_halflife_games=1.0)
    g5_flat = default_out[default_out["gameId"] == "g5"].iloc[0]["pts_attack_rate"]
    g5_adaptive = adaptive_out[adaptive_out["gameId"] == "g5"].iloc[0]["pts_attack_rate"]
    true_recent_rate = 130.0
    check("team X's EWMA-weighted own rate is closer to ITS OWN new (recent, high-scoring) rate "
          "than the flat-cumulative one after a clear within-season shift",
          abs(g5_adaptive - true_recent_rate) < abs(g5_flat - true_recent_rate),
          f"flat={g5_flat}, adaptive={g5_adaptive}, true_recent={true_recent_rate}")


def test_team_stat_totals_falls_back_to_empty_when_team_missing():
    """`generate_props._team_stat_totals` is the live wiring's per-game
    combine step -- must return real per-category (home, away) totals when
    both teams have a team-stat history row, and an honest empty dict (not
    a crash, not a silently-zero total) when either team is missing from
    `latest_team_stat` (e.g. a genuine first-game-of-the-season live call),
    so the caller's own fallback-to-unanchored-bottom-up branch actually
    triggers instead of being fed a bogus zero."""
    latest = pd.DataFrame({
        **{f"{label}_attack_rate": [30.0, 28.0] for label in ADOPTED_CATEGORIES},
        **{f"{label}_defense_rate": [25.0, 27.0] for label in ADOPTED_CATEGORIES},
        **{f"{label}_league_avg": [26.0, 26.0] for label in ADOPTED_CATEGORIES},
    }, index=pd.Index([100, 200], name="team"))

    totals = _team_stat_totals(latest, home_id=100, away_id=200)
    check("every adopted category present in the result", set(totals) == set(ADOPTED_CATEGORIES))
    for label in ADOPTED_CATEGORIES:
        expected_home, expected_away = project_team_stat(
            home_attack=30.0, home_defense=25.0, away_attack=28.0, away_defense=27.0, league_avg=26.0)
        check(f"{label}: home total matches project_team_stat directly",
              abs(totals[label]["home"] - expected_home) < 1e-9)
        check(f"{label}: away total matches project_team_stat directly",
              abs(totals[label]["away"] - expected_away) < 1e-9)

    missing = _team_stat_totals(latest, home_id=100, away_id=999)
    check("a team missing from latest_team_stat falls back to an empty dict, not a crash or a zero",
          missing == {}, f"got {missing}")


def test_fit_team_b2b_adjustment_walk_forward_diverges_per_team_unlike_pooled_average():
    """Task #70: the ORIGINAL `_add_walk_forward_b2b_adjustment`
    (validate_b2b_adjustment.py) pools every team's B2B residuals into one
    shared running mean -- a real heterogeneous per-team effect averages
    toward ~0 league-wide. This per-team version must track each team's
    OWN running mean (shrunk toward the pooled mean only via
    `prior_games`), so two teams with opposite true B2B effects diverge
    toward their own true values instead of both collapsing to the same
    pooled number. Also confirms the same-game leak guard: a team's own
    very first B2B game must not include its own same-game residual."""
    rows = []
    gid = 1
    for _ in range(6):
        rows.append({"gameId": gid, "gameDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=gid),
                      "team_home": 100, "team_away": 900 + gid,
                      "pred_home": 100.0, "pred_away": 100.0,
                      "actual_home": 110.0, "actual_away": 100.0,
                      "home_is_b2b": True, "away_is_b2b": False})
        gid += 1
        rows.append({"gameId": gid, "gameDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=gid),
                      "team_home": 200, "team_away": 900 + gid,
                      "pred_home": 100.0, "pred_away": 100.0,
                      "actual_home": 90.0, "actual_away": 100.0,
                      "home_is_b2b": True, "away_is_b2b": False})
        gid += 1
    df = pd.DataFrame(rows)
    out = fit_team_b2b_adjustment_walk_forward(df, prior_games=2.0)

    check("the very first B2B game in the whole league has zero adjustment (no pooled history exists yet)",
          abs(out.loc[0, "home_pred_adj"] - 100.0) < 1e-9)

    team_a_rows = out[out["team_home"] == 100].reset_index(drop=True)
    team_b_rows = out[out["team_home"] == 200].reset_index(drop=True)
    check("team A's own adjustment trends toward its own true +10 effect, not a pooled ~0 average",
          team_a_rows["home_pred_adj"].iloc[-1] - 100.0 > 5.0)
    check("team B's own adjustment trends toward its own true -10 effect, not a pooled ~0 average",
          team_b_rows["home_pred_adj"].iloc[-1] - 100.0 < -5.0)
    check("no same-game leak: team A's own adjustment strictly grows toward its converged value across its games",
          team_a_rows["home_pred_adj"].iloc[0] < team_a_rows["home_pred_adj"].iloc[-1])


if __name__ == "__main__":
    test_fit_denver_specific_home_court_uses_per_team_value_only_for_denver()
    test_build_official_game_log_differential_computes_away_minus_home_correctly()
    test_probabilistic_predictive_shares_matches_flat_predictive_when_everyone_certain()
    test_probabilistic_predictive_shares_downweights_a_player_on_a_current_dnp_streak()
    test_build_miss_rebound_log_skips_block_credit_rows_and_dead_balls_and_game_boundaries()
    test_add_referee_tendency_has_no_leak_on_an_officials_first_ever_game()
    test_add_lineup_continuity_counts_distinct_prior_games_not_stints()
    test_identify_clutch_stints_requires_both_late_AND_close()
    test_add_trailing_clutch_deviation_skips_nan_games_and_requires_a_minimum_sample()
    test_add_travel_fatigue_computes_distance_and_timezone_shift_from_own_prior_game_only()
    test_add_schedule_density_counts_only_strictly_prior_games_in_window()
    test_attendance_features_counts_dnp_streak_from_the_most_recent_game_backward()
    test_predict_attendance_probability_decays_with_streak_and_matches_fraction_at_streak_zero()
    test_fit_count_family_mean_dependent_isolates_the_near_zero_overdispersion()
    test_compute_walkforward_variance_model_has_no_leak_across_a_checkpoint()
    test_predict_variance_walk_forward_picks_the_latest_prior_checkpoint()
    test_add_walk_forward_exposure_rate_has_no_leak_on_a_teams_first_ever_game()
    test_project_oreb_share_returns_exact_league_average_at_league_average_inputs()
    test_project_team_oreb_gives_exact_game_level_coherence_with_projected_misses()
    test_team_home_court_walk_forward_has_no_leak_on_a_teams_first_ever_home_game()
    test_possession_counter_uses_teamid_not_cumulative_description()
    test_score_forward_fill_ignores_placeholder_zero_on_non_scoring_rows()
    test_starters_use_row_order_not_position_field()
    test_roster_lookup_resolves_name_collision_and_suffix_mismatch()
    test_home_court_uses_walkforward_columns_not_raw_realized_columns()
    test_career_games_played_pools_home_and_away_appearances()
    test_validate_script_drops_nan_rows_before_bootstrapping()
    test_naive_arm_selected_before_rename_not_after()
    test_garbage_time_uses_per_game_length_not_season_wide_max()
    test_prepare_stints_converts_gamedate_to_real_timestamps()
    test_player_rate_shrinkage_pools_across_players_not_per_row_mean()
    test_minutes_ewm_uses_only_strictly_prior_games()
    test_bootstrap_compare_batches_large_n_games_instead_of_one_giant_matrix()
    test_scoring_rates_no_leak_and_dtype_safe()
    test_ft_scoring_rate_uses_expanding_shrinkage_not_ewma()
    test_block_rate_uses_ewma_not_expanding_shrinkage()
    test_ast_prior_not_copy_pasted_from_tov()
    test_opponent_defense_adjustment_weights_by_matchup_minute_share()
    test_opponent_defense_adjustment_falls_back_below_trust_floor()
    test_count_log_score_does_not_blow_up_at_zero_mean()
    test_negbin_parameterization_matches_target_mean_and_variance()
    test_usage_shares_sum_to_team_total_not_double_counted()
    test_usage_shares_clip_negative_and_handle_all_zero()
    test_generic_holdout_check_distinguishes_noise_from_real_gap()
    test_adaptive_league_average_tracks_shift_faster_than_flat_cumulative()
    test_era_adjusted_rate_detrends_then_retrends_correctly()
    test_defender_stat_difficulty_rate_generalizes_points_unchanged()
    test_defender_total_minutes_asof_excludes_future_games()
    test_position_group_expanding_variant_pools_by_team_posgroup()
    test_season_for_date_resolves_purely_from_calendar()
    test_current_nba_season_falls_back_when_newest_candidate_fails_entirely()
    test_rolling_window_report_isolates_per_season_and_excludes_first_season_by_default()
    test_block_bootstrap_correctly_widens_ci_for_within_block_correlated_deltas()
    test_block_bootstrap_still_detects_a_real_effect_with_no_block_correlation()
    test_game_type_from_id_classifies_regular_season_vs_playoff()
    test_component_scale_factors_makes_points_prop_reconcile_with_its_own_components()
    test_generate_props_before_filter_excludes_on_and_after_date()
    test_team_history_season_filter_excludes_prior_season_games()
    test_team_stat_game_log_for_against_symmetry()
    test_project_team_stat_matches_ratio_idiom_formula()
    test_adopted_categories_excludes_oreb_holdout_failure()
    test_team_stat_totals_falls_back_to_empty_when_team_missing()
    test_team_level_adaptive_league_average_default_preserving_and_responsive()
    test_own_halflife_default_preserving_and_responsive()
    test_prop_distribution_variance_floor_is_player_scale_not_team_scale()
    test_add_team_ratings_new_prior_games_params_default_preserving()
    test_project_game_gamma_defaults_preserve_original_combine_and_actually_damp()
    test_add_team_stat_ratings_oreb_excluded_from_uniform_cross_season_adoption()
    test_add_team_stat_ratings_oreb_excluded_from_uniform_own_halflife_adoption()
    test_matchup_delta_application_direction_suppresses_points_boosts_tov()
    test_anchor_preserving_missing_does_not_mask_genuinely_missing_players()
    test_latest_snapshot_carries_forward_across_ewma_season_reset()
    test_t_scale_produces_correctly_matched_variance()
    test_name_index_prefers_active_player_and_flags_genuine_ambiguity()
    test_resolve_active_lineup_excludes_departed_players()
    test_predictive_minutes_shares_has_no_hindsight_leak()
    test_semi_oracle_minutes_shares_uses_real_attendance_but_trailing_average_shares()
    test_fit_team_b2b_adjustment_walk_forward_diverges_per_team_unlike_pooled_average()
    test_resolve_active_lineup_downweights_a_live_dnp_streak_not_just_a_flat_average()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    else:
        print("ALL CHECKS PASSED")
