"""Walk-forward shrunk team-level PACE/OFF_RATING/DEF_RATING, and the
closed-form pace x rating combine that turns two teams' ratings into a
projected final score for a game.

`build_team_game_log` merges the cached schedule (real final scores, home/
away, date/season) with the cached advanced-box-score team rows (that
game's realized pace/ORtg/DRtg) into one row per team per game -- the shared
input every walk-forward rating in this module is built from.
"""

import pandas as pd

from src.ingest.fetch_schedule import FIRST_DEV_SEASON, season_str
from src.models.shrinkage import add_walk_forward_mean, add_walk_forward_rate
from src.utils.paths import DATA_PROCESSED, DATA_RAW

PRIOR_GAMES_RATING = 12.0  # ADOPTED 2026-08-01 (was 15.0) -- see LEAGUE_AVG_HALFLIFE_GAMES_RATING's
# docstring below for the full joint-fix writeup; MODEL_DOCUMENTATION.md Sec29.
PRIOR_GAMES_PACE = 15.0

CROSS_SEASON_WEIGHT_RATING = 0.3  # ADOPTED 2026-08-01 (was 0.0, i.e. a full within-season reset --
# shrinkage.py's own docstring flagged this parameter as "a genuinely untested hypothesis for NBA
# specifically... not an assumed-correct default" since the day it was introduced, and it was never
# actually swept in this entire project until the full-model audit's research pass flagged it.
# Swept 0.1-1.0 on the recent-dev slice (all vs the already-adopted joint-fix defaults above): EVERY
# value showed a real improvement on total_mae AND margin_mae, the strongest dev-only result of any
# lever tested this session; 0.25-0.35 additionally improved SU. Confirmed on real holdout at 0.3
# (the strongest all-around candidate): total_mae -0.0835 REAL IMPROVEMENT, margin_mae -0.0763 REAL
# IMPROVEMENT, su NOISE (no harm) -- a second genuine, clean win for Phase 1's core rating engine
# found through the audit's research levers, on top of the LEAGUE_AVG_HALFLIFE_GAMES_RATING/
# PRIOR_GAMES_RATING joint fix above. See run_cross_season_weight_holdout_check.py /
# MODEL_DOCUMENTATION.md Sec33.

OWN_HALFLIFE_GAMES_RATING = 40.0  # ADOPTED 2026-08-01 (was None, i.e. a flat equally-weighted
# in-season cumulative mean of the team's OWN rating history) -- a structurally distinct lever from
# both constants below: CROSS_SEASON_WEIGHT_RATING changes the early-season PRIOR only,
# LEAGUE_AVG_HALFLIFE_GAMES_RATING changes the shared TARGET only, but neither touches how a team's
# own in-season games are weighted against each other (game 1 of a season counted exactly as much
# as last night's game). Dev-only Stage 1 (recent-dev slice) swept 5/10/20/40/80 games: 5 was a REAL
# REGRESSION (too aggressive, thrashes on noise), 10-80 all cleared the net-win bar, 20 and 40
# cleared it most cleanly (real improvement on BOTH total_mae and margin_mae). Stage 2 (full dev
# range) confirmed both, with 40 the stronger all-around candidate. Confirmed on real holdout (vs.
# current config): total_mae -0.0499 REAL IMPROVEMENT, margin_mae -0.0412 REAL IMPROVEMENT, su NOISE
# (no harm) -- the fourth genuine, holdout-confirmed win found via the audit's research levers this
# session. The candidate's OWN dev-vs-holdout gap on margin_mae still shows a real widening (the
# same scoring-era-drift phenomenon Sec29 diagnosed, not eliminated by this fix either) -- but it
# copes with that difficulty measurably better than the prior (flat cumulative) config did, same
# decision logic as Sec29/33. See run_own_halflife_holdout_check.py / MODEL_DOCUMENTATION.md Sec36.

LEAGUE_AVG_HALFLIFE_GAMES_RATING = 2000.0  # ADOPTED 2026-08-01 (was flat/infinite-memory, i.e. no
# EWMA at all) -- the first configuration to genuinely resolve Phase 1's long-open margin/scoring-
# era-drift regression (Sec9.5). Two levers were tried INDEPENDENTLY first and neither alone was a
# clean win: Sec24 (recency-weighted league-average target alone) showed a REAL REGRESSION on
# margin_mae at every halflife tested; Sec26 (reduced shrinkage strength alone) showed a REAL
# margin_mae IMPROVEMENT but a REAL total_mae REGRESSION of larger magnitude -- a tradeoff, not a
# win. Testing them TOGETHER (prior_games_rating=12.0 + this halflife) for the first time (a
# full-model-audit research-pass lever, 2026-08-01) gave a genuine net win on BOTH metrics, on REAL
# HOLDOUT data: total_mae -0.2605 (REAL IMPROVEMENT), margin_mae -0.0194 (REAL IMPROVEMENT), su
# NOISE (no harm) -- the first result in this entire investigation (Sec24/26/this) to clear both
# metrics at once. The candidate's OWN dev-vs-holdout GAP on margin_mae still shows a real widening
# (the underlying scoring-era-drift phenomenon is real and not eliminated) -- but it copes with that
# same real difficulty measurably BETTER than the prior (flat-cumulative, prior_games=15) config
# did, which is the actual decision-relevant comparison. See run_joint_margin_fix_holdout_check.py /
# validate_joint_margin_fix.py for the full sweep and one-time confirmatory read.


def build_team_game_log(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per team per game, for every season in [start_year, end_year]
    that has cached schedule + box score data. Regular season only (a team's
    rating shouldn't be trained on playoff-intensity/rotation-shortened
    games mixed in with regular-season ones -- playoffs are a downstream
    prediction target, not dev-training signal, same principle as excluding
    spring training from the MLB sibling's true-talent fit)."""
    schedule_frames, box_frames = [], []
    for y in range(start_year, end_year + 1):
        sched_path = DATA_RAW / f"schedule_{season_str(y)}.parquet"
        box_path = DATA_RAW / f"boxscore_adv_team_{season_str(y)}.parquet"
        if not sched_path.exists() or not box_path.exists():
            continue
        schedule_frames.append(pd.read_parquet(sched_path))
        box_frames.append(pd.read_parquet(box_path))

    if not schedule_frames:
        return pd.DataFrame()

    schedule = pd.concat(schedule_frames, ignore_index=True)
    schedule = schedule[schedule["seasonType"] == "Regular Season"]
    box = pd.concat(box_frames, ignore_index=True)

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        merged = schedule.merge(
            box, left_on=["gameId", f"{side}TeamId"], right_on=["gameId", "teamId"], how="inner")
        rows.append(pd.DataFrame({
            "gameId": merged["gameId"],
            "gameDate": pd.to_datetime(merged["gameDate"]),
            "season": merged["season"],
            "team": merged[f"{side}TeamId"],
            "opponent": merged[f"{other}TeamId"],
            "is_home": side == "home",
            "actualScore": merged[f"{side}Score"],
            "opponentScore": merged[f"{other}Score"],
            "pace": merged["pace"],
            "offRtg": merged["offensiveRating"],
            "defRtg": merged["defensiveRating"],
            "possessions": merged["possessions"],
        }))

    log = pd.concat(rows, ignore_index=True)
    return log.sort_values(["gameDate", "gameId", "is_home"], ascending=[True, True, False]).reset_index(drop=True)


def add_team_ratings(log: pd.DataFrame, cross_season_weight: float = CROSS_SEASON_WEIGHT_RATING,
                      league_avg_halflife_games: float | None = LEAGUE_AVG_HALFLIFE_GAMES_RATING,
                      prior_games_rating: float = PRIOR_GAMES_RATING,
                      prior_games_pace: float = PRIOR_GAMES_PACE,
                      own_halflife_games: float | None = OWN_HALFLIFE_GAMES_RATING) -> pd.DataFrame:
    """Adds walk-forward shrunk pace_shrunk_mean, rtg_attack_rate (OFF),
    rtg_defense_rate (DEF), and their league-average companions.

    `cross_season_weight` (default `CROSS_SEASON_WEIGHT_RATING`, ADOPTED
    2026-08-01 -- see that constant's own docstring): blends a team's own
    immediately-preceding season rate into its early-season prior instead
    of resetting fully to the league average. Pass `0.0` explicitly for
    the ORIGINAL full-reset behavior.

    `league_avg_halflife_games` (default `LEAGUE_AVG_HALFLIFE_GAMES_RATING`,
    ADOPTED 2026-08-01 -- see that constant's own docstring for the full
    joint-fix writeup): passed straight through to `shrinkage.py`'s
    primitives. Pass `None` explicitly to get the ORIGINAL flat infinite-
    memory league average instead (kept available, e.g. for a script that
    specifically wants to reproduce a pre-Sec29 historical result).

    `prior_games_rating`/`prior_games_pace` (default the module-level
    constants -- `PRIOR_GAMES_RATING` is also 2026-08-01-adopted, see its
    own docstring): overridable shrinkage-STRENGTH knobs, distinct from
    `league_avg_halflife_games` (which changes what the blending TARGET
    tracks, not how strongly a team's own rating is pulled toward it).
    Neither lever alone was a clean win (Sec24 target-only, Sec26
    strength-only) -- both were needed together (Sec29).

    `own_halflife_games` (default `OWN_HALFLIFE_GAMES_RATING`, ADOPTED
    2026-08-01 -- see that constant's own docstring): recency-weights the
    team's own within-season rating history (applied to rtg only, not
    pace -- untested for pace). Pass `None` explicitly for the ORIGINAL
    flat, equally-weighted in-season cumulative mean instead."""
    log = add_walk_forward_mean(log, "pace", prior_games_pace, prefix="pace",
                                 cross_season_weight=cross_season_weight,
                                 league_avg_halflife_games=league_avg_halflife_games)
    log = add_walk_forward_rate(log, "offRtg", "defRtg", prior_games_rating, prefix="rtg",
                                 cross_season_weight=cross_season_weight,
                                 league_avg_halflife_games=league_avg_halflife_games,
                                 own_halflife_games=own_halflife_games)
    return log


def project_game(home_pace: float, away_pace: float, league_avg_pace: float,
                  home_oRtg: float, home_dRtg: float, away_oRtg: float, away_dRtg: float,
                  league_avg_rtg: float, home_court_mult: float,
                  gamma_rtg: float = 1.0, gamma_pace: float = 1.0) -> tuple[float, float, float]:
    """The Phase 1 closed-form combine (see MODEL_DOCUMENTATION.md/the plan
    for the derivation): a multiplicative-ratio idiom, one home-court
    multiplier applied symmetrically (home's projected rating scaled up,
    away's scaled down by the same factor) rather than as an additive bonus,
    so it composes cleanly with the ratio structure of the rest of the
    combine. Returns (projected_pace, projected_home_score, projected_away_score).

    `gamma_rtg`/`gamma_pace` (default 1.0, exactly reproducing the original combine
    byte-for-byte): exponents on each deviation-from-league-average ratio term, e.g.
    `(home_oRtg / league_avg_rtg) ** gamma_rtg` instead of the bare ratio. ADDED
    2026-08-02 (external-review follow-up): regressing REALIZED log-rating deviation on the
    walk-forward-PREDICTED log-deviation (dev-only, n=21,474 team-game observations) found the
    implicit gamma=1.0 assumption is measurably wrong -- real outcomes regress toward league
    average MORE than the combine assumes (rating: slope_off=0.907, slope_def=0.868, both real
    overconfidence for teams far from average) while pace UNDER-extrapolates in the opposite
    direction (slope_home=1.038, slope_away=1.097). A single shared gamma per axis is the
    simplest testable hypothesis (not separate off/def or home/away exponents) -- see
    `validate_combine_gamma.py` for the sweep and one-time holdout read. `1.0` preserves the
    original, unscaled ratio for any caller that wants it (e.g. reproducing a pre-existing
    historical result)."""
    projected_pace = league_avg_pace * (home_pace / league_avg_pace) ** gamma_pace * (away_pace / league_avg_pace) ** gamma_pace

    projected_home_rtg = (league_avg_rtg * (home_oRtg / league_avg_rtg) ** gamma_rtg * (away_dRtg / league_avg_rtg) ** gamma_rtg
                          * home_court_mult)
    projected_away_rtg = (league_avg_rtg * (away_oRtg / league_avg_rtg) ** gamma_rtg * (home_dRtg / league_avg_rtg) ** gamma_rtg
                          / home_court_mult)

    projected_home_score = projected_home_rtg * projected_pace / 100.0
    projected_away_score = projected_away_rtg * projected_pace / 100.0
    return projected_pace, projected_home_score, projected_away_score


if __name__ == "__main__":
    from src.ingest.fetch_schedule import current_nba_season
    current = current_nba_season()
    log = build_team_game_log(FIRST_DEV_SEASON, current)
    print(f"team-game log: {len(log)} rows ({log['gameId'].nunique()} games) "
          f"across seasons {sorted(log['season'].unique())}", flush=True)
    if not log.empty:
        rated = add_team_ratings(log)
        out_path = DATA_PROCESSED / "team_game_log_rated.parquet"
        rated.to_parquet(out_path, index=False)
        print(f"saved {out_path}", flush=True)
