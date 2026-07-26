"""Phase 2 (review #1.4): Monte Carlo drive-state game simulator, the NFL
analog of this workspace's MLB `game_simulator.py`. Reuses the MLB
architectural PATTERN (combine context into a distribution -> sample ->
bootstrap-resample a real historical state transition -> loop to a terminal
condition -> wrap in N Monte Carlo trials -> aggregate), not its mechanics
(base-out/innings are baseball-specific; this uses drive state and a
game-clock countdown instead, built on drive_transitions.py's real historical
drive-outcome tables).

Mechanics, and where they simplify real NFL rules (all clearly approximations,
not claimed as exact):
  - A game is a strict alternating sequence of drives (every real drive-ending
    event -- score, punt, turnover, turnover on downs, safety, end of half --
    changes possession; there is no "same team keeps the ball" case), so who
    has the ball next never needs to be modeled explicitly, only START FIELD
    POSITION does (via drive_transitions.py's samplers).
  - Two 1800-second halves. Each half opens with a real, bootstrap-resampled
    kickoff-return field position (drive_transitions.build_opening_position_pool);
    every subsequent drive's start position is resampled conditioned on how
    the PRIOR drive ended. A coin toss (50/50) decides who receives the
    opening kickoff; the other team receives to open the second half
    (standard NFL rule).
  - Each drive's outcome (result category, points, time elapsed) is bootstrap-
    resampled from real historical drives matching (start field-position
    bucket, market-implied-total quantile for the possessing team) --
    DriveOutcomeSampler. v1 conditioned on Layer-1 EPA off/def terciles;
    validated (validate_game_simulator.py) and found to underperform the
    current market-blend point-estimate approach on every metric -- v2
    replaces that with the same market-implied-team-total signal already
    validated in game_environment.py's game-script v2, continuous and
    already opponent-adjusted rather than a coarse 3-bucket split. v2 got
    margin/win-probability to parity with the current model (and beat it
    straight-up) but TOTAL still lagged clearly (MAE 11.63 vs 10.20) --
    each side's score was simulated fully independently, missing the real
    game-level correlation between the two teams' scoring (weather, pace,
    game script affect both offenses together). A v3 experiment (shared
    "game-environment percentile" drawn once per trial, nudging both teams'
    drive draws together -- DriveOutcomeSampler.sample's shared_percentile)
    was TESTED AND REVERTED: it improved margin variance/Brier slightly but
    badly inflated TOTAL variance (MAE 11.63->14.48, CRPS 8.06->11.63),
    since correlating both sides toward the same percentile shrinks the
    variance of their DIFFERENCE (margin) while inflating the variance of
    their SUM (total) -- the opposite of the intended fix. v2 (market-implied
    quantile only, no cross-team correlation) is the current, shipped state;
    the shared_percentile mechanism is kept available in DriveOutcomeSampler,
    disabled by default, for a future more careful attempt.
  - OVERTIME is simplified to sudden-death (first drive to score of any kind
    ends the game), capped at MAX_OT_DRIVES to guard against a pathological
    scoreless loop -- real modern NFL OT rules (both teams get a possession
    unless the first score is a touchdown) are more complex and vary by era
    across this dataset's 2016-2025 span; this is a clearly-documented
    approximation, not a claim of exact rule fidelity. OT affects a small
    minority of games and only adds a modest amount of additional score.
"""

import numpy as np
import pandas as pd

from src.models.drive_transitions import (
    DriveOutcomeSampler,
    NextDrivePositionSampler,
    assign_implied_total_quantile,
    build_drive_table,
    build_opening_position_pool,
)

HALF_SECONDS = 1800
MAX_OT_DRIVES = 12  # guards against a pathological scoreless-OT loop; real OT ends almost always within 2-4 drives


class GameSimulator:
    def __init__(
        self,
        drive_sampler: DriveOutcomeSampler,
        next_pos_sampler: NextDrivePositionSampler,
        opening_pool: np.ndarray,
        seed: int = 0,
    ):
        self.drive_sampler = drive_sampler
        self.next_pos_sampler = next_pos_sampler
        self.opening_pool = opening_pool
        self.rng = np.random.default_rng(seed)

    def _sample_opening_position(self) -> float:
        return float(self.opening_pool[self.rng.integers(0, len(self.opening_pool))])

    def _run_half(self, receiving_team: str, quantiles: dict, seconds: float, score: dict, shared_pct: float) -> None:
        """Mutates `score` in place. quantiles: {"home": int, "away": int} --
        each team's own market-implied-total quantile bin for this game.
        shared_pct: this trial's shared game-environment percentile (see
        DriveOutcomeSampler docstring) -- same value used for every drive,
        both teams, this whole simulated game."""
        offense = receiving_team
        start_pos = self._sample_opening_position()
        elapsed = 0.0
        while elapsed < seconds:
            defense = "away" if offense == "home" else "home"
            start_bucket = int(min(90, max(0, start_pos // 10 * 10)))
            row = self.drive_sampler.sample(start_bucket, quantiles[offense], shared_percentile=shared_pct)
            duration = float(row["time_elapsed"]) if pd.notna(row["time_elapsed"]) and row["time_elapsed"] > 0 else 20.0

            # Explicit half-boundary check (review round 4, #6): this candidate drive's own
            # sampled duration (from the clean, duration-uncontaminated pool) wouldn't fit in
            # the time remaining -- the real generating process is "this drive got cut off by
            # the clock," not "it completed normally right as the half happened to end," so
            # resolve it from the real end-of-half point distribution instead of this row's
            # own (would-be, uninterrupted) outcome.
            if elapsed + duration > seconds:
                eoh_row = self.drive_sampler.sample_end_of_half(start_bucket)
                score[offense] += float(eoh_row["offense_points"])
                score[defense] += float(eoh_row["defense_points"])
                break

            score[offense] += float(row["offense_points"])
            score[defense] += float(row["defense_points"])
            elapsed += duration
            start_pos = self.next_pos_sampler.sample(row["result"])
            offense = defense

    def _run_overtime(self, first_offense: str, quantiles: dict, score: dict, shared_pct: float) -> None:
        offense = first_offense
        start_pos = self._sample_opening_position()
        for _ in range(MAX_OT_DRIVES):
            defense = "away" if offense == "home" else "home"
            start_bucket = int(min(90, max(0, start_pos // 10 * 10)))
            row = self.drive_sampler.sample(start_bucket, quantiles[offense], shared_percentile=shared_pct)
            score[offense] += float(row["offense_points"])
            score[defense] += float(row["defense_points"])
            if score["home"] != score["away"]:
                return
            start_pos = self.next_pos_sampler.sample(row["result"])
            offense = defense
        # pathological scoreless-OT guard exhausted -- leave tied (a real,
        # if extremely rare, NFL outcome under old rules; under current
        # rules this branch should essentially never trigger)

    def simulate_one(self, home_quantile: int, away_quantile: int) -> tuple:
        quantiles = {"home": home_quantile, "away": away_quantile}
        score = {"home": 0.0, "away": 0.0}
        # v3 experiment (shared_pct correlating both teams' draws): TESTED AND
        # REVERTED -- see DriveOutcomeSampler's docstring. It improved margin
        # variance/Brier marginally but badly inflated TOTAL variance (MAE
        # 11.63->14.48, CRPS 8.06->11.63): correlating both sides toward the
        # same percentile shrinks the variance of their DIFFERENCE (margin)
        # while inflating the variance of their SUM (total) -- the opposite
        # of the intended fix. shared_pct=None below disables it while
        # keeping the (validated-negative) mechanism available in
        # DriveOutcomeSampler for a future, more careful attempt (e.g. a
        # much smaller CORRELATION_WEIGHT, or correlating only a subset of
        # drives rather than every one).
        shared_pct = None
        opening_receiver = "home" if self.rng.random() < 0.5 else "away"
        second_half_receiver = "away" if opening_receiver == "home" else "home"
        self._run_half(opening_receiver, quantiles, HALF_SECONDS, score, shared_pct)
        self._run_half(second_half_receiver, quantiles, HALF_SECONDS, score, shared_pct)
        if score["home"] == score["away"]:
            ot_first = "home" if self.rng.random() < 0.5 else "away"
            self._run_overtime(ot_first, quantiles, score, shared_pct)
        return score["home"], score["away"]

    def simulate_game(self, home_quantile: int, away_quantile: int, n_trials: int = 300) -> dict:
        home_scores = np.empty(n_trials)
        away_scores = np.empty(n_trials)
        for i in range(n_trials):
            h, a = self.simulate_one(home_quantile, away_quantile)
            home_scores[i] = h
            away_scores[i] = a
        margins = home_scores - away_scores
        totals = home_scores + away_scores
        return {
            "home_score_mean": float(home_scores.mean()), "home_score_std": float(home_scores.std()),
            "away_score_mean": float(away_scores.mean()), "away_score_std": float(away_scores.std()),
            "margin_mean": float(margins.mean()), "margin_std": float(margins.std()),
            "total_mean": float(totals.mean()), "total_std": float(totals.std()),
            "home_win_prob": float((margins > 0).mean()), "push_prob": float((margins == 0).mean()),
            "margins": margins, "totals": totals,
        }


def build_simulator_for_season_range(pbp: pd.DataFrame, schedules: pd.DataFrame, bin_edges=None, seed: int = 0):
    """Convenience constructor: build all resampling components from real
    historical data in one call. Pass bin_edges (fit on TRAIN via
    drive_transitions.fit_implied_total_bin_edges) when constructing a
    simulator whose pools are meant to be scored against separate TEST data;
    leave None to fit fresh from this same data. Returns (simulator, bin_edges)."""
    drives = build_drive_table(pbp)
    drives, bin_edges = assign_implied_total_quantile(drives, schedules, bin_edges=bin_edges)

    # "End of half" drives don't end because of a score/turnover, they end because the clock
    # hit 0:00, so their recorded time_elapsed is whatever fraction of a drive fit before the
    # half expired -- a real, confirmed truncation (mean 42.6s vs. 165.8s for every other
    # result category, ~7% of the pool). Excluded from the DURATION pool below (round 3, #3).
    #
    # Round 4, #6: excluding them ENTIRELY (round 3's original fix) also discarded their
    # OUTCOME -- nearly always zero points, a real, common NFL event (a drive that gets cut
    # off by the clock), not an artifact. That's why total got slightly worse after the round-3
    # fix: the remaining pool's average points-per-drive rose along with the now-accurate
    # duration. The principled fix models the half boundary explicitly instead of pool
    # surgery: draw a normal candidate drive from the clean (duration-uncontaminated) pool as
    # usual; if ITS sampled duration would exceed the half's remaining time, the real
    # generating process is "this drive got cut short," not "it completed normally and the
    # half happened to end" -- so resolve it via DriveOutcomeSampler.sample_end_of_half()
    # instead, drawing from the REAL end-of-half point distribution specifically (see
    # GameSimulator._run_half). This should fix drive count AND points-per-drive together
    # instead of trading one against the other.
    end_of_half_drives = drives[drives["result"] == "End of half"]
    drives_for_sampling = drives[drives["result"] != "End of half"]
    drive_sampler = DriveOutcomeSampler(drives_for_sampling, seed=seed, end_of_half_drives=end_of_half_drives)
    # NextDrivePositionSampler still excludes end-of-half rows entirely (not just for
    # duration) -- a half-ending drive's `next_start_yardline_100` is the OTHER team's
    # second-half-opening kickoff position (build_drive_table's shift(-1) is grouped by game,
    # not by half), not a legitimate "what happens after this result" transition. This is
    # moot in practice once the explicit clock check below fires (an end-of-half resolution
    # ends the half, so next_pos_sampler is never consulted for it), but kept excluded here
    # for the same reason as before regardless.
    next_pos_sampler = NextDrivePositionSampler(drives_for_sampling, seed=seed + 1)
    opening_pool = build_opening_position_pool(drives)
    sim = GameSimulator(drive_sampler, next_pos_sampler, opening_pool, seed=seed + 2)
    return sim, bin_edges
