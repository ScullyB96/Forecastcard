"""NFL analog of a baseball TransitionTable (Phase 2, review #1.4): builds a
real historical drive-outcome table from PBP for game_simulator.py to
bootstrap-resample from, instead of hand-modeling down-by-down mechanics.

nflverse's PBP data already computes drive-level summaries per play (fixed_drive,
fixed_drive_result, drive_time_of_possession, etc.) -- confirmed this covers
everything needed without re-deriving down/distance mechanics ourselves:
starting field position (first play's yardline_100), a real, verified point
total per drive (posteam_score_post/defteam_score_post delta, not a category
->points guess -- confirmed against real 2024 data: Touchdown mean=6.94,
Field goal mean=2.96, Safety/Opp-touchdown correctly credit the DEFENSE, not
the offense), time elapsed, and a categorical result
(Touchdown/Field goal/Missed field goal/Punt/Turnover/Turnover on downs/
Safety/Opp touchdown/End of half).

Two resampling pools, matching the MLB pattern of bootstrap-resampling real
historical state transitions rather than modeling mechanics from scratch:

1. DRIVE OUTCOMES, stratified by (start field-position bucket, market-implied
   team total quantile). v1 conditioned on Layer-1 EPA off/def terciles (3
   discrete buckets each); validated end-to-end (validate_game_simulator.py)
   and found to underperform the current market-blend point-estimate
   approach on every metric (MAE, CRPS, straight-up%, Brier) -- almost
   certainly because 3-bucket Layer-1 terciles are a much coarser, less
   precise team-strength signal than the real market line the current
   approach is already anchored on. v2 (current) replaces this with the
   MARKET-IMPLIED team total for the possessing team in that specific
   historical game (implied_team_total = total_line/2 +/- spread_line/2,
   the same signal validated in game_environment.py's game-script v2) --
   continuous (quantile-binned, not 3 discrete buckets) and already
   opponent-adjusted (a team's implied total nets out the specific defense
   they're facing that week), so it directly captures real matchup-specific
   market expectations rather than a static, context-free offense/defense
   split.
2. NEXT-DRIVE STARTING FIELD POSITION, conditioned on how the prior drive
   ended (result category) -- avoids modeling punt distance/return yards
   explicitly; a punt's real next-drive starting position is drawn directly
   from the empirical distribution of what real punts in that era produced.
"""

import numpy as np
import pandas as pd

FIELD_BUCKET_SIZE = 10  # 10-yard starting-field-position buckets, 0-100
N_IMPLIED_TOTAL_QUANTILES = 5


def build_drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, fixed_drive): start_yardline_100, result,
    offense_points, defense_points, time_elapsed (seconds), num_plays."""
    reg = pbp[
        (pbp["season_type"] == "REG")
        & pbp["fixed_drive"].notna()
        & pbp["yardline_100"].notna()
        & pbp["game_seconds_remaining"].notna()
    ].sort_values("game_seconds_remaining", ascending=False)

    g = reg.groupby(["season", "week", "game_id", "fixed_drive"])
    drives = g.agg(
        posteam=("posteam", "first"),
        defteam=("defteam", "first"),
        result=("fixed_drive_result", "first"),
        start_secs=("game_seconds_remaining", "first"),
        end_secs=("game_seconds_remaining", "last"),
        posteam_score_start=("posteam_score", "first"),
        posteam_score_end=("posteam_score_post", "last"),
        defteam_score_start=("defteam_score", "first"),
        defteam_score_end=("defteam_score_post", "last"),
        num_plays=("play_type", "size"),
        qtr_start=("qtr", "first"),
    ).reset_index()

    # BUG (caught before shipping): a drive's first PBP row is often the
    # kickoff itself, whose yardline_100 reflects the KICKING team's line
    # (e.g. "kicks from ARI 35"), not the receiving team's real post-return
    # starting position ("Touchback to the BUF 30" -> yardline_100=70, on
    # the very next row). Using the literal first row silently fed the
    # simulator the wrong field position (off by ~35-40 yards, and in the
    # wrong direction) for every drive that starts with a kickoff. Fixed by
    # taking the first non-special-teams play's yardline_100 instead.
    scrimmage = reg[~reg["play_type"].isin(["kickoff", "extra_point"])]
    start_pos = scrimmage.groupby(["game_id", "fixed_drive"])["yardline_100"].first().rename("start_yardline_100")
    drives = drives.merge(start_pos, on=["game_id", "fixed_drive"], how="left")

    # clip at 9: a single real drive cannot legitimately score more than a
    # TD+2pt (8); ~0.68% of drives show higher values (checked directly), a
    # fixed_drive-boundary data-quality tail (most tagged "Opp touchdown" --
    # almost certainly a score immediately followed by another before the
    # drive counter increments) rather than a real single-drive outcome.
    # Uncapped, these would occasionally feed a badly wrong point value into
    # the resampling pool.
    drives["offense_points"] = (drives["posteam_score_end"] - drives["posteam_score_start"]).clip(lower=0, upper=9)
    drives["defense_points"] = (drives["defteam_score_end"] - drives["defteam_score_start"]).clip(lower=0, upper=9)
    drives["time_elapsed"] = (drives["start_secs"] - drives["end_secs"]).clip(lower=0)
    drives["start_bucket"] = (
        (drives["start_yardline_100"] // FIELD_BUCKET_SIZE * FIELD_BUCKET_SIZE).clip(0, 90)
    ).astype("Int64")

    # next drive's real starting field position, within the same game -- the
    # ground truth for "what starting position did this exact result
    # actually lead to", used by the next-drive-position resampler below.
    drives = drives.sort_values(["game_id", "fixed_drive"])
    drives["next_start_yardline_100"] = drives.groupby("game_id")["start_yardline_100"].shift(-1)
    return drives.drop(columns=["posteam_score_start", "posteam_score_end", "defteam_score_start", "defteam_score_end"])


def fit_implied_total_bin_edges(implied_totals: pd.Series, n_quantiles: int = N_IMPLIED_TOTAL_QUANTILES) -> np.ndarray:
    """Fit quantile bin edges from a (TRAIN-only) distribution of implied team
    totals, for reuse when scoring TEST/live games -- bins must be fixed at
    fit time, not re-derived per season, to stay walk-forward honest."""
    qs = np.linspace(0, 1, n_quantiles + 1)
    edges = np.quantile(implied_totals.dropna(), qs)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def implied_total_quantile_bin(value: float, bin_edges: np.ndarray) -> int:
    return int(np.digitize([value], bin_edges)[0] - 1)


def assign_implied_total_quantile(
    drives: pd.DataFrame, schedules: pd.DataFrame, bin_edges: np.ndarray | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach each drive's OWN GAME's market-implied team total for the
    POSSESSING team (implied_team_total = total_line/2 +/- spread_line/2,
    same signal as game_environment.py's game-script v2) and bucket it into
    quantiles. Pass bin_edges (fit on TRAIN via fit_implied_total_bin_edges)
    when scoring TEST/live data; leave None to fit fresh from this data
    (TRAIN-time construction). Returns (drives_with_quantile, bin_edges) so
    the same edges can be reused later."""
    sched = schedules[schedules["game_type"] == "REG"][["game_id", "home_team", "away_team", "spread_line", "total_line"]]
    drives = drives.merge(sched, on="game_id", how="left")
    drives["posteam_implied_total"] = np.where(
        drives["posteam"] == drives["home_team"],
        drives["total_line"] / 2 + drives["spread_line"] / 2,
        drives["total_line"] / 2 - drives["spread_line"] / 2,
    )
    if bin_edges is None:
        bin_edges = fit_implied_total_bin_edges(drives["posteam_implied_total"])
    drives["implied_total_quantile"] = pd.cut(
        drives["posteam_implied_total"], bins=bin_edges, labels=False, include_lowest=True
    )
    return drives, bin_edges


class DriveOutcomeSampler:
    """Bootstrap-resample a real historical drive matching (start field
    bucket, market-implied-total quantile for the possessing team), falling
    back to the field-position bucket alone when a specific combination has
    too few real drives to sample from -- rare buckets shouldn't silently
    resample from an empty or tiny, high-variance pool.

    v2 (shared-environment correlation): validate_game_simulator.py found
    TOTAL predictions lagged margin/win-probability significantly (MAE 11.63
    vs the current model's 10.20) after v1 shipped -- each side's score was
    simulated fully independently, missing the real game-level correlation
    between the two teams' scoring (weather, pace, game script, a fast-paced
    shootout vs. a low-scoring grind affect BOTH offenses together, not one
    at a time). `sample()` now accepts an optional `shared_percentile` (drawn
    ONCE per simulated trial, in [0,1], and passed to every drive draw for
    BOTH teams that trial): pools are pre-sorted by offense_points, and the
    drawn index is a weighted mix of that shared percentile and independent
    per-drive randomness (CORRELATION_WEIGHT controls the mix) -- a
    high-percentile trial nudges every drive, for both sides, toward the
    higher-scoring end of its own (start bucket, implied-total quantile)
    pool, correlating the two teams' totals within a trial while still
    preserving real per-drive variance from the independent-randomness share."""

    MIN_POOL_SIZE = 40
    CORRELATION_WEIGHT = 0.45
    EOH_MIN_POOL_SIZE = 20

    def __init__(self, drives: pd.DataFrame, seed: int = 0, end_of_half_drives: pd.DataFrame | None = None):
        self.rng = np.random.default_rng(seed)
        self.drives = drives.dropna(subset=["start_bucket", "implied_total_quantile"])
        self._pools: dict[tuple, pd.DataFrame] = {}
        for key, grp in self.drives.groupby(["start_bucket", "implied_total_quantile"]):
            self._pools[key] = grp.sort_values("offense_points").reset_index(drop=True)
        self._bucket_only_pools = {
            b: g.sort_values("offense_points").reset_index(drop=True) for b, g in self.drives.groupby("start_bucket")
        }
        self._all_sorted = self.drives.sort_values("offense_points").reset_index(drop=True)

        # End-of-half outcome pool (review round 4, #6): kept SEPARATE from the main pools
        # above, which exclude these rows for their DURATION (a real artifact -- see
        # game_simulator.py's build_simulator_for_season_range docstring). Their OUTCOME
        # (nearly always zero points) is a real, common NFL event -- a drive that gets cut
        # off by the clock -- and belongs back in the model as what actually happens at the
        # half boundary, via sample_end_of_half() below, not thrown away entirely.
        eoh = end_of_half_drives if end_of_half_drives is not None else drives.iloc[0:0]
        self._eoh_bucket_pools = {
            b: g.reset_index(drop=True) for b, g in eoh.groupby("start_bucket") if len(g) >= self.EOH_MIN_POOL_SIZE
        }
        self._eoh_all = eoh.reset_index(drop=True) if len(eoh) > 0 else eoh

    def sample(self, start_bucket: int, implied_total_quantile: int, shared_percentile: float | None = None):
        key = (start_bucket, implied_total_quantile)
        pool = self._pools.get(key)
        if pool is None or len(pool) < self.MIN_POOL_SIZE:
            # fall back: same bucket, ignore quantile (still real, still
            # conditioned on field position, just not matchup-quality-aware)
            pool = self._bucket_only_pools.get(start_bucket)
        if pool is None or len(pool) == 0:
            pool = self._all_sorted  # last-resort: any real drive at all

        if shared_percentile is None:
            idx = self.rng.integers(0, len(pool))
        else:
            u = self.CORRELATION_WEIGHT * shared_percentile + (1 - self.CORRELATION_WEIGHT) * self.rng.random()
            idx = min(int(u * len(pool)), len(pool) - 1)
        return pool.iloc[idx]

    def sample_end_of_half(self, start_bucket: int):
        """A real historical drive that was itself cut off by the clock, matching
        start-field-position bucket where there's enough real data, falling back to the
        overall end-of-half pool otherwise. Used when a normally-sampled drive's duration
        (from the clean, non-truncated pool via sample() above) wouldn't fit in the
        remaining half -- the real generating process is "this drive got cut short," not
        "this drive completed normally then the half happened to end," so its outcome
        should come from real cut-short drives specifically, not the general pool."""
        pool = self._eoh_bucket_pools.get(start_bucket)
        if pool is None or len(pool) == 0:
            pool = self._eoh_all
        idx = self.rng.integers(0, len(pool))
        return pool.iloc[idx]


class NextDrivePositionSampler:
    """Bootstrap-resample a real next-drive starting field position,
    conditioned on how the prior drive ended (result category) -- avoids
    modeling punt distance/return yards or kickoff mechanics explicitly."""

    def __init__(self, drives: pd.DataFrame, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        valid = drives.dropna(subset=["next_start_yardline_100", "result"])
        self._pools = {r: g["next_start_yardline_100"].to_numpy() for r, g in valid.groupby("result")}
        self._all = valid["next_start_yardline_100"].to_numpy()

    def sample(self, result: str) -> float:
        pool = self._pools.get(result)
        if pool is None or len(pool) < 20:
            pool = self._all
        return float(pool[self.rng.integers(0, len(pool))])


def build_opening_position_pool(drives: pd.DataFrame) -> np.ndarray:
    """Real starting field positions for the very first drive of a half
    (post-opening-kickoff or post-halftime-kickoff) -- used to seed each
    simulated half realistically (touchbacks, returns, occasional onside
    kicks all show up for free since these are real historical openers)."""
    openers = drives[drives.groupby("game_id").cumcount() == 0]
    return openers["start_yardline_100"].dropna().to_numpy()
