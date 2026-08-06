"""Review round 2, #3.3: §9.6's original props-simulator validation reported correlation
(saturated near 1, 0.999x) and Brier (deltas ~0.002) -- metrics that can't discriminate a
decision-relevant improvement, the same saturation failure mode §6.4 already documented for
population-level calibration. This adds three metrics that can: log-loss (punishes tail
overconfidence harder than Brier), tail calibration specifically at the top 5% of predicted
P(targets>=7), and an explicit vs-the-vig framing (~4-5% typical two-way hold at standard
pricing) stating plainly whether the improvement is large enough to matter for betting.

Measurement only -- no pipeline wiring change. Walk-forward: residual pools + share engine
fit on TRAIN=2018-2021 only, scored on TEST=2022-2025.

**2026-08 addition**: also compares TouchCountSimulator's default (raw, non-demeaned
bootstrap) against its `demean=True` option -- an external review flagged the raw bootstrap's
real, non-uniform per-tier mean bias (fringe +0.67 SD, role_player +0.26 SD, primary ~0.01 SD
on TRAIN) as reinjecting an unintended bias into every simulated trial. Tested directly rather
than assumed: see MODEL_DOCUMENTATION.md §9.6.1 for the result and why it's the OPPOSITE of
what that framing predicted.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.models.player_usage import ShareEngine, build_player_week_shares
from src.models.props_simulator import TouchCountSimulator, build_standardized_residuals
from src.utils.paths import DATA_RAW

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
THRESHOLDS = [3, 5, 7]
N_TRIALS = 1000


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    shares = build_player_week_shares(weekly)

    tgt_engine = ShareEngine()
    tgt_result = tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")

    train_result = tgt_result[tgt_result["season"].isin(TRAIN_SEASONS)]
    resid_table = build_standardized_residuals(train_result, "target_share_calc", "team_targets", "targets")
    print("per-tier residual mean (TRAIN) -- the bias the demeaning comparison below tests:")
    print(resid_table.groupby("usage_tier")["standardized_resid"].agg(["mean", "std", "count"]).round(3))
    simulator = TouchCountSimulator(resid_table, seed=0)
    simulator_demeaned = TouchCountSimulator(resid_table, seed=0, demean=True)

    test = tgt_result[tgt_result["season"].isin(TEST_SEASONS)].copy()
    test = test[(test["pregame_target_share_calc"] > 0.03) & (test["team_targets"] > 0)]
    print(f"TEST=2022-2025, n={len(test)} meaningfully-involved player-weeks\n")

    print("=== R3.10: log-loss, Brier, and correlation at each threshold ===\n")
    diffs_by_threshold = {}
    for thresh in THRESHOLDS:
        p_naive = 1 - stats.binom.cdf(thresh - 1, test["team_targets"], test["pregame_target_share_calc"])

        sim_probs = np.empty(len(test))
        sim_probs_demeaned = np.empty(len(test))
        for i, row in enumerate(test.itertuples()):
            sims = simulator.simulate(row.pregame_target_share_calc, row.team_targets, n_trials=N_TRIALS)
            sims_dm = simulator_demeaned.simulate(row.pregame_target_share_calc, row.team_targets, n_trials=N_TRIALS)
            sim_probs[i] = (sims >= thresh).mean()
            sim_probs_demeaned[i] = (sims_dm >= thresh).mean()

        y = (test["targets"] >= thresh).to_numpy().astype(float)
        corr_naive = np.corrcoef(p_naive, y)[0, 1]
        corr_sim = np.corrcoef(sim_probs, y)[0, 1]
        corr_dm = np.corrcoef(sim_probs_demeaned, y)[0, 1]
        print(f"--- threshold >= {thresh} targets ---")
        print(f"  naive Binomial:        corr={corr_naive:.4f}  Brier={brier(p_naive, y):.5f}  log-loss={log_loss(p_naive, y):.5f}")
        print(f"  bootstrap (raw, ship): corr={corr_sim:.4f}  Brier={brier(sim_probs, y):.5f}  log-loss={log_loss(sim_probs, y):.5f}")
        print(f"  bootstrap (demeaned):  corr={corr_dm:.4f}  Brier={brier(sim_probs_demeaned, y):.5f}  log-loss={log_loss(sim_probs_demeaned, y):.5f}")
        diffs_by_threshold[thresh] = np.abs(sim_probs - p_naive)
        print(f"  |bootstrap - naive| probability gap: mean={diffs_by_threshold[thresh].mean():.4f}  median={np.median(diffs_by_threshold[thresh]):.4f}\n")

        if thresh == 7:
            print("=== R3.10: tail calibration, top 5% of predicted P(targets>=7) ===")
            for label, p in [("naive", p_naive), ("bootstrap raw", sim_probs), ("bootstrap demeaned", sim_probs_demeaned)]:
                cutoff = np.quantile(p, 0.95)
                top = y[p >= cutoff]
                print(f"  {label}: predicted-mean in top 5%={p[p >= cutoff].mean():.3f}  realized rate={top.mean():.3f}  n={len(top)}")
            print()

    print("=== R3.10: vs-the-vig framing ===")
    print("A standard two-way prop line at -110/-110 holds ~4.55% of the total stake -- the")
    print("bootstrap-vs-naive probability gap at the most decision-relevant threshold (>=7")
    print("targets) is the direct measure of how differently the two approaches would price a")
    print("real bet, in the same units as that hold:")
    gap7 = diffs_by_threshold[7]
    print(f"  mean |gap|={gap7.mean():.4f} ({gap7.mean()*100:.2f} points of probability), "
          f"median={np.median(gap7):.4f}, 95th pct={np.quantile(gap7, 0.95):.4f}")
    print("(Interpretation belongs in MODEL_DOCUMENTATION.md §9.6, not hardcoded here -- this")
    print("script reports the numbers, not a pre-committed verdict on what they mean.)")
