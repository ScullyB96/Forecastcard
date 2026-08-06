"""Review round 2, #2.1/#2.2: since the margin market blend was removed (§4.3), the
game-side margin model's ENTIRE live betting edge lives in the QB-swap + symmetric-CB
adjustment layer applied on top of the market spread_line -- but that layer has only ever
been graded on MAE/signed bias on its flagged subset (§6.1.1, §5.1), never the full
ATS%-with-bootstrap-CI panel that actually decided the market blend's fate. The CB
coefficient specifically (the sole survivor of a wide search: top-2/3/4/5 corners,
Out/Doubtful/Questionable, several position candidates, three residual bases) has also
never been run through rolling-origin CV the way the market blend was (§10.5). This script
closes both gaps: a fixed-split ATS/CRPS/bias panel on the CB-flagged and QB-swap-flagged
subsets, plus a rolling-origin refit of the CB coefficient with a snap-rank discriminator.

Imports SWAP_B_MARKET directly from weekly_update.py (not a second hardcoded copy) so this
validation can never silently drift from the live production value -- exactly the
stale-coupling bug class review round 2 was about (#1.1, #4.2).

Extended for review round 3, #1: the original CB-flagged ATS panel below is RESUBSTITUTION,
not out-of-sample -- JOINT_COEFS_FORWARD's coefficient is fit on pooled 2018-2025 data,
which includes the TEST=2022-2025 window the panel scores against. Added a walk-forward
(per-fold, strictly-prior-seasons-only) version of the same panel, plus a permutation test
on the full fit-and-score procedure (shuffle each team's real CB-out flags across its own
played weeks within season, preserving its real annual flag count, re-run the whole
pooled-fit-then-score procedure 1000x, and locate the real number in the resulting null).
See MODEL_DOCUMENTATION.md §6.1.1 for the full writeup and the resulting decision to shrink
JOINT_COEFS_FORWARD's CB coefficient toward the honest fold median.
"""

import numpy as np
import pandas as pd

from src.models.injury_adjustment import (
    JOINT_COEFS_FORWARD,
    STATUS_OUT,
    apply_joint_adjustment,
    build_presumed_starters_by_name,
    compute_injury_flags,
    _latest_season_range_file,
)
from src.models.qb_adjustment import QbRatingEngine, build_qb_week_table, build_starter_sequence
from src.models.scoring import ats_win_rate, crps_gaussian, fit_residual_sigma, signed_bias
from src.pipeline.weekly_update import SWAP_B_MARKET
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.rolling_origin_cv import sign_consistency
from src.utils.stats import fit_linear

TRAIN_SEASONS = {2018, 2019, 2020, 2021}
TEST_SEASONS = {2022, 2023, 2024, 2025}
ALL_SEASONS = sorted(TRAIN_SEASONS | TEST_SEASONS)


def build_net_swap_delta(schedules: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """Per-game [game_id, net_swap_delta] -- the general in-season swap-delta signal
    (§5.1's QbRatingEngine + build_starter_sequence), same construction qb_adjustment.py's
    own validation demo uses, NOT weekly_update.py's narrow offseason-bootstrap-only wiring
    (that path only ever fires for the one 2026-week-1 Clay bootstrap; this retrospective
    check needs the general mechanism §5.1/§6.1.1 was actually validated against)."""
    qb_weeks = build_qb_week_table(weekly)
    starters = build_starter_sequence(schedules)
    engine = QbRatingEngine()
    starters = engine.run_with_starters(starters, qb_weeks)
    starters["swap_delta"] = np.where(
        starters["qb_changed"], starters["pregame_qb_rating"] - starters["prev_qb_rating"], 0.0
    )
    starters["swap_delta"] = starters["swap_delta"].fillna(0.0)

    home = starters.rename(columns={"team": "home_team_x", "swap_delta": "home_swap_delta"})[
        ["game_id", "home_team_x", "home_swap_delta"]
    ]
    away = starters.rename(columns={"team": "away_team_x", "swap_delta": "away_swap_delta"})[
        ["game_id", "away_team_x", "away_swap_delta"]
    ]
    sched = schedules[schedules["game_type"] == "REG"][["game_id", "home_team", "away_team"]]
    merged = sched.merge(home, left_on=["game_id", "home_team"], right_on=["game_id", "home_team_x"], how="left")
    merged = merged.merge(away, left_on=["game_id", "away_team"], right_on=["game_id", "away_team_x"], how="left")
    merged["home_swap_delta"] = merged["home_swap_delta"].fillna(0.0)
    merged["away_swap_delta"] = merged["away_swap_delta"].fillna(0.0)
    merged["net_swap_delta"] = merged["home_swap_delta"] - merged["away_swap_delta"]
    return merged[["game_id", "net_swap_delta"]]


def load_cb_search_inputs(seasons_dir=DATA_RAW):
    """Loads the two real, unshuffled tables the whole corner-count x designation search
    (review round 4, #1) is built from: injury designations (season, week, team, name_norm,
    report_status) and snap-share-ranked CB pools for every corner-count 2-5
    (top_cb_by_n[n] = build_presumed_starters_by_name(..., n, ...)). Precomputed ONCE and
    reused across every permutation and every grid config -- only WHICH player-weeks count
    as "flagged" changes per shuffle/config, not these underlying real pools."""
    from src.ingest.name_matching import norm_name

    inj = pd.read_parquet(_latest_season_range_file(seasons_dir, "injuries"))
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["name_norm"] = inj["full_name"].apply(norm_name)

    sc = pd.read_parquet(_latest_season_range_file(seasons_dir, "snap_counts"))
    sc["name_norm"] = sc["player"].apply(norm_name)
    sc["season"] = sc["season"].astype(int)
    sc["week"] = sc["week"].astype(int)
    top_cb_by_n = {n: build_presumed_starters_by_name(sc, "CB", n, "defense_pct") for n in (2, 3, 4, 5)}
    return inj, top_cb_by_n


def build_cb_flags_for_config(inj: pd.DataFrame, top_cb_by_n: dict, n_corners: int, designations: set) -> pd.DataFrame:
    """Per (season, week, team): cb_out (count) + out_min_rank, for a SPECIFIC
    (corner-count, designation-set) configuration -- generalizes the original top-3/Out-only
    construction so review round 4 #1's grid search can sweep both axes cheaply from the
    precomputed inputs above."""
    top_cb = top_cb_by_n[n_corners]
    out = inj[inj["report_status"].isin(designations)][["season", "week", "team", "name_norm"]].drop_duplicates()
    out["is_out"] = True
    merged = top_cb.merge(out, on=["season", "week", "team", "name_norm"], how="left")
    merged["is_out"] = merged["is_out"].fillna(False)
    cb_out = merged.groupby(["season", "week", "team"])["is_out"].sum().reset_index(name="cb_out")
    out_ranked = merged[merged["is_out"]].groupby(["season", "week", "team"])["rank"].min().reset_index(name="out_min_rank")
    return cb_out.merge(out_ranked, on=["season", "week", "team"], how="left")


def build_cb_flags_with_rank(seasons_dir=DATA_RAW) -> pd.DataFrame:
    """The project's current, already-decided configuration (top-3 corners, "Out" only) --
    thin wrapper around the generalized builder above, kept so existing callers/output don't
    change."""
    inj, top_cb_by_n = load_cb_search_inputs(seasons_dir)
    return build_cb_flags_for_config(inj, top_cb_by_n, 3, {STATUS_OUT})


def fit_symmetric_cb_tstat(train: pd.DataFrame) -> tuple[float, float, float]:
    """Fits residual ~ (away_cb_flag - home_cb_flag) and returns (intercept, slope, |t-stat
    of slope|) -- the selection-criterion value used by the grid search (review round 4,
    #1): a single, uniformly-computable summary of "how confident is this configuration that
    the effect is real," in the same spirit as the t-stats historically used to choose top-3
    over top-2/4/5 corners and the symmetric constraint over independent home/away effects."""
    x = (train["away_cb_flag"] - train["home_cb_flag"]).to_numpy(dtype=float)
    y = train["resid"].to_numpy(dtype=float)
    a, b = fit_linear(pd.Series(x), pd.Series(y))
    resid = y - (a + b * x)
    n = len(x)
    if n <= 2 or np.allclose(x, x[0]):
        return a, b, 0.0
    sigma2 = (resid ** 2).sum() / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se_b = np.sqrt(sigma2 / sxx) if sxx > 0 else np.inf
    tstat = abs(b / se_b) if se_b > 0 else 0.0
    return a, b, tstat


def build_production_margin(layer1: pd.DataFrame, flags: pd.DataFrame, net_swap: pd.DataFrame) -> pd.DataFrame:
    """layer1 must already have [game_id, season, week, team_a(home), team_b(away),
    spread_line, actual_margin]. Returns layer1 with market_margin/production_margin/
    home_cb_flag/away_cb_flag/net_swap_delta columns added."""
    df = layer1.merge(net_swap, on="game_id", how="left")
    df["net_swap_delta"] = df["net_swap_delta"].fillna(0.0)

    adj = apply_joint_adjustment(df, flags, coefs=JOINT_COEFS_FORWARD)
    m = df.merge(flags.rename(columns={"team": "team_a", "cb_out": "cb_out_home"})[["season", "week", "team_a", "cb_out_home"]],
                 on=["season", "week", "team_a"], how="left")
    m = m.merge(flags.rename(columns={"team": "team_b", "cb_out": "cb_out_away"})[["season", "week", "team_b", "cb_out_away"]],
                on=["season", "week", "team_b"], how="left")
    df["home_cb_flag"] = (m["cb_out_home"].fillna(0) >= 1).astype(int)
    df["away_cb_flag"] = (m["cb_out_away"].fillna(0) >= 1).astype(int)

    df["market_margin"] = df["spread_line"]
    df["production_margin"] = df["market_margin"] + SWAP_B_MARKET * df["net_swap_delta"] + adj
    return df


if __name__ == "__main__":
    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")

    sched_lines = schedules[schedules["game_type"] == "REG"][["game_id", "spread_line"]]
    layer1 = layer1.merge(sched_lines, on="game_id", how="left")
    layer1 = layer1.dropna(subset=["spread_line"])

    flags = build_cb_flags_with_rank()
    net_swap = build_net_swap_delta(schedules, weekly)
    df = build_production_margin(layer1, flags, net_swap)

    test = df[df["season"].isin(TEST_SEASONS)].copy()
    sigma = fit_residual_sigma(test["production_margin"], test["actual_margin"])
    print(f"TEST-wide residual sigma (for CRPS on subsets below): {sigma:.3f}\n")

    print("=== R1.3: full ATS panel, CB-flagged subset ===")
    cb_flagged = test[(test["home_cb_flag"] == 1) | (test["away_cb_flag"] == 1)]
    ats = ats_win_rate(cb_flagged["production_margin"], cb_flagged["market_margin"], cb_flagged["actual_margin"])
    bias = signed_bias(cb_flagged["production_margin"], cb_flagged["actual_margin"])
    crps = crps_gaussian(cb_flagged["production_margin"], cb_flagged["actual_margin"], sigma)
    print(f"n={ats['n']} (+{ats['n_push']} push)  ATS%={ats['win_rate']:.4f}  CI=[{ats['ci'][0]:.3f},{ats['ci'][1]:.3f}]")
    print(f"signed bias={bias['bias']:+.3f}  CI=[{bias['ci'][0]:+.3f},{bias['ci'][1]:+.3f}]")
    print(f"CRPS={crps['crps']:.3f}")

    print("\n=== R1.3: full ATS panel, QB-swap-flagged subset ===")
    swap_flagged = test[test["net_swap_delta"] != 0.0]
    ats2 = ats_win_rate(swap_flagged["production_margin"], swap_flagged["market_margin"], swap_flagged["actual_margin"])
    bias2 = signed_bias(swap_flagged["production_margin"], swap_flagged["actual_margin"])
    crps2 = crps_gaussian(swap_flagged["production_margin"], swap_flagged["actual_margin"], sigma)
    print(f"n={ats2['n']} (+{ats2['n_push']} push)  ATS%={ats2['win_rate']:.4f}  CI=[{ats2['ci'][0]:.3f},{ats2['ci'][1]:.3f}]")
    print(f"signed bias={bias2['bias']:+.3f}  CI=[{bias2['ci'][0]:+.3f},{bias2['ci'][1]:+.3f}]")
    print(f"CRPS={crps2['crps']:.3f}")

    print("\n=== R1.4: rolling-origin CV on the symmetric CB coefficient ===")
    rows = []
    fold_scored_rows = []  # accumulates each fold's OWN test-season rows, scored with THAT fold's coefficient
    for i in range(4, len(ALL_SEASONS)):
        train_seasons = set(ALL_SEASONS[:i])
        test_season = ALL_SEASONS[i]
        train = df[df["season"].isin(train_seasons)].copy()
        base_a, base_b = fit_linear(train["market_margin"], train["actual_margin"])
        # symmetric CB regression: residual ~ (away_cb_flag - home_cb_flag), on the market-margin residual
        train["resid"] = train["actual_margin"] - train["market_margin"]
        cb_diff = train["away_cb_flag"] - train["home_cb_flag"]
        cb_a, cb_b = fit_linear(cb_diff, train["resid"])
        rows.append({"train_seasons": f"{min(train_seasons)}-{max(train_seasons)}", "test_season": test_season, "cb_coef": cb_b})

        fold_test = df[df["season"] == test_season].copy()
        fold_cb_diff = fold_test["away_cb_flag"] - fold_test["home_cb_flag"]
        fold_test["fold_production_margin"] = (
            fold_test["market_margin"] + SWAP_B_MARKET * fold_test["net_swap_delta"] + cb_a + cb_b * fold_cb_diff
        )
        fold_scored_rows.append(fold_test)
    fold_df = pd.DataFrame(rows)
    print(fold_df.round(3).to_string(index=False))
    print(f"\nsign consistency: {sign_consistency(fold_df, 'cb_coef'):.3f}")
    print(f"magnitude range: [{fold_df['cb_coef'].min():.3f}, {fold_df['cb_coef'].max():.3f}]  median={fold_df['cb_coef'].median():.3f}")

    print("\n=== S2.1 (review round 3, #1): WALK-FORWARD (honest, out-of-sample) ATS on the CB-flagged subset ===")
    print("The panel above used JOINT_COEFS_FORWARD's pooled 2018-2025 coefficient (2.977) to score")
    print("TEST=2022-2025 -- but that coefficient was FIT ON data including the very seasons being")
    print("scored (resubstitution), not an honest out-of-sample test. This uses ONLY each fold's own")
    print("strictly-prior-seasons coefficient (from the rolling-origin loop above) to score that")
    print("fold's own test season, pooled across all 4 folds -- the number that corresponds to what")
    print("could actually have been bet.")
    walk_forward = pd.concat(fold_scored_rows, ignore_index=True)
    wf_cb_flagged = walk_forward[(walk_forward["home_cb_flag"] == 1) | (walk_forward["away_cb_flag"] == 1)]
    wf_ats = ats_win_rate(wf_cb_flagged["fold_production_margin"], wf_cb_flagged["market_margin"], wf_cb_flagged["actual_margin"])
    wf_bias = signed_bias(wf_cb_flagged["fold_production_margin"], wf_cb_flagged["actual_margin"])
    print(f"n={wf_ats['n']} (+{wf_ats['n_push']} push)  ATS%={wf_ats['win_rate']:.4f}  CI=[{wf_ats['ci'][0]:.3f},{wf_ats['ci'][1]:.3f}]")
    print(f"signed bias={wf_bias['bias']:+.3f}  CI=[{wf_bias['ci'][0]:+.3f},{wf_bias['ci'][1]:+.3f}]")
    print(f"(resubstitution ATS% from the panel above, for direct comparison: {ats['win_rate']:.4f})")
    print("CORRECTION (review round 4, #1): ATS is a sign-of-disagreement metric -- for a")
    print("single-signed adjustment on top of the market line it is entirely magnitude-")
    print("independent, and every fold coefficient above is positive, so the bet direction is")
    print("identical in every fold. The walk-forward and resubstitution ATS numbers are very")
    print("nearly the SAME statistic (the raw win rate of 'bet against the team missing a")
    print("top-3 corner'), not independent corroboration that the in-sample coefficient was")
    print("sound -- their closeness is evidence the metric is blind to the contamination, not")
    print("evidence the contamination didn't matter. The permutation test below is the only")
    print("check that can actually speak to this.")

    print("\n=== S2.2 (review round 3, #1): permutation test on the FULL fit-and-score procedure ===")
    print("Null hypothesis: CB-out flags carry no real information. Shuffle each team's real CB-out")
    print("flags across its own played weeks WITHIN season (preserving that team's real annual flag")
    print("count exactly), re-run the SAME resubstitution fit-and-score procedure that produced the")
    print("original 62.4% (pooled 2018-2025 fit, scored on TEST=2022-2025), 1000 times, and see where")
    print("the real number falls in the resulting null distribution.")
    rng = np.random.default_rng(0)
    n_perm = 1000
    null_ats = np.empty(n_perm)
    flags_idx = flags.set_index(["season", "team"])
    for p in range(n_perm):
        shuffled_flags = flags.copy()
        for (season, team), grp in flags.groupby(["season", "team"]):
            shuffled_flags.loc[grp.index, "cb_out"] = rng.permutation(grp["cb_out"].to_numpy())
        perm_df = build_production_margin(layer1, shuffled_flags, net_swap)
        perm_test = perm_df[perm_df["season"].isin(TEST_SEASONS)]
        perm_cb_flagged = perm_test[(perm_test["home_cb_flag"] == 1) | (perm_test["away_cb_flag"] == 1)]
        predicted_edge = perm_cb_flagged["production_margin"] - perm_cb_flagged["market_margin"]
        actual_edge = perm_cb_flagged["actual_margin"] - perm_cb_flagged["market_margin"]
        decided = ~np.isclose(actual_edge, 0.0)
        null_ats[p] = (np.sign(predicted_edge[decided]) == np.sign(actual_edge[decided])).mean() if decided.sum() else np.nan
    valid_null = null_ats[~np.isnan(null_ats)]
    percentile = (valid_null < ats["win_rate"]).mean()
    print(f"null distribution (n={len(valid_null)} permutations): mean={valid_null.mean():.4f}  std={valid_null.std():.4f}  "
          f"95th pct={np.quantile(valid_null, 0.95):.4f}  99th pct={np.quantile(valid_null, 0.99):.4f}")
    print(f"real resubstitution ATS%={ats['win_rate']:.4f}  ->  percentile in null distribution: {percentile*100:.1f}")
    print(f"real walk-forward ATS%={wf_ats['win_rate']:.4f}  ->  percentile in null distribution: {(valid_null < wf_ats['win_rate']).mean()*100:.1f}")
    print("CAVEAT (review round 4, #1): this null only prices in 'fit and scored on the same")
    print("games' (resubstitution). It does NOT price in the SPECIFICATION SEARCH that chose")
    print("this exact configuration (top-3 corners, Out-only, symmetric) out of the family")
    print("actually explored historically. Back-of-envelope: the expected MAXIMUM of n draws")
    print("from this null (mean %.1f, std %.1f) is roughly %.1f%% at n=10 and %.1f%% at n=15 --" %
          (valid_null.mean()*100, valid_null.std()*100, (valid_null.mean()+1.54*valid_null.std())*100, (valid_null.mean()+1.74*valid_null.std())*100))
    print("close to the observed 61.8-62.4%. T2.1 below extends the permutation to re-run the")
    print("actual specification search per shuffle, which is the correct reference.")

    print("\n=== T2.1 (review round 4, #1): permutation test extended to the FULL specification search ===")
    print("For each shuffle, re-run a grid search over corner-count x designation-set (12")
    print("configs -- a faithful, cheaply-reproducible core of the real historical search; the")
    print("symmetric-vs-independent choice is left fixed at symmetric, since that was a one-time")
    print("architectural decision already locked in, not an ongoing sweep like corner-count and")
    print("designation are), select the winner by |t-stat| (the same kind of criterion that")
    print("originally chose top-3-over-top-2/4/5 and symmetric-over-independent), then score")
    print("THAT shuffle's own winning config's ATS. This is the null of 'the best result the")
    print("whole procedure finds on noise' -- the correct reference for a number that survived")
    print("the whole procedure, not just one fit.")
    inj, top_cb_by_n = load_cb_search_inputs()
    designation_grid = [{STATUS_OUT}, {STATUS_OUT, "Doubtful"}, {STATUS_OUT, "Doubtful", "Questionable"}]
    corner_grid = [2, 3, 4, 5]
    config_grid = [(n, d) for n in corner_grid for d in designation_grid]
    print(f"grid size: {len(config_grid)} configs x N permutations")

    def flags_for_config_shuffled(n_corners, designations, rng):
        cfg_flags = build_cb_flags_for_config(inj, top_cb_by_n, n_corners, designations)
        shuffled = cfg_flags.copy()
        for (season, team), grp in cfg_flags.groupby(["season", "team"]):
            shuffled.loc[grp.index, "cb_out"] = rng.permutation(grp["cb_out"].to_numpy())
        return shuffled

    def score_config(cfg_flags):
        """Fits the symmetric coefficient on pooled 2018-2025 (matching the original
        resubstitution procedure), returns (|t-stat|, ats_win_rate) using that same
        pooled-fit coefficient to score TEST -- exactly how the real 62.4% was produced."""
        d = layer1.merge(net_swap, on="game_id", how="left")
        d["net_swap_delta"] = d["net_swap_delta"].fillna(0.0)
        m = d.merge(cfg_flags.rename(columns={"team": "team_a", "cb_out": "cb_out_home"})[["season", "week", "team_a", "cb_out_home"]], on=["season", "week", "team_a"], how="left")
        m = m.merge(cfg_flags.rename(columns={"team": "team_b", "cb_out": "cb_out_away"})[["season", "week", "team_b", "cb_out_away"]], on=["season", "week", "team_b"], how="left")
        d["home_cb_flag"] = (m["cb_out_home"].fillna(0) >= 1).astype(int)
        d["away_cb_flag"] = (m["cb_out_away"].fillna(0) >= 1).astype(int)
        d["market_margin"] = d["spread_line"]
        d["resid"] = d["actual_margin"] - d["market_margin"]
        cb_a, cb_b, tstat = fit_symmetric_cb_tstat(d)

        test_d = d[d["season"].isin(TEST_SEASONS)]
        flagged = test_d[(test_d["home_cb_flag"] == 1) | (test_d["away_cb_flag"] == 1)]
        d_margin = flagged["market_margin"] + SWAP_B_MARKET * flagged["net_swap_delta"] + cb_a + cb_b * (flagged["away_cb_flag"] - flagged["home_cb_flag"])
        predicted_edge = d_margin - flagged["market_margin"]
        actual_edge = flagged["actual_margin"] - flagged["market_margin"]
        decided = ~np.isclose(actual_edge, 0.0)
        win_rate = (np.sign(predicted_edge[decided]) == np.sign(actual_edge[decided])).mean() if decided.sum() else np.nan
        return tstat, win_rate

    N_SPEC_PERM = 300
    rng2 = np.random.default_rng(1)
    spec_null_ats = np.empty(N_SPEC_PERM)
    for p in range(N_SPEC_PERM):
        results = [score_config(flags_for_config_shuffled(n, d, rng2)) for n, d in config_grid]
        best = max(results, key=lambda r: r[0])
        spec_null_ats[p] = best[1]
        if (p + 1) % 50 == 0:
            print(f"  {p+1}/{N_SPEC_PERM} spec-search permutations done...")
    valid_spec_null = spec_null_ats[~np.isnan(spec_null_ats)]
    print(f"\nfull-spec-search null (n={len(valid_spec_null)}): mean={valid_spec_null.mean():.4f}  std={valid_spec_null.std():.4f}  "
          f"95th pct={np.quantile(valid_spec_null, 0.95):.4f}  99th pct={np.quantile(valid_spec_null, 0.99):.4f}")
    print(f"real walk-forward ATS%={wf_ats['win_rate']:.4f}  ->  percentile in FULL-SPEC-SEARCH null: "
          f"{(valid_spec_null < wf_ats['win_rate']).mean()*100:.1f}")

    print("\n=== T2.2 (review round 4, #2): why does the single-config null center at %.1f%% not 50%%? ===" % (valid_null.mean()*100))
    print("Re-running the ORIGINAL single-config permutation with WALK-FORWARD (not resubstitution)")
    print("fitting: if the off-center null was purely the resubstitution advantage, this should")
    print("move noticeably closer to 50%.")
    N_WF_NULL = 300
    rng3 = np.random.default_rng(2)
    wf_null = np.empty(N_WF_NULL)
    for p in range(N_WF_NULL):
        shuffled_flags = flags.copy()
        for (season, team), grp in flags.groupby(["season", "team"]):
            shuffled_flags.loc[grp.index, "cb_out"] = rng3.permutation(grp["cb_out"].to_numpy())
        perm_df = build_production_margin(layer1, shuffled_flags, net_swap)
        fold_rows_p = []
        for i in range(4, len(ALL_SEASONS)):
            train_seasons_p = set(ALL_SEASONS[:i]); test_season_p = ALL_SEASONS[i]
            train_p = perm_df[perm_df["season"].isin(train_seasons_p)].copy()
            train_p["resid"] = train_p["actual_margin"] - train_p["market_margin"]
            cb_diff_p = train_p["away_cb_flag"] - train_p["home_cb_flag"]
            cb_a_p, cb_b_p = fit_linear(cb_diff_p, train_p["resid"])
            fold_test_p = perm_df[perm_df["season"] == test_season_p].copy()
            fold_cb_diff_p = fold_test_p["away_cb_flag"] - fold_test_p["home_cb_flag"]
            fold_test_p["fold_production_margin"] = fold_test_p["market_margin"] + SWAP_B_MARKET * fold_test_p["net_swap_delta"] + cb_a_p + cb_b_p * fold_cb_diff_p
            fold_rows_p.append(fold_test_p)
        wf_p = pd.concat(fold_rows_p, ignore_index=True)
        wf_p_flagged = wf_p[(wf_p["home_cb_flag"] == 1) | (wf_p["away_cb_flag"] == 1)]
        pred_edge_p = wf_p_flagged["fold_production_margin"] - wf_p_flagged["market_margin"]
        act_edge_p = wf_p_flagged["actual_margin"] - wf_p_flagged["market_margin"]
        decided_p = ~np.isclose(act_edge_p, 0.0)
        wf_null[p] = (np.sign(pred_edge_p[decided_p]) == np.sign(act_edge_p[decided_p])).mean() if decided_p.sum() else np.nan
    valid_wf_null = wf_null[~np.isnan(wf_null)]
    print(f"walk-forward-fitted null (n={len(valid_wf_null)}): mean={valid_wf_null.mean():.4f}  std={valid_wf_null.std():.4f}")
    print(f"resubstitution-fitted null (from above, for comparison): mean={valid_null.mean():.4f}  std={valid_null.std():.4f}")
    print("If walk-forward's null mean is materially closer to 0.50, the resubstitution-advantage")
    print("explanation is confirmed (the null is behaving correctly). If a real gap from 0.50")
    print("remains, that's a residual asymmetry worth investigating (candidates: home/away")
    print("flag-incidence imbalance interacting with the bet rule, or a scoring-direction issue).")

    print("\n=== R1.4: snap-rank discriminator (CB-flagged games only, TEST) ===")
    ranked = cb_flagged.merge(
        flags.rename(columns={"team": "team_a"})[["season", "week", "team_a", "out_min_rank"]].rename(columns={"out_min_rank": "rank_home"}),
        on=["season", "week", "team_a"], how="left",
    )
    ranked = ranked.merge(
        flags.rename(columns={"team": "team_b"})[["season", "week", "team_b", "out_min_rank"]].rename(columns={"out_min_rank": "rank_away"}),
        on=["season", "week", "team_b"], how="left",
    )
    ranked["flagged_rank"] = ranked[["rank_home", "rank_away"]].min(axis=1)
    for rk in sorted(ranked["flagged_rank"].dropna().unique()):
        sub = ranked[ranked["flagged_rank"] == rk]
        b = signed_bias(sub["production_margin"], sub["actual_margin"])
        mae = (sub["production_margin"] - sub["actual_margin"]).abs().mean()
        print(f"rank={int(rk)} (CB{int(rk)+1}): n={len(sub)}  MAE={mae:.3f}  signed_bias={b['bias']:+.3f}")

    print("\n=== T3.1 (review round 4, #3): signed bias on the CB-flagged holdout, +-2.977 vs +-2.446 ===")
    print("NOTE: this uses the CURRENT production formula (market line + QB-swap + CB term) as the")
    print("base, not the historical Layer-1/blend-residual-basis methodology that originally produced")
    print("v2->v3's +1.596->+0.542 bias figures (that base no longer exists in production -- margin's")
    print("market blend was removed entirely). This directly answers what the shrink costs TODAY.")
    for coef_val, label in [(2.977, "v3 (+-2.977)"), (2.446, "v4, shrunk (+-2.446)")]:
        adj = (coef_val * (cb_flagged["away_cb_flag"] - cb_flagged["home_cb_flag"]))
        # rebuild margin at this coefficient: market_margin + swap adj (unchanged) + this coefficient's CB term
        # (the joint-model intercept/skill/OL terms are 0 in JOINT_COEFS_FORWARD v3/v4, so this isolates the CB term cleanly)
        base_no_cb = cb_flagged["market_margin"] + SWAP_B_MARKET * cb_flagged["net_swap_delta"] - 0.018
        margin_at_coef = base_no_cb + adj
        b = signed_bias(margin_at_coef, cb_flagged["actual_margin"])
        mae = (margin_at_coef - cb_flagged["actual_margin"]).abs().mean()
        print(f"{label:24s}  n={len(cb_flagged)}  MAE={mae:.3f}  signed_bias={b['bias']:+.3f}  CI=[{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]")
    print("(Documented as an explicit trade, not a one-sided precaution -- see MODEL_DOCUMENTATION.md")
    print(" §6.1.1's staking note: sizing should use the coefficient's own implied edge and CI,")
    print(" which this bias check informs, not the ATS number, which the shrink can't move at all.)")

    print("\n=== T3.2 (2026-08, follow-up on the literature-benchmark check): empirical MAE/CRPS/bias")
    print("    curve across candidate CB coefficient magnitudes, including the real-world-informed")
    print("    range (0 to ~1.5, per the Sports Insights oddsmaker survey -- MODEL_DOCUMENTATION.md")
    print("    §6.1.1) ===")
    print("ATS can't discriminate magnitude (every candidate implies the same bet direction) -- MAE/")
    print("CRPS/signed-bias can. This is the honest test of whether shrinking toward the literature-")
    print("plausible range costs real accuracy on our own held-out data, or is close to free.")
    for coef_val in [0.0, 0.5, 1.0, 1.5, 2.0, 2.446, 2.977]:
        adj = coef_val * (cb_flagged["away_cb_flag"] - cb_flagged["home_cb_flag"])
        base_no_cb = cb_flagged["market_margin"] + SWAP_B_MARKET * cb_flagged["net_swap_delta"] - 0.018
        margin_at_coef = base_no_cb + adj
        b = signed_bias(margin_at_coef, cb_flagged["actual_margin"])
        mae = (margin_at_coef - cb_flagged["actual_margin"]).abs().mean()
        crps_val = crps_gaussian(margin_at_coef, cb_flagged["actual_margin"], sigma)["crps"]
        label = {0.0: "0.0 (no CB term)", 0.5: "0.5 (lit: elite skill-position ceiling)",
                 1.5: "1.5 (lit: best-defender speculative ceiling)", 2.446: "2.446 (current production)",
                 2.977: "2.977 (pre-shrink v3)"}.get(coef_val, f"{coef_val}")
        print(f"  {label:42s}  MAE={mae:.3f}  CRPS={crps_val:.3f}  signed_bias={b['bias']:+.3f}  CI=[{b['ci'][0]:+.3f},{b['ci'][1]:+.3f}]")
    best_by_mae = min(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.446, 2.977],
        key=lambda c: (cb_flagged["market_margin"] + SWAP_B_MARKET * cb_flagged["net_swap_delta"] - 0.018
                       + c * (cb_flagged["away_cb_flag"] - cb_flagged["home_cb_flag"]) - cb_flagged["actual_margin"]).abs().mean(),
    )
    print(f"\nMAE-minimizing candidate among those tested: {best_by_mae}")
    print("(Interpretation belongs in MODEL_DOCUMENTATION.md §6.1.1, not hardcoded here.)")
