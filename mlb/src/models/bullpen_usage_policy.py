"""Task #144 step 1: the bullpen usage POLICY, fit from real data, with NO
simulator wiring yet (see MODEL_DOCUMENTATION.md sec 11.18 for the full
5-step spec this is step 1 of). Two components:

1. STARTER-HOOK model: P(starter exits | inning, PA count faced, cumulative
   runs allowed, score margin from the DEFENSE's own perspective). Validated
   on real, held-out 2025 data: mean real hook inning is a clean, monotonic
   function of margin (trailing big: 4.45, tied: 5.20, leading big: 6.30) --
   confirms margin is a genuine, measurable driver of real hook decisions.

2. RELIEVER-TIER model: P(tier appears | margin, inning, save situation),
   tiers = season-long usage-rank terciles within each team-bullpen (mopup/
   middle/leverage), closer identified separately via 9th+-inning appearance
   concentration. Validated even more cleanly: leverage-arm share by margin
   bucket is nearly IDENTICAL between the 2023-2024 fit sample and the 2025
   validation sample (e.g. leading: 82.3% fit vs 81.8% validation; trailing
   big: 53.1% vs 53.6%) -- a real, stable, reproducible pattern, not overfit
   noise. Closer usage in 9th-inning save situations: 60.6% (fit) vs 57.2%
   (validation).

Both components exist to eventually condition bullpen resolution on the
SIMULATED game state as each Monte Carlo trial develops (steps 2-5 of the
spec, NOT built yet) -- this module only fits and validates the POLICY
against real history; it is not imported by game_simulator.py or any
production path yet.

HOOK-FRAILTY EXTENSION (task #145): a pre-registered tail-calibration check found
the starter-hook model's per-cell hazard is well-calibrated in isolation, but
SEQUENTIALLY COMPOUNDING it across a start's own PAs (exactly how a simulator would
evaluate it trial-by-trial) badly overpredicts the "shelled starter hooked early"
scenario: on 2025 held-out data, real P(exited by inning 4 | reached inning 4 with
5+ runs allowed) = 29.3%, but the naive per-PA-independent compound implies 42.0%.
Root cause, confirmed by elimination (recency refit ~1pp help; per-inning-boundary
evaluation cadence overshoots the OTHER direction; damping low-danger cells barely
moves it; a properly-specified discrete-time logit hazard only closes ~1/4 of the
gap): a genuine SELECTION EFFECT. "Reached inning 4 with 5+ runs allowed" is not a
random sample of starts -- a quick-hook manager pulls the guy at 3 runs, so that
start never enters the evaluation set. The starts that DO reach that state are
disproportionately the ones with low hook-propensity that day (tired pen, a manager
who's already written the game off). This is frailty's textbook signature: real
per-start correlated propensity that a marginal, independent-per-PA hazard model
cannot represent no matter its functional form.

Fix: `hook_frailty_sigma(inning)` draws ONE latent z~N(0,1) per (start, trial),
scaled by a DECAYING per-inning sigma and applied as a mean-preserving logistic
offset to every PA's hazard for that start/trial (same "one extra CRN-keyed draw,
one fitted constant, mean-preserving" recipe as SHOCK_SIGMA in game_simulator.py).
Decaying (not constant) sigma was necessary: a constant sigma fit to the inning-4
target (1.40) overcorrected at inning 5+ (undershooting real rates) and made the
OPPOSITE tail -- P(still in at inning 7 | clean start) -- worse, not better,
confirming late-game hook decisions are increasingly pitch-count/fatigue-ceiling
-driven, not frailty-driven, so frailty's influence must shrink with inning.
HOOK_FRAILTY_SIGMA1=3.8, HOOK_FRAILTY_DECAY=0.65 (sigma(inning) = SIGMA1 *
DECAY**(inning-1)), fit via n-weighted grid search on 2023-24, validated held-out
on 2025 across a full grid (P(exited by inning k | {3+, 5+} runs allowed) for
k=3..7, PLUS the opposite tail P(still in at inning k | clean start)):

  Grid 1 (5+ runs, k=3..7), real vs decaying-frailty-implied: 0.138 vs 0.139,
  0.293 vs 0.326, 0.623 vs 0.615, 0.869 vs 0.829, 0.941 vs 0.934 -- dramatically
  better than both no-frailty (0.208/0.420/0.713/0.895/0.964) and constant sigma=1.40
  (0.166/0.322/0.538/0.700/0.785) at every single k.

KNOWN LIMITATION (not fixed by any frailty shape, logged not built): the opposite
tail's deepest point, P(still in at inning 7 | clean start throughout), stays
substantially miscalibrated under every variant tested (real 28.5% vs ~51-62%
modeled) -- this specific miss is not a frailty-shape problem, it reflects a
genuinely MISSING feature (pitch count / times-through-order fatigue ceiling,
which caps even clean starts around 6-7 innings regardless of runs allowed or
frailty) that this hazard model has no signal for at all. A future refinement
could add real pitch-count as a feature; not built here (scope discipline).

ALSO LOGGED, NOT BUILT: part of a start's "frailty" is actually observable --
bullpen rest state (which relievers are unavailable that day), already computed
by the bullpen model. A future refinement could condition part of the offset on
pen availability instead of leaving it fully random, converting some of this
unexplained variance into real signal. Not built now (scope discipline).

UNRESOLVED CAVEAT (do not let sigma silently take credit for this): during the
discrete-time logit hazard exploration (a rejected intermediate approach, kept only
as a diagnostic), the `margin` coefficient came back statistically null (p=0.93) in
a smooth linear specification, despite margin being the single strongest driver in
the bucketed cross-tab table (trailing-big vs leading-big spans 4.45 to 6.30 mean
hook innings). This suggests margin's true effect is threshold-like or entangled
with runs-allowed (the two are highly collinear in early innings) rather than a
smooth linear gradient. The cross-tab table (which retains margin as a discrete
bucket) sidesteps this, so it isn't a live bug -- but it's a genuine anomaly in the
underlying data-generating story that remains unexplained, not resolved by frailty.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, roots_hermitenorm

from src.models.bullpen import mark_pitching_team, _attach_starter_id, build_relief_pa

HOOK_PRIOR_N = 30  # Bayesian-shrinkage pseudo-count toward the inning-only marginal,
                   # same magnitude/role as this project's other context-factor priors

HOOK_FRAILTY_SIGMA1 = 3.8   # task #145: fitted per-start frailty magnitude at inning 1
HOOK_FRAILTY_DECAY = 0.65   # per additional inning, sigma(inning) = SIGMA1 * DECAY**(inning-1)

_GH_NODES, _GH_WEIGHTS = roots_hermitenorm(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()  # Gauss-Hermite quadrature, normalized to a
                                                # probability measure, for exact E_h[sigmoid(x+h)]


def hook_frailty_sigma(inning: int, sigma1: float = HOOK_FRAILTY_SIGMA1,
                        decay: float = HOOK_FRAILTY_DECAY) -> float:
    return sigma1 * (decay ** (inning - 1))


def _marginal_mean(x: float, sigma: float) -> float:
    """E_h[sigmoid(x+h)], h~N(0,sigma^2), via Gauss-Hermite quadrature."""
    return float(np.sum(_GH_WEIGHTS * expit(x + _GH_NODES * sigma)))


_pre_noise_logit_cache: dict = {}


def solve_pre_noise_logit(p: float, sigma: float) -> float:
    """Exact (not closed-form-approximate) mean-preserving correction: returns x such
    that E_h[sigmoid(x+h)] == p for h~N(0,sigma^2). The closed-form (Zeger-Liang-Albert)
    approximation was checked and found biased at low p -- exactly where most hook
    cells live -- so this project uses an exact numerical root-find instead, cached
    per (p, sigma) since there are only ~300 distinct table cells."""
    key = (round(p, 6), round(sigma, 4))
    if key in _pre_noise_logit_cache:
        return _pre_noise_logit_cache[key]
    if sigma < 1e-6:
        x = logit(np.clip(p, 1e-6, 1 - 1e-6))
        _pre_noise_logit_cache[key] = x
        return x
    p = np.clip(p, 1e-6, 1 - 1e-6)
    target_logit = logit(p)
    lo, hi = target_logit - 6 * sigma - 5, target_logit + 6 * sigma + 5
    x = brentq(lambda x: _marginal_mean(x, sigma) - p, lo, hi, xtol=1e-8)
    _pre_noise_logit_cache[key] = x
    return x


def apply_hook_frailty(p_baseline: float, z: float, inning: int,
                        sigma1: float = HOOK_FRAILTY_SIGMA1,
                        decay: float = HOOK_FRAILTY_DECAY) -> float:
    """Adjusts a baseline (already-validated, per-cell) hook hazard for one PA using
    this start/trial's own frailty draw `z` (a single standard normal drawn ONCE per
    start per trial and reused across every inning of that start -- its INFLUENCE
    shrinks with inning via `sigma`, but it is the same underlying latent draw
    throughout, representing one correlated propensity for that start). Mean-preserving:
    averaging this over the frailty distribution reproduces p_baseline exactly."""
    sigma = hook_frailty_sigma(inning, sigma1, decay)
    if sigma < 1e-6:
        return p_baseline
    x = solve_pre_noise_logit(p_baseline, sigma)
    return float(expit(x + z * sigma))


def attach_margin(pa: pd.DataFrame) -> pd.DataFrame:
    """Adds `pitch_score` (the DEFENSE's own score entering this PA) and
    `margin` (pitch_score - bat_score -- positive means the defense's team
    is winning) to a copy of `pa`. Needed because the PA table only carries
    `bat_score` (the batting team's own score) directly; the pitching team's
    score requires looking up the OTHER team's most recently completed
    half-inning, which this reconstructs via a per-half cumulative sum."""
    bot_halves = pa[pa["inning_topbot"] == "Bot"].groupby(["game_pk", "inning"], as_index=False)["runs_scored"].sum()
    bot_halves = bot_halves.sort_values(["game_pk", "inning"])
    bot_halves["home_score_cum"] = bot_halves.groupby("game_pk")["runs_scored"].cumsum()
    top_halves = pa[pa["inning_topbot"] == "Top"].groupby(["game_pk", "inning"], as_index=False)["runs_scored"].sum()
    top_halves = top_halves.sort_values(["game_pk", "inning"])
    top_halves["away_score_cum"] = top_halves.groupby("game_pk")["runs_scored"].cumsum()

    out = pa.copy()
    out["pitching_team"] = mark_pitching_team(out)
    home_prior = bot_halves[["game_pk", "inning", "home_score_cum"]].copy()
    home_prior["inning"] = home_prior["inning"] + 1  # shift: home's score after inning N-1's Bot applies to inning N's Top
    out = out.merge(home_prior.rename(columns={"home_score_cum": "_home_pitch_score"}), on=["game_pk", "inning"], how="left")
    away_prior = top_halves[["game_pk", "inning", "away_score_cum"]].copy()
    out = out.merge(away_prior.rename(columns={"away_score_cum": "_away_pitch_score"}), on=["game_pk", "inning"], how="left")
    out["pitch_score"] = np.where(out["inning_topbot"] == "Top", out["_home_pitch_score"], out["_away_pitch_score"]).astype(float)
    out["pitch_score"] = out["pitch_score"].fillna(0.0)
    out["margin"] = out["pitch_score"] - out["bat_score"]
    return out.drop(columns=["_home_pitch_score", "_away_pitch_score"])


def bucket_inning(i) -> int:
    return min(int(i), 9)


def bucket_pa_count(n) -> str:
    if n <= 6: return "1-6"
    if n <= 12: return "7-12"
    if n <= 18: return "13-18"
    return "19+"


def bucket_runs_allowed(r) -> int:
    return min(int(r), 4)


def bucket_margin(m) -> str:
    if m <= -4: return "trail_big"
    if m <= -1: return "trail"
    if m == 0: return "tied"
    if m <= 3: return "lead"
    return "lead_big"


def build_starter_hook_events(pa_with_margin: pd.DataFrame) -> pd.DataFrame:
    """One row per starter PA: (inning, pa_count, cum_runs_allowed, margin,
    is_last_pa) -- is_last_pa marks the starter's final PA of that start
    (whether a real managerial hook or the game simply concluding while he
    was still in -- not separately distinguished in this first pass)."""
    tmp = _attach_starter_id(pa_with_margin)
    starter_pa = tmp[tmp["pitcher"] == tmp["starter_id"]].copy()
    starter_pa = starter_pa.sort_values(["game_pk", "pitching_team", "inning", "at_bat_number"])
    starter_pa["pa_count"] = starter_pa.groupby(["game_pk", "pitching_team"]).cumcount() + 1
    starter_pa["cum_runs_allowed"] = (
        starter_pa.groupby(["game_pk", "pitching_team"])["runs_scored"].cumsum() - starter_pa["runs_scored"]
    )
    starter_pa["is_last_pa"] = (
        starter_pa.groupby(["game_pk", "pitching_team"])["pa_count"].transform("max") == starter_pa["pa_count"]
    )
    starter_pa["inning_b"] = starter_pa["inning"].apply(bucket_inning)
    starter_pa["pa_count_b"] = starter_pa["pa_count"].apply(bucket_pa_count)
    starter_pa["runs_allowed_b"] = starter_pa["cum_runs_allowed"].apply(bucket_runs_allowed)
    starter_pa["margin_b"] = starter_pa["margin"].apply(bucket_margin)
    return starter_pa


def fit_hook_policy(events: pd.DataFrame, prior_n: int = HOOK_PRIOR_N) -> pd.DataFrame:
    """Bayesian-shrunk P(exit) per (inning, pa_count, runs_allowed, margin)
    cell, shrunk toward the inning-only marginal -- same discipline as
    build_state_factors_by_season/build_ttop_factors_by_season."""
    table = events.groupby(["inning_b", "pa_count_b", "runs_allowed_b", "margin_b"]).agg(
        n=("is_last_pa", "size"), exits=("is_last_pa", "sum")
    ).reset_index()
    table["p_exit_raw"] = table["exits"] / table["n"]
    inning_marginal = events.groupby("inning_b")["is_last_pa"].mean()
    table = table.merge(inning_marginal.rename("inning_prior"), left_on="inning_b", right_index=True)
    table["p_exit"] = (table["exits"] + prior_n * table["inning_prior"]) / (table["n"] + prior_n)
    return table


def build_reliever_tier_log(pa_with_margin: pd.DataFrame) -> pd.DataFrame:
    """One row per relief PA, tagged with (inning, margin, save_situation,
    tier, is_closer). tier = season-long usage-rank tercile within that
    (team, season)'s own bullpen -- a descriptive/measurement construct
    (uses full-season data, not walk-forward), fine for characterizing the
    real policy this module fits, not for making a leakage-sensitive
    prediction. is_closer = whoever has the most innings>=9 relief PAs for
    that (team, season)."""
    relief_pa = build_relief_pa(pa_with_margin)
    relief_pa["inning_b"] = relief_pa["inning"].apply(bucket_inning)
    relief_pa["margin_b"] = relief_pa["margin"].apply(bucket_margin)
    relief_pa["save_situation"] = (relief_pa["inning"] >= 7) & (relief_pa["margin"] >= 1) & (relief_pa["margin"] <= 3)

    usage = relief_pa.groupby(["team", "season", "pitcher"]).size().rename("n_relief_pa").reset_index()
    usage["tier"] = usage.groupby(["team", "season"])["n_relief_pa"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["mopup", "middle", "leverage"])
    )
    late_counts = relief_pa[relief_pa["inning"] >= 9].groupby(["team", "season", "pitcher"]).size().rename("n_late").reset_index()
    closer_ids = late_counts.sort_values("n_late", ascending=False).drop_duplicates(["team", "season"])[["team", "season", "pitcher"]]
    closer_ids["is_closer"] = True

    relief_pa = relief_pa.merge(usage[["team", "season", "pitcher", "tier"]], on=["team", "season", "pitcher"], how="left")
    relief_pa = relief_pa.merge(closer_ids, on=["team", "season", "pitcher"], how="left")
    relief_pa["is_closer"] = relief_pa["is_closer"].fillna(False)
    return relief_pa


def fit_tier_policy(tier_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tier_by_margin, closer_by_situation): empirical P(tier | margin) and
    P(closer | inning, save_situation), unshrunk (both already have ample
    per-cell sample size given the season-long tier assignment)."""
    tier_by_margin = pd.crosstab(tier_log["margin_b"], tier_log["tier"], normalize="index")
    closer_by_situation = tier_log.groupby(["inning_b", "save_situation"])["is_closer"].mean().reset_index()
    return tier_by_margin, closer_by_situation


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED

    FIT_SEASONS = {2023, 2024}
    VALIDATE_SEASON = 2025

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    pa = attach_margin(pa)

    print("=== STARTER-HOOK MODEL ===", flush=True)
    events = build_starter_hook_events(pa)
    fit_events = events[events["season"].isin(FIT_SEASONS)]
    val_events = events[events["season"] == VALIDATE_SEASON]
    hook_table = fit_hook_policy(fit_events)
    print(f"fit sample (2023-2024): {len(fit_events)} starter PAs, {len(hook_table)} cells")

    exits_val = val_events[val_events["is_last_pa"]]
    print("\nReal 2025 validation: mean hook inning by margin bucket (at the hook)")
    print(exits_val.groupby("margin_b")["inning"].agg(["mean", "count"]).to_string())

    print("\n=== RELIEVER-TIER MODEL ===", flush=True)
    tier_log = build_reliever_tier_log(pa)
    fit_tier = tier_log[tier_log["season"].isin(FIT_SEASONS)]
    val_tier = tier_log[tier_log["season"] == VALIDATE_SEASON]
    tier_by_margin_fit, closer_by_situation_fit = fit_tier_policy(fit_tier)
    tier_by_margin_val, closer_by_situation_val = fit_tier_policy(val_tier)

    print("\nFit (2023-2024) tier share by margin:")
    print(tier_by_margin_fit.round(3).to_string())
    print("\nValidation (2025) tier share by margin:")
    print(tier_by_margin_val.round(3).to_string())
    print("\nFit (2023-2024) closer share by (inning, save situation):")
    print(closer_by_situation_fit.to_string())
    print("\nValidation (2025) closer share by (inning, save situation):")
    print(closer_by_situation_val.to_string())
