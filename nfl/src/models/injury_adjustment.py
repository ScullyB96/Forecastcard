"""Key-injury adjustment for Layer 1, covering four positions validated to
matter: WR/TE, RB, offensive line (T/G/C), and cornerback.

Two different identification methods are used depending on data availability:

  - WR/TE and RB: identified by pregame target/carry share (from Layer 2),
    joined to the injury report by gsis_id (reliable direct ID match).
  - Offensive line and cornerback: box-score stats don't cover linemen or
    corners, so starters are identified by snap-count share instead. The
    fantasy-oriented ID crosswalks (nfl_data_py's import_ids, and even
    weekly_rosters' own pfr_id field) cover O-line at under 1% -- fantasy
    platforms don't track them -- so these are joined to the injury report by
    normalized NAME within the same (season, week, team), a small,
    disambiguated search space rather than a risky global fuzzy match.

In every case, a player's share/snap-pct is forward-filled across weeks they
didn't play, before picking the "presumed starter." This matters: a player
who is genuinely ruled Out has zero stats that week and no row in the box
score, so naively picking "top performer among players who played this week"
would circularly exclude exactly the injured players being detected.

Positions tested and REJECTED (see project notes for details): DE/EDGE
(sign flips train-to-test), safety, linebacker (both: no improvement, one
with a backwards coefficient sign). Not included here.

CB detection uses the top THREE cornerbacks by snap share, not two -- the
modern NFL plays enough nickel (3-CB sub-packages) that the "3rd" corner is
a near-full-time starter. Tested top-2 through top-5: top-3 was the clear
best (test MAE 9.948 -> 9.919 over top-2), and top-4/5 dilute the signal
enough that the home-side coefficient's sign becomes noise.

Also tested: including "Doubtful" and "Questionable" designations alongside
"Out". Both made things worse (Doubtful broke the home-CB coefficient's sign;
Questionable pushed test MAE above the unadjusted baseline entirely) --
"Out" is the only designation reliable enough to use, matching common
sports-betting wisdom that "Questionable" is close to a coin flip.

Final joint model, fit on 2018-2021 train and validated on 2022-2025 holdout
(each individually validated first, then refit together to confirm they
don't double-count the same "team is banged up" signal -- they don't, pairwise
correlations are all under 0.03):

    margin_adjustment = -0.38
        + 1.43 * away_skill_out       (away WR/TE or RB presumed starter Out)
        + 1.45 * away_ol_out          (away O-line presumed starter Out)
        - 1.22 * home_cb_out          (home top-3 CB presumed starter Out)
        + 3.18 * away_cb_out          (away top-3 CB presumed starter Out)

Overall test MAE: 10.000 -> 9.906. CB carries by far the largest effect and
works in BOTH directions (unlike the offense positions, which only showed a
real away-side effect) -- losing a top corner is the single most exploitable
injury signal found in this project.

FORWARD-USE COEFFICIENTS (JOINT_COEFS_FORWARD), v1 (SUPERSEDED, see v2 below):
predict_2026.py recalibrates its margin model on 2022-2025 rather than
2018-2021 (see that module's docstring -- 2019-2020 badly understate
home-field advantage, a COVID crowd-size artifact). Refitting JOINT_COEFS
against that new base prediction's residuals, using only 2022-2025, moved
every coefficient a lot (e.g. home_cb: -1.22 -> -4.67) -- too much to be pure
signal given the sample is the same size as before (~150-300 flagged games
total across all 4 signals). That's consistent with these already being
estimated from rare events, not evidence of a real regime change. Rather than
pick either 4-year window, each game's residual is computed against ITS OWN
era's well-fit calibration (old games against the old fit, new games against
the new fit -- both are ~unbiased within their own era), then pooled across
all 8 seasons for one lower-variance regression:

    margin_adjustment = -0.34
        + 2.56 * away_skill_out
        + 1.20 * away_ol_out
        - 3.43 * home_cb_out
        + 3.79 * away_cb_out

FORWARD-USE COEFFICIENTS, v2 (current -- fit against the BLEND residual, not
the Layer-1 residual): a third-party review (2026-07) flagged that the live
pipeline (weekly_update.py) applies JOINT_COEFS_FORWARD and the QB swap_delta
adjustment ON TOP OF the already market-blended prediction (margin blend
weight on our own model is only 0.02-0.17 -- the blend is 88-98% the closing
line), while both were validated against PURE LAYER-1 residuals. Since the
market already prices in real injury/QB news, this is very likely
double-counting the same information twice.

Refit against `actual_margin - blended_pred` (same era-consistent-residual
pooling as v1, now scoring each era with that era's own market blend too, not
just its own Layer-1 calibration; n=2127 pooled games) confirms this: every
coefficient shrinks, and TWO of the four terms are no longer individually
distinguishable from zero once market pricing is netted out --
away_skill_flag (t=+1.64) and away_ol_flag (t=+1.14). Both CB terms survive
(home_cb_flag t=-2.43, away_cb_flag t=+3.27) -- corners are the one injury
signal genuinely not fully priced into the market number already:

    margin_adjustment = -0.08
        + 0.00 * away_skill_out   (dropped -- not distinguishable from 0 on blend residual)
        + 0.00 * away_ol_out      (dropped -- not distinguishable from 0 on blend residual)
        - 2.58 * home_cb_out
        + 3.35 * away_cb_out

The QB swap_delta coefficient (SWAP_B in weekly_update.py/predict_2026.py, not
stored here) shrinks the same way: 6.616 (Layer-1 residual) -> 2.978 (blend
residual, same pooling), t=+1.53 -- kept despite being just under conventional
significance, since a stale-rating-catches-up mechanism is well-established
and not obviously spurious, and it's a single coefficient rather than a
4-way multicollinear fit.

Honest empirical caveat: a strict walk-forward check (fit OLD_ERA=2018-2021
only, score NEW_ERA=2022-2025 blind) shows the v1 (double-counting) and v2
(corrected) coefficients are statistically indistinguishable on point-MAE
(9.459 vs 9.493, difference well within noise) -- so this fix is not
primarily an accuracy win. It matters because v1's larger coefficients
overstate the true size of the edge, which is what actually gets used for bet
sizing (Kelly-style staking on an overstated edge is a real, quantifiable risk
that MAE alone doesn't surface -- exactly the "wrong objective function"
critique the same review raised separately).

FORWARD-USE COEFFICIENTS, v3 (current -- symmetric CB constraint, review
#1.6): v2 left home_cb_flag (-2.576) and away_cb_flag (+3.351) as independent,
differently-sized coefficients. Reparameterizing as ONE coefficient on
(away_cb_flag - home_cb_flag) -- forcing the home and away effects to be
equal and opposite -- roughly doubles the effective sample for that
coefficient (every CB-out game contributes to the same estimate, not two
separate ones) and the t-stat jumps from -2.43/+3.27 (independent) to +4.01
(symmetric, pooled 8-season). A strict walk-forward check (fit OLD_ERA only,
score NEW_ERA blind) confirms the symmetric version generalizes better, not
just fits better in-sample: on the CB-flagged holdout subset (n=183), MAE
9.968 (v2) -> 9.836 (v3), and signed bias +1.596 (v2, a real miscalibration)
-> +0.542 (v3):

    margin_adjustment = -0.02
        + 0.00 * away_skill_out   (dropped, see v2)
        + 0.00 * away_ol_out      (dropped, see v2)
        - 2.98 * home_cb_out
        + 2.98 * away_cb_out

The QB swap_delta coefficient is refit jointly with this symmetric term:
2.970 (from 2.978 in v2 -- negligible change, as expected since it's a
separate, uncorrelated predictor).

FORWARD-USE COEFFICIENTS, v4 (current -- shrunk toward the rolling-origin fold
median, review round 3 #1): v3's 2.977 is a pooled 2018-2025 fit, which is
RESUBSTITUTION for any evaluation scoring TEST=2022-2025 -- it was fit on data
including the very seasons such an evaluation checks. The honest,
out-of-sample estimates come from src/models/validate_adjustment_layer.py's
rolling-origin check (refit on strictly-prior seasons only): 1.590, 2.353,
2.539, 2.629 across the four folds -- EVERY one below 2.977. Shrunk to the
fold median, 2.446, as an asymmetric-downside precaution (see the constant's
own comment below for the full reasoning, including the permutation-test and
walk-forward-ATS results that came back reassuring on direction and magnitude
of edge even though the point-estimate coefficient itself was inflated).

Used only by predict_2026.py and weekly_update.py. JOINT_COEFS (above) stays
as originally validated for historical backtesting (predict.py), since that
number needs to reflect exactly what was walk-forward tested, not a later
refinement.
"""

import numpy as np
import pandas as pd

from src.ingest.name_matching import norm_name
from src.utils.paths import DATA_PROCESSED, DATA_RAW
from src.utils.stats import fit_linear

STATUS_OUT = "Out"

# PURE LAYER-1 residual basis (fit 2018-2021, scored 2022-2025 holdout -- see module
# docstring above). Frozen exactly as originally validated; used only by predict.py for
# honest historical backtesting, which must reflect what was actually walk-forward tested.
# Do NOT refit or "sync" this to JOINT_COEFS_FORWARD's blend-residual basis below (review
# round 2, #1.3 -- same same-name-different-basis trap SWAP_B_MARKET/SWAP_B_LAYER1 guards
# against in weekly_update.py/predict_2026.py).
JOINT_COEFS = {
    "intercept": -0.382,
    "away_skill_flag": 1.434,
    "away_ol_flag": 1.447,
    "home_cb_flag": -1.217,
    "away_cb_flag": 3.184,
}

# pooled 8-season, era-consistent BLEND-residual fit, symmetric CB constraint
# (v3) -- see module docstring. Used by predict_2026.py and weekly_update.py,
# applied on top of the market-blended prediction. away_skill_flag/
# away_ol_flag are zeroed (not distinguishable from 0 once market pricing is
# netted out); kept as explicit keys, not removed, so apply_joint_adjustment's
# signature doesn't need to change and a future refit can re-populate them if
# warranted. home_cb_flag/away_cb_flag are now equal-magnitude, opposite-sign
# by construction (fit as one coefficient on away_cb_flag - home_cb_flag).
#
# v4 (review round 3, #1): the pooled v3 fit above (2.977) is RESUBSTITUTION for
# any TEST-window evaluation -- it's fit on 2018-2025 pooled, which includes the
# very 2022-2025 games any such evaluation scores. src/models/validate_adjustment_layer.py's
# rolling-origin check (fit on strictly-prior seasons only) gives four honest,
# out-of-sample estimates -- 1.590, 2.353, 2.539, 2.629 -- and every single one sits
# below 2.977. Shrunk to the fold MEDIAN (2.446) here as an asymmetric-downside
# precaution: if the pooled estimate is genuinely inflated, a ~0.5pt-per-game
# systematic error on ~46 CB-flagged games/season is a real, avoidable cost; if the
# effect is exactly as strong as pooled suggests, shrinking gives up a little edge.
# A permutation test (shuffle each team's real CB-out flags across its own played
# weeks within season, preserving its real annual flag count, re-run the full
# fit-and-score procedure 1000x) put the real ATS% at the ~97.5th percentile of the
# single-configuration null -- real signal, not purely a forking-paths artifact of
# the wide coefficient search, but not overwhelming either given how wide that
# search was (top-2/3/4/5 corners x 3 injury-report designations x several position
# candidates x 3 residual bases x the symmetry constraint). CORRECTED (review round
# 4, #1): that single-config null only prices in "fit and scored on the same
# games," not the specification search itself -- extending the permutation to
# re-run the actual corner-count x designation grid search per shuffle puts the
# real number at only the ~95th percentile of the corrected null, right at the
# edge of conventional significance, not comfortably in the tail. Also (round 4,
# #1): the WALK-FORWARD ATS% (61.8%) came back almost identical to the flawed
# resubstitution number (62.4%) NOT because the in-sample fit was vindicated, but
# because ATS is a sign-of-disagreement metric and every fold coefficient has the
# same sign -- the two numbers are nearly the same statistic, not independent
# corroboration. See MODEL_DOCUMENTATION.md §6.1.1 for the full, current writeup;
# held provisionally, not proven as strongly as round 3 first reported. Shrinking
# here still protects the point-estimate margin prediction specifically (every
# honest fold estimate sits below 2.977), independent of how the ATS question
# above resolves.
#
# WHY THIS IS A FROZEN CONSTANT, NOT AUTO-REFIT EVERY RUN (unlike LEAGUE_AVG_PLAYS,
# the EB-fitted prior_weights, or the TD-probability calibrator, all of which
# `weekly_update.py` recomputes fresh from current data every pipeline run): those
# are single, cheap, mechanical recalculations (a mean, a closed-form MoM estimate,
# a 2-parameter OLS fit) with no real judgment call involved. This coefficient is
# the opposite -- it took 4 review rounds of real scrutiny (resubstitution bias,
# a wide specification search, a permutation test, a lookahead audit) to arrive at
# the current value, and silently auto-refitting it every week would re-run that
# entire judgment-laden process with zero human review of the result. Deliberately
# left as a frozen production value, updated only through a reviewed refit
# (matching how `JOINT_COEFS`/`SWAP_B_LAYER1` are already treated) -- if a future
# session revisits this, that should be an explicit, documented refit (e.g. once
# a season, with the same validate_adjustment_layer.py scrutiny), not an automatic
# recompute inside the live pipeline.
JOINT_COEFS_FORWARD = {
    "intercept": -0.018,
    "away_skill_flag": 0.0,
    "away_ol_flag": 0.0,
    "home_cb_flag": -2.446,
    "away_cb_flag": 2.446,
}


def build_presumed_starter_by_id(df: pd.DataFrame, share_col: str, positions: list[str]) -> pd.DataFrame:
    """WR/TE and RB: forward-filled share, keyed by gsis player_id."""
    pool = df[df["position"].isin(positions)][
        ["season", "week", "recent_team", "player_id", "player_display_name", share_col]
    ].copy()
    all_weeks = pool[["season", "week"]].drop_duplicates()

    panels = []
    for (season, team), grp in pool.groupby(["season", "recent_team"]):
        players = grp["player_id"].unique()
        weeks = sorted(all_weeks[all_weeks["season"] == season]["week"].unique())
        panel = pd.MultiIndex.from_product([weeks, players], names=["week", "player_id"]).to_frame(index=False)
        panel = panel.merge(
            grp[["week", "player_id", "player_display_name", share_col]], on=["week", "player_id"], how="left"
        )
        panel = panel.sort_values(["player_id", "week"])
        panel[share_col] = panel.groupby("player_id")[share_col].ffill()
        panel["season"] = season
        panel["recent_team"] = team
        panels.append(panel)

    full = pd.concat(panels, ignore_index=True).dropna(subset=[share_col])
    return full.sort_values(share_col, ascending=False).drop_duplicates(["season", "week", "recent_team"])


def build_presumed_starters_by_name(
    snap_counts: pd.DataFrame, position: str, n_starters: int, snap_col: str
) -> pd.DataFrame:
    """O-line and CB: forward-filled snap share, keyed by normalized name
    (no reliable ID crosswalk covers these positions). n_starters=2 for T/G/CB
    (two per side), 1 for C (one starting center)."""
    pool = snap_counts[snap_counts["position"] == position][["season", "week", "team", "name_norm", snap_col]].copy()
    pool = pool.groupby(["season", "week", "team", "name_norm"], as_index=False)[snap_col].max()
    all_weeks = pool[["season", "week"]].drop_duplicates()

    panels = []
    for (season, team), grp in pool.groupby(["season", "team"]):
        players = grp["name_norm"].unique()
        weeks = sorted(all_weeks[all_weeks["season"] == season]["week"].unique())
        panel = pd.MultiIndex.from_product([weeks, players], names=["week", "name_norm"]).to_frame(index=False)
        panel = panel.merge(grp[["week", "name_norm", snap_col]], on=["week", "name_norm"], how="left")
        panel = panel.sort_values(["name_norm", "week"])
        panel[snap_col] = panel.groupby("name_norm")[snap_col].ffill()
        panel["season"] = season
        panel["team"] = team
        panels.append(panel)

    full = pd.concat(panels, ignore_index=True).dropna(subset=[snap_col])
    full = full.sort_values(["season", "week", "team", snap_col], ascending=[True, True, True, False])
    full["rank"] = full.groupby(["season", "week", "team"]).cumcount()
    return full[full["rank"] < n_starters]


def flag_out_by_id(presumed: pd.DataFrame, injuries: pd.DataFrame, flag_name: str) -> pd.DataFrame:
    out = injuries[injuries["report_status"] == STATUS_OUT][["season", "week", "gsis_id"]].drop_duplicates()
    merged = presumed.merge(
        out, left_on=["season", "week", "player_id"], right_on=["season", "week", "gsis_id"], how="left", indicator=True
    )
    merged[flag_name] = merged["_merge"] == "both"
    return merged.drop(columns=["_merge", "gsis_id"])


def flag_team_week_out_by_name(presumed: pd.DataFrame, injuries: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """Collapse a presumed-starters table (possibly >1 starter/team/week) to a
    team-week count of how many of them are Out, matched by normalized name."""
    out = injuries[injuries["report_status"] == STATUS_OUT][["season", "week", "team", "name_norm"]].drop_duplicates()
    out["is_out"] = True
    merged = presumed.merge(out, on=["season", "week", "team", "name_norm"], how="left")
    merged["is_out"] = merged["is_out"].fillna(False).astype(int)
    return merged.groupby(["season", "week", "team"])["is_out"].sum().reset_index(name=out_col)


def _latest_season_range_file(seasons_dir, prefix: str):
    """Picks the widest/most-recent `<prefix>_<start>_<end>.parquet` file in
    seasons_dir, rather than a hardcoded literal -- BUG FIX 2026-07 (found
    during a full-codebase review): this function used to hardcode
    `injuries_2016_2025.parquet`/`snap_counts_2016_2025.parquet` regardless of
    what season range the pipeline had actually just fetched. Harmless while
    2026 data isn't published yet (the fallback path happens to still write
    the 2025-ending filename), but once a real `..._2016_2026.parquet` file
    exists, the old hardcoded read would have silently kept using the stale
    2025-ending one forever -- no crash, just an entire season of missing
    injury data. Picks by highest end-year, matching this project's
    `{prefix}_{min}_{max}.parquet` naming convention everywhere else."""
    candidates = sorted(seasons_dir.glob(f"{prefix}_*_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"no {prefix}_*.parquet file found in {seasons_dir}")

    def _end_year(path):
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(candidates, key=_end_year)


def compute_injury_flags(seasons_dir=DATA_RAW, processed_dir=DATA_PROCESSED) -> pd.DataFrame:
    """Build the four team-week injury flags used by the joint model. Returns
    columns [season, week, team, skill_out, ol_out, cb_out] (skill_out/ol_out/
    cb_out are counts; threshold at >=1 to get the binary flags used in
    JOINT_COEFS)."""
    inj = pd.read_parquet(_latest_season_range_file(seasons_dir, "injuries"))
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)

    tgt = pd.read_parquet(processed_dir / "player_target_share_ratings.parquet")
    carry = pd.read_parquet(processed_dir / "player_carry_share_ratings.parquet")
    top_wr = flag_out_by_id(build_presumed_starter_by_id(tgt, "pregame_target_share_calc", ["WR", "TE"]), inj, "wr_out")
    top_rb = flag_out_by_id(build_presumed_starter_by_id(carry, "pregame_carry_share_calc", ["RB"]), inj, "rb_out")
    skill = pd.concat(
        [
            top_wr.rename(columns={"recent_team": "team"})[["season", "week", "team", "wr_out"]],
            top_rb.rename(columns={"recent_team": "team", "rb_out": "wr_out"})[["season", "week", "team", "wr_out"]],
        ]
    )
    skill_out = skill.groupby(["season", "week", "team"])["wr_out"].max().reset_index(name="skill_out")

    sc = pd.read_parquet(_latest_season_range_file(seasons_dir, "snap_counts"))
    sc["name_norm"] = sc["player"].apply(norm_name)
    sc["season"] = sc["season"].astype(int)
    sc["week"] = sc["week"].astype(int)
    inj["name_norm"] = inj["full_name"].apply(norm_name)

    top_t = build_presumed_starters_by_name(sc, "T", 2, "offense_pct")
    top_g = build_presumed_starters_by_name(sc, "G", 2, "offense_pct")
    top_c = build_presumed_starters_by_name(sc, "C", 1, "offense_pct")
    ol_all = pd.concat([top_t, top_g, top_c])
    ol_out = flag_team_week_out_by_name(ol_all, inj, "ol_out")

    top_cb = build_presumed_starters_by_name(sc, "CB", 3, "defense_pct")  # top-3: includes the nickel corner
    cb_out = flag_team_week_out_by_name(top_cb, inj, "cb_out")

    flags = skill_out.merge(ol_out, on=["season", "week", "team"], how="outer")
    flags = flags.merge(cb_out, on=["season", "week", "team"], how="outer")
    for c in ["skill_out", "ol_out", "cb_out"]:
        flags[c] = flags[c].fillna(0).astype(int)
    return flags


def apply_joint_adjustment(games: pd.DataFrame, flags: pd.DataFrame, coefs: dict = None) -> pd.Series:
    """games must have [season, week, team_a (home), team_b (away)]. Returns
    the pts-adjustment to add to the base margin prediction (home - away).
    coefs defaults to JOINT_COEFS (the historically-validated fit); pass
    JOINT_COEFS_FORWARD for 2026 forward-looking predictions -- see module
    docstring for why they differ."""
    coefs = coefs if coefs is not None else JOINT_COEFS
    m = games.merge(
        flags.rename(columns={"team": "team_a"}), on=["season", "week", "team_a"], how="left", suffixes=("", "_home")
    )
    m = m.merge(
        flags.rename(columns={"team": "team_b"}), on=["season", "week", "team_b"], how="left", suffixes=("_home", "_away")
    )
    for c in ["skill_out_home", "ol_out_home", "cb_out_home", "skill_out_away", "ol_out_away", "cb_out_away"]:
        if c not in m.columns:
            m[c] = 0
        m[c] = m[c].fillna(0)

    away_skill_flag = (m["skill_out_away"] >= 1).astype(float)
    away_ol_flag = (m["ol_out_away"] >= 1).astype(float)
    home_cb_flag = (m["cb_out_home"] >= 1).astype(float)
    away_cb_flag = (m["cb_out_away"] >= 1).astype(float)

    # NOTE: the intercept is applied to every game, flagged or not -- this matches
    # how it was validated (X_test @ coefs, with an all-ones intercept column).
    # It's small (~-0.3 pts) and reflects a tiny leftover test-period bias, not a
    # standalone injury effect.
    return (
        coefs["intercept"]
        + coefs["away_skill_flag"] * away_skill_flag
        + coefs["away_ol_flag"] * away_ol_flag
        + coefs["home_cb_flag"] * home_cb_flag
        + coefs["away_cb_flag"] * away_cb_flag
    )


if __name__ == "__main__":
    flags = compute_injury_flags()
    flags.to_parquet(DATA_PROCESSED / "injury_flags.parquet", index=False)
    print(f"saved injury_flags.parquet ({len(flags)} team-weeks)")

    layer1 = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    TRAIN = {2018, 2019, 2020, 2021}
    TEST = {2022, 2023, 2024, 2025}
    train_all = layer1[layer1["season"].isin(TRAIN)]
    base_a, base_b = fit_linear(train_all["pregame_rating_diff"], train_all["actual_margin"])
    layer1["base_pred"] = base_a + base_b * layer1["pregame_rating_diff"]
    layer1["adjustment"] = apply_joint_adjustment(layer1, flags)
    layer1["adjusted_pred"] = layer1["base_pred"] + layer1["adjustment"]

    test = layer1[layer1["season"].isin(TEST)]
    mae_before = (test["base_pred"] - test["actual_margin"]).abs().mean()
    mae_after = (test["adjusted_pred"] - test["actual_margin"]).abs().mean()
    print(f"overall test MAE: before={mae_before:.4f}  after={mae_after:.4f}")

    flagged = test[~np.isclose(test["adjustment"], JOINT_COEFS["intercept"])]  # any real flag active, not just intercept
    print(
        f"flagged games (n={len(flagged)}): MAE before={(flagged['base_pred']-flagged['actual_margin']).abs().mean():.3f}"
        f"  after={(flagged['adjusted_pred']-flagged['actual_margin']).abs().mean():.3f}"
    )
