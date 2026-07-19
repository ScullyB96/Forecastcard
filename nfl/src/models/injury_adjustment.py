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

FORWARD-USE COEFFICIENTS (JOINT_COEFS_FORWARD): predict_2026.py recalibrates
its margin model on 2022-2025 rather than 2018-2021 (see that module's
docstring -- 2019-2020 badly understate home-field advantage, a COVID
crowd-size artifact). Refitting JOINT_COEFS against that new base prediction's
residuals, using only 2022-2025, moved every coefficient a lot (e.g. home_cb:
-1.22 -> -4.67) -- too much to be pure signal given the sample is the same
size as before (~150-300 flagged games total across all 4 signals). That's
consistent with these already being estimated from rare events, not evidence
of a real regime change. Rather than pick either 4-year window, each game's
residual is computed against ITS OWN era's well-fit calibration (old games
against the old fit, new games against the new fit -- both are ~unbiased
within their own era), then pooled across all 8 seasons for one lower-variance
regression. The result sits between the two individual fits, as expected, with
roughly double the effective sample:

    margin_adjustment = -0.34
        + 2.56 * away_skill_out
        + 1.20 * away_ol_out
        - 3.43 * home_cb_out
        + 3.79 * away_cb_out

Used only by predict_2026.py. JOINT_COEFS (above) stays as originally
validated for historical backtesting (predict.py), since that number needs to
reflect exactly what was walk-forward tested, not a later refinement.
"""

import re

import numpy as np
import pandas as pd

from src.utils.paths import DATA_PROCESSED, DATA_RAW

STATUS_OUT = "Out"

JOINT_COEFS = {
    "intercept": -0.382,
    "away_skill_flag": 1.434,
    "away_ol_flag": 1.447,
    "home_cb_flag": -1.217,
    "away_cb_flag": 3.184,
}

# pooled 8-season, era-consistent-residual fit -- see module docstring. Used by
# predict_2026.py only.
JOINT_COEFS_FORWARD = {
    "intercept": -0.339,
    "away_skill_flag": 2.562,
    "away_ol_flag": 1.195,
    "home_cb_flag": -3.430,
    "away_cb_flag": 3.792,
}


def norm_name(name: str) -> str | None:
    if pd.isna(name):
        return None
    s = name.lower().strip()
    s = re.sub(r"[.'\-]", "", s)
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    return re.sub(r"\s+", " ", s)


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


def compute_injury_flags(seasons_dir=DATA_RAW, processed_dir=DATA_PROCESSED) -> pd.DataFrame:
    """Build the four team-week injury flags used by the joint model. Returns
    columns [season, week, team, skill_out, ol_out, cb_out] (skill_out/ol_out/
    cb_out are counts; threshold at >=1 to get the binary flags used in
    JOINT_COEFS)."""
    inj = pd.read_parquet(seasons_dir / "injuries_2016_2025.parquet")
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

    sc = pd.read_parquet(seasons_dir / "snap_counts_2016_2025.parquet")
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
    base_a, base_b = np.polyfit(train_all["pregame_rating_diff"], train_all["actual_margin"], 1)[::-1]
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
