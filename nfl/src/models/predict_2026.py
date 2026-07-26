"""Week 1 2026 predictions: our own Layer 1 + QB-adjusted model, run forward
to the real 2026 schedule (extracted from Clay's PDF), compared against
ESPN's implied scores for the same games.

Honest framing: our model only knows what happened through the 2025 season.
For any team whose actual starting QB changed in the 2026 offseason (a trade,
free-agent signing, retirement, or rookie takeover), our rating is built on
the WRONG quarterback and will be unreliable until real 2026 data accumulates.
Those teams are flagged explicitly -- Clay/ESPN's number, which incorporates
real offseason information we have no other way to see, should be weighted
more heavily for them. For QB-stable teams, our independently-built model is
a legitimate second opinion.

CALIBRATION WINDOW: this script fits the margin/total regressions on 2022-2025,
NOT the 2018-2021 window used everywhere else in this project for historical
backtesting. That's deliberate, not an inconsistency: 2018-2021 was chosen for
backtesting so a disjoint 2022-2025 holdout could validate it honestly. But
2019-2020 both show badly anomalous home-field advantage (season mean margin
-0.14 and +0.05, vs. 1.7-2.7 every other year 2016-2025) -- almost certainly
the empty/reduced-crowd COVID seasons -- and 2018/2020 both run 3-4 points
higher on total scoring than 2022-2025. A calibration fit on that window
systematically underpredicts margin and overpredicts totals when applied to
a normal season, which is exactly what a per-season bias check showed: margin
bias -0.9 to -1.8 pts and total bias +0.7 to +3.0 pts in every one of
2022-2025. For an actual 2026 forecast, there's no leakage concern in using
the most recent, non-anomalous seasons instead -- 2022-2025 is what 2026 will
most resemble.
"""

import numpy as np
import pandas as pd

from src.ingest.name_matching import build_qb_name_to_id, find_qb_changes
from src.models.injury_adjustment import JOINT_COEFS_FORWARD, apply_joint_adjustment, compute_injury_flags
from src.models.qb_adjustment import QbRatingEngine, build_qb_week_table, build_starter_sequence
from src.models.ratings import PowerRatingEngine, build_dataset
from src.utils.paths import DATA_RAW, DATA_PROCESSED
from src.utils.stats import fit_linear

CALIBRATION_SEASONS = {2022, 2023, 2024, 2025}


def fit_calibration(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    return fit_linear(x, y)


if __name__ == "__main__":
    seasons = list(range(2016, 2026))
    raw = build_dataset(seasons)
    matchups = raw.rename(columns={"home_team_x": "team_a", "away_team_x": "team_b"})

    rating_engine = PowerRatingEngine()
    rated = rating_engine.run_walk_forward(matchups)
    rating_engine._maybe_new_season(2026)  # apply preseason mean-reversion for the new season

    train = rated[rated["season"].isin(CALIBRATION_SEASONS)]
    base_a, base_b = fit_calibration(train["pregame_rating_diff"], train["actual_margin"])

    clay_schedule = pd.read_parquet(DATA_RAW / "clay_2026_schedule.parquet")
    clay_players = pd.read_parquet(DATA_RAW / "clay_2026_player_projections.parquet")
    schedules = pd.read_parquet(DATA_RAW / "schedules_2016_2025.parquet")

    # total-points calibration: total_signal + is_indoor (validated in game_environment.py --
    # dome/closed-roof games score ~4 pts more; a naive total_signal-only fit here would silently
    # regress that improvement). Roof is per-stadium, so it's looked up by each 2026 HOME team's
    # most recent listed roof type.
    reg_sched = schedules[schedules["game_type"] == "REG"]
    team_roof = (
        reg_sched.sort_values(["season", "week"]).groupby("home_team").last()["roof"]
        .isin(["dome", "closed"]).astype(float).to_dict()
    )
    layer1_full = pd.read_parquet(DATA_PROCESSED / "layer1_games_with_ratings.parquet")
    layer1_full["actual_total"] = layer1_full["home_score"] + layer1_full["away_score"]
    roof_lookup = reg_sched[["game_id", "roof"]].assign(is_indoor=lambda d: d["roof"].isin(["dome", "closed"]).astype(float))
    layer1_full = layer1_full.merge(roof_lookup[["game_id", "is_indoor"]], on="game_id", how="left")
    total_train = layer1_full[layer1_full["season"].isin(CALIBRATION_SEASONS)]
    X_train = np.column_stack(
        [np.ones(len(total_train)), total_train["pregame_total_signal"], total_train["is_indoor"]]
    )
    total_coefs, *_ = np.linalg.lstsq(X_train, total_train["actual_total"].to_numpy(), rcond=None)

    week1 = clay_schedule[(clay_schedule["week"] == 1) & clay_schedule["is_home"]].copy()
    qb_changes = find_qb_changes(schedules, clay_players)
    changed_teams = set(qb_changes.loc[qb_changes["likely_change"], "team"])

    # offseason QB-swap point adjustment: same swap_b (points per unit of EPA/dropback
    # rating gap) validated for in-season swaps in qb_adjustment.py, applied here to the
    # gap between each flagged team's outgoing 2025 starter and incoming 2026 one. Most of
    # the incoming QBs (Tua, Watson, Kyler Murray, Geno Smith) are established veterans with
    # real rating histories elsewhere, so this is a real point estimate, not just a flag --
    # a true rookie with no NFL history (e.g. a Day 1 draft pick) defaults to 0.0 (league
    # average), the same cold-start assumption QbRatingEngine uses everywhere else.
    # NOTE: this script never blends our_margin with the market line (unlike
    # weekly_update.py, the live pipeline), so 6.616 -- fit against the pure
    # Layer-1 residual -- is the correct value HERE. Named _LAYER1 (review
    # round 2, #1.3) to make the residual basis explicit in the name itself,
    # since a bare "SWAP_B" shared with weekly_update.py's differently-scaled
    # constant is exactly the kind of same-name-different-meaning trap that's
    # bitten this project before (np.polyfit's arg order, three times). Do not
    # "sync" this to weekly_update.py's SWAP_B_MARKET=2.970 (refit 2026-07
    # against the market-blend residual specifically because that pipeline
    # applies it on top of a blended prediction); doing so here would apply a
    # blend-fit coefficient to an unblended prediction, a different bug. See
    # injury_adjustment.py's module docstring for the full refit rationale.
    SWAP_B_LAYER1 = 6.616
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    qb_weeks = build_qb_week_table(weekly)
    qb_starters = build_starter_sequence(schedules)
    qb_engine = QbRatingEngine()
    qb_engine.run_with_starters(qb_starters, qb_weeks)
    qb_name_to_id = build_qb_name_to_id(schedules)

    def qb_rating_by_name(name: str) -> float:
        qb_id = qb_name_to_id.get(name)
        return qb_engine.predict(qb_id) if qb_id is not None else 0.0

    team_qb_swap_delta = {}
    for row in qb_changes.itertuples():
        if row.team in changed_teams:
            team_qb_swap_delta[row.team] = qb_rating_by_name(row.projected_2026) - qb_rating_by_name(row.starter_2025)

    rows = []
    for r in week1.itertuples():
        home, away = r.team, r.opp
        home_net, away_net = rating_engine.nets(home, away)
        rating_diff = home_net - away_net
        total_signal = home_net + away_net
        home_qb_swap = team_qb_swap_delta.get(home, 0.0)
        away_qb_swap = team_qb_swap_delta.get(away, 0.0)
        our_margin = base_a + base_b * rating_diff + SWAP_B_LAYER1 * (home_qb_swap - away_qb_swap)
        is_indoor = team_roof.get(home, 0.0)
        our_total = total_coefs[0] + total_coefs[1] * total_signal + total_coefs[2] * is_indoor
        our_home_pts = (our_total + our_margin) / 2
        our_away_pts = (our_total - our_margin) / 2
        flags = []
        if home in changed_teams:
            flags.append(f"{home} QB change")
        if away in changed_teams:
            flags.append(f"{away} QB change")
        rows.append(
            {
                "season": 2026,
                "week": 1,
                "team_a": home,
                "team_b": away,
                "home": home,
                "away": away,
                "our_margin_pre_injury": our_margin,
                "our_total": our_total,
                "is_indoor": is_indoor,
                "clay_home_pts": r.team_pts,
                "clay_away_pts": r.opp_pts,
                "clay_total": round(r.team_pts + r.opp_pts, 1),
                "clay_margin": round(r.team_pts - r.opp_pts, 1),
                "clay_win_prob_home": r.win_prob,
                "flags": ", ".join(flags) if flags else "",
            }
        )

    result = pd.DataFrame(rows)

    # injury adjustment: 2026 has no in-season injury reports yet (they don't exist
    # pre-season), so this mostly no-ops for week 1 -- it's here so the same code
    # path applies automatically once real 2026 injury data starts flowing in-season.
    injury_flags = compute_injury_flags()
    result["injury_adjustment"] = apply_joint_adjustment(result, injury_flags, coefs=JOINT_COEFS_FORWARD)
    result["our_margin"] = (result["our_margin_pre_injury"] + result["injury_adjustment"]).round(1)
    result["our_home_pts"] = ((result["our_total"] + result["our_margin"]) / 2).round(1)
    result["our_away_pts"] = ((result["our_total"] - result["our_margin"]) / 2).round(1)
    result["our_total"] = result["our_total"].round(1)
    result["total_diff"] = (result["our_total"] - result["clay_total"]).round(1)

    result = result.sort_values("home")[
        ["home", "away", "our_home_pts", "our_away_pts", "our_margin", "our_total",
         "clay_home_pts", "clay_away_pts", "clay_margin", "clay_total", "total_diff",
         "clay_win_prob_home", "flags"]
    ]
    pd.set_option("display.width", 200)
    print("=== WEEK 1, 2026: our model vs ESPN (Clay) ===\n")
    print(result.to_string(index=False))

    print(f"\n--- TOTALS check ---")
    print(f"our total calibration: total = {total_coefs[0]:.2f} + {total_coefs[1]:.2f}*total_signal + {total_coefs[2]:.2f}*is_indoor")
    print(f"our avg total: {result['our_total'].mean():.1f}   Clay avg total: {result['clay_total'].mean():.1f}")
    print(f"mean |our_total - clay_total|: {result['total_diff'].abs().mean():.2f} pts")
    print(f"biggest total disagreements:")
    print(result.reindex(result['total_diff'].abs().sort_values(ascending=False).index)[["home","away","our_total","clay_total","total_diff"]].head(5).to_string(index=False))

    print(f"\n{len(changed_teams)} teams flagged with an offseason QB change our model can't see:")
    print(", ".join(sorted(changed_teams)))
    print("\nfor those teams' games, weight Clay's projection more heavily than ours.")

    result.to_parquet(DATA_PROCESSED / "week1_2026_predictions.parquet", index=False)
    qb_changes.to_parquet(DATA_PROCESSED / "qb_changes_2026.parquet", index=False)
    print("\nsaved week1_2026_predictions.parquet, qb_changes_2026.parquet")
