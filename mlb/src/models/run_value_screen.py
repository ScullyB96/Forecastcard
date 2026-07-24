"""Stage-0 pre-simulation screen for candidate rate-multiplier signals
(2026-07-22, per an external reviewer's suggested cost-reduction funnel).

Before spending a ~50-minute-plus Monte Carlo backtest run on a candidate
signal, estimate its plausible impact directly from linear weights: convert
the candidate's before/after per-category rate shift into an expected-runs
shift using published run values, aggregate to a per-game expected-margin
shift over REAL historical PAs (not hypothetical average PA counts), and
report the distribution across games. If the overwhelming majority of games
show a negligible shift, the candidate cannot plausibly move straight-up
accuracy and can be killed without simulating a single trial -- this is
deliberately an upper-bound estimate, not a substitute for the full simulator
(it ignores the renormalization-driven redistribution across OTHER outcome
categories that `combine_matchup_distribution` performs, and ignores every
context factor -- park, weather, platoon, etc. -- entirely). Use it to REJECT
candidates cheaply, not to CONFIRM one; a candidate that passes this screen
still needs the full backtest.
"""

import numpy as np
import pandas as pd

# Standard published linear weights (run value of each outcome relative to
# average, per Tango/Lichtman/Dolphin's "The Book" -- the same source this
# project already cites for its odds-ratio matchup methodology). The core six
# (walk/HBP/1B/2B/3B/HR) are the well-established figures; the remaining
# categories are reasonable, clearly-flagged approximations (this project's
# own established convention for anything not independently sourced --
# see true_talent.py's own "REASONABLE STARTING VALUES, not yet empirically
# re-validated" precedent) since no single canonical run-value figure exists
# for e.g. triple_play or catcher_interf.
LINEAR_WEIGHTS = {
    "strikeout": -0.30,       # a strikeout forecloses productive-out advancement entirely,
                              # slightly worse than a generic ball-in-play out
    "field_out": -0.27,       # standard generic contact-out run value
    "walk": 0.32,
    "intent_walk": 0.32,      # same base-running run value as a regular walk
    "hit_by_pitch": 0.34,
    "single": 0.47,
    "double": 0.77,
    "triple": 1.04,
    "home_run": 1.40,
    "double_play": -0.50,     # worse than one out -- removes two outs on one play
    "sac_fly": -0.15,         # gives up an out; often scores a run situationally, but
                              # averaged across contexts this is a net modest cost
    "sac_bunt": -0.15,        # same approximation as sac_fly -- an out given up for advancement
    "fielders_choice": -0.27, # a net out occurs even though the batter reaches -- treated
                              # like a generic out for the team's overall run-value purposes
    "field_error": 0.47,      # batter reaches base; treated like a single for the
                              # BATTING team's run-value benefit (the error is the defense's cost)
    "catcher_interf": 0.32,   # batter awarded first base -- same run value as a walk
    "triple_play": -0.75,     # three outs on one play, roughly 3x a generic out (bounded,
                              # not literally additive, but this is a rare-enough category
                              # that the exact figure barely matters)
}


def run_value(rates: dict[str, float]) -> float:
    """Expected run value of one PA's outcome distribution, in runs relative
    to a league-average PA (0 by construction if `rates` exactly matched the
    league mean for every category and the weights were perfectly calibrated
    to sum to zero at that mean -- in practice a small, ignorable offset)."""
    return sum(rates.get(o, 0.0) * w for o, w in LINEAR_WEIGHTS.items())


def screen_impact(pa: pd.DataFrame, rates_before_col: str, rates_after_col: str,
                   outcome: str, group_cols: list[str] = ("game_pk",)) -> pd.DataFrame:
    """Given a PA-level table with two existing columns holding a single
    outcome category's BEFORE and AFTER probability (e.g. the pitcher's own
    home_run rate before/after a candidate multiplier), compute the per-PA
    expected-run delta via LINEAR_WEIGHTS[outcome], then aggregate to a
    per-group (default: per-game) total expected-run shift.

    This is a single-CATEGORY screen (matches how every multiplier in this
    project actually operates -- one category's rate is rescaled, then
    `combine_matchup_distribution` renormalizes the whole vector afterward).
    Ignoring that renormalization's second-order redistribution into OTHER
    categories is deliberate -- it would require running the actual odds-ratio
    combine, which defeats the purpose of a cheap pre-simulation filter."""
    delta_per_pa = (pa[rates_after_col] - pa[rates_before_col]) * LINEAR_WEIGHTS[outcome]
    tmp = pa[list(group_cols)].copy()
    tmp["run_value_delta"] = delta_per_pa
    return tmp.groupby(list(group_cols))["run_value_delta"].sum().reset_index()


def summarize_screen(per_game_delta: pd.Series, materiality_threshold: float = 0.02) -> dict:
    """Plain-English summary of a screen_impact() result: how many games
    would plausibly see a real margin shift, and how big is the tail."""
    abs_delta = per_game_delta.abs()
    return {
        "n_games": len(per_game_delta),
        "median_abs_runs": abs_delta.median(),
        "p90_abs_runs": abs_delta.quantile(0.90),
        "p99_abs_runs": abs_delta.quantile(0.99),
        "max_abs_runs": abs_delta.max(),
        "frac_above_threshold": (abs_delta > materiality_threshold).mean(),
        "verdict": (
            "LIKELY CANNOT MOVE SU -- kill without simulating"
            if (abs_delta > materiality_threshold).mean() < 0.05
            else "passes screen -- worth a full backtest"
        ),
    }


if __name__ == "__main__":
    from src.models.expected_stats import pitcher_hr_allowed_multiplier, player_game_gb_fb_rate_snapshot
    from src.models.true_talent import build_debut_rate, build_pregame_rates
    from src.utils.paths import DATA_PROCESSED

    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")
    HIT_EVENTS = {"single", "double", "triple", "home_run"}
    BIP_OUT_EVENTS = {"field_out", "double_play", "fielders_choice", "field_error",
                      "sac_fly", "sac_bunt", "triple_play", "catcher_interf"}
    bip_events = list(HIT_EVENTS | BIP_OUT_EVENTS)

    print("=== POSITIVE CONTROL: GB/FB pitcher HR-share (confirmed real, +1.53pp SU at n=7237) ===")
    pregame_hr = build_pregame_rates(pa, "pitcher", "home_run")[
        ["game_pk", "at_bat_number", "pitcher", "pregame_rate"]
    ].rename(columns={"pregame_rate": "pitcher_hr_rate"})
    gbfb_snap_full = player_game_gb_fb_rate_snapshot(pa)  # per-game snapshot; need PA-level for this screen
    # build_pitcher_gb_fb_rate_by_season gives a PA-level (not just per-game-first) rate --
    # reuse it directly for a proper per-PA screen instead of the once-per-game snapshot.
    from src.models.expected_stats import build_pitcher_gb_fb_rate_by_season
    gbfb_pa = build_pitcher_gb_fb_rate_by_season(pa)[
        ["game_pk", "at_bat_number", "pitcher", "pregame_gb_rate_pitcher", "pregame_fb_rate_pitcher"]
    ]
    merged = pa[["game_pk", "at_bat_number", "pitcher", "outcome"]].merge(
        pregame_hr, on=["game_pk", "at_bat_number", "pitcher"], how="inner"
    ).merge(gbfb_pa, on=["game_pk", "at_bat_number", "pitcher"], how="inner")

    # approximate a full rates dict per PA using just league-average-ish placeholders for the
    # other BIP categories (the multiplier only needs the BIP total, not each category exactly,
    # to compute existing_share -- close enough for a screen, not the real simulator).
    league_bip_rate = pa["outcome"].isin(bip_events).mean()
    merged["bip_rate_approx"] = league_bip_rate

    def _hr_after(row):
        rates = {"home_run": row["pitcher_hr_rate"]}
        for o in bip_events:
            if o != "home_run":
                rates[o] = 0.0
        rates["home_run_bip_total_placeholder"] = row["bip_rate_approx"]
        # build a minimal rates dict where sum(bip_events) == bip_rate_approx and
        # home_run == pitcher_hr_rate (the only two quantities pitcher_hr_allowed_multiplier reads)
        fake_rates = {o: 0.0 for o in bip_events}
        fake_rates["home_run"] = row["pitcher_hr_rate"]
        remaining = row["bip_rate_approx"] - row["pitcher_hr_rate"]
        other = [o for o in bip_events if o != "home_run"]
        for o in other:
            fake_rates[o] = max(remaining, 0.0) / len(other)
        mult = pitcher_hr_allowed_multiplier(
            fake_rates, row["pregame_gb_rate_pitcher"], row["pregame_fb_rate_pitcher"]
        )
        return row["pitcher_hr_rate"] * mult

    merged["pitcher_hr_rate_after"] = merged.apply(_hr_after, axis=1)
    game_impact = screen_impact(
        merged.rename(columns={"pitcher_hr_rate": "hr_before", "pitcher_hr_rate_after": "hr_after"}),
        "hr_before", "hr_after", "home_run"
    )
    print(summarize_screen(game_impact["run_value_delta"]))

    print("\n=== NEGATIVE CONTROL: rookie/debut prior (confirmed NULL, CI includes zero at n=7237) ===")
    league_rate_hr = (pa["outcome"] == "home_run").mean()
    debut_rate_hr = build_debut_rate(pa, "batter", "home_run", 2026, league_rate_hr)
    print(f"league_rate={league_rate_hr:.4f} debut_rate={debut_rate_hr:.4f} "
          f"per-PA run-value delta for a rookie: {(debut_rate_hr - league_rate_hr) * LINEAR_WEIGHTS['home_run']:+.5f} runs")
    print("(rookie prior affects ~15 outcome categories at once per rookie-PA, not just home_run --"
          " a full per-PA screen would sum all 15 category deltas; home_run alone already shows"
          " this is a small, sub-hundredth-of-a-run-per-PA effect, consistent with the confirmed null)")
