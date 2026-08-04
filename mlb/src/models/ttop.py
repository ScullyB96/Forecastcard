"""Times-through-the-order penalty (TTOP): a pitcher gets measurably less
effective each time he faces the same lineup again in one game, independent
of pitch count/fatigue -- confirmed by published research (Wharton/Baseball
Prospectus/SABR) to be driven by hitter FAMILIARITY, not tiredness, and to
show near-zero pitcher-specific predictive signal (r~=0.03), meaning a
LEAGUE-AVERAGE magnitude is the correct choice here, not a per-pitcher one.

Confirmed directly in our own data before building this: strikeout rate
falls each pass (23.6% -> 20.8% -> 19.6%), while contact-quality outcomes
rise (home_run 2.9% -> 3.3% -> 3.5%, single 13.9% -> 14.5% -> 14.9%) --
exactly the published direction and shape. Statcast's own `n_thruorder_pitcher`
column (added to build_pa_table.py's PA_COLUMNS for this) already labels
every PA with which pass through the order it represents -- no separate
tracking needed to build the factor table, only to APPLY it during
simulation (see game_simulator.py, which tracks how many times each lineup
slot has faced the CURRENT pitcher so far this game).

Capped at 3 (times_through >= 3 all bucketed together): the penalty is
documented to flatten out after the 3rd pass, and a 4th/5th pass is
vanishingly rare in our own data (1748 and 1 PA respectively across four
full seasons) -- not enough samples to estimate a distinct factor anyway.
"""

import pandas as pd

from src.models.game_simulator import OUTCOMES

MAX_TIMES_THROUGH = 3
TTOP_FACTOR_PRIOR_PA = 5000  # same shrinkage fix as build_state_factors_by_season -- see its docstring
TTOP_FACTOR_CLIP_MIN, TTOP_FACTOR_CLIP_MAX = 0.03, 20.0  # task #160 (2026-07-26 correctness
                                                           # audit): this table had no clip at all,
                                                           # unlike every sibling factor table
                                                           # (weather.py's WEATHER_FACTOR_CLIP_MIN/MAX,
                                                           # park_factors.py's renormalization) --
                                                           # real current values (0.041x-3.81x) sit
                                                           # comfortably inside this band, so it's
                                                           # pure insurance against a future data
                                                           # update sparsening a cell further, not a
                                                           # change to today's real factors.


def build_ttop_factors_by_season(pa: pd.DataFrame) -> dict[int, dict[int, dict[str, float]]]:
    """Walk-forward-safe (prior-seasons-only): factors[season][times_through][outcome],
    fit WITHIN-PITCHER on STARTER PAs only (2026-08-03 audit, finding M4).

    The previous pooled estimator (rate(o | TT) / rate(o | overall), all PAs)
    conflated three things: (1) COMPOSITION -- 56.4% of TT1 PAs are thrown by
    relievers, whose rate mix differs from starters', so every reliever got
    e.g. x1.055 strikeout at TT1 on top of profile rates that are already
    reliever-high; decomposition on the 2023-24 window showed the pooled TT2
    home_run bump (+7.5%) is ~100% composition; (2) day-level SURVIVOR
    SELECTION -- reaching TT3 requires having a good day, so pooled TT3
    "improvements" (the -5.8% walk reduction is entirely this) are selection,
    not familiarity, and the hook-frailty mechanism (task #144/#145) already
    models that same day-level selection explicitly; (3) the real
    FAMILIARITY effect the published TTOP literature describes. Only (3)
    belongs in this multiplier.

    New estimator, per (times_through, outcome):
        factor = observed events / EXPECTED events,
    where expected sums each PA's own pitcher's overall starter-PA rate in
    the fit window -- every pitcher is compared to HIMSELF, so pitcher-mix
    composition cancels exactly. Fit on starter PAs only (the pitcher who
    threw his team's first PA of the game), then renormalized so the
    starter-PA-mix-weighted mean factor is exactly 1.0 per outcome: a
    starter's total expected rates are preserved and the factor only
    REDISTRIBUTES across passes. Same Bayesian shrinkage toward 1 and clip
    as before (rare outcomes). Relievers should get NO factor at all --
    game_simulator.simulate_half_inning skips TTOP for profiles stamped
    apply_ttop=False (see the reliever-profile build sites in props.py /
    validate_game_simulator.py).

    Residual caveat (deliberate, documented): a within-pitcher expected
    baseline removes composition but only part of day-level selection (TT3
    PAs still come disproportionately from good days, which biases the
    familiarity penalty CONSERVATIVELY, toward 1) -- the remaining overlap
    with the hook mechanism is a Phase D re-baseline question, not fixable
    by this table alone."""
    out = {}
    capped_all = pa["n_thruorder_pitcher"].clip(upper=MAX_TIMES_THROUGH)
    # starter = the pitcher of each (game, batting-side)'s first PA
    first_pa = pa.sort_values("at_bat_number").drop_duplicates(["game_pk", "inning_topbot"])
    starter_keys = set(zip(first_pa["game_pk"], first_pa["inning_topbot"], first_pa["pitcher"]))
    is_starter_pa = pd.Series(
        list(zip(pa["game_pk"], pa["inning_topbot"], pa["pitcher"])), index=pa.index
    ).isin(starter_keys)

    for season in sorted(pa["season"].unique()):
        prior_mask = pa["season"] < season
        if not prior_mask.any():
            # task #160 (2026-07-26 correctness audit) -- same real look-
            # ahead leak and same fix as catcher_framing.py/weather.py/
            # umpire_factor.py's identical cold-start pattern: the true
            # first season previously fell back to ITS OWN full data.
            out[season] = {}
            continue
        ref = pa[prior_mask & is_starter_pa]
        capped = capped_all.loc[ref.index]

        # each pitcher's own overall starter-PA outcome rates in the window
        pitcher_rates = (
            pd.crosstab(ref["pitcher"], ref["outcome"], normalize="index")
            .reindex(columns=OUTCOMES, fill_value=0.0)
        )
        expected_per_pa = pitcher_rates.reindex(ref["pitcher"]).to_numpy()  # (n_pa, n_outcomes)
        expected_df = pd.DataFrame(expected_per_pa, index=ref.index, columns=OUTCOMES)

        pooled = ref["outcome"].value_counts(normalize=True)
        factors = {}
        for times_through, g in ref.groupby(capped):
            observed = g["outcome"].value_counts()
            expected = expected_df.loc[g.index].sum()
            cell = {}
            for o in OUTCOMES:
                pool_rate = pooled.get(o, 1e-9)
                # shrink the observed/expected ratio toward 1 with
                # TTOP_FACTOR_PRIOR_PA pseudo-PA at the pooled rate --
                # same small-sample discipline as before, adapted to the
                # observed-vs-expected form
                num = observed.get(o, 0) + TTOP_FACTOR_PRIOR_PA * pool_rate
                den = expected[o] + TTOP_FACTOR_PRIOR_PA * pool_rate
                factor = num / den if den > 1e-9 else 1.0
                cell[o] = max(TTOP_FACTOR_CLIP_MIN, min(TTOP_FACTOR_CLIP_MAX, factor))
            factors[int(times_through)] = cell
        # renormalize to the starter PA mix so E_mix[factor] == 1 per outcome
        mix = capped.value_counts(normalize=True)
        for o in OUTCOMES:
            norm = sum(mix.get(tt, 0.0) * cell_f[o] for tt, cell_f in factors.items())
            if norm > 1e-9:
                for cell_f in factors.values():
                    cell_f[o] = cell_f[o] / norm
        out[season] = factors
    return out
