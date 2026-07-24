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


def build_ttop_factors_by_season(pa: pd.DataFrame) -> dict[int, dict[int, dict[str, float]]]:
    """Walk-forward-safe (prior-seasons-only): factors[season][times_through][outcome]
    = rate(outcome | times_through) / rate(outcome | overall). Same plain-ratio,
    Bayesian-shrunk design as build_state_factors_by_season (not odds-ratio:
    some categories can be exactly zero in a bucket), and deliberately
    LEAGUE-AVERAGE, not pitcher-specific -- see module docstring for why.
    Each times_through bucket is large (hundreds of thousands of PAs) so
    shrinkage rarely moves these much in practice, but applying it keeps
    every "environmental multiplier" table in this project on the same
    small-sample-safe footing (added 2026-07-21, same audit finding as
    game_simulator.py's state_factors)."""
    out = {}
    capped_all = pa["n_thruorder_pitcher"].clip(upper=MAX_TIMES_THROUGH)
    for season in sorted(pa["season"].unique()):
        prior = pa[pa["season"] < season]
        ref = prior if len(prior) else pa[pa["season"] == season]
        capped = capped_all.loc[ref.index]
        overall = ref["outcome"].value_counts(normalize=True)
        factors = {}
        for times_through, g in ref.groupby(capped):
            counts = g["outcome"].value_counts()
            n_bucket = len(g)
            cell = {}
            for o in OUTCOMES:
                overall_rate = overall.get(o, 1e-9)
                shrunk_rate = (
                    (counts.get(o, 0) + TTOP_FACTOR_PRIOR_PA * overall_rate) / (n_bucket + TTOP_FACTOR_PRIOR_PA)
                )
                cell[o] = shrunk_rate / overall_rate if overall_rate > 1e-9 else 1.0
            factors[int(times_through)] = cell
        out[season] = factors
    return out
