"""Measures the platoon structural fix's shared (population-level) same/opp-
hand displacement directly from data, instead of assuming or fitting an
exponent against a holdout (2026-08-04, follow-up to the 2026-08-03
fallback-only fix and its 2026-08-04 observed-leg follow-up).

The estimand: for each (batter-hand, pitcher-hand) cell and outcome, the
league-wide odds ratio of ACTUAL outcome rate vs. a PLATOON-DISABLED
prediction (matchup-combine x state x park x TTO, no platoon term at all),
using only PRIOR seasons' data (walk-forward, no leakage). This directly
measures "how much should the platoon adjustment move this cell," in the
model's own native currency, with the real exposure composition and any
real selection effects (e.g. lefty relievers being deployed specifically
against tough same-handed batters) baked in by construction -- not
approximated by an assumed or swept exponent.

Kept as a SEPARATE script/module from platoon_splits.py rather than computed
inline there, because the pieces needed (state/park/TTO factors, batter/
pitcher pregame rates) live in game_simulator.py and true_talent.py, and
game_simulator.py itself imports build_platoon_multipliers FROM
platoon_splits.py -- importing back would be circular. This mirrors the
project's existing "offseason fitted-constants refresh ritual" (task #155)
pattern for exactly this kind of measured, periodically-refreshed constant:
run this script, it persists a small JSON, platoon_splits.py loads it lazily
at runtime with a documented fallback for anything missing.

IMPORTANT -- this is a measurement of the residual structure GIVEN THE REST
OF THE STACK AS OF MEASUREMENT TIME, not an isolated platoon-only quantity:
the "platoon-disabled prediction" baseline still includes state/park/TTO
live, so each measured cell already absorbs the AVERAGE of any OTHER
handedness-correlated effect in the stack (e.g. the platoon x times-through-
the-order interaction documented as an open item in MODEL_REVIEW_RESPONSE.md
-- strikeout-specific, real, not yet built). If a future factor is added
that itself correlates with batter/pitcher handedness, this measurement
MUST be re-run with that factor active BEFORE the new factor ships --
otherwise the handedness-correlated share gets counted in both places, the
exact double-counting failure mode this whole file exists to prevent,
reintroduced one level up. Re-run on the general offseason cadence (task
#155) too, but don't treat that as sufficient on its own: also confirmed via
a 3-season stability check that the measured values aren't just noise, they
show a real, more-than-noise MONOTONIC drift (strikeout's LHB-vs-LHP cell:
1.116 in 2024, 1.089 in 2025, 1.063 in 2026) -- plausibly real drift in
matchup curation, plausibly the 2026 partial season, but either way this
means the refresh is load-bearing, not routine housekeeping to deprioritize."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.matchup import odds
from src.models.props import OUTCOMES
from src.models.validate_game_simulator import build_shared_tables

# Must match platoon_splits.MEASURED_SHARED_TERM_PATH -- lives alongside the
# source (NOT data/processed/, which is gitignored) so it ships with the code
# and is available immediately in production. See that module's own comment
# for why: production rebuilds pa_table/state/park/ttop fresh from raw data
# on every full run, but this measurement refreshes on the offseason cadence
# (task #155), not nightly -- it needs to persist across deploys on its own.
OUTPUT_PATH = Path(__file__).parent / "platoon_shared_term.json"


def measure_for_season(pa: pd.DataFrame, shared: dict, season: int) -> dict:
    """{outcome: {"LL": ratio, "LR": ratio, "RL": ratio, "RR": ratio}} for
    one season, using ONLY that season's own real PAs (the walk-forward
    safety lives in state/park/ttop/league_rates/batter_snap/pitcher_snap
    themselves, all built from strictly-prior seasons by their own
    functions -- see build_shared_tables)."""
    pa_season = pa[pa["season"] == season]
    cols = ["batter", "pitcher", "game_pk", "home_team", "stand", "p_throws",
            "pre_state", "n_thruorder_pitcher", "outcome"]
    df = pa_season[cols].copy()
    df["times_through"] = df["n_thruorder_pitcher"].clip(upper=3)

    batter_snap = shared["batter_snap"]
    pitcher_snap = shared["pitcher_snap"]
    batter_snap = batter_snap[batter_snap["season"] == season][
        ["batter", "game_pk"] + [f"pregame_rate_{o}" for o in OUTCOMES]
    ].rename(columns={f"pregame_rate_{o}": f"batter_rate_{o}" for o in OUTCOMES})
    pitcher_snap = pitcher_snap[pitcher_snap["season"] == season][
        ["pitcher", "game_pk"] + [f"pregame_rate_{o}" for o in OUTCOMES]
    ].rename(columns={f"pregame_rate_{o}": f"pitcher_rate_{o}" for o in OUTCOMES})
    df = df.merge(batter_snap, on=["batter", "game_pk"], how="inner")
    df = df.merge(pitcher_snap, on=["pitcher", "game_pk"], how="inner")
    if df.empty:
        return {}

    league_rates = shared["league_rates"].get(season, {})
    state_factors = shared["state_factors"].get(season, {})
    ttop_factors = shared["ttop_factors"].get(season, {})
    park_wide = shared["park_factors_wide"]

    df["state_row"] = df["pre_state"].map(state_factors)
    df["ttop_row"] = df["times_through"].map(ttop_factors)

    result = {}
    for outcome in OUTCOMES:
        lg = league_rates.get(outcome, np.nan)
        if not (0 < lg < 1):
            continue
        b = pd.to_numeric(df[f"batter_rate_{outcome}"], errors="coerce").clip(1e-6, 1 - 1e-6)
        p = pd.to_numeric(df[f"pitcher_rate_{outcome}"], errors="coerce").clip(1e-6, 1 - 1e-6)
        m_odds = odds(lg) * (odds(b) / odds(lg)) * (odds(p) / odds(lg))
        p0 = (m_odds / (1 + m_odds)).astype(float).to_numpy(dtype=np.float64)

        state_v = df["state_row"].apply(lambda c: c.get(outcome, 1.0) if isinstance(c, dict) else 1.0).to_numpy(dtype=np.float64)
        ttop_v = df["ttop_row"].apply(lambda c: c.get(outcome, 1.0) if isinstance(c, dict) else 1.0).to_numpy(dtype=np.float64)
        if outcome in park_wide.columns:
            park_v = np.array(
                [park_wide.loc[(t, season), outcome] if (t, season) in park_wide.index else 1.0 for t in df["home_team"]],
                dtype=np.float64,
            )
        else:
            park_v = np.ones(len(df))

        pred_no_platoon = p0 * state_v * park_v * ttop_v
        actual = (df["outcome"] == outcome).to_numpy(dtype=np.float64)

        cells = {}
        for stand in ["L", "R"]:
            for p_throws in ["L", "R"]:
                mask = (df["stand"] == stand).to_numpy() & (df["p_throws"] == p_throws).to_numpy()
                if mask.sum() < 200:  # too few real PAs this season for a stable cell measurement
                    continue
                actual_rate = actual[mask].mean()
                pred_rate = pred_no_platoon[mask].mean()
                cells[f"{stand}{p_throws}"] = float(odds(actual_rate) / odds(pred_rate))
        if cells:
            result[outcome] = cells
    return result


if __name__ == "__main__":
    from src.utils.paths import DATA_PROCESSED as _DP

    pa = pd.read_parquet(_DP / "pa_table_2023_2026.parquet")
    seasons = sorted(pa["season"].unique())
    test_seasons = set(seasons)

    print(f"building shared tables once for seasons {seasons}...", flush=True)
    shared = build_shared_tables(pa, test_seasons)
    print("done.\n", flush=True)

    # "_meta" is a real key, not a comment -- JSON has no comment syntax, and this
    # warning needs to survive in the data file itself, not just this script's own
    # docstring, since a future editor may open/regenerate the JSON without reading
    # the script. platoon_splits.py's loader only ever looks up str(season) keys, so
    # this key is silently ignored by every real consumer.
    out = {
        "_meta": (
            "Measures the residual structure GIVEN THE REST OF THE STACK AS OF "
            "MEASUREMENT TIME (platoon-disabled baseline still includes state/park/TTO "
            "live) -- re-run this script BEFORE shipping any new handedness-correlated "
            "factor (e.g. a platoon x times-through-the-order joint factor), not just on "
            "the offseason cadence, or the handedness-correlated share gets counted "
            "twice. Also shows a real, more-than-noise MONOTONIC drift across seasons "
            "(strikeout LHB-vs-LHP: 1.116 in 2024, 1.089 in 2025, 1.063 in 2026) -- the "
            "refresh is load-bearing, not routine housekeeping. See "
            "measure_platoon_shared_term.py's own module docstring for the full context."
        ),
    }
    for season in seasons:
        print(f"measuring season {season}...", flush=True)
        measured = measure_for_season(pa, shared, season)
        if measured:
            out[str(season)] = measured

    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved measured shared-term table to {OUTPUT_PATH}")
    print(json.dumps(out, indent=2))
