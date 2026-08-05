"""Does the park-geometry HR/XBH factor (built for the ORACLE backtest,
tasks #189-191, real-but-full-stack-immaterial on SU/Brier -- see
MODEL_DOCUMENTATION.md sec 11.44) move anything on the actual PROP
probabilities it's conceptually aimed at (p_1plus_hr, hits), now that it's
wired into props.py's live/predictive path (2026-08-05)?

A game-level SU/Brier null doesn't necessarily mean a prop-level null --
the effect could be real and material for a specific batter's specific
prop line even while washing out in the aggregate score/win outcome. This
is the direct test of that, using validate_prop_calibration.py's own
existing calibration_report machinery -- OFF vs ON, same real games/seasons,
same trial count.
"""
import pandas as pd

from src.models.validate_prop_calibration import collect_prop_predictions, report_calibration
from src.utils.paths import DATA_PROCESSED

N_GAMES = 300
N_TRIALS = 150
SEASONS = [2024, 2025]


def main():
    pa = pd.read_parquet(DATA_PROCESSED / "pa_table_2023_2026.parquet")

    print("=== OFF (geometry factors disabled -- current production behavior) ===", flush=True)
    off = collect_prop_predictions(pa, seasons=SEASONS, n_games=N_GAMES, n_trials=N_TRIALS,
                                    geometry_hr_enabled=False, geometry_xbh_enabled=False)
    print("\n=== ON (geometry_hr_enabled=True, geometry_xbh_enabled=True) ===", flush=True)
    on = collect_prop_predictions(pa, seasons=SEASONS, n_games=N_GAMES, n_trials=N_TRIALS,
                                   geometry_hr_enabled=True, geometry_xbh_enabled=True)

    print("\n\n" + "=" * 30 + " p_1plus_hr (HR prop -- geometry_hr_factor's direct target) " + "=" * 30)
    report_calibration(off["batter_props"], "p_1plus_hr", "actual_1plus_hr", "OFF")
    report_calibration(on["batter_props"], "p_1plus_hr", "actual_1plus_hr", "ON")

    print("\n\n" + "=" * 30 + " p_2plus_hits (closest existing prop to a TB/XBH signal) " + "=" * 30)
    report_calibration(off["batter_props"], "p_2plus_hits", "actual_2plus_hits", "OFF")
    report_calibration(on["batter_props"], "p_2plus_hits", "actual_2plus_hits", "ON")

    print("\n\n" + "=" * 30 + " mean_hits (continuous MAE check) " + "=" * 30)
    for label, res in [("OFF", off), ("ON", on)]:
        bp = res["batter_props"]
        if "mean_hits" in bp.columns and "hits" in bp.columns:
            mae = (bp["mean_hits"] - bp["hits"]).abs().mean()
            corr = bp["mean_hits"].corr(bp["hits"])
            print(f"{label}: mean_hits MAE={mae:.4f}  corr={corr:.4f}  n={len(bp)}")


if __name__ == "__main__":
    main()
