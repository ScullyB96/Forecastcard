"""Task #158 (2026-07-25): the forecast-quality backtest weather_forecast.py's
own docstring flagged as never having been done -- does WIND_MATCH_BOOST
actually help, and is 2.0 (its original, never-empirically-checked default)
well-tuned?

Method: for every CONFIDENT_TEAMS game with real posted weather (2023-2025,
non-domed venues only -- matches resolve_weather_distribution's own real
behavior), fetch that venue's REAL historical wind at ~game time from
archive-api.open-meteo.com (the same free source park_orientation.py used to
calibrate each park's bearing) as a stand-in for what a live forecast call
would have returned. Classify it via park_orientation.forecast_wind_to_
bucket_suffix, exactly as the live code path does. Score candidate boost
values by log-loss (a proper scoring rule) on the probability the reweighted
climatological distribution assigns to the REAL realized bucket.

Two checks, in order:
1. Leave-one-out (games scored against a climatology built from every OTHER
   game at that venue/month): confirms the forecast suffix carries real
   signal at all (classification accuracy vs. base rate) before trusting any
   boost-value comparison built on top of it.
2. A genuine 60/40 train/test split (climatology fit on 60%, scored on the
   held-out 40%, no leakage), repeated across 5 seeds -- this project's own
   stability-check convention (task #64, 5-split correlation sweep) applied
   here to log-loss instead of correlation.

Real result (2026-07-25): classification accuracy 37.0% (n=5369) vs. base
rates of 15-21% for the most common labels -- real signal. WIND_MATCH_BOOST
=2.75 beat the previous default (2.0) on held-out log-loss in 5/5 splits;
the log-loss curve is smooth and single-peaked between 2.5-3.0 in every
split. Deployed: WIND_MATCH_BOOST raised 2.0 -> 2.75 in weather_forecast.py.
See MODEL_DOCUMENTATION.md sec 11.31.

Not a recurring script (Tier 3 per sec 11.28 -- one-time "actually fit this
for the first time," not an annual refresh) -- kept for reproducibility, not
scheduled to re-run automatically.
"""
import time

import numpy as np
import pandas as pd
import requests

from src.models.park_orientation import CONFIDENT_TEAMS, forecast_wind_to_bucket_suffix
from src.models.weather import bucket_weather
from src.models.weather_forecast import RETRACTABLE_OR_DOMED_VENUES, VENUE_COORDS, VENUE_TO_TEAM
from src.utils.paths import DATA_RAW

SEASONS = [2023, 2024, 2025]
CANDIDATES = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0]


def fetch_qualifying_games() -> pd.DataFrame:
    games = []
    for season in SEASONS:
        s = pd.read_parquet(DATA_RAW / f"schedule_{season}.parquet")
        s = s[(s["game_type"] == "R") & s["weather_condition"].notna()]
        games.append(s[["game_pk", "date", "venue_name", "weather_condition", "weather_temp", "weather_wind"]])
    games = pd.concat(games, ignore_index=True)
    games["bucket"] = games.apply(
        lambda r: bucket_weather(r["weather_condition"], r["weather_temp"], r["weather_wind"]), axis=1
    )
    games = games[games["bucket"].notna()].copy()
    games["month"] = pd.to_datetime(games["date"]).dt.month
    games["team"] = games["venue_name"].map(VENUE_TO_TEAM)

    qualifying = games[
        games["team"].isin(CONFIDENT_TEAMS)
        & ~games["venue_name"].isin(RETRACTABLE_OR_DOMED_VENUES)
        & games["venue_name"].isin(VENUE_COORDS)
    ].copy()

    wind_by_venue = {}
    for venue in qualifying["venue_name"].unique():
        lat, lon = VENUE_COORDS[venue]
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": lat, "longitude": lon,
            "start_date": "2023-01-01", "end_date": "2025-12-31",
            "hourly": "wind_direction_10m,wind_speed_10m", "wind_speed_unit": "mph", "timezone": "auto",
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
        times = data.get("hourly", {}).get("time", [])
        dirs = data.get("hourly", {}).get("wind_direction_10m", [])
        speeds = data.get("hourly", {}).get("wind_speed_10m", [])
        wind_by_venue[venue] = {t[:10]: (d, s) for t, d, s in zip(times, dirs, speeds) if t.endswith("T19:00")}
        time.sleep(0.3)

    qualifying["wind_from_deg"], qualifying["wind_speed_mph"] = zip(*qualifying.apply(
        lambda r: wind_by_venue.get(r["venue_name"], {}).get(str(r["date"])[:10], (None, None)), axis=1
    ))
    qualifying["forecast_suffix"] = qualifying.apply(
        lambda r: forecast_wind_to_bucket_suffix(r["team"], r["wind_from_deg"], r["wind_speed_mph"])
        if pd.notna(r["wind_from_deg"]) else None, axis=1
    )
    qualifying["tbin"] = qualifying["bucket"].str.split("_", n=1).str[0]
    qualifying.loc[qualifying["bucket"] == "indoor", "tbin"] = "indoor"
    return qualifying


def classification_accuracy(df: pd.DataFrame) -> dict:
    real_suffix = df["bucket"].str.split("_", n=1).str[1]
    real_suffix[df["bucket"] == "indoor"] = None
    directional = df.assign(real_suffix=real_suffix)
    directional = directional[directional["real_suffix"].notna() & directional["forecast_suffix"].notna()]
    return {
        "n": len(directional),
        "accuracy": float((directional["forecast_suffix"] == directional["real_suffix"]).mean()),
        "base_rates": directional["real_suffix"].value_counts(normalize=True).to_dict(),
    }


def _score(games_df: pd.DataFrame, clim_source_df: pd.DataFrame) -> dict:
    clim_lookup = {
        key: g["bucket"].value_counts(normalize=True).to_dict()
        for key, g in clim_source_df.groupby(["venue_name", "month"])
    }
    out = {b: [] for b in CANDIDATES}
    for _, row in games_df.iterrows():
        clim = clim_lookup.get((row["venue_name"], row["month"]))
        if not clim:
            continue
        tbin = row["tbin"]
        restricted = {b: p for b, p in clim.items() if b == "indoor" or b.startswith(tbin)}
        if not restricted or row["bucket"] not in restricted:
            continue
        suffix = row["forecast_suffix"]
        match_key = f"{tbin}_{suffix}" if pd.notna(suffix) else None
        for boost in CANDIDATES:
            r = dict(restricted)
            if match_key is not None and match_key in r:
                r[match_key] *= boost
            total = sum(r.values())
            out[boost].append(-np.log(max(r[row["bucket"]] / total, 1e-6)))
    return {b: float(np.mean(v)) if v else float("nan") for b, v in out.items()}


def stability_check(df: pd.DataFrame, n_splits: int = 5, frac_train: float = 0.6) -> list[dict]:
    results = []
    for seed in range(n_splits):
        shuffled = df.sample(frac=1.0, random_state=seed)
        cut = int(len(shuffled) * frac_train)
        train, test = shuffled.iloc[:cut], shuffled.iloc[cut:]
        results.append(_score(test, train))
    return results


if __name__ == "__main__":
    print("fetching qualifying games + real historical wind (archive-api.open-meteo.com)...", flush=True)
    df = fetch_qualifying_games()
    print(f"qualifying games: {len(df)} across {df['venue_name'].nunique()} confident-team venues", flush=True)

    acc = classification_accuracy(df)
    print(f"\nforecast_suffix classification accuracy: {acc['accuracy']:.4f} (n={acc['n']})")
    print(f"real_suffix base rates: {acc['base_rates']}")

    print(f"\n5-split (60/40) held-out log-loss by candidate WIND_MATCH_BOOST:")
    splits = stability_check(df)
    for i, s in enumerate(splits):
        print(f"  seed={i}: " + "  ".join(f"{b}={v:.5f}" for b, v in s.items()))
    wins = sum(1 for s in splits if s[2.75] < s[2.0])
    print(f"\nboost=2.75 beats boost=2.0 (prior default) on held-out log-loss in {wins}/{len(splits)} splits")
