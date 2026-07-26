"""Weather for a game whose real conditions aren't posted yet (see
generate_daily_props.py -- confirmed a future game's weather stays empty
even fetched the night before). Three-tier fallback, each honestly labeled
rather than silently guessed:

  1. Real posted conditions (handled upstream in props.py -- unchanged).
  2. A REAL temperature forecast (free, no-key Open-Meteo API) narrows the
     climatological wind/bucket distribution below to just the buckets
     matching that forecasted temp bin, then samples wind from history.
  3. Pure climatological: this park's own historical distribution of real
     weather buckets for this calendar month, sampled per Monte Carlo trial
     (not a single fixed "average" -- propagates real day-to-day weather
     uncertainty the same way every other stochastic element in this
     project is handled).

Wind DIRECTION is now ALSO forecast-driven (added 2026-07-21), for parks
where park_orientation.py's data-calibrated home-plate-to-CF bearing is
confident enough to trust (see park_orientation.CONFIDENT_TEAMS) -- two
earlier attempts at per-park orientation failed (an external numeric source
never existed as extractable text; an independent from-scratch derivation
from real OpenStreetMap field geometry got a textbook park confidently
wrong), and this project's decision until now was that guessing wind
direction wrong would silently flip a sign, worse than not modeling it. The
THIRD attempt (park_orientation.py) sidesteps needing any external source or
geometry: it back-calculates each park's orientation from real historical
(MLB-reported relative wind label, real historical geographic wind bearing)
pairs, validated on 3 known-orientation parks before trusting it at scale.

Even with a validated bearing, the classification from a raw forecast
bearing to a relative label carries real residual uncertainty (each
confident park's own calibration error is 24-60 degrees, not a precise
measurement) -- this is applied as a MODEST reweighting of the climatological
distribution (WIND_MATCH_BOOST), not a hard restriction to a single bucket
the way the temperature forecast is: a bucket matching the forecast wind's
predicted direction gets more probability mass, but buckets that don't match
still keep some, unlike temperature (which IS restricted hard, since a
specific forecasted reading is comparatively reliable and carries no
geometry-translation uncertainty).

UPDATE (task #158, 2026-07-25): this piece now HAS been validated against a
real forecast-quality backtest -- see MODEL_DOCUMENTATION.md sec 11.31.
Using each CONFIDENT_TEAMS game's real historical wind (archive-api.open-
meteo.com at ~game time, the same real proxy a live forecast call would
have returned) as the "forecast," classification accuracy against the real
posted bucket (37.0%, n=5369) clearly beats the base rate (15-21% for the
most common labels) -- real signal, not noise. A genuine train/test split
(climatology fit on a random 60%, scored leakage-free on the held-out 40%,
5 seeds) found WIND_MATCH_BOOST=2.75 beats the previous default of 2.0 on
held-out log-loss in 5/5 splits -- see the constant's own comment below.

VENUE_COORDS is keyed by venue_name (not team) -- confirmed necessary
directly from our own data: several teams have played at more than one
venue within our 2023-2026 window (Athletics: Oakland Coliseum -> Sutter
Health Park; Rays: Tropicana Field -> George M. Steinbrenner Field after
hurricane damage), so keying by team would silently point a relocated
team's forecast at the wrong city. Only primary, high-frequency venues are
covered -- rare one-off international/exhibition games (Tokyo Dome, London
Stadium, Mexico City, etc., confirmed present in our data as single-digit
game counts) fall through to no forecast, just the climatological layer.
"""

import datetime as _dt

import pandas as pd
import requests

from src.models.park_orientation import forecast_wind_to_bucket_suffix
from src.models.weather import bucket_weather, temp_bin

VENUE_COORDS = {
    "Sutter Health Park": (38.5805, -121.5310),
    "Truist Park": (33.8908, -84.4678),
    "Chase Field": (33.4455, -112.0667),
    "Oriole Park at Camden Yards": (39.2838, -76.6217),
    "Fenway Park": (42.3467, -71.0972),
    "Wrigley Field": (41.9484, -87.6553),
    "Rate Field": (41.8299, -87.6338),
    "Guaranteed Rate Field": (41.8299, -87.6338),
    "Great American Ball Park": (39.0979, -84.5063),
    "Progressive Field": (41.4962, -81.6852),
    "Coors Field": (39.7559, -104.9942),
    "Comerica Park": (42.3390, -83.0485),
    "Daikin Park": (29.7573, -95.3555),
    "Minute Maid Park": (29.7573, -95.3555),
    "Kauffman Stadium": (39.0517, -94.4803),
    "Angel Stadium": (33.8003, -117.8827),
    "Dodger Stadium": (34.0739, -118.2400),
    "loanDepot park": (25.7781, -80.2196),
    "American Family Field": (43.0280, -87.9712),
    "Target Field": (44.9817, -93.2776),
    "Citi Field": (40.7571, -73.8458),
    "Yankee Stadium": (40.8296, -73.9262),
    "Oakland Coliseum": (37.7516, -122.2005),
    "Citizens Bank Park": (39.9061, -75.1665),
    "PNC Park": (40.4469, -80.0057),
    "Petco Park": (32.7073, -117.1566),
    "T-Mobile Park": (47.5914, -122.3325),
    "Oracle Park": (37.7786, -122.3893),
    "Busch Stadium": (38.6226, -90.1928),
    "George M. Steinbrenner Field": (27.9803, -82.5322),
    "Tropicana Field": (27.7683, -82.6482),
    "Globe Life Field": (32.7473, -97.0842),
    "Rogers Centre": (43.6414, -79.3894),
    "Nationals Park": (38.8730, -77.0074),
}

# venue_name -> team_abbr, for looking up park_orientation.py's calibrated
# bearing (keyed by team, not venue) -- built the same way VENUE_COORDS
# itself was: each team's PRIMARY venue in our own 2023-2026 cached data, not
# assumed from general knowledge (a mistake caught directly: an earlier pass
# assumed Oakland Coliseum was still primary for the Athletics and calibrated
# their bearing against the wrong city's historical wind entirely -- checked
# real venue_name game counts afterward and found zero "ATH"-coded games at
# Oakland Coliseum at all in our cached window; that team's Oakland-era games
# use a separate historical "OAK" team code). Venues that map to more than
# one team (none currently) or that aren't any team's primary venue (e.g.
# Guaranteed Rate Field, the pre-rename Rate Field name; Minute Maid Park,
# the pre-rename Daikin Park name) still map to their real team for
# forward-compatibility, since VENUE_COORDS keeps both name variants.
VENUE_TO_TEAM = {
    "Sutter Health Park": "ATH", "Oakland Coliseum": "ATH",
    "Truist Park": "ATL", "Chase Field": "AZ",
    "Oriole Park at Camden Yards": "BAL", "Fenway Park": "BOS",
    "Wrigley Field": "CHC", "Rate Field": "CWS", "Guaranteed Rate Field": "CWS",
    "Great American Ball Park": "CIN", "Progressive Field": "CLE",
    "Coors Field": "COL", "Comerica Park": "DET",
    "Daikin Park": "HOU", "Minute Maid Park": "HOU",
    "Kauffman Stadium": "KC", "Angel Stadium": "LAA", "Dodger Stadium": "LAD",
    "loanDepot park": "MIA", "American Family Field": "MIL", "Target Field": "MIN",
    "Citi Field": "NYM", "Yankee Stadium": "NYY",
    "Citizens Bank Park": "PHI", "PNC Park": "PIT", "Petco Park": "SD",
    "T-Mobile Park": "SEA", "Oracle Park": "SF", "Busch Stadium": "STL",
    "George M. Steinbrenner Field": "TB", "Tropicana Field": "TB",
    "Globe Life Field": "TEX", "Rogers Centre": "TOR", "Nationals Park": "WSH",
}

# a bucket matching the forecast wind's predicted direction gets its
# probability multiplied by this before renormalizing -- modest, not a hard
# restriction (see module docstring): the calibration's own real residual
# uncertainty (24-60 degrees per confident park) means treating a single
# predicted label as certain would risk exactly the "confidently wrong sign"
# failure mode this project has twice already caught and rejected elsewhere.
# Raised from 2.0 (2026-07-25, task #158): a real forecast-quality backtest
# (leave-one-out + 5-split train/test, n=5369 confident-team games,
# archive-api.open-meteo.com real historical wind as the forecast proxy)
# found 2.75 beats 2.0 on held-out log-loss in 5/5 splits; the log-loss
# curve is smooth and single-peaked between 2.5-3.0 across every split, so
# 2.75 sits at the stable middle rather than chasing one split's exact
# minimum. See MODEL_DOCUMENTATION.md sec 11.31.
WIND_MATCH_BOOST = 2.75

# Retractable-roof/domed parks: whether the roof is open on a given day is a
# managerial decision, not weather itself, and unknowable in advance -- the
# climatological layer already reflects each park's real historical
# roof-open frequency for that month, which is more honest than assuming a
# forecasted OUTDOOR temperature applies when the roof might be closed.
RETRACTABLE_OR_DOMED_VENUES = {
    "Chase Field", "Daikin Park", "Minute Maid Park", "American Family Field",
    "Globe Life Field", "Rogers Centre", "loanDepot park", "Tropicana Field",
}

FORECAST_HOUR = 19  # approximate typical game-time local hour; Open-Meteo's
                     # timezone=auto returns hourly timestamps in LOCAL time.


def fetch_temperature_forecast(lat: float, lon: float, target_date: str, hour: int = FORECAST_HOUR) -> float | None:
    """Forecasted temperature (degF) at approximately game time on
    target_date, or None if the date is too far out for a forecast, the
    request fails, or that hour isn't in the returned data."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "timezone": "auto",
            "start_date": target_date, "end_date": target_date,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None
    times = data.get("hourly", {}).get("time", [])
    temps = data.get("hourly", {}).get("temperature_2m", [])
    target_ts = f"{target_date}T{hour:02d}:00"
    if target_ts in times:
        return temps[times.index(target_ts)]
    return None


def fetch_wind_forecast(lat: float, lon: float, target_date: str, hour: int = FORECAST_HOUR) -> tuple[float | None, float | None]:
    """(wind_from_deg, wind_speed_mph) at approximately game time on
    target_date -- (None, None) under the same failure conditions as
    fetch_temperature_forecast. wind_from_deg is the standard meteorological
    convention (the compass direction the wind is COMING FROM), matching
    what park_orientation.forecast_wind_to_bucket_suffix expects."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "hourly": "wind_direction_10m,wind_speed_10m",
            "wind_speed_unit": "mph", "timezone": "auto",
            "start_date": target_date, "end_date": target_date,
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None, None
    times = data.get("hourly", {}).get("time", [])
    dirs = data.get("hourly", {}).get("wind_direction_10m", [])
    speeds = data.get("hourly", {}).get("wind_speed_10m", [])
    target_ts = f"{target_date}T{hour:02d}:00"
    if target_ts in times:
        idx = times.index(target_ts)
        return dirs[idx], speeds[idx]
    return None, None


def build_historical_game_buckets(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per historical REGULAR-SEASON game with real posted weather:
    its venue, calendar month, and weather bucket (see weather.bucket_weather).
    Deliberately pooled across all cached seasons, not walk-forward-by-season
    -- a park's seasonal climate pattern is stable year to year, not a
    skill/trend that could leak forward-looking information (same reasoning
    as blowout.py's pooled position-player profile)."""
    reg = schedule[(schedule["game_type"] == "R") & schedule["weather_condition"].notna()].copy()
    reg["bucket"] = reg.apply(lambda r: bucket_weather(r["weather_condition"], r["weather_temp"], r["weather_wind"]), axis=1)
    reg["month"] = pd.to_datetime(reg["date"]).dt.month
    return reg[reg["bucket"].notna()][["venue_name", "month", "bucket"]]


def climatological_bucket_distribution(game_buckets: pd.DataFrame, venue_name: str, month: int) -> dict:
    """{bucket: probability} from this park's own real history in this
    calendar month. Empty dict if this venue has no historical weather data
    (e.g. a rare one-off international/exhibition site)."""
    sub = game_buckets[(game_buckets["venue_name"] == venue_name) & (game_buckets["month"] == month)]
    if sub.empty:
        return {}
    return sub["bucket"].value_counts(normalize=True).to_dict()


def resolve_weather_distribution(game_buckets: pd.DataFrame, venue_name: str, target_date: str) -> dict:
    """The bucket-probability distribution to sample from for a game with no
    posted weather yet -- climatological, narrowed by a real temperature
    forecast (hard restriction -- see module docstring), then further
    reweighted (soft boost, not a restriction) toward the wind-direction
    bucket implied by a real wind forecast, for parks confident enough to
    trust (see park_orientation.CONFIDENT_TEAMS)."""
    month = pd.Timestamp(target_date).month
    dist = climatological_bucket_distribution(game_buckets, venue_name, month)
    if not dist or venue_name in RETRACTABLE_OR_DOMED_VENUES or venue_name not in VENUE_COORDS:
        return dist

    lat, lon = VENUE_COORDS[venue_name]
    forecast_temp = fetch_temperature_forecast(lat, lon, target_date)
    if forecast_temp is None:
        return dist
    tbin = temp_bin(forecast_temp)
    if tbin is None:
        return dist
    restricted = {b: p for b, p in dist.items() if b == "indoor" or b.startswith(tbin)}
    if not restricted:
        return dist

    team = VENUE_TO_TEAM.get(venue_name)
    if team is not None:
        wind_from_deg, wind_speed_mph = fetch_wind_forecast(lat, lon, target_date)
        suffix = forecast_wind_to_bucket_suffix(team, wind_from_deg, wind_speed_mph) if wind_from_deg is not None else None
        if suffix is not None:
            match_key = f"{tbin}_{suffix}"
            if match_key in restricted:
                restricted = dict(restricted)
                restricted[match_key] *= WIND_MATCH_BOOST

    total = sum(restricted.values())
    return {b: p / total for b, p in restricted.items()}


def sample_weather_bucket(dist: dict, rng) -> str | None:
    """Sample one bucket from a distribution built above -- called once per
    Monte Carlo trial (not once per game), so real day-to-day weather
    uncertainty propagates into the simulated outcome distribution."""
    if not dist:
        return None
    buckets, probs = zip(*dist.items())
    return rng.choice(buckets, p=probs)
