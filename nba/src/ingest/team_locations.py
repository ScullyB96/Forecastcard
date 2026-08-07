"""Static per-team home-arena location lookup (task #60, travel/timezone-
fatigue diagnostic) -- same convention as `team_codes.py`: a fixed, public,
no-relocation-in-window fact table, not a fetched-and-cached one (all 30
current franchises' home arenas/cities are stable across the entire
2015-16..present dev/holdout range).

`tz_zone` is a SIMPLIFIED 4-value US timezone bucket (0=Eastern,
1=Central, 2=Mountain, 3=Pacific), not a precise UTC-offset-with-DST
calculation -- adequate for a "how many zones did this team just cross"
fatigue proxy, not for anything requiring exact wall-clock arithmetic.
Arizona (Phoenix, no DST) and Denver/Utah's Mountain-zone DST observance
are both bucketed at their season-long conventional zone (2) -- a known,
accepted simplification, not an oversight.
"""

TEAM_LOCATIONS: dict[str, dict] = {
    "ATL": {"lat": 33.7573, "lon": -84.3963, "tz_zone": 0},
    "BOS": {"lat": 42.3662, "lon": -71.0621, "tz_zone": 0},
    "BKN": {"lat": 40.6826, "lon": -73.9754, "tz_zone": 0},
    "CHA": {"lat": 35.2251, "lon": -80.8392, "tz_zone": 0},
    "CHI": {"lat": 41.8807, "lon": -87.6742, "tz_zone": 1},
    "CLE": {"lat": 41.4965, "lon": -81.6882, "tz_zone": 0},
    "DAL": {"lat": 32.7905, "lon": -96.8103, "tz_zone": 1},
    "DEN": {"lat": 39.7487, "lon": -105.0077, "tz_zone": 2},
    "DET": {"lat": 42.3410, "lon": -83.0553, "tz_zone": 0},
    "GSW": {"lat": 37.7680, "lon": -122.3877, "tz_zone": 3},
    "HOU": {"lat": 29.7508, "lon": -95.3621, "tz_zone": 1},
    "IND": {"lat": 39.7640, "lon": -86.1555, "tz_zone": 0},
    "LAC": {"lat": 34.0430, "lon": -118.2673, "tz_zone": 3},
    "LAL": {"lat": 34.0430, "lon": -118.2673, "tz_zone": 3},
    "MEM": {"lat": 35.1382, "lon": -90.0505, "tz_zone": 1},
    "MIA": {"lat": 25.7814, "lon": -80.1870, "tz_zone": 0},
    "MIL": {"lat": 43.0451, "lon": -87.9172, "tz_zone": 1},
    "MIN": {"lat": 44.9795, "lon": -93.2761, "tz_zone": 1},
    "NOP": {"lat": 29.9490, "lon": -90.0821, "tz_zone": 1},
    "NYK": {"lat": 40.7505, "lon": -73.9934, "tz_zone": 0},
    "OKC": {"lat": 35.4634, "lon": -97.5151, "tz_zone": 1},
    "ORL": {"lat": 28.5392, "lon": -81.3839, "tz_zone": 0},
    "PHI": {"lat": 39.9012, "lon": -75.1720, "tz_zone": 0},
    "PHX": {"lat": 33.4457, "lon": -112.0712, "tz_zone": 2},
    "POR": {"lat": 45.5316, "lon": -122.6668, "tz_zone": 3},
    "SAC": {"lat": 38.5802, "lon": -121.4997, "tz_zone": 3},
    "SAS": {"lat": 29.4269, "lon": -98.4375, "tz_zone": 1},
    "TOR": {"lat": 43.6435, "lon": -79.3791, "tz_zone": 0},
    "UTA": {"lat": 40.7683, "lon": -111.9011, "tz_zone": 2},
    "WAS": {"lat": 38.8981, "lon": -77.0209, "tz_zone": 0},
}

assert len(TEAM_LOCATIONS) == 30, f"expected 30 current NBA franchises, got {len(TEAM_LOCATIONS)}"
