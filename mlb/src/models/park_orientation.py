"""Each park's real home-plate-to-center-field compass bearing -- needed to
convert a FORECASTED wind reading (a raw meteorological compass bearing, e.g.
"wind from 240 degrees") into the same park-relative label MLB's own
historical weather_wind field already uses for played games (e.g. "Out To
RF"). No trustworthy source for this exists as plain numeric text (checked
twice: Baseball Almanac/Hardball Times/ballparks.com are diagram-only images
with no extractable degree values) and an independent from-scratch attempt to
derive it from real OpenStreetMap field-polygon geometry via PCA also failed
its own sanity check (confidently wrong on Fenway Park, a textbook NE-oriented
park -- computed 236 degrees/SW; inconsistent across different OSM ways for
Comerica Park). See project memory for both failed attempts.

This module's bearings come from a THIRD, genuinely different method that
sidesteps needing any external source or geometry at all: since MLB's own
historical weather_wind field is already reported park-relative, a park's true
orientation can be back-calculated by cross-referencing many (real MLB-
reported relative label, real historical GEOGRAPHIC wind bearing at that
park/date) pairs from a free historical weather archive
(archive-api.open-meteo.com, no key needed) -- grid-searching the home-to-CF
bearing B that best explains all of them at once (each relative label implies
an expected wind travel-direction under a candidate B: out_CF=B, out_RF=B+60,
out_LF=B-60, in_*=180 degrees opposite the matching out_*, cross_LtoR=B+90,
cross_RtoL=B-90 -- offsets reflecting a batter's left/right hand as they face
center field). Validated on 3 parks with well-known real orientations before
trusting it at scale: Fenway Park and Wrigley Field both recovered a
northeast bearing (matching their well-known orientation), Comerica Park
recovered southeast/"southward" (matching Hardball Times' explicit claim) --
all three with a SHARP single-peaked fit, not a scattered/ambiguous one.

MEAN_ERROR_DEG is each park's own calibration quality (mean circular error, in
degrees, between the real historical wind and what the best-fit bearing would
have predicted) -- every park beats the ~90-degree pure-chance baseline
(confirming real signal everywhere), but none is a tight fit (24-73 degree
range), so this is real, not-fully-resolved uncertainty, not a precise
measurement. CONFIDENT_TEAMS gates which parks are trusted enough to drive a
live forecast-based wind-direction adjustment -- below-threshold parks
(mostly small-sample retractable-roof venues with little real directional-
wind history to calibrate against) fall back to the existing climatological-
only wind handling rather than risk a wrong, confidently-applied label."""

# {team_abbr: home-plate-to-center-field bearing, degrees, 0=true north}
# ATH corrected 2026-07-21: originally calibrated against Oakland Coliseum's
# historical wind data by mistake -- checked venue_name game counts directly
# afterward and found our own cached 2023-2026 data has ZERO Oakland Coliseum
# games under the "ATH" team code at all (the Athletics' 2023-2024 Oakland
# seasons are stored under a separate historical "OAK" team code, and only
# "ATH"-coded games -- all at Sutter Health Park -- were actually used to
# pull the real MLB wind labels). Re-ran the calibration against Sutter
# Health Park's real coordinates instead (38.5805, -121.5310, matching
# weather_forecast.py's own VENUE_COORDS): 64 -> 38 degrees.
PARK_ORIENTATION_DEG = {
    "BAL": 40, "BOS": 38, "NYY": 78, "TB": 46, "TOR": 2,
    "CWS": 129, "CLE": 1, "DET": 144, "KC": 68, "MIN": 88,
    "ATH": 38, "HOU": 54, "LAA": 45, "SEA": 80, "TEX": 55,
    "ATL": 185, "MIA": 142, "NYM": 46, "PHI": 18, "WSH": 26,
    "CHC": 34, "CIN": 139, "MIL": 140, "PIT": 121, "STL": 73,
    "AZ": 17, "COL": 351, "LAD": 345, "SD": 0, "SF": 75,
}

# {team_abbr: mean circular error, degrees} from the calibration above
MEAN_ERROR_DEG = {
    "BAL": 44.0, "BOS": 55.3, "NYY": 51.8, "TB": 59.9, "TOR": 66.9,
    "CWS": 38.6, "CLE": 52.8, "DET": 46.5, "KC": 53.6, "MIN": 52.3,
    "ATH": 44.5, "HOU": 72.7, "LAA": 49.0, "SEA": 42.3, "TEX": 68.2,
    "ATL": 53.9, "MIA": 56.5, "NYM": 36.2, "PHI": 37.2, "WSH": 45.3,
    "CHC": 43.7, "CIN": 42.6, "MIL": 55.6, "PIT": 43.5, "STL": 51.4,
    "AZ": 55.6, "COL": 69.0, "LAD": 38.1, "SD": 29.7, "SF": 23.7,
}

# below this mean-error threshold, trust this park's calibrated bearing
# enough to drive a live forecast-based wind-direction adjustment. Chosen to
# exclude only the handful of clearly-weakest fits (mostly small-sample
# retractable-roof parks -- HOU/TEX/TOR -- plus Coors Field/COL, which is
# fully open-air but still landed a weak 69-degree fit) at 67-73 degrees,
# while keeping every park with a meaningfully-better-than-chance fit.
CONFIDENCE_ERROR_THRESHOLD_DEG = 60.0

CONFIDENT_TEAMS = {t for t, e in MEAN_ERROR_DEG.items() if e < CONFIDENCE_ERROR_THRESHOLD_DEG}

# same offsets used during calibration -- the classification rule for turning
# a real wind travel-direction into a park-relative label given a bearing B.
_OFFSET_TO_LABEL = [
    (0, "out_CF"), (60, "out_RF"), (-60, "out_LF"),
    (180, "in_CF"), (240, "in_RF"), (120, "in_LF"),
    (90, "cross_LtoR"), (-90, "cross_RtoL"),
]


def forecast_wind_to_bucket_suffix(team: str, wind_from_deg: float, wind_speed_mph: float) -> str | None:
    """Convert a raw FORECASTED meteorological wind reading (wind_from_deg:
    standard convention, the compass direction the wind is COMING FROM;
    wind_speed_mph) into the same park-relative bucket suffix bucket_weather
    produces for a real historical game (see weather.py) -- e.g. "out_RF",
    "cross_LtoR", "calm". Returns None if this team's calibration isn't
    confident enough to trust (see CONFIDENT_TEAMS) or wind_from_deg is
    missing -- callers should treat None as "fall back to the existing
    climatological wind handling," never as "assume calm."""
    if team not in CONFIDENT_TEAMS or wind_from_deg is None:
        return None
    if wind_speed_mph < 5:
        return "calm"
    travel_deg = (wind_from_deg + 180) % 360
    bearing = PARK_ORIENTATION_DEG[team]
    best_label, best_diff = None, 361
    for offset, label in _OFFSET_TO_LABEL:
        expected = (bearing + offset) % 360
        diff = abs(travel_deg - expected) % 360
        diff = min(diff, 360 - diff)
        if diff < best_diff:
            best_diff, best_label = diff, label
    return best_label
