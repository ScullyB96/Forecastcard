"""Weather effects, from the MLB Stats API's own per-game weather fields
(weather_condition, weather_temp, weather_wind -- already cached in the
schedule tables, already reported RELATIVE TO THE PARK, e.g. "Out To RF"
means blowing toward right field in THAT park's own orientation regardless of
its geographic compass direction -- this is what makes a lefty/righty-
specific effect tractable without needing separate per-park field-geometry
modeling: "Out To RF" always disproportionately helps a lefty pull hitter,
in every park, by construction of how the API already reports it). This
FIELD-specific distinction (RF vs. CF vs. LF, not just "out" generically) is
what bucket_weather actually keys on -- see its docstring for the real data
confirming the effect and why an earlier version of this module collapsed
that distinction away by mistake.

Confirmed on real data before building anything: "Roof Closed" games show
near-uniformly controlled conditions (72degF median, tight distribution; 0 mph
wind in 98% of games) -- correctly treated as a separate "indoor" bucket with
no outdoor weather effect, rather than binned by the meaningless recorded
temp/wind of a controlled climate.

Same empirical bucket-factor design as build_state_factors_by_season in
game_simulator.py: factor[bucket][stand][outcome] = rate(outcome | bucket,
batter stand) / rate(outcome | overall) -- a weighted decomposition (overall
IS the frequency-weighted average of the bucket rates), so unlike the
original park_factors bug this does NOT need a population-mean-1.0
normalization fix; the structure guarantees it already. Walk-forward-safe:
each season's factors are built only from strictly prior seasons.
"""

import pandas as pd

from src.models.game_simulator import OUTCOMES

WEATHER_FACTOR_PRIOR_PA = 5000  # pseudo-PA of Bayesian shrinkage toward the
                                # group's own overall rate, see build_weather_factors_by_season

# Hard safety-net bound (added 2026-07-21, after an audit found this was the
# only "environmental multiplier" table in the project relying purely on
# shrinkage with no hard clip, unlike park_factors.py's mean-1.0 renorm and
# defense_factor.py's explicit clip). Confirmed the CURRENT real table stays
# well inside this range on its own (0.19x-14.3x, both extremes on
# triple_play -- the same ultra-rare category park_factors.py already shows
# a comparable 0.00005x-28x range for, an accepted characteristic of
# astronomically-rare-event ratios, not a live bug) -- this bound is pure
# insurance against a future data update producing a sparser cell, not a
# response to a currently-observed extreme.
WEATHER_FACTOR_CLIP_MIN, WEATHER_FACTOR_CLIP_MAX = 0.1, 20.0


def temp_bin(temp) -> str | None:
    """One of "<60"/"60s"/"70s"/"80s"/"90+", or None if temp is missing/NaN.
    Factored out of bucket_weather so weather_forecast.py's forecast-temp
    layer can bin a real forecasted temperature the same way a historical
    reading is binned."""
    try:
        temp = float(temp)
    except (TypeError, ValueError):
        return None
    if pd.isna(temp):
        # NaN comparisons (temp < 60, etc.) are all False in Python, so
        # without this explicit check NaN silently falls through to the
        # final "else" branch below ("90+") instead of being recognized as
        # missing -- confirmed to actually happen: every game with not-yet-
        # posted weather (checked live the night before a future game) was
        # silently getting bucketed as a hot, calm day instead of correctly
        # getting no adjustment at all. Caught before this ever ran against
        # real future-game predictions.
        return None
    if temp < 60:
        return "<60"
    if temp < 70:
        return "60s"
    if temp < 80:
        return "70s"
    if temp < 90:
        return "80s"
    return "90+"


def bucket_weather(condition: str, temp, wind: str) -> str | None:
    """One of "indoor", or "{temp_bin}_{wind_bin}" for outdoor games.
    wind_bin is "calm" (light/no wind or an unpredictable "Varies" reading),
    or "{direction}_{field}" for a real directional wind (in/out -- FIELD-
    SPECIFIC: RF/CF/LF, not collapsed -- see below) or "cross_LtoR"/
    "cross_RtoL" for a crosswind. Below 5 mph is treated as calm regardless
    of direction -- MLB's own reported wind reading is too unreliable/
    inconsistent to trust a direction label at that speed.

    Field-specific (not collapsed to a generic "in"/"out"), because a real,
    meaningful lefty/righty-specific asymmetry was confirmed on our own data
    (2026-07-20) that a collapsed direction-only bucket structurally cannot
    capture: the lefty-vs-righty home_run rate gap is roughly 4x larger on
    "Out To RF" days (+0.50pp) than "Out To CF" days (+0.13pp) -- lefties
    specifically pull toward RF, so wind blowing out to RF specifically (not
    just "out" generically) is what should disproportionately help them.
    Deliberately does NOT also split by wind speed (light/strong, as an
    earlier version did) on top of the field split -- checked sample sizes
    first: full field+speed granularity would push several real (bucket,
    season, temp-bin) cells uncomfortably thin for the walk-forward-safe
    per-season split this project uses throughout, especially in early
    seasons with few/no prior years -- and field-specificity is the newer,
    currently-uncaptured signal being added here, so it takes priority over
    preserving speed granularity within an already-more-granular split."""
    if condition in ("Roof Closed", "Dome"):
        return "indoor"
    tbin = temp_bin(temp)
    if tbin is None:
        return None

    if not isinstance(wind, str):
        return f"{tbin}_calm"
    speed_match = pd.Series([wind]).str.extract(r"^(\d+)")[0].iloc[0]
    speed = float(speed_match) if speed_match is not None else 0.0
    if speed < 5:
        return f"{tbin}_calm"

    if "In From RF" in wind:
        return f"{tbin}_in_RF"
    if "In From CF" in wind:
        return f"{tbin}_in_CF"
    if "In From LF" in wind:
        return f"{tbin}_in_LF"
    if "Out To RF" in wind:
        return f"{tbin}_out_RF"
    if "Out To CF" in wind:
        return f"{tbin}_out_CF"
    if "Out To LF" in wind:
        return f"{tbin}_out_LF"
    if "L To R" in wind:
        return f"{tbin}_cross_LtoR"
    if "R To L" in wind:
        return f"{tbin}_cross_RtoL"
    return f"{tbin}_calm"  # "Calm", "None", "Varies"


def attach_weather_bucket(pa: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Merge each PA with its game's weather bucket (see bucket_weather)."""
    game_weather = schedule[["game_pk", "weather_condition", "weather_temp", "weather_wind"]].drop_duplicates("game_pk")
    game_weather["weather_bucket"] = game_weather.apply(
        lambda r: bucket_weather(r["weather_condition"], r["weather_temp"], r["weather_wind"]), axis=1
    )
    return pa.merge(game_weather[["game_pk", "weather_bucket"]], on="game_pk", how="left")


def build_weather_factors_by_season(pa_with_bucket: pd.DataFrame) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """Walk-forward-safe (prior-seasons-only): factors[season][bucket][group][outcome],
    where `group` is "{stand}_{pull_tercile}" (e.g. "L_high_pull") -- REQUIRES
    a `pull_tercile` column already attached (see spray.py's
    build_pull_rate_snapshot/pull_tercile; a batter with no pull-rate history
    yet should be passed as "mid_pull", the neutral default). `stand` is the
    batter's ACTUAL side for that PA ("L"/"R", from Statcast's own `stand`
    column -- already resolved correctly for switch-hitters per-PA, unlike
    the fixed "S" placeholder used for a player's default profile elsewhere
    in this project) -- this is what makes the lefty/righty-specific
    park-relative wind effect (e.g. "Out To RF" helping lefties) fall out
    naturally from the data, no separate field-geometry model needed.

    Splitting further by pull_tercile on top of stand is NOT a redundant
    refinement -- confirmed real, additional signal beyond stand alone
    (2026-07-20): among lefty batters, the home_run-rate gap between
    pull-side wind (out_RF) and opposite-field wind (out_LF) is near zero for
    low-pull hitters but a real, meaningfully larger +0.46pp for high-pull
    hitters -- a stand-only factor washes this out by averaging pull-heavy
    and balanced hitters together.

    Deliberately normalized against each GROUP's OWN overall rate (not the
    single league-wide pooled rate): confirmed on real data that lefty
    batters have a somewhat higher overall home_run rate than righties
    league-wide, independent of weather (and, separately, that pull tendency
    itself correlates with power). Every player's true-talent rate is
    already individually estimated (see true_talent.py), so normalizing
    against the pooled rate here would double-count those generic gaps on
    top of each player's own rate; normalizing per-group isolates just the
    WEATHER-SPECIFIC deviation this module is actually responsible for."""
    pa = pa_with_bucket[pa_with_bucket["weather_bucket"].notna() & pa_with_bucket["pull_tercile"].notna()].copy()
    pa["group"] = pa["stand"] + "_" + pa["pull_tercile"]
    out = {}
    for season in sorted(pa["season"].unique()):
        prior = pa[pa["season"] < season]
        if prior.empty:
            # task #160 (2026-07-26 correctness audit) -- same real leak and
            # same fix as catcher_framing.py/umpire_factor.py's identical
            # cold-start pattern: the true first season previously fell back
            # to ITS OWN full data instead of a neutral default. An empty
            # dict here means the caller's own `.get(season, {}).get(bucket)`
            # lookup (validate_game_simulator.py) correctly returns None
            # (treated as "no weather adjustment"), matching how a missing
            # park-factor season is already handled.
            out[season] = {}
            continue
        ref = prior
        overall_counts_by_group = {group: g["outcome"].value_counts() for group, g in ref.groupby("group")}
        overall_n_by_group = {group: len(g) for group, g in ref.groupby("group")}

        factors = {}
        for (bucket, group), g in ref.groupby(["weather_bucket", "group"]):
            bucket_counts = g["outcome"].value_counts()
            n_bucket = len(g)
            overall_counts = overall_counts_by_group[group]
            n_overall = overall_n_by_group[group]
            group_factors = {}
            for o in OUTCOMES:
                overall_rate = overall_counts.get(o, 0) / n_overall if n_overall else 0.0
                # Bayesian-shrink the (small-sample) bucket rate toward the
                # (large-sample) overall rate before taking the ratio -- same
                # WEATHER_FACTOR_PRIOR_PA pseudo-PA principle already used for
                # park factors (park_factors.py), applied here after a real,
                # serious bug was found: an un-shrunk bucket_rate/overall_rate
                # ratio is unbounded as n_bucket shrinks, and the NEW
                # field-specific-wind x pull-tercile split (added 2026-07-21)
                # multiplies the number of (bucket, group) cells enough that
                # several rare outcome categories (triple_play, catcher_interf,
                # sac_bunt) had cells with almost no real samples -- confirmed
                # directly: factor values up to 73x were found across the
                # table before this fix, an extreme silently capable of
                # distorting a specific game's whole outcome distribution.
                shrunk_bucket_rate = (
                    (bucket_counts.get(o, 0) + WEATHER_FACTOR_PRIOR_PA * overall_rate)
                    / (n_bucket + WEATHER_FACTOR_PRIOR_PA)
                )
                factor = shrunk_bucket_rate / overall_rate if overall_rate > 1e-9 else 1.0
                group_factors[o] = max(WEATHER_FACTOR_CLIP_MIN, min(WEATHER_FACTOR_CLIP_MAX, factor))
            factors.setdefault(bucket, {})[group] = group_factors
        out[season] = factors
    return out


def build_venue_weather_norms(game_buckets, weather_factors_by_season: dict) -> dict:
    """norms[season][venue_name][group][outcome] = this venue's EXPECTED
    weather factor under its own climatological bucket distribution
    (2026-08-03 audit, finding M10). The park factor (park_factors.py) is
    measured from real home/road outcome rates, so each park's AVERAGE
    climate is already inside it -- applying the absolute weather factor on
    top double-counts that average (measured: per-venue mean applied HR
    weather factor spanned 0.979 at Petco/SF to 1.020 at TB, a systematic
    ~+/-2% HR-scale bias at climate-extreme venues). Dividing each game's
    bucket factor by this norm (see park_relative_weather_factors) makes
    weather contribute only the DEVIATION from the park's own typical
    conditions, leaving the average where it belongs.

    game_buckets: build_historical_game_buckets's output (venue_name,
    month, bucket; pooled across seasons -- a park's climate mix is stable,
    see that function's docstring for why pooling isn't leakage). The norm
    is the game-frequency-weighted mean of the season's own walk-forward
    bucket factors, accumulated per (group, outcome) so buckets missing a
    sparse cell just renormalize instead of biasing the mean."""
    venue_dist = {
        v: g["bucket"].value_counts(normalize=True).to_dict()
        for v, g in game_buckets.groupby("venue_name")
    }
    norms = {}
    for season, factors in weather_factors_by_season.items():
        norms[season] = {}
        if not factors:
            continue  # cold-start season -- no factors, callers apply none either
        for venue, dist in venue_dist.items():
            acc = {}  # group -> outcome -> [prob-weighted factor sum, prob mass seen]
            for bucket, p in dist.items():
                bucket_factors = factors.get(bucket)
                if not bucket_factors:
                    continue
                for group, ofs in bucket_factors.items():
                    g = acc.setdefault(group, {})
                    for o, f in ofs.items():
                        s = g.setdefault(o, [0.0, 0.0])
                        s[0] += p * f
                        s[1] += p
            norms[season][venue] = {
                group: {o: s[0] / s[1] for o, s in ofs.items() if s[1] > 1e-9}
                for group, ofs in acc.items()
            }
    return norms


def park_relative_weather_factors(bucket_factors: dict | None, venue_norm: dict | None) -> dict | None:
    """One game's applied weather factors, re-expressed RELATIVE to the
    venue's climatological norm (finding M10, see build_venue_weather_norms).
    None/missing-norm inputs pass through unchanged (absolute behavior --
    graceful for unknown venues and cold-start seasons)."""
    if bucket_factors is None or not venue_norm:
        return bucket_factors
    out = {}
    for group, ofs in bucket_factors.items():
        norm = venue_norm.get(group)
        if not norm:
            out[group] = ofs
            continue
        out[group] = {
            o: (f / norm[o]) if norm.get(o, 0.0) > 1e-9 else f
            for o, f in ofs.items()
        }
    return out
