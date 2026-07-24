"""Residuals-first check (Sec18), decomposing the mean-total undershoot
(Sec16.5) by situation bucket (EV / PP+PK / other) per season -- adjudicates
between every live hypothesis at once: if the undershoot concentrates in
"other" (where empty-net attempts live), the falsified aggregate xG-goals
check (Sec16.4-ish, which cleared at ratio 1.0008) doesn't rule out a TOI
MISALLOCATION specifically in that bucket, since xG being fine doesn't mean
the model assigns the right number of MINUTES to it. If the bias spreads
evenly across buckets, the TOI mechanism is dead and the walk-forward
intercept wins by elimination.

Uses the SAME `_combine` function team_strength_situational.py's own
`predict_situational_lambda` calls, so the per-bucket lambdas here are
identical to what the production combine actually computes -- not a
re-derivation that could itself introduce a discrepancy.
"""

import pandas as pd

from src.models.shrinkage import add_walk_forward_mean
from src.models.team_strength_situational import (
    _combine, add_walk_forward_situational_strength, build_team_game_situational_log,
)
from src.models.validate_baseline import fit_home_ice_multiplier
from src.models.validate_cross_season import CROSS_SEASON_WEIGHT, PRIOR_MINUTES_MULTIPLIER
from src.models.validate_drift import HALFLIFE_GAMES
from src.utils.paths import DATA_RAW

PRIOR_GAMES_TOI = 10


def run_decomposition(min_season: int = 20102011, use_sh_term: bool = False) -> pd.DataFrame:
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    moneypuck = pd.read_parquet(DATA_RAW / "moneypuck_all_teams.parquet")

    log = build_team_game_situational_log(schedule, moneypuck)
    log = add_walk_forward_situational_strength(log, league_avg_halflife_games=HALFLIFE_GAMES,
                                                 cross_season_weight=CROSS_SEASON_WEIGHT,
                                                 prior_minutes_multiplier=PRIOR_MINUTES_MULTIPLIER)
    for col in ("ev_league_avg_for_per60", "ev_league_avg_against_per60",
                "pp_league_avg_for_per60", "pk_league_avg_against_per60", "pk_league_avg_for_per60",
                "other_league_avg_for_per60", "other_league_avg_against_per60"):
        log[col] = log[col].bfill()
    log["league_avg_ev_toi_min"] = log["ev_toi_min"].shift(1).expanding().mean().bfill()
    log["league_avg_other_toi_min"] = log["other_toi_min"].shift(1).expanding().mean().bfill()
    log = add_walk_forward_mean(log, "pp_toi_min", PRIOR_GAMES_TOI, "pp_toi")
    log = add_walk_forward_mean(log, "pk_toi_min", PRIOR_GAMES_TOI, "pk_toi")
    log["pp_toi_shrunk_mean"] = log["pp_toi_shrunk_mean"].bfill()
    log["pk_toi_shrunk_mean"] = log["pk_toi_shrunk_mean"].bfill()

    home_ice_multiplier = fit_home_ice_multiplier(log)

    games = log[log["is_home"]].copy()
    away_cols = ["gameId", "ev_attack_rate_per60", "ev_defense_rate_per60",
                 "ev_league_avg_for_per60", "ev_league_avg_against_per60",
                 "pp_attack_rate_per60", "pk_defense_rate_per60",
                 "pp_league_avg_for_per60", "pk_league_avg_against_per60", "pk_league_avg_for_per60",
                 "other_attack_rate_per60", "other_defense_rate_per60",
                 "other_league_avg_for_per60", "other_league_avg_against_per60",
                 "pp_toi_shrunk_mean", "pk_toi_shrunk_mean"]
    away_side = log[~log["is_home"]][away_cols]
    games = games.merge(away_side, on="gameId", suffixes=("_home", "_away"))
    games = games[games["season"] >= min_season]

    # Real actual goals per situation bucket, straight from MoneyPuck (not this
    # project's own model output) -- the ground truth this section compares against.
    mp = moneypuck[moneypuck["situation"].isin(["5on5", "5on4", "4on5", "other"])].copy()
    from src.ingest.team_codes import normalize_moneypuck_team_code
    mp["team"] = [normalize_moneypuck_team_code(t, s) for t, s in zip(mp["team"], mp["season"])]
    actual_by_bucket = mp.pivot_table(index=["gameId", "team"], columns="situation", values="goalsFor",
                                       aggfunc="sum").reset_index()
    actual_by_bucket.columns.name = None
    actual_by_bucket = actual_by_bucket.rename(
        columns={"5on5": "actual_ev", "5on4": "actual_pp", "4on5": "actual_pk", "other": "actual_other"})

    results = []
    for row in games.itertuples():
        home_pp_toi = (row.pp_toi_shrunk_mean_home + row.pk_toi_shrunk_mean_away) / 2
        away_pp_toi = (row.pp_toi_shrunk_mean_away + row.pk_toi_shrunk_mean_home) / 2

        home_ev = _combine(row.ev_attack_rate_per60_home, row.ev_defense_rate_per60_away,
                            row.ev_league_avg_for_per60_home, row.ev_league_avg_against_per60_away,
                            row.league_avg_ev_toi_min)
        away_ev = _combine(row.ev_attack_rate_per60_away, row.ev_defense_rate_per60_home,
                            row.ev_league_avg_for_per60_away, row.ev_league_avg_against_per60_home,
                            row.league_avg_ev_toi_min)
        home_pp = _combine(row.pp_attack_rate_per60_home, row.pk_defense_rate_per60_away,
                            row.pp_league_avg_for_per60_home, row.pk_league_avg_against_per60_away, home_pp_toi)
        away_pp = _combine(row.pp_attack_rate_per60_away, row.pk_defense_rate_per60_home,
                            row.pp_league_avg_for_per60_away, row.pk_league_avg_against_per60_home, away_pp_toi)
        home_other = _combine(row.other_attack_rate_per60_home, row.other_defense_rate_per60_away,
                               row.other_league_avg_for_per60_home, row.other_league_avg_against_per60_away,
                               row.league_avg_other_toi_min)
        away_other = _combine(row.other_attack_rate_per60_away, row.other_defense_rate_per60_home,
                               row.other_league_avg_for_per60_away, row.other_league_avg_against_per60_home,
                               row.league_avg_other_toi_min)

        home_sh = away_sh = 0.0
        if use_sh_term:
            home_sh = row.pk_league_avg_for_per60_home * (away_pp_toi / 60)
            away_sh = row.pk_league_avg_for_per60_home * (home_pp_toi / 60)

        results.append({
            "gameId": row.gameId, "season": row.season,
            "home_team": row.team, "away_team": row.opponent,
            "pred_ev_home": home_ev * home_ice_multiplier, "pred_ev_away": away_ev / home_ice_multiplier,
            "pred_pp_home": (home_pp + home_sh) * home_ice_multiplier,
            "pred_pp_away": (away_pp + away_sh) / home_ice_multiplier,
            "pred_other_home": home_other * home_ice_multiplier, "pred_other_away": away_other / home_ice_multiplier,
        })
    r = pd.DataFrame(results)

    home_actual = actual_by_bucket.rename(columns={"team": "home_team"}).add_suffix("_h").rename(
        columns={"gameId_h": "gameId", "home_team_h": "home_team"})
    away_actual = actual_by_bucket.rename(columns={"team": "away_team"}).add_suffix("_a").rename(
        columns={"gameId_a": "gameId", "away_team_a": "away_team"})
    r = r.merge(home_actual, on=["gameId", "home_team"], how="left")
    r = r.merge(away_actual, on=["gameId", "away_team"], how="left")
    return r


if __name__ == "__main__":
    import sys
    use_sh = "--sh" in sys.argv
    r = run_decomposition(use_sh_term=use_sh)
    print(f"{len(r)} games decomposed, use_sh_term={use_sh}")

    r["pred_ev_total"] = r["pred_ev_home"] + r["pred_ev_away"]
    r["actual_ev_total"] = r["actual_ev_h"] + r["actual_ev_a"]
    r["pred_pp_total"] = r["pred_pp_home"] + r["pred_pp_away"]
    r["actual_pp_total"] = r["actual_pp_h"] + r["actual_pk_h"] + r["actual_pp_a"] + r["actual_pk_a"]
    r["pred_other_total"] = r["pred_other_home"] + r["pred_other_away"]
    r["actual_other_total"] = r["actual_other_h"] + r["actual_other_a"]

    by_season = r.groupby("season").agg(
        n=("gameId", "count"),
        pred_ev=("pred_ev_total", "mean"), actual_ev=("actual_ev_total", "mean"),
        pred_pp=("pred_pp_total", "mean"), actual_pp=("actual_pp_total", "mean"),
        pred_other=("pred_other_total", "mean"), actual_other=("actual_other_total", "mean"),
    )
    by_season["gap_ev"] = by_season["pred_ev"] - by_season["actual_ev"]
    by_season["gap_pp"] = by_season["pred_pp"] - by_season["actual_pp"]
    by_season["gap_other"] = by_season["pred_other"] - by_season["actual_other"]

    pd.set_option("display.width", 160)
    print(by_season[["n", "gap_ev", "gap_pp", "gap_other"]])
    print("\noverall mean gaps:")
    print(f"  EV: {by_season['gap_ev'].mean():+.4f}")
    print(f"  PP+PK: {by_season['gap_pp'].mean():+.4f}")
    print(f"  other: {by_season['gap_other'].mean():+.4f}")
