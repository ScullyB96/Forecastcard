"""Player props for Week 1, 2026: projected targets, carries, receptions,
receiving/rushing yards, and TD probability for real players in real games,
tying together every validated component built this session:

  - Layer 1 (ratings.py) + calibration: predicted margin/total per game
  - Game environment (game_environment.py): pass rate (with game-script
    adjustment from predicted margin) and league-average plays/game (team-
    specific pace was tested and rejected -- see that module's docstring)
  - Layer 2 (player_usage.py): target share, carry share, TD-per-touch rate,
    yards-per-target/carry, catch rate

Signal strength is NOT uniform across these and shouldn't be treated as if it
were. Validated end-to-end (validate_props_pipeline.py) on 2022-2025 holdout:
target/carry share and TD probability are strong (targets MAE 1.92 vs. naive
2.47; carries MAE 3.18 vs. naive 5.18; TD-probability calibration corr 0.992
across deciles, with a fitted correction applied below for a top-decile
overconfidence). Yards and catch rate are real but much weaker (0.5-3.7%
better than naive) -- yards-per-touch is dominated by boom/bust variance a
single player's "skill" barely dents. Included because the signal is genuine,
not because it's strong.

Rookie fallback: a player with no NFL history (no gsis_id match against our
name lookup) gets share/rate ratings that all default to 0 in our own
engines -- correct behavior for the engines themselves (there's no data to
rate them on), but wrong for a props output, since a real rookie starter
should get real projected volume. For those specific players, this script
falls back to Clay/ESPN's own season-total projection (attempts, targets,
carries, TDs, yards, divided by projected games) -- using our own validated
engine for the players it was built for, and outside expertise exactly where
we have a known, structural blind spot.
"""

import numpy as np
import pandas as pd

from src.ingest.name_matching import (
    build_lastname_pos_to_id,
    build_lastname_team_pos_to_id,
    build_name_to_id,
    norm_name,
)
from src.models.game_environment import PassRateEngine, build_pass_rate_table
from src.models.player_usage import ShareEngine, TdRateEngine, build_player_week_shares
from src.utils.paths import DATA_PROCESSED, DATA_RAW

LEAGUE_AVG_PLAYS = 62.859

# TD-probability calibration (validate_props_pipeline.py): raw Poisson-derived probability
# runs a bit overconfident at the high end (top decile predicted 51.0%, actual rate was 42.1%).
# Fitted on 2022-2025 holdout, corr 0.992 across deciles.
TD_PROB_CALIB_A = 0.046
TD_PROB_CALIB_B = 0.769


if __name__ == "__main__":
    weekly = pd.read_parquet(DATA_RAW / "weekly_player_stats_2016_2025.parquet")
    rosters = pd.read_parquet(DATA_RAW / "weekly_rosters_2016_2025.parquet")
    pbp = pd.read_parquet(DATA_RAW / "pbp_2016_2025.parquet")

    # --- rebuild current-state engines by walking forward through everything we have ---
    shares = build_player_week_shares(weekly)
    tgt_engine = ShareEngine()
    tgt_engine.run_walk_forward(shares, "target_share_calc", gate_col="involved")
    carry_engine = ShareEngine()
    carry_engine.run_walk_forward(shares, "carry_share_calc", gate_col="involved")

    league_rec_td_rate = shares["receiving_tds"].sum() / shares["targets"].sum()
    league_rush_td_rate = shares["rushing_tds"].sum() / shares["carries"].sum()
    rec_td_engine = TdRateEngine(league_rate=league_rec_td_rate, prior_weight=30.0)
    rec_td_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_tds", "targets")
    rush_td_engine = TdRateEngine(league_rate=league_rush_td_rate, prior_weight=30.0)
    rush_td_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_tds", "carries")

    # yards/catch-rate: real but much weaker signal than share or TD rate (validated via
    # prior_weight sweep -- interior optimum exists but the MAE improvement over naive is only
    # 0.5-3.7%, vs. the large gains from share/TD-rate). Included because it's genuine signal,
    # not because it's strong -- set expectations accordingly downstream.
    league_ypt = shares["receiving_yards"].sum() / shares["targets"].sum()
    league_ypc = shares["rushing_yards"].sum() / shares["carries"].sum()
    league_catch_rate = shares["receptions"].sum() / shares["targets"].sum()
    ypt_engine = TdRateEngine(league_rate=league_ypt, prior_weight=80.0)
    ypt_engine.run_walk_forward(shares[shares["targets"] > 0], "receiving_yards", "targets")
    ypc_engine = TdRateEngine(league_rate=league_ypc, prior_weight=300.0)
    ypc_engine.run_walk_forward(shares[shares["carries"] > 0], "rushing_yards", "carries")
    catch_rate_engine = TdRateEngine(league_rate=league_catch_rate, prior_weight=15.0)
    catch_rate_engine.run_walk_forward(shares[shares["targets"] > 0], "receptions", "targets")

    pr_table = build_pass_rate_table(pbp)
    pr_engine = PassRateEngine()
    pr_engine.run_walk_forward(pr_table)

    with open(DATA_PROCESSED / "environment_calibration.txt") as fh:
        calib = dict(line.strip().split("=") for line in fh if line.strip())
    script_slope = float(calib["script_slope"])
    script_intercept = float(calib["script_intercept"])

    name_to_id = build_name_to_id(rosters)
    lastname_team_pos_to_id = build_lastname_team_pos_to_id(rosters, season=2025)
    lastname_pos_to_id = build_lastname_pos_to_id(rosters, season=2025)

    week1_games = pd.read_parquet(DATA_PROCESSED / "week1_2026_predictions.parquet")
    clay_players = pd.read_parquet(DATA_RAW / "clay_2026_player_projections.parquet")
    clay_players["games"] = clay_players["games"].clip(lower=1)

    rows = []
    for g in week1_games.itertuples():
        for team, opp_team, team_margin in [(g.home, g.away, g.our_margin), (g.away, g.home, -g.our_margin)]:
            pass_rate = pr_engine.predict(team) + script_intercept + script_slope * team_margin
            pass_rate = min(max(pass_rate, 0.30), 0.80)
            pass_attempts = LEAGUE_AVG_PLAYS * pass_rate
            rush_attempts = LEAGUE_AVG_PLAYS * (1 - pass_rate)

            team_players = clay_players[
                (clay_players["team"] == team) & clay_players["position"].isin(["WR", "TE", "RB", "QB"])
            ]
            for p in team_players.itertuples():
                name_key = norm_name(p.player)
                pid = name_to_id.get(name_key)
                if pid is None:
                    # nickname fallback (Ken/Kenneth, Cam/Cameron, ...): last name + team + position
                    # is tight enough to be safe -- only resolves when it's unambiguous for that team.
                    last_name = name_key.split()[-1] if name_key else None
                    pid = lastname_team_pos_to_id.get((last_name, team, p.position))
                if pid is None:
                    # third tier: player changed teams in the offseason, so team no longer matches --
                    # drop it and require league-wide uniqueness on last_name+position instead.
                    pid = lastname_pos_to_id.get((last_name, p.position))
                has_history = pid is not None and (tgt_engine.predict(pid) > 0 or carry_engine.predict(pid) > 0)

                if has_history:
                    tgt_share = tgt_engine.predict(pid)
                    carry_share = carry_engine.predict(pid)
                    rec_td_rate = rec_td_engine.predict(pid)
                    rush_td_rate = rush_td_engine.predict(pid)
                    ypt_rate = ypt_engine.predict(pid)
                    ypc_rate = ypc_engine.predict(pid)
                    catch_rate = catch_rate_engine.predict(pid)
                    source = "our model"
                else:
                    # rookie / no-history fallback: Clay's own season projection, per game
                    tgt_share = (p.targets / p.games) / pass_attempts if pass_attempts > 0 else 0.0
                    carry_share = (p.rush_att / p.games) / rush_attempts if rush_attempts > 0 else 0.0
                    rec_td_rate = (p.rec_td / p.targets) if p.targets > 0 else league_rec_td_rate
                    rush_td_rate = (p.rush_td / p.rush_att) if p.rush_att > 0 else league_rush_td_rate
                    ypt_rate = (p.rec_yds / p.targets) if p.targets > 0 else league_ypt
                    ypc_rate = (p.rush_yds / p.rush_att) if p.rush_att > 0 else league_ypc
                    catch_rate = (p.receptions / p.targets) if p.targets > 0 else league_catch_rate
                    source = "Clay fallback (no NFL history)"

                proj_targets = pass_attempts * tgt_share
                proj_carries = rush_attempts * carry_share
                proj_receptions = proj_targets * catch_rate
                proj_rec_yards = proj_targets * ypt_rate
                proj_rush_yards = proj_carries * ypc_rate
                proj_rec_tds = proj_targets * rec_td_rate
                proj_rush_tds = proj_carries * rush_td_rate
                raw_td_prob = 1 - np.exp(-(proj_rec_tds + proj_rush_tds))
                td_prob = np.clip(TD_PROB_CALIB_A + TD_PROB_CALIB_B * raw_td_prob, 0.0, 1.0)

                rows.append(
                    {
                        "team": team, "opp": opp_team, "player": p.player, "position": p.position,
                        "proj_targets": round(proj_targets, 1), "proj_carries": round(proj_carries, 1),
                        "proj_receptions": round(proj_receptions, 1),
                        "proj_rec_yards": round(proj_rec_yards, 1), "proj_rush_yards": round(proj_rush_yards, 1),
                        "proj_rec_tds": round(proj_rec_tds, 3), "proj_rush_tds": round(proj_rush_tds, 3),
                        "td_probability": round(td_prob, 3), "source": source,
                    }
                )

    props = pd.DataFrame(rows)
    props = props[(props["proj_targets"] >= 1.0) | (props["proj_carries"] >= 1.0)]
    props = props.sort_values("td_probability", ascending=False)

    pd.set_option("display.width", 160)
    print("=== WEEK 1, 2026: top 20 projected TD probabilities ===\n")
    print(props.head(20).to_string(index=False))

    print(f"\n=== sample: full team breakdown, ARI @ LAC (out) — showing LAC's offense ===\n")
    print(props[props["team"] == "LAC"].to_string(index=False))

    props.to_parquet(DATA_PROCESSED / "week1_2026_player_props.parquet", index=False)
    print(f"\nsaved week1_2026_player_props.parquet ({len(props)} player-rows)")
