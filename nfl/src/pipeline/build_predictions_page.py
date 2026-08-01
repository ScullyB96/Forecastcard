"""Render the latest week's game predictions (predictions_{season}_wk{week}.parquet)
as a single self-contained HTML page -- teams and predicted scores, one card per
game. Auto-detects the most recent predictions file in data/processed unless a
season/week is passed explicitly.
"""

import argparse
import glob
import re

import pandas as pd

from src.utils.paths import DATA_PROCESSED

# (full name, primary hex, secondary hex) -- primary used as the card's accent stripe.
TEAMS = {
    "ARI": ("Cardinals", "#97233F", "#000000"),
    "ATL": ("Falcons", "#A71930", "#000000"),
    "BAL": ("Ravens", "#241773", "#9E7C0C"),
    "BUF": ("Bills", "#00338D", "#C60C30"),
    "CAR": ("Panthers", "#0085CA", "#101820"),
    "CHI": ("Bears", "#0B162A", "#C83803"),
    "CIN": ("Bengals", "#FB4F14", "#000000"),
    "CLE": ("Browns", "#311D00", "#FF3C00"),
    "DAL": ("Cowboys", "#041E42", "#869397"),
    "DEN": ("Broncos", "#FB4F14", "#002244"),
    "DET": ("Lions", "#0076B6", "#B0B7BC"),
    "GB": ("Packers", "#203731", "#FFB612"),
    "HOU": ("Texans", "#03202F", "#A71930"),
    "IND": ("Colts", "#002C5F", "#A2AAAD"),
    "JAX": ("Jaguars", "#101820", "#D7A22A"),
    "KC": ("Chiefs", "#E31837", "#FFB81C"),
    "LA": ("Rams", "#003594", "#FFA300"),
    "LAC": ("Chargers", "#0080C6", "#FFC20E"),
    "LV": ("Raiders", "#000000", "#A5ACAF"),
    "MIA": ("Dolphins", "#008E97", "#FC4C02"),
    "MIN": ("Vikings", "#4F2683", "#FFC62F"),
    "NE": ("Patriots", "#002244", "#C60C30"),
    "NO": ("Saints", "#101820", "#D3BC8D"),
    "NYG": ("Giants", "#0B2265", "#A71930"),
    "NYJ": ("Jets", "#125740", "#000000"),
    "PHI": ("Eagles", "#004C54", "#A5ACAF"),
    "PIT": ("Steelers", "#101820", "#FFB612"),
    "SEA": ("Seahawks", "#002244", "#69BE28"),
    "SF": ("49ers", "#AA0000", "#B3995D"),
    "TB": ("Buccaneers", "#D50A0A", "#34302B"),
    "TEN": ("Titans", "#0C2340", "#4B92DB"),
    "WAS": ("Commanders", "#5A1414", "#FFB612"),
}


def latest_predictions_file() -> tuple[int, int, str]:
    candidates = glob.glob(str(DATA_PROCESSED / "predictions_*_wk*.parquet"))
    if not candidates:
        raise FileNotFoundError("no predictions_*.parquet files found in data/processed")
    parsed = []
    for path in candidates:
        m = re.search(r"predictions_(\d+)_wk(\d+)\.parquet$", path)
        if m:
            parsed.append((int(m.group(1)), int(m.group(2)), path))
    season, week, path = max(parsed)
    return season, week, path


def _statline_rows(team_props: pd.DataFrame) -> str:
    if team_props.empty:
        return "<div class=\"stat-row\"><span class=\"stat-name\">No roster data available</span></div>"
    rows = []
    qb = team_props[team_props["position"] == "QB"]
    if not qb.empty:
        r = qb.iloc[0]
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-name\">{r['player']} <small>QB</small></span>"
            f"<span class=\"stat-line\">{r['proj_completions']:.0f}/{r['proj_pass_attempts']:.0f}, "
            f"{r['proj_passing_yards']:.0f} yd, {r['proj_passing_tds']:.1f} TD, {r['proj_interceptions']:.1f} INT</span></div>"
        )
    non_qb = team_props[team_props["position"] != "QB"]
    if "status_note" in non_qb.columns:
        non_qb = non_qb[non_qb["status_note"] != "OUT"]
    skill = non_qb.sort_values("td_probability", ascending=False).head(4)
    for r in skill.itertuples():
        bits = []
        if r.proj_targets > 0.3:
            bits.append(f"{r.proj_receptions:.0f} rec, {r.proj_rec_yards:.0f} yd")
        if r.proj_carries > 0.3:
            bits.append(f"{r.proj_carries:.0f} car, {r.proj_rush_yards:.0f} yd")
        bits.append(f"{r.td_probability:.0%} TD")
        rows.append(
            f"<div class=\"stat-row\"><span class=\"stat-name\">{r.player} <small>{r.position}</small></span>"
            f"<span class=\"stat-line\">{', '.join(bits)}</span></div>"
        )
    return "\n".join(rows)


def _moneyline_badge(moneyline) -> str:
    """Renders a game simulator moneyline (§9.5) as a small badge -- '' if unavailable (no
    market line published yet for this game, same fallback case margin/total already handle)."""
    if moneyline is None or (isinstance(moneyline, float) and pd.isna(moneyline)):
        return ""
    moneyline = int(moneyline)
    return f'<span class="moneyline">{"+" if moneyline > 0 else ""}{moneyline}</span>'


def build_page(season: int, week: int, games: pd.DataFrame, props: pd.DataFrame | None = None) -> str:
    cards = []
    for g in games.itertuples():
        home, away = g.home, g.away
        home_name = TEAMS.get(home, (home, "#555555", "#999999"))
        away_name = TEAMS.get(away, (away, "#555555", "#999999"))
        home_win = g.our_home_pts >= g.our_away_pts

        game_props = props[props["game_id"] == g.game_id] if props is not None else None
        stats_block = ""
        if game_props is not None and not game_props.empty:
            away_stats = _statline_rows(game_props[game_props["team"] == away])
            home_stats = _statline_rows(game_props[game_props["team"] == home])
            stats_block = f"""
          <details class="statlines">
            <summary>Expected statlines</summary>
            <div class="stat-team">{away}</div>
            {away_stats}
            <div class="stat-team">{home}</div>
            {home_stats}
          </details>"""

        # Monte Carlo game simulator's moneyline/win-probability (§9.5) -- computed and
        # validated since round 2, never actually surfaced on this page until now. Recentered
        # on our_margin/our_total as of round 4, so it's consistent with the displayed score.
        away_ml = _moneyline_badge(getattr(g, "sim_away_moneyline", None))
        home_ml = _moneyline_badge(getattr(g, "sim_home_moneyline", None))
        win_prob = getattr(g, "sim_home_win_prob", None)
        confidence_note = ""
        if win_prob is not None and not (isinstance(win_prob, float) and pd.isna(win_prob)):
            fav = home if win_prob >= 0.5 else away
            fav_prob = win_prob if win_prob >= 0.5 else 1 - win_prob
            margin_std = getattr(g, "sim_margin_std", None)
            total_std = getattr(g, "sim_total_std", None)
            spread_bits = f"&plusmn;{margin_std:.0f} margin" if margin_std is not None and not pd.isna(margin_std) else ""
            total_bits = f"&plusmn;{total_std:.0f} total" if total_std is not None and not pd.isna(total_std) else ""
            range_bits = " / ".join(b for b in (spread_bits, total_bits) if b)
            confidence_note = (
                f'<div class="confidence">{fav} {fav_prob:.0%} to win'
                + (f" &middot; {range_bits}" if range_bits else "")
                + "</div>"
            )

        cards.append(f"""
        <div class="card">
          <div class="row {'winner' if not home_win else ''}">
            <span class="swatch" style="background:{away_name[1]}"></span>
            <span class="team">{away}<small>{away_name[0]}</small></span>
            {away_ml}
            <span class="score">{g.our_away_pts:.0f}</span>
          </div>
          <div class="row {'winner' if home_win else ''}">
            <span class="swatch" style="background:{home_name[1]}"></span>
            <span class="team">{home}<small>{home_name[0]}</small></span>
            {home_ml}
            <span class="score">{g.our_home_pts:.0f}</span>
          </div>{confidence_note}{stats_block}
        </div>""")

    return f"""<title>Week {week} &middot; {season} Predictions</title>
<style>
:root {{
  --bg: #0e1116;
  --panel: #161b22;
  --border: #262d38;
  --ink: #e8ecf1;
  --muted: #7f8a99;
  --accent: #e8a33d;
}}
:root[data-theme="light"] {{
  --bg: #f4f5f7;
  --panel: #ffffff;
  --border: #dde1e7;
  --ink: #1a1f27;
  --muted: #667085;
  --accent: #b5720a;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg: #f4f5f7;
    --panel: #ffffff;
    --border: #dde1e7;
    --ink: #1a1f27;
    --muted: #667085;
    --accent: #b5720a;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  padding: 2.5rem 1.25rem 4rem;
}}
.wrap {{ max-width: 900px; margin: 0 auto; }}
header {{ margin-bottom: 2rem; text-align: center; }}
.eyebrow {{
  font-size: 0.75rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
}}
h1 {{
  font-size: clamp(1.6rem, 4vw, 2.3rem);
  margin: 0.35rem 0 0;
  text-wrap: balance;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1rem;
}}
.card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.85rem 1.1rem;
}}
.row {{
  display: grid;
  grid-template-columns: 10px 1fr auto auto;
  align-items: center;
  gap: 0.7rem;
  padding: 0.4rem 0;
}}
.row + .row {{ border-top: 1px solid var(--border); }}
.swatch {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
}}
.team {{
  font-weight: 600;
  font-size: 1rem;
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  color: var(--muted);
}}
.team small {{
  font-weight: 400;
  font-size: 0.8rem;
  color: var(--muted);
  opacity: 0.75;
}}
.row.winner .team {{ color: var(--ink); }}
.score {{
  font-variant-numeric: tabular-nums;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--muted);
  min-width: 2ch;
  text-align: right;
}}
.row.winner .score {{ color: var(--accent); }}
.moneyline {{
  font-variant-numeric: tabular-nums;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--muted);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.1rem 0.5rem;
  white-space: nowrap;
}}
.confidence {{
  font-size: 0.75rem;
  color: var(--muted);
  padding: 0.3rem 0 0.1rem;
  border-top: 1px dashed var(--border);
  margin-top: 0.15rem;
}}
.statlines {{
  margin-top: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px solid var(--border);
}}
.statlines summary {{
  cursor: pointer;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.stat-team {{
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin: 0.6rem 0 0.15rem;
}}
.stat-row {{
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  font-size: 0.82rem;
  padding: 0.2rem 0;
  color: var(--muted);
}}
.stat-name {{ color: var(--ink); white-space: nowrap; }}
.stat-name small {{ color: var(--muted); font-weight: 400; }}
.stat-line {{
  font-variant-numeric: tabular-nums;
  text-align: right;
  white-space: nowrap;
}}
footer {{
  text-align: center;
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 2.5rem;
}}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">{season} Season</div>
    <h1>Week {week} Predictions</h1>
  </header>
  <div class="grid">
    {''.join(cards)}
  </div>
  <footer>Generated by the model's own power ratings, QB and injury adjustments &mdash; predicted final scores.</footer>
</div>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.season and args.week:
        path = DATA_PROCESSED / f"predictions_{args.season}_wk{args.week}.parquet"
        season, week = args.season, args.week
    else:
        season, week, path = latest_predictions_file()

    games = pd.read_parquet(path)
    props_path = DATA_PROCESSED / f"props_{season}_wk{week}.parquet"
    props = pd.read_parquet(props_path) if props_path.exists() else None
    html = build_page(season, week, games, props)
    out_path = args.out or str(DATA_PROCESSED / f"predictions_page_{season}_wk{week}.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(games)} games, week {week} {season})")
