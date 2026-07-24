"""Residuals-first check (external review, follow-up 2026-07-23), run BEFORE
building any local-transfer parameter: compares the model's own (uninflated,
theta=1) independent-Poisson REGULATION-score margin distribution against the
real regulation-margin distribution, bin by bin. Sec4.13/Sec10.1 only checked
margin=0 (the tie deficit); this extends that check to every margin bin to
distinguish two hypotheses about WHERE the model's tie-mass deficit is
funded from on the real side:

- LOCAL-TRANSFER hypothesis (external review): the model's deficit at margin=0
  is offset by a matching EXCESS concentrated specifically at |margin|=1 --
  i.e. real games "steal" mass from what would have been a one-goal decision
  and convert it into a tie. This would mean a single local (x+1,x)/(x,x+1)
  <-> (x,x) transfer parameter is the structurally correct fix, and that a
  GLOBAL uniform rescale (Sec10's theta) is funding the diagonal from the
  wrong place (all off-diagonal cells, including blowouts).
- GLOBAL-RESCALE hypothesis: the offsetting excess is spread thinly across
  many/all margins, not concentrated at |margin|=1 -- meaning Sec10's uniform
  rescale was already the right shape, and the margin-MAE harm found in
  Sec10.3 is an unavoidable cost of fixing the tie rate this way, not a sign
  the fix was the wrong shape.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.models.overtime_shootout import add_regulation_score
from src.models.validate_dixon_coles_regulation import HOME_REG_RATIO, AWAY_REG_RATIO
from src.models.validate_drift import run_validation as run_current_best
from src.models.final_holdout_check import DEV_MAX_SEASON
from src.utils.paths import DATA_RAW

MAX_GOALS = 12
MARGIN_RANGE = range(-8, 9)  # -8..+8; real dev-set margins never exceed this (checked below)


def _margin_mass_by_bin(lambda_home: float, lambda_away: float, max_goals: int = MAX_GOALS) -> dict:
    home_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    joint = np.outer(home_pmf, away_pmf)
    out = {}
    for k in MARGIN_RANGE:
        xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
        out[k] = joint[(xs - ys) == k].sum()
    return out


if __name__ == "__main__":
    schedule = pd.read_parquet(DATA_RAW / "nhl_schedule_2008_2026.parquet")
    schedule = schedule[(schedule["gameType"] == 2) & (schedule["gameState"] == "OFF")]
    schedule_ot = add_regulation_score(schedule)

    base, _adj = run_current_best()
    base = base[(base["season"] >= 20102011) & (base["season"] < DEV_MAX_SEASON)].copy()
    reg_cols = schedule_ot[["gameId", "reg_home_score", "reg_away_score"]]
    base = base.merge(reg_cols, on="gameId", how="left")
    base["lambda_home_reg"] = base["lambda_home"] * HOME_REG_RATIO
    base["lambda_away_reg"] = base["lambda_away"] * AWAY_REG_RATIO
    base["real_margin"] = base["reg_home_score"] - base["reg_away_score"]

    n = len(base)
    print(f"real regulation-margin range: [{base['real_margin'].min()}, {base['real_margin'].max()}] "
          f"across {n} dev-set games")

    real_freq = base["real_margin"].value_counts(normalize=True).reindex(MARGIN_RANGE, fill_value=0.0)

    predicted_sum = {k: 0.0 for k in MARGIN_RANGE}
    for row in base.itertuples():
        mass = _margin_mass_by_bin(row.lambda_home_reg, row.lambda_away_reg)
        for k in MARGIN_RANGE:
            predicted_sum[k] += mass[k]
    predicted_avg = {k: v / n for k, v in predicted_sum.items()}

    print(f"\n{'margin':>7} {'real_freq':>10} {'model_pred':>11} {'real-model':>11}")
    total_real_minus_model = 0.0
    for k in MARGIN_RANGE:
        diff = real_freq[k] - predicted_avg[k]
        total_real_minus_model += diff
        flag = "  <-- tie" if k == 0 else ("  <-- |margin|=1" if abs(k) == 1 else "")
        print(f"{k:>7} {real_freq[k]:>10.4f} {predicted_avg[k]:>11.4f} {diff:>+11.4f}{flag}")
    print(f"\nsum of (real-model) across all bins shown: {total_real_minus_model:+.4f} "
          f"(should be ~0 if MARGIN_RANGE captures ~all real mass; nonzero residual is tail mass "
          f"outside [-8,+8])")

    deficit_at_0 = real_freq[0] - predicted_avg[0]
    excess_at_pm1 = -(real_freq[1] - predicted_avg[1]) - (real_freq[-1] - predicted_avg[-1])
    excess_elsewhere = sum(-(real_freq[k] - predicted_avg[k]) for k in MARGIN_RANGE if abs(k) not in (0, 1))
    print(f"\ntie deficit (real - model at margin=0): {deficit_at_0:+.4f}")
    print(f"model excess at |margin|=1 specifically (model - real, summed both signs): {excess_at_pm1:+.4f}")
    print(f"model excess at all OTHER margins (|margin|>=2, summed): {excess_elsewhere:+.4f}")
    print(f"fraction of the offsetting excess concentrated at |margin|=1: "
          f"{excess_at_pm1/(excess_at_pm1+excess_elsewhere)*100:.1f}%")
