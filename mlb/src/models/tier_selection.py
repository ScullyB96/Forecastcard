"""Pure, dependency-free tier-conditioned reliever-selection math (task #144
step 4), split into its own module for the SAME reason hook_frailty.py exists:
game_simulator.py needs to call this directly from inside simulate_game's
per-inning loop (where the CURRENT trial's live margin is known), but cannot
import bullpen.py or bullpen_usage_policy.py without a circular import (both
of those import game_simulator.py for OUTCOMES).

Reweights WITHIN the pregame availability-weighted roster (see
bullpen.build_team_bullpen_roster) -- state-conditioning chooses AMONG
available arms, it does not override rest-day availability. Tier labels are
derived directly from each reliever's own walk-forward roster weight (top
third of trailing usage+rest weight = "leverage", middle third = "middle",
bottom third = "mopup") rather than bullpen_usage_policy.build_reliever_tier_log's
season-long tercile, which that module's own docstring flags as NOT
walk-forward-safe (a measurement-only simplification, fine for characterizing
the real policy, not for a live predictive mechanism). See
MODEL_DOCUMENTATION.md sec 11.18/11.22/11.23 for the full story.
"""
import numpy as np

from src.models.hook_frailty import bucket_inning, bucket_margin


def tier_label_from_roster_weights(weights: dict) -> dict:
    """{pid: tier} for a team's own walk-forward roster weights (see
    bullpen.build_team_bullpen_roster) -- ranks by weight descending, top
    third = "leverage", middle third = "middle", bottom third = "mopup"."""
    if not weights:
        return {}
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    third = max(1, round(n / 3))
    labels = {}
    for i, (pid, _) in enumerate(ranked):
        if i < third:
            labels[pid] = "leverage"
        elif i < 2 * third:
            labels[pid] = "middle"
        else:
            labels[pid] = "mopup"
    return labels


def sample_tier_conditioned_reliever(available: dict, tier_labels: dict, tier_by_margin: dict,
                                      closer_by_situation: dict, closer_id, inning: int, margin: int,
                                      profile_lookup, uniform_fn) -> tuple:
    """Draws ONE reliever for THIS inning (or hook moment), conditioned on the
    CURRENT (trial-specific) margin and save-situation, from the WITHOUT-
    REPLACEMENT `available` pool (mutated in place -- a pid is removed once
    drawn, matching bullpen.sample_bullpen_plan's own convention).

    uniform_fn(sub_idx): returns a uniform in [0,1) for sub-decision `sub_idx`
    (0=closer-check, 1=tier-pick, 2=within-tier-pick) -- CRN-keyed or
    plain-rng, supplied by the caller (game_simulator.py).

    Returns (pid, profile) or (None, None) if `available` is empty (caller
    applies its own fallback_profile, matching sample_bullpen_plan's
    existing convention)."""
    if not available:
        return None, None

    inning_b = bucket_inning(inning)
    save_situation = inning >= 7 and 1 <= margin <= 3
    margin_b = bucket_margin(margin)

    closer_prob = closer_by_situation.get((inning_b, save_situation), 0.0)
    if closer_id is not None and closer_id in available and uniform_fn(0) < closer_prob:
        pid = closer_id
        available.pop(pid)
        return pid, profile_lookup(pid)

    tier_probs = tier_by_margin.get(margin_b, {"mopup": 1 / 3, "middle": 1 / 3, "leverage": 1 / 3})
    tier_order = ("leverage", "middle", "mopup")
    tiers_present = [t for t in tier_order if any(tier_labels.get(pid) == t for pid in available)]
    if not tiers_present:
        # no tier label at all for anyone left (shouldn't normally happen) --
        # fall back to a flat weighted draw over the whole remaining pool.
        ids, weights = zip(*available.items())
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        u = uniform_fn(2)
        cum = np.cumsum(weights)
        idx = int(np.searchsorted(cum, u, side="right"))
        pid = ids[min(idx, len(ids) - 1)]
        available.pop(pid)
        return pid, profile_lookup(pid)

    probs_present = np.array([tier_probs.get(t, 0.0) for t in tiers_present], dtype=float)
    total = probs_present.sum()
    probs_present = probs_present / total if total > 0 else np.full(len(tiers_present), 1.0 / len(tiers_present))
    u1 = uniform_fn(1)
    cum1 = np.cumsum(probs_present)
    chosen_tier = tiers_present[min(int(np.searchsorted(cum1, u1, side="right")), len(tiers_present) - 1)]

    pool = {pid: w for pid, w in available.items() if tier_labels.get(pid) == chosen_tier}
    ids, weights = zip(*pool.items())
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    u2 = uniform_fn(2)
    cum2 = np.cumsum(weights)
    idx = int(np.searchsorted(cum2, u2, side="right"))
    pid = ids[min(idx, len(ids) - 1)]
    available.pop(pid)
    return pid, profile_lookup(pid)
