"""Counter-based random numbers (CRN) for paired A/B simulation (2026-07-22,
per an external reviewer's suggestion). Purely additive/opt-in infrastructure
-- every existing caller of the simulator is completely unaffected unless it
explicitly opts in.

The problem this solves: `validate_game_simulator.py`'s `rng` is a single
`np.random.default_rng(seed)` stream consumed sequentially across the whole
run. Two arms (e.g. "signal on" vs. "signal off") that produce even ONE
different outcome diverge in EVERY subsequent random draw from that point on
-- they are not a paired comparison of the same underlying randomness, just
two independent Monte Carlo runs that happen to start aligned. This is why
`ab_significance.py`'s bootstrap CI (not a naive point-estimate delta) has
been the standard decision tool this session -- the unpaired noise floor is
real and has produced at least one false-positive verdict already (jet-lag,
see MODEL_DOCUMENTATION.md §11.7).

CRN fixes the ROOT CAUSE instead of just measuring around it: each stochastic
decision gets a uniform derived from a hash of a SEMANTIC key (which game,
which trial, which half-inning, which at-bat, which kind of decision) rather
than "whatever's next in a shared stream." Two arms that build the exact same
key for the same decision draw the EXACT SAME uniform, so a rate-multiplier
change only flips outcomes near the changed probability thresholds -- every
other decision in the game stays synchronized. This collapses the variance of
a paired delta dramatically (the reviewer's estimate: matching a 500-trial
unpaired comparison's precision at 50-100 CRN-paired trials).

Key design choice: the "at-bat index within this half-inning" key component
is robust to divergence exactly where it needs to be. Batting order itself
never depends on random outcomes (batter 1,2,3,...,9,1,2,... regardless of
what happens) -- only how many at-bats occur before 3 outs are recorded does.
So "half-inning H, at-bat index N" identifies the SAME semantic decision in
both arms for every at-bat that occurs in both (i.e. everything before the
half-inning's actual out-count history first differs) -- exactly the
"stay synchronized until the real point of divergence" property CRN needs.
"""

import numpy as np
from scipy.stats import norm

# splitmix64 finalizer constants (public domain, widely used for fast,
# well-distributed integer hashing -- doesn't need cryptographic strength,
# just good uniformity and perfect reproducibility given the same inputs).
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF


def crn_uniform(*keys: int) -> float:
    """Deterministic uniform in [0, 1) derived from an arbitrary tuple of
    integer keys -- same keys always produce the same output, different keys
    are effectively independent draws. Not cryptographic, just fast and
    well-distributed enough for Monte Carlo use."""
    h = _GOLDEN
    for k in keys:
        h = (h + (int(k) & _MASK64) + _GOLDEN) & _MASK64
        h = ((h ^ (h >> 30)) * _MIX1) & _MASK64
        h = ((h ^ (h >> 27)) * _MIX2) & _MASK64
        h = (h ^ (h >> 31)) & _MASK64
    return h / 2**64


def crn_bool(threshold: float, *keys: int) -> bool:
    """True with probability `threshold`, deterministically keyed."""
    return crn_uniform(*keys) < threshold


def crn_choice(options, probs, *keys: int):
    """Sample one of `options` via inverse-CDF against `probs`, deterministically keyed.
    `options`/`probs` must be in the same fixed order for both arms being compared
    (true by construction here -- both always iterate OUTCOMES in the same order)."""
    u = crn_uniform(*keys)
    cum = np.cumsum(probs)
    idx = int(np.searchsorted(cum, u, side="right"))
    return options[min(idx, len(options) - 1)]

def crn_index(n: int, *keys: int) -> int:
    """A uniform-random index in [0, n), deterministically keyed -- replaces
    rng.integers(0, n) (e.g. TransitionTable's historical-row resampling)."""
    u = crn_uniform(*keys)
    return min(int(u * n), n - 1)


def crn_normal(mean: float, std: float, *keys: int) -> float:
    """A deterministic N(mean, std) draw, keyed like every other crn_* helper
    -- via the inverse-CDF (probit) of a single crn_uniform draw, so it costs
    exactly one hash instead of needing a Box-Muller pair. u is clipped away
    from the [0,1) endpoints (matching this module's other numerical-safety
    clips, e.g. matchup.odds' probability clip) since norm.ppf(0) = -inf."""
    u = min(max(crn_uniform(*keys), 1e-9), 1 - 1e-9)
    return mean + std * norm.ppf(u)


# Fixed small integer tags for "decision_type" key components -- keeping
# these as named constants (not raw literals scattered at call sites) so a
# typo can't silently collide two different decision types onto the same key.
DECISION_OUTCOME = 1
DECISION_SB_ATTEMPT = 2
DECISION_SB_SUCCESS = 3
DECISION_TRANSITION = 4
DECISION_BULLPEN_PICK = 5
DECISION_CLOSER_USAGE = 6
DECISION_WEATHER_BUCKET = 7
DECISION_PITCHER_SHOCK = 8  # task #137: latent per-pitcher-appearance "stuff" shock, see game_simulator.py
DECISION_HOOK_FRAILTY = 9  # task #144/#145: per-PA state-conditioned starter-hook decision, see game_simulator.py
DECISION_TIER_SELECT = 10  # task #144 step 4: state-conditioned reliever-tier selection, see game_simulator.py


def half_inning_ordinal(inning: int, is_bottom: bool) -> int:
    """A single deterministic integer identifying "top of inning 1", "bottom
    of inning 1", "top of inning 2", ... -- the key component that anchors
    every per-at-bat decision to a specific half-inning, independent of the
    stochastic outcomes that determine how many at-bats it actually contains."""
    return (inning - 1) * 2 + (1 if is_bottom else 0)
