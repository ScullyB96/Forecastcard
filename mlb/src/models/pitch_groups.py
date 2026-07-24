"""Shared pitch-level classification helpers -- pitch_type groups and count
leverage groups. Originally private to catcher_framing.py; promoted here
(2026-07-21) so the new pitch-by-pitch modeling work (pitch_talent.py,
build_pitch_table.py) and any other future pitch-level consumer share one
definition instead of duplicating it."""

# Pitch-type groups -- coarse enough that every cell (crossed with a count
# group) has real sample size, unlike the full ~12-value pitch_type column.
FASTBALL_TYPES = {"FF", "SI", "FC", "FA"}
BREAKING_TYPES = {"SL", "CU", "ST", "KC", "SV", "KN"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "EP"}


def pitch_group(pitch_type: str) -> str:
    if pitch_type in FASTBALL_TYPES:
        return "FB"
    if pitch_type in BREAKING_TYPES:
        return "BR"
    if pitch_type in OFFSPEED_TYPES:
        return "OS"
    return "OTHER"


def count_group(balls: int, strikes: int) -> str:
    """2-strike counts (a real called-strike-calling-tendency confound -- some
    umpires/situations see more generous calls once a batter is in danger of
    a called third strike) and hitter-ahead counts (2-0/3-0/2-1/3-1, where an
    "expanding zone" effect is also documented) are kept distinct from a
    neutral baseline, matching the two count-leverage effects most discussed
    in real umpire/framing research."""
    if strikes == 2:
        return "2strike"
    if balls >= 2 and strikes <= 1:
        return "hitter_ahead"
    return "neutral"


def swing_count_group(balls: int, strikes: int) -> str:
    """Like count_group, but splits 3-0 out of "hitter_ahead" into its own
    bucket -- for the swing-VS-take decision specifically, 3-0 is not a mild
    variant of "hitter ahead," it's a real behavioral discontinuity. Real 2025
    Statcast swing rates: 2-0 40.3%, 2-1 58.2%, 3-1 54.0%, but 3-0 only 8.1%
    -- the "take almost automatically" 3-0 approach would get washed out by
    count_group's single hitter_ahead bucket. Only use this for swing-vs-take
    rate estimation; whiff-vs-foul-vs-inplay and ball-vs-called-strike showed
    no comparable discontinuity on real data, so they keep the plain
    count_group buckets."""
    if strikes == 2:
        return "2strike"
    if balls == 3 and strikes == 0:
        return "3-0"
    if balls >= 2 and strikes <= 1:
        return "hitter_ahead"
    return "neutral"
