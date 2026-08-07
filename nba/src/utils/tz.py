"""Real 'what day is it' for slate-selection purposes.

Every automated pipeline run happens on Railway's UTC-clock container, but
every schedule decision (which day's games to predict, when the site's
displayed slate should flip) is inherently a US Eastern-time concept --
NBA games are scheduled on Eastern time. Using this instead of bare
`datetime.date.today()` anywhere "today"/"tomorrow" matters for slate
selection avoids a day-boundary mismatch during the hours each night
where UTC has already rolled to the next calendar day but Eastern hasn't
yet -- confirmed as a real live bug in MLB's equivalent (never previously
hit here only because NBA's plain `date.today()` default happened to
coincide with its cron's UTC hour, not because it was actually
timezone-aware)."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def eastern_today():
    return datetime.now(EASTERN).date()


def default_slate_date():
    """The date an automated run should target when no explicit date is
    passed. Time-of-day aware, not a fixed offset: firing in the evening
    (the night-before pass, populating tomorrow's slate ahead of the
    midnight flip) targets tomorrow; firing in the morning (the
    refresh pass) targets today. Works for both cron firings with no
    mode flag or argument needed -- the real Eastern wall-clock hour at
    the moment each one actually runs decides which slate it means."""
    now = datetime.now(EASTERN)
    if now.hour >= 17:
        return now.date() + timedelta(days=1)
    return now.date()


TARGET_FIRING_HOURS_ET = (21, 10)  # 9pm ET night-before pass, 10am ET morning refresh pass


def is_scheduled_firing_hour(now: datetime | None = None, target_hours: tuple = TARGET_FIRING_HOURS_ET) -> bool:
    """DST fix (previously "documented, not solved" in railway.toml):
    Railway's cron scheduler is UTC-only with no timezone/DST awareness at
    all (confirmed via Railway's own docs -- no per-cron timezone setting
    exists), so a single literal UTC cron expression is only ever correct
    for ONE of EDT/EST and silently fires an hour off-target for the other
    (squarely within the Nov-Mar NBA season). Fixed at the application
    level instead of the platform level: `railway.toml` now fires FOUR
    UTC times a day (both the EDT and EST versions of each Eastern target
    hour), and this function is the real DST-correct decision of which of
    those firings should actually do the work -- True only when the
    CURRENT Eastern wall-clock hour exactly equals one of `target_hours`.
    `now` is injectable (defaults to the real current time) purely for
    testability -- no other caller should ever pass it."""
    if now is None:
        now = datetime.now(EASTERN)
    return now.hour in target_hours
