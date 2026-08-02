"""Real 'what day is it' for slate-selection purposes.

Every automated pipeline run happens on Railway's UTC-clock container, but
every schedule decision (which day's games to predict, when RotoWire's
lineups apply, when the site's displayed slate should flip) is inherently
a US Eastern-time concept -- MLB games are scheduled on Eastern time.
Using this instead of bare `datetime.date.today()` anywhere "today"/
"tomorrow" matters for slate selection avoids a day-boundary mismatch
during the hours each night where UTC has already rolled to the next
calendar day but Eastern hasn't yet (task #163's follow-up: a real bug
where the site jumped to tomorrow's slate while it was still today in
Eastern time, traced to exactly this UTC/Eastern mismatch).
"""

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
    lineup-confirmation refresh pass) targets today. Works for both cron
    firings with no mode flag or argument needed -- the real Eastern
    wall-clock hour at the moment each one actually runs decides which
    slate it means."""
    now = datetime.now(EASTERN)
    if now.hour >= 17:
        return now.date() + timedelta(days=1)
    return now.date()
