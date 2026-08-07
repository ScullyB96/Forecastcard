"""Cron-firing gate for `railway.toml`'s DST workaround (four UTC firings/
day, covering both the EDT and EST versions of 9pm/10am Eastern -- see
`tz.is_scheduled_firing_hour`'s own docstring for why this exists).

Exit code 0 means "this is a real target Eastern hour -- proceed with the
pipeline"; nonzero means "an extra DST-workaround firing -- skip
harmlessly." `railway.toml`'s startCommand depends on that EXACT polarity
(`python -m src.utils.tz_gate || exit 0; <the real pipeline chain>` --
gate fails -> whole script exits 0 immediately, pipeline never runs; gate
succeeds -> falls through into the real chain) -- don't invert this
without updating that shell chain too. Deliberately never exits nonzero
in a way that would make Railway report the deployment as failed: a
skipped extra firing is expected, routine behavior, not an error.

Run as `python -m src.utils.tz_gate`.
"""

import sys

from src.utils.tz import EASTERN, is_scheduled_firing_hour
from datetime import datetime

if __name__ == "__main__":
    now_eastern = datetime.now(EASTERN)
    if is_scheduled_firing_hour():
        print(f"tz_gate: Eastern hour {now_eastern.hour} is a real target firing -- proceeding.", flush=True)
        sys.exit(0)
    print(f"tz_gate: Eastern hour {now_eastern.hour} is a DST-workaround extra firing -- skipping.", flush=True)
    sys.exit(1)
