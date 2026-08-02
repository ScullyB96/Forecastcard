"""Known season-opener dates for sports that are between seasons (2026
offseason) -- shown as a countdown on that sport's page/section instead of a
bare "no runs yet" once the real pipeline starts producing slates.

NFL and NHL dates are from each league's own official schedule-release
announcement (confirmed, not estimated). The NBA has NOT released its
2026-27 schedule as of this writing (expected mid-August 2026) -- its date
here is a media estimate based on the league's usual "third Tuesday of
October" opening-night pattern, flagged as such via `confirmed=False` so
the UI can be honest about it rather than presenting a guess as fact."""

OPENERS = {
    "nfl": {
        "kickoff_utc": "2026-09-10T00:20:00+00:00",  # Wed Sep 9, 8:20pm ET
        "label": "Seahawks vs. Patriots, NFL Kickoff Game (Seattle)",
        "confirmed": True,
    },
    "nhl": {
        "kickoff_utc": "2026-09-29T21:00:00+00:00",  # Tue Sep 29, 5pm ET (earliest opening-night game)
        "label": "Panthers @ Hurricanes, opening night",
        "confirmed": True,
    },
    "nba": {
        "kickoff_utc": "2026-10-20T23:30:00+00:00",  # estimated Tue Oct 20, ~7:30pm ET
        "label": "Opening night (estimated -- full schedule not yet released)",
        "confirmed": False,
    },
}
