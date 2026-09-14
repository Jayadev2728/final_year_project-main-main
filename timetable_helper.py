"""
timetable_helper.py
--------------------
Reads timetable.json and returns the current session info
based on the current day and hour.

Usage:
    from timetable_helper import get_current_session, get_session_email

"""
import json
import os
import config
from datetime import datetime


TIMETABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "timetable.json"
)


def load_timetable():
    if not os.path.exists(TIMETABLE_PATH):
        return {}
    with open(TIMETABLE_PATH, "r") as f:
        return json.load(f)


def get_current_session():
    """
    Returns current period info.
    If current time does not match any timetable slot,
    returns a DEMO session so the system always works.
    """
    now  = datetime.now()
    day  = now.strftime("%A")
    hour = now.strftime("%H:00")

    tt      = load_timetable()
    day_tt  = tt.get(day, {})
    session = day_tt.get(hour, None)

    if session:
        # Convert period to 12hr format for display
        h = int(hour.split(":")[0])
        ampm = "AM" if h < 12 else "PM"
        h12  = h % 12 or 12
        period_display = f"{h12}:00 {ampm}"
        return {**session, "day": day, "period": period_display}

    # ── No timetable match — return DEMO session ──────────────
        h    = int(hour.split(":")[0])
    ampm = "AM" if h < 12 else "PM"
    h12  = h % 12 or 12
    period_display = f"{h12}:00 {ampm}"

    return {
        "subject": "Demo Session",
        "faculty": "Dr. Puneeth S P",
        "email":   "puneeth@biet.ac.in",
        "class":   "6th Sem IS&E",
        "day":     day,
        "period":  period_display
    }


def get_session_email():
    """
    Returns the faculty email for the current period,
    or falls back to config.REPORT_RECIPIENT_EMAIL.
    """
    session = get_current_session()
    if session and session.get("email"):
        return session["email"]
    try:
        import config
        return config.REPORT_RECIPIENT_EMAIL
    except Exception:
        return None


def get_all_todays_sessions():
    """Returns all periods for today as a list."""
    day = datetime.now().strftime("%A")
    tt  = load_timetable()
    day_tt = tt.get(day, {})
    return [
        {**v, "period": k}
        for k, v in sorted(day_tt.items())
    ]