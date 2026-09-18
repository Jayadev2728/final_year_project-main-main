"""
timetable_helper.py
--------------------
Reads timetable.json and returns current session info.
Keys in timetable.json use 12hr format: "08:00 AM", "10:30 AM" etc.
"""
import json
import os
from datetime import datetime

TIMETABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "timetable.json"
)

def load_timetable():
    if not os.path.exists(TIMETABLE_PATH):
        return {}
    with open(TIMETABLE_PATH, "r") as f:
        return json.load(f)

def slot_to_mins(key):
    """Convert '08:00 AM' or '10:30 AM' to total minutes since midnight."""
    parts = key.strip().split(" ")
    h, m  = map(int, parts[0].split(":"))
    ampm  = parts[1] if len(parts) > 1 else "AM"
    if ampm == "PM" and h != 12:
        h += 12
    if ampm == "AM" and h == 12:
        h = 0
    return h * 60 + m

def get_current_session():
    """
    Returns current period info dict or Demo Session.
    Matches by finding which slot has started and
    not yet been replaced by the next one.
    """
    now     = datetime.now()
    day     = now.strftime("%A")
    nowMins = now.hour * 60 + now.minute

    tt      = load_timetable()
    day_tt  = tt.get(day, {})

    # Sort slots by time
    keys = sorted(day_tt.keys(), key=slot_to_mins)

    current_key = None
    for i, key in enumerate(keys):
        slot_mins = slot_to_mins(key)
        next_mins = slot_to_mins(keys[i + 1]) if i + 1 < len(keys) else slot_mins + 60
        if slot_mins <= nowMins < next_mins:
            current_key = key
            break

    if current_key:
        info = day_tt[current_key]
        return {
            "subject": info.get("subject"),
            "faculty": info.get("faculty"),
            "email":   info.get("email"),
            "class":   info.get("class"),
            "day":     day,
            "period":  current_key   # already in 12hr format e.g. "08:00 AM"
        }

    # No match — Demo Session
    h    = now.hour
    ampm = "AM" if h < 12 else "PM"
    h12  = h % 12 or 12
    period_display = f"{h12}:{str(now.minute).zfill(2)} {ampm}"

    return {
        "subject": "Mentoring Session",
        "faculty": "Dr. Puneeth S P",
        "email":   "jayadevhn27@gmail.com",
        "class":   "7th Sem IS&E 'A'",
        "day":     day,
        "period":  period_display
    }

def get_all_todays_sessions():
    """Returns all periods for today sorted by time."""
    day    = datetime.now().strftime("%A")
    tt     = load_timetable()
    day_tt = tt.get(day, {})
    return [
        {**v, "period": k}
        for k, v in sorted(day_tt.items(), key=lambda x: slot_to_mins(x[0]))
    ]

def get_session_email():
    """Returns faculty email for current period or fallback."""
    session = get_current_session()
    if session and session.get("email"):
        return session["email"]
    try:
        import config
        return config.REPORT_RECIPIENT_EMAIL
    except Exception:
        return None