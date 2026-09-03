"""
database.py
------------
SQLite database layer for ClassSentinel.

This module is the single source of truth for the schema and all
read/write operations. Two different programs use it:
  1. detection_system.py  -> WRITES data (attendance, alerts, engagement)
  2. backend/app.py       -> READS data to serve the dashboard API

Keeping all SQL in one place means both programs always agree on the
schema, and the dashboard can never see half-written rows.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager
import os

# Single shared DB file at the project root, regardless of which
# script (detection_system.py or backend/app.py) imports this module.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classentinel.db")


def normalize_id(student_id):
    """Uppercase + strip, so '4bd23IS050' and '4BD23IS050' are treated as the same student."""
    return student_id.strip().upper()


def normalize_name(student_name):
    """Capitalizes each word, but preserves short all-caps words (KL, JS, BS)
    as initials instead of mangling them — Python's plain .title() turns
    'Prashanth KL' into 'Prashanth Kl', which is wrong for Indian naming
    conventions that commonly include initials."""
    words = student_name.strip().split()
    fixed = []
    for word in words:
        if len(word) <= 3 and word.isalpha():
            fixed.append(word.upper())   # treat short words as initials: kl/KL/Kl -> KL
        else:
            fixed.append(word.capitalize())
    return " ".join(fixed)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # allows detection script + API to read/write concurrently
    return conn


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates all tables if they don't already exist. Safe to call every startup."""
    with db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(session_id, student_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS phone_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                student_id TEXT,
                student_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drowsy_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                student_id TEXT,
                student_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS engagement_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                score INTEGER NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_presence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                UNIQUE(session_id, student_id)
            )
        """)
    _migrate_add_column("phone_alerts", "student_id", "TEXT")
    _migrate_add_column("phone_alerts", "student_name", "TEXT")
    _migrate_add_column("drowsy_alerts", "student_id", "TEXT")
    _migrate_add_column("drowsy_alerts", "student_name", "TEXT")


def _migrate_add_column(table, column, coltype):
    """Adds a column to an existing table if it's not already there. Needed
    because init_db()'s CREATE TABLE IF NOT EXISTS won't add new columns to
    a database file created by an older version of this schema."""
    with db_cursor() as cur:
        cur.execute(f"PRAGMA table_info({table})")
        existing_columns = [row["name"] for row in cur.fetchall()]
        if column not in existing_columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


# ── Sessions ─────────────────────────────────────────────────────
def start_session():
    """Call once when the detection script starts. Returns the new session_id."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (start_time) VALUES (?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
        )
        return cur.lastrowid


def end_session(session_id):
    """Call once when the detection script exits (e.g. on 'q')."""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id)
        )


def get_latest_session():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None


def get_session(session_id):
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_sessions():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]


# ── Attendance ───────────────────────────────────────────────────
def mark_attendance(session_id, student_id, student_name, status):
    """Insert-or-ignore so a student is only ever marked once per session.
    IDs/names are normalized so casing differences don't create duplicate students."""
    student_id   = normalize_id(student_id)
    student_name = normalize_name(student_name)
    now = datetime.now()
    with db_cursor() as cur:
        cur.execute(
            """INSERT OR IGNORE INTO attendance
               (session_id, student_id, student_name, date, time, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, student_id, student_name,
             now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status)
        )
        return cur.rowcount > 0  # True only if this was a new insert


def is_marked(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone() is not None


def is_marked_today(student_id):
    """Checks ALL sessions today, not just the current one — this is what stops
    re-running main.py several times in one day from creating duplicate rows
    for the same student (the exact issue visible in the old attendance.csv)."""
    student_id = normalize_id(student_id)
    today = datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
            (student_id, today)
        )
        return cur.fetchone() is not None


def get_attendance(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM attendance WHERE session_id = ? ORDER BY time",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def get_todays_attendance_status(student_id):
    """Looks up a student's official attendance status for TODAY, regardless
    of which session actually recorded it. This is what lets a later
    session's report correctly show someone as 'Late'/'On Time' even though
    same-day dedup meant THIS session didn't write a new attendance row for
    them — they were already marked earlier today."""
    student_id = normalize_id(student_id)
    today = datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
            (student_id, today)
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ── Session presence (separate from once-a-day attendance) ─────────
def mark_seen(session_id, student_id, student_name):
    """Upserts a 'this student was seen in this session' record — separate
    from mark_attendance's once-a-day dedup. This is what a session's
    report/dashboard uses to show 'who was actually present', since the
    attendance table deliberately won't write a new row for someone
    already marked earlier the same day."""
    student_id   = normalize_id(student_id)
    student_name = normalize_name(student_name)
    now = datetime.now().strftime("%H:%M:%S")
    with db_cursor() as cur:
        cur.execute(
            """INSERT INTO session_presence (session_id, student_id, student_name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, student_id) DO UPDATE SET last_seen = excluded.last_seen""",
            (session_id, student_id, student_name, now, now)
        )


def get_session_presence(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM session_presence WHERE session_id = ? ORDER BY first_seen",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def get_absentees(session_id):
    """
    Enrolled students who were NOT detected at all in this session.
    Enrollment comes from the student photo folders -- the same source
    main.py builds its face database from -- since "who is enrolled" is
    never itself written to the database.
    """
    import config
    from pathlib import Path

    root = Path(config.STUDENT_PHOTOS_DIR)
    enrolled = {}
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            parts = folder.name.split("_", 1)
            if len(parts) != 2:
                continue
            sid = normalize_id(parts[0])
            sname = normalize_name(parts[1])
            enrolled[sid] = sname

    present_ids = {p["student_id"] for p in get_session_presence(session_id)}

    absentees = [
        {"student_id": sid, "student_name": sname}
        for sid, sname in enrolled.items()
        if sid not in present_ids
    ]
    return sorted(absentees, key=lambda a: a["student_name"])


# ── Alerts ───────────────────────────────────────────────────────


# ── Alerts ───────────────────────────────────────────────────────
def log_phone_alert(session_id, student_id=None, student_name=None):
    """student_id/student_name are the closest recognized face to the phone
    at detection time (best-effort proximity match, not guaranteed correct
    if multiple people are close together) — pass None/None if no
    recognized face was near enough to attribute confidently."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO phone_alerts (session_id, time, student_id, student_name) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), student_id, student_name)
        )


def log_drowsy_alert(session_id, student_id=None, student_name=None):
    """student_id/student_name are the closest recognized face to the
    drowsy landmark set at detection time — None/None if no recognized
    face was near enough to attribute confidently."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO drowsy_alerts (session_id, time, student_id, student_name) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), student_id, student_name)
        )


def get_phone_alerts(session_id, limit=50):
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, session_id, time, student_id, student_name "
            "FROM phone_alerts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, int(limit))
        )
        return [dict(r) for r in cur.fetchall()]


def get_drowsy_alerts(session_id, limit=50):
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, session_id, time, student_id, student_name "
            "FROM drowsy_alerts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, int(limit))
        )
        return [dict(r) for r in cur.fetchall()]


def get_phone_alert_count(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as c FROM phone_alerts WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone()["c"]


def get_drowsy_alert_count(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as c FROM drowsy_alerts WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone()["c"]


def get_student_summary(session_id):
    """THE unified per-student table: one row per student actually present
    in this session, combining attendance status, phone alerts, drowsy
    alerts, and a derived engagement score. This is what the report and
    dashboard show instead of several disconnected tables."""
    import config
    presence = get_session_presence(session_id)
    rows = []
    for p in presence:
        sid, sname  = p["student_id"], p["student_name"]
        attendance  = get_todays_attendance_status(sid)
        # Being in this list means session_presence confirms they were
        # detected THIS session -- "Present" is always correct here, even
        # if the once-a-day attendance row belongs to an earlier session
        # from today.
        status      = attendance["status"] if attendance else "Present"
        phone_count  = get_phone_alert_count(session_id, sid)
        drowsy_count = get_drowsy_alert_count(session_id, sid)
        engagement   = max(0, 100 - phone_count * config.PHONE_ALERT_PENALTY
                                    - drowsy_count * config.DROWSY_ALERT_PENALTY)
        at_risk      = (phone_count >= config.AT_RISK_PHONE_THRESHOLD
                         or drowsy_count >= config.AT_RISK_DROWSY_THRESHOLD)
        rows.append({
            "student_id":    sid,
            "student_name":  sname,
            "attendance":    status,
            "first_seen":    p["first_seen"],
            "last_seen":     p["last_seen"],
            "phone_alerts":  phone_count,
            "drowsy_alerts": drowsy_count,
            "engagement":    engagement,
            "at_risk":       at_risk
        })
    return rows


# ── Engagement ───────────────────────────────────────────────────
def log_engagement(session_id, score):
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO engagement_log (session_id, time, score) VALUES (?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), score)
        )


def get_engagement_log(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM engagement_log WHERE session_id = ? ORDER BY id",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]

def get_engagement_heatmap(session_id):
    """
    Session engagement bucketed into 1-minute segments (averaged) — same
    underlying engagement_log data as the line chart, grouped for a
    compact minute-by-minute heatmap strip instead of a dense point-per-
    10-seconds view.
    """
    with db_cursor() as cur:
        cur.execute(
            """SELECT substr(time, 1, 5) as minute, AVG(score) as avg_score
               FROM engagement_log
               WHERE session_id = ?
               GROUP BY minute
               ORDER BY minute""",
            (session_id,)
        )
        rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["avg_score"] = round(r["avg_score"], 1)
    return rows



def get_student_profile(student_id, limit_sessions=20):
    """
    Full cross-session history for one student: attendance status,
    alert counts, and derived engagement for each of their last N
    sessions, plus summary averages. Used by the dashboard's
    per-student history view.
    """
    import config
    student_id = normalize_id(student_id)

    with db_cursor() as cur:
        cur.execute(
            """SELECT student_name FROM session_presence
               WHERE student_id = ? ORDER BY id DESC LIMIT 1""",
            (student_id,)
        )
        row = cur.fetchone()
    if row is None:
        return None
    student_name = row["student_name"]

    with db_cursor() as cur:
        cur.execute(
            """SELECT s.id as session_id, s.start_time,
                      a.status as attendance_status,
                      sp.first_seen
               FROM sessions s
               LEFT JOIN attendance a
                 ON a.session_id = s.id AND a.student_id = ?
               LEFT JOIN session_presence sp
                 ON sp.session_id = s.id AND sp.student_id = ?
               ORDER BY s.id DESC
               LIMIT ?""",
            (student_id, student_id, limit_sessions)
        )
        rows = [dict(r) for r in cur.fetchall()]

    sessions = []
    total_present = 0
    total_phone = 0
    total_drowsy = 0
    engagement_sum = 0
    engagement_count = 0

    for r in rows:
        present = r["first_seen"] is not None
        if present:
            total_present += 1
            phone_count  = get_phone_alert_count(r["session_id"], student_id)
            drowsy_count = get_drowsy_alert_count(r["session_id"], student_id)
            engagement   = max(0, 100 - phone_count * config.PHONE_ALERT_PENALTY
                                        - drowsy_count * config.DROWSY_ALERT_PENALTY)
            total_phone  += phone_count
            total_drowsy += drowsy_count
            engagement_sum += engagement
            engagement_count += 1
        else:
            phone_count = drowsy_count = engagement = None

        sessions.append({
            "session_id":    r["session_id"],
            "start_time":    r["start_time"],
            "present":       present,
            # Presence in THIS session is the real signal -- the once-a-day
            # attendance row only ties to whichever session first recorded
            # them that day, so a later same-day session can be present
            # with no row there at all.
            "attendance":    (r["attendance_status"] or "Present") if present else "Absent",
            "phone_alerts":  phone_count,
            "drowsy_alerts": drowsy_count,
            "engagement":    engagement
        })

    sessions = list(reversed(sessions))  # oldest -> newest

    return {
        "student_id":          student_id,
        "student_name":        student_name,
        "sessions_counted":    len(rows),
        "sessions_present":    total_present,
        "attendance_pct":      round(100 * total_present / len(rows), 1) if rows else 0,
        "avg_engagement":      round(engagement_sum / engagement_count, 1) if engagement_count else 0,
        "total_phone_alerts":  total_phone,
        "total_drowsy_alerts": total_drowsy,
        "sessions":            sessions
    }

def export_attendance_csv(session_id, path="attendance.csv"):
    """Writes attendance for one session out to CSV, matching the format your
    evaluators/report already expect. The database stays the single source of
    truth; CSV is just a export view of it, so the two can never drift apart
    the way attendance.csv and attendence.csv did before."""
    import csv
    rows = get_attendance(session_id)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["student_id", "student_name", "date", "time", "status"])
        for r in rows:
            writer.writerow([r["student_id"], r["student_name"], r["date"], r["time"], r["status"]])

# ── Trends (across multiple sessions) ───────────────────────────────
def get_attendance_trend(limit_sessions=10):
    """
    Class-wide attendance % per session, most recent `limit_sessions` first.
    Used for a trend chart: 'how did overall attendance move over the
    last N classes'. A session's total headcount is taken as the number
    of distinct students who have EVER been enrolled (based on
    student_photos), so % is comparable across sessions even if a
    session had fewer detections.
    """
    import config
    from pathlib import Path

    root = Path(config.STUDENT_PHOTOS_DIR)
    total_enrolled = 0
    if root.is_dir():
        total_enrolled = len([f for f in root.iterdir() if f.is_dir()])
    total_enrolled = max(1, total_enrolled)  # avoid divide-by-zero

    with db_cursor() as cur:
        cur.execute(
            """SELECT s.id as session_id, s.start_time,
                      COUNT(DISTINCT a.student_id) as present_count
               FROM sessions s
               LEFT JOIN attendance a ON a.session_id = s.id
               GROUP BY s.id
               ORDER BY s.id DESC
               LIMIT ?""",
            (limit_sessions,)
        )
        rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        r["total_enrolled"] = total_enrolled
        r["attendance_pct"] = round(100 * r["present_count"] / total_enrolled, 1)

    return list(reversed(rows))  # oldest -> newest, so a chart reads left-to-right


def get_student_attendance_history(student_id, limit_sessions=15):
    """
    One student's attendance status across their last N sessions where
    a session actually occurred (not just sessions they missed entirely
    with zero record) — used for a per-student trend on their profile view.
    """
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            """SELECT s.id as session_id, s.start_time,
                      a.status, a.time as marked_time
               FROM sessions s
               LEFT JOIN attendance a
                 ON a.session_id = s.id AND a.student_id = ?
               ORDER BY s.id DESC
               LIMIT ?""",
            (student_id, limit_sessions)
        )
        rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        r["status"] = r["status"] or "Absent"

    return list(reversed(rows))


# ── Summary (used for the dashboard's top cards) ───────────────────
def get_summary(session_id):
    presence   = get_session_presence(session_id)
    phone      = get_phone_alerts(session_id, limit=100000)
    drowsy     = get_drowsy_alerts(session_id, limit=100000)
    engagement = get_engagement_log(session_id)
    avg_score  = round(sum(e["score"] for e in engagement) / len(engagement), 1) if engagement else 0
    return {
        "total_marked":   len(presence),
        "phone_alerts":   len(phone),
        "drowsy_alerts":  len(drowsy),
        "avg_engagement": avg_score
    }