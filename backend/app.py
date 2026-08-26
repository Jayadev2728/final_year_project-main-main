"""
backend/app.py
---------------
Flask REST API for ClassSentinel.

Reads from the same SQLite database that detection_system.py writes to,
and serves the frontend dashboard. Run this AFTER detection_system.py
has started (so a session already exists), then open:

    http://localhost:5000

The dashboard polls these endpoints every few seconds to stay live.
"""

import os
import sys
import tempfile
from functools import wraps
from flask import Flask, jsonify, send_from_directory, send_file, request, redirect, url_for, session
from flask_cors import CORS
from werkzeug.security import check_password_hash

# Allow "import database" and "import config" to work when this file is run from backend/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database as db
import config
from report_generator import generate_report_pdf

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.secret_key = config.SECRET_KEY
CORS(app, supports_credentials=True)

db.init_db()


# ── Authentication ───────────────────────────────────────────────
PUBLIC_PATHS = {"/login", "/api/login"}


@app.before_request
def require_login():
    if request.path in PUBLIC_PATHS:
        return
    # Let CSS/JS/font assets load on the login page itself before login.
    if request.path.startswith("/static/") or request.path.endswith((".css", ".js", ".woff2", ".woff")):
        return
    if not session.get("logged_in"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect(url_for("login_page"))


@app.route("/login")
def login_page():
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    username = data.get("username", "")
    password = data.get("password", "")
    if username == config.TEACHER_USERNAME and check_password_hash(config.TEACHER_PASSWORD_HASH, password):
        session["logged_in"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Invalid username or password"}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


# ── Frontend ─────────────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Sessions ─────────────────────────────────────────────────────
@app.route("/api/sessions")
def api_sessions():
    return jsonify(db.get_all_sessions())


@app.route("/api/session/latest")
def api_latest_session():
    return jsonify(db.get_latest_session() or {})


# ── Attendance ───────────────────────────────────────────────────
@app.route("/api/attendance/<int:session_id>")
def api_attendance(session_id):
    return jsonify(db.get_attendance(session_id))


@app.route("/api/student-summary/<int:session_id>")
def api_student_summary(session_id):
    return jsonify(db.get_student_summary(session_id))


# ── Alerts ───────────────────────────────────────────────────────
@app.route("/api/alerts/<int:session_id>")
def api_alerts(session_id):
    return jsonify({
        "phone":  db.get_phone_alerts(session_id),
        "drowsy": db.get_drowsy_alerts(session_id)
    })


# ── Engagement ───────────────────────────────────────────────────
@app.route("/api/engagement/<int:session_id>")
def api_engagement(session_id):
    return jsonify(db.get_engagement_log(session_id))


# ── Summary (top cards) ──────────────────────────────────────────
@app.route("/api/summary/<int:session_id>")
def api_summary(session_id):
    return jsonify(db.get_summary(session_id))

# ── Trends ───────────────────────────────────────────────────────
@app.route("/api/trends/attendance")
def api_attendance_trend():
    return jsonify(db.get_attendance_trend(limit_sessions=10))


@app.route("/api/trends/student/<student_id>")
def api_student_attendance_history(student_id):
    return jsonify(db.get_student_attendance_history(student_id, limit_sessions=15))


@app.route("/api/engagement/heatmap/<int:session_id>")
def api_engagement_heatmap(session_id):
    return jsonify(db.get_engagement_heatmap(session_id))


@app.route("/api/student/<student_id>/profile")
def api_student_profile(student_id):
    profile = db.get_student_profile(student_id)
    if profile is None:
        return jsonify({"error": "Student not found"}), 404
    return jsonify(profile)


# ── PDF report ───────────────────────────────────────────────────
@app.route("/api/report/<int:session_id>/pdf")
def api_report_pdf(session_id):
    session = db.get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        output_path = tmp.name
    generate_report_pdf(session_id, output_path)

    date_part = session["start_time"].split(" ")[0]
    filename = f"ClassSentinel_Session{session_id}_{date_part}.pdf"
    return send_file(output_path, mimetype="application/pdf", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    print("ClassSentinel backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)