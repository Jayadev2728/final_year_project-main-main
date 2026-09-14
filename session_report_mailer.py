"""
session_report_mailer.py  (UPDATED — timetable-aware)
--------------------------
Generates a PDF report for a session and (optionally) emails it right
after the session ends. Called once from main.py's cleanup step.

Changes from original:
  - Email recipient is now dynamically resolved from timetable.json
    based on the current day and hour (which faculty was teaching).
  - Falls back to config.REPORT_RECIPIENT_EMAIL if no match found.
  - Email subject and body now include subject name and faculty name.
  - PDF is ALWAYS saved locally regardless of email status.
"""

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import config
from report_generator import generate_report_pdf

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def generate_and_send(session_id, start_time_str):
    Path(REPORTS_DIR).mkdir(exist_ok=True)

    date_part  = start_time_str.split(" ")[0]
    filename   = f"SmartMonitor_Session{session_id}_{date_part}.pdf"
    output_path = os.path.join(REPORTS_DIR, filename)

    # ── Generate PDF ───────────────────────────────────────────
    try:
        generate_report_pdf(session_id, output_path)
        print(f"[REPORT] Saved: {output_path}")
    except Exception as exc:
        print(f"[REPORT ERROR] Could not generate PDF: {exc}")
        return

    if not getattr(config, "EMAIL_REPORTS_ENABLED", False):
        return

    # ── Resolve recipient from timetable ───────────────────────
    recipient, subject_name, faculty_name = _resolve_timetable_context()

    try:
        _send_email(output_path, session_id, filename,
                    recipient, subject_name, faculty_name)
        print(f"[REPORT] Emailed to {recipient} "
              f"({faculty_name} / {subject_name})")
    except Exception as exc:
        print(f"[REPORT ERROR] Email not sent: {exc}")


def _resolve_timetable_context():
    """
    Returns (recipient_email, subject_name, faculty_name).
    Uses timetable.json to find the current period's faculty.
    Falls back to config defaults if not found.
    """
    try:
        from timetable_helper import get_current_session
        info = get_current_session()
        if info:
            return (
                info.get("email", config.REPORT_RECIPIENT_EMAIL),
                info.get("subject", "N/A"),
                info.get("faculty", "Faculty"),
            )
    except Exception as e:
        print(f"[TIMETABLE] Could not resolve session info: {e}")

    # Fallback
    return config.REPORT_RECIPIENT_EMAIL, "N/A", "Faculty"


def _send_email(pdf_path, session_id, filename,
                recipient, subject_name, faculty_name):
    msg = EmailMessage()
    msg["Subject"] = (
        f"SmartMonitor — Session #{session_id} Report | "
        f"{subject_name} | {faculty_name}"
    )
    msg["From"] = config.EMAIL_SENDER_ADDRESS
    msg["To"]   = recipient

    msg.set_content(
        f"Dear {faculty_name},\n\n"
        f"Please find attached the automated attendance and behavior report "
        f"for Session #{session_id}.\n\n"
        f"Subject  : {subject_name}\n"
        f"Faculty  : {faculty_name}\n\n"
        f"The report includes:\n"
        f"  - Attendance summary with timestamps and late/on-time status\n"
        f"  - Phone use alerts per student\n"
        f"  - Drowsiness alerts per student\n"
        f"  - Per-student engagement scores\n\n"
        f"This email was generated automatically by the Smart Attendance "
        f"and Student Monitoring System.\n\n"
        f"Regards,\n"
        f"Smart Attendance System\n"
        f"Department of IS&E, BIET Davangere"
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=filename
        )

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(config.EMAIL_SENDER_ADDRESS, config.EMAIL_SENDER_APP_PASSWORD)
        server.send_message(msg)