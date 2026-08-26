"""
session_report_mailer.py
--------------------------
Generates a PDF report for a session and (optionally) emails it, right
after that session ends. Called once from main_sface_test.py's cleanup
step -- never touches the live detection loop.

The PDF is ALWAYS generated and saved locally into reports/. Emailing
it is best-effort on top of that: if EMAIL_REPORTS_ENABLED is False
(the default) or sending fails for any reason, the PDF still exists
locally and nothing else is affected.
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

    date_part = start_time_str.split(" ")[0]
    filename = f"ClassSentinel_Session{session_id}_{date_part}.pdf"
    output_path = os.path.join(REPORTS_DIR, filename)

    try:
        generate_report_pdf(session_id, output_path)
        print(f"[REPORT] Saved: {output_path}")
    except Exception as exc:
        print(f"[REPORT ERROR] Could not generate PDF: {exc}")
        return

    if not getattr(config, "EMAIL_REPORTS_ENABLED", False):
        return

    try:
        _send_email(output_path, session_id, filename)
        print(f"[REPORT] Emailed to {config.REPORT_RECIPIENT_EMAIL}")
    except Exception as exc:
        print(f"[REPORT ERROR] Email not sent: {exc}")


def _send_email(pdf_path, session_id, filename):
    msg = EmailMessage()
    msg["Subject"] = f"ClassSentinel Session #{session_id} Report"
    msg["From"] = config.EMAIL_SENDER_ADDRESS
    msg["To"] = config.REPORT_RECIPIENT_EMAIL
    msg.set_content(
        f"Attached is the automatically generated report for session #{session_id}.\n\n"
        f"This email was sent automatically by ClassSentinel."
    )

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=filename
        )

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.EMAIL_SENDER_ADDRESS, config.EMAIL_SENDER_APP_PASSWORD)
        server.send_message(msg)