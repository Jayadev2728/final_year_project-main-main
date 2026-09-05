"""
report_generator.py
---------------------
Builds a clean PDF report for one session: attendance, phone-use alerts,
drowsiness alerts, and an engagement summary. Pulls from the same
database the dashboard reads from, so the PDF and the live dashboard can
never show different numbers.

Used by backend/app.py's /api/report/<session_id>/pdf endpoint.
"""

from fpdf import FPDF
from datetime import datetime
import database as db

TEAL  = (42, 170, 164)
DARK  = (18, 35, 56)
GREY  = (104, 119, 137)
LIGHT = (242, 245, 248)
RED   = (230, 60, 70)
AMBER = (230, 150, 30)


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 25, "F")
        self.set_font("Helvetica", "B", 17)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, "SmartMonitor", ln=True)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(210, 225, 238)
        self.cell(0, 5, "SMART CLASSROOM MONITORING & ANALYTICS", ln=True)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.8)
        self.line(10, 20, 200, 20)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(*LIGHT)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 7, f"Smart Attendance | Session Performance Report | Page {self.page_no()}", align="C")

    def section_title(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*DARK)
        self.cell(0, 8, text, ln=True)
        self.set_draw_color(*LIGHT)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def table_header(self, widths, labels):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*DARK)
        self.set_text_color(255, 255, 255)
        for w, label in zip(widths, labels):
            self.cell(w, 8, label, border=0, fill=True, align="L")
        self.ln()

    def table_row(self, widths, values, fill):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.set_fill_color(*LIGHT) if fill else self.set_fill_color(255, 255, 255)
        for w, val in zip(widths, values):
            self.cell(w, 7, str(val), border=0, fill=True, align="L")
        self.ln()


def generate_report_pdf(session_id, output_path):
    session    = db.get_session(session_id)
    students   = db.get_student_summary(session_id)
    phone      = db.get_phone_alerts(session_id, limit=100000)
    drowsy     = db.get_drowsy_alerts(session_id, limit=100000)
    summary    = db.get_summary(session_id)
    absentees  = db.get_absentees(session_id)

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Session info ──────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GREY)
    start = session["start_time"] if session else "-"
    end   = session["end_time"] if session and session["end_time"] else "In progress"
    pdf.cell(0, 6, f"Session #{session_id}   |   Start: {start}   |   End: {end}", ln=True)
    pdf.ln(2)

    # ── Summary cards (as a compact row) ────────────────────────
    pdf.section_title("Executive Summary")
    widths = [47, 47, 47, 47]
    labels = ["Students Present", "Phone Alerts", "Drowsy Alerts", "Avg Engagement"]
    values = [summary["total_marked"], summary["phone_alerts"], summary["drowsy_alerts"], f"{summary['avg_engagement']}%"]
    pdf.table_header(widths, labels)
    pdf.table_row(widths, values, fill=True)
    pdf.ln(4)

    # ── Engagement analytics ───────────────────────────────────
    engagement = db.get_engagement_log(session_id)
    scores = [float(e.get("score", 0)) for e in engagement if e.get("score") is not None]
    pdf.section_title("Engagement Analysis")
    if scores:
        avg_score = sum(scores) / len(scores)
        peak_score = max(scores)
        low_score = min(scores)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*DARK)
        pdf.cell(62, 7, f"Average: {avg_score:.1f}%")
        pdf.cell(62, 7, f"Peak: {peak_score:.1f}%")
        pdf.cell(62, 7, f"Lowest: {low_score:.1f}%", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, f"Recorded observations: {len(scores)}", ln=True)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No engagement observations were recorded.", ln=True)

    # ── Unified per-student summary — the main table ────────────
        pdf.section_title(f"Student-Level Performance ({len(students)} present)")
    if students:
        widths = [28, 40, 22, 22, 22, 22, 24]
        pdf.table_header(widths, ["ID", "Name", "Attend.", "First Seen", "Phone", "Drowsy", "Engage."])
        for i, s in enumerate(students):
            pdf.table_row(widths, [
                s["student_id"], s["student_name"], s["attendance"],
                s["first_seen"], s["phone_alerts"], s["drowsy_alerts"], f"{s['engagement']}%"
            ], fill=(i % 2 == 0))
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No students detected in this session.", ln=True)

    # ── Absentees ────────────────────────────────────────────
    pdf.section_title(f"Absentees ({len(absentees)})")
    if absentees:
        widths = [40, 90]
        pdf.table_header(widths, ["ID", "Name"])
        for i, a in enumerate(absentees):
            pdf.table_row(widths, [a["student_id"], a["student_name"]], fill=(i % 2 == 0))
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "Everyone enrolled was present.", ln=True)

            # ── Alert logs (deduplicated per student) ──────────────────────
    pdf.ln(4)

    # ---------------------------------------------------------------
    # PHONE ALERTS
    # ---------------------------------------------------------------
    phone_by_student = {}

    for row in phone:
        student_id = row.get("student_id")
        student_name = row.get("student_name") or "Unidentified"

        key = student_id if student_id is not None else student_name

        if key not in phone_by_student:
            phone_by_student[key] = {
                "student_name": student_name,
                "first_time": row.get("time", "-"),
                "count": 0
            }

        phone_by_student[key]["count"] += 1

    pdf.section_title(
        f"Phone-Use Alert Summary ({len(phone_by_student)} students)"
    )

    if phone_by_student:
        widths = [45, 65, 30]

        pdf.table_header(
            widths,
            ["First Detected", "Student", "Occurrences"]
        )

        for i, item in enumerate(phone_by_student.values()):
            pdf.table_row(
                widths,
                [
                    item["first_time"],
                    item["student_name"],
                    item["count"]
                ],
                fill=(i % 2 == 0)
            )
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(0, 6, "No phone usage detected.", ln=True)


    # ---------------------------------------------------------------
    # DROWSINESS ALERTS
    # ---------------------------------------------------------------

    drowsy_by_student = {}

    for row in drowsy:
        student_id = row.get("student_id")
        student_name = row.get("student_name") or "Unidentified"

        # Use student ID when available.
        # Otherwise use the student name.
        key = student_id if student_id else student_name

        if key not in drowsy_by_student:
            drowsy_by_student[key] = {
                "time": row.get("time", "-"),
                "student_id": student_id,
                "student_name": student_name,
                "count": 0
            }

        drowsy_by_student[key]["count"] += 1

    unique_drowsy = list(drowsy_by_student.values())

    pdf.section_title(
        f"Drowsiness Alert Summary ({len(unique_drowsy)} students)"
    )

    if unique_drowsy:
        widths = [40, 70, 35, 35]

        pdf.table_header(
            widths,
            ["First Alert", "Student", "Status", "Occurrences"]
        )

        for i, row in enumerate(unique_drowsy):
            pdf.table_row(
                widths,
                [
                    row["time"],
                    row["student_name"],
                    "Drowsy",
                    row["count"]
                ],
                fill=(i % 2 == 0)
            )

    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GREY)
        pdf.cell(
            0,
            6,
            "No drowsiness detected.",
            ln=True
        )

    # ---------------------------------------------------------------
    # CREATE THE PDF
    # ---------------------------------------------------------------

    pdf.output(output_path)

    return output_path