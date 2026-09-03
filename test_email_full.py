import database as db
from session_report_mailer import generate_and_send

session_id = 13
session_info = db.get_session(session_id)
print(f"Testing report + email for session {session_id}...")
generate_and_send(session_id, session_info["start_time"])
print("Done.")