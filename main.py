"""
main.py
--------
ClassSentinel core pipeline: LBPH face recognition (attendance), YOLOv8
phone detection, MediaPipe EAR drowsiness detection.

Changes from the original version, based on real issues found in this
project's own attendance.csv:
  - All tunable values now live in config.py (no more different thresholds
    hardcoded in different files)
  - Student IDs/names are normalized (uppercase ID, title-case name) so
    "4bd23IS050" and "4BD23IS050" are treated as one person
  - Attendance writes to SQLite (single source of truth) via database.py,
    and is exported to attendance.csv at the end — so the CSV can never
    drift from the DB the way attendance.csv/attendence.csv did before
  - Won't re-mark someone already marked earlier THE SAME DAY, even across
    separate runs of this script (fixes repeated rows from re-testing)
  - YOLO only runs every Nth frame (config.YOLO_EVERY_N_FRAMES) instead of
    every frame — meaningful FPS improvement, especially over Iriun/network
  - Camera read failures retry a few times instead of instantly ending the
    session (one dropped frame over WiFi shouldn't kill a whole class period)
  - Warns at startup about likely-duplicate student folders (e.g. "Sanjana"
    and "JS Sanjana"), which otherwise silently split one person's training
    data into two separate LBPH classes and hurt recognition accuracy
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher
from ultralytics import YOLO

import config
import database as db
import detector

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
    print("Warning: MediaPipe import failed. Drowsiness detection will be disabled.")
    print(f"Import error: {e}")
    mp_face_mesh = None

# ── EAR Calculation ──────────────────────────────────────────
def calculate_EAR(eye_points, landmarks, w, h):
    p = []
    for idx in eye_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        p.append((x, y))
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    h1 = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h1)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ── Initialize database ──────────────────────────────────────
db.init_db()
session_id = db.start_session()
print(f"Session {session_id} started.")

# ── Load enrolled students (with normalization + duplicate warning) ──
print("Loading enrolled students...")

known_faces, known_names, known_ids = [], [], []
folder_by_norm_id, folder_by_norm_name = {}, {}   # for duplicate detection

face_detector = cv2.CascadeClassifier(config.get_cascade_path())   # still used for LBPH training crop below

if not os.path.isdir(config.STUDENT_PHOTOS_DIR):
    print(f"'{config.STUDENT_PHOTOS_DIR}' folder not found. Enroll students first with enroll_student.py")
    raise SystemExit

for student_folder in os.listdir(config.STUDENT_PHOTOS_DIR):
    folder_path = os.path.join(config.STUDENT_PHOTOS_DIR, student_folder)
    if not os.path.isdir(folder_path):
        continue
    parts = student_folder.split("_", 1)
    if len(parts) < 2:
        print(f"Skipping '{student_folder}' — folder name must be <id>_<name>")
        continue

    student_id   = db.normalize_id(parts[0])
    student_name = db.normalize_name(parts[1])

    # Duplicate detection: same normalized ID under a different folder
    if student_id in folder_by_norm_id and folder_by_norm_id[student_id] != student_folder:
        print(f"WARNING: '{student_folder}' has the same ID as '{folder_by_norm_id[student_id]}' "
              f"— these will be merged as one student ({student_id}). If they're different "
              f"people, fix the ID in one of the folder names.")
    folder_by_norm_id[student_id] = student_folder

    # Duplicate detection: very similar name under a different ID (e.g. "Sanjana" vs "JS Sanjana")
    for existing_name, existing_folder in folder_by_norm_name.items():
        similarity = SequenceMatcher(None, student_name, existing_name).ratio()
        if similarity > 0.6 and existing_folder != student_folder:
            print(f"WARNING: '{student_folder}' looks similar to '{existing_folder}' "
                  f"({student_name} vs {existing_name}). If this is the same person enrolled "
                  f"twice, merge the folders — otherwise recognition accuracy will suffer.")
    folder_by_norm_name[student_name] = student_folder

    for photo_file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, photo_file))
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(img, gray)
        for (x, y, w_f, h_f) in faces:
            face_roi = cv2.resize(gray[y:y+h_f, x:x+w_f], (200, 200))
            known_faces.append(face_roi)
            known_names.append(student_name)
            known_ids.append(student_id)

if len(known_faces) == 0:
    print("No usable training photos found. Run enroll_student.py first.")
    raise SystemExit

unique_names = sorted(set(known_names))
name_to_id = {}
for name, sid in zip(known_names, known_ids):
    name_to_id.setdefault(name, sid)
labels = [unique_names.index(name) for name in known_names]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(known_faces, np.array(labels))
print(f"Loaded {len(unique_names)} students ({len(known_faces)} training photos).")

# ── Load Models ───────────────────────────────────────────────
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")
face_mesh = None
if mp_face_mesh is not None:
    try:
        # We process one detected face crop at a time below.  static_image_mode=True
        # makes FaceMesh re-localize the face inside each crop instead of relying on
        # tracking from a different person's previous crop.  This is more reliable
        # when several students are visible and especially when faces are small.
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as e:
        print("Warning: Failed to initialize MediaPipe FaceMesh. Drowsiness detection will be disabled.")
        print(f"Initialization error: {e}")
print("All models loaded. Starting system...")

# ── Tracking variables ────────────────────────────────────────
already_marked_this_run = {}
track_attendance_identity = {}  # track_id -> first confirmed student ID
last_seen_write = {}          # student_id -> datetime, throttles session_presence updates
yolo_frame_count = 0

# Recognition tracking state. A detector box can move by a few pixels from one
# frame to the next. We therefore keep a short-lived track for each face and
# require several consistent LBPH predictions before changing its identity.
face_tracks = {}
next_track_id = 1
TRACK_MAX_MISSED_FRAMES = 8
TRACK_MAX_CENTER_DISTANCE_RATIO = 0.85
TRACK_IOU_MIN = 0.12
DROWSY_MIN_FACE_SIZE = 45
RECOGNITION_HISTORY_SIZE = 11
RECOGNITION_MIN_VOTES = 7
RECOGNITION_SWITCH_VOTES = 10
RECOGNITION_UNKNOWN_GRACE_FRAMES = 5
RECOGNITION_CONFIRM_RATIO = 0.65
RECOGNITION_SWITCH_RATIO = 0.85

def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def _match_face_tracks(detected_faces, frame_w, frame_h):
    """Match current detector boxes to persistent face tracks."""
    global next_track_id
    assignments = {}
    used_tracks = set()
    max_distance = max(40.0, min(frame_w, frame_h) * TRACK_MAX_CENTER_DISTANCE_RATIO)

    candidates = []
    for di, box in enumerate(detected_faces):
        x, y, fw, fh = box
        cx, cy = x + fw / 2.0, y + fh / 2.0
        for tid, state in face_tracks.items():
            if tid in used_tracks:
                continue
            tx, ty, tw, th = state["box"]
            tcx, tcy = tx + tw / 2.0, ty + th / 2.0
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            iou = _box_iou(box, state["box"])
            if dist <= max_distance or iou >= TRACK_IOU_MIN:
                # Lower score is better. IoU gets priority over raw distance.
                scale = max(fw, fh, tw, th, 1)
                score = (dist / scale) - (iou * 2.5)
                candidates.append((score, di, tid))

    for _, di, tid in sorted(candidates):
        if di in assignments or tid in used_tracks:
            continue
        assignments[di] = tid
        used_tracks.add(tid)

    for di in range(len(detected_faces)):
        if di not in assignments:
            tid = next_track_id
            next_track_id += 1
            face_tracks[tid] = {
                "box": detected_faces[di],
                "missed": 0,
                "votes": [],
                "confirmed_name": None,
                "confirmed_id": None,
                "unknown_streak": 0,
                "last_confidence": None,
            }
            assignments[di] = tid

    for di, tid in assignments.items():
        state = face_tracks[tid]
        state["box"] = detected_faces[di]
        state["missed"] = 0

    assigned_ids = set(assignments.values())
    for tid, state in list(face_tracks.items()):
        if tid not in assigned_ids:
            state["missed"] += 1
            if state["missed"] > TRACK_MAX_MISSED_FRAMES:
                del face_tracks[tid]

    return assignments

def _stable_recognition(track, candidate_name, confidence):
    """Confirm an identity from repeated predictions, not one LBPH frame."""
    if candidate_name is not None:
        track["votes"].append(candidate_name)
        track["unknown_streak"] = 0
    else:
        track["votes"].append(None)
        track["unknown_streak"] += 1

    if len(track["votes"]) > RECOGNITION_HISTORY_SIZE:
        track["votes"].pop(0)

    valid_votes = [v for v in track["votes"] if v is not None]
    if not valid_votes:
        return track["confirmed_name"]

    counts = Counter(valid_votes)
    winner, winner_count = counts.most_common(1)[0]
    winner_ratio = winner_count / len(valid_votes)
    current = track["confirmed_name"]

    if current is None:
        if winner_count >= RECOGNITION_MIN_VOTES and winner_ratio >= RECOGNITION_CONFIRM_RATIO:
            track["confirmed_name"] = winner
            return winner
        return None

    if winner == current:
        return current

    current_count = counts.get(current, 0)
    if (winner_count >= RECOGNITION_SWITCH_VOTES
            and winner_ratio >= RECOGNITION_SWITCH_RATIO
            and winner_count > current_count + 3):
        track["confirmed_name"] = winner
        return winner

    return current

def mark_attendance(student_id, student_name):
    student_id = db.normalize_id(student_id)
    if student_id in already_marked_this_run:
        return
    if not config.ALLOW_REMARK_SAME_DAY and db.is_marked_today(student_id):
        already_marked_this_run[student_id] = True   # don't check the DB again every frame
        return

    now = datetime.now()
    class_start = now.replace(hour=config.CLASS_START_HOUR, minute=config.CLASS_START_MINUTE, second=0)
    status = "Late" if now > class_start else "On Time"

    inserted = db.mark_attendance(session_id, student_id, student_name, status)
    if inserted:
        already_marked_this_run[student_id] = True
        print(f"Marked: {student_name} - {status}")

def mark_seen(student_id, student_name):
    """Records 'present in this session' regardless of the once-a-day
    attendance dedup — throttled to avoid a DB write every single frame."""
    student_id = db.normalize_id(student_id)
    now = datetime.now()
    if student_id not in last_seen_write or (now - last_seen_write[student_id]).seconds >= 5:
        db.mark_seen(session_id, student_id, student_name)
        last_seen_write[student_id] = now

# ── Engagement Score ──────────────────────────────────────────
def calculate_engagement(face_present, phone_detected, drowsy):
    score = 100
    if not face_present: score -= 40
    if phone_detected:   score -= 35
    if drowsy:           score -= 25
    return max(score, 0)

def get_score_color(score):
    if score >= 70: return (0, 255, 0)
    elif score >= 40: return (0, 165, 255)
    else: return (0, 0, 255)

# ── Start camera (with retry on dropped frames) ─────────────────
def open_camera():
    if config.USE_DSHOW:
        return cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_DSHOW)
    return cv2.VideoCapture(config.CAMERA_SOURCE)

cap = open_camera()
if not cap.isOpened():
    print(f"\nERROR: Could not open camera at CAMERA_SOURCE={config.CAMERA_SOURCE}.")
    print("This usually means one of:")
    print(f"  - No camera exists at index {config.CAMERA_SOURCE} on this machine right now")
    print("    (Iriun's index can change between reconnects — run camera.py to recheck)")
    print("  - Another app is currently using the camera")
    print("  - Antivirus/Windows privacy settings are blocking camera access")
    print("Run 'python camera.py' to find the correct index, then update CAMERA_SOURCE in config.py.")
    db.end_session(session_id)
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Requested {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}, camera actually gave {actual_w}x{actual_h}.")
if actual_w < config.CAPTURE_WIDTH:
    print("Note: camera didn't honor the higher resolution request — check Iriun's quality")
    print("setting on your phone, it may be capping the stream below what was requested.")
print("ClassSentinel running... Press Q to quit and generate report.")

# ── Runtime state used by the live loop ────────────────────────
# These are initialized explicitly so the first frame can safely use every
# subsystem without relying on a previous frame.
session_start = datetime.now()
last_yolo_results = None

# Drowsiness state is maintained independently for each recognized student.
drowsy_state = {}
last_drowsy_log_by_key = {}

# Phone/engagement logging throttles.
last_phone_log_by_key = {}
last_engage_log = None

# Face-crop / EAR parameters for adaptive drowsiness.
FACE_CROP_PADDING = 0.15
FACE_CROP_SIZE = 320
EAR_MIN_VALID = 0.05
EAR_MAX_VALID = 0.50
EAR_SMOOTH_ALPHA = 0.35
EAR_HISTORY_SIZE = 15
EAR_BASELINE_SAMPLES = 20
EAR_CLOSED_RATIO = 0.72

consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 30   # ~1 second of dropped frames at 30fps before giving up

while True:
    ret, frame = cap.read()
    if not ret:
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print("Camera feed lost mid-session (dropped connection). Ending session.")
            break
        time.sleep(0.05)
        continue
    consecutive_failures = 0

    h, w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = datetime.now()

    # ── Attendance / stable face recognition ───────────────────
    faces = detector.detect_faces(frame, gray_frame)
    track_assignments = _match_face_tracks(faces, w, h)
    recognized_faces_this_frame = []   # stable identities used by drowsiness/phone attribution

    for face_index, (x, y, fw, fh) in enumerate(faces):
        track_id = track_assignments[face_index]
        track = face_tracks[track_id]
        face_roi = cv2.resize(gray_frame[y:y+fh, x:x+fw], (200, 200))

        candidate_name = None
        confidence = 999.0
        try:
            label, confidence = recognizer.predict(face_roi)
            if 0 <= label < len(unique_names) and confidence < config.CONFIDENCE_THRESHOLD:
                candidate_name = unique_names[label]
        except Exception as e:
            if config.DEBUG_PRINT_CONFIDENCE:
                print(f"Recognition error on track {track_id}: {e}")

        stable_name = _stable_recognition(track, candidate_name, confidence)

        if config.DEBUG_PRINT_CONFIDENCE:
            votes = [v for v in track["votes"] if v is not None]
            print(
                f"Track {track_id}: raw={candidate_name or 'Unknown'} "
                f"conf={confidence:.1f} stable={stable_name or 'Recognizing'} "
                f"votes={votes[-RECOGNITION_HISTORY_SIZE:]}"
            )

        if stable_name is not None:
            name = stable_name
            uid = track.get("confirmed_id") or name_to_id.get(name)
            color = (0, 255, 0)
            if uid is not None:
                # Lock the first confirmed student ID to this physical face track.
                # A later LBPH flicker cannot create attendance for another student.
                locked_uid = track_attendance_identity.get(track_id)
                if locked_uid is None:
                    track_attendance_identity[track_id] = uid
                    locked_uid = uid

                if locked_uid == uid:
                    mark_attendance(uid, name)
                    mark_seen(uid, name)

                recognized_faces_this_frame.append({
                    "name": name, "id": locked_uid,
                    "cx": x + fw / 2, "cy": y + fh / 2,
                    "track_id": track_id
                })
        else:
            name, color = "Recognizing...", (0, 255, 255)

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(frame, f"T{track_id}: {name}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── Drowsiness (face-specific crop + adaptive EAR) ───────────
    # IMPORTANT: MediaPipe is run on each detector face crop, not on the whole
    # classroom frame.  A far-away face therefore gets enlarged before eye
    # landmarks are calculated, and the EAR belongs directly to that detected
    # face instead of being matched later using a nearest-nose heuristic.
    drowsy = False
    drowsy_names_this_frame = []
    face_present = len(faces) > 0

    if face_mesh is not None:
        for (x, y, fw, fh) in faces:
            # Very small detections do not contain enough eye pixels for a reliable
            # EAR measurement.  Recognition can still work, but drowsiness should
            # remain neutral instead of producing a false positive.
            if fw < DROWSY_MIN_FACE_SIZE or fh < DROWSY_MIN_FACE_SIZE:
                continue

            # Find the recognized student attached to THIS detector box.
            matched_name, matched_id = None, None
            box_cx, box_cy = x + fw / 2.0, y + fh / 2.0
            best_box_distance = None
            for f in recognized_faces_this_frame:
                dist = ((f["cx"] - box_cx) ** 2 + (f["cy"] - box_cy) ** 2) ** 0.5
                if best_box_distance is None or dist < best_box_distance:
                    best_box_distance = dist
                    matched_name, matched_id = f["name"], f["id"]

            # Only raise a per-student drowsiness event after the face is recognized.
            # This prevents an unknown/poor-quality face from being attached to a
            # student's drowsiness history.
            if matched_id is None:
                continue

            # Expand the detected face slightly, clamp to the frame, then upscale.
            pad_x = int(fw * FACE_CROP_PADDING)
            pad_y = int(fh * FACE_CROP_PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_CUBIC)

            try:
                # static_image_mode=True above intentionally re-detects the face
                # inside this crop, which is more robust than tracking between
                # unrelated students' crops.
                mesh_result = face_mesh.process(face_crop)
                if not mesh_result.multi_face_landmarks:
                    continue

                landmarks = mesh_result.multi_face_landmarks[0].landmark
                crop_h, crop_w = face_crop.shape[:2]
                left_ear = calculate_EAR(LEFT_EYE, landmarks, crop_w, crop_h)
                right_ear = calculate_EAR(RIGHT_EYE, landmarks, crop_w, crop_h)
                avg_ear = (left_ear + right_ear) / 2.0

                if not np.isfinite(avg_ear) or not (EAR_MIN_VALID <= avg_ear <= EAR_MAX_VALID):
                    continue

                key = matched_id
                state = drowsy_state.setdefault(key, {
                    "start": None,
                    "streak": 0,
                    "history": [],
                    "baseline": None,
                    "baseline_samples": 0,
                    "smoothed": None,
                })

                # Exponential smoothing removes single-frame landmark spikes.
                if state["smoothed"] is None:
                    state["smoothed"] = avg_ear
                else:
                    state["smoothed"] = (
                        EAR_SMOOTH_ALPHA * avg_ear +
                        (1.0 - EAR_SMOOTH_ALPHA) * state["smoothed"]
                    )
                smoothed_ear = state["smoothed"]

                state["history"].append(smoothed_ear)
                if len(state["history"]) > EAR_HISTORY_SIZE:
                    state["history"].pop(0)

                # Build a personal open-eye baseline during the first valid samples.
                # Use the upper half of recent EAR values so a blink/closed-eye frame
                # does not pull the baseline downward.
                if state["baseline_samples"] < EAR_BASELINE_SAMPLES:
                    state["baseline_samples"] += 1
                    recent = state["history"]
                    if recent:
                        sorted_recent = sorted(recent)
                        upper_start = max(0, len(sorted_recent) // 2)
                        upper_values = sorted_recent[upper_start:]
                        state["baseline"] = float(np.median(upper_values))

                baseline = state["baseline"]
                if baseline is None or baseline <= 0:
                    baseline = config.EAR_THRESHOLD / EAR_CLOSED_RATIO

                # Personal threshold.  Keep config.EAR_THRESHOLD as a safety ceiling
                # so the adaptive threshold cannot become absurdly high.  The lower
                # adaptive value is what fixes people whose naturally open-eye EAR is
                # below the old universal 0.23 threshold.
                adaptive_threshold = min(
                    config.EAR_THRESHOLD,
                    baseline * EAR_CLOSED_RATIO
                )

                cv2.putText(
                    frame,
                    f"EAR:{smoothed_ear:.2f}",
                    (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

                if config.DEBUG_PRINT_EAR:
                    print(
                        f"EAR raw:{avg_ear:.3f} smooth:{smoothed_ear:.3f} "
                        f"baseline:{baseline:.3f} threshold:{adaptive_threshold:.3f} "
                        f"({matched_name})"
                    )

                # Require continuous low EAR for DROWSY_SECONDS.  A normal blink
                # should not be long enough to trigger the alert.
                if state["baseline_samples"] >= EAR_BASELINE_SAMPLES and smoothed_ear < adaptive_threshold:
                    state["streak"] = 0
                    if state["start"] is None:
                        state["start"] = now

                    elapsed = (now - state["start"]).total_seconds()
                    if elapsed >= config.DROWSY_SECONDS:
                        drowsy = True
                        drowsy_names_this_frame.append(matched_name)
                        last_log = last_drowsy_log_by_key.get(key)
                        if last_log is None or (now - last_log).total_seconds() >= 5:
                            db.log_drowsy_alert(session_id, matched_id, matched_name)
                            last_drowsy_log_by_key[key] = now
                else:
                    state["streak"] += 1
                    if state["streak"] >= config.EAR_RESET_TOLERANCE_FRAMES:
                        state["start"] = None
                        state["streak"] = 0

                # Draw eye landmark points back onto the original frame.
                for idx in LEFT_EYE + RIGHT_EYE:
                    px = x1 + int(landmarks[idx].x * (x2 - x1))
                    py = y1 + int(landmarks[idx].y * (y2 - y1))
                    cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

            except Exception as e:
                if config.DEBUG_PRINT_EAR:
                    print(f"EAR/FaceMesh error for {matched_name}: {e}")

    # ── Phone Detection (throttled) ────────────────────────────
    phone_detected = False
    yolo_frame_count += 1
    if yolo_frame_count % config.YOLO_EVERY_N_FRAMES == 0:
        last_yolo_results = yolo_model(frame, verbose=False)
    yolo_results = last_yolo_results

    if yolo_results is not None:
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls[0]) == 67:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

                    # Attribute to the nearest recognized face, if one is close enough
                    phone_cx, phone_cy = (x1 + x2) / 2, (y1 + y2) / 2
                    attributed_name, attributed_id = None, None
                    best_distance = None
                    for f in recognized_faces_this_frame:
                        dist = ((f["cx"] - phone_cx) ** 2 + (f["cy"] - phone_cy) ** 2) ** 0.5
                        if best_distance is None or dist < best_distance:
                            best_distance = dist
                            attributed_name, attributed_id = f["name"], f["id"]
                    max_distance = w * config.FACE_MATCH_MAX_DISTANCE_RATIO
                    if best_distance is None or best_distance > max_distance:
                        attributed_name, attributed_id = None, None

                    label_text = f"PHONE - {attributed_name}" if attributed_name else "PHONE - Unidentified"
                    cv2.putText(frame, label_text, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    phone_key = attributed_id if attributed_id else "unidentified"
                    last_log = last_phone_log_by_key.get(phone_key)
                    if last_log is None or (now - last_log).seconds >= 5:
                        db.log_phone_alert(session_id, attributed_id, attributed_name)
                        last_phone_log_by_key[phone_key] = now

    # ── Engagement + UI ────────────────────────────────────────
    score = calculate_engagement(face_present, phone_detected, drowsy)
    score_color = get_score_color(score)
    if last_engage_log is None or (now - last_engage_log).seconds >= 10:
        db.log_engagement(session_id, score)
        last_engage_log = now

    cv2.rectangle(frame, (15,15), (310,50), (40,40,40), -1)
    cv2.rectangle(frame, (15,15), (15+int(score*2.9), 50), score_color, -1)
    cv2.putText(frame, f"Engagement: {score}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(frame, f"Marked : {len(already_marked_this_run)}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Phone  : {'YES' if phone_detected else 'NO'}", (160,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if phone_detected else (0,255,0), 2)
    cv2.putText(frame, f"Drowsy : {'YES' if drowsy else 'NO'}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if drowsy else (0,255,0), 2)
    if drowsy:
        cv2.putText(frame, f"DROWSY: {', '.join(sorted(set(drowsy_names_this_frame)))}", (20,135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
    if phone_detected:
        cv2.putText(frame, "PHONE DETECTED!", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    if config.SESSION_DURATION_MINUTES is not None:
        remaining = max(0, config.SESSION_DURATION_MINUTES - (now - session_start).total_seconds() / 60)
        mins, secs = int(remaining), int((remaining % 1) * 60)
        cv2.putText(frame, f"Time left: {mins:02d}:{secs:02d}", (w - 220, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("ClassSentinel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ── Auto-stop after the configured duration ────────────────
    if config.SESSION_DURATION_MINUTES is not None:
        elapsed_minutes = (now - session_start).total_seconds() / 60
        if elapsed_minutes >= config.SESSION_DURATION_MINUTES:
            print(f"\nSession duration ({config.SESSION_DURATION_MINUTES} min) reached. Ending session.")
            break

cap.release()
cv2.destroyAllWindows()
db.end_session(session_id)
db.export_attendance_csv(session_id, config.ATTENDANCE_CSV)
print(f"Session {session_id} closed. Attendance exported to {config.ATTENDANCE_CSV}.")
print("Run 'python backend/app.py' and open http://localhost:5000 to view the dashboard.")