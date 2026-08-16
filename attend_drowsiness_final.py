"""
ClassSentinel - Proper OpenCV SFace + YuNet Recognition Test
--------------------------------------------------------------
This is an isolated recognition test. It does NOT modify main.py and does
not write attendance.

Pipeline follows the official OpenCV SFace design:
Camera/Enrollment image
    -> YuNet face detector + 5 landmarks
    -> SFace alignCrop()
    -> SFace feature()
    -> cosine similarity
    -> robust student matching
    -> temporal confirmation

Models:
    face_recognition_sface_2021dec.onnx
    face_detection_yunet_2023mar.onnx
"""

import os
import time
from datetime import datetime
import urllib.request
from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np

import config
import database as db


# ================================================================
# PATHS / MODEL SETTINGS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SFACE_MODEL = PROJECT_ROOT / "face_recognition_sface_2021dec.onnx"
YUNET_MODEL = PROJECT_ROOT / "face_detection_yunet_2023mar.onnx"

SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/"
    "resolve/main/face_recognition_sface_2021dec.onnx?download=true"
)

YUNET_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/"
    "resolve/main/face_detection_yunet_2023mar.onnx?download=true"
)

# SFace reference cosine threshold.
SFACE_COSINE_THRESHOLD = 0.363

# A match must also be clearly better than the runner-up.
SFACE_MIN_MARGIN = 0.025

# Distance-aware recognition:
# below this size we do not attempt identity recognition.
MIN_FACE_SIZE = 60

# Below this size the face is considered "low quality" even if detected.
RELIABLE_FACE_SIZE = 90

# Image-quality gates.
MIN_BLUR_SCORE = 18.0
MIN_DETECTION_CONFIDENCE = 0.80
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 225.0

# Temporal confirmation.
CONFIRM_FRAMES = 5
CONFIRM_RATIO = 0.80

# A different identity needs substantially stronger evidence to replace
# an already locked identity.
SWITCH_CONFIRM_FRAMES = 10
SWITCH_CONFIRM_RATIO = 0.90

TOP_K = 5

# Track settings.
MAX_MISSED_FRAMES = 12
TRACK_CENTER_DISTANCE_RATIO = 1.60
TRACK_IOU_THRESHOLD = 0.10


# ================================================================
# MODEL DOWNLOAD / VALIDATION
# ================================================================

def ensure_model(path: Path, url: str, minimum_bytes: int, label: str):
    if path.exists() and path.stat().st_size >= minimum_bytes:
        return

    if path.exists():
        print(f"Invalid/small {label} model found. Replacing it...")
        try:
            path.unlink()
        except Exception:
            pass

    print(f"Downloading {label} model...")
    print(f"  {url}")

    temp_path = path.with_suffix(path.suffix + ".download")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ClassSentinel/1.0"}
        )

        with urllib.request.urlopen(request, timeout=90) as response, open(temp_path, "wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break

                out.write(chunk)
                downloaded += len(chunk)

                if total:
                    print(
                        f"\r  {downloaded / 1024 / 1024:.1f} / "
                        f"{total / 1024 / 1024:.1f} MB",
                        end=""
                    )

        print()

        if temp_path.stat().st_size < minimum_bytes:
            raise RuntimeError(
                f"Downloaded {label} model is only "
                f"{temp_path.stat().st_size / 1024:.1f} KB."
            )

        temp_path.replace(path)
        print(f"{label} model ready: {path.name}")

    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

        raise RuntimeError(
            f"Could not download {label} model.\n"
            f"Place the official ONNX model at:\n{path}\n"
            f"Original error: {exc}"
        )


# Validate SFace by a sane minimum size. The downloaded file is the
# official model and is much larger than the YuNet detector.
ensure_model(
    SFACE_MODEL,
    SFACE_URL,
    minimum_bytes=30_000_000,
    label="SFace"
)

# YuNet 2023mar is intentionally a very small model (~227-233 KB).
# Do NOT use a 500 KB minimum here. The official OpenCV Zoo file is
# approximately 233 KB and has this SHA-256.
YUNET_EXPECTED_SHA256 = (
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)


def validate_yunet_model(path: Path):
    if not path.exists():
        return False

    size = path.stat().st_size

    # The official model is around 233 KB. Reject obviously tiny pointer files.
    if size < 180_000 or size > 400_000:
        return False

    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().lower() == YUNET_EXPECTED_SHA256


if not validate_yunet_model(YUNET_MODEL):
    print("ERROR: YuNet model is missing or invalid.")
    print(f"Expected file: {YUNET_MODEL}")
    print()
    print("Do NOT let the script delete the model.")
    print("Download the official OpenCV YuNet 2023mar model and place it")
    print("in the project root, then run this file again.")
    print()
    raise SystemExit(1)

print(
    f"YuNet model verified: {YUNET_MODEL.name} "
    f"({YUNET_MODEL.stat().st_size / 1024:.1f} KB)"
)


# ================================================================
# OPENCV MODEL INITIALIZATION
# ================================================================

if not hasattr(cv2, "FaceRecognizerSF"):
    raise SystemExit(
        "ERROR: FaceRecognizerSF is unavailable.\n"
        "Install opencv-contrib-python in the active environment."
    )

if not hasattr(cv2, "FaceDetectorYN"):
    raise SystemExit(
        "ERROR: FaceDetectorYN is unavailable.\n"
        "Your OpenCV installation is too old or incomplete."
    )


try:
    sface = cv2.FaceRecognizerSF.create(
        str(SFACE_MODEL),
        "",
        cv2.dnn.DNN_BACKEND_OPENCV,
        cv2.dnn.DNN_TARGET_CPU,
    )
except Exception as exc:
    raise SystemExit(f"ERROR: Could not load SFace model:\n{exc}")


yunet = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL),
    "",
    (320, 320),
    0.80,       # confidence threshold
    0.30,       # NMS threshold
    5000,
    cv2.dnn.DNN_BACKEND_OPENCV,
    cv2.dnn.DNN_TARGET_CPU,
)

print("SFace loaded.")
print("YuNet loaded.")


# ================================================================
# STUDENT LOADING
# ================================================================

def normalize_id(value):
    return db.normalize_id(str(value))


def normalize_name(value):
    return db.normalize_name(str(value))


def load_enrolled_students():
    students = []

    root = Path(config.STUDENT_PHOTOS_DIR)

    if not root.is_dir():
        raise RuntimeError(
            f"Student photo directory not found: {root}"
        )

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        parts = folder.name.split("_", 1)

        if len(parts) != 2:
            print(f"Skipping folder with invalid name: {folder.name}")
            continue

        student_id = normalize_id(parts[0])
        student_name = normalize_name(parts[1])

        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            students.append({
                "id": student_id,
                "name": student_name,
                "image": image,
                "file": str(image_path),
            })

    return students


enrollment_images = load_enrolled_students()

if not enrollment_images:
    raise SystemExit("No enrollment photos found.")


print(
    f"Loaded {len(enrollment_images)} enrollment photos "
    f"from {len(set(x['id'] for x in enrollment_images))} students."
)


# ================================================================
# YUNET HELPERS
# ================================================================

def detect_faces(image, top_k=5000):
    h, w = image.shape[:2]

    yunet.setInputSize((w, h))
    yunet.setTopK(top_k)

    _, faces = yunet.detect(image)

    if faces is None:
        return np.empty((0, 15), dtype=np.float32)

    return faces


def valid_face(face):
    if face is None or len(face) < 15:
        return False

    x, y, w, h = face[:4]

    return (
        w >= MIN_FACE_SIZE
        and h >= MIN_FACE_SIZE
        and w > 0
        and h > 0
    )


def face_quality(image, face):
    """
    Measure whether the detected face contains enough visual information
    for reliable recognition.

    Returns:
        dict with size, blur, brightness, detector confidence and flags.
    """
    x, y, w, h = map(int, face[:4])

    ih, iw = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)

    if x2 <= x1 or y2 <= y1:
        return {
            "usable": False,
            "reliable": False,
            "size": 0,
            "blur": 0.0,
            "brightness": 0.0,
            "confidence": 0.0,
            "reason": "invalid-crop",
        }

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return {
            "usable": False,
            "reliable": False,
            "size": min(w, h),
            "blur": 0.0,
            "brightness": 0.0,
            "confidence": float(face[14]),
            "reason": "empty-crop",
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    confidence = float(face[14])
    size = int(min(w, h))

    detector_ok = confidence >= MIN_DETECTION_CONFIDENCE
    size_ok = size >= MIN_FACE_SIZE
    blur_ok = blur_score >= MIN_BLUR_SCORE
    brightness_ok = MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS

    usable = (
        detector_ok
        and size_ok
        and blur_ok
        and brightness_ok
    )

    reliable = (
        usable
        and size >= RELIABLE_FACE_SIZE
    )

    if not size_ok:
        reason = "face-too-small"
    elif not detector_ok:
        reason = "low-detection-confidence"
    elif not blur_ok:
        reason = "too-blurry"
    elif not brightness_ok:
        reason = "poor-lighting"
    elif not reliable:
        reason = "low-resolution"
    else:
        reason = "good"

    return {
        "usable": usable,
        "reliable": reliable,
        "size": size,
        "blur": blur_score,
        "brightness": brightness,
        "confidence": confidence,
        "reason": reason,
    }


def recognition_quality_ok(quality):
    return bool(quality["usable"])


def reliable_recognition_quality(quality):
    return bool(
        quality["reliable"]
        and quality["size"] >= RELIABLE_FACE_SIZE
    )


# ================================================================
# SFACE FEATURE EXTRACTION
# ================================================================

def extract_feature(image, face):
    """
    IMPORTANT:
    This follows OpenCV's official SFace flow:
        original image + YuNet's 5 landmarks
        -> SFace.alignCrop()
        -> SFace.feature()
    """

    try:
        aligned = sface.alignCrop(image, face[:14])

        if aligned is None or aligned.size == 0:
            return None

        feature = sface.feature(aligned)

        if feature is None or feature.size == 0:
            return None

        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)

        norm = np.linalg.norm(feature)

        if norm <= 1e-8:
            return None

        return feature / norm

    except Exception:
        return None


# ================================================================
# BUILD ENROLLMENT EMBEDDINGS
# ================================================================

def build_embedding_database():
    database = {}

    usable = 0
    skipped = 0

    for item in enrollment_images:
        image = item["image"]

        faces = detect_faces(image, top_k=1)

        if len(faces) == 0:
            skipped += 1
            continue

        # Enrollment images are expected to contain one student.
        # Pick the highest-confidence detected face.
        face = max(faces, key=lambda row: float(row[14]))

        if not valid_face(face):
            skipped += 1
            continue

        feature = extract_feature(image, face)

        if feature is None:
            skipped += 1
            continue

        student_id = item["id"]

        database.setdefault(
            student_id,
            {
                "name": item["name"],
                "features": [],
            }
        )

        database[student_id]["features"].append(feature)
        usable += 1

    print(
        f"Enrollment embeddings: {usable} usable, "
        f"{skipped} skipped."
    )

    for student_id, data in database.items():
        print(
            f"  {data['name']}: "
            f"{len(data['features'])} embeddings"
        )

    return database


embedding_db = build_embedding_database()

if not embedding_db:
    raise SystemExit(
        "ERROR: No usable SFace embeddings were created."
    )


# ================================================================
# ROBUST MATCHING
# ================================================================

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)

    return float(np.dot(a, b))


def match_student(feature):
    """
    Compare one live embedding against every enrollment embedding.

    For each student:
        sort similarities
        take strongest TOP_K
        average them

    Then require:
        best score >= threshold
        AND
        best score sufficiently above runner-up
    """

    if feature is None:
        return None

    candidates = []

    for student_id, data in embedding_db.items():
        scores = [
            cosine_similarity(feature, stored)
            for stored in data["features"]
        ]

        scores.sort(reverse=True)

        if not scores:
            continue

        k = min(TOP_K, len(scores))

        # Robust score, rather than one lucky enrollment photo.
        robust_score = float(np.mean(scores[:k]))

        # Also keep the strongest individual score for diagnostics.
        peak_score = float(scores[0])

        candidates.append(
            (
                robust_score,
                peak_score,
                student_id,
                data["name"],
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)

    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    best_score = best[0]
    second_score = second[0] if second else -1.0
    margin = best_score - second_score

    if best_score < SFACE_COSINE_THRESHOLD:
        return {
            "name": None,
            "id": None,
            "score": best_score,
            "peak": best[1],
            "margin": margin,
            "reason": "below-threshold",
            "candidates": candidates,
        }

    if second is not None and margin < SFACE_MIN_MARGIN:
        return {
            "name": None,
            "id": None,
            "score": best_score,
            "peak": best[1],
            "margin": margin,
            "reason": "ambiguous",
            "candidates": candidates,
        }

    return {
        "name": best[3],
        "id": best[2],
        "score": best_score,
        "peak": best[1],
        "margin": margin,
        "reason": "accepted",
        "candidates": candidates,
    }


# ================================================================
# SIMPLE FACE TRACKING
# ================================================================

tracks = {}
next_track_id = 0


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_center(box):
    x, y, w, h = box
    return (
        x + w / 2.0,
        y + h / 2.0,
    )


def center_distance(a, b):
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)

    return float(np.hypot(ax - bx, ay - by))


def assign_tracks(face_rows):
    global next_track_id

    detections = [
        tuple(map(int, row[:4]))
        for row in face_rows
    ]

    assignments = {}
    used_tracks = set()
    candidates = []

    for det_index, box in enumerate(detections):
        for track_id, track in tracks.items():

            if track_id in used_tracks:
                continue

            old_box = track["bbox"]

            iou = bbox_iou(box, old_box)

            max_size = max(
                box[2],
                box[3],
                old_box[2],
                old_box[3],
                1,
            )

            distance = center_distance(box, old_box)

            if (
                iou >= TRACK_IOU_THRESHOLD
                or distance <= TRACK_CENTER_DISTANCE_RATIO * max_size
            ):
                score = (
                    iou * 10.0
                    - distance / max_size
                )

                candidates.append(
                    (
                        score,
                        det_index,
                        track_id,
                    )
                )

    candidates.sort(reverse=True)

    used_detections = set()

    for _, det_index, track_id in candidates:

        if det_index in used_detections:
            continue

        if track_id in used_tracks:
            continue

        assignments[det_index] = track_id

        used_detections.add(det_index)
        used_tracks.add(track_id)

    for det_index, box in enumerate(detections):

        if det_index not in assignments:

            track_id = next_track_id
            next_track_id += 1

            tracks[track_id] = {
                "bbox": box,
                "history": deque(maxlen=max(CONFIRM_FRAMES, SWITCH_CONFIRM_FRAMES)),
                "stable_name": None,
                "stable_id": None,
                "missed": 0,
            }

            assignments[det_index] = track_id

    for det_index, track_id in assignments.items():

        tracks[track_id]["bbox"] = detections[det_index]
        tracks[track_id]["missed"] = 0

    assigned_ids = set(assignments.values())

    for track_id in list(tracks.keys()):

        if track_id not in assigned_ids:

            tracks[track_id]["missed"] += 1

            if tracks[track_id]["missed"] > MAX_MISSED_FRAMES:
                del tracks[track_id]

    return assignments


# ================================================================
# TEMPORAL IDENTITY CONFIRMATION
# ================================================================

def update_identity(track, result):
    """
    Identity state machine.

    Important behavior:
    - A weak/unknown observation NEVER replaces a confirmed identity.
    - A different student needs 10 strong observations with >=90% agreement
      before a switch is permitted.
    - This prevents Jayadev -> Akash -> Sanjana flickering caused by
      low-quality frames.
    """
    if result is None or result.get("id") is None:
        track["history"].append(None)

        if track["stable_name"] is not None:
            return track["stable_name"], track["stable_id"], True

        return None, None, False

    student_id = result["id"]
    student_name = result["name"]

    track["history"].append(
        (student_id, student_name)
    )

    valid = [
        item
        for item in track["history"]
        if item is not None
    ]

    if not valid:
        return track["stable_name"], track["stable_id"], False

    counts = Counter(valid)
    candidate, count = counts.most_common(1)[0]
    ratio = count / len(valid)

    # Initial confirmation.
    if track["stable_id"] is None:
        if (
            count >= CONFIRM_FRAMES
            and ratio >= CONFIRM_RATIO
        ):
            track["stable_id"] = candidate[0]
            track["stable_name"] = candidate[1]
            return (
                track["stable_name"],
                track["stable_id"],
                True,
            )

        return None, None, False

    current = (
        track["stable_id"],
        track["stable_name"],
    )

    # Same identity. Keep it locked.
    if candidate == current:
        return (
            track["stable_name"],
            track["stable_id"],
            True,
        )

    # A different identity must prove itself much more strongly.
    if (
        count >= SWITCH_CONFIRM_FRAMES
        and ratio >= SWITCH_CONFIRM_RATIO
    ):
        track["stable_id"] = candidate[0]
        track["stable_name"] = candidate[1]

    return (
        track["stable_name"],
        track["stable_id"],
        True,
    )




# ================================================================
# CLASS SENTINEL PRODUCTION PIPELINE
# ================================================================

import mediapipe as mp
from ultralytics import YOLO

db.init_db()
session_id = db.start_session()
print(f"Session {session_id} started.")

# Drowsiness
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EAR_HISTORY_SIZE = 15
EAR_BASELINE_SAMPLES = 20
EAR_SMOOTH_ALPHA = 0.35
EAR_CLOSED_RATIO = 0.72
EAR_MIN_VALID = 0.08
EAR_MAX_VALID = 0.60
DROWSY_MIN_FACE_SIZE = 45
FACE_CROP_PADDING = 0.30
FACE_CROP_SIZE = 480

# Far-distance EAR safeguards
EAR_FAILURE_TOLERANCE_FRAMES = 8
EAR_DROWSY_CONFIRM_FRAMES = 4
EAR_AWAKE_CONFIRM_FRAMES = 3

def calculate_EAR(points, landmarks, w, h):
    p = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in points]
    horizontal = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    if horizontal <= 1e-8:
        return float("nan")
    v1 = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    return (v1+v2)/(2.0*horizontal)

try:
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
except Exception as exc:
    print(f"WARNING: MediaPipe unavailable: {exc}")
    face_mesh = None

print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")
print("All models loaded. Starting ClassSentinel...")

already_marked_this_run = {}
last_seen_write = {}
drowsy_state = {}
last_drowsy_log_by_key = {}
last_phone_log_by_key = {}
last_engage_log = None
yolo_frame_count = 0
last_yolo_results = None
session_start = datetime.now()

def mark_attendance(student_id, student_name):
    student_id = db.normalize_id(student_id)
    if student_id in already_marked_this_run:
        return
    if not config.ALLOW_REMARK_SAME_DAY and db.is_marked_today(student_id):
        already_marked_this_run[student_id] = True
        return
    inserted = db.mark_attendance(
        session_id, student_id, student_name, "Present"
    )
    if inserted:
        already_marked_this_run[student_id] = True
        print(f"Marked: {student_name} - Present")

def mark_seen(student_id, student_name):
    now = datetime.now()
    if (student_id not in last_seen_write or
        (now-last_seen_write[student_id]).total_seconds() >= 5):
        db.mark_seen(session_id, student_id, student_name)
        last_seen_write[student_id] = now

def engagement(face, phone, sleepy):
    return max(0, 100 - (40 if not face else 0) -
               (35 if phone else 0) - (25 if sleepy else 0))

def score_color(score):
    return (0,255,0) if score >= 70 else ((0,165,255) if score >= 40 else (0,0,255))

def open_camera():
    if getattr(config, "USE_DSHOW", False):
        return cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_DSHOW)
    return cv2.VideoCapture(config.CAMERA_SOURCE)

cap = open_camera()
if not cap.isOpened():
    db.end_session(session_id)
    raise SystemExit(f"Could not open camera source: {config.CAMERA_SOURCE}")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Requested {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}, camera actually gave {actual_w}x{actual_h}.")
print("ClassSentinel running. Press Q to quit.")
print("Recognition: WORKING YuNet + SFace pipeline.")
print("Attendance status: Present.")

failures = 0
frame_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            failures += 1
            if failures >= 30:
                print("Camera feed lost. Ending session.")
                break
            time.sleep(0.05)
            continue
        failures = 0
        frame_counter += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = datetime.now()

        # ---------- SFace recognition: keep the tested pipeline ----------
        faces = detect_faces(frame, top_k=5000)
        valid_faces = [f for f in faces if valid_face(f)]
        assignments = assign_tracks(valid_faces)
        recognized = []
        drowsy_names = []
        drowsy = False

        for i, face in enumerate(valid_faces):
            x,y,fw,fh = map(int, face[:4])
            tid = assignments[i]
            track = tracks[tid]
            quality = face_quality(frame, face)
            result = None

            if recognition_quality_ok(quality):
                feature = extract_feature(frame, face)
                if feature is not None:
                    result = match_student(feature)
                    if (result is not None and
                        not reliable_recognition_quality(quality) and
                        (result["score"] < 0.400 or result["margin"] < 0.045)):
                        result = None

            stable_name, stable_id, locked = update_identity(track, result)

            if stable_name is not None:
                if quality["reliable"]:
                    label, color = stable_name, (0,255,0)
                elif quality["usable"]:
                    label, color = f"{stable_name} | LOW QUALITY", (0,255,255)
                else:
                    label, color = f"{stable_name} | TRACKED", (0,200,255)

                mark_attendance(stable_id, stable_name)
                mark_seen(stable_id, stable_name)
                recognized.append({
                    "name": stable_name, "id": stable_id,
                    "cx": x+fw/2, "cy": y+fh/2
                })
            elif result is not None and result.get("id") is not None:
                label, color = "Verifying...", (0,255,255)
            else:
                label, color = "Unknown", (0,0,255)

            if frame_counter % 10 == 0:
                if result is None:
                    print(f"Track {tid}: {label} | quality={quality['reason']} | size={quality['size']} | blur={quality['blur']:.1f} | brightness={quality['brightness']:.1f}")
                else:
                    print(f"Track {tid}: {label} | score={result['score']:.3f} | peak={result['peak']:.3f} | margin={result['margin']:.3f} | quality={quality['reason']} | size={quality['size']} | blur={quality['blur']:.1f} | {result['reason']}")

            cv2.rectangle(frame,(x,y),(x+fw,y+fh),color,2)
            cv2.putText(frame,f"{label} [T{tid}]",(x,max(25,y-10)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
            cv2.putText(frame,f"size:{quality['size']} blur:{quality['blur']:.0f}",(x,min(h-10,y+fh+20)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)

        # ---------- Drowsiness V3: diagnostic + adaptive EAR ----------
        # Recognition/SFace is intentionally unchanged.
        if face_mesh is not None:
            for face in valid_faces:
                x, y, fw, fh = map(int, face[:4])
                if fw < DROWSY_MIN_FACE_SIZE or fh < DROWSY_MIN_FACE_SIZE:
                    continue

                cx, cy = x + fw / 2, y + fh / 2
                match = min(
                    recognized,
                    key=lambda r: float(np.hypot(r["cx"] - cx, r["cy"] - cy)),
                    default=None
                )
                if match is None:
                    continue

                key = match["id"]
                st = drowsy_state.setdefault(
                    key,
                    {
                        "start": None,
                        "history": [],
                        "baseline_values": [],
                        "baseline": None,
                        "smoothed": None,
                        "valid_frames": 0,
                        "failed_frames": 0,
                        "closed_frames": 0,
                        "open_frames": 0,
                        "calibrated": False,
                    }
                )

                pad_x = int(fw * 0.35)
                pad_y = int(fh * 0.35)
                x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
                x2, y2 = min(w, x+fw+pad_x), min(h, y+fh+pad_y)

                crop = rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Far-distance enhancement:
                # enlarge the face more aggressively while preserving detail.
                # The detected face may be only ~60-90 px wide at long range.
                # A larger crop gives Face Mesh more eye pixels to work with.
                crop = cv2.resize(
                    crop, (800, 800), interpolation=cv2.INTER_LANCZOS4
                )

                # Mild local contrast enhancement. This is deliberately
                # conservative so normal/near faces are not over-processed.
                lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=1.5,
                    tileGridSize=(8, 8)
                )
                l_channel = clahe.apply(l_channel)
                crop = cv2.cvtColor(
                    cv2.merge((l_channel, a_channel, b_channel)),
                    cv2.COLOR_LAB2RGB
                )

                try:
                    mr = face_mesh.process(crop)

                    if not mr.multi_face_landmarks:
                        st["failed_frames"] += 1
                        cv2.putText(
                            frame, "EYES: NOT FOUND",
                            (max(0, x1), max(22, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (0, 0, 255), 2
                        )
                        continue

                    lm = mr.multi_face_landmarks[0].landmark
                    ch, cw = crop.shape[:2]

                    le = calculate_EAR(LEFT_EYE, lm, cw, ch)
                    re = calculate_EAR(RIGHT_EYE, lm, cw, ch)

                    if not np.isfinite(le) or not np.isfinite(re):
                        st["failed_frames"] += 1
                        continue

                    ear = (le + re) / 2.0

                    # Reject impossible landmark geometry.
                    if not (0.05 <= ear <= 0.65):
                        st["failed_frames"] += 1
                        continue

                    st["failed_frames"] = 0
                    st["valid_frames"] += 1

                    if st["smoothed"] is None:
                        st["smoothed"] = ear
                    else:
                        st["smoothed"] = (
                            0.25 * ear + 0.75 * st["smoothed"]
                        )

                    smooth = st["smoothed"]
                    st["history"].append(smooth)
                    if len(st["history"]) > 20:
                        st["history"].pop(0)

                    # Calibrate only while the eyes appear open.
                    # This prevents a user starting the camera with eyes closed
                    # from creating a useless baseline.
                    if (
                        st["baseline"] is None
                        and len(st["baseline_values"]) < 40
                        and smooth > 0.20
                    ):
                        st["baseline_values"].append(smooth)

                    if (
                        st["baseline"] is None
                        and len(st["baseline_values"]) >= 20
                    ):
                        st["baseline"] = float(
                            np.median(st["baseline_values"])
                        )
                        st["calibrated"] = True
                        print(
                            f"EAR calibrated: {match['name']} "
                            f"baseline={st['baseline']:.3f}"
                        )

                    # If calibration has not completed, use a conservative
                    # temporary baseline only for diagnostics.
                    baseline = st["baseline"]
                    if baseline is None:
                        baseline = max(
                            0.26,
                            float(np.median(st["history"]))
                            if st["history"] else 0.30
                        )

                    threshold = max(
                        0.13,
                        min(0.24, baseline * 0.70)
                    )

                    eyes_closed = smooth < threshold

                    if eyes_closed:
                        st["closed_frames"] += 1
                        st["open_frames"] = 0

                        if st["closed_frames"] >= 5:
                            if st["start"] is None:
                                st["start"] = now

                    else:
                        st["open_frames"] += 1
                        st["closed_frames"] = 0

                        if st["open_frames"] >= 4:
                            st["start"] = None

                    elapsed = 0.0
                    if st["start"] is not None:
                        elapsed = (now - st["start"]).total_seconds()

                    # Trigger only after sustained closure.
                    if elapsed >= config.DROWSY_SECONDS:
                        drowsy = True
                        if match["name"] not in drowsy_names:
                            drowsy_names.append(match["name"])

                        last = last_drowsy_log_by_key.get(key)
                        if (
                            last is None
                            or (now-last).total_seconds() >= 5
                        ):
                            db.log_drowsy_alert(
                                session_id, key, match["name"]
                            )
                            last_drowsy_log_by_key[key] = now

                    # Draw eye points.
                    for idx in LEFT_EYE + RIGHT_EYE:
                        px = x1 + int(lm[idx].x * (x2-x1))
                        py = y1 + int(lm[idx].y * (y2-y1))
                        cv2.circle(
                            frame, (px, py), 2, (255, 255, 0), -1
                        )

                    # On-screen diagnostics.
                    status = "CLOSED" if eyes_closed else "OPEN"
                    status_color = (
                        (0, 0, 255) if eyes_closed else (0, 255, 0)
                    )

                    cv2.putText(
                        frame,
                        f"EAR {smooth:.3f}  BASE {baseline:.3f}",
                        (max(0, x1), min(h-45, y2+18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (255, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        f"EYES: {status}  THR:{threshold:.3f}",
                        (max(0, x1), min(h-25, y2+40)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48,
                        status_color, 2
                    )

                    # Show when the system is operating in the far-face
                    # regime, so testing can distinguish distance effects.
                    if fw < 90:
                        cv2.putText(
                            frame,
                            "FAR-EAR ENHANCED",
                            (max(0, x1), max(20, y1-45)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45,
                            (0, 255, 255), 2
                        )

                    if st["start"] is not None:
                        cv2.putText(
                            frame,
                            f"CLOSED: {elapsed:.1f}s",
                            (max(0, x1), max(20, y1-28)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0, 0, 255), 2
                        )

                    if config.DEBUG_PRINT_EAR:
                        print(
                            f"EAR | {match['name']} | "
                            f"face={fw}x{fh} | "
                            f"raw={ear:.3f} | "
                            f"smooth={smooth:.3f} | "
                            f"base={baseline:.3f} | "
                            f"thr={threshold:.3f} | "
                            f"eyes={status} | "
                            f"closed={elapsed:.1f}s | "
                            f"calibrated={st['calibrated']}"
                        )

                except Exception as exc:
                    st["failed_frames"] += 1
                    if config.DEBUG_PRINT_EAR:
                        print(
                            f"EAR ERROR | {match['name']} | "
                            f"face={fw}x{fh} | {exc}"
                        )

        # ---------- YOLO phone detection ----------
        phone=False
        yolo_frame_count+=1
        if yolo_frame_count % config.YOLO_EVERY_N_FRAMES==0:
            last_yolo_results=yolo_model(frame,verbose=False)
        if last_yolo_results is not None:
            for rr in last_yolo_results:
                for box in rr.boxes:
                    if int(box.cls[0])!=67:
                        continue
                    phone=True
                    px1,py1,px2,py2=map(int,box.xyxy[0])
                    cv2.rectangle(frame,(px1,py1),(px2,py2),(0,0,255),2)
                    pcx,pcy=(px1+px2)/2,(py1+py2)/2
                    nearest=min(recognized,key=lambda r:float(np.hypot(r["cx"]-pcx,r["cy"]-pcy)),default=None)
                    name=nearest["name"] if nearest else None
                    sid=nearest["id"] if nearest else None
                    cv2.putText(frame,f"PHONE - {name or 'Unidentified'}",(px1,max(20,py1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                    key=sid or "unidentified"
                    last=last_phone_log_by_key.get(key)
                    if last is None or (now-last).total_seconds()>=5:
                        db.log_phone_alert(session_id,sid,name)
                        last_phone_log_by_key[key]=now

        # ---------- Engagement / UI ----------
        score=engagement(len(valid_faces)>0,phone,drowsy)
        col=score_color(score)
        if last_engage_log is None or (now-last_engage_log).total_seconds()>=10:
            db.log_engagement(session_id,score)
            last_engage_log=now

        cv2.putText(frame,f"Engagement: {score}%",(20,40),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
        cv2.putText(frame,f"Marked : {len(already_marked_this_run)}",(20,75),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
        cv2.putText(frame,f"Phone  : {'YES' if phone else 'NO'}",(160,75),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255) if phone else (0,255,0),2)
        cv2.putText(frame,f"Drowsy : {'YES' if drowsy else 'NO'}",(20,100),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255) if drowsy else (0,255,0),2)
        if drowsy:
            cv2.putText(frame,f"DROWSY: {', '.join(sorted(set(drowsy_names)))}",(20,135),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),3)
        if phone:
            cv2.putText(frame,"PHONE DETECTED!",(20,170),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        if config.SESSION_DURATION_MINUTES is not None:
            remaining=max(0,config.SESSION_DURATION_MINUTES-(now-session_start).total_seconds()/60)
            cv2.putText(frame,f"Time left: {int(remaining):02d}:{int((remaining%1)*60):02d}",(w-220,75),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)

        cv2.imshow("ClassSentinel",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
        if config.SESSION_DURATION_MINUTES is not None and (now-session_start).total_seconds()/60>=config.SESSION_DURATION_MINUTES:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    db.end_session(session_id)
    db.export_attendance_csv(session_id,config.ATTENDANCE_CSV)
    print(f"Session {session_id} closed. Attendance exported to {config.ATTENDANCE_CSV}.")