"""
detector.py
------------
Shared face detection backend, used by main.py, enroll_student.py, and
check_enrollment.py so all three always detect faces the same way.

Two backends, switched via config.FACE_DETECTOR_BACKEND:
  "dnn"  (default) - OpenCV's SSD + ResNet-10 DNN face detector. Much more
          robust to textured backgrounds (curtains, patterned walls) and
          distant/angled faces than Haar — this is what fixed the ghost
          detections and distance issues found during testing.
  "haar" - the original Haar cascade + NMS approach, kept as an automatic
          fallback if the DNN model files are missing, and useful if you
          want to compare both for your report.

Both return the same format — a list of (x, y, w, h) tuples — so the
rest of the pipeline doesn't need to know which backend is active.
"""

import os
import cv2
import config

_dnn_net = None
_haar_detector = None


def _get_haar_detector():
    global _haar_detector
    if _haar_detector is None:
        _haar_detector = cv2.CascadeClassifier(config.get_cascade_path())
    return _haar_detector


def _get_dnn_net():
    global _dnn_net
    if _dnn_net is None:
        base = os.path.dirname(os.path.abspath(__file__))
        prototxt = os.path.join(base, "deploy.prototxt")
        model    = os.path.join(base, "res10_300x300_ssd_iter_140000.caffemodel")
        if not (os.path.exists(prototxt) and os.path.exists(model)):
            raise FileNotFoundError(
                "DNN face detector model files not found (deploy.prototxt / "
                "res10_300x300_ssd_iter_140000.caffemodel). Make sure both are in the "
                "project root, or set config.FACE_DETECTOR_BACKEND = 'haar' to use the fallback."
            )
        _dnn_net = cv2.dnn.readNetFromCaffe(prototxt, model)
    return _dnn_net


def _detect_haar(gray_frame):
    detector = _get_haar_detector()
    boxes, _reject_levels, level_weights = detector.detectMultiScale3(
        gray_frame,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=config.FACE_MIN_SIZE,
        outputRejectLevels=True
    )
    if len(boxes) == 0:
        return []
    scores = [float(w) for w in level_weights]
    indices = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, b)) for b in boxes],
        scores=scores,
        score_threshold=config.FACE_MIN_CONFIDENCE_SCORE,
        nms_threshold=config.FACE_NMS_IOU_THRESHOLD
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
    return [tuple(boxes[i]) for i in indices]


def _detect_dnn(bgr_frame):
    net = _get_dnn_net()
    h, w = bgr_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence >= config.DNN_FACE_CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def detect_faces(bgr_frame, gray_frame=None):
    """Main entry point. Pass the color (BGR) frame — gray_frame is only
    needed for the Haar backend and computed automatically if omitted."""
    if config.FACE_DETECTOR_BACKEND == "dnn":
        try:
            return _detect_dnn(bgr_frame)
        except FileNotFoundError as e:
            print(f"Warning: {e}\nFalling back to Haar cascade for this run.")
            config.FACE_DETECTOR_BACKEND = "haar"
    if gray_frame is None:
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return _detect_haar(gray_frame)