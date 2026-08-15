"""
tracker.py
-----------
LBPH re-predicts fresh every single frame with no memory of previous
frames, so a name can flicker between correct and "Unknown" (or between
two names) as confidence jitters slightly around the threshold from one
frame to the next. This smooths that out: each detected face is matched
to the same position from the previous frame, and the displayed name is
a majority vote over its last few raw predictions rather than whatever
this one frame happened to say.
"""

from collections import deque, Counter
import config


class FaceTracker:
    def __init__(self):
        self.tracks = {}   # track_id -> {"cx","cy","history": deque}
        self.next_id = 0

    def update(self, detections):
        """detections: list of dicts with cx, cy, raw_name (a real name or
        'Unknown'). Adds a 'stable_name' key to each and returns the list."""
        used_ids = set()

        for det in detections:
            best_id, best_dist = None, None
            for tid, t in self.tracks.items():
                if tid in used_ids:
                    continue
                dist = ((t["cx"] - det["cx"]) ** 2 + (t["cy"] - det["cy"]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None and best_dist <= config.FACE_TRACK_MAX_DISTANCE:
                track_id = best_id
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {"cx": det["cx"], "cy": det["cy"],
                                          "history": deque(maxlen=config.FACE_TRACK_HISTORY_LEN)}

            track = self.tracks[track_id]
            track["cx"], track["cy"] = det["cx"], det["cy"]
            track["history"].append(det["raw_name"])
            used_ids.add(track_id)

            counts = Counter(track["history"])
            top_name, top_count = counts.most_common(1)[0]
            agreement = top_count / len(track["history"])
            det["stable_name"] = top_name if agreement >= config.FACE_TRACK_MIN_AGREEMENT else "Unknown"

        # drop tracks not matched this frame — they've left the scene
        self.tracks = {tid: t for tid, t in self.tracks.items() if tid in used_ids}
        return detections