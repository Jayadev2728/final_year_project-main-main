"""
voice_alerts.py
----------------
Background text-to-speech worker for ClassSentinel.

Speaking directly inside the detection loop would block frame capture
for the ~1-2 seconds each phrase takes — the same FPS problem already
fixed for phone detection. So a background thread with a queue picks
up alert phrases and speaks them without ever blocking main.py's
frame loop. Call announce(...) from anywhere in main.py; it returns
immediately.
"""

import threading
import queue
import time

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

_alert_queue = queue.Queue()
_last_spoken = {}  # key -> last spoken time, per-student+type cooldown
COOLDOWN_SECONDS = 12  # don't repeat the same alert for the same person too often


def _worker():
    if not _TTS_AVAILABLE:
        print("WARNING: pyttsx3 not installed. Voice alerts disabled. Run: pip install pyttsx3")
        return

    while True:
        text = _alert_queue.get()
        if text is None:
            break
        try:
            # A fresh engine per phrase avoids a known pyttsx3 issue on
            # Windows (SAPI5) where reusing one engine across multiple
            # say()/runAndWait() calls only speaks the first phrase and
            # silently does nothing on every call after that.
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as exc:
            print(f"Voice alert error: {exc}")


_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()


def announce(key, text):
    """
    key:  a unique string identifying this alert (e.g. "drowsy_4BD23IS050")
          used only for cooldown tracking — never spoken aloud.
    text: the phrase actually spoken.
    """
    now = time.time()
    last = _last_spoken.get(key, 0)
    if now - last < COOLDOWN_SECONDS:
        return
    _last_spoken[key] = now
    _alert_queue.put(text)