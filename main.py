"""
main.py

Real-time facial emotion recognition via webcam.

This script implements a high-performance video processing pipeline:
1. Main Thread: Handles camera I/O, UI rendering, and user input.
2. Worker Thread: Processes frames for emotion in the background (async).
3. Logging: Records every detection to a CSV file and prints a session summary.

Press 'q' to quit.
Press 's' to save a snapshot.
"""

import cv2
import time
import threading
import logging
import sys
import os
from datetime import datetime

from detector import FaceDetector
from utils.logger import EmotionLogger

# --- Global Settings ---
WINDOW_TITLE = "Emotion Detector"
TARGET_FPS   = 15
FRAME_DELAY  = 1.0 / TARGET_FPS
SNAPSHOT_DIR = "snapshots"

# Suppress DeepFace verbose logging to keep the terminal clean
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

class EmotionWorker(threading.Thread):
    """
    Background worker that continuously analyzes the latest available frame.
    Prevents the UI from lagging due to DeepFace's processing time.
    """
    def __init__(self, detector: FaceDetector):
        super().__init__(daemon=True)
        self.detector   = detector
        self._lock      = threading.Lock()
        self._new_frame = None          # Input: latest frame to process
        self._result    = (None, [])    # Output: (annotated_frame, bboxes)
        self._event     = threading.Event()
        self._stop_flag = False

    def submit(self, frame):
        """Submit a new frame for analysis. Overwrites any unprocessed frame."""
        with self._lock:
            self._new_frame = frame.copy()
        self._event.set()

    def get_result(self):
        """Retrieve the latest processed results without blocking."""
        with self._lock:
            return self._result

    def stop(self):
        """Signal the worker to terminate."""
        self._stop_flag = True
        self._event.set()

    def run(self):
        """Worker loop: waits for a frame, analyzes it, and caches the result."""
        while not self._stop_flag:
            self._event.wait()
            self._event.clear()
            if self._stop_flag:
                break

            with self._lock:
                frame = self._new_frame
                self._new_frame = None

            if frame is None:
                continue

            try:
                # Perform the heavy lifting
                annotated, bboxes = self.detector.detect_faces(frame)
                with self._lock:
                    self._result = (annotated, bboxes)
            except Exception as e:
                log.error(f"Worker analysis error: {e}")

# --- UI Helper Functions ---

def draw_fps_overlay(frame, fps: float):
    """Draws a semi-transparent FPS counter in the top-right corner."""
    label = f"FPS: {fps:.1f}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    h, w = frame.shape[:2]
    margin = 8
    x1, y1 = w - tw - margin * 2, margin
    x2, y2 = w - margin, th + margin * 2
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, label, (x1 + margin, y2 - margin), font, scale, (0, 255, 180), thick, cv2.LINE_AA)

def draw_status(frame, text: str, color=(200, 200, 200)):
    """Draws a status message at the bottom-left of the frame."""
    h = frame.shape[0]
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def draw_emotion_chart(frame, bboxes, detector):
    """Draws a compact horizontal bar chart of all emotion probabilities."""
    if not bboxes or 'all_emotions' not in bboxes[0]:
        return

    emotions = bboxes[0]['all_emotions']
    x_offset, y_offset = 10, frame.shape[0] - 160
    chart_w, chart_h = 200, 140
    bar_h, spacing = 15, 4
    
    # Background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset - 5, y_offset - 20), (x_offset + chart_w + 5, y_offset + chart_h + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "Probabilities", (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    emotion_order = ['happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust']
    for i, emo_name in enumerate(emotion_order):
        val = emotions.get(emo_name, emotions.get(emo_name + 'd', 0))
        bar_w = int((val / 100) * chart_w)
        y = y_offset + (i * (bar_h + spacing))
        color = detector._get_color(emo_name)
        
        cv2.rectangle(frame, (x_offset, y), (x_offset + bar_w, y + bar_h), color, -1)
        cv2.rectangle(frame, (x_offset, y), (x_offset + chart_w, y + bar_h), (100, 100, 100), 1)
        
        label = f"{emo_name}: {val:.1f}%"
        cv2.putText(frame, label, (x_offset + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

# --- Main Application Loop ---

def main():
    print("Opening webcam ...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n[ERROR] Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize components
    detector = FaceDetector()
    emotion_log = EmotionLogger(log_dir=os.path.dirname(os.path.abspath(__file__)))
    worker = EmotionWorker(detector)
    worker.start()

    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

    prev_time, fps, frame_count = time.time(), 0.0, 0
    display_frame, last_bboxes, snapshot_feedback_until = None, [], 0
    SUBMIT_EVERY = 2

    print(f"Press 'q' to quit, 's' to take a snapshot.\n")

    while True:
        loop_start = time.time()
        ret, raw_frame = cap.read()
        if not ret: continue

        frame_count += 1
        if frame_count % SUBMIT_EVERY == 0:
            worker.submit(raw_frame)

        # Update display frame based on latest worker results
        annotated, bboxes = worker.get_result()
        if bboxes and bboxes is not last_bboxes:
            emotion_log.log_frame(bboxes)
            last_bboxes = bboxes

        display_frame = annotated.copy() if annotated is not None else raw_frame.copy()
        if annotated is None: draw_status(display_frame, "Initializing model ...", (0, 200, 255))

        # Update FPS and render overlays
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        draw_fps_overlay(display_frame, fps)
        draw_emotion_chart(display_frame, bboxes, detector)

        # Status text
        if bboxes:
            emotions = ", ".join(b['emotion'].upper() for b in bboxes)
            draw_status(display_frame, f"Detected: {emotions}")
        elif annotated is not None:
            draw_status(display_frame, "No faces detected")

        # Visual feedback for snapshot
        if time.time() < snapshot_feedback_until:
            cv2.putText(display_frame, "Snapshot saved!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_TITLE, display_frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            dom_emotion = bboxes[0]['emotion'] if bboxes else "none"
            filename = f"{dom_emotion}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            cv2.imwrite(filepath, display_frame)
            print(f"Snapshot saved: {filepath}")
            snapshot_feedback_until = time.time() + 1.0

        # Frame rate cap
        elapsed = time.time() - loop_start
        if FRAME_DELAY - elapsed > 0:
            time.sleep(FRAME_DELAY - elapsed)

    # Cleanup
    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    emotion_log.close()
    print("Done.")

if __name__ == "__main__":
    main()
