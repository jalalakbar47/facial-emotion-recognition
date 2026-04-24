"""
web/app.py

Refined Flask application for EmotionAI.
Force-disables caching and explicitly sets paths to ensure the latest 
templates are always served.
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import sys
import os
import time
from datetime import datetime
import csv
import numpy as np
import threading

# Get absolute path of this script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

# FORCE DISABLE CACHE
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

detector_path = os.path.abspath(os.path.join(base_dir, '..'))
if detector_path not in sys.path:
    sys.path.append(detector_path)
from detector import FaceDetector

detector = FaceDetector()
detector_lock = threading.Lock()

latest_data = {
    "emotions": {
        "happy": 0, "sad": 0, "angry": 0, "surprise": 0, 
        "neutral": 0, "fear": 0, "disgust": 0
    },
    "dominant": "Searching...",
    "confidence": 0,
    "last_frame": None,
    "is_running": True
}

class CameraThread(threading.Thread):
    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.camera = None
        self.stop_flag = False

    def get_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.camera

    def release_camera(self):
        if self.camera:
            self.camera.release()
            self.camera = None

    def run(self):
        global latest_data
        while not self.stop_flag:
            if not latest_data["is_running"]:
                self.release_camera()
                time.sleep(0.5)
                continue
            cap = self.get_camera()
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            with detector_lock:
                annotated, bboxes = self.detector.detect_faces(frame)
            if bboxes:
                latest_data["emotions"] = bboxes[0]['all_emotions']
                latest_data["dominant"] = bboxes[0]['emotion']
                latest_data["confidence"] = bboxes[0]['confidence']
            else:
                latest_data["dominant"] = "Searching..."
                latest_data["confidence"] = 0
            ret, buffer = cv2.imencode('.jpg', annotated)
            latest_data["last_frame"] = buffer.tobytes()
            time.sleep(0.05)

cam_thread = CameraThread(detector)
cam_thread.start()

@app.after_request
def add_header(r):
    """Add headers to both force latest IE rendering engine or Chrome Frame, and also to cache as little as possible."""
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def index():
    return render_template('index.html', t=time.time()) # Add cache-buster variable

@app.route('/video_feed')
def video_feed():
    def stream():
        while True:
            if latest_data["is_running"] and latest_data["last_frame"] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_data["last_frame"] + b'\r\n')
            else:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Camera Paused", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
            time.sleep(0.04)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    latest_data["is_running"] = True
    return jsonify({"status": "running"})

@app.route('/stop_camera')
def stop_camera():
    latest_data["is_running"] = False
    return jsonify({"status": "stopped"})

@app.route('/emotion_data')
def emotion_data():
    return jsonify({
        "emotions": latest_data["emotions"],
        "dominant": latest_data["dominant"],
        "confidence": latest_data["confidence"],
        "is_running": latest_data["is_running"]
    })

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filestr = file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400
    with detector_lock:
        _, bboxes = detector.detect_faces(img)
    if bboxes:
        global latest_data
        latest_data["emotions"] = bboxes[0]['all_emotions']
        latest_data["dominant"] = bboxes[0]['emotion']
        latest_data["confidence"] = bboxes[0]['confidence']
    return jsonify({'status': 'success', 'results': bboxes})

@app.route('/snapshot')
def snapshot():
    if not latest_data["is_running"]: return jsonify({'status': 'error', 'message': 'Paused'}), 400
    cap = cam_thread.get_camera()
    success, frame = cap.read()
    if success:
        os.makedirs('../snapshots', exist_ok=True)
        filename = f"web_snap_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        cv2.imwrite(os.path.join('../snapshots', filename), frame)
        return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'status': 'error'}), 500

@app.route('/logs_data')
def logs_data():
    log_file = '../emotion_log.csv'
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            logs = list(reader)[-20:]
            logs.reverse()
    return jsonify(logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
