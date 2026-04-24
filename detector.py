"""
detector.py

This module handles face detection using OpenCV's Haar Cascades and 
emotion recognition using the DeepFace library.
"""

import cv2
from deepface import DeepFace
import logging

# Configure logging to suppress verbose DeepFace output while keeping our info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Main detection class that combines OpenCV for fast face localization 
    and DeepFace for deep-learning based emotion analysis.
    """

    # BGR color map for visual feedback based on the 7 standard emotions
    EMOTION_COLORS = {
        'happy':     (0,   255,  0),    # Green
        'sad':       (255, 100,  0),    # Blue
        'angry':     (0,   0,   255),   # Red
        'surprise':  (0,   255, 255),   # Yellow
        'neutral':   (255, 255, 255),   # White
        'fear':      (0,   165, 255),   # Orange
        'disgust':   (128, 0,   128),   # Purple
    }

    # Normalization map to handle minor naming variations in DeepFace output
    EMOTION_KEY_MAP = {
        'happy':     'happy',
        'sad':       'sad',
        'angry':     'angry',
        'surprise':  'surprise',
        'surprised': 'surprise',
        'neutral':   'neutral',
        'fear':      'fear',
        'fearful':   'fear',
        'disgust':   'disgust',
        'disgusted': 'disgust',
    }

    def __init__(self):
        """Initializes the detector and loads the pre-trained Haar Cascade XML."""
        logger.info("Initializing Face Detector with Emotion Recognition...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def _get_color(self, emotion: str):
        """
        Maps an emotion string to its designated BGR color.
        
        Args:
            emotion: String representing the detected emotion.
        Returns:
            Tuple (B, G, R) for OpenCV drawing.
        """
        key = self.EMOTION_KEY_MAP.get(emotion.lower(), 'neutral')
        return self.EMOTION_COLORS.get(key, (255, 255, 255))

    def _analyze_emotion(self, face_roi):
        """
        Invokes DeepFace to analyze a cropped face region for emotions.
        
        Args:
            face_roi: Cropped BGR image segment containing exactly one face.
        Returns:
            Tuple containing:
            - dominant_emotion (str)
            - confidence_score (float)
            - all_emotions (dict of all 7 probabilities)
        """
        try:
            results = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            # DeepFace returns a list when analyze is called
            res = results[0] if isinstance(results, list) else results
            dominant = res['dominant_emotion']
            confidence = float(res['emotion'][dominant])
            all_emotions = res['emotion']
            return dominant, confidence, all_emotions
        except Exception as e:
            logger.warning(f"Emotion analysis failed for ROI: {e}")
            return 'neutral', 0.0, {}

    def detect_faces(self, frame):
        """
        Detects all faces in a frame and applies emotion analysis to each.
        
        Args:
            frame: The full BGR image frame from a webcam or file.
        Returns:
            Tuple containing:
            - annotated_frame: A copy of the frame with bboxes and text labels.
            - bboxes: A list of dicts with raw detection data (region, emotion, etc).
        """
        # Convert to grayscale for Haar Cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade (fast)
        raw_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        annotated_frame = frame.copy()
        bboxes = []

        # Process each detected face
        for (x, y, w, h) in raw_faces:
            # Crop the face for DeepFace analysis
            face_roi = frame[y:y + h, x:x + w]
            emotion, confidence, all_emotions = self._analyze_emotion(face_roi)
            color = self._get_color(emotion)

            # --- 1. Draw bounding box ---
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

            # --- 2. Create styled label background ---
            label = f"{emotion.upper()}  {confidence:.1f}%"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thick)

            pad = 4
            bg_x1, bg_y1 = x, max(0, y - text_h - baseline - pad * 2)
            bg_x2, bg_y2 = x + text_w + pad * 2, max(text_h + baseline, y)
            cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, cv2.FILLED)

            # --- 3. Draw label text with auto-contrast ---
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
            
            cv2.putText(
                annotated_frame, label, (bg_x1 + pad, bg_y2 - baseline - pad),
                font, scale, text_color, thick, cv2.LINE_AA
            )

            # --- 4. Record data for logging/UI ---
            # Ensure all numeric types are standard Python floats/ints for JSON serialization
            clean_emotions = {k: float(v) for k, v in all_emotions.items()}
            
            bboxes.append({
                'region':       (int(x), int(y), int(w), int(h)),
                'emotion':      emotion,
                'confidence':   float(confidence),
                'all_emotions': clean_emotions,
            })

        return annotated_frame, bboxes
