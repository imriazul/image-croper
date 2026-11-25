"""Face detection module using MediaPipe Face Landmarker (Tasks API)."""

import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

# Model download URL
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PACKAGE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


@dataclass
class FaceLandmarks:
    """Detected face landmark data."""

    # Bounding box (x, y, width, height) in pixels
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    # Key landmark points (x, y) in pixels
    left_eye: Tuple[float, float] = (0.0, 0.0)
    right_eye: Tuple[float, float] = (0.0, 0.0)
    nose_tip: Tuple[float, float] = (0.0, 0.0)
    mouth_center: Tuple[float, float] = (0.0, 0.0)
    chin: Tuple[float, float] = (0.0, 0.0)
    forehead: Tuple[float, float] = (0.0, 0.0)
    left_ear: Tuple[float, float] = (0.0, 0.0)
    right_ear: Tuple[float, float] = (0.0, 0.0)

    # Face center
    face_center: Tuple[float, float] = (0.0, 0.0)

    # Detection confidence
    confidence: float = 0.0

    # Head dimensions
    head_top: float = 0.0
    head_bottom: float = 0.0
    head_left: float = 0.0
    head_right: float = 0.0

    # Head width and height
    head_width: float = 0.0
    head_height: float = 0.0

    # Inter-pupillary distance
    ipd: float = 0.0

    # Roll angle (head tilt)
    roll_angle: float = 0.0


@dataclass
class FaceDetectionResult:
    """Result of face detection on an image."""

    success: bool = False
    primary_face: Optional[FaceLandmarks] = None
    all_faces: List[FaceLandmarks] = field(default_factory=list)
    num_faces: int = 0
    rejection_reason: str = ""
    image_width: int = 0
    image_height: int = 0


def _download_model():
    """Download the Face Landmarker model if not present."""
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Downloading Face Landmarker model (first run only)...")
    try:
        try:
            import requests
            resp = requests.get(MODEL_URL, stream=True, timeout=60)
            resp.raise_for_status()
            tmp_path = MODEL_PATH + ".tmp"
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, MODEL_PATH)
        except ImportError:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Face Landmarker model: {e}\n"
            f"You can manually download it from:\n{MODEL_URL}\n"
            f"and place it at: {MODEL_PATH}"
        )


class FaceDetector:
    """MediaPipe Face Landmarker-based face detector."""

    # Key Face Mesh landmark indices (same for Tasks API)
    LANDMARK_LEFT_EYE = 468       # Left iris center
    LANDMARK_RIGHT_EYE = 473      # Right iris center
    LANDMARK_LEFT_EYE_OUTER = 33
    LANDMARK_LEFT_EYE_INNER = 133
    LANDMARK_RIGHT_EYE_OUTER = 263
    LANDMARK_RIGHT_EYE_INNER = 362
    LANDMARK_NOSE_TIP = 1
    LANDMARK_MOUTH_LEFT = 61
    LANDMARK_MOUTH_RIGHT = 291
    LANDMARK_CHIN = 152
    LANDMARK_FOREHEAD = 10
    LANDMARK_LEFT_EAR = 234
    LANDMARK_RIGHT_EAR = 454

    def __init__(self, min_confidence: float = 0.5, max_faces: int = 5):
        self.min_confidence = min_confidence
        self.max_faces = max_faces
        self._landmarker: Optional[mp_vision.FaceLandmarker] = None
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self):
        """Initialize the MediaPipe Face Landmarker."""
        if self._initialized:
            return

        # Download model if needed
        _download_model()

        base_options = mp_python.BaseOptions(
            model_asset_path=MODEL_PATH
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_confidence,
            min_face_presence_confidence=self.min_confidence,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._initialized = True
        logger.info("MediaPipe Face Landmarker initialized successfully")

    def detect(
        self,
        image: np.ndarray,
        min_face_size_ratio: float = 0.05,
        min_face_area_px: int = 2500,
    ) -> FaceDetectionResult:
        """
        Detect faces in an image and extract landmarks.

        Args:
            image: BGR image as numpy array
            min_face_size_ratio: Minimum face area as ratio of image area
            min_face_area_px: Minimum face area in pixels

        Returns:
            FaceDetectionResult with detection info
        """
        if not self._initialized:
            self.initialize()

        result = FaceDetectionResult()
        h, w = image.shape[:2]
        result.image_width = w
        result.image_height = h

        # Check minimum resolution
        if w < 100 or h < 100:
            result.rejection_reason = "Image resolution too low"
            return result

        # Convert BGR to RGB and create MediaPipe Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )

        # Run face landmarker (serialized for thread safety)
        try:
            with self._lock:
                mp_result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.error(f"MediaPipe processing error: {e}")
            result.rejection_reason = f"Face detection error: {str(e)}"
            return result

        if not mp_result.face_landmarks:
            result.rejection_reason = "No face detected"
            return result

        # Extract landmarks for each detected face
        image_area = w * h
        valid_faces = []

        for face_lms in mp_result.face_landmarks:
            landmarks = self._extract_landmarks(face_lms, w, h)

            # Check face size
            face_area = landmarks.head_width * landmarks.head_height
            face_area_ratio = face_area / image_area

            if face_area < min_face_area_px:
                continue
            if face_area_ratio < min_face_size_ratio:
                continue

            valid_faces.append(landmarks)

        if not valid_faces:
            result.rejection_reason = "No sufficiently large face detected"
            return result

        result.all_faces = valid_faces
        result.num_faces = len(valid_faces)

        # Select primary face (largest by area)
        primary = max(valid_faces, key=lambda f: f.head_width * f.head_height)
        result.primary_face = primary

        # Check for ambiguity with multiple faces
        if len(valid_faces) > 1:
            areas = sorted(
                [f.head_width * f.head_height for f in valid_faces], reverse=True
            )
            if len(areas) > 1 and areas[1] > areas[0] * 0.6:
                result.rejection_reason = (
                    f"Multiple faces detected ({len(valid_faces)}) - "
                    "primary subject is ambiguous"
                )
                return result

        # Validate primary face
        validation = self._validate_face(primary, w, h)
        if validation:
            result.rejection_reason = validation
            return result

        result.success = True
        return result

    def _extract_landmarks(
        self, face_landmarks: list, img_width: int, img_height: int
    ) -> FaceLandmarks:
        """Extract structured landmarks from Face Landmarker result."""
        result = FaceLandmarks()

        def to_px(idx):
            lm = face_landmarks[idx]
            return (lm.x * img_width, lm.y * img_height)

        # Eyes - use iris landmarks if available (indices 468-477)
        num_landmarks = len(face_landmarks)
        if num_landmarks > self.LANDMARK_RIGHT_EYE:
            result.left_eye = to_px(self.LANDMARK_LEFT_EYE)
            result.right_eye = to_px(self.LANDMARK_RIGHT_EYE)
        else:
            # Fallback to eye corner midpoints
            le_outer = to_px(self.LANDMARK_LEFT_EYE_OUTER)
            le_inner = to_px(self.LANDMARK_LEFT_EYE_INNER)
            result.left_eye = (
                (le_outer[0] + le_inner[0]) / 2,
                (le_outer[1] + le_inner[1]) / 2,
            )
            re_outer = to_px(self.LANDMARK_RIGHT_EYE_OUTER)
            re_inner = to_px(self.LANDMARK_RIGHT_EYE_INNER)
            result.right_eye = (
                (re_outer[0] + re_inner[0]) / 2,
                (re_outer[1] + re_inner[1]) / 2,
            )

        result.nose_tip = to_px(self.LANDMARK_NOSE_TIP)

        mouth_left = to_px(self.LANDMARK_MOUTH_LEFT)
        mouth_right = to_px(self.LANDMARK_MOUTH_RIGHT)
        result.mouth_center = (
            (mouth_left[0] + mouth_right[0]) / 2,
            (mouth_left[1] + mouth_right[1]) / 2,
        )

        result.chin = to_px(self.LANDMARK_CHIN)
        result.forehead = to_px(self.LANDMARK_FOREHEAD)
        result.left_ear = to_px(self.LANDMARK_LEFT_EAR)
        result.right_ear = to_px(self.LANDMARK_RIGHT_EAR)

        # Face center
        eye_center_x = (result.left_eye[0] + result.right_eye[0]) / 2
        face_center_y = (result.forehead[1] + result.chin[1]) / 2
        result.face_center = (eye_center_x, face_center_y)

        # Head boundaries from all landmarks
        all_x = [face_landmarks[i].x * img_width for i in range(min(468, num_landmarks))]
        all_y = [face_landmarks[i].y * img_height for i in range(min(468, num_landmarks))]

        result.head_left = min(all_x)
        result.head_right = max(all_x)
        result.head_top = min(all_y)
        result.head_bottom = max(all_y)

        result.head_width = result.head_right - result.head_left
        result.head_height = result.head_bottom - result.head_top

        result.bbox = (
            int(result.head_left),
            int(result.head_top),
            int(result.head_width),
            int(result.head_height),
        )

        # Inter-pupillary distance
        result.ipd = (
            (result.left_eye[0] - result.right_eye[0]) ** 2
            + (result.left_eye[1] - result.right_eye[1]) ** 2
        ) ** 0.5

        # Roll angle
        dx = result.right_eye[0] - result.left_eye[0]
        dy = result.right_eye[1] - result.left_eye[1]
        result.roll_angle = np.degrees(np.arctan2(dy, dx))

        # Confidence from landmark presence
        z_values = [abs(face_landmarks[i].z) for i in range(min(468, num_landmarks))]
        result.confidence = max(0, 1.0 - np.mean(z_values))

        return result

    def _validate_face(
        self, face: FaceLandmarks, img_width: int, img_height: int
    ) -> str:
        """
        Validate a detected face for passport photo suitability.
        Returns empty string if valid, or rejection reason.
        """
        margin = 0.02

        if face.forehead[1] < img_height * margin:
            return "Head is cut off at the top (forehead not fully visible)"

        if face.chin[1] > img_height * (1 - margin):
            return "Head is cut off at the bottom (chin not fully visible)"

        if face.head_left < img_width * margin and face.head_right < img_width * 0.4:
            return "Face is severely cut off on the left side"
        if face.head_right > img_width * (1 - margin) and face.head_left > img_width * 0.6:
            return "Face is severely cut off on the right side"

        eye_distance = face.ipd
        if eye_distance < 10:
            return "Eyes not reliably detected (too close together)"

        if abs(face.roll_angle) > 30:
            return f"Excessive head tilt ({face.roll_angle:.1f}°)"

        face_area = face.head_width * face.head_height
        image_area = img_width * img_height
        if face_area / image_area < 0.01:
            return "Face is too small in the image"

        return ""

    def close(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False
