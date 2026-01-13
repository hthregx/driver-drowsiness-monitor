# MediaPipe Face + Pose

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import urllib.request

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as image_module

Point = Tuple[float, float]

# 6 điểm/1 mắt để tính EAR: [p1, p2, p3, p4, p5, p6]
EYE_EAR_IDX = {
    "left":  [33, 160, 158, 133, 153, 144],
    "right": [362, 385, 387, 263, 373, 380],
}

# ROI mắt (polygon) dùng để debug/crop nếu cần
EYE_ROI_IDX = {
    "left":  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
}

# EOP baseline: reuse 6 điểm như EAR (nâng cao mới cần iris)
EOP_IDX = EYE_EAR_IDX

# Iris/Pupil center estimation indices (từ 468 landmarks)
# Sử dụng các điểm xung quanh để ước tính center của iris
IRIS_CENTER_IDX = {
    "left":  [468, 469, 470, 471, 472],  # Iris landmarks (nếu có refine)
    "right": [473, 474, 475, 476, 477],
}

# Alternative: ước tính pupil center từ các điểm mắt
# Sử dụng điểm giữa của 4 điểm góc mắt
PUPIL_ESTIMATE_IDX = {
    "left":  [33, 133, 7, 163],   # 4 điểm góc mắt để ước tính center
    "right": [362, 263, 249, 390],
}

# Model URL for face landmarker
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _get_model_path() -> str:
    """Download and cache the face landmarker model."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".mediapipe_models")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "face_landmarker.task")
    
    if not os.path.exists(model_path):
        print(f"Downloading face landmarker model to {model_path}...")
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        print("Download complete!")
    
    return model_path


@dataclass
class EyeLandmarks:
    left_eye6: List[Point]
    right_eye6: List[Point]
    left_roi: List[Point]
    right_roi: List[Point]
    left_pupil_center: Optional[Point] = None  # Estimated pupil center
    right_pupil_center: Optional[Point] = None


class EyeLandmarker:
    """MediaPipe FaceMesh -> trả 6 điểm mắt + ROI mắt (pixel coords)."""

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True):
        model_path = _get_model_path()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._frame_count = 0

    def close(self):
        self._detector.close()

    @staticmethod
    def _to_xy(lm, w: int, h: int) -> Point:
        """Convert normalized landmark (0-1) to pixel coordinates."""
        return (lm.x * w, lm.y * h)

    def detect(self, bgr_frame, timestamp_ms: Optional[int] = None) -> Optional[EyeLandmarks]:
        """
        Detect eye landmarks from BGR frame.
        
        Args:
            bgr_frame: BGR image from OpenCV
            timestamp_ms: Timestamp in milliseconds (for video mode). If None, uses frame counter.
        """
        h, w = bgr_frame.shape[:2]
        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = image_module.Image(image_format=image_module.ImageFormat.SRGB, data=rgb)
        
        # Use frame counter if timestamp not provided
        if timestamp_ms is None:
            timestamp_ms = self._frame_count * 33  # Assume ~30 FPS (33ms per frame)
            self._frame_count += 1
        
        # Detect landmarks
        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        # Get first face landmarks (normalized 0-1)
        landmarks = result.face_landmarks[0]
        
        # Convert normalized landmarks to pixel coordinates
        pts = [self._to_xy(lm, w, h) for lm in landmarks]

        left6 = [pts[i] for i in EYE_EAR_IDX["left"]]
        right6 = [pts[i] for i in EYE_EAR_IDX["right"]]
        left_roi = [pts[i] for i in EYE_ROI_IDX["left"]]
        right_roi = [pts[i] for i in EYE_ROI_IDX["right"]]
        
        # Estimate pupil center từ 4 điểm góc mắt
        def estimate_pupil_center(corner_indices: List[int]) -> Point:
            """Ước tính center của pupil từ các điểm góc mắt."""
            corner_pts = [pts[i] for i in corner_indices]
            cx = sum(p[0] for p in corner_pts) / len(corner_pts)
            cy = sum(p[1] for p in corner_pts) / len(corner_pts)
            return (cx, cy)
        
        left_pupil = estimate_pupil_center(PUPIL_ESTIMATE_IDX["left"])
        right_pupil = estimate_pupil_center(PUPIL_ESTIMATE_IDX["right"])

        return EyeLandmarks(
            left_eye6=left6,
            right_eye6=right6,
            left_roi=left_roi,
            right_roi=right_roi,
            left_pupil_center=left_pupil,
            right_pupil_center=right_pupil,
        )
