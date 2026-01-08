from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

from core.config import FaceLandmarkerConfig
from core.types import FaceMeshOutput

class FaceMeshTracker:
    """
    MediaPipe Tasks - FaceLandmarker wrapper.
    Returns face landmarks (typically 478 points with this model bundle).
    x,y are normalized; z is relative depth.
    """

    def __init__(self, cfg: FaceLandmarkerConfig):
        self.cfg = cfg

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=RunningMode.VIDEO,
            num_faces=cfg.num_faces,
            min_face_detection_confidence=cfg.min_face_detection_confidence,
            min_face_presence_confidence=cfg.min_face_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[FaceMeshOutput]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(timestamp * 1000)

        res = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            return None

        lm_list = res.face_landmarks[0]
        n = len(lm_list)

        lm_norm = np.zeros((n, 3), dtype=np.float32)
        lm_px   = np.zeros((n, 3), dtype=np.float32)

        for i, lm in enumerate(lm_list):
            x_n, y_n, z_n = float(lm.x), float(lm.y), float(lm.z)
            lm_norm[i] = (x_n, y_n, z_n)

            x_px = x_n * w
            y_px = y_n * h
            z_scaled = z_n * w
            lm_px[i] = (x_px, y_px, z_scaled)

        return FaceMeshOutput(
            timestamp=timestamp,
            frame_size=(h, w),
            landmarks_norm=lm_norm,
            landmarks_px=lm_px,
        )

    def close(self) -> None:
        self.landmarker.close()
