import cv2
import numpy as np
from typing import Optional
from core.types import FaceMeshOutput

def draw_facemesh(frame_bgr: np.ndarray, out: Optional[FaceMeshOutput]) -> np.ndarray:
    if out is None:
        return frame_bgr

    img = frame_bgr.copy()
    pts = out.landmarks_px[:, :2].astype(int)

    # draw sampled points for speed
    for (x, y) in pts[::2]:
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    return img
