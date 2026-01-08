from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class FaceMeshOutput:
    """Normalized + pixel-space face mesh output for downstream modules (EAR, MAR, head pose, etc.)."""
    timestamp: float
    frame_size: Tuple[int, int]      # (h, w)
    landmarks_norm: np.ndarray       # (468, 3) float32: x,y,z normalized
    landmarks_px: np.ndarray         # (468, 3) float32: x_px,y_px,z_scaled
