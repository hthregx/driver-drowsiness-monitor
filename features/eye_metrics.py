from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Optional
import numpy as np

RIGHT_EYE_6 = [33, 160, 158, 133, 153, 144]
LEFT_EYE_6  = [362, 385, 387, 263, 373, 380]

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def ear_from_landmarks(landmarks_px: np.ndarray, idxs: list[int]) -> float:
    pts = [landmarks_px[i, :2].astype(np.float32) for i in idxs]
    p1, p2, p3, p4, p5, p6 = pts
    num = _dist(p2, p6) + _dist(p3, p5)
    den = 2.0 * _dist(p1, p4) + 1e-6
    return num / den

@dataclass
class EyeMetricsConfig:
    ear_thresh: float = 0.21
    min_closed_frames: int = 3
    window_s: float = 60.0
    ema_alpha: float = 0.35

    # PERCLOS stability (warm-up)
    min_perclos_coverage_s: float = 10.0   # chỉ hiển thị PERCLOS khi buffer phủ >= 10s
    min_perclos_samples: int = 120         # hoặc đủ số frame tối thiểu (tuỳ FPS)

class EyeMetrics:
    def __init__(self, cfg: EyeMetricsConfig):
        self.cfg = cfg
        self._ema: Optional[float] = None
        self._closed_counter: int = 0
        self._buf: Deque[Tuple[float, bool]] = deque()

    def update(self, timestamp: float, landmarks_px: np.ndarray) -> dict:
        n = int(landmarks_px.shape[0])

        if n <= 387:
            self._append(timestamp, False)
            return self._pack(None, None, False, n, timestamp)

        ear_r = ear_from_landmarks(landmarks_px, RIGHT_EYE_6)
        ear_l = ear_from_landmarks(landmarks_px, LEFT_EYE_6)
        ear = (ear_r + ear_l) / 2.0

        if self._ema is None:
            self._ema = ear
        else:
            a = float(self.cfg.ema_alpha)
            self._ema = a * ear + (1.0 - a) * self._ema

        if self._ema < float(self.cfg.ear_thresh):
            self._closed_counter += 1
        else:
            self._closed_counter = 0

        is_closed = (self._closed_counter >= int(self.cfg.min_closed_frames))
        self._append(timestamp, bool(is_closed))
        return self._pack(float(ear), float(self._ema), bool(is_closed), n, timestamp)

    def _append(self, t: float, closed: bool) -> None:
        self._buf.append((t, closed))
        while self._buf and (t - self._buf[0][0] > float(self.cfg.window_s)):
            self._buf.popleft()

    def _perclos(self) -> float:
        if not self._buf:
            return 0.0
        closed = sum(1 for _, c in self._buf if c)
        return 100.0 * closed / len(self._buf)

    def _coverage(self, now: float) -> float:
        if not self._buf:
            return 0.0
        return max(0.0, now - self._buf[0][0])

    def _pack(self, ear: Optional[float], ear_smooth: Optional[float], is_closed: bool, n: int, now: float) -> dict:
        cov = self._coverage(now)
        nwin = len(self._buf)
        per = self._perclos()
        valid = (cov >= float(self.cfg.min_perclos_coverage_s)) and (nwin >= int(self.cfg.min_perclos_samples))
        return {
            "ear": ear,
            "ear_smooth": ear_smooth,
            "is_closed": is_closed,
            "perclos": float(per),
            "perclos_valid": bool(valid),
            "window_covered_s": float(cov),
            "n_window": int(nwin),
            "n_landmarks": int(n),
        }
