# EOP, PERCLOS, EAR

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional

Point = Tuple[float, float]

def _dist(a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


# ===== 1) EAR =====
def compute_ear(eye6: List[Point]) -> float:
    """EAR from 6 points [p1,p2,p3,p4,p5,p6]."""
    if len(eye6) != 6:
        raise ValueError("eye6 must have 6 points")
    p1, p2, p3, p4, p5, p6 = eye6
    num = _dist(p2, p6) + _dist(p3, p5)
    den = 2.0 * _dist(p1, p4)
    return float(num / den) if den > 1e-9 else 0.0


# ===== 2) EOP (baseline) =====
def compute_eop(eye6: List[Point]) -> float:
    """EOP baseline = (độ mở dọc trung bình) / (độ rộng mắt)."""
    if len(eye6) != 6:
        raise ValueError("eye6 must have 6 points")
    p1, p2, p3, p4, p5, p6 = eye6
    width = _dist(p1, p4)
    open_ = (_dist(p2, p6) + _dist(p3, p5)) / 2.0
    return float(open_ / width) if width > 1e-9 else 0.0


# ===== 2b) EOP "over the pupil" =====
def compute_eop_over_pupil(
    eye6: List[Point], 
    pupil_center: Optional[Point],
    upper_lid_idx: Tuple[int, int] = (1, 2),  # p2, p3 (upper lid)
    lower_lid_idx: Tuple[int, int] = (4, 5),  # p5, p6 (lower lid)
) -> float:
    """
    EOP "over the pupil" - tính độ mở mắt dựa trên khoảng cách từ mi mắt đến pupil center.
    
    Định nghĩa: EOP = (khoảng cách từ upper lid đến pupil + khoảng cách từ lower lid đến pupil) / (đường kính mắt)
    
    Args:
        eye6: 6 điểm mắt [p1, p2, p3, p4, p5, p6]
        pupil_center: Center của pupil (x, y). Nếu None, fallback về EOP baseline.
        upper_lid_idx: Indices của upper eyelid points (default: p2, p3)
        lower_lid_idx: Indices của lower eyelid points (default: p5, p6)
    
    Returns:
        EOP value (0.0 = closed, 1.0+ = fully open)
    """
    if len(eye6) != 6:
        raise ValueError("eye6 must have 6 points")
    
    # Fallback về baseline nếu không có pupil center
    if pupil_center is None:
        return compute_eop(eye6)
    
    p1, p2, p3, p4, p5, p6 = eye6
    pc = pupil_center
    
    # Tính khoảng cách từ upper lid đến pupil
    upper_dist = (_dist(p2, pc) + _dist(p3, pc)) / 2.0
    
    # Tính khoảng cách từ lower lid đến pupil
    lower_dist = (_dist(p5, pc) + _dist(p6, pc)) / 2.0
    
    # Độ mở mắt = tổng khoảng cách từ mi mắt đến pupil
    eye_open = upper_dist + lower_dist
    
    # Đường kính mắt (horizontal width)
    eye_diameter = _dist(p1, p4)
    
    # EOP = độ mở / đường kính
    return float(eye_open / eye_diameter) if eye_diameter > 1e-9 else 0.0


# ===== 3) Debounce blink =====
@dataclass
class EyeState:
    is_closed: bool
    closed_frames: int
    last_closed_start_ts: Optional[float]


class EyeClosureDetector:
    """Chỉ coi closed nếu EAR < threshold liên tục >= MIN_CLOSED_FRAMES."""

    def __init__(self, ear_thresh: float, min_closed_frames: int):
        self.ear_thresh = float(ear_thresh)
        self.min_closed_frames = int(min_closed_frames)
        self._closed_frames = 0
        self._is_closed = False
        self._start_ts: Optional[float] = None

    def update(self, ear: float, ts: float) -> EyeState:
        if ear < self.ear_thresh:
            self._closed_frames += 1
            if self._closed_frames == 1:
                self._start_ts = ts
        else:
            self._closed_frames = 0
            self._start_ts = None

        self._is_closed = self._closed_frames >= self.min_closed_frames

        if self._is_closed and self._start_ts is None:
            self._start_ts = ts

        return EyeState(self._is_closed, self._closed_frames, self._start_ts)


# ===== 4) PERCLOS (sliding window) =====
class PerclosWindow:
    def __init__(self, window_s: float):
        self.window_s = float(window_s)
        self._buf: Deque[Tuple[float, int]] = deque()  # (ts, 0/1)

    def reset(self):
        """Reset PERCLOS window (xóa tất cả dữ liệu cũ)."""
        self._buf.clear()

    def update_perclos(self, ts: float, is_closed: bool) -> float:
        self._buf.append((ts, 1 if is_closed else 0))
        cutoff = ts - self.window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if not self._buf:
            return 0.0
        return float(sum(v for _, v in self._buf) / len(self._buf))
