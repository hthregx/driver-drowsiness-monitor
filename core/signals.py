from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DrowsinessSignals:
    # time
    timestamp: float

    # face
    face_present: bool

    # eye metrics
    ear_smooth: Optional[float] = None
    eye_closed: bool = False
    perclos: Optional[float] = None
    perclos_valid: bool = False

    # teammate modules (plug-in)
    yawn: bool = False
    yawn_conf: Optional[float] = None

    neck_pitch_deg: Optional[float] = None   # cúi gật (pitch) > threshold => nguy cơ
    neck_valid: bool = False

@dataclass
class AlertDecision:
    level: str                 # "NORMAL" | "WARNING" | "ALARM"
    reasons: List[str]
