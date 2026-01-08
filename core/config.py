from dataclasses import dataclass

@dataclass
class FaceMeshConfig:
    max_num_faces: int = 1
    refine_landmarks: bool = False  # True nếu muốn iris landmarks (nặng hơn)
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

@dataclass
class VideoConfig:
    source: str = "0"  # "0" webcam; hoặc đường dẫn video
    width: int = 0
    height: int = 0

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class FaceLandmarkerConfig:
    model_path: str = str(PROJECT_ROOT / "assets" / "models" / "face_landmarker.task")
    num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
