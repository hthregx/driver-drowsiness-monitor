import cv2
import time
from typing import Optional, Tuple, Union

class VideoStream:
    """
    Unified frame reader for webcam (int) or video file (str).
    Returns (ok, frame_bgr, timestamp).
    """
    def __init__(self, source: Union[int, str], width: int = 0, height: int = 0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        if width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[bool, Optional[any], float]:
        ok, frame = self.cap.read()
        ts = time.time()
        return ok, frame, ts

    def release(self) -> None:
        self.cap.release()
