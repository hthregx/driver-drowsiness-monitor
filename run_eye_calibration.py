from __future__ import annotations
import time
from typing import List

import cv2
import numpy as np

from src.landmarks import EyeLandmarker
from src.eye_metrics import compute_ear
from src.utils import load_config, save_config


def median(xs: List[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float32))) if xs else 0.0


def main():
    cfg = load_config("config.yaml")
    cam_id = int(cfg["VIDEO"]["CAM_ID"])
    w = int(cfg["VIDEO"]["WIDTH"])
    h = int(cfg["VIDEO"]["HEIGHT"])
    flip = bool(cfg["VIDEO"].get("FLIP", True))

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    lm = EyeLandmarker()

    def collect(seconds: float, title: str) -> List[float]:
        xs: List[float] = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            eyes = lm.detect(frame)
            if eyes is not None:
                ear = (compute_ear(eyes.left_eye6) + compute_ear(eyes.right_eye6)) / 2.0
                xs.append(ear)
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            remain = seconds - (time.time() - t0)
            cv2.putText(frame, f"{title}  ({remain:0.1f}s)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press q to quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("EAR calibration", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        return xs

    try:
        print("Phase 1: OPEN eyes ~12s")
        open_x = collect(12.0, "Phase 1: EYES OPEN")
        time.sleep(0.6)
        print("Phase 2: CLOSE eyes ~6s")
        closed_x = collect(6.0, "Phase 2: EYES CLOSED")

        mo = median(open_x)
        mc = median(closed_x)
        thr = (mo + mc) / 2.0 if (mo > 0 and mc > 0) else float(cfg["EYE"]["EAR_THRESH"])

        cfg["EYE"]["EAR_THRESH"] = float(round(thr, 4))
        save_config(cfg, "config.yaml")

        print("=== Result ===")
        print(f"median_open   = {mo:.4f} (n={len(open_x)})")
        print(f"median_closed = {mc:.4f} (n={len(closed_x)})")
        print(f"EAR_THRESH    = {thr:.4f} -> saved to config.yaml")

    finally:
        lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
