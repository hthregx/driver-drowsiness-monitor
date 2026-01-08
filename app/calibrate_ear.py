import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from core.config import FaceLandmarkerConfig
from io_utils.video_stream import VideoStream
from vision.facemesh_tracker import FaceMeshTracker
from vision.draw_facemesh import draw_facemesh
from features.eye_metrics import ear_from_landmarks, RIGHT_EYE_6, LEFT_EYE_6

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--open_s", type=float, default=10.0)
    p.add_argument("--closed_s", type=float, default=5.0)
    p.add_argument("--out", type=str, default="artifacts/calibration/ear_calib.json")
    return p.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    stream = VideoStream(source)
    tracker = FaceMeshTracker(FaceLandmarkerConfig())

    state = "IDLE"  # IDLE -> OPEN -> CLOSED -> DONE
    t0 = None
    open_ears = []
    closed_ears = []

    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                break

            out = tracker.process(frame, ts)
            vis = draw_facemesh(frame, out)

            key = cv2.waitKey(1) & 0xFF

            if state == "IDLE":
                cv2.putText(vis, "Press SPACE to start EAR calibration", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                if key == ord(' '):
                    state = "OPEN"
                    t0 = ts
                    open_ears.clear()
                    closed_ears.clear()

            elif state == "OPEN":
                remain = max(0.0, args.open_s - (ts - t0))
                cv2.putText(vis, f"OPEN eyes normally... {remain:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                if out is not None and out.landmarks_px.shape[0] > 387:
                    ear_r = ear_from_landmarks(out.landmarks_px, RIGHT_EYE_6)
                    ear_l = ear_from_landmarks(out.landmarks_px, LEFT_EYE_6)
                    open_ears.append((ear_r + ear_l) / 2.0)

                if ts - t0 >= args.open_s:
                    state = "CLOSED"
                    t0 = ts

            elif state == "CLOSED":
                remain = max(0.0, args.closed_s - (ts - t0))
                cv2.putText(vis, f"CLOSE your eyes... {remain:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                if out is not None and out.landmarks_px.shape[0] > 387:
                    ear_r = ear_from_landmarks(out.landmarks_px, RIGHT_EYE_6)
                    ear_l = ear_from_landmarks(out.landmarks_px, LEFT_EYE_6)
                    closed_ears.append((ear_r + ear_l) / 2.0)

                if ts - t0 >= args.closed_s:
                    state = "DONE"

            if state == "DONE":
                if len(open_ears) < 10 or len(closed_ears) < 10:
                    cv2.putText(vis, "Not enough samples. Press R to retry.", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    mo = float(np.median(open_ears))
                    mc = float(np.median(closed_ears))
                    thresh = (mo + mc) / 2.0
                    payload = {
                        "median_open": mo,
                        "median_closed": mc,
                        "ear_thresh": thresh,
                        "n_open": len(open_ears),
                        "n_closed": len(closed_ears),
                        "open_s": args.open_s,
                        "closed_s": args.closed_s,
                    }
                    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

                    cv2.putText(vis, f"Calibration DONE. ear_thresh={thresh:.3f}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(vis, f"Saved: {args.out}", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.putText(vis, "Press R to retry, Q to quit.", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                if key in (ord('r'), ord('R')):
                    state = "IDLE"
                    t0 = None
                if key in (ord('q'), ord('Q')):
                    break

            cv2.imshow("EAR Calibration", vis)
            if key in (ord('q'), ord('Q')) and state != "DONE":
                break

    finally:
        stream.release()
        tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
