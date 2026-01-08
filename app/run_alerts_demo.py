import argparse
import cv2

from core.config import FaceLandmarkerConfig
from core.signals import DrowsinessSignals
from io_utils.video_stream import VideoStream
from vision.facemesh_tracker import FaceMeshTracker
from vision.draw_facemesh import draw_facemesh
from features.eye_metrics import EyeMetrics, EyeMetricsConfig
from alerts.engine import AlertEngine, AlertConfig
from alerts.notifier import Notifier, NotifierConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--ear_thresh", type=float, default=0.21)
    p.add_argument("--window_s", type=float, default=60.0)
    return p.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    stream = VideoStream(source)
    tracker = FaceMeshTracker(FaceLandmarkerConfig())

    eye = EyeMetrics(EyeMetricsConfig(
        ear_thresh=args.ear_thresh,
        window_s=args.window_s,
    ))

    engine = AlertEngine(AlertConfig())
    notifier = Notifier(NotifierConfig())

    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                break

            out = tracker.process(frame, ts)
            vis = draw_facemesh(frame, out)

            face_present = (out is not None)
            ear_smooth = None
            eye_closed = False
            perclos = None
            perclos_valid = False

            if face_present:
                m = eye.update(out.timestamp, out.landmarks_px)
                ear_smooth = m.get("ear_smooth")
                eye_closed = bool(m.get("is_closed", False))
                perclos = m.get("perclos")
                perclos_valid = bool(m.get("perclos_valid", False))

            # placeholders for teammate modules (fill later)
            yawn = False
            neck_pitch_deg = None
            neck_valid = False

            signals = DrowsinessSignals(
                timestamp=ts,
                face_present=face_present,
                ear_smooth=ear_smooth,
                eye_closed=eye_closed,
                perclos=perclos,
                perclos_valid=perclos_valid,
                yawn=yawn,
                neck_pitch_deg=neck_pitch_deg,
                neck_valid=neck_valid,
            )

            decision = engine.update(signals)
            notifier.notify(decision.level)

            # Overlay
            color = (0, 255, 0) if decision.level == "NORMAL" else (0, 255, 255) if decision.level == "WARNING" else (0, 0, 255)
            cv2.putText(vis, f"STATE: {decision.level}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if ear_smooth is not None:
                cv2.putText(vis, f"EAR_smooth: {ear_smooth:.3f} closed={eye_closed}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            if perclos is not None:
                if perclos_valid:
                    cv2.putText(vis, f"PERCLOS({args.window_s:.0f}s): {perclos:.1f}%", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                else:
                    cv2.putText(vis, "PERCLOS: warming up", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            if decision.reasons:
                cv2.putText(vis, "Reasons: " + ", ".join(decision.reasons)[:60], (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Alerts Demo", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stream.release()
        tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
