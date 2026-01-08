import argparse
import cv2

from core.config import FaceLandmarkerConfig
from io_utils.video_stream import VideoStream
from vision.facemesh_tracker import FaceMeshTracker
from vision.draw_facemesh import draw_facemesh
from features.eye_metrics import EyeMetrics, EyeMetricsConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--ear_thresh", type=float, default=0.21)
    p.add_argument("--min_closed_frames", type=int, default=3)
    p.add_argument("--window_s", type=float, default=60.0)
    p.add_argument("--ema_alpha", type=float, default=0.35)
    return p.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    stream = VideoStream(source)
    tracker = FaceMeshTracker(FaceLandmarkerConfig())

    metrics = EyeMetrics(EyeMetricsConfig(
        ear_thresh=args.ear_thresh,
        min_closed_frames=args.min_closed_frames,
        window_s=args.window_s,
        ema_alpha=args.ema_alpha,
    ))

    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                break

            out = tracker.process(frame, ts)
            vis = draw_facemesh(frame, out)

            if out is None:
                cv2.putText(vis, "Face: NOT FOUND", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                m = metrics.update(out.timestamp, out.landmarks_px)
                n = m.get("n_landmarks", out.landmarks_px.shape[0])

                cv2.putText(vis, f"Face: OK ({n})", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if m.get("ear_smooth") is None:
                    cv2.putText(vis, "EAR: N/A", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    closed = bool(m["is_closed"])
                    ear_s = float(m["ear_smooth"])

                    cv2.putText(vis, f"EAR_smooth: {ear_s:.3f} | closed={closed}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if closed else (0, 255, 0), 2)

                    # ---- PERCLOS with warm-up ----
                    if not m.get("perclos_valid", False):
                        cov = float(m.get("window_covered_s", 0.0))
                        nwin = int(m.get("n_window", 0))
                        cv2.putText(
                            vis,
                            f"PERCLOS warming up: {cov:.1f}s, n={nwin}",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        perclos = float(m["perclos"])
                        cv2.putText(
                            vis,
                            f"PERCLOS({args.window_s:.0f}s): {perclos:.1f}%",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                        )

            cv2.imshow("EAR + PERCLOS", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stream.release()
        tracker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
