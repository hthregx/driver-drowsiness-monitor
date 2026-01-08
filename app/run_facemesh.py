import argparse
import cv2

from core.config import FaceLandmarkerConfig
from io_utils.video_stream import VideoStream
from vision.facemesh_tracker import FaceMeshTracker
from vision.draw_facemesh import draw_facemesh

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 for webcam, or path to a video file")
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--save", type=str, default="", help="optional output video path (.mp4)")
    p.add_argument("--refine", action="store_true", help="enable refine_landmarks (heavier)")
    p.add_argument("--no_view", action="store_true", help="headless mode (no cv2.imshow)")
    return p.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    stream = VideoStream(source, width=args.width, height=args.height)
    tracker = FaceMeshTracker(FaceLandmarkerConfig())

    writer = None
    try:
        while True:
            ok, frame, ts = stream.read()
            if not ok:
                break

            out = tracker.process(frame, ts)
            vis = draw_facemesh(frame, out)

            if out is None:
                cv2.putText(vis, "Face: NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                cv2.putText(vis, "Face: OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            if args.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w, h))
                writer.write(vis)

            if not args.no_view:
                cv2.imshow("FaceMesh 3D (x,y,z)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        stream.release()
        tracker.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
