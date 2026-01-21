"""
run_head_pose_demo.py
Webcam + MediaPipe PoseLandmarker (VIDEO mode)
Neck angle vs Oy (neck->nose), EMA smoothing, timers WARNING/DROWSY, nodding
ROI "cổ -> đầu", quality gate (angle + yaw proxy)
CSV logging

Usage:
  python run_head_pose_demo.py --model models/pose_landmarker_lite.task --cam 0
Keys:
  q : quit
"""

from __future__ import annotations
import argparse
import os
import time
import csv
from typing import Tuple

import cv2

from neck_module import NeckModule, NeckCfg

VIS_TH = 0.5  # visibility threshold


def lm_to_px(lm, w: int, h: int) -> Tuple[float, float]:
    return (lm.x * w, lm.y * h)


def lm_visible(lm) -> bool:
    v = getattr(lm, "visibility", 1.0)
    return (v is not None) and (v >= VIS_TH)


def safe_rect(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    """Vẽ bbox an toàn để tránh crash khi tọa độ bị ngược/âm."""
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_info(frame, out, fps: float, pose_ok: bool, msg: str = ""):
    h, w = frame.shape[:2]
    panel_w, panel_h = 470, 220

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 40
    dy = 28

    if (not pose_ok) or (out is None):
        cv2.putText(frame, "NO RELIABLE POSE", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        y += dy
        if msg:
            cv2.putText(frame, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y += dy
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        return

    state = out.get("state", "NORMAL")
    gate_ok = bool(out.get("gate_ok", True))
    quality = float(out.get("quality", 0.0))
    nodding = bool(out.get("nodding", False))

    angle_signed = float(out.get("angle_signed", 0.0))
    angle_abs = float(out.get("angle_abs", abs(angle_signed)))
    yaw = out.get("yaw", None)

    warn_timer = float(out.get("warn_timer", 0.0))
    drowsy_timer = float(out.get("drowsy_timer", 0.0))

    state_col = {
        "NORMAL": (0, 255, 0),
        "WARNING": (0, 255, 255),
        "DROWSY": (0, 0, 255),
        "NO_POSE": (0, 0, 255),
    }.get(state, (255, 255, 255))

    gate_col = (0, 255, 0) if gate_ok else (0, 0, 255)

    cv2.putText(frame, f"STATE: {state}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_col, 2)
    y += dy

    cv2.putText(frame, f"Angle: {angle_signed:+.1f} deg   |Angle|: {angle_abs:.1f}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)
    y += dy

    if yaw is not None:
        cv2.putText(frame, f"Yaw proxy: {float(yaw):.3f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y += dy - 6

    cv2.putText(frame, f"Gate: {'OK' if gate_ok else 'CLOSED'}   Quality: {quality:.2f}",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, gate_col, 2)
    y += dy

    cv2.putText(frame, f"Hold: WARN={warn_timer:.1f}s   DROWSY={drowsy_timer:.1f}s",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    y += dy

    if not gate_ok:
        cv2.putText(frame, "OFF-ANGLE: GATE CLOSED", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y += dy

    if nodding:
        cv2.putText(frame, "NOD!", (w - 150, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to pose_landmarker_lite.task")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out_csv", default="outputs/neck_log.csv")
    ap.add_argument("--ema_alpha", type=float, default=0.25)
    ap.add_argument("--warn_hold", type=float, default=0.8)
    ap.add_argument("--drowsy_hold", type=float, default=1.2)
    args = ap.parse_args()

    # MediaPipe
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception as e:
        print("❌ Không thể import mediapipe. Error:", repr(e))
        return

    if not os.path.exists(args.model):
        print(f"❌ Model không tồn tại: {args.model}")
        return

    # Pose landmarker (VIDEO)
    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Webcam
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    # Neck module config
    cfg = NeckCfg(
        ema_alpha=args.ema_alpha,
        warn_hold_s=args.warn_hold,
        drowsy_hold_s=args.drowsy_hold
    )
    neck = NeckModule(cfg)

    # CSV logging
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    csv_f = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow([
        "frame_id", "timestamp_ms",
        "pose_ok", "gate_ok", "quality",
        "state",
        "angle_signed", "angle_abs", "yaw",
        "warn_timer", "drowsy_timer",
        "nodding",
        "ls_x", "ls_y", "rs_x", "rs_y", "nose_x", "nose_y"
    ])

    frame_id = 0
    t_prev = time.time()
    fps_smooth = 30.0

    # landmark indices (MediaPipe Pose)
    IDX_NOSE = 0
    IDX_LS = 11
    IDX_RS = 12
    IDX_LE = 7
    IDX_RE = 8

    print("=" * 60)
    print("🎥 NECK ANGLE DEMO (PoseLandmarker)")
    print("q: quit")
    print("=" * 60)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)

        # FPS calc
        t_now = time.time()
        dt = max(1e-6, t_now - t_prev)
        t_prev = t_now
        fps_inst = 1.0 / dt
        fps_smooth = 0.1 * fps_inst + 0.9 * fps_smooth

        # detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result_mp = landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_ok = False
        out = None
        msg = ""

        if result_mp.pose_landmarks and len(result_mp.pose_landmarks) > 0:
            lms = result_mp.pose_landmarks[0]

            nose_lm = lms[IDX_NOSE]
            ls_lm = lms[IDX_LS]
            rs_lm = lms[IDX_RS]

            if not (lm_visible(nose_lm) and lm_visible(ls_lm) and lm_visible(rs_lm)):
                pose_ok = False
                msg = "Low visibility (<0.5) on NOSE/SHOULDERS"
            else:
                pose_ok = True
                ls = lm_to_px(ls_lm, w, h)
                rs = lm_to_px(rs_lm, w, h)
                nose = lm_to_px(nose_lm, w, h)

                # optional ears
                left_ear = None
                right_ear = None
                le_lm = lms[IDX_LE]
                re_lm = lms[IDX_RE]
                if lm_visible(le_lm) and lm_visible(re_lm):
                    left_ear = lm_to_px(le_lm, w, h)
                    right_ear = lm_to_px(re_lm, w, h)

                out = neck.update(
                    t=t_now, w=w, h=h,
                    ls=ls, rs=rs, nose=nose,
                    left_ear=left_ear, right_ear=right_ear,
                    yawn_conf=None
                )

                # ROI bbox (an toàn)
                x1, y1, x2, y2 = out["roi"]
                safe_rect(frame, x1, y1, x2, y2, color=(0, 255, 0), thickness=2)

                # draw key points
                nx, ny = int(nose[0]), int(nose[1])
                lxs, lys = int(ls[0]), int(ls[1])
                rxs, rys = int(rs[0]), int(rs[1])
                cx, cy = int(out["neck"][0]), int(out["neck"][1])

                cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)      # nose
                cv2.circle(frame, (lxs, lys), 5, (255, 255, 0), -1)    # L shoulder
                cv2.circle(frame, (rxs, rys), 5, (255, 255, 0), -1)    # R shoulder
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)      # neck
                tx, ty = int(out["head_top"][0]), int(out["head_top"][1])
                cv2.circle(frame, (tx, ty), 6, (0, 255, 255), -1)


                # log
                writer.writerow([
                    frame_id, timestamp_ms,
                    1,
                    int(out["gate_ok"]),
                    f"{out['quality']:.3f}",
                    out["state"],
                    f"{out['angle_signed']:.3f}",
                    f"{out['angle_abs']:.3f}",
                    f"{out['yaw']:.3f}" if out["yaw"] is not None else "",
                    f"{out['warn_timer']:.3f}",
                    f"{out['drowsy_timer']:.3f}",
                    int(out["nodding"]),
                    f"{ls[0]:.1f}", f"{ls[1]:.1f}",
                    f"{rs[0]:.1f}", f"{rs[1]:.1f}",
                    f"{nose[0]:.1f}", f"{nose[1]:.1f}",
                ])

        if not pose_ok:
            draw_info(frame, None, fps_smooth, pose_ok=False, msg=msg)
            writer.writerow([frame_id, timestamp_ms, 0, 0, "", "NOPOSE", "", "", "", "", "", 0, "", "", "", "", "", ""])
        else:
            draw_info(frame, out, fps_smooth, pose_ok=True)

        cv2.imshow("Neck Angle Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_id += 1

    csv_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved log: {args.out_csv}")


if __name__ == "__main__":
    main()
