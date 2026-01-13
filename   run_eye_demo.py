from __future__ import annotations
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_config
from src.landmarks import EyeLandmarker
from src.eye_metrics import compute_ear, EyeClosureDetector, PerclosWindow


def ensure_parent(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config("config.yaml")

    cam_id = int(cfg["VIDEO"]["CAM_ID"])
    w = int(cfg["VIDEO"]["WIDTH"])
    h = int(cfg["VIDEO"]["HEIGHT"])
    flip = bool(cfg["VIDEO"].get("FLIP", True))

    ear_thresh = float(cfg["EYE"]["EAR_THRESH"])
    min_closed_frames = int(cfg["EYE"]["MIN_CLOSED_FRAMES"])
    window_s = float(cfg["EYE"]["PERCLOS_WINDOW_S"])
    warn = float(cfg["EYE"]["PERCLOS_WARN"])
    drowsy = float(cfg["EYE"]["PERCLOS_DROWSY"])
    severe = float(cfg["EYE"].get("PERCLOS_SEVERE", 0.35))  # optional
    hard_close_s = float(cfg["EYE"]["HARD_CLOSE_S"])

    csv_path = str(cfg["OUTPUTS"]["CSV_PATH"])
    plot_path = str(cfg["OUTPUTS"]["PERCLOS_PLOT_PATH"])
    ensure_parent(csv_path)
    ensure_parent(plot_path)

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    lm = EyeLandmarker()
    closure = EyeClosureDetector(ear_thresh=ear_thresh, min_closed_frames=min_closed_frames)
    perclos = PerclosWindow(window_s=window_s)

    rows = []
    t0 = time.time()

    # ====== Robust continuous-close tracking (based on debounced is_closed) ======
    prev_is_closed = False
    close_start_ts: float | None = None
    last_close_dur = 0.0
    last_close_end_ts: float | None = None
    grace_period_s = 0.5  # giữ trạng thái DROWSY thêm 0.5s sau khi vừa mở nếu nhắm lâu

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            ts = time.time()
            t_rel = ts - t0

            eyes = lm.detect(frame)

            ear = np.nan
            is_closed = False
            closed_cont_s = 0.0
            p = np.nan
            status = "NO_FACE"
            long_close_recent = False

            if eyes is not None:
                # --- EAR ---
                ear = (compute_ear(eyes.left_eye6) + compute_ear(eyes.right_eye6)) / 2.0

                # --- Debounce closure (blink filtering) ---
                st = closure.update(ear=ear, ts=ts)
                is_closed = st.is_closed

                # --- Continuous close duration (DÙNG is_closed đã debounce) ---
                # transition OPEN -> CLOSED
                if is_closed and (not prev_is_closed):
                    close_start_ts = ts
                    last_close_dur = 0.0  # reset for this closure

                # while CLOSED
                if is_closed and close_start_ts is not None:
                    closed_cont_s = ts - close_start_ts
                    last_close_dur = closed_cont_s  # keep updating

                # transition CLOSED -> OPEN
                if (not is_closed) and prev_is_closed:
                    if close_start_ts is not None:
                        last_close_dur = ts - close_start_ts
                    last_close_end_ts = ts
                    close_start_ts = None

                # grace check: vừa mở mắt sau khi nhắm lâu
                if (not is_closed) and (last_close_end_ts is not None):
                    if (ts - last_close_end_ts) <= grace_period_s and last_close_dur >= hard_close_s:
                        long_close_recent = True
                    elif (ts - last_close_end_ts) > grace_period_s:
                        # hết grace thì clear (để UI không giữ hoài)
                        last_close_end_ts = None
                        # last_close_dur giữ lại cũng được, nhưng không còn effect

                prev_is_closed = is_closed

                # --- PERCLOS (chỉ update khi có face) ---
                p = perclos.update_perclos(ts=ts, is_closed=is_closed)

                # --- STATUS LOGIC (đúng & nhất quán) ---
                # DROWSY: nhắm lâu hiện tại, hoặc vừa nhắm lâu xong (grace), hoặc xu hướng quá cao
                if is_closed and closed_cont_s >= hard_close_s:
                    status = "DROWSY"
                elif long_close_recent:
                    status = "DROWSY"
                elif p >= severe:
                    status = "DROWSY"
                elif p >= drowsy:
                    # PERCLOS cao: nếu đang closed -> drowsy, nếu đang open -> warn (window còn nhớ)
                    status = "DROWSY" if is_closed else "WARN"
                elif p >= warn:
                    status = "WARN"
                else:
                    status = "OK"

                # draw eye points
                for (x, y) in eyes.left_eye6 + eyes.right_eye6:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # ====== Overlay ======
            cv2.putText(
                frame,
                f"EAR(th={ear_thresh:.3f}): {0.0 if ear!=ear else ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            show_cont = closed_cont_s
            if (not is_closed) and long_close_recent:
                show_cont = max(show_cont, last_close_dur)

            cv2.putText(
                frame,
                f"Closed:{int(is_closed)} cont:{show_cont:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"PERCLOS@{window_s:.0f}s: {0.0 if p!=p else p:.3f} warn={warn:.2f} drowsy={drowsy:.2f} severe={severe:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            color = (0, 0, 255) if status in ("WARN", "DROWSY") else (0, 255, 0)
            cv2.putText(
                frame,
                f"STATUS: {status}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            cv2.putText(
                frame,
                "Press q to quit",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Eye Task (EAR + PERCLOS)", frame)

            # ====== Log rows ======
            rows.append(
                [
                    t_rel,
                    ts,
                    "" if ear != ear else float(ear),
                    int(is_closed),
                    "" if p != p else float(p),
                    float(show_cont),
                    status,
                ]
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        lm.close()
        cap.release()
        cv2.destroyAllWindows()

    # ====== Save CSV ======
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["t_rel_s", "timestamp", "ear", "is_closed", "perclos", "closed_cont_s", "status"])
        wr.writerows(rows)  

    # ====== Plot PERCLOS ======
    t_vals, p_vals = [], []
    for r in rows:
        if r[4] != "":
            t_vals.append(float(r[0]))
            p_vals.append(float(r[4]))

    if len(t_vals) >= 5:
        plt.figure(figsize=(12, 5))
        plt.plot(t_vals, p_vals)
        plt.axhline(warn)
        plt.axhline(drowsy)
        plt.axhline(severe)
        plt.title(f"PERCLOS over time (window={window_s:.0f}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("PERCLOS")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
