1) Thông số phân loại trạng thái (NeckCfg)

normal_deg = 15.0
→ |Angle| < 15° ⇒ NORMAL

warn_deg = 25.0
→ 15–25° ⇒ WARNING
→ >25° ⇒ DROWSY

warn_hold_s = 1.2
→ phải giữ trong vùng WARNING đủ 1.2s mới lên WARNING

drowsy_hold_s = 1.8
→ phải giữ trong vùng DROWSY đủ 1.8s mới lên DROWSY

ema_alpha = 0.25
→ làm mượt góc, giảm rung (EMA smoothing)

2) Gate (lọc tư thế không tin cậy)

gate_angle_deg = 40.0
→ |Angle| ≥ 40° ⇒ Gate CLOSED ⇒ NO_POSE

gate_yaw_ratio = 0.30
→ yaw proxy ≥ 0.30 ⇒ Gate CLOSED ⇒ NO_POSE

Khi Gate CLOSED: timers không tăng (giảm nhẹ theo decay)

3) Quality (điểm tin cậy)

norm_angle = 45.0 (≥ gate_angle)

norm_yaw = 0.30 (khớp gate_yaw_ratio)
→ Quality giảm khi nghiêng nhiều/quay đầu nhiều.

4) Nodding (gật đầu ép DROWSY)

nod_angle_th = 28.0

nod_rate_th = 45.0 deg/s

nod_refractory_s = 1.0
→ nếu “gật mạnh + nhanh” ⇒ ép sang DROWSY (khi gate OK)

5) Timers decay + reset khi NORMAL

decay_ok = 1.0 (khi gate OK nhưng không trong vùng)

decay_gate_fail = 0.5 (khi gate fail, giảm mềm hơn)

Bạn đã yêu cầu: khi quay về NORMAL thì WARN=0s và DROWSY=0s (reset về 0)

6) BƯỚC 2 (điểm neck)

Bạn chọn Cách 1:

Neck = trung điểm 2 vai, đẩy lên 0.15 * sw (sw = độ rộng vai)
→ ổn định hơn, ít ăn xuống ngực.

7) Dấu chấm đỉnh đầu (không dùng mắt)

Dot “ngọn đầu” = head_top = nose + k*(nose - neck)

k = head_top_extend ≈ 0.9 (tăng/giảm để chấm đúng tóc)

8) ROI (khung xanh) dạng hình vuông, khít

ROI tính từ neck (chấm tím) lên head_top (chấm đỉnh đầu)

Ép vuông: width = height = side

pad ≈ 0.08–0.10 để khít/thoáng nhẹ

9) Phím chạy demo (theo README bạn gửi)
Usage:
  python run_head_pose_demo.py --model models/pose_landmarker_lite.task --cam 0
  
c : recalibrate baseline (ngồi thẳng ~10s) (nếu code hiện tại có phần baseline)

q : thoát