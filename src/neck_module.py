# neck_module.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ema(prev: Optional[float], cur: float, alpha: float) -> float:
    return cur if prev is None else alpha * cur + (1 - alpha) * prev


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_neck_to_head_signed(neck: Tuple[float, float], head: Tuple[float, float]) -> float:
    """
    Signed angle (deg) between vector (neck->head) and Oy axis (upward).
    In image coords, Oy(up) = (0, -1).
      0°  : head straight up
      +   : lean to the right
      -   : lean to the left
    """
    vx = head[0] - neck[0]
    vy = head[1] - neck[1]
    n = math.hypot(vx, vy) + 1e-6
    vx /= n
    vy /= n

    dot = max(-1.0, min(1.0, -vy))       # dot with (0, -1)
    ang = math.degrees(math.acos(dot))   # 0..180
    return ang if vx >= 0 else -ang


def yaw_ratio_proxy(nose: Tuple[float, float], left_ear: Tuple[float, float], right_ear: Tuple[float, float]) -> float:
    dL = dist2d(nose, left_ear)
    dR = dist2d(nose, right_ear)
    return abs(dL - dR) / (dL + dR + 1e-6)


def head_neck_bbox(
    neck: Tuple[float, float],
    head_top: Tuple[float, float],
    w: int, h: int,
    pad: float = 0.08 # dư nhẹ cho tóc/cổ
) -> Tuple[int, int, int, int]:
    """
    ROI HÌNH VUÔNG:
    - Đáy: cổ (neck)
    - Đỉnh: đỉnh đầu (head_top)
    - Rộng = Cao
    """
    # chiều cao từ cổ -> đỉnh đầu
    side = abs(neck[1] - head_top[1])
    side = max(1.0, side)

    # padding nhẹ
    side = side * (1.0 + pad)

    # tâm theo cổ
    cx = neck[0]

    # y: trên = đỉnh đầu, dưới = cổ
    y2 = int(neck[1] + 0.05 * side)
    y1 = int(y2 - side)

    # x theo hình vuông
    x1 = int(cx - side / 2)
    x2 = int(cx + side / 2)

    # clamp
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


@dataclass
class NeckCfg:
    ema_alpha: float = 0.25

    # thresholds on |angle_smooth|
    normal_deg: float = 15.0      # |angle| < 15 => NORMAL
    warn_deg: float = 25.0        # 15..25 (inclusive) => WARNING; >25 => DROWSY

    warn_hold_s: float = 1.2
    drowsy_hold_s: float = 1.8

    # nodding: angle lớn + tốc độ lớn   
    nod_angle_th: float = 28.0
    nod_rate_th: float = 45.0 # deg/s
    nod_refractory_s: float = 1.0

    # quality gate
    gate_angle_deg: float = 40.0
    gate_yaw_ratio: float = 0.30

    # quality normalizers
    norm_angle: float = 45.0
    norm_yaw: float = 0.30

    # timer decay when conditions not met
    decay_ok: float = 1.0          # decay when gate ok but not in region
    decay_gate_fail: float = 0.5   # decay when gate fail (mềm hơn)


@dataclass
class NeckState:
    angle_s: Optional[float] = None
    yaw_s: Optional[float] = None

    warn_timer: float = 0.0
    drowsy_timer: float = 0.0

    last_angle: Optional[float] = None
    last_t: Optional[float] = None
    last_nod_t: float = -1e9


class NeckModule:
    def __init__(self, cfg: NeckCfg):
        self.cfg = cfg
        self.st = NeckState()

    def update(
        self,
        t: float,
        w: int, h: int,
        ls: Tuple[float, float], rs: Tuple[float, float],
        nose: Tuple[float, float],
        left_ear: Optional[Tuple[float, float]] = None,
        right_ear: Optional[Tuple[float, float]] = None,
        yawn_conf: Optional[float] = None
    ) -> Dict:
        # cổ = trung điểm vai
        sw = max(1.0, dist2d(ls, rs))
        neck = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2 - 0.15 * sw)
        # TOP OF HEAD (đỉnh đầu) 
        vx = nose[0] - neck[0]
        vy = nose[1] - neck[1]
        head_top = (
            nose[0] + 0.9 * vx,
            nose[1] + 0.9 * vy
        )


        # góc cổ so với Oy (neck->nose)
        angle_raw = angle_neck_to_head_signed(neck, nose)

        # yaw proxy (optional)
        yaw_raw = None
        yaw_available = (left_ear is not None and right_ear is not None)
        if yaw_available:
            yaw_raw = yaw_ratio_proxy(nose, left_ear, right_ear)

        # EMA smoothing
        self.st.angle_s = ema(self.st.angle_s, angle_raw, self.cfg.ema_alpha)
        if yaw_raw is not None:
            self.st.yaw_s = ema(self.st.yaw_s, yaw_raw, self.cfg.ema_alpha)

        angle_signed = float(self.st.angle_s)
        yaw = float(self.st.yaw_s) if (yaw_raw is not None and self.st.yaw_s is not None) else None
        angle_abs = abs(angle_signed)

        # dt
        dt = 0.0
        if self.st.last_t is not None:
            dt = max(1e-6, t - self.st.last_t)
        self.st.last_t = t

        # quality gate
        yaw_ok = True
        if yaw_available and yaw is not None:
            yaw_ok = (yaw < self.cfg.gate_yaw_ratio)
        gate_ok = (angle_abs < self.cfg.gate_angle_deg) and yaw_ok

        # quality score (0..1)
        q_angle = clamp(1 - angle_abs / (self.cfg.norm_angle + 1e-6), 0, 1)
        q_yaw = 1.0 if yaw is None else clamp(1 - yaw / (self.cfg.norm_yaw + 1e-6), 0, 1)
        quality = q_angle * q_yaw

        yawn_conf_adj = None
        if yawn_conf is not None:
            yawn_conf_adj = yawn_conf * quality

        # timers with hold
        if not gate_ok:
            # giảm nhẹ timers để đỡ nhảy
            self.st.warn_timer = max(0.0, self.st.warn_timer - self.cfg.decay_gate_fail * dt)
            self.st.drowsy_timer = max(0.0, self.st.drowsy_timer - self.cfg.decay_gate_fail * dt)
        else:
            # WARNING: 10..15 (inclusive)
            if self.cfg.normal_deg <= angle_abs <= self.cfg.warn_deg:
                self.st.warn_timer += dt
            else:
                self.st.warn_timer = max(0.0, self.st.warn_timer - self.cfg.decay_ok * dt)

            # DROWSY: >15 (strict)
            if angle_abs > self.cfg.warn_deg:
                self.st.drowsy_timer += dt
            else:
                self.st.drowsy_timer = max(0.0, self.st.drowsy_timer - self.cfg.decay_ok * dt)

        # nodding: angle lớn + tốc độ lớn (chỉ tính khi có dt)
        nodding = False
        if self.st.last_angle is not None and dt > 0:
            rate = (angle_signed - self.st.last_angle) / dt
            if (
                angle_abs >= self.cfg.nod_angle_th
                and abs(rate) >= self.cfg.nod_rate_th
                and (t - self.st.last_nod_t) >= self.cfg.nod_refractory_s
            ):
                nodding = True
                self.st.last_nod_t = t
        self.st.last_angle = angle_signed

        # state
        if not gate_ok:
            state = "NO_POSE"
        else:
            if self.st.drowsy_timer >= self.cfg.drowsy_hold_s:
                state = "DROWSY"
            elif self.st.warn_timer >= self.cfg.warn_hold_s:
                state = "WARNING"
            else:
                state = "NORMAL"

        # nếu nodding thì ép lên DROWSY (chỉ khi gate_ok)
        if gate_ok and nodding and state in ("NORMAL", "WARNING"):
            state = "DROWSY"

        # ROI nhỏ (cổ -> đầu)

        roi = head_neck_bbox(neck, head_top, w, h)
        return {
            "neck": neck,
            "head_top": head_top,
            "roi": roi,

            "angle_raw": float(angle_raw),
            "yaw_raw": float(yaw_raw) if yaw_raw is not None else None,

            "angle_signed": float(angle_signed),
            "angle_abs": float(angle_abs),
            "yaw": float(yaw) if yaw is not None else None,

            "angle_s": float(angle_signed),
            "yaw_s": float(yaw) if yaw is not None else None,

            "gate_ok": bool(gate_ok),
            "quality": float(quality),

            "warn_timer": float(self.st.warn_timer),
            "drowsy_timer": float(self.st.drowsy_timer),

            "state": state,
            "nodding": bool(nodding),

            "yawn_conf_adj": yawn_conf_adj,
        }
