from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from core.signals import DrowsinessSignals, AlertDecision

@dataclass
class AlertConfig:
    # Eye / microsleep
    microsleep_s: float = 1.2              # mắt đóng liên tục >= 1.2s => ALARM

    # PERCLOS thresholds (percent)
    perclos_warn: float = 20.0
    perclos_alarm: float = 30.0

    # Neck pitch thresholds (deg)
    neck_warn_deg: float = 18.0            # cúi gật mức warn (sustained)
    neck_alarm_deg: float = 25.0           # cúi gật mức alarm (sustained)
    neck_sustain_s: float = 1.0            # duy trì >= 1s mới tính

    # Yawn policy (simple; teammate có thể thay)
    yawn_warn: bool = True                # chỉ cần yawn=True => WARNING (demo)

    # Hysteresis / cooldown
    cooldown_s: float = 3.0               # cần ổn định >= 3s để hạ cảnh báo

class AlertEngine:
    """
    State machine: NORMAL -> WARNING -> ALARM with hysteresis.
    """
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._state: str = "NORMAL"
        self._last_state_change_t: Optional[float] = None

        # timers
        self._eye_closed_start: Optional[float] = None
        self._neck_high_start: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def update(self, s: DrowsinessSignals) -> AlertDecision:
        reasons: List[str] = []

        # If no face -> degrade to NORMAL (or keep previous? choose NORMAL to avoid false alarm)
        if not s.face_present:
            self._reset_timers()
            return self._set_state("NORMAL", s.timestamp, ["no_face"])

        # --- Eye microsleep ---
        eye_alarm = False
        if s.eye_closed:
            if self._eye_closed_start is None:
                self._eye_closed_start = s.timestamp
            closed_dur = s.timestamp - self._eye_closed_start
            if closed_dur >= self.cfg.microsleep_s:
                eye_alarm = True
                reasons.append(f"microsleep>= {self.cfg.microsleep_s:.1f}s")
        else:
            self._eye_closed_start = None

        # --- PERCLOS ---
        perclos_warn = False
        perclos_alarm = False
        if s.perclos_valid and (s.perclos is not None):
            if s.perclos >= self.cfg.perclos_alarm:
                perclos_alarm = True
                reasons.append(f"perclos>={self.cfg.perclos_alarm:.0f}%")
            elif s.perclos >= self.cfg.perclos_warn:
                perclos_warn = True
                reasons.append(f"perclos>={self.cfg.perclos_warn:.0f}%")

        # --- Neck pitch sustained ---
        neck_warn = False
        neck_alarm = False
        if s.neck_valid and (s.neck_pitch_deg is not None):
            pitch = float(s.neck_pitch_deg)
            if pitch >= self.cfg.neck_warn_deg:
                if self._neck_high_start is None:
                    self._neck_high_start = s.timestamp
                neck_dur = s.timestamp - self._neck_high_start
                if pitch >= self.cfg.neck_alarm_deg and neck_dur >= self.cfg.neck_sustain_s:
                    neck_alarm = True
                    reasons.append(f"neck_pitch>={self.cfg.neck_alarm_deg:.0f}deg")
                elif neck_dur >= self.cfg.neck_sustain_s:
                    neck_warn = True
                    reasons.append(f"neck_pitch>={self.cfg.neck_warn_deg:.0f}deg")
            else:
                self._neck_high_start = None
        else:
            self._neck_high_start = None

        # --- Yawn ---
        yawn_warn = False
        if self.cfg.yawn_warn and s.yawn:
            yawn_warn = True
            reasons.append("yawn_detected")

        # --- Decide target level ---
        target = "NORMAL"
        if eye_alarm or perclos_alarm or neck_alarm:
            target = "ALARM"
        elif perclos_warn or neck_warn or yawn_warn:
            target = "WARNING"

        # --- Hysteresis / cooldown ---
        return self._apply_hysteresis(target, s.timestamp, reasons)

    def _apply_hysteresis(self, target: str, now: float, reasons: List[str]) -> AlertDecision:
        # escalate immediately
        if self._state == "NORMAL":
            if target in ("WARNING", "ALARM"):
                return self._set_state(target, now, reasons)
            return self._set_state("NORMAL", now, [])
        if self._state == "WARNING":
            if target == "ALARM":
                return self._set_state("ALARM", now, reasons)
            if target == "NORMAL":
                if self._cooldown_ok(now):
                    return self._set_state("NORMAL", now, ["cooldown_ok"])
                return AlertDecision(level="WARNING", reasons=reasons or ["cooldown_wait"])
            return AlertDecision(level="WARNING", reasons=reasons)
        if self._state == "ALARM":
            if target in ("WARNING", "NORMAL"):
                if self._cooldown_ok(now):
                    return self._set_state("WARNING" if target == "WARNING" else "NORMAL", now, ["cooldown_ok"])
                return AlertDecision(level="ALARM", reasons=reasons or ["cooldown_wait"])
            return AlertDecision(level="ALARM", reasons=reasons)

        return self._set_state(target, now, reasons)

    def _cooldown_ok(self, now: float) -> bool:
        if self._last_state_change_t is None:
            return True
        return (now - self._last_state_change_t) >= self.cfg.cooldown_s

    def _set_state(self, new_state: str, now: float, reasons: List[str]) -> AlertDecision:
        if new_state != self._state:
            self._state = new_state
            self._last_state_change_t = now
        return AlertDecision(level=self._state, reasons=reasons)

    def _reset_timers(self) -> None:
        self._eye_closed_start = None
        self._neck_high_start = None
