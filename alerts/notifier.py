from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time

try:
    import winsound
except Exception:
    winsound = None

@dataclass
class NotifierConfig:
    wav_enabled: bool = True
    wav_path: str = "assets/sounds/alarm.wav"
    min_restart_interval_s: float = 1.5  # tránh restart quá dày
    beep_fallback: bool = True
    beep_freq: int = 1600
    beep_ms: int = 180

class Notifier:
    """
    Call notify(level) every frame (or whenever you have a decision).
    - When level == ALARM: start playing alarm (async).
    - When level != ALARM: stop alarm.
    """
    def __init__(self, cfg: NotifierConfig):
        self.cfg = cfg
        self._playing = False
        self._last_start_t = 0.0

    def notify(self, level: str) -> None:
        if level == "ALARM":
            self._start_alarm()
        else:
            self._stop_alarm()

    def _start_alarm(self) -> None:
        now = time.time()
        if self._playing and (now - self._last_start_t) < self.cfg.min_restart_interval_s:
            return

        if winsound is None:
            # fallback
            if self.cfg.beep_fallback:
                print("[ALARM] winsound unavailable")
            return

        wav = Path(self.cfg.wav_path)
        if self.cfg.wav_enabled and wav.exists():
            # async playback
            winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._playing = True
            self._last_start_t = now
        else:
            # fallback beep pattern
            if self.cfg.beep_fallback:
                winsound.Beep(int(self.cfg.beep_freq), int(self.cfg.beep_ms))
                self._playing = False

    def _stop_alarm(self) -> None:
        if winsound is None:
            self._playing = False
            return
        if self._playing:
            # purge any currently playing sound
            winsound.PlaySound(None, winsound.SND_PURGE)
            self._playing = False
