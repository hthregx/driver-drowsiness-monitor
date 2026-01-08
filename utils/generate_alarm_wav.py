import wave, math, struct
from pathlib import Path

OUT = Path("assets/sounds/alarm.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Siren parameters
sr = 44100
duration_s = 2.0
base_f1 = 900
base_f2 = 1700
sweep_hz = 1.8  # how fast it sweeps up/down
amp = 0.95      # 0..1 (keep near 1)

n = int(sr * duration_s)
with wave.open(str(OUT), "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(sr)

    for i in range(n):
        t = i / sr
        # Triangle wave sweep between f1 and f2
        phase = (t * sweep_hz) % 1.0
        tri = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)  # 0..1..0
        freq = base_f1 + (base_f2 - base_f1) * tri

        # Tone
        sample = amp * math.sin(2 * math.pi * freq * t)

        # Hard gate to make it more "alarming"
        gate = 1.0 if (math.sin(2 * math.pi * 6.0 * t) > 0) else 0.35
        sample *= gate

        s = int(max(-1.0, min(1.0, sample)) * 32767)
        wf.writeframes(struct.pack("<h", s))

print("WAV generated:", OUT.resolve())
