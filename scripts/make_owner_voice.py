#!/usr/bin/env python3
"""
Build assets/audio/owner_voice.wav: what the owner's voice sounds like at 1 m, for the sim's ears.

The sim plays it from the owner's position whenever they talk to Nudge (voice mic or GUI text box),
so the robot can work out WHERE the owner is from the direction and loudness of their voice
(nudge-brain/hearing.py speech score + sound_localizer.py + owner_belief.py).

Source: the Piper TTS test sentences in assets/audio/voice_tests/ (22.05 kHz), silences trimmed,
joined into one continuous ~8 s utterance, resampled to the ears' 16 kHz and normalised.

    pixi run python scripts/make_owner_voice.py
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "audio" / "voice_tests"
OUT = ROOT / "assets" / "audio" / "owner_voice.wav"
CLIPS = ["teach_bedtime.wav", "cmd_kitchen.wav", "teach_kettle.wav", "cmd_come.wav"]
TARGET_RMS = 0.12  # normal speech, a bit quieter than the doorbell clip


def trim(x: np.ndarray, thr: float = 0.02) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x) > thr)
    return x[idx[0]: idx[-1] + 1] if len(idx) else x


def main() -> None:
    parts = []
    for name in CLIPS:
        sr, d = wavfile.read(str(SRC / name))
        x = d.astype(np.float64) / 32768.0
        if x.ndim > 1:
            x = x[:, 0]
        if sr != 16000:
            x = resample_poly(x, 16000, sr)
        parts.append(trim(x))
        parts.append(np.zeros(int(0.12 * 16000)))  # a breath between sentences
    y = np.concatenate(parts)
    y *= TARGET_RMS / (np.sqrt(np.mean(y * y)) + 1e-9)
    y = np.clip(y, -0.98, 0.98)
    wavfile.write(str(OUT), 16000, (y * 32767).astype(np.int16))
    print(f"wrote {OUT} ({len(y) / 16000:.1f} s, rms {np.sqrt(np.mean(y * y)):.3f})")


if __name__ == "__main__":
    main()
