"""Play an arbitrary WAV clip from a position into the robot's ears (P4 test sources).

gui action `{"action": "play", "file": "unknown/glass_break.wav", "pos": [x, y], "level": 1.0,
"repeat": 1}`: `file` is relative to assets/audio/ (or absolute), 16 kHz mono/stereo WAV. Mixed in via
`SimSensors.synthesize_stereo_audio(extra_sources=...)` like the TV, so the brain hears it with
distance, walls, ILD and ITD. Not an active registry sound: nothing in the sim answers it.
State ground truth (judge only): `state.played_gt` = [file names playing].
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"
WINDOW = 8000  # sensors.AUDIO_WINDOW_SAMPLES (0.5 s at 16 kHz)


def _load(path: Path) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    x = data.astype(np.float32)
    if np.issubdtype(data.dtype, np.integer):
        x /= float(np.iinfo(data.dtype).max)
    if sr != 16000:
        x = resample_poly(x, 16000, sr).astype(np.float32)
    return x


class ClipPlayer:
    def __init__(self) -> None:
        self.playing: List[Dict[str, Any]] = []
        self._cache: Dict[str, np.ndarray] = {}

    def play(self, payload: dict) -> None:
        name = str(payload.get("file", ""))
        path = Path(name) if Path(name).is_absolute() else AUDIO_DIR / name
        try:
            pos = payload.get("pos") or [payload.get("x"), payload.get("y")]
            xy = (float(pos[0]), float(pos[1]))
            if name not in self._cache:
                self._cache[name] = _load(path)
        except Exception as exc:
            print(f"[sim] play: can't play {name!r} ({exc})")
            return
        repeat = max(1, int(payload.get("repeat", 1)))
        clip = np.tile(self._cache[name], repeat)
        self.playing.append({"file": name, "clip": clip, "cursor": 0, "pos": xy,
                             "level": float(payload.get("level", 1.0))})
        print(f"[sim] PLAY {name} x{repeat} ({len(clip) / 16000:.1f}s) from ({xy[0]:.1f}, {xy[1]:.1f})")

    def window(self) -> list:
        """`extra_sources` entries for this 0.5 s audio window."""
        out = []
        for p in list(self.playing):
            seg = p["clip"][p["cursor"]:p["cursor"] + WINDOW]
            p["cursor"] += WINDOW
            if len(seg) < WINDOW:
                seg = np.pad(seg, (0, WINDOW - len(seg)))
            out.append((seg, p["pos"], p["level"]))
            if p["cursor"] >= len(p["clip"]):
                self.playing.remove(p)
        return out

    def names(self) -> list:
        return [p["file"] for p in self.playing]

    def stop(self) -> None:
        self.playing.clear()
