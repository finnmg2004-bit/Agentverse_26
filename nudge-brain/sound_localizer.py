#!/usr/bin/env python3
"""
Sound-source localisation from the robot's own ears (`nudge-brain/sound_localizer.py`).

Instead of looking a sound up by its label ("doorbell -> entry"), the robot works out WHERE
it is from what it hears, the way a real dog turns its head and moves:

  * Every 0.5 s audio window gives two cues (hearing.py):
      - direction: interaural level/time difference -> sin(relative azimuth). This is
        front/back ambiguous (sin(a) == sin(180 - a)); turning or moving resolves it.
      - loudness: RMS, compared with what a known source level would produce at each
        candidate point given distance and the walls in between (house floor plan).
  * A grid of candidate points covering the house (0.5 m) accumulates the log-likelihood of
    every measurement taken from wherever the robot was standing -> triangulation.
  * The estimate is the posterior mean; its spread is the uncertainty. Converged once a few
    measurements agree to within ~1.5 m.

The acoustic model mirrors sim/sensors.py (inverse-distance attenuation, 0.45x per wall up
to 3 walls); source loudness comes from the calibrated per-sound `level` and the reference
clips in assets/audio/.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sim.house_layout import HOUSE_BOUNDS, ROOMS, SOUND_SOURCES, WALL_SEGMENTS_2D, get_room_at

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_RES = 0.5
SIGMA_SIN = 0.18          # direction cue noise (in sin(azimuth) units)
SIGMA_LOG_LEVEL = 0.45    # loudness cue noise (natural-log units, ~ +-55 %)
MIN_MEAS = 3
CONVERGED_STD = 1.5       # metres
MAX_MEAS = 40
OTHER_SOUND_MAX = 0.35    # skip windows where another sound is also loud (mixed cues)


def _clip_rms() -> Dict[str, float]:
    """Reference RMS of each known sound's clip (the robot's calibration recordings)."""
    from scipy.io import wavfile

    out: Dict[str, float] = {}
    for name in SOUND_SOURCES:
        path = PROJECT_ROOT / "assets" / "audio" / f"{name}.wav"
        try:
            _, data = wavfile.read(str(path))
            if data.ndim > 1:
                data = data[:, 0]
            x = data.astype(np.float64) / 32768.0
            out[name] = float(np.sqrt(np.mean(x * x))) or 0.2
        except Exception:
            out[name] = 0.2
    return out


def _walls_crossed(p: Tuple[float, float], pts: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Vectorised count of wall segments crossed by p -> each point in pts. Returns (N,) ints."""
    ax, ay = p
    bx, by = pts[:, 0:1], pts[:, 1:2]             # (N, 1)
    cx, cy, dx, dy = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]  # (S,)

    def ccw(x1, y1, x2, y2, x3, y3):
        return (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3 - x1)

    o1 = ccw(ax, ay, cx, cy, dx, dy)               # (S,)
    o2 = ccw(bx, by, cx, cy, dx, dy)               # (N, S)
    o3 = ccw(ax, ay, bx, by, cx, cy)               # (N, S)
    o4 = ccw(ax, ay, bx, by, dx, dy)               # (N, S)
    return ((o1 != o2) & (o3 != o4)).sum(axis=1)


class SoundTrack:
    """Accumulated evidence for one sound (reset when that sound goes silent)."""

    def __init__(self, sound: str, grid: np.ndarray) -> None:
        self.sound = sound
        self.ll = np.zeros(len(grid))
        self.meas: List[Dict[str, Any]] = []  # {"pose", "rel_az_deg", "rms"}
        self.estimate: Optional[Tuple[float, float]] = None
        self.std: float = float("inf")
        self.room_probs: Dict[str, float] = {}  # posterior mass per room (the mean point can sit in
                                                # the wrong room while the spread straddles a wall)


class SoundLocalizer:
    def __init__(self) -> None:
        b = HOUSE_BOUNDS
        xs = np.arange(b["x_min"] + GRID_RES / 2, b["x_max"], GRID_RES)
        ys = np.arange(b["y_min"] + GRID_RES / 2, b["y_max"], GRID_RES)
        pts = [(x, y) for y in ys for x in xs if get_room_at(x, y) in ROOMS]
        self.grid = np.array(pts, dtype=np.float64)
        self.grid_rooms = [get_room_at(x, y) for x, y in pts]
        self.segs = np.array([[a[0], a[1], c[0], c[1]] for a, c in WALL_SEGMENTS_2D], dtype=np.float64)
        self.ref_rms = _clip_rms()
        self.tracks: Dict[str, SoundTrack] = {}

    # ------------------------------------------------------------------ model
    def _expected_rms(self, sound: str, robot_xy: Tuple[float, float]) -> np.ndarray:
        d = np.maximum(0.25, np.hypot(self.grid[:, 0] - robot_xy[0], self.grid[:, 1] - robot_xy[1]))
        walls = np.minimum(_walls_crossed(robot_xy, self.grid, self.segs), 3)
        level = float(SOUND_SOURCES[sound].get("level", 1.0))
        gain = level * np.maximum(0.12, (1.0 / (1.0 + 0.22 * d)) * (0.45 ** walls))
        return gain * self.ref_rms[sound]

    # ------------------------------------------------------------------ update
    def update(self, res: Dict[str, Any], pose: Tuple[float, float, float]) -> Optional[str]:
        """Feed one hearing result. Returns the sound name if its estimate changed."""
        label = res.get("label")
        if label not in SOUND_SOURCES or float(res.get("confidence", 0.0)) < 0.5:
            return None
        scores = res.get("scores", {})
        if any(float(v) > OTHER_SOUND_MAX for k, v in scores.items() if k != label):
            return None
        track = self.tracks.get(label)
        if track is None:
            track = self.tracks[label] = SoundTrack(label, self.grid)
        if len(track.meas) >= MAX_MEAS:
            return None

        x, y, yaw = pose
        rel_meas = math.radians(float(res.get("rel_azimuth_deg", 0.0)))
        rms = float(res.get("rms", 0.0))
        if rms <= 0.0:
            return None

        # Direction likelihood (front/back ambiguous by construction: compares sin values).
        ang = np.arctan2(self.grid[:, 1] - y, self.grid[:, 0] - x) - yaw
        ll_dir = -0.5 * ((np.sin(ang) - math.sin(rel_meas)) / SIGMA_SIN) ** 2
        # Loudness likelihood.
        exp_rms = self._expected_rms(label, (x, y))
        ll_lvl = -0.5 * ((np.log(rms) - np.log(exp_rms)) / SIGMA_LOG_LEVEL) ** 2
        track.ll += ll_dir + ll_lvl
        track.meas.append({"pose": (x, y, yaw), "rel_az_deg": math.degrees(rel_meas), "rms": rms})

        w = np.exp(track.ll - track.ll.max())
        w /= w.sum()
        mx, my = float(w @ self.grid[:, 0]), float(w @ self.grid[:, 1])
        var = float(w @ ((self.grid[:, 0] - mx) ** 2 + (self.grid[:, 1] - my) ** 2))
        track.estimate = (mx, my)
        track.std = math.sqrt(max(var, 0.0))
        probs: Dict[str, float] = {}
        for room, wi in zip(self.grid_rooms, w):
            probs[room] = probs.get(room, 0.0) + float(wi)
        track.room_probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        return label

    def reset(self, sound: Optional[str] = None) -> None:
        if sound is None:
            self.tracks.clear()
        else:
            self.tracks.pop(sound, None)

    # ------------------------------------------------------------------ queries
    def converged(self, sound: str) -> bool:
        t = self.tracks.get(sound)
        return t is not None and len(t.meas) >= MIN_MEAS and t.std <= CONVERGED_STD

    def estimate(self, sound: str) -> Optional[Dict[str, Any]]:
        t = self.tracks.get(sound)
        if t is None or t.estimate is None:
            return None
        room = get_room_at(*t.estimate)
        top = {r: round(p, 3) for r, p in list(t.room_probs.items())[:3]}
        return {"xy": t.estimate, "std": t.std, "n": len(t.meas), "room": room if room in ROOMS else None,
                "converged": self.converged(sound), "meas": t.meas[-8:],
                "room_probs": top, "room_prob": dict(t.room_probs)}
