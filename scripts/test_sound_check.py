#!/usr/bin/env python3
"""
Offline check for "Is the sound real?" (nudge-brain/sound_check.py; no sim, no brain process).

For many robot poses around the house, plays a doorbell / knock either from the REAL source (front
door) or from the living-room TV (house_layout.TV_SOURCE) through the sim's ear model
(sim/sensors.py `_mix_source` + ambient noise), runs the brain's hearing (nudge-brain/hearing.py) and,
on the first window the sound is heard, the sound check. Reports per case: verdict and how long the
check waited. A real door must never be called "tv"; TV sounds should mostly be caught.

    pixi run python scripts/test_sound_check.py [--noise quiet tv] [-v]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

from sim.house_layout import ROBOT_START_POSE, ROOM_SCAN_POINTS, SOUND_SOURCES, TV_SOURCE  # noqa: E402
from sim.sensors import AUDIO_WINDOW_SAMPLES, AmbientNoise, SimSensors, ensure_audio_assets  # noqa: E402
from hearing import StereoHearing  # noqa: E402
from sound_check import CHECK_SOUNDS, SoundCheck  # noqa: E402

NEW_SOUND_CONF = 0.5  # nudge-brain/main.py
MAX_WINDOWS = 8       # give up if the sound is never heard


def poses():
    yield "start", ROBOT_START_POSE
    for room, (x, y) in ROOM_SCAN_POINTS.items():
        for yaw_deg in (0, 90, 180, 270):
            yield f"{room}@{yaw_deg}", (x, y, math.radians(yaw_deg))


def run_case(hearing, check, noise, clip, src_xy, level, pose, sound):
    # quiet windows first: flush the hearing buffers, let the check learn the noise floor
    for _ in range(4):
        left, right = noise.generate(AUDIO_WINDOW_SAMPLES)
        res = hearing.process_audio(_pcm(left, right), pose)
        check.observe(res)
    for w in range(MAX_WINDOWS):
        left, right = noise.generate(AUDIO_WINDOW_SAMPLES)
        idx = (np.arange(AUDIO_WINDOW_SAMPLES) + w * AUDIO_WINDOW_SAMPLES) % len(clip)
        SimSensors._mix_source(clip[idx], src_xy, level, pose, left, right)
        res = hearing.process_audio(_pcm(left, right), pose)
        check.observe(res)
        if res.get("label") == sound and float(res.get("confidence", 0.0)) >= NEW_SOUND_CONF:
            v = check.check(sound, res, pose, t=w * 0.5)
            if v != "wait":
                return v, check.status()
    return "unheard", None


def _pcm(left, right):
    st = np.clip(np.stack([left, right], axis=-1), -0.99, 0.99)
    return (st * 32767).astype(np.int16).astype(np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", nargs="*", default=["quiet", "tv"])
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    clips = ensure_audio_assets()
    hearing = StereoHearing()
    false_tv = 0
    for noise_name in a.noise:
        noise = AmbientNoise(np.random.default_rng(1))
        noise.set(noise_name)
        tally = {}
        for sound in CHECK_SOUNDS:
            for where in ("real", "tv"):
                if where == "real":
                    xy, level = SOUND_SOURCES[sound]["pos"][:2], float(SOUND_SOURCES[sound].get("level", 1.0))
                else:
                    xy, level = TV_SOURCE["pos"][:2], float(TV_SOURCE["level"])
                for name, pose in poses():
                    check = SoundCheck()
                    v, st = run_case(hearing, check, noise, clips[sound], xy, level, pose, sound)
                    key = (sound, where)
                    tally.setdefault(key, {}).setdefault(v, 0)
                    tally[key][v] += 1
                    bad = where == "real" and v == "tv"
                    false_tv += bad
                    if a.verbose or bad or name == "start":
                        extra = f"  {st['reason']} (margin {st['margin']:+.1f}, {st['windows']} win)" if st else ""
                        print(f"{noise_name:6} {sound:8} from {where:4} robot {name:18} -> {v:7}{extra}"
                              f"{'  <-- REAL DOOR CALLED TV' if bad else ''}")
        print(f"--- {noise_name}: " + "  ".join(f"{s}/{w}: {d}" for (s, w), d in tally.items()))
    print(f"\n{false_tv} real door sound(s) wrongly called TV")
    return 1 if false_tv else 0


if __name__ == "__main__":
    sys.exit(main())
