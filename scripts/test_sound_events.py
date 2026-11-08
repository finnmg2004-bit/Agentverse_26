#!/usr/bin/env python3
"""
Offline check of `hearing.SoundEvents` (contract C4: taught tasks fire on `phase == "start"`):
each ring of a sound must produce exactly ONE start event (no repeats while it keeps ringing),
and no start for the other registry sounds.

Uses the sim's own ear synthesis (`sim.sensors.SimSensors.synthesize_stereo_audio`, background
noise profiles included) and the brain's `StereoHearing` + `SoundEvents`, so no sim/brain processes.

    pixi run python scripts/test_sound_events.py                     # kettle + oven_timer, all noise
    pixi run python scripts/test_sound_events.py --sounds doorbell knock --noise quiet tv
    pixi run python scripts/test_sound_events.py --sounds none        # background noise only: any start is false
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nudge-brain")]

from sim.house_layout import SOUND_SOURCES  # noqa: E402
from sim.sensors import NOISE_NAMES, SimSensors  # noqa: E402
from sim.world import build_nudge_model  # noqa: E402
from hearing import SoundEvents, StereoHearing  # noqa: E402

# Robot listening spots: same room, next door, across the house.
SPOTS = {
    "stove": (13.4, 5.6),        # right next to the kettle / oven timer (loud, clipping)
    "kitchen": (8.0, 2.4),
    "robot_home": (8.0, -2.0),
    "entry": (-1.4, -2.0),
    "dining": (-4.5, 1.6),
    "living_room": (-8.5, 4.6),
}
HEADINGS = (0.0, math.pi / 2, math.pi, -math.pi / 2)


def ears(pcm: bytes) -> np.ndarray:
    """Sim PCM -> what the brain's bridge hands to hearing: float32 stereo in [-1, 1]."""
    return np.frombuffer(pcm, np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
WIN_S = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sounds", nargs="*", default=["kettle", "oven_timer"])
    ap.add_argument("--noise", nargs="*", default=NOISE_NAMES)
    ap.add_argument("--rings", type=int, default=2)
    ap.add_argument("--on-s", type=float, default=20.0)
    ap.add_argument("--off-s", type=float, default=6.0)
    ap.add_argument("--headings", type=int, default=len(HEADINGS))
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        model, data = build_nudge_model(verbose=False)
        sensors = SimSensors(model, data, enable_camera=False)
        hearing = StereoHearing()

    t0 = time.monotonic()
    bad = 0
    table = []
    for sound in args.sounds:
        for noise in args.noise:
            sensors.noise.set(noise)
            per_ring = Counter()   # starts per ring -> how many rings had that count
            other = Counter()      # false starts of other sounds
            stops_mid = 0          # stop events while the sound was still on
            for spot, (x, y) in SPOTS.items():
                for yaw in HEADINGS[: args.headings]:
                    pose = (x, y, yaw)
                    ev = SoundEvents()
                    t = 0.0
                    for _ in range(6):  # flush the rolling buffer with silence
                        pcm, _ = sensors.synthesize_stereo_audio(pose, {})
                        hearing.process_audio(ears(pcm), pose)
                    for _ring in range(args.rings):
                        starts = 0
                        for phase, dur in (("on", args.on_s), ("off", args.off_s)):
                            for _w in range(int(dur / WIN_S)):
                                on = {} if sound == "none" else {sound: phase == "on"}
                                pcm, _ = sensors.synthesize_stereo_audio(pose, on)
                                res = hearing.process_audio(ears(pcm), pose)
                                for e in ev.update(res, t):
                                    if e["sound"] != sound:
                                        if e["phase"] == "start":
                                            other[e["sound"]] += 1
                                    elif e["phase"] == "start":
                                        starts += 1
                                    elif phase == "on":
                                        stops_mid += 1
                                t += WIN_S
                        per_ring[starts] += 1
            n = sum(per_ring.values())
            ok = per_ring[1]
            repeats = sum(c for k, c in per_ring.items() if k > 1)
            bad += repeats + sum(other.values())
            table.append((sound, noise, ok, n, per_ring[0], repeats, stops_mid, dict(other)))
            print(f"{sound:11s} {noise:7s} exactly-one {ok:3d}/{n}  missed {per_ring[0]:3d}  "
                  f"repeated {repeats:3d}  stop-while-on {stops_mid:3d}  other-starts {dict(other) or '-'}",
                  flush=True)
    print(f"[sound-events] {time.monotonic() - t0:.0f}s; "
          + ("no repeats, no false starts" if bad == 0 else f"{bad} repeat/false starts"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
