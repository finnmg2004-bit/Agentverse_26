#!/usr/bin/env python3
"""
Offline check for "Nudge hears where the owner is" (no sim, no brain process):
the owner talks from each owner spot while the robot stands at its start pose; the sim's ear model
(sim/sensors.py) mixes the voice clip, the brain's hearing (nudge-brain/hearing.py) scores it, and
the sound localiser (nudge-brain/sound_localizer.py) estimates the room.

Checks: speech is heard, no household sound fires from speech alone (no false missions), and the
room estimate after ~3 s of talking. Prints one line per (noise, owner room).

    pixi run python scripts/test_owner_voice.py [--noise quiet tv vacuum]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))

from sim.house_layout import OWNER_SPOTS, OWNER_VOICE, ROBOT_START_POSE  # noqa: E402
from sim.sensors import AUDIO_WINDOW_SAMPLES, AmbientNoise, SimSensors  # noqa: E402
from hearing import StereoHearing  # noqa: E402
from sound_localizer import SoundLocalizer  # noqa: E402

WINDOWS = 6  # 3 s of talking
MISSION_CONF = 0.5  # main.py NEW_SOUND_CONF: one window this high starts a mission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", nargs="*", default=["quiet", "tv"])
    a = ap.parse_args()
    sr, d = wavfile.read(str(ROOT / "assets" / "audio" / str(OWNER_VOICE["file"])))
    clip = d.astype(np.float32) / 32768.0
    hearing = StereoHearing()
    bad = 0
    wrong = 0
    for noise_name in a.noise:
        noise = AmbientNoise(np.random.default_rng(1))
        noise.set(noise_name)
        for room, spot in OWNER_SPOTS.items():
            loc = SoundLocalizer()
            pose = ROBOT_START_POSE
            speech, false_hits, rel = [], [], []
            for w in range(WINDOWS):
                left, right = noise.generate(AUDIO_WINDOW_SAMPLES)
                idx = (np.arange(AUDIO_WINDOW_SAMPLES) + w * AUDIO_WINDOW_SAMPLES) % len(clip)
                SimSensors._mix_source(clip[idx], spot, float(OWNER_VOICE["level"]), pose, left, right)
                st = np.clip(np.stack([left, right], axis=-1), -0.99, 0.99)
                pcm = (st * 32767).astype(np.int16).astype(np.float32) / 32768.0
                res = hearing.process_audio(pcm, pose)
                speech.append(res["speech"])
                hits = {k: v for k, v in res["scores"].items() if v >= 0.35}
                if hits:
                    false_hits.append(hits)
                rel.append(res["rel_azimuth_deg"])
                if res["speech"] >= 0.5 and not hits:
                    loc.update(dict(res, label="owner_voice", confidence=res["speech"]), pose)
            est = loc.estimate("owner_voice")
            truth = math.degrees(math.atan2(spot[1] - pose[1], spot[0] - pose[0]) - pose[2])
            truth = (truth + 180) % 360 - 180
            top = est["room_probs"] if est else {}
            best = next(iter(top), None)
            ok = best == room
            # a false MISSION = a household sound >= main.py NEW_SOUND_CONF (0.5) from the voice alone;
            # a wrong room from ONE pose (front/back ambiguity, no head turn) is soft evidence, not a failure
            missions = [h for h in false_hits if max(h.values()) >= MISSION_CONF]
            bad += bool(missions)
            wrong += not ok
            print(f"{noise_name:6} owner in {room:12} speech {np.mean(speech):.2f} (min {min(speech):.2f})  "
                  f"az {np.mean(rel):+5.0f} (true {truth:+4.0f})  -> {best} {top}"
                  f"{'' if ok else '  WRONG ROOM'}{'  FALSE SOUND ' + str(false_hits[:2]) if false_hits else ''}"
                  f"{'  FALSE MISSION' if missions else ''}")
    print(f"\n{bad} case(s) with a false mission, {wrong} wrong top room(s) from one robot pose")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
