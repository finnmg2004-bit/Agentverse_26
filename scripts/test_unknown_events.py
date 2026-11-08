#!/usr/bin/env python3
"""Offline checks for P4 unknown sounds (`nudge-brain/unknown_events.py`), no sim, no Ollama.

1. Detection: the synthesized unknown clips (assets/audio/unknown/) through hearing.process_audio
   -> the expected category is detected (2-window debounce).
2. No false triggers: every registry clip (doorbell, knock, smoke, kettle, oven timer) and the owner's
   voice, through the same path, never produce a non-log unknown event.
3. Decisions (rule fallback, Gemma offline): danger floor = alert_owner by "safety", busy mission =
   no hijack, benign = log_only, emergency = logged only.

    pixi run python scripts/test_unknown_events.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nudge-brain"))
os.environ.setdefault("NUDGE_RUN_LOG", "0")

from hearing import StereoHearing  # noqa: E402
from sound_localizer import SoundLocalizer  # noqa: E402
from world_model import WorldModel  # noqa: E402
from ask import AskOwner  # noqa: E402
import unknown_events as ue  # noqa: E402

FAILS = []


def check(ok: bool, what: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILS.append(what)


class StubAgent:
    enabled, llm_online, model = False, False, None

    def __init__(self):
        self.decisions = []

    def cancel(self):
        pass

    def submit(self, event, world):
        self.submitted = event


class StubExec:
    class _S:
        name = ""
    skill = _S()

    def execute(self, action, world):
        self.last = action


class FakeBrain:
    def __init__(self):
        self.world = WorldModel()
        self.agent = StubAgent()
        self.executor = StubExec()
        self.ask = AskOwner()
        self.latched = {}


def windows(path: Path, gain: float = 1.0):
    _, d = wavfile.read(str(path))
    x = d.astype(np.float32) / 32768.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.tile(x, 2) * gain
    for s in range(0, len(x) - 8000, 8000):
        m = x[s:s + 8000]
        # a little to the left: L louder (ILD) for a non-trivial bearing
        yield np.stack([m * 1.1, m * 0.9], axis=-1)


REG_MAX = {}


def run_clip(hearing, det: "ue.UnknownEvents", brain, path: Path, gain=0.5):
    cats = set()
    for st in windows(path, gain):
        res = hearing.process_audio(st, brain.world.robot_pose)
        for k, v in (res.get("scores") or {}).items():
            REG_MAX[(path.stem, k)] = max(REG_MAX.get((path.stem, k), 0.0), float(v))
        for c in det.candidates(res, brain.latched):
            cats.add(c)
        det.on_audio(res, brain)
    return cats


def main() -> int:
    hearing = StereoHearing()
    if hearing.tflite_interpreter is None:
        print("YAMNet missing (pixi run fetch-models): skipping")
        return 0
    loc = SoundLocalizer()
    print("1. detection of the unknown clips")
    expect = {"glass_break": "glass breaking", "baby_cry": "baby crying", "crash": "crash",
              "water_running": "water running"}
    for name, cat in expect.items():
        brain = FakeBrain()
        det = ue.UnknownEvents(brain.agent, loc)
        cats = run_clip(hearing, det, brain, ROOT / "assets" / "audio" / "unknown" / f"{name}.wav")
        fired = det.pending["cat"] if det.pending else None
        reg = max(((k, v) for (c, k), v in REG_MAX.items() if c == name), key=lambda kv: kv[1])
        check(reg[1] < 0.5, f"{name}: no registry sound fires (max {reg[0]} {reg[1]:.2f})")
        if cat is None:
            print(f"  info {name}: categories {sorted(cats)}, event {fired}")
            check(fired is None or ue.RULES.get(fired) == "log_only" or fired not in ue.DANGER,
                  f"{name}: no danger event")
        else:
            check(fired == cat, f"{name}: event {fired!r} (expected {cat!r}; seen {sorted(cats)})")

    print("2. no false unknown events from registry sounds / the owner's voice")
    for name in ["doorbell", "knock", "smoke_alarm", "kettle", "oven_timer", "owner_voice"]:
        for gain in (0.15, 0.6):
            brain = FakeBrain()
            det = ue.UnknownEvents(brain.agent, loc)
            run_clip(hearing, det, brain, ROOT / "assets" / "audio" / f"{name}.wav", gain)
            fired = det.pending["cat"] if det.pending else None
            check(fired is None or ue.RULES.get(fired) == "log_only",
                  f"{name} x{gain}: no unknown event (got {fired!r})")

    print("2b. no unknown events from the background-noise profiles (60 s each)")
    from sim.sensors import NOISE_NAMES, AmbientNoise
    for prof in NOISE_NAMES:
        noise = AmbientNoise(np.random.default_rng(1))
        noise.set(prof)
        brain = FakeBrain()
        det = ue.UnknownEvents(brain.agent, loc)
        fired = []
        for i in range(120):
            l, r = noise.generate(8000)
            res = hearing.process_audio(np.stack([l, r], axis=-1), brain.world.robot_pose)
            brain.world.t_sim = i * 0.5
            det.on_audio(res, brain)
            if det.pending:
                fired.append(det.pending["cat"])
                det.pending = None
        bad = [c for c in fired if ue.RULES.get(c) != "log_only"]
        check(not bad, f"noise {prof}: events {fired or 'none'}")

    print("3. decisions (rules, Gemma offline)")
    est = {"room": "kitchen", "p": 0.8, "xy": (11, 3), "room_probs": {"kitchen": 0.8}}

    def decide(cat, mission=None, emergency=False):
        brain = FakeBrain()
        det = ue.UnknownEvents(brain.agent, loc)
        if mission:
            brain.world.start_mission(mission)
        p = {"cat": cat, "cls": cat, "conf": 0.9, "room": "kitchen", "est": est, "danger": cat in ue.DANGER,
             "busy": brain.world.mission is not None, "t0": 0.0, "t_sim": 0.0, "classes": [], "result": None}
        p["options"] = ["alert_owner"] if p["danger"] else (["log_only"] if p["busy"] else list(ue.ACTIONS))
        p["rule"] = ue.RULES.get(cat, "investigate") if ue.RULES.get(cat, "investigate") in p["options"] else p["options"][0]
        det.decide(p, {"error": "offline"}, 0.1, brain)
        return det.status, brain

    st, b = decide("glass breaking")
    check(st["action"] == "alert_owner" and st["by"] == "safety", f"glass -> {st['action']} by {st['by']}")
    check(b.world.mission is not None and b.world.mission.get("task_kind") == "find_alert",
          "glass -> find-and-tell mission started")
    st, b = decide("glass breaking", mission="doorbell")
    check(b.world.mission["sound"] == "doorbell" and "after the doorbell" in st["executed"],
          f"glass during doorbell -> not hijacked ({st['executed']})")
    st, b = decide("baby crying", mission="doorbell")
    check(b.world.mission["sound"] == "doorbell" and st["action"] == "log_only", "baby during doorbell -> logged")
    st, b = decide("baby crying")
    check(st["action"] == "ask_owner" and b.ask.current is not None, "baby -> ask_owner (C5 ask open)")
    st, b = decide("dog barking")
    check(st["action"] == "log_only" and b.world.mission is None, "dog -> log_only")
    st, b = decide("crash", mission="smoke_alarm")
    check(b.world.mission["sound"] == "smoke_alarm" and "fire" in st["executed"], "crash in a fire -> logged only")
    st, b = decide("water running")
    check(st["action"] == "ask_owner", "water running -> ask_owner")

    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
