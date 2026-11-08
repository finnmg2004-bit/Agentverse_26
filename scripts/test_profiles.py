#!/usr/bin/env python3
"""
Owner profile check (contract C10, sim/owner_profile.py): each profile switches on the right outputs.

Builds the real NudgeSimulator in-process (headless, no brain), switches the profile through the
same `gui` handler the control GUI uses, feeds it a doorbell alert and then a smoke alarm the way
the brain would (cmd alert_mode + brain status), steps the sim and checks what actually happens:
  * screen  : flank-screen material lit (tier colour, emission > 0) and the floating label text
  * light   : light-bar centre lens lit at least once in 1.2 s (blink pattern), ends strobe in danger
  * speech  : RobotVoice.say_if_changed called with the message (or "[sim] SPEECH OFF" instead)
  * haptic  : guide handle swings up (always on)
  * state   : state["profile"] / state["outputs"] (published every 50 ms)

    pixi run python scripts/test_profiles.py

Uses its own ports (5565-5567), never the sim ports 5555-5559, so it needs no test lock and never
talks to a running sim / brain / GUI. Speech and sound output are muted (the speech call is spied on).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Silent test: no TTS / no laptop-speaker sounds even where they default to on (set before importing the sim).
os.environ.setdefault("NUDGE_SPEECH", "0")
os.environ.setdefault("NUDGE_SOUND_OUT", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

import mujoco  # noqa: E402

from sim.owner_profile import PROFILE_NAMES, channels  # noqa: E402
from sim.run_sim import NudgeSimulator  # noqa: E402

DOORBELL_SAY = "Someone is at the front door. Take my handle."
SMOKE_SAY = "Fire alarm! Take my handle, we go outside now."


def mat(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)


class Probe:
    def __init__(self) -> None:
        self.sim = NudgeSimulator(pub_endpoint="tcp://127.0.0.1:5565", sub_endpoint="tcp://127.0.0.1:5566",
                                  gui_endpoint="tcp://127.0.0.1:5567", enable_camera=False, owner_wander=False,
                                  owner_seed=1)
        m = self.sim.model
        self.screen_mat = mat(m, "screen_left_mat")
        self.center_mat = mat(m, "lightbar_center_mat")
        self.red_mat = mat(m, "lightbar_red_mat")
        assert self.screen_mat >= 0 and self.center_mat >= 0 and self.red_mat >= 0, "screen / light bar missing"
        self.spoken: list[str] = []
        orig = self.sim.voice.say_if_changed

        def spy(text, urgent=False):
            if text and text != self.sim.voice.last_text:
                self.spoken.append(text)
            return orig(text, urgent=urgent)

        self.sim.voice.say_if_changed = spy
        self.dt = float(m.opt.timestep)

    def alert(self, sound: str, alert_mode: str, say: str) -> None:
        s = self.sim
        s.cmd_alert_mode, s.mission_status = alert_mode, "active"
        s.handle_brain_status({"episode": s.episode, "mission": {"status": "active", "sound": sound, "say": say}})

    def clear(self) -> None:
        s = self.sim
        s.cmd_alert_mode, s.mission_status = "none", "idle"
        s.handle_brain_status({"episode": s.episode, "mission": {"status": "idle"}})
        self.run(0.3)
        self.spoken.clear()

    def run(self, seconds: float) -> dict:
        """Step the sim; return what the outputs did over that time."""
        m, seen = self.sim.model, {"screen": False, "center": False, "red": False}
        n = int(seconds / self.dt)
        for i in range(n):
            self.sim.last_cmd_time = __import__("time").monotonic()  # no brain: keep the "cmd" fresh
            self.sim.step_once(self.dt)
            if i % 5 == 0:
                seen["screen"] |= float(m.mat_emission[self.screen_mat]) > 0.0
                seen["center"] |= float(m.mat_emission[self.center_mat]) > 0.5   # idle centre glow is 0.12
                seen["red"] |= float(m.mat_emission[self.red_mat]) > 0.5
        seen["label"] = bool(self.sim.screen_text)
        seen["handle_up"] = self.sim.locomotion.handle.swing > 0.5
        return seen


def main() -> int:
    p = Probe()
    s = p.sim
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(f"    {'ok  ' if cond else 'FAIL'} {what}")
        if not cond:
            failures.append(what)

    check(s.profile == "deafblind" and s.outputs == channels("deafblind"), "default profile is deafblind (all on)")
    s.handle_gui({"action": "profile", "mode": "nonsense"})
    check(s.profile == "deafblind", "unknown profile is ignored")

    for prof in PROFILE_NAMES:
        want = channels(prof)
        print(f"\n=== profile {prof}: expect {', '.join(c for c, v in want.items() if v)}")
        s.handle_gui({"action": "profile", "mode": prof})  # same path as the control GUI (`gui` topic)
        check(s.profile == prof, f"state.profile == {prof}")
        p.clear()

        # Doorbell (tier 3): only the profile's own channels.
        p.alert("doorbell", "1_pulse", DOORBELL_SAY)
        seen = p.run(1.6)
        check(s.outputs == want, f"doorbell: outputs == {want}")
        check(seen["screen"] == want["screen"], f"doorbell: screen lit = {want['screen']}")
        check(seen["label"] == want["screen"], f"doorbell: floating 'Nudge says' label = {want['screen']}")
        check(seen["center"] == want["light"], f"doorbell: light bar centre lit = {want['light']}")
        check(not seen["red"], "doorbell: no danger strobe")
        check((DOORBELL_SAY in p.spoken) == want["speech"], f"doorbell: spoken = {want['speech']}")
        check(seen["handle_up"], "doorbell: haptic handle up (always on)")
        check(s.nudge_says == DOORBELL_SAY, "doorbell: NUDGE SAYS kept (state.nudge_says)")

        # Smoke alarm (tier 1): emergency override -> everything on.
        p.alert("smoke_alarm", "continuous", SMOKE_SAY)
        seen = p.run(1.0)
        check(all(s.outputs.values()), "smoke: EMERGENCY -> all outputs on")
        check(seen["screen"] and seen["label"], "smoke: screen lit + label")
        check(seen["center"] and seen["red"], "smoke: light bar centre + danger strobe")
        check(SMOKE_SAY in p.spoken, "smoke: spoken")
        if not want["speech"]:
            check(DOORBELL_SAY not in p.spoken, "smoke: the stale doorbell message is NOT spoken")
        check(seen["handle_up"], "smoke: haptic handle up")

        p.clear()
        check(s.outputs == want, f"after the emergency: back to the {prof} outputs")

    # Switching to a speech profile mid-message speaks the current message once.
    print("\n=== deaf -> blind mid-message")
    s.handle_gui({"action": "profile", "mode": "deaf"})
    p.clear()
    p.alert("kettle", "2_pulse", "The kettle is boiling in the kitchen. Take my handle.")
    p.run(0.2)
    check(not p.spoken, "deaf: kettle message not spoken")
    s.handle_gui({"action": "profile", "mode": "blind"})
    p.run(0.2)
    check(p.spoken == ["The kettle is boiling in the kitchen. Take my handle."], "switch to blind: spoken once")
    seen = p.run(1.3)
    check(not seen["screen"] and not seen["center"], "blind: screen + light bar centre off")

    # [R] reset keeps the owner profile (owner setting, not scene state).
    s.reset_scene()
    check(s.profile == "blind", "reset keeps the profile")

    s.close()
    print(f"\n--> test_profiles: {'PASS' if not failures else 'FAIL'} ({len(failures)} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
