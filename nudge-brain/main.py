#!/usr/bin/env python3
"""
Nudge Brain orchestrator (`nudge-brain/main.py`) — run with `pixi run brain`.

20 Hz loop:
  sensors (ZMQ) -> hearing / vision -> world model
  owner requests (mic [T] -> Whisper, or typed via :5558) -> intent (LLM / keywords) -> mission
  -> safety pre-emption (smoke alarm) -> discrete events -> agent (LLM / rules, 2 s watchdog)
  -> skill executor + navigator -> safety speed filter -> `cmd` to the sim -> dashboard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import safety  # noqa: E402
from agent import NudgeAgent  # noqa: E402
from alerts import VOICE_REQUEST, HapticAlerts  # noqa: E402
from bridge import BrainBridge  # noqa: E402
from hearing import StereoHearing  # noqa: E402
from navigator import Navigator  # noqa: E402
from skills import Skill, SkillExecutor  # noqa: E402
from vision import OwnerDetector  # noqa: E402
from voice import VoiceInput  # noqa: E402
from world_model import WorldModel  # noqa: E402

LOOP_HZ = 20.0
ALERT_SOUNDS = ("doorbell", "knock")
NEW_SOUND_CONF = 0.5
SILENCE_WINDOWS_TO_UNLATCH = 3  # 1.5 s of silence before the same sound can start a new mission
UNLATCH_BELOW = 0.25
OWNER_SEEN_COOLDOWN = 3.0


class Brain:
    def __init__(self, show_dashboard: bool, voice: bool = True) -> None:
        self.bridge = BrainBridge()
        self.world = WorldModel()
        self.hearing = StereoHearing()
        self.vision = OwnerDetector()
        self.nav = Navigator()
        self.alerts = HapticAlerts()
        self.executor = SkillExecutor(self.nav, self.alerts)
        self.agent = NudgeAgent()
        self.voice = VoiceInput(enabled=voice)
        self.dashboard = None
        if show_dashboard:
            from dashboard import Dashboard

            self.dashboard = Dashboard()

        self.latched: dict = {}  # sound -> consecutive windows not heard
        self.last_owner_seen_event = 0.0
        self.episode = None  # sim episode we are acting on; a change means the sim was reset ([R])
        self.world.log(f"Brain up | hearing={self.hearing.backend_name} vision={self.vision.backend} "
                       f"agent={self.agent.backend} voice={self.voice.backend}")

    def reset(self, episode) -> None:
        """Sim was reset: drop the mission, plan, alerts, owner belief and pending agent decision."""
        self.agent.cancel()
        self.agent.cancel_request()
        self.nav.clear()
        self.alerts.clear()
        self.executor.skill = Skill()
        self.world = WorldModel()
        self.latched = {}
        self.last_owner_seen_event = 0.0
        self.episode = episode
        self.world.log(f"RESET by sim (episode {episode}) - ready")

    # ------------------------------------------------------------------ perception
    def on_audio(self) -> None:
        meta, stereo = self.bridge.latest_audio
        res = self.hearing.process_audio(stereo, self.world.robot_pose)
        self.world.update_sound(res)
        scores = res.get("scores", {})

        # Release latches only once a sound is truly silenced (not just quieter because we moved away),
        # so a still-ringing alarm can't restart a mission that was just completed.
        for s in list(self.latched):
            heard = scores.get(s, 0.0) >= UNLATCH_BELOW
            self.latched[s] = 0 if heard else self.latched[s] + 1
            if self.latched[s] >= SILENCE_WINDOWS_TO_UNLATCH:
                del self.latched[s]

        # 1. Safety pre-emption: smoke alarm overrides everything.
        if safety.smoke_detected(res):
            if not self.world.is_emergency() and "smoke_alarm" not in self.latched:
                self.latched["smoke_alarm"] = 0
                self.world.log(f"SAFETY PRE-EMPT: smoke_alarm {scores['smoke_alarm']:.2f} >= 0.60")
                self.agent.cancel()
                self.world.start_mission("smoke_alarm")
                self.alerts.set("continuous")
                self.agent.submit("new_sound", self.world)
            return

        # 2. Doorbell / knock start (or update) a mission.
        label, conf = res.get("label"), float(res.get("confidence", 0.0))
        if label in ALERT_SOUNDS and conf >= NEW_SOUND_CONF and label not in self.latched:
            self.latched[label] = 0
            if self.world.is_emergency():
                return
            if self.world.mission is None or self.world.is_request():
                if self.world.is_request():  # a doorbell matters more than the owner's own errand
                    self.world.log(f"{label} interrupts the owner's request")
                self.world.log(f"Heard {label} ({conf:.2f}) az {res.get('world_azimuth_deg'):+.0f} deg")
                self.world.start_mission(label)
                self.agent.submit("new_sound", self.world)
            elif self.world.mission["sound"] != label:
                self.world.log(f"Mission sound updated: {label}")
                self.world.mission["sound"] = label
                if self.alerts.mode != "none":
                    self.alerts.set_for_sound(label)

    def on_camera(self) -> None:
        meta, rgb, depth = self.bridge.latest_camera
        pose = meta.get("robot_pose", self.world.robot_pose)
        det = self.vision.detect(rgb, depth, (float(pose[0]), float(pose[1]), float(pose[2])))
        self.world.update_owner_detection(det)
        now = time.monotonic()
        if (det["found"] and self.world.mission is not None and self.world.phase == "searching"
                and now - self.last_owner_seen_event > OWNER_SEEN_COOLDOWN):
            self.last_owner_seen_event = now
            self.world.log(f"Owner seen in {self.world.owner_room} at {det['world_xy']} ({det['backend']})")
            self.agent.submit("owner_seen", self.world)

    # ------------------------------------------------------------------ owner requests
    def on_owner_input(self) -> None:
        """Collect requests (mic / voice file / chat GUI) and start understanding them."""
        incoming = list(self.bridge.requests)
        self.bridge.requests.clear()
        heard = self.voice.poll()
        if heard is not None:
            if heard.get("warning"):
                self.world.log(f"VOICE warning: {heard['warning']}")
            if heard.get("text"):
                incoming.append(heard)
            else:
                self.world.log(f"VOICE: {heard.get('error', 'nothing heard')}")
        for req in incoming:
            extra = f", stt {heard['stt_s']:.2f}s" if req is heard and "stt_s" in req else ""
            self.world.log(f'OWNER SAID ({req["channel"]}{extra}): "{req["text"]}"')
            self.agent.parse_request(req["text"])

        parsed = self.agent.poll_request()
        if parsed is not None:
            self.apply_request(parsed)

    def stop_mission(self, why: str) -> None:
        self.agent.cancel()
        self.nav.clear()
        self.executor.skill = Skill()
        self.alerts.clear()
        self.world.cancel_mission(why)

    def apply_request(self, r: dict) -> None:
        intent, target = r["intent"], r.get("target")
        self.world.log(f"REQUEST -> {intent}" + (f"({target})" if target else "")
                       + (f"({r['object']})" if r.get("object") else "") + f" [{r['source']} {r['latency']:.2f}s]")
        if self.world.is_emergency():  # the smoke-alarm evacuation can't be cancelled or redirected
            self.world.reply("Fire alarm. Follow me out first.")
            return
        if intent == "stop":
            if self.world.mission is not None:
                self.stop_mission("owner said stop")
            self.world.reply(r["say"])
            return
        if intent in ("come_here", "go_to"):
            if self.world.mission is not None:
                self.stop_mission("replaced by a new owner request")
            self.world.start_mission(VOICE_REQUEST, target=target, request=r["text"])
            self.world.reply(r["say"])
            self.agent.submit("owner_request", self.world)
            return
        self.world.reply(r["say"])  # find_object (not a capability yet) / unknown

    # ------------------------------------------------------------------ loop
    def tick(self) -> bool:
        upd = self.bridge.poll()
        if upd["state"]:
            episode = self.bridge.latest_state.get("episode")
            if self.episode is None:
                self.episode = episode
            elif episode != self.episode:
                self.reset(episode)
            self.world.update_state(self.bridge.latest_state)
        if upd["audio"]:
            self.on_audio()
        if upd["camera"]:
            self.on_camera()
        self.on_owner_input()

        # LiDAR doorway check: re-plan around a blocked doorway (e.g. the chair).
        door = self.nav.check_doorways(self.world.robot_pose, self.world.lidar_ranges,
                                       ignore=self.world.owner_pos if self.world.owner_recently_seen(3) else None)
        if door:
            self.world.blocked_doors[door] = time.monotonic()
            self.world.log(f"LiDAR: {door} BLOCKED -> re-planned route")

        out = self.executor.step(self.world)
        if out.event:
            if out.event == "owner_alerted" and self.world.is_request() and not self.world.mission["target"]:
                self.world.log("Robot is with the owner")  # "come here" is done once Nudge has nudged them
                self.world.finish_mission()
                self.alerts.clear()
            elif out.event == "guide_complete":
                if self.world.mission:
                    self.world.log(f"Owner delivered to {self.world.mission['target']}")
                self.world.finish_mission()
                self.alerts.clear()
            elif self.world.mission is not None:
                self.agent.submit(out.event, self.world)

        action = self.agent.poll(self.world)
        if action is not None:
            self.executor.execute(action, self.world)

        vx, vy, wz = safety.filter_cmd(out.vx, out.vy, out.wz, self.world, self.world.lidar_ranges,
                                       allow_contact=out.allow_contact)
        if not self.world.connected or time.monotonic() - self.world.last_state_wall > 1.0:
            vx = vy = wz = 0.0
        self.world.nav_path = self.nav.path_points(self.world.robot_pose[:2]) if self.nav.waypoints else []
        status, status_text = self.world.mission_status()
        self.bridge.send_cmd(vx, vy, wz, alert_mode=self.alerts.mode, guiding=out.guiding,
                             mission_status=status, mission_text=status_text, episode=self.episode)

        if self.dashboard is not None:
            key = self.dashboard.render(self.world, self.agent, self.bridge.latest_camera,
                                        self.executor.skill.label(), self.alerts, self.world.nav_path,
                                        voice=self.voice)
            if key == "quit":
                return False
            if key == "talk":
                if self.voice.listen():
                    self.world.log("VOICE: listening... (speak now)")
                else:
                    self.world.log(f"VOICE: can't listen ({self.voice.backend}, {self.voice.state})")
            elif key:
                self.bridge.send_sim_event(key)
        return True

    def run(self, duration: float = 0.0) -> None:
        period = 1.0 / LOOP_HZ
        start = time.monotonic()
        try:
            while True:
                t0 = time.monotonic()
                if not self.tick():
                    break
                if duration and t0 - start > duration:
                    break
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            self.bridge.send_cmd(0.0, 0.0, 0.0)
            self.bridge.close()
            self.voice.close()
            if self.dashboard is not None:
                self.dashboard.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Nudge brain: perception, navigation, agent, safety, dashboard")
    ap.add_argument("--no-dashboard", action="store_true", help="Run without the OpenCV window")
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = forever)")
    ap.add_argument("--no-voice", action="store_true", help="Don't load the microphone / speech-to-text")
    args = ap.parse_args()
    Brain(show_dashboard=not args.no_dashboard, voice=not args.no_voice).run(args.duration)


if __name__ == "__main__":
    main()
