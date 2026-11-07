#!/usr/bin/env python3
"""
Nudge Brain orchestrator (`nudge-brain/main.py`) — run with `pixi run brain`.

20 Hz loop:
  sensors (ZMQ) -> hearing / vision -> world model
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
from agent import DEFAULT_SAY, NudgeAgent  # noqa: E402
from alerts import HapticAlerts  # noqa: E402
from bridge import BrainBridge  # noqa: E402
from hearing import StereoHearing  # noqa: E402
from navigator import Navigator  # noqa: E402
from skills import Skill, SkillExecutor  # noqa: E402
from vision import OwnerDetector  # noqa: E402
from world_model import WorldModel  # noqa: E402

LOOP_HZ = 20.0
STATUS_HZ = 10.0  # `brain` status for the control GUI (gui/nudge_control.py)
ALERT_SOUNDS = ("doorbell", "knock")
NEW_SOUND_CONF = 0.5
SILENCE_WINDOWS_TO_UNLATCH = 3  # 1.5 s of silence before the same sound can start a new mission
UNLATCH_BELOW = 0.25
OWNER_SEEN_COOLDOWN = 3.0


class Brain:
    def __init__(self, show_dashboard: bool) -> None:
        self.bridge = BrainBridge()
        self.world = WorldModel()
        self.hearing = StereoHearing()
        self.vision = OwnerDetector()
        self.nav = Navigator()
        self.alerts = HapticAlerts()
        self.executor = SkillExecutor(self.nav, self.alerts)
        self.agent = NudgeAgent()
        self.dashboard = None
        if show_dashboard:
            from dashboard import Dashboard

            self.dashboard = Dashboard()

        self.latched: dict = {}  # sound -> consecutive windows not heard
        self.last_owner_seen_event = 0.0
        self.episode = None  # sim episode we are acting on; a change means the sim was reset ([R])
        self.last_status_pub = 0.0
        self.world.log(f"Brain up | hearing={self.hearing.backend_name} vision={self.vision.backend} "
                       f"agent={self.agent.backend}")

    def reset(self, episode) -> None:
        """Sim was reset: drop the mission, plan, alerts, owner belief and pending agent decision."""
        self.agent.cancel()
        self.agent.decisions.clear()
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
            if self.world.mission is None:
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

        # LiDAR doorway check: re-plan around a blocked doorway (e.g. the chair).
        door = self.nav.check_doorways(self.world.robot_pose, self.world.lidar_ranges,
                                       ignore=self.world.owner_pos if self.world.owner_recently_seen(3) else None)
        if door:
            self.world.blocked_doors[door] = time.monotonic()
            self.world.log(f"LiDAR: {door} BLOCKED -> re-planned route")

        out = self.executor.step(self.world)
        if out.event:
            if out.event == "guide_complete":
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
        now = time.monotonic()
        if now - self.last_status_pub >= 1.0 / STATUS_HZ:
            self.last_status_pub = now
            self.bridge.send_status(self.status(status, status_text))

        if self.dashboard is not None:
            key = self.dashboard.render(self.world, self.agent, self.bridge.latest_camera,
                                        self.executor.skill.label(), self.alerts, self.world.nav_path)
            if key == "quit":
                return False
            if key:
                self.bridge.send_sim_event(key)
        return True

    def status(self, mission_status: str, mission_text: str) -> dict:
        """Everything the control GUI shows about the brain (published on topic `brain`)."""
        w = self.world
        snd = w.sound or {}
        det = w.last_detection or {}
        return {
            "t": round(w.t_sim, 2),
            "episode": self.episode,
            "mission": {"status": mission_status, "text": mission_text,
                        "sound": w.mission["sound"] if w.mission else None,
                        "emergency": w.is_emergency(),
                        # what Nudge tells the owner (Gemma's message, else a default per sound)
                        "say": (w.mission.get("say") or DEFAULT_SAY.get(w.mission["sound"], ""))
                        if w.mission else ""},
            "phase": w.phase,
            "skill": self.executor.skill.label(),
            "alert_mode": self.alerts.mode,
            "sound": {k: snd.get(k) for k in ("label", "confidence", "scores", "world_azimuth_deg",
                                              "estimated_room", "backend")},
            "detection": {k: det.get(k) for k in ("found", "bbox", "confidence", "distance", "backend")},
            "owner": {"pos": w.owner_pos, "room": w.owner_room, "source": w.owner_source,
                      "seen_recently": w.owner_recently_seen(2.0)},
            "nav_path": [[round(x, 2), round(y, 2)] for x, y in (w.nav_path or [])],
            "searched_rooms": sorted(w.searched_rooms),
            "blocked_doors": sorted(w.blocked_doors),
            "agent": {"backend": self.agent.backend, "pending": self.agent.pending_info(),
                      "decisions": list(self.agent.decisions), "llm": self.agent.stats()},
            "events": list(w.events),
            "backends": {"hearing": self.hearing.backend_name, "vision": self.vision.backend},
        }

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
            if self.dashboard is not None:
                self.dashboard.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Nudge brain: perception, navigation, agent, safety, dashboard")
    ap.add_argument("--no-dashboard", action="store_true", help="Run without the OpenCV window")
    ap.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = forever)")
    args = ap.parse_args()
    Brain(show_dashboard=not args.no_dashboard).run(args.duration)


if __name__ == "__main__":
    main()
