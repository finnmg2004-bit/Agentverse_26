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
from evacuation import ESCALATE_AFTER_S, EvacuationPlanner  # noqa: E402
from sound_localizer import SoundLocalizer  # noqa: E402
from rules import MIN_VOICE_CONFIDENCE, CommandListener, CommandParser, validate  # noqa: E402

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
        self.localizer = SoundLocalizer()
        self.evac = EvacuationPlanner(self.nav.planner)
        # What the owner tells the robot (GUI text box / voice): rules.py
        self.cmd_listener = CommandListener()
        self.cmd_parser = CommandParser(self.agent)
        # The owner's no-go areas; read self.world at call time (reset() replaces the world).
        self.nav.avoid_provider = lambda: self.world.avoid_for_planning()
        self.last_tick = time.monotonic()
        self.dashboard = None
        if show_dashboard:
            from dashboard import Dashboard

            self.dashboard = Dashboard()

        self.latched: dict = {}  # sound -> consecutive windows not heard
        self.last_owner_seen_event = 0.0
        self.episode = None  # sim episode we are acting on; a change means the sim was reset ([R])
        self.last_status_pub = 0.0
        self.last_heard = None  # last thing the owner told the robot (voice or GUI), for the GUI echo
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
        self.localizer.reset()
        self.latched = {}
        self.last_owner_seen_event = 0.0
        self.episode = episode
        self.world.log(f"RESET by sim (episode {episode}) - ready")

    # ------------------------------------------------------------------ perception
    def on_audio(self) -> None:
        meta, stereo = self.bridge.latest_audio
        res = self.hearing.process_audio(stereo, self.world.robot_pose)
        scores = res.get("scores", {})

        # Localise the sound from the ears (direction + loudness, accumulated as the robot moves).
        changed = self.localizer.update(res, self.world.robot_pose)
        if changed:
            est = self.localizer.estimate(changed)
            self.world.sound_estimates[changed] = est
            if changed == "smoke_alarm" and self.world.is_emergency() and self.world.phase != "guiding":
                self.plan_evacuation("fire location refined by hearing")
        est = self.world.sound_estimates.get(res.get("label"))
        res["estimated_room"] = est["room"] if est and est["converged"] else None
        self.world.update_sound(res)

        # Release latches only once a sound is truly silenced (not just quieter because we moved away),
        # so a still-ringing alarm can't restart a mission that was just completed.
        for s in list(self.latched):
            heard = scores.get(s, 0.0) >= UNLATCH_BELOW
            self.latched[s] = 0 if heard else self.latched[s] + 1
            if self.latched[s] >= SILENCE_WINDOWS_TO_UNLATCH:
                del self.latched[s]
                self.localizer.reset(s)
                self.world.sound_estimates.pop(s, None)

        # 1. Safety pre-emption: smoke alarm overrides everything.
        if safety.smoke_detected(res):
            if not self.world.is_emergency() and "smoke_alarm" not in self.latched:
                self.latched["smoke_alarm"] = 0
                self.world.log(f"SAFETY PRE-EMPT: smoke_alarm {scores['smoke_alarm']:.2f} >= 0.60")
                self.agent.cancel()
                self.world.start_mission("smoke_alarm")
                self.alerts.set("continuous")
                self.plan_evacuation("smoke alarm heard")
                self.agent.submit("new_sound", self.world)
            return

        # 2. Doorbell / knock start (or update) a mission.
        label, conf = res.get("label"), float(res.get("confidence", 0.0))
        if label in ALERT_SOUNDS and conf >= NEW_SOUND_CONF and label not in self.latched:
            self.latched[label] = 0
            if self.world.is_emergency():
                return
            if self.world.mission is None:
                pref = self.world.rules.sound_rule(label)  # never applies to danger sounds
                if pref is not None and pref["policy"] in ("ignore", "quiet_hours"):
                    self.world.log(f"Heard {label} - not alerting: owner's rule '{pref['summary']}'")
                    return
                self.world.log(f"Heard {label} ({conf:.2f}) az {res.get('world_azimuth_deg'):+.0f} deg")
                self.world.start_mission(label)
                if pref is not None and pref["policy"] == "expect":
                    name = label.replace("_", " ").capitalize()
                    self.world.mission["say"] = (f"{name}. Maybe your parcel. Hold my handle. "
                                                 f"I will take you to the front door.")
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

    # ------------------------------------------------------------------ owner commands
    def on_user_command(self, msg: dict) -> None:
        """A sentence from the owner (GUI text box or voice). Parsed in the background by Gemma."""
        text = str(msg.get("text", "")).strip()
        source = str(msg.get("source", "gui"))
        if not text:
            return
        conf = float(msg.get("confidence", 1.0) or 0.0)
        # echo for the control GUI's "Heard:" line (also when the robot was too unsure to act on it)
        self.last_heard = {"text": text, "source": source, "confidence": conf, "t": round(self.world.t_sim, 1)}
        if source == "voice" and conf < MIN_VOICE_CONFIDENCE:
            self.world.log(f'OWNER SAID ({source}, {conf:.2f}, too unsure): "{text}"')
            self.world.say("Sorry, I did not catch that. Please say it again.", "confirm")
            return
        self.world.log(f'OWNER SAID ({source}): "{text}"')
        self.cmd_parser.submit(dict(msg, text=text, source=source))

    def apply_command(self, result: dict) -> None:
        """Validate a parsed instruction in plain code, enforce it, and confirm it to the owner."""
        w = self.world
        msg, parsed = result["msg"], result["parsed"]
        instr, say = validate(parsed, w.is_emergency(), str(msg.get("text", "")))
        w.log(f"COMMAND parsed by {result['source']} in {result['latency']:.2f}s: {parsed.get('kind')}"
              + (f" -> {instr['type']}" if instr else " -> refused / not understood"))
        if instr is not None:
            kind = instr["type"]
            if kind in ("avoid_area", "sound_preference"):
                rule = w.rules.add(dict(instr, text=msg.get("text"), source=msg.get("source")))
                w.log(f"RULE #{rule['id']}: {rule['summary']}")
                if kind == "avoid_area" and self.nav.goal is not None:
                    self.nav._plan(w.robot_pose[:2])  # the new rule applies to the current route at once
            elif kind == "allow_area":
                if not w.rules.remove_area(instr["area"]):
                    say = "OK. That place was not blocked."
                elif self.nav.goal is not None:
                    self.nav._plan(w.robot_pose[:2])
            elif kind == "go_to":
                self.agent.cancel()
                w.start_request(instr["target"], str(msg.get("text", "")))
                w.mission["say"] = say
                self.agent.submit("user_command", w)
            elif kind == "stop":
                self.agent.cancel()
                self.nav.clear()
                self.executor.skill = Skill()
                self.alerts.clear()
                if w.mission is not None:
                    w.log(f"Mission {w.mission['sound']} CANCELLED by the owner")
                    w.mission = None
                    w.phase = "idle"
        w.say(say, "confirm")

    # ------------------------------------------------------------------ emergency
    def plan_evacuation(self, why: str) -> bool:
        """Choose the safest exit for the owner given the fire's location. Returns True if it changed."""
        w = self.world
        fire = self.evac.fire_estimate(self.localizer)
        start = w.owner_pos if (w.owner_pos is not None and w.owner_recently_seen(30.0)) else w.robot_pose[:2]
        decision = self.evac.choose(start, fire, set(w.blocked_doors), avoid=w.rules.avoid_rects())
        old = w.evac_target() if w.evac else None
        w.fire, w.evac = fire, decision
        if w.mission is not None and w.is_emergency():
            w.mission["target"] = decision["target"]
        if decision["target"] != old:
            via = " ACROSS the owner's no-go area (much shorter)" if decision.get("via") == "across" else ""
            keep = " keeping off " + ", ".join(w.rules.avoid_areas()) if w.rules.avoid_areas() and not via else ""
            w.log(f"EVAC ({why}): fire in {fire['room']} ({fire['source']}) -> exit {decision['target']}{via}{keep}, "
                  f"avoiding {len(decision['hazard_doors'])} doorways")
            return True
        return False

    def check_escalation(self) -> None:
        w = self.world
        if (w.is_emergency() and not w.escalated and w.phase == "searching"
                and time.monotonic() - w.mission["started"] > ESCALATE_AFTER_S):
            w.escalated = True
            w.log(f"ESCALATION: owner not found {ESCALATE_AFTER_S:.0f} s into a fire -> "
                  "emergency services called (simulated)")

    # ------------------------------------------------------------------ loop
    def tick(self) -> bool:
        now = time.monotonic()
        self.world.belief.predict(now - self.last_tick)  # the owner may have walked off
        self.last_tick = now
        if not self.world.owner_recently_seen(2.0):
            self.world.owner_room = self.world.belief.most_likely()
        self.check_escalation()
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
        for msg in self.cmd_listener.poll():
            self.on_user_command(msg)
        for result in self.cmd_parser.poll():
            self.apply_command(result)

        # LiDAR doorway check: re-plan around a blocked doorway (e.g. the chair).
        door = self.nav.check_doorways(self.world.robot_pose, self.world.lidar_ranges,
                                       ignore=self.world.owner_pos if self.world.owner_recently_seen(3) else None)
        if door:
            self.world.blocked_doors[door] = time.monotonic()
            self.world.log(f"LiDAR: {door} BLOCKED -> re-planned route")
            # Mid-evacuation: a blocked door or exit changes the options -> the agent re-chooses the
            # exit (the navigator already re-routed, so the robot keeps moving meanwhile).
            if self.world.is_emergency():
                self.plan_evacuation(f"{door} blocked")
                if self.world.phase == "guiding":
                    self.agent.submit("path_blocked", self.world)

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
            if action.get("escalate") and not self.world.escalated:
                self.world.escalated = True
                self.world.log("AGENT CALLED EMERGENCY SERVICES (simulated)")
            if self.world.is_emergency() and action.get("tool") == "guide_owner_to" and self.world.evac:
                # The agent may pick a different safe exit than the planner's shortest one.
                self.world.evac["target"] = action["args"]["target"]
                self.world.mission["target"] = action["args"]["target"]
            self.executor.execute(action, self.world)

        vx, vy, wz = safety.filter_cmd(out.vx, out.vy, out.wz, self.world, self.world.lidar_ranges,
                                       allow_contact=out.allow_contact)
        if not self.world.connected or time.monotonic() - self.world.last_state_wall > 1.0:
            vx = vy = wz = 0.0
        self.world.nav_path = self.nav.path_points(self.world.robot_pose[:2]) if self.nav.waypoints else []
        status, status_text = self.world.mission_status()
        if self.world.escalated and status == "active":
            status_text += "  |  EMERGENCY SERVICES CALLED"
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
                        "say": (w.mission.get("say") or DEFAULT_SAY.get(w.mission["sound"], "")
                                or (w.nudge_says or {}).get("text", ""))
                        if w.mission else ""},
            # What the owner told the robot, as enforced (rules.py), and the latest message for the owner
            "rules": w.rules.status(),
            "nudge_says": w.nudge_says,
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
                      "decisions": list(self.agent.decisions), "llm": self.agent.llm_stats(),
                      "menu": self.agent.last_menu},
            "events": list(w.events),
            "backends": {"hearing": self.hearing.backend_name, "vision": self.vision.backend},
            # last sentence the owner said/typed, as received (control GUI "Heard:" line)
            "heard": self.last_heard,
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
            self.cmd_listener.close()
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
