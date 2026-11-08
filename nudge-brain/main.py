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
import os
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
from hearing import SoundEvents, StereoHearing  # noqa: E402
from navigator import Navigator  # noqa: E402
from skills import GoToRoom, Skill, SkillExecutor  # noqa: E402
from vision import OwnerDetector  # noqa: E402
from world_model import WorldModel  # noqa: E402
from evacuation import ESCALATE_AFTER_S, EvacuationPlanner  # noqa: E402
from sound_localizer import SoundLocalizer  # noqa: E402
from rules import MIN_VOICE_CONFIDENCE, CommandListener, CommandParser, forget_reply, validate  # noqa: E402
from tasks import OwnerActivity, TaskRunner, TaskStore, confirmation  # noqa: E402
from ask import AnswerListener, AskOwner  # noqa: E402
from sim.house_layout import ALERT_SOUNDS, SOUND_SOURCES  # noqa: E402  (non-danger sounds, from the registry)

LOOP_HZ = 20.0
STATUS_HZ = 10.0  # `brain` status for the control GUI (gui/nudge_control.py)
NEW_SOUND_CONF = 0.5
SILENCE_WINDOWS_TO_UNLATCH = 3  # 1.5 s of silence before the same sound can start a new mission
UNLATCH_BELOW = 0.25
OWNER_SEEN_COOLDOWN = 3.0
# E1.5: after teaching, ask "... OK?" (C5 `ask`); yes keeps the task, no drops it, no answer keeps it
ASK_CONFIRM = os.environ.get("NUDGE_ASK_CONFIRM", "0") == "1"
SAY_HOLD_S = 20.0  # top-level `say` in the brain status: how long a reply stays on the robot screen


class TaskAdapter:
    """Runs taught-task jobs (tasks.TaskRunner, contract C1) through the brain's mission machinery.

    "mission" (find owner -> nudge -> guide to X) is an owner-request mission, exactly like
    "take me to X". "find_alert" (find owner -> nudge -> tell them, no guiding) is the same mission
    ended by plain code once the owner is alerted. "say" just speaks; "go_to_room" walks there.
    A task mission is never started over another mission (busy) and a smoke alarm replaces it
    (start_mission) - the runner then fails the task.
    """

    def __init__(self, brain: "Brain") -> None:
        self.brain = brain
        self.mission = None  # the mission dict this adapter started (identity check)
        self.kind = None
        self.result = None  # "ok" / "failed" once known

    def reset(self) -> None:
        self.mission, self.kind, self.result = None, None, None

    def busy(self) -> bool:
        w = self.brain.world
        return w.mission is not None and w.mission is not self.mission

    def emergency(self) -> bool:
        return self.brain.world.is_emergency()

    def start_job(self, job: dict, task: dict) -> None:
        b, w = self.brain, self.brain.world
        self.mission, self.kind, self.result = None, job["kind"], None
        say = job.get("say") or job.get("text") or ""
        if job["kind"] == "find_alert" and (w.guiding_active or w.dist_to_owner() < 1.6):
            # already with the owner: just tell them
            w.say(say or f"Reminder: {task['name']}.", "task")
            self.result = "ok"
        elif job["kind"] in ("mission", "find_alert"):
            b.agent.cancel()
            target = job.get("target") or w.best_search_room()
            w.start_request(target, f"task {task['name']}")
            w.mission.update(task=task["name"], task_kind=job["kind"])
            if say:
                w.mission["say"] = say
            self.mission = w.mission
            b.agent.submit("user_command", w)
        elif job["kind"] == "say":
            w.say(say, "task")
            self.result = "ok"
        elif job["kind"] == "go_to_room":
            b.executor.execute({"tool": "go_to_room", "args": {"room": job["room"]}}, w)
        else:
            self.result = "failed"

    def job_status(self, job: dict) -> str:
        if self.result is not None:
            return self.result
        if self.kind == "go_to_room":
            return "running" if isinstance(self.brain.executor.skill, GoToRoom) else "ok"
        w = self.brain.world
        return "running" if (self.mission is not None and w.mission is self.mission) else "failed"

    def finished(self, mission) -> None:
        """The brain completed `mission` (owner delivered / alerted): the job is done."""
        if mission is not None and mission is self.mission:
            self.result = "ok"

    def cancel_job(self, job: dict) -> None:
        b, w = self.brain, self.brain.world
        if self.mission is not None and w.mission is self.mission:  # never touches a smoke mission
            b.agent.cancel()
            b.nav.clear()
            b.executor.skill = Skill()
            b.alerts.clear()
            w.mission = None
            w.phase = "idle"
        elif self.kind == "go_to_room" and isinstance(b.executor.skill, GoToRoom):
            b.nav.clear()
            b.executor.skill = Skill()
        self.mission, self.result = None, "failed"


class Brain:
    def __init__(self, show_dashboard: bool) -> None:
        self.bridge = BrainBridge()
        self.world = WorldModel()
        self.hearing = StereoHearing()
        self.sound_events = SoundEvents()  # debounced start/stop + "what did I hear" log (Ricktho, 10.5a)
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
        # Taught jobs (tasks.py): assets/tasks.json, emptied at brain start unless NUDGE_TASKS_KEEP=1
        # (so a task taught in one test run never leaks into the next).
        self.task_store = TaskStore()
        if os.environ.get("NUDGE_TASKS_KEEP", "0") != "1" and len(self.task_store):
            self.task_store.clear()
        self.task_adapter = TaskAdapter(self)
        self.ask = AskOwner()  # one open question to the owner (C5), answered by voice / GUI
        self.answer_listener = AnswerListener() if ASK_CONFIRM else None
        self.task_runner = TaskRunner(self.task_store, self.task_adapter,
                                      log=lambda text: self.world.log(text))
        self.new_sound_events: list = []  # SoundEvents since the last task tick (sound triggers)
        self.owner_activity = OwnerActivity()  # idle / owner_entered_room triggers (E1.6)
        self.owner_spoke = False  # a sentence from the owner since the last task tick (= activity)
        self.seen_wall = self.tap_wall = 0.0  # last owner sighting / handle tap fed to owner_activity
        # The owner's no-go areas; read self.world at call time (reset() replaces the world).
        self.nav.avoid_provider = lambda: self.world.avoid_for_planning()
        self.last_tick = time.monotonic()
        self.dashboard = None
        if show_dashboard:
            from dashboard import Dashboard

            self.dashboard = Dashboard()

        self.latched: dict = {}  # sound -> consecutive windows not heard
        self.deferred: set = set()  # sounds heard during a mission for another place (alert after it)
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
        self.deferred = set()
        self.sound_events.reset()
        self.new_sound_events = []
        self.task_runner.reset()
        self.task_adapter.reset()
        self.owner_activity.reset()
        self.owner_spoke = False
        self.ask.clear()
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
        for ev in self.sound_events.update(res, self.world.t_sim):
            self.new_sound_events.append(ev)
            extra = f" after {ev['duration_s']:.1f}s" if ev["phase"] == "stop" else f" ({ev['confidence']:.2f})"
            self.world.log(f"SOUND {ev['phase']}: {ev['label']}{extra}")

        # Release latches only once a sound is truly silenced (not just quieter because we moved away),
        # so a still-ringing alarm can't restart a mission that was just completed.
        for s in list(self.latched):
            heard = scores.get(s, 0.0) >= UNLATCH_BELOW
            self.latched[s] = 0 if heard else self.latched[s] + 1
            if self.latched[s] >= SILENCE_WINDOWS_TO_UNLATCH:
                del self.latched[s]
                self.deferred.discard(s)
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

        # 2. Doorbell / knock / kettle (non-danger registry sounds) start (or update) a mission.
        #    A sound that needs the owner somewhere else than the running mission is deferred and
        #    starts its own mission when the current one ends (if it is still sounding).
        label, conf = res.get("label"), float(res.get("confidence", 0.0))
        deferred_ready = label in self.deferred and self.world.mission is None
        if label in ALERT_SOUNDS and conf >= NEW_SOUND_CONF and (label not in self.latched or deferred_ready):
            self.latched[label] = 0
            self.deferred.discard(label)
            if self.world.is_emergency():
                return
            task_name = self.task_runner.sound_task(label)
            if task_name is not None:
                # E1.4: the owner taught their own response to this sound (e.g. "when the kettle boils,
                # just tell me"); the task's sound trigger runs it (queued if a mission is active).
                self.world.log(f"Heard {label} ({conf:.2f}) - owner's task '{task_name}' replaces the "
                               f"built-in alert")
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
            elif (self.world.mission["sound"] != label
                  and SOUND_SOURCES[label]["target_landmark"] != self.world.mission["target"]):
                self.deferred.add(label)
                self.world.log(f"Heard {label} during the {self.world.mission['sound']} mission: "
                               f"will alert after it")
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
        self.owner_spoke = True  # they spoke / typed: awake and OK (idle trigger, E1.6)
        # echo for the control GUI's "Heard:" line (also when the robot was too unsure to act on it)
        self.last_heard = {"text": text, "source": source, "confidence": conf, "t": round(self.world.t_sim, 1)}
        if source == "voice" and conf < MIN_VOICE_CONFIDENCE:
            self.world.log(f'OWNER SAID ({source}, {conf:.2f}, too unsure): "{text}"')
            self.world.say("Sorry, I did not catch that. Please say it again.", "confirm")
            return
        if self.ask.answer(msg, plain_command=True):  # "yes" / "no" to the open question
            self.world.log(f'OWNER ANSWERED ({source}): "{text}"')
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
            elif kind == "teach":
                task = self.task_runner.store.add(dict(instr["task"], source=msg.get("source", "")))
                w.log(f"TASK taught: {task['name']} {task['trigger']} -> {[st['skill'] for st in task['steps']]}")
                if ASK_CONFIRM and not w.is_emergency():
                    say = self.ask_to_keep(task)
            elif kind == "forget_task":
                removed = self.task_runner.remove(instr["name"])
                say = forget_reply(removed, instr["name"])
                if removed is not None:
                    w.log(f"TASK forgotten: {removed['name']}")
            elif kind == "stop":
                stopped = self.task_runner.stop(w.t_sim)
                if stopped:
                    w.log(f"Task {stopped} STOPPED by the owner")
                self.agent.cancel()
                self.nav.clear()
                self.executor.skill = Skill()
                self.alerts.clear()
                if w.mission is not None:
                    w.log(f"Mission {w.mission['sound']} CANCELLED by the owner")
                    w.mission = None
                    w.phase = "idle"
        w.say(say, "confirm")

    # ------------------------------------------------------------------ ask the owner (C5)
    def ask_to_keep(self, task: dict) -> str:
        """Taught task: ask "Every day at 10 pm I'll take you to bed. OK?". The task is stored already,
        so it also works if nobody answers; "no" forgets it again. Returns the question."""
        summary = confirmation(task)[4:]  # without the leading "OK. "
        name = task["name"]

        def on_answer(choice):
            w = self.world
            if choice == "no":
                removed = self.task_runner.remove(name)
                w.log(f"ASK answered no -> task {name} " + ("dropped" if removed else "was already gone"))
                w.say("OK. I won't do that.", "confirm")
            elif choice == "yes":
                w.log(f"ASK answered yes -> task {name} kept")
                w.say(f"OK. {summary}", "confirm")
            else:
                w.log(f"ASK not answered -> task {name} kept")
                w.say(f"No answer, so I kept it. {summary}", "confirm")

        question = f"{summary} OK?"
        qid = self.ask.ask(question, self.world.t_sim, on_answer)
        self.world.log(f'ASK {qid}: "{question}"')
        return question

    def tick_ask(self) -> None:
        if self.answer_listener is not None:
            for msg in self.answer_listener.poll():
                if self.ask.answer(msg):
                    self.world.log(f'OWNER ANSWERED ({msg.get("source", "?")}): "{msg.get("text", "")}"')
        if self.ask.current is not None and self.world.is_emergency():
            self.ask.clear()  # a fire: no questions now (the task stays, as if unanswered)
        self.ask.tick(self.world.t_sim)

    # ------------------------------------------------------------------ taught tasks
    def tick_tasks(self) -> None:
        """Trigger watcher + runner (tasks.py). Runs a task only when no other mission is active."""
        st = self.bridge.latest_state or {}
        events, self.new_sound_events = self.new_sound_events, []
        # idle / owner_entered_room: from brain-side evidence only (sightings, handle, speech)
        w = self.world
        seen_xy = seen_room = None
        if w.owner_pos is not None and w.owner_last_seen_wall > self.seen_wall:
            self.seen_wall = w.owner_last_seen_wall
            seen_xy, seen_room = w.owner_pos, w.owner_room
        active = self.owner_spoke or w.guiding_active or w.last_tap_wall > self.tap_wall
        self.owner_spoke, self.tap_wall = False, w.last_tap_wall
        now = OwnerActivity.now_s(st.get("sim_clock"), st.get("sim_day"), w.t_sim)
        idle_s, owner_events = self.owner_activity.update(now, seen_xy, seen_room, active)
        for ev in owner_events:
            if any(t["trigger"].get("event") == ev["event"] and t["trigger"].get("room") == ev["room"]
                   for t in self.task_store.tasks.values()):
                w.log(f"Owner went into {ev['room']} (from {ev['from']})")
        self.task_runner.tick(w.t_sim, clock=st.get("sim_clock"), day=st.get("sim_day"),
                              sound_events=events, owner_idle_s=idle_s, events=owner_events)

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
        self.tick_ask()
        self.tick_tasks()

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
            m = self.world.mission
            if out.event == "guide_complete":
                if self.world.mission:
                    self.world.log(f"Owner delivered to {self.world.mission['target']}")
                self.task_adapter.finished(m)
                self.world.finish_mission()
                self.alerts.clear()
            elif (m is not None and m.get("task_kind") == "find_alert"
                  and out.event in ("owner_alerted", "handle_taken")):
                # taught "find me and tell me": the owner has been nudged and told -> done, no guiding
                self.world.say(m.get("say") or f"Reminder: {m['task']}.", "task")
                self.task_adapter.finished(m)
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
            "ask": self.ask.status(),  # C5: open question to the owner, or None
            # the same reply as plain text for the robot screen / speaker outside a mission (sim reads
            # `say` when no mission message is active: "Sorry, I can't do that yet...", task reminders,
            # rule confirmations); cleared after SAY_HOLD_S so the screen doesn't show it forever
            "say": (w.nudge_says or {}).get("text", "")
            if w.nudge_says and w.t_sim - float(w.nudge_says.get("t", 0.0)) <= SAY_HOLD_S else "",
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
            # debounced sound start/stop events + what Nudge heard in the last 5 min (hearing.SoundEvents)
            "sound_events": list(self.sound_events.log)[-10:],
            "heard_sounds": self.sound_events.what_did_i_hear(since_t=w.t_sim - 300.0),
            # taught tasks (contract C3, + "summary" / "queued")
            "tasks": self.task_runner.status(),
            # brain-side owner activity for idle tasks (E1.6): minutes without a sign of activity
            "owner_activity": self.owner_activity.status(),
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
            if self.answer_listener is not None:
                self.answer_listener.close()
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
