#!/usr/bin/env python3
"""
Teachable tasks (`nudge-brain/tasks.py`) — overnight contract C1 / C3 (overnight/README.md).

The owner teaches Nudge a new job in one sentence ("every night at 10, take me to bed"); the
command parser (rules.py, E1.2) turns it into a task program, and this module does the rest in
plain code:

  validate_task(task)   C1 check: known skills / places / sounds only, <= 8 steps, tier-1 sounds
                        refused (a smoke alarm can never be taught over or delayed)
  TaskStore             assets/tasks.json (gitignored; NUDGE_TASKS_FILE overrides)
  TriggerWatcher        time (state.sim_clock / sim_day), sound (SoundEvents "start"), idle
                        (owner hasn't moved for N minutes), event (owner_entered_room)
  TaskRunner            runs ONE task at a time as a list of jobs through an adapter that owns the
                        real mission machinery (main.py wires it, E1.3). It never starts while a
                        higher-priority mission is active, and a smoke alarm fails a running task.

Steps are compiled into jobs that the existing machinery already knows how to do:
  find_owner [say|alert_owner] [offer_handle] guide_owner_to X  -> {"kind": "mission", target X, say}
  find_owner [say|alert_owner] (no guiding)                     -> {"kind": "find_alert", say}
  say (owner not searched for)                                  -> {"kind": "say", text}
  go_to_room R                                                  -> {"kind": "go_to_room", room R}
  wait S                                                        -> {"kind": "wait", s}  (runner-internal)
No imports from main.py / skills.py, so it can be unit-tested with a fake clock (scripts/test_tasks.py).
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sim.house_layout import LANDMARKS, ROOMS, SOUND_SOURCES

SKILLS = ("find_owner", "alert_owner", "say", "offer_handle", "guide_owner_to", "go_to_room", "wait")
TRIGGER_KINDS = ("time", "sound", "idle", "event")
EVENTS = ("owner_entered_room",)
MAX_STEPS = 8
MAX_TEXT = 140
MAX_WAIT_S = 600.0
TIME_GRACE_MIN = 30  # a clock jump past a task's time by more than this does not fire it (stale)
CANT_DO = "Sorry, I can't do that yet. I can find you, remind you, or take you somewhere."

# Words owners use for places -> house_layout keys ("bed" = the guest bedroom, C1).
PLACE_ALIASES = {
    "bed": "guest", "bedroom": "guest", "my bed": "guest", "guest_bedroom": "guest",
    "living": "living_room", "lounge": "living_room", "sofa": "living_room", "couch": "living_room",
    "front": "front_door", "door": "front_door", "front_door_inside": "front_door",
    "kitchen_stove": "stove", "oven": "stove", "family": "family_room", "garden": "family_room",
}
PLACE_LABELS = {"guest": "bed", "living_room": "the living room", "front_door": "the front door",
                "stove": "the stove", "kettle": "the kettle", "kitchen": "the kitchen",
                "dining": "the dining room", "entry": "the hall", "family_room": "the family room"}

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "assets" / "tasks.json"


# ---------------------------------------------------------------------------- helpers
def places() -> List[str]:
    return sorted(set(LANDMARKS) | set(ROOMS))


def place_key(name: Any) -> Optional[str]:
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace(" ", "_")
    key = PLACE_ALIASES.get(key, PLACE_ALIASES.get(key.replace("_", " "), key))
    return key if key in LANDMARKS or key in ROOMS else None


def place_label(key: str) -> str:
    if key in PLACE_LABELS:
        return PLACE_LABELS[key]
    if key in ROOMS:
        return "the " + str(ROOMS[key].get("label", key)).lower()
    return "the " + key.replace("_", " ")


def teachable_sounds() -> List[str]:
    """Registry sounds a task may react to: everything except tier 1 (danger)."""
    return [s for s, spec in SOUND_SOURCES.items()
            if int(spec.get("tier", 3)) != 1 and spec.get("alert_mode") != "continuous"]


def parse_hhmm(text: Any) -> Optional[int]:
    """'22:00' / '7:05' -> minutes since midnight, else None."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(text or ""))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return h * 60 + mi if 0 <= h < 24 and 0 <= mi < 60 else None


def fmt_hhmm(minutes: int) -> str:
    minutes %= 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def spoken_time(minutes: int) -> str:
    """22:00 -> '10 pm', 07:30 -> '7:30 am' (short and plain, for the screen and speech)."""
    h, mi = divmod(minutes % 1440, 60)
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if mi == 0 else f"{h12}:{mi:02d} {suffix}"


def _name(raw: Any, trigger: Dict[str, Any]) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(raw or "").lower()).strip("_")[:24]
    if name:
        return name
    kind = trigger.get("kind")
    if kind == "time":
        return "at_" + str(trigger.get("at", "")).replace(":", "")
    if kind == "sound":
        return "on_" + str(trigger.get("sound", ""))
    return f"{kind}_task"


# ---------------------------------------------------------------------------- validator (C1)
def validate_task(task: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Check + normalise a task program. Returns (task, "") or (None, why it was refused).
    Never raises: anything malformed (e.g. from the LLM) is just refused."""
    if not isinstance(task, dict):
        return None, "not a task"
    trig = task.get("trigger")
    if not isinstance(trig, dict) or trig.get("kind") not in TRIGGER_KINDS:
        return None, "unknown trigger"
    kind = trig["kind"]
    t: Dict[str, Any] = {"kind": kind}
    if kind == "time":
        at = parse_hhmm(trig.get("at"))
        if at is None:
            return None, "time must be HH:MM"
        repeat = trig.get("repeat", "daily")
        if repeat not in ("daily", "once"):
            return None, "repeat must be daily or once"
        t.update(at=fmt_hhmm(at), repeat=repeat)
    elif kind == "sound":
        sound = trig.get("sound")
        if sound not in SOUND_SOURCES:
            return None, f"I can't hear '{sound}' yet"
        if sound not in teachable_sounds():
            return None, "danger sounds can't be changed: I always get you out"
        t.update(sound=sound, phase="start")
    elif kind == "idle":
        try:
            minutes = float(trig.get("minutes"))
        except (TypeError, ValueError):
            return None, "idle needs minutes"
        if not 1 <= minutes <= 24 * 60:
            return None, "idle minutes out of range"
        t.update(minutes=round(minutes, 1))
    else:  # event
        if trig.get("event") not in EVENTS:
            return None, "unknown event"
        room = trig.get("room")
        room = place_key(room) if room is not None else None
        if room is None or room not in ROOMS:
            return None, "unknown room"
        t.update(event=trig["event"], room=room)

    steps_in = task.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        return None, "no steps"
    if len(steps_in) > MAX_STEPS:
        return None, f"too many steps (max {MAX_STEPS})"
    steps: List[Dict[str, Any]] = []
    for st in steps_in:
        if not isinstance(st, dict) or st.get("skill") not in SKILLS:
            return None, f"unknown skill {st.get('skill') if isinstance(st, dict) else st!r}"
        skill, args = st["skill"], st.get("args") or {}
        if not isinstance(args, dict):
            return None, f"bad args for {skill}"
        out: Dict[str, Any] = {}
        if skill in ("say", "alert_owner"):
            text = " ".join(str(args.get("text", "")).split())[:MAX_TEXT]
            if skill == "say" and not text:
                return None, "say needs text"
            out["text"] = text
        elif skill == "guide_owner_to":
            target = place_key(args.get("target"))
            if target is None:
                return None, f"unknown place '{args.get('target')}'"
            out["target"] = target
        elif skill == "go_to_room":
            room = place_key(args.get("room"))
            if room is None or room not in ROOMS:
                return None, f"unknown room '{args.get('room')}'"
            out["room"] = room
        elif skill == "wait":
            try:
                s = float(args.get("s"))
            except (TypeError, ValueError):
                return None, "wait needs seconds"
            if not 0 < s <= MAX_WAIT_S:
                return None, "wait out of range"
            out["s"] = round(s, 1)
        steps.append({"skill": skill, "args": out})

    until = task.get("until")
    if until is not None and not isinstance(until, (int, float)):
        until = None  # only a sim_day number is understood; anything else = no end
    return {"name": _name(task.get("name"), t), "trigger": t, "steps": steps, "until": until,
            "source": str(task.get("source", "")), "text": str(task.get("text", ""))[:200]}, ""


# ---------------------------------------------------------------------------- words
def describe_trigger(trigger: Dict[str, Any]) -> str:
    kind = trigger.get("kind")
    if kind == "time":
        when = spoken_time(parse_hhmm(trigger["at"]) or 0)
        return f"Every day at {when}" if trigger.get("repeat", "daily") == "daily" else f"At {when}"
    if kind == "sound":
        label = str(SOUND_SOURCES.get(trigger.get("sound"), {}).get("label", trigger.get("sound")))
        return f"When I hear the {label.lower()}"
    if kind == "idle":
        m = float(trigger.get("minutes", 0))
        span = f"{m / 60:g} hours" if m >= 60 else f"{m:g} minutes"
        return f"If you haven't moved for {span}"
    return f"When you go into {place_label(str(trigger.get('room')))}"


def describe_steps(steps: List[Dict[str, Any]]) -> str:
    """Plain words for what the task does, from the robot's point of view ("I'll ...")."""
    parts: List[str] = []
    guides = any(st["skill"] == "guide_owner_to" for st in steps)  # keep it short: the message is said later
    for st in steps:
        a = st.get("args", {})
        s = st["skill"]
        if s == "find_owner":
            parts.append("find you")
        elif s in ("say", "alert_owner") and guides:
            continue
        elif s in ("say", "alert_owner") and a.get("text"):
            parts.append(f'tell you "{a["text"]}"')
        elif s == "alert_owner":
            parts.append("nudge you")
        elif s == "guide_owner_to":
            parts.append(f"take you to {place_label(a['target'])}")
        elif s == "go_to_room":
            parts.append(f"go to {place_label(a['room'])}")
        elif s == "wait":
            parts.append(f"wait {a['s']:g} s")
    # "find you and take you to bed" reads better than "find you, take you to bed"
    if len(parts) >= 2 and parts[0] == "find you" and parts[-1].startswith("take you"):
        parts = parts[1:]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def confirmation(task: Dict[str, Any]) -> str:
    """'OK. Every day at 10 pm I'll take you to bed.'"""
    out = f"OK. {describe_trigger(task['trigger'])} I'll {describe_steps(task['steps'])}."
    return out.replace('.".', '."').replace('!".', '!"').replace('?".', '?"')  # no 'tell you "X.".'


# ---------------------------------------------------------------------------- compile to jobs
def compile_jobs(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group steps into jobs the existing mission machinery can run (see module docstring)."""
    jobs: List[Dict[str, Any]] = []
    i = 0
    while i < len(steps):
        s, a = steps[i]["skill"], steps[i].get("args", {})
        if s in ("find_owner", "alert_owner", "offer_handle", "guide_owner_to"):
            # an owner-contact run: find_owner / alert / say / offer_handle ... [guide_owner_to]
            job: Dict[str, Any] = {"kind": "find_alert", "say": ""}
            while i < len(steps):
                s, a = steps[i]["skill"], steps[i].get("args", {})
                if s in ("say", "alert_owner"):
                    if a.get("text"):
                        job["say"] = (job["say"] + " " + a["text"]).strip()[:MAX_TEXT]
                elif s == "guide_owner_to":
                    job.update(kind="mission", target=a["target"])
                    i += 1
                    break
                elif s not in ("find_owner", "offer_handle"):
                    break
                i += 1
            jobs.append(job)
            continue
        if s == "say":
            jobs.append({"kind": "say", "text": a["text"]})
        elif s == "go_to_room":
            jobs.append({"kind": "go_to_room", "room": a["room"]})
        elif s == "wait":
            jobs.append({"kind": "wait", "s": float(a["s"])})
        i += 1
    return jobs


# ---------------------------------------------------------------------------- store
class TaskStore:
    """Taught tasks, persisted as JSON (assets/tasks.json, gitignored). Name is the key."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path or os.environ.get("NUDGE_TASKS_FILE") or DEFAULT_FILE)
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        self.tasks = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in raw if isinstance(raw, list) else []:
            task, _ = validate_task(item)
            if task is not None:
                task["state"] = item.get("state", "idle") if item.get("state") in ("idle", "done") else "idle"
                task["last_run"] = item.get("last_run")
                self.tasks[task["name"]] = task

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(list(self.tasks.values()), indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:  # never crash the brain over a file
            print(f"[tasks] could not save {self.path}: {exc}", flush=True)

    def add(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Add (or replace, same name) an already-validated task."""
        task = dict(task, state="idle", last_run=None)
        self.tasks[task["name"]] = task
        self.save()
        return task

    def remove(self, name: str) -> Optional[Dict[str, Any]]:
        key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        task = self.tasks.pop(key, None)
        if task is None:  # "forget the bed task" -> bedtime; match on a prefix / contained word
            hits = [n for n in self.tasks if key and (n.startswith(key) or key in n)]
            if len(hits) == 1:
                task = self.tasks.pop(hits[0])
        if task is not None:
            self.save()
        return task

    def clear(self) -> None:
        self.tasks = {}
        self.save()

    def __iter__(self):
        return iter(list(self.tasks.values()))

    def __len__(self) -> int:
        return len(self.tasks)


# ---------------------------------------------------------------------------- triggers
class TriggerWatcher:
    """Decides which tasks are due. Feed it each brain tick; returns task names that fired."""

    def __init__(self) -> None:
        self.prev_abs: Optional[int] = None  # previous sim clock, minutes since day 0
        self.idle_armed: Dict[str, bool] = {}

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def clock_abs(clock: Optional[str], day: Optional[int]) -> Optional[int]:
        m = parse_hhmm(clock)
        return None if m is None else int(day or 0) * 1440 + m

    def update(self, tasks, clock: Optional[str] = None, day: Optional[int] = None,
               sound_events: Optional[List[Dict[str, Any]]] = None, owner_idle_s: Optional[float] = None,
               events: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        now = self.clock_abs(clock, day)
        prev = self.prev_abs
        if now is not None:
            # first reading or the clock went back (sim reset): nothing has been crossed
            self.prev_abs = now
            if prev is None or now < prev:
                prev = now
        fired: List[str] = []
        for task in tasks:
            if task.get("state") == "done":
                continue
            if task.get("until") is not None and day is not None and day > task["until"]:
                continue
            trig, name = task["trigger"], task["name"]
            kind = trig["kind"]
            if kind == "time" and now is not None and prev is not None and now > prev:
                at = parse_hhmm(trig["at"]) or 0
                for d in range(prev // 1440, now // 1440 + 1):
                    t_abs = d * 1440 + at
                    if prev < t_abs <= now and now - t_abs <= TIME_GRACE_MIN:
                        fired.append(name)
                        break
            elif kind == "sound":
                for ev in sound_events or ():
                    if ev.get("phase") == "start" and ev.get("sound") == trig["sound"]:
                        fired.append(name)
                        break
            elif kind == "idle" and owner_idle_s is not None:
                over = owner_idle_s >= float(trig["minutes"]) * 60.0
                if over and self.idle_armed.get(name, True):
                    fired.append(name)
                self.idle_armed[name] = not over  # fire once per still stretch
            elif kind == "event":
                for ev in events or ():
                    if ev.get("event") == trig["event"] and ev.get("room") == trig["room"]:
                        fired.append(name)
                        break
        return fired

    def next_run(self, task: Dict[str, Any]) -> Optional[str]:
        """'22:00' (today) / 'day 2 22:00' for time tasks, a short phrase otherwise; None when done."""
        if task.get("state") == "done":
            return None
        trig = task["trigger"]
        if trig["kind"] != "time":
            return describe_trigger(trig).lower()
        at = parse_hhmm(trig["at"]) or 0
        if self.prev_abs is None:
            return trig["at"]
        day, now_m = divmod(self.prev_abs, 1440)
        return trig["at"] if at > now_m else f"day {day + 1} {trig['at']}"


# ---------------------------------------------------------------------------- runner
class TaskRunner:
    """Runs taught tasks, one at a time, as jobs through an adapter (main.py, E1.3):

        adapter.busy() -> bool                 a built-in / higher-priority mission is active
        adapter.emergency() -> bool            tier-1 danger: fails any running task
        adapter.start_job(job, task) -> None   start a "mission" / "find_alert" / "say" / "go_to_room" job
        adapter.job_status(job) -> str         "running" | "ok" | "failed"
        adapter.cancel_job(job) -> None
    `wait` jobs are timed here on the sim clock `t` passed to tick().
    """

    def __init__(self, store: TaskStore, adapter, log: Callable[[str], None] = print) -> None:
        self.store, self.adapter, self.log = store, adapter, log
        self.watcher = TriggerWatcher()
        self.pending: List[str] = []
        self.current: Optional[Dict[str, Any]] = None  # {"name", "jobs", "i", "job", "t_job"}

    def reset(self) -> None:
        """Sim reset: forget what was running / queued (the tasks themselves are kept)."""
        self.watcher.reset()
        self.pending.clear()
        self.current = None
        for task in self.store:
            if task.get("state") == "running":
                task["state"] = "idle"

    def fire(self, name: str) -> bool:
        """Queue a task by name (also for tests / a GUI 'run now' button)."""
        if name in self.store.tasks and name not in self.pending and (
                self.current is None or self.current["name"] != name):
            self.pending.append(name)
            return True
        return False

    def tick(self, t: float, **trigger_inputs) -> None:
        for name in self.watcher.update(self.store, **trigger_inputs):
            if self.fire(name):
                self.log(f"TASK {name}: triggered ({describe_trigger(self.store.tasks[name]['trigger'])})")
        cur = self.current
        if cur is not None:
            if self.adapter.emergency():
                if cur["job"] is not None and cur["job"]["kind"] != "wait":
                    self.adapter.cancel_job(cur["job"])
                self._end("failed", "pre-empted by a danger sound", t)
                return
            self._advance(t)
            return
        if self.pending and not self.adapter.busy() and not self.adapter.emergency():
            name = self.pending.pop(0)
            task = self.store.tasks.get(name)
            if task is None:
                return
            task["state"] = "running"
            self.current = {"name": name, "jobs": compile_jobs(task["steps"]), "i": -1, "job": None, "t_job": t}
            self.log(f"TASK {name}: start ({len(self.current['jobs'])} jobs)")
            self._next_job(t)

    def _next_job(self, t: float) -> None:
        cur = self.current
        cur["i"] += 1
        if cur["i"] >= len(cur["jobs"]):
            self._end("done", "", t)
            return
        job = cur["jobs"][cur["i"]]
        cur["job"], cur["t_job"] = job, t
        if job["kind"] != "wait":
            self.adapter.start_job(copy.deepcopy(job), self.store.tasks[cur["name"]])

    def _advance(self, t: float) -> None:
        cur = self.current
        job = cur["job"]
        if job["kind"] == "wait":
            status = "ok" if t - cur["t_job"] >= job["s"] else "running"
        else:
            status = self.adapter.job_status(job)
        if status == "ok":
            self._next_job(t)
        elif status == "failed":
            self._end("failed", f"{job['kind']} failed", t)

    def _end(self, state: str, why: str, t: float) -> None:
        name = self.current["name"]
        self.current = None
        task = self.store.tasks.get(name)
        if task is None:
            return
        repeat_once = task["trigger"]["kind"] == "time" and task["trigger"].get("repeat") == "once"
        task["state"] = "done" if (state == "done" and repeat_once) else ("failed" if state == "failed" else "idle")
        task["last_run"] = {"t": round(t, 1), "result": state, "why": why}
        self.store.save()
        self.log(f"TASK {name}: {state}" + (f" ({why})" if why else ""))

    def stop(self, t: float, why: str = "stopped by the owner") -> Optional[str]:
        """Owner said "stop": end the running task (failed) and drop the queue. Returns its name."""
        self.pending.clear()
        if self.current is None:
            return None
        name = self.current["name"]
        if self.current["job"] is not None and self.current["job"]["kind"] != "wait":
            self.adapter.cancel_job(self.current["job"])
        self._end("failed", why, t)
        return name

    def remove(self, name: str) -> Optional[Dict[str, Any]]:
        """Forget a task; if it is running, stop it first."""
        task = self.store.remove(name)
        if task is not None:
            if self.current is not None and self.current["name"] == task["name"]:
                if self.current["job"] is not None and self.current["job"]["kind"] != "wait":
                    self.adapter.cancel_job(self.current["job"])
                self.current = None
            if task["name"] in self.pending:
                self.pending.remove(task["name"])
        return task

    def sound_task(self, sound: str) -> Optional[str]:
        """Name of a taught task triggered by `sound` (E1.4): it REPLACES the built-in response to
        that sound. Tier-1 (danger) sounds never: the validator refuses them, and this re-checks."""
        if int(SOUND_SOURCES.get(sound, {}).get("tier", 1)) <= 1:
            return None
        for task in self.store:
            trig = task.get("trigger", {})
            if trig.get("kind") == "sound" and trig.get("sound") == sound and task.get("state") != "done":
                return task["name"]
        return None

    def status(self) -> List[Dict[str, Any]]:
        """Contract C3: brain status `tasks`."""
        out = []
        for task in self.store:
            state = task.get("state", "idle")
            if self.current is not None and self.current["name"] == task["name"]:
                state = "running"
            out.append({"name": task["name"], "trigger": task["trigger"], "steps": task["steps"],
                        "next_run": self.watcher.next_run(task), "source": task.get("source", ""),
                        "state": state, "last_run": task.get("last_run"),
                        "summary": confirmation(task)[4:], "queued": task["name"] in self.pending})
        return out
