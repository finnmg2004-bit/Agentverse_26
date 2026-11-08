#!/usr/bin/env python3
"""
Milestone 8a: Event-driven Local Agent (`nudge-brain/agent.py`).

The LLM (local Gemma via Ollama) is the decision-maker. It wakes ONLY on discrete events
(new_sound, arrived_in_room, owner_seen, owner_not_found, owner_alerted, owner_no_response,
handle_taken, path_blocked, timeout) and chooses ONE action.

How a decision is made
  1. `build_menu(event, world)` turns live perception into a menu of LEGAL options, each with
     the facts that matter, e.g.
        B  go_to_room(guest)          search Guest bedroom: owner probability 42 %, 1 room away
        D  guide_owner_to(front_door_outside)   exit: 14 m walk, closest to fire 9.1 m
     The safety layer decides what is legal (no exit through the fire room, no blocked exit,
     no alert before the owner is located, ...). The LLM decides which legal option is best.
  2. The LLM answers with a JSON schema (Ollama structured output) that only allows the menu's
     letters, a short reason, and - in a fire - whether to call emergency services. It cannot
     produce an invalid tool call, and the answer is ~15 tokens, so it is fast.
  3. `safety.validate_action` re-checks the choice against the world at decision time.
  4. Fallbacks: if Ollama is offline, the watchdog expires (NUDGE_WATCHDOG_S, default 4.0 s), or
     the answer is unusable, the deterministic policy decides. Every decision also records what
     the rules WOULD have chosen, so the dashboard shows when the LLM disagreed.
  5. When nudging the owner is on the menu, the LLM also writes `say`: the short, plain message
     Nudge shows / speaks to the owner (control GUI "Nudge says"; DEFAULT_SAY otherwise).
  6. NUDGE_LLM_EVENTS=new_sound,owner_not_found (comma list) limits which events consult the
     LLM; the others go straight to the rules. Default: every event.
Every decision is recorded in `self.decisions` (event, source, choice, reasoning, latency, what
the rules would have done, notes, raw reply, owner message) for the control GUI's reasoning panel.
The smoke-alarm pre-emption itself (interrupt + continuous alert) stays in plain code
(main.py / safety.py) and is instant; the LLM then makes the emergency decisions within it.

Model: env NUDGE_LLM_MODEL, else the first installed of gemma4:e2b, gemma3:4b, gemma3:1b.
Set NUDGE_LLM=off to force the deterministic policy.
Set NUDGE_LLM_GPU_LAYERS=<n> to force n model layers onto the GPU (26 on a 2 GB MX450).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from sim.house_layout import LANDMARKS, ROOMS, SOUND_SOURCES

import safety
from world_model import room_hops

OLLAMA_URL = os.environ.get("NUDGE_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_CANDIDATES = ["gemma4:e2b", "gemma3:4b", "gemma3:1b"]
MAX_NUDGES = 3
LETTERS = "ABCDEFGHIJ"
_EVENTS_ENV = os.environ.get("NUDGE_LLM_EVENTS", "").strip()
LLM_EVENTS = {e.strip() for e in _EVENTS_ENV.split(",") if e.strip()} or None  # None = every event

# What Nudge tells the owner when the LLM hasn't written a message (rules / emergency).
DEFAULT_SAY = {
    "doorbell": "Doorbell. Hold my handle. I will take you to the front door.",
    "knock": "Someone is knocking. Hold my handle. I will take you to the front door.",
    "smoke_alarm": "Smoke alarm! Hold my handle. We are going outside now.",
}
# Sounds added to the registry later (e.g. kettle) bring their own default message.
DEFAULT_SAY.update({k: str(v["say"]) for k, v in SOUND_SOURCES.items() if k not in DEFAULT_SAY and v.get("say")})

SYSTEM_PROMPT = """You are the decision-maker inside Nudge, a robot assistance dog for a deafblind owner.
The owner cannot hear household sounds and cannot see. You hear for them: you find them, nudge them
(a touch they feel), vibrate a code on your handle, and lead them by the handle.

You are woken when something happens. Read the situation and choose the ONE best option by its letter.
Priorities, in order:
1. Safety. In a fire, get the owner out of the house fast, by an exit whose route stays away from the fire.
2. Reach the owner quickly. Search the room with the HIGHEST owner probability first. Only prefer a
   closer room when the probabilities are similar (within about 10 percentage points).
3. Make sure the owner understood (they tap the handle) before leading them - unless there is a fire
   and you have already tried twice: then lead them anyway.
4. Bring the owner to where the sound needs them (e.g. the front door for the doorbell).
In a fire, call emergency services if the owner has not been found after about 20 seconds, or if the
owner is in danger and not responding.

Answer only with JSON: {"choice": "<letter>", "reason": "<at most 8 words>"} and, when asked,
"call_emergency_services": true or false, and "say": the message you show and speak to the owner when
you nudge them - what happened and where you will take them, in short plain words (for many owners,
e.g. BSL users, English is a second language). Example say: "Doorbell. Hold my handle. I will take you to the front door.\""""


# --------------------------------------------------------------------------- #
# Deterministic policy (fallback + "what the rules would do" for comparison)
# --------------------------------------------------------------------------- #
def _scan_here_useful(world) -> bool:
    """Is spinning to scan the current room worth it? Not if it was scanned already, or if the
    camera already ruled it out (e.g. walking in through an open archway: owner_belief.cleared)."""
    here = world.robot_room
    return here in ROOMS and here not in world.searched_rooms and not world.belief.cleared(here)


def _search_step(world, scan_first: bool) -> Dict[str, Any]:
    """Next search move. Never go_to_room(<current room>): that is what look_around is for.
    scan_first (just arrived / timed out in a room): scan it if that can still find the owner, else
    walk on. Otherwise (a new sound, a finished scan): the best room by probability and distance."""
    if scan_first and _scan_here_useful(world):
        return {"tool": "look_around", "args": {}}
    room = world.best_search_room()
    if room == world.robot_room:
        return {"tool": "look_around", "args": {}}
    return {"tool": "go_to_room", "args": {"room": room}}


def deterministic_policy(event: str, world) -> Dict[str, Any]:
    """Rule-based fallback that always produces a safe, valid tool call."""
    if world.is_emergency():
        return safety.emergency_action(world, event)
    mission = world.mission
    if mission is None:
        return {"tool": "get_state", "args": {}}
    sound, target = mission["sound"], mission["target"]

    if event == "user_command":  # owner asked to be taken somewhere
        if world.guiding_active or world.dist_to_owner() < 1.6:
            return {"tool": "guide_owner_to", "args": {"target": target}}
        if world.owner_recently_seen(3.0):
            return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
        return {"tool": "go_to_room", "args": {"room": world.best_search_room()}}
    if event == "owner_no_response":
        # #20: after MAX_NUDGES unanswered nudges main.py alerts the emergency contact; an owner who
        # doesn't respond can't be led, so Nudge stays with them until they do (or help arrives).
        if world.nudge_attempts < MAX_NUDGES and not _contact_alerted(world):
            return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
        return {"tool": "stay_with_owner", "args": {}}
    if event in ("new_sound", "owner_seen") and world.owner_recently_seen(3.0):
        return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
    if event in ("new_sound", "owner_not_found"):
        return _search_step(world, scan_first=False)
    if event == "arrived_in_room":
        # 3 s, like the prompt's "visible now" (was 2 s: a sighting 2-3 s old meant a pointless scan)
        if world.owner_recently_seen(3.0):
            return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
        return _search_step(world, scan_first=True)
    if event in ("owner_alerted", "handle_taken", "path_blocked"):
        return {"tool": "guide_owner_to", "args": {"target": target}}
    if event == "guide_complete":
        return {"tool": "get_state", "args": {}}
    # timeout / unknown: carry on leading if the owner holds the handle, else re-acquire the owner
    if world.guiding_active:
        return {"tool": "guide_owner_to", "args": {"target": target}}
    if world.owner_recently_seen(3.0):
        return {"tool": "alert_owner", "args": {"sound": sound, "place": target}}
    return _search_step(world, scan_first=True)


# --------------------------------------------------------------------------- #
# Menu of legal options, built from live perception
# --------------------------------------------------------------------------- #
EVENT_TEXT = {
    "new_sound": "A sound was just heard.",
    "arrived_in_room": "You just arrived in a room.",
    "owner_seen": "Your camera just spotted the owner.",
    "owner_not_found": "You looked but the owner is not here.",
    "owner_alerted": "The alert was delivered (nudge + vibration).",
    "owner_no_response": "You nudged the owner and vibrated the code, but they did not tap the handle.",
    "handle_taken": "The owner took your handle.",
    "path_blocked": "Your LiDAR found the planned route blocked.",
    "timeout": "Your last action took too long.",
    "user_command": "The owner just asked you to do something.",
}


def _contact_alerted(world) -> bool:
    """#20: the owner's emergency contact has been alerted (owner not responding, no fire)."""
    esc = getattr(world, "escalation", None)
    return bool(esc and esc.get("called") and esc.get("contact") == "emergency_contact")


def _owner_declined_999(world) -> bool:
    """#13: the owner answered "no" to "Shall I call 999?" in this fire."""
    esc = getattr(world, "escalation", None)
    return bool(esc and esc.get("contact") == "999" and esc.get("answer") == "no" and not esc.get("called"))


def _escalation_lines(world) -> List[str]:
    """What the agent should know about calls for help (C8) and the camera check (C7)."""
    lines: List[str] = []
    esc = getattr(world, "escalation", None) or {}
    if world.is_emergency():
        if world.escalated:
            lines.append("Emergency services (999) have already been called.")
        elif _owner_declined_999(world):
            lines.append("The owner said NOT to call 999. Only call if the owner is lost or not responding.")
        elif esc.get("asked") and esc.get("answer") is None:
            lines.append("You asked the owner \"Shall I call 999?\" and are waiting for the answer "
                         "(no answer = call). Keep leading them out.")
    elif _contact_alerted(world):
        lines.append(f"The owner did not respond ({esc.get('detail', '')}): their emergency contact has "
                     "been alerted. Stay with them until they respond.")
    if getattr(world, "owner_lying_down", None) and world.owner_lying_down():
        lines.append("Camera check: the owner is LYING DOWN.")
    return lines


def _areas_text(world) -> str:
    """Owner's no-go areas in words, e.g. 'the new carpet in the living room' ('' if none)."""
    from rules import area_label
    return " and ".join(area_label(a) for a in world.rules.avoid_areas())


def _label(room: Optional[str]) -> str:
    return ROOMS[room]["label"] if room in ROOMS else str(room)


_PLANNER = None


def _walk_m(world, goal) -> float:
    """Walking distance on the floor-plan map (falls back to 1.3 x straight line)."""
    global _PLANNER
    x, y = world.robot_pose[:2]
    try:
        if _PLANNER is None:
            from sim.nav_map import GridPlanner, load_or_build
            _PLANNER = GridPlanner(load_or_build(verbose=False))
        rects, cost = world.avoid_for_planning()
        wps = _PLANNER.plan((x, y), goal, set(world.blocked_doors), rects, cost)
        if wps:
            pts = [(x, y)] + [(w[0], w[1]) for w in wps]
            return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:]))
    except Exception:
        pass
    return 1.3 * ((goal[0] - x) ** 2 + (goal[1] - y) ** 2) ** 0.5


def _situation(event: str, world) -> List[str]:
    m = world.mission
    lines = [f"Event: {EVENT_TEXT.get(event, event)}", f"You are in the {_label(world.robot_room)}."]
    if m is not None:
        sound = m["sound"]
        est = world.sound_estimates.get(sound)
        where = f", localised by your ears near the {_label(est['room'])}" if est and est.get("room") else ""
        if world.is_emergency():
            fire = world.fire or {}
            elapsed = time.monotonic() - m["started"]
            lines.append(f"EMERGENCY: smoke alarm{where}. Fire believed in the {_label(fire.get('room'))} "
                         f"({fire.get('source', 'unknown')}). Alarm started {elapsed:.0f} s ago. "
                         f"Goal: get the owner out of the house.")
            if world.escalated or _owner_declined_999(world):
                pass  # see _escalation_lines
            elif not world.owner_recently_seen(10.0) and not world.guiding_active and elapsed > 20:
                lines.append(f"The owner is STILL MISSING {elapsed:.0f} s into the fire "
                             f"(more than 20 s: emergency services should be called now).")
            elif world.nudge_attempts >= 2 and not world.guiding_active:
                lines.append("The owner is not responding in a fire (emergency services should be called).")
        elif sound == "request":
            lines.append(f"The owner asked: \"{m.get('request_text', '')}\". Goal: take the owner to {m['target']}.")
        else:
            lines.append(f"Sound: {sound}{where}. Goal: bring the owner to {m['target']}.")
    if world.guiding_active:
        lines.append("The owner is holding your handle.")
    elif world.owner_recently_seen(3.0) and world.owner_pos is not None:
        lines.append(f"Owner: visible now in the {_label(world.owner_room)}, {world.dist_to_owner():.1f} m from you.")
    elif world.owner_pos is not None:
        age = time.monotonic() - world.owner_last_seen_wall
        lines.append(f"Owner: not visible; last seen {age:.0f} s ago in the {_label(world.owner_room)}.")
    else:
        lines.append("Owner: not seen yet.")
    oc = getattr(world, "owner_check", None)  # Gemma's camera look at the owner (C7; only with NUDGE_OWNER_CHECK=1)
    if oc and oc.get("state") != "not_visible" and oc.get("age", 99.0) <= 10.0:
        lines.append(f"Camera look at the owner {oc['age']:.0f} s ago: {oc['description']} "
                     f"({oc['state'].replace('_', ' ')}).")
    fallen = getattr(world, "fallen", None)  # fallen.py: someone lying on the floor YOLO can't see
    if fallen and fallen.get("stage") != "ok":
        lines.append(f"The camera sees the owner LYING ON THE FLOOR ({fallen.get('description', '')}); "
                     f"Nudge is checking on them (stage: {fallen.get('stage')}).")
    if world.owner_ack == "acknowledged":
        lines.append("The owner acknowledged the alert (tapped the handle).")
    elif world.nudge_attempts:
        lines.append(f"The owner has not acknowledged yet after {world.nudge_attempts} nudge(s).")
    lines += _escalation_lines(world)
    if world.blocked_doors:
        lines.append("Blocked doorways seen by LiDAR: " + ", ".join(sorted(world.blocked_doors)) + ".")
    rules = world.rules.summaries()
    if rules:
        lines.append("The owner's rules: " + "; ".join(rules) + ".")
        if world.is_emergency() and world.rules.avoid_areas():
            lines.append("In a fire you may break a place rule ONLY if it is clearly faster or safer "
                         "- then say why in your reason.")
    if world.is_emergency() and world.evac:
        for target, d in world.evac.get("details", {}).items():
            if d.get("unavailable"):
                lines.append(f"Exit {target} is NOT usable: {d['unavailable']}.")
    return lines


def build_menu(event: str, world) -> Tuple[List[str], List[Dict[str, Any]]]:
    """(situation lines, options). Every option is a legal tool call plus the facts behind it."""
    opts: List[Dict[str, Any]] = []

    def add(tool: str, args: Dict[str, Any], facts: str) -> None:
        if any(o["tool"] == tool and o["args"] == args for o in opts):
            return
        opts.append({"tool": tool, "args": args, "facts": facts})

    m = world.mission
    if m is None:
        return _situation(event, world), [{"tool": "get_state", "args": {}, "facts": "no task: stay put"}]
    sound, emergency = m["sound"], world.is_emergency()

    # Leading the owner (after an alert, or while they hold the handle)
    lead_now = (world.guiding_active or event in ("owner_alerted", "handle_taken", "path_blocked")
                or (event == "user_command" and world.dist_to_owner() < 1.6))
    keep_off = _areas_text(world)
    # #20: the owner did not respond to MAX_NUDGES nudges (no fire): stay with them instead of leading
    stay_now = (not emergency and event == "owner_no_response" and not world.guiding_active
                and (_contact_alerted(world) or world.nudge_attempts >= MAX_NUDGES))
    if stay_now:
        add("stay_with_owner", {},
            f"stay next to the owner with the handle held out until they respond (no tap after "
            f"{world.nudge_attempts} nudges" + ("; their emergency contact has been alerted" if _contact_alerted(world)
                                                else "") + ")")
    if lead_now or event == "owner_no_response":
        if emergency and world.evac:
            details = world.evac.get("details", {})
            usable = [d for d in details.values() if not d.get("unavailable") and d.get("fire_dist") is not None]
            best_far = max((d["fire_dist"] for d in usable), default=0.0)
            planner_pick = world.evac.get("target")

            def fire_words(dist: float) -> str:
                # Plain safety words: a small model misread "passes 1.6 m from the fire" as good
                # (rehearsal: "Terrace exit is much closer to the fire"). Eden, covering Finn.
                if dist < 3.0:
                    return f"route passes NEAR the fire ({dist:.1f} m) - dangerous"
                if dist >= best_far - 0.5:
                    return f"route stays FAR from the fire ({dist:.1f} m) - safest"
                return f"route stays {dist:.1f} m from the fire"

            for target, d in details.items():
                if d.get("unavailable"):
                    continue
                if d.get("around_ok", True):
                    add("guide_owner_to", {"target": target},
                        f"lead the owner out via {target}: {d['length']:.0f} m walk, "
                        + fire_words(d["fire_dist"])
                        + (" (the evacuation planner's pick)" if target == planner_pick else "")
                        + (f", keeps off {keep_off}" if keep_off else ""))
                if "cross_length" in d:
                    add("guide_owner_to", {"target": target, "via": "across"},
                        f"lead the owner out via {target} ACROSS {keep_off} (breaks the owner's rule): "
                        f"{d['cross_length']:.0f} m walk"
                        + (f" instead of {d['length']:.0f} m" if d.get("around_ok", True) else
                           " (no other way)")
                        + ", " + fire_words(d["cross_fire_dist"]))
        elif not emergency and not stay_now:
            why = "where they asked to go" if sound == "request" else f"where the {sound} needs them"
            add("guide_owner_to", {"target": m["target"]},
                f"lead the owner by the handle to {m['target']} ({why})"
                + (f", route keeps off {keep_off}" if keep_off else ""))

    # Nudging (again)
    if not world.guiding_active and world.owner_recently_seen(4.0) and world.owner_pos is not None:
        if event == "owner_no_response":
            add("alert_owner", {"sound": sound, "place": m["target"]},
                f"nudge the owner again (attempt {world.nudge_attempts + 1}); they have not tapped yet")
        elif not lead_now:
            add("alert_owner", {"sound": sound, "place": m["target"]},
                f"walk {world.dist_to_owner():.1f} m to the owner, nudge them and vibrate the {sound} code")

    # Searching
    if (not world.guiding_active and not lead_now and event != "owner_no_response"
            and not world.owner_recently_seen(3.0)):
        blocked = set(world.blocked_doors)
        here = world.robot_room
        if here in ROOMS and here not in world.searched_rooms:
            add("look_around", {}, f"scan the {_label(here)} you are in (owner probability "
                                   f"{world.belief.p.get(here, 0):.0%}, "
                                   + ("your camera already saw most of it on the way in: little chance)"
                                      if world.belief.cleared(here) else "not scanned yet)"))
        ranked = [(r, p) for r, p in sorted(world.belief.p.items(), key=lambda kv: -kv[1]) if r != here]
        for r, p in ranked[:4]:
            if room_hops(here, r, blocked) >= 99:
                continue
            metres = _walk_m(world, ROOMS[r]["center"])
            add("go_to_room", {"room": r},
                f"search the {_label(r)}: owner probability {p:.0%}, about {metres:.0f} m walk "
                f"(~{metres / 0.9:.0f} s)" + (f", keeping off {keep_off}" if keep_off else ""))

    # Make sure the rules' own choice is always on the menu (it is legal by construction).
    rule = deterministic_policy(event, world)
    if rule["tool"] != "get_state":
        add(rule["tool"], rule.get("args", {}), "default plan")
    return _situation(event, world), opts[: len(LETTERS)]


def _menu_text(situation: List[str], options: List[Dict[str, Any]], ask_escalation: bool,
               ask_say: bool = False) -> str:
    lines = ["Situation:"] + [f"- {s}" for s in situation] + ["", "Options:"]
    for letter, o in zip(LETTERS, options):
        args = ", ".join(f"{v}" for v in o["args"].values())
        lines.append(f"{letter}) {o['tool']}({args}) - {o['facts']}")
    if ask_escalation:
        lines.append("")
        lines.append("Also answer call_emergency_services: true or false.")
    if ask_say:
        lines.append("Also answer say: the message for the owner when you nudge them.")
    return "\n".join(lines)


def _schema(n_options: int, ask_escalation: bool, ask_say: bool = False) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "choice": {"type": "string", "enum": list(LETTERS[:n_options])},
        "reason": {"type": "string", "maxLength": 80},
    }
    required = ["choice", "reason"]
    if ask_escalation:
        props["call_emergency_services"] = {"type": "boolean"}
        required.append("call_emergency_services")
    if ask_say:
        props["say"] = {"type": "string", "maxLength": 90}
        required.append("say")
    return {"type": "object", "properties": props, "required": required}


def explain_decision(event: str, action: Dict[str, Any], world) -> str:
    """Plain-English reasoning for a rules / safety / watchdog decision (control GUI)."""
    tool = action.get("tool")
    args = action.get("args") or {}
    mission = world.mission or {}
    prefix = "SMOKE ALARM protocol: " if world.is_emergency() else ""
    keep_off = _areas_text(world)
    if tool == "alert_owner":
        return (f"{prefix}owner located in the {_label(world.owner_room)} -> walk up, nudge them and give "
                f"the {args.get('sound', mission.get('sound', '?'))} vibration code")
    if (tool == "go_to_room" and world.robot_room not in world.searched_rooms
            and world.belief.cleared(world.robot_room) and event == "arrived_in_room"):
        return (f"{prefix}the camera already ruled out the {_label(world.robot_room)} on the way in -> skip the "
                f"scan and search '{args.get('room')}' next" + (f", keeping off {keep_off}" if keep_off else ""))
    if tool == "go_to_room":
        searched = ", ".join(sorted(world.searched_rooms)) or "none yet"
        return (f"{prefix}owner not in view after '{event}' -> search the most likely room "
                f"'{args.get('room')}' (searched: {searched})" + (f", keeping off {keep_off}" if keep_off else ""))
    if tool == "look_around":
        return f"{prefix}in the {_label(world.robot_room)} but owner not visible -> spin to scan with the camera"
    if tool == "guide_owner_to":
        why = "owner is holding the handle" if world.guiding_active else f"owner has been alerted ('{event}')"
        if args.get("via") == "across":
            return (f"{prefix}{why} -> lead them to {args.get('target')} ACROSS {keep_off}: the fire overrides "
                    f"the owner's place rule because that route is much shorter")
        return f"{prefix}{why} -> lead them to {args.get('target')}" + (f", keeping off {keep_off}" if keep_off else "")
    if tool == "stay_with_owner":
        return (f"owner did not tap the handle after {world.nudge_attempts} nudge(s) -> emergency contact "
                "alerted; stay next to them with the handle out until they respond")
    if tool == "get_state":
        return "no active task -> stay put"
    return f"{prefix}{tool}({', '.join(str(v) for v in args.values())})"


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class NudgeAgent:
    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.llm_online = False
        self.enabled = os.environ.get("NUDGE_LLM", "on").lower() not in ("off", "0", "false")
        self.watchdog = safety.LLMWatchdog()
        self._lock = threading.Lock()
        self._request_id = 0
        self._pending: Optional[Dict[str, Any]] = None  # {"id", "event", "options", "rule", "result"}
        self.last_decision: Dict[str, Any] = {"tool": "-", "args": {}, "source": "-", "reason": "", "latency": 0.0}
        self.last_menu: List[str] = []
        self.history: List[str] = []
        self.stats = {"llm": 0, "agreed": 0, "fallback": 0}
        # Recent decisions for the dashboard timeline and the control GUI's reasoning panel:
        # {"t", "tool", "args", "source", "latency", "agreed", "event", "reasoning", "rule", "note",
        #  "llm_raw", "model", "say", "escalate"}
        self.decisions: List[Dict[str, Any]] = []
        self.llm_calls = 0
        self.llm_latencies: Deque[float] = deque(maxlen=30)
        self._note = ""  # why the LLM's answer was not used, attached to the next decision
        if self.enabled:
            threading.Thread(target=self._probe_loop, daemon=True).start()

    # ------------------------------------------------------------------ Ollama
    def _http(self, path: str, payload: Optional[dict], timeout: float) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _probe_loop(self) -> None:
        """Find an installed Gemma model; re-check every 20 s so the agent recovers if Ollama starts later."""
        while True:
            if not self.llm_online:
                try:
                    tags = self._http("/api/tags", None, timeout=1.0)
                    names = [m.get("name", "") for m in tags.get("models", [])]
                    wanted = [os.environ["NUDGE_LLM_MODEL"]] if os.environ.get("NUDGE_LLM_MODEL") else MODEL_CANDIDATES
                    for cand in wanted:
                        match = next((n for n in names if n == cand or n.startswith(cand + ":") or n.split(":")[0] == cand), None)
                        if match:
                            self.model = match
                            break
                    if self.model:
                        # Warm-up so the first real event isn't slowed by model loading.
                        # First load can take ~60 s on a small GPU (measured 58 s on an MX450).
                        self._chat("Situation:\n- warm-up\n\nOptions:\nA) get_state() - nothing to do",
                                   _schema(1, False), timeout=180.0)
                        self.llm_online = True
                        print(f"[agent] Local LLM online: {self.model} via Ollama")
                    else:
                        print(f"[agent] Ollama running but none of {wanted} installed; using deterministic policy")
                except Exception:
                    pass
            time.sleep(20.0)

    @staticmethod
    def _options(num_predict: int = 48) -> Dict[str, Any]:
        opts: Dict[str, Any] = {"temperature": 0.0, "num_predict": num_predict}
        # Per-machine GPU offload. Ollama's auto-fit is conservative on small cards (an MX450 2 GB
        # gets 0/36 layers -> ~10 s/decision on CPU); forcing 26 layers there gives ~4 s.
        # Too high = CUDA out-of-memory (30+ on 2 GB). Unset = let Ollama decide.
        layers = os.environ.get("NUDGE_LLM_GPU_LAYERS")
        if layers:
            opts["num_gpu"] = int(layers)
        return opts

    def _chat(self, user: str, schema: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        resp = self._http("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,  # structured output: only the menu's letters are possible
            # Gemma 4 reasons ("thinks") by default, which eats num_predict and leaves an empty answer
            # well past the watchdog. Decisions here are one choice, so answer directly.
            "think": False,
            "keep_alive": "30m",
            # the owner message ("say") needs room for ~20 more tokens
            "options": self._options(96 if "say" in schema.get("properties", {}) else 48),
        }, timeout=timeout)
        return json.loads(resp["message"]["content"])

    def _llm_worker(self, req_id: int, prompt: str, schema: Dict[str, Any], event: str = "") -> None:
        t0 = time.perf_counter()
        try:
            result: Any = self._chat(prompt, schema, timeout=safety.WATCHDOG_S + 10.0)
        except Exception as exc:
            result = {"error": str(exc)}
        dt = time.perf_counter() - t0
        self.llm_latencies.append(dt)
        with self._lock:
            current = self._pending is not None and self._pending["id"] == req_id
            if current:
                self._pending["result"] = result
        print(f"[agent] LLM #{req_id} '{event}' answered in {dt:.2f} s"
              f"{'' if current else ' (too late: superseded or watchdog already decided)'}", flush=True)

    # ------------------------------------------------------------------ public API
    @property
    def backend(self) -> str:
        if not self.enabled:
            return "deterministic (LLM off)"
        return f"{self.model} (Ollama)" if self.llm_online else "deterministic (Ollama offline)"

    def submit(self, event: str, world) -> None:
        """Wake the agent on a discrete event. A newer event supersedes a pending one."""
        world.log(f"EVENT {event}")
        situation, options = build_menu(event, world)
        rule = deterministic_policy(event, world)
        # The owner's "no" to "Shall I call 999?" stands; the no-response / not-found rules still call.
        ask_esc = world.is_emergency() and not world.escalated and not _owner_declined_999(world)
        ask_say = any(o["tool"] == "alert_owner" for o in options)
        prompt = _menu_text(situation, options, ask_esc, ask_say)
        self.last_menu = prompt.splitlines()
        consult = (self.enabled and self.llm_online and len(options) > 1
                   and (LLM_EVENTS is None or event in LLM_EVENTS))
        with self._lock:
            self._request_id += 1
            self._pending = {"id": self._request_id, "event": event, "options": options, "rule": rule,
                             "result": None, "ask_esc": ask_esc, "llm": consult}
            req_id = self._request_id
        self.watchdog.start()
        if consult:
            self.llm_calls += 1
            threading.Thread(target=self._llm_worker,
                             args=(req_id, prompt, _schema(len(options), ask_esc, ask_say), event),
                             daemon=True).start()

    def cancel(self) -> None:
        with self._lock:
            self._pending = None
        self.watchdog.stop()

    def pending_info(self) -> Optional[Dict[str, Any]]:
        """The event being decided right now (control GUI 'thinking...' line)."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        return {"event": pending["event"], "elapsed": round(self.watchdog.elapsed(), 2),
                "waiting_for_llm": bool(pending.get("llm") and pending["result"] is None)}

    def llm_stats(self) -> Dict[str, Any]:
        lat = list(self.llm_latencies)
        return {"calls": self.llm_calls, "timeout_s": safety.WATCHDOG_S,
                "avg_s": round(sum(lat) / len(lat), 2) if lat else None,
                "max_s": round(max(lat), 2) if lat else None,
                "agreed": self.stats["agreed"], "llm": self.stats["llm"], "fallback": self.stats["fallback"]}

    def poll(self, world) -> Optional[Dict[str, Any]]:
        """Return a decided action once available (LLM, or rules on watchdog / offline / bad answer)."""
        with self._lock:
            pending = self._pending
        if pending is None:
            return None
        event, options, rule = pending["event"], pending["options"], pending["rule"]

        if len(options) == 1:
            o = options[0]
            return self._finish({"tool": o["tool"], "args": o["args"]}, "only option", o["facts"], rule, world,
                                event)
        if not pending.get("llm"):
            if self.enabled and self.llm_online:  # NUDGE_LLM_EVENTS: routine event, rules on purpose
                return self._finish(rule, "rules", "routine step", rule, world, event)
            self.stats["fallback"] += 1
            return self._finish(rule, "rules", "LLM offline", rule, world, event)

        result = pending["result"]
        if result is not None:
            raw = json.dumps(result)[:300]
            if isinstance(result, dict) and result.get("choice") in LETTERS[: len(options)]:
                o = options[LETTERS.index(result["choice"])]
                action = {"tool": o["tool"], "args": dict(o["args"])}
                if pending["ask_esc"] and result.get("call_emergency_services") is True:
                    action["escalate"] = True
                ok, why = safety.validate_action(action, world)
                if ok:
                    self.stats["llm"] += 1
                    say = str(result.get("say") or "").strip()[:120]
                    if say and world.mission is not None:
                        world.mission["say"] = say
                    return self._finish(action, "llm", str(result.get("reason", ""))[:70], rule, world,
                                        event, raw, say)
                world.log(f"LLM choice {result['choice']} rejected by safety: {why}")
                self._note = f"LLM chose {o['tool']}({', '.join(str(v) for v in o['args'].values())}) - rejected by safety: {why}"
            else:
                world.log(f"LLM error: {str(result)[:70]}")
                self._note = f"LLM answer unusable: {str(result)[:120]}"
                if isinstance(result, dict) and "error" in result:
                    self.llm_online = False  # probe loop will reconnect
            self.stats["fallback"] += 1
            return self._finish(rule, "rules", "LLM answer unusable", rule, world, event, raw)

        if self.watchdog.expired():
            world.log(f"WATCHDOG: LLM > {safety.WATCHDOG_S:.1f} s, fallback rules take over")
            self._note = f"LLM took > {safety.WATCHDOG_S:.1f} s - watchdog switched to the rules"
            self.stats["fallback"] += 1
            return self._finish(rule, "watchdog", "LLM too slow", rule, world, event)
        return None

    def _finish(self, action: Dict[str, Any], source: str, reason: str, rule: Dict[str, Any],
                world, event: str = "", llm_raw: Optional[str] = None, say: str = "") -> Dict[str, Any]:
        latency = self.watchdog.elapsed()
        self.cancel()
        agreed = action["tool"] == rule.get("tool") and action.get("args", {}) == rule.get("args", {})
        if source == "llm" and agreed:
            self.stats["agreed"] += 1
        rule_txt = f"{rule.get('tool')}({', '.join(str(v) for v in rule.get('args', {}).values())})"
        self.last_decision = {"tool": action["tool"], "args": action.get("args", {}), "source": source,
                              "reason": reason, "latency": round(latency, 2),
                              "agreed": agreed, "rule": rule_txt, "escalate": bool(action.get("escalate"))}
        args = ", ".join(f"{v}" for v in action.get("args", {}).values())
        self.history.append(f"{action['tool']}({args})")
        reasoning = reason if source in ("llm", "only option") else explain_decision(event, action, world)
        self.decisions = (self.decisions + [{"t": round(world.t_sim, 1), "tool": action["tool"],
                                              "args": action.get("args", {}), "source": source,
                                              "latency": round(latency, 2), "agreed": agreed,
                                              "event": event, "reasoning": reasoning, "rule": rule_txt,
                                              "note": self._note, "llm_raw": llm_raw, "say": say,
                                              "model": self.model if source == "llm" else None,
                                              "escalate": bool(action.get("escalate"))}])[-25:]
        self._note = ""
        extra = "" if agreed or source != "llm" else f"  (rules would: {rule_txt})"
        why = f" - {reason}" if source == "llm" and reason else ""
        world.log(f"DECIDE [{source} {latency:.2f}s] {action['tool']}({args}){why}{extra}")
        return action
