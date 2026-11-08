#!/usr/bin/env python3
"""
Shared scenario model for the LLM eval set and the fine-tuning data (menu-based agent).

The runtime agent (`nudge-brain/agent.py`) shows the LLM a menu of LEGAL options built from live
perception (`build_menu`) and asks for ONE letter (`{"choice", "reason"}` + `call_emergency_services`
in a fire + `say`, the owner message, when nudging is on the menu), constrained by a JSON schema
(`_schema`). A *scenario* here is a small dict describing one
decision point (event + belief state). `render(sc)` builds a real `WorldModel` from it, calls the
agent's own `build_menu` / `_menu_text` / `_schema`, and labels it with the agent's own
`deterministic_policy` (the rule's option is always on the menu by construction):

  expected choice  = letter of the menu option equal to deterministic_policy(event, world)
  expected escalate (only when asked) = the situation text says emergency services "should be
                     called" (owner missing > 20 s into a fire, or >= 2 unanswered nudges) — the
                     same condition `_situation` writes into the prompt.

Walking distances in the menu and the fire-aware exits come from the real floor-plan planner
(`sim.nav_map.GridPlanner`) and `evacuation.EvacuationPlanner`, so prompts match the runtime.

Scenario keys (all optional except `event`):
  event          new_sound | owner_seen | arrived_in_room | owner_not_found | owner_alerted |
                 owner_no_response | handle_taken | path_blocked | timeout
  mission        None | doorbell | knock | smoke_alarm
  robot_room     ROOMS key (robot at the room's scan point unless `robot_pose` is given)
  robot_pose     [x, y, yaw]
  belief         {room: P(owner in room)} for some rooms; the rest share the remaining mass by prior
  owner_pos      [x, y] of the last sighting (default: owner spot of `owner_room`)
  owner_room     room of the last sighting (needed when seen_ago is set and owner_pos is not)
  seen_ago       seconds since the owner was last seen (None = not seen this episode)
  owner_visible  bool, owner in the camera right now
  guiding        bool, owner holding the handle (owner pos then follows the robot)
  phase          idle | searching | approaching | guiding
  searched       rooms already scanned
  blocked        doorways / exits believed blocked (e.g. door_guest_living, exit_back)
  nudges         unanswered nudges so far; ack: True if the owner tapped the handle
  elapsed        seconds since the mission started (fire timing)
  escalated      emergency services already called
  heard_room     room the ears localised the sound to (None = not localised yet)
  fire_room      room of the fire for smoke_alarm (default: the installed alarm's room, kitchen)
  fire_heard     True = fire location from the ears ("heard"), else "installed alarm"
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "nudge-brain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import agent  # noqa: E402
import safety  # noqa: E402
from evacuation import EvacuationPlanner  # noqa: E402
from sim.house_layout import ROOMS, SOUND_SOURCES  # noqa: E402
from world_model import WorldModel  # noqa: E402

# Seconds-since-seen values this close to a policy / menu threshold are ambiguous in wall time
# (menu and label are computed microseconds apart), so generators keep clear of them.
AGE_THRESHOLDS = (2.0, 3.0, 4.0, 8.0, 10.0, 30.0)
ELAPSED_THRESHOLDS = (20.0,)

_PLANNER = None
_EVAC = None


def planner():
    """The brain's GridPlanner on the static nav map. Uses the scene-checked cache when the
    nerv-world submodule is present, else the cached map file of this or the main checkout."""
    global _PLANNER, _EVAC
    if _PLANNER is None:
        from sim.nav_map import MAP_PATH, GridPlanner, NavMap, load_or_build
        nav = None
        try:
            nav = load_or_build(verbose=False)
        except Exception:
            for cand in (MAP_PATH, ROOT.parents[2] / "assets" / "maps" / MAP_PATH.name):
                nav = NavMap.load(cand)
                if nav is not None:
                    break
        if nav is None:
            raise SystemExit("nav map not found: run `pixi run build-map` (needs `pixi run fetch-world`)")
        _PLANNER = GridPlanner(nav)
        _EVAC = EvacuationPlanner(_PLANNER)
        agent._PLANNER = _PLANNER  # the menu's walking distances use the same planner
    return _PLANNER


def _belief(w: WorldModel, spec: Optional[Dict[str, float]]) -> None:
    if not spec:
        return
    prior = dict(w.belief.p)
    rest = 1.0 - sum(spec.values())
    other = sum(v for r, v in prior.items() if r not in spec) or 1.0
    p = {r: (spec[r] if r in spec else max(0.0, rest) * prior[r] / other) for r in prior}
    w.belief.p = w.belief._norm(p)


def make_world(sc: Dict[str, Any]) -> WorldModel:
    planner()
    w = WorldModel()
    now = time.monotonic()
    room = sc.get("robot_room", "entry")
    if sc.get("robot_pose") is not None:
        w.robot_pose = tuple(float(v) for v in sc["robot_pose"])  # type: ignore[assignment]
    else:
        cx, cy = ROOMS[room]["center"]
        w.robot_pose = (cx, cy, 0.0)
    w.robot_room = room
    w.guiding_active = bool(sc.get("guiding", False))
    w.searched_rooms = set(sc.get("searched", []))
    w.blocked_doors = {d: now for d in sc.get("blocked", [])}
    w.nudge_attempts = int(sc.get("nudges", 0))
    w.owner_ack = "acknowledged" if sc.get("ack") else (
        f"no response after {w.nudge_attempts} nudge(s)" if w.nudge_attempts else None)
    _belief(w, sc.get("belief"))

    # Owner sighting
    seen_ago = sc.get("seen_ago")
    w.owner_visible = bool(sc.get("owner_visible", False))
    if w.guiding_active:
        x, y, yaw = w.robot_pose
        c, s = math.cos(yaw), math.sin(yaw)
        w.owner_pos = (x - 0.50 * c + 0.34 * s, y - 0.50 * s - 0.34 * c)
        w.owner_source = "handle"
        seen_ago = 0.0 if seen_ago is None else seen_ago
    elif sc.get("owner_pos") is not None:
        w.owner_pos = tuple(float(v) for v in sc["owner_pos"])  # type: ignore[assignment]
    elif seen_ago is not None:
        w.owner_pos = tuple(ROOMS[sc.get("owner_room", room)]["owner_spot"])  # type: ignore[assignment]
    if seen_ago is not None and w.owner_pos is not None:
        from sim.house_layout import get_room_at
        r = get_room_at(*w.owner_pos)
        if r in ROOMS:
            w.owner_room = r
        if w.owner_source != "handle":
            w.owner_source = "vision"
        w.owner_last_seen_wall = now - float(seen_ago)
        if not sc.get("belief"):  # a sighting collapses the belief, then the walking model leaks it
            w.belief.see(w.owner_pos)
            w.belief.predict(float(seen_ago))
    if not w.owner_recently_seen(2.0):  # main.tick: unseen owner -> most likely room
        w.owner_room = w.belief.most_likely()

    # Mission (+ fire / evacuation exactly like main.plan_evacuation)
    mission = sc.get("mission")
    if mission:
        w.mission = {"sound": mission, "target": str(SOUND_SOURCES[mission]["target_landmark"]),
                     "started": now - float(sc.get("elapsed", 5.0)), "started_sim": 0.0}
        w.escalated = bool(sc.get("escalated", False))
        if sc.get("heard_room"):
            w.sound_estimates[mission] = {"room": sc["heard_room"], "converged": True}
        if mission == "smoke_alarm":
            fire_room = sc.get("fire_room", "kitchen")
            if fire_room == "kitchen" and not sc.get("fire_heard"):
                sx, sy, _ = SOUND_SOURCES["smoke_alarm"]["pos"]
                w.fire = {"xy": (sx, sy), "room": "kitchen", "std": None, "source": "installed alarm"}
            else:
                fx, fy = ROOMS[fire_room]["center"]
                w.fire = {"xy": (fx, fy), "room": fire_room, "std": 1.5, "source": "heard"}
            start = w.owner_pos if (w.owner_pos is not None and w.owner_recently_seen(30.0)) else w.robot_pose[:2]
            _EVAC.reset()  # each case is a fresh mission: no exit hysteresis carried over from the last one
            w.evac = _EVAC.choose(start, w.fire, set(w.blocked_doors))
            w.mission["target"] = w.evac["target"]
    w.phase = sc.get("phase", ("guiding" if w.guiding_active else "searching") if mission else "idle")
    return w


def call_str(action: Optional[Dict[str, Any]]) -> str:
    if not action:
        return "-"
    args = ", ".join(f"{v}" for v in (action.get("args") or {}).values())
    return f"{action['tool']}({args})"


def escalation_expected(w: WorldModel) -> bool:
    """The condition under which agent._situation tells the LLM to call emergency services."""
    if not w.is_emergency() or w.escalated:
        return False
    elapsed = time.monotonic() - w.mission["started"]
    if not w.owner_recently_seen(10.0) and not w.guiding_active and elapsed > 20:
        return True
    return w.nudge_attempts >= 2 and not w.guiding_active


def _find(options: List[Dict[str, Any]], tool: str, args: Optional[Dict[str, Any]] = None) -> Optional[int]:
    return next((i for i, o in enumerate(options)
                 if o["tool"] == tool and (args is None or o["args"] == args)), None)


def sensible_label(event: str, w: WorldModel, options: List[Dict[str, Any]],
                   rule: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """The rules' call, except in three states where deterministic_policy contradicts what the
    menu tells the LLM (found by reviewing base-model "errors"; agent.py itself is unchanged):

      seen_2_3s     arrived_in_room, owner seen 2-3 s ago: the prompt says "visible now" and offers
                    the alert, the rules' 2 s cut-off would scan the room instead  -> alert_owner
      guiding       timeout while the owner holds the handle (non-fire): the rules re-nudge them
                    -> keep guiding (guide_owner_to the mission target)
      timeout_search  timeout, owner not seen: the rules always (re)scan the current room, even an
                    already-scanned or 1 % one -> go_to_room(best_search_room) unless that is the
                    current room (what emergency_action already does for a timeout in a fire)
      here          the rules' go_to_room(<the room the robot is in>) while look_around() is on the
                    menu -> look_around (same effect, without the pointless walk to the centre)
    Returns (label, override name or None)."""
    if event == "arrived_in_room" and rule["tool"] == "look_around" and not w.guiding_active \
            and w.owner_recently_seen(3.0):
        i = _find(options, "alert_owner")
        if i is not None:
            return options[i], "seen_2_3s"
    if event == "timeout" and not w.is_emergency():
        if w.guiding_active:
            i = _find(options, "guide_owner_to", {"target": w.mission["target"]})
            if i is not None:
                return options[i], "guiding"
        elif rule["tool"] == "look_around":
            best = w.best_search_room()
            if best != w.robot_room:
                i = _find(options, "go_to_room", {"room": best})
                if i is not None:
                    return options[i], "timeout_search"
    if rule["tool"] == "go_to_room" and rule["args"].get("room") == w.robot_room:
        i = _find(options, "look_around")
        if i is not None:
            return options[i], "here"
    return rule, None


def render(sc: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the runtime agent would compute for this decision point, plus the labels."""
    w = make_world(sc)
    situation, options = agent.build_menu(sc["event"], w)
    rule = agent.deterministic_policy(sc["event"], w)
    rule = {"tool": rule["tool"], "args": dict(rule.get("args") or {})}
    ask_esc = w.is_emergency() and not w.escalated
    ask_say = any(o["tool"] == "alert_owner" for o in options)  # as in NudgeAgent.submit
    prompt = agent._menu_text(situation, options, ask_esc, ask_say)
    schema = agent._schema(len(options), ask_esc, ask_say)
    label, override = sensible_label(sc["event"], w, options, rule)
    label = {"tool": label["tool"], "args": dict(label.get("args") or {})}
    idx = _find(options, label["tool"], label["args"])
    return {
        "world": w, "prompt": prompt, "schema": schema, "options": options, "ask_esc": ask_esc,
        "ask_say": ask_say,
        "rule": rule, "label": label, "override": override,
        "choice": None if idx is None else agent.LETTERS[idx],
        "escalate": escalation_expected(w) if ask_esc else None,
        # The LLM is only asked when there is a real choice (agent.submit: len(options) > 1).
        "llm_asked": len(options) > 1 and sc.get("mission") is not None,
    }


def option_action(r: Dict[str, Any], letter: str) -> Optional[Dict[str, Any]]:
    if letter not in agent.LETTERS[: len(r["options"])]:
        return None
    o = r["options"][agent.LETTERS.index(letter)]
    return {"tool": o["tool"], "args": dict(o["args"])}


def is_valid(action: Dict[str, Any], sc: Dict[str, Any]) -> Tuple[bool, str]:
    """Would safety.validate_action accept this call in this scenario?"""
    return safety.validate_action(action, make_world(sc))


def coarse_key(sc: Dict[str, Any]) -> str:
    """Train/eval disjointness key: the decision-relevant state, ignoring poses / exact numbers."""
    def bucket(v, cuts):
        return None if v is None else sum(v > t for t in cuts)
    belief = sc.get("belief") or {}
    top = sorted(belief, key=lambda r: -belief[r])[:2]
    return json.dumps([
        sc["event"], sc.get("mission"), sc.get("robot_room", "entry"), top,
        bucket(sc.get("seen_ago"), AGE_THRESHOLDS), bool(sc.get("guiding")), sorted(sc.get("searched", [])),
        sorted(sc.get("blocked", [])), int(sc.get("nudges", 0)), bool(sc.get("ack")),
        bucket(sc.get("elapsed"), ELAPSED_THRESHOLDS), bool(sc.get("escalated")), sc.get("fire_room"),
    ])


def reason_for(r: Dict[str, Any], event: str) -> str:
    """Short (<= 8 words) reason for the training target."""
    tool = r["label"]["tool"]
    fire = r["world"].is_emergency()
    if r["override"] == "guiding":
        return "owner holds the handle, keep leading"
    if tool == "guide_owner_to":
        if fire:
            return "fire: lead owner out the safest exit"
        if event == "owner_no_response":
            return "nudged three times, lead them anyway"
        return "owner ready, lead them to the door"
    if tool == "alert_owner":
        if event == "owner_no_response":
            return "no tap yet, nudge again"
        return "fire: owner found, alert them now" if fire else "owner located, nudge and alert them"
    if tool == "look_around":
        return "scan this room for the owner"
    if tool == "go_to_room":
        if r["override"] == "timeout_search":
            return "room already scanned, search the likeliest"
        return "search the most likely nearby room"
    return "no task"


def answer(r: Dict[str, Any], event: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"choice": r["choice"], "reason": reason_for(r, event)}
    if r["ask_esc"]:
        out["call_emergency_services"] = bool(r["escalate"])
    if r["ask_say"]:  # the owner message: main's own DEFAULT_SAY for the mission sound
        out["say"] = agent.DEFAULT_SAY[r["world"].mission["sound"]]
    return out


def chat_messages(sc: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Training example: system + user exactly as at runtime, assistant = labelled JSON answer."""
    r = render(sc)
    msgs = [{"role": "system", "content": agent.SYSTEM_PROMPT},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": json.dumps(answer(r, sc["event"]))}]
    return msgs, r
