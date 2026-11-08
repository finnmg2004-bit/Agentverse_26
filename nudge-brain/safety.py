#!/usr/bin/env python3
"""
Milestone 8b: Plain-code Safety Layer (`nudge-brain/safety.py`).
Hard invariants that hold no matter what the LLM says:
  1. Smoke-alarm pre-emption: smoke_alarm confidence >= 0.6 interrupts any plan, sets the
     `continuous` alert and runs the deterministic evacuation to `back_exit_outside`.
  2. 2.0 s LLM watchdog: if the agent hasn't produced a valid tool call in time,
     deterministic fallback rules decide instead.
  3. Proximity speed cap: |v| <= 0.4 m/s whenever within 1.0 m of the owner.
  4. LiDAR bumper: no forward motion with an obstacle < 0.30 m ahead (except a deliberate nudge).
  5. Action validation: LLM tool calls with bad arguments or that break the emergency
     protocol are rejected.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from sim.house_layout import LANDMARKS, ROOMS, SOUND_SOURCES

SMOKE_THRESHOLD = 0.60
import os  # noqa: E402

# LLM answer deadline. 4.0 s: Gemma answers in ~0.4-0.6 s on an RTX 5050 laptop but spikes to ~3 s,
# so 4 s keeps it deciding; raise it (NUDGE_WATCHDOG_S=6) on a slow GPU, lower it for a snappier
# worst case (the robot holds its current action while waiting).
WATCHDOG_S = float(os.environ.get("NUDGE_WATCHDOG_S", "4.0"))
MAX_NUDGES = 3
FIRE_MARGIN_M = 1.0  # an LLM-chosen exit route may not pass more than this much closer to the fire than the planner's
OWNER_SPEED_CAP = 0.40
OWNER_CAP_RADIUS = 1.0
BUMPER_RANGE = 0.30
EVACUATION_TARGET = "back_exit_outside"  # default only; evacuation.py picks the safest exit per fire

TOOLS = ("get_state", "go_to_room", "look_around", "alert_owner", "offer_handle",
         "guide_owner_to", "investigate_sound", "stay_with_owner")


class LLMWatchdog:
    def __init__(self, timeout_s: float = WATCHDOG_S) -> None:
        self.timeout_s = timeout_s
        self.started: Optional[float] = None

    def start(self) -> None:
        self.started = time.monotonic()

    def stop(self) -> None:
        self.started = None

    def elapsed(self) -> float:
        return 0.0 if self.started is None else time.monotonic() - self.started

    def expired(self) -> bool:
        return self.started is not None and self.elapsed() > self.timeout_s


def smoke_detected(sound_result: Dict[str, Any]) -> bool:
    return float(sound_result.get("scores", {}).get("smoke_alarm", 0.0)) >= SMOKE_THRESHOLD


def _evac_args(world) -> Dict[str, Any]:
    """guide_owner_to args for the planned exit; `via: across` when the plan breaks an owner's place rule."""
    args: Dict[str, Any] = {"target": world.evac_target()}
    if world.evac and world.evac.get("via") == "across":
        args["via"] = "across"
    return args


def emergency_action(world, event: str = "new_sound") -> Dict[str, Any]:
    """Deterministic evacuation policy used on pre-emption and for every emergency event."""
    if world.guiding_active or event in ("owner_alerted", "handle_taken", "path_blocked"):
        return {"tool": "guide_owner_to", "args": _evac_args(world)}
    if event == "owner_no_response":
        # In a fire, two unanswered nudges are enough: lead them out anyway.
        if world.nudge_attempts < 2 and world.owner_recently_seen(4.0):
            return {"tool": "alert_owner", "args": {"sound": "smoke_alarm", "place": world.evac_target()}}
        return {"tool": "guide_owner_to", "args": _evac_args(world)}
    if world.owner_recently_seen(2.0):
        return {"tool": "alert_owner", "args": {"sound": "smoke_alarm", "place": world.evac_target()}}
    if event == "arrived_in_room":
        return {"tool": "look_around", "args": {}}
    return {"tool": "go_to_room", "args": {"room": world.best_search_room()}}


def validate_action(action: Dict[str, Any], world) -> Tuple[bool, str]:
    """Reject malformed or unsafe tool calls coming from the LLM."""
    tool = action.get("tool")
    args = action.get("args") or {}
    if tool not in TOOLS:
        return False, f"unknown tool {tool!r}"
    if tool == "get_state" and world.mission is not None:
        return False, "mission active: must act, state is already in the prompt"
    if tool == "stay_with_owner":  # #20: owner not responding, emergency contact alerted
        if world.is_emergency():
            return False, "fire: lead the owner out, never just stay"
        if world.owner_pos is None:
            return False, "owner location unknown"
    if tool == "go_to_room" and args.get("room") not in ROOMS:
        return False, f"bad room {args.get('room')!r}"
    if tool == "guide_owner_to":
        target = args.get("target")
        if target not in ROOMS and target not in LANDMARKS:
            return False, f"bad target {target!r}"
        if world.is_emergency():
            safe = [t for t, d in ((world.evac or {}).get("details") or {}).items() if not d.get("unavailable")]
            if target not in (safe or [world.evac_target()]):
                return False, f"emergency: {target} is not a safe exit (safe: {safe})"
            # The LLM may pick any usable exit, but never one whose route passes clearly closer to
            # the fire than the planner's choice (seen in rehearsal: Gemma picked the terrace exit
            # "much closer to the fire" over the planner's front door).
            details = (world.evac or {}).get("details") or {}
            best = (world.evac or {}).get("target")
            d, b = details.get(target) or {}, details.get(best) or {}
            if (target != best and d.get("fire_dist") is not None and b.get("fire_dist") is not None
                    and d["fire_dist"] < b["fire_dist"] - FIRE_MARGIN_M):
                return False, (f"emergency: the route to {target} passes {d['fire_dist']:.1f} m from the fire; "
                               f"{best} keeps {b['fire_dist']:.1f} m")
        if args.get("via") == "across":
            # Breaking an owner's place rule is only ever allowed in an emergency, and only on an
            # exit route the planner found to be much shorter that way.
            if not world.is_emergency():
                return False, "owner's place rules may only be broken in an emergency"
            if "cross_length" not in ((world.evac or {}).get("details") or {}).get(target, {}):
                return False, f"no shorter route across the no-go area to {target}"
        if not (world.guiding_active or world.dist_to_owner() < 1.6):
            return False, "owner not at the handle"
    if tool == "alert_owner":
        if args.get("sound") not in SOUND_SOURCES and args.get("sound") != "request":
            return False, f"bad sound {args.get('sound')!r}"
        if not world.owner_recently_seen(8.0):
            return False, "owner location unknown"
    return True, "ok"


def filter_cmd(vx: float, vy: float, wz: float, world, lidar: Sequence[float],
               allow_contact: bool = False) -> Tuple[float, float, float]:
    """Apply the proximity speed cap and the LiDAR bumper to a velocity command."""
    if world.dist_to_owner() <= OWNER_CAP_RADIUS or world.guiding_active:
        speed = math.hypot(vx, vy)
        if speed > OWNER_SPEED_CAP:
            vx, vy = vx * OWNER_SPEED_CAP / speed, vy * OWNER_SPEED_CAP / speed
    if not allow_contact and vx > 0.0 and lidar:
        n = len(lidar)
        front = min(lidar[n // 2 - 3: n // 2 + 3])
        if front < BUMPER_RANGE:
            vx = 0.0
    return vx, vy, wz
