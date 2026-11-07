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
WATCHDOG_S = 2.0
OWNER_SPEED_CAP = 0.40
OWNER_CAP_RADIUS = 1.0
BUMPER_RANGE = 0.30
EVACUATION_TARGET = "back_exit_outside"

TOOLS = ("get_state", "go_to_room", "look_around", "alert_owner", "offer_handle",
         "guide_owner_to", "investigate_sound")


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


def emergency_action(world, event: str = "new_sound") -> Dict[str, Any]:
    """Deterministic evacuation policy used on pre-emption and for every emergency event."""
    if world.guiding_active or event in ("owner_alerted", "handle_taken"):
        return {"tool": "guide_owner_to", "args": {"target": EVACUATION_TARGET}}
    if world.owner_recently_seen(2.0):
        return {"tool": "alert_owner", "args": {"sound": "smoke_alarm", "place": EVACUATION_TARGET}}
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
    if tool == "go_to_room" and args.get("room") not in ROOMS:
        return False, f"bad room {args.get('room')!r}"
    if tool == "guide_owner_to":
        target = args.get("target")
        if target not in ROOMS and target not in LANDMARKS:
            return False, f"bad target {target!r}"
        if world.is_emergency() and target != EVACUATION_TARGET:
            return False, "emergency: must evacuate to back_exit_outside"
        if not (world.guiding_active or world.dist_to_owner() < 1.6):
            return False, "owner not at the handle"
    if tool == "alert_owner":
        if args.get("sound") not in SOUND_SOURCES and args.get("sound") != "voice_request":
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
