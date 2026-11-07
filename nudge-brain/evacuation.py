#!/usr/bin/env python3
"""
Fire-aware evacuation planning (`nudge-brain/evacuation.py`).

Replaces the fixed "always the back door" rule with a decision:
  1. Fire location = where the robot's ears localised the smoke alarm (sound_localizer),
     falling back to the alarm's installed position until the estimate has enough evidence.
  2. Hazard doorways = every doorway of the fire room. The owner is never LED through them
     (unless they are already inside the fire room, where the nearest way out wins).
  3. Every exterior exit is scored: walking distance from the owner along the floor-plan
     planner + a penalty for passing near the fire. Exits in the fire room and exits LiDAR
     has seen blocked are excluded.
  4. Re-planned whenever a doorway or exit is found blocked mid-evacuation.
Escalation: if the owner hasn't been found within ESCALATE_AFTER_S of the alarm, the robot
raises an emergency-services call (simulated: dashboard + viewer overlay + log).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from sim.house_layout import (
    DOORWAY_INFO,
    EXIT_TARGETS,
    EXITS,
    LANDMARKS,
    SMOKE_EVACUATION_TARGET,
    SOUND_SOURCES,
    get_room_at,
)

FIRE_NEAR_M = 4.0          # paths closer than this to the fire are penalised
FIRE_PENALTY = 8.0         # metres of extra cost per metre inside FIRE_NEAR_M
ESCALATE_AFTER_S = 40.0
MIN_FIRE_MEAS = 4          # trust the ears' fire location only after this many windows...
MAX_FIRE_STD = 4.5         # ...and once it has narrowed down to this spread (m)


def hazard_doors(fire_room: Optional[str]) -> Set[str]:
    if not fire_room:
        return set()
    doors = {n for n, d in DOORWAY_INFO.items() if fire_room in (d["a"], d["b"])}
    doors |= {n for n, e in EXITS.items() if e["room"] == fire_room}
    return doors


def _path_len(start: Tuple[float, float], wps: List[Tuple[float, float, Any]]) -> float:
    pts = [start] + [(w[0], w[1]) for w in wps]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _min_dist_to(point: Tuple[float, float], start: Tuple[float, float], wps) -> float:
    pts = [start] + [(w[0], w[1]) for w in wps]
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / L2))
        best = min(best, math.hypot(ax + t * dx - point[0], ay + t * dy - point[1]))
    return best


class EvacuationPlanner:
    def __init__(self, planner) -> None:
        self.planner = planner  # sim.nav_map.GridPlanner (shared with the navigator)

    def fire_estimate(self, localizer) -> Dict[str, Any]:
        est = localizer.estimate("smoke_alarm") if localizer is not None else None
        if est is not None and est["n"] >= MIN_FIRE_MEAS and est["std"] <= MAX_FIRE_STD and est["room"]:
            return {"xy": est["xy"], "room": est["room"], "std": est["std"], "source": "heard"}
        sx, sy, _ = SOUND_SOURCES["smoke_alarm"]["pos"]
        return {"xy": (sx, sy), "room": get_room_at(sx, sy), "std": None, "source": "installed alarm"}

    def choose(self, start: Tuple[float, float], fire: Dict[str, Any], blocked: Set[str]) -> Dict[str, Any]:
        """Pick the safest exit for someone standing at `start`. Returns the decision + scores."""
        fire_xy, fire_room = fire["xy"], fire["room"]
        owner_in_fire_room = get_room_at(*start) == fire_room
        hazards = set() if owner_in_fire_room else hazard_doors(fire_room)
        scores: Dict[str, Optional[float]] = {}
        details: Dict[str, Dict[str, Any]] = {}  # per exit target: length / fire_dist, or why unusable
        for exit_name, target in EXIT_TARGETS.items():
            e = EXITS[exit_name]
            scores[target] = None
            if exit_name in blocked:
                details[target] = {"unavailable": "blocked (seen by LiDAR)"}
                continue
            if e["room"] == fire_room and not owner_in_fire_room:
                details[target] = {"unavailable": f"it is in the {fire_room}, where the fire is"}
                continue
            wps = self.planner.plan(start, LANDMARKS[target], blocked | hazards)
            if wps is None:
                details[target] = {"unavailable": "no route that avoids the fire"}
                continue
            near = _min_dist_to(fire_xy, start, wps)
            length = _path_len(start, wps)
            scores[target] = length + FIRE_PENALTY * max(0.0, FIRE_NEAR_M - near)
            details[target] = {"length": length, "fire_dist": near, "cost": scores[target]}
        valid = {k: v for k, v in scores.items() if v is not None}
        if valid:
            target = min(valid, key=valid.get)
            reason = f"shortest safe exit ({valid[target]:.0f} m)"
        else:
            target = SMOKE_EVACUATION_TARGET
            reason = "no safe route found: default exit"
        return {"target": target, "scores": scores, "details": details, "hazard_doors": hazards, "reason": reason}
