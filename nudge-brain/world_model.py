#!/usr/bin/env python3
"""
Milestone 7a: World Model (`nudge-brain/world_model.py`).
Single shared belief state for the brain:
  - Robot pose / room / LiDAR (from sim `state` telemetry)
  - Owner belief (last seen position, room, visibility) from vision or the guide handle
  - Latest hearing result and the active mission (doorbell / knock / smoke_alarm)
  - Rooms already searched, doorways currently believed blocked, and an event log
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from sim.legacy_box_house import (
    ROBOT_START_POSE,
    ROOM_GRAPH,
    ROOM_ORDER,
    ROOMS,
    SOUND_SOURCES,
    get_room_at,
)

# Prior knowledge: where the owner usually sits before the robot has seen them.
OWNER_PRIOR_ROOM = "living_room"


def room_hops(start: str, goal: str, blocked_doors: Optional[Set[str]] = None) -> int:
    """Breadth-first hop count between rooms over the doorway graph (99 if unreachable)."""
    if start == goal:
        return 0
    blocked = blocked_doors or set()
    frontier = [start]
    seen = {start}
    hops = 0
    while frontier:
        hops += 1
        nxt: List[str] = []
        for room in frontier:
            for (a, b), door in ROOM_GRAPH.items():
                if a != room or b in seen or door in blocked:
                    continue
                if b == goal:
                    return hops
                seen.add(b)
                nxt.append(b)
        frontier = nxt
    return 99


class WorldModel:
    """Tracks everything the brain believes about the robot, owner, sounds, and task."""

    def __init__(self) -> None:
        self.t_sim: float = 0.0
        self.robot_pose: Tuple[float, float, float] = ROBOT_START_POSE
        self.robot_room: str = get_room_at(ROBOT_START_POSE[0], ROBOT_START_POSE[1])
        self.lidar_ranges: List[float] = [4.0] * 36
        self.guiding_active: bool = False
        self.connected: bool = False
        self.last_state_wall: float = 0.0

        # Owner belief
        self.owner_pos: Optional[Tuple[float, float]] = None
        self.owner_room: str = OWNER_PRIOR_ROOM
        self.owner_visible: bool = False
        self.owner_last_seen_wall: float = -1e9
        self.owner_source: str = "prior"  # prior | vision | handle
        self.last_detection: Dict[str, Any] = {"found": False}

        # Hearing
        self.sound: Dict[str, Any] = {
            "label": "silence",
            "confidence": 0.0,
            "scores": {"doorbell": 0.0, "knock": 0.0, "smoke_alarm": 0.0},
        }

        # Mission / task
        self.mission: Optional[Dict[str, Any]] = None  # {"sound", "target", "started", "started_sim"}
        # Outcome of the last finished mission, kept until a new mission starts or the sim resets:
        # {"status": "complete", "sound", "target", "duration_s" (sim time)}
        self.last_result: Optional[Dict[str, Any]] = None
        self.phase: str = "idle"  # idle | searching | approaching | guiding | done
        self.searched_rooms: Set[str] = set()
        self.blocked_doors: Dict[str, float] = {}  # door name -> wall time marked
        self.nav_path: List[Tuple[float, float]] = []

        self.events: Deque[str] = deque(maxlen=10)

    # ------------------------------------------------------------------ updates
    def update_state(self, msg: Dict[str, Any]) -> None:
        pose = msg.get("robot_pose", list(self.robot_pose))
        self.t_sim = float(msg.get("t", self.t_sim))
        self.robot_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
        room = str(msg.get("robot_room", self.robot_room))
        self.robot_room = "bedroom" if room == "back_exit_outside" else room
        self.lidar_ranges = [float(r) for r in msg.get("lidar_ranges", self.lidar_ranges)]
        self.guiding_active = bool(msg.get("guiding_active", False))
        self.connected = True
        self.last_state_wall = time.monotonic()

        # While the owner holds the handle we know where they are without seeing them:
        # they walk at the handle point 0.5 m behind / 0.14 m left of the robot.
        if self.guiding_active:
            x, y, yaw = self.robot_pose
            c, s = math.cos(yaw), math.sin(yaw)
            self._set_owner((x - 0.50 * c - 0.14 * s, y - 0.50 * s + 0.14 * c), "handle")

    def update_owner_detection(self, det: Dict[str, Any]) -> None:
        self.last_detection = det
        self.owner_visible = bool(det.get("found"))
        if self.owner_visible and det.get("world_xy") is not None:
            wx, wy = det["world_xy"]
            self._set_owner((float(wx), float(wy)), "vision")

    def _set_owner(self, xy: Tuple[float, float], source: str) -> None:
        self.owner_pos = xy
        room = get_room_at(xy[0], xy[1])
        if room in ROOMS:
            self.owner_room = room
        self.owner_last_seen_wall = time.monotonic()
        self.owner_source = source
        # Once the owner is located, the search bookkeeping is no longer needed.
        self.searched_rooms.clear()

    def update_sound(self, result: Dict[str, Any]) -> None:
        self.sound = result

    # ------------------------------------------------------------------ queries
    def owner_recently_seen(self, within_s: float) -> bool:
        return (time.monotonic() - self.owner_last_seen_wall) <= within_s

    def dist_to_owner(self) -> float:
        if self.owner_pos is None:
            return float("inf")
        return math.hypot(self.robot_pose[0] - self.owner_pos[0], self.robot_pose[1] - self.owner_pos[1])

    def is_emergency(self) -> bool:
        return self.mission is not None and self.mission["sound"] == "smoke_alarm"

    def active_blocked_doors(self, ttl_s: float = 30.0) -> Set[str]:
        now = time.monotonic()
        for door in [d for d, t in self.blocked_doors.items() if now - t > ttl_s]:
            del self.blocked_doors[door]
            self.log(f"Doorway {door} re-opened for planning")
        return set(self.blocked_doors)

    def best_search_room(self) -> str:
        """Owner's believed room first, then the nearest unsearched room."""
        if self.owner_room not in self.searched_rooms:
            return self.owner_room
        candidates = [r for r in ROOM_ORDER if r not in self.searched_rooms]
        if not candidates:
            # Searched everywhere: start again, skipping the room we're standing in.
            self.searched_rooms = {self.robot_room}
            candidates = [r for r in ROOM_ORDER if r != self.robot_room]
        blocked = self.active_blocked_doors()
        start_idx = ROOM_ORDER.index(self.robot_room) if self.robot_room in ROOM_ORDER else 0
        return min(
            candidates,
            key=lambda r: (
                room_hops(self.robot_room, r, blocked),
                (ROOM_ORDER.index(r) - start_idx) % len(ROOM_ORDER),
            ),
        )

    # ------------------------------------------------------------------ mission
    def start_mission(self, sound: str) -> None:
        target = str(SOUND_SOURCES[sound]["target_landmark"])
        self.mission = {"sound": sound, "target": target, "started": time.monotonic(),
                        "started_sim": self.t_sim}
        self.last_result = None
        self.phase = "searching"
        self.searched_rooms.clear()
        self.log(f"MISSION {sound.upper()} -> owner to {target}")

    def finish_mission(self) -> None:
        if self.mission:
            duration = self.t_sim - float(self.mission.get("started_sim", self.t_sim))
            self.last_result = {"status": "complete", "sound": self.mission["sound"],
                                "target": self.mission["target"], "duration_s": round(duration, 1)}
            self.log(f"Mission {self.mission['sound']} complete ({duration:.1f} s sim time)")
        self.mission = None
        self.phase = "idle"

    def mission_status(self) -> Tuple[str, str]:
        """("idle" | "active" | "complete", human-readable detail) for the sim overlay & dashboard."""
        if self.mission is not None:
            return "active", f"{self.mission['sound'].upper()} -> owner to {self.mission['target']}"
        r = self.last_result
        if r is not None:
            return "complete", (f"{r['sound'].upper()}: owner delivered to {r['target']} "
                                f"in {r['duration_s']:.1f} s")
        return "idle", ""

    def log(self, text: str) -> None:
        stamp = f"{self.t_sim:6.1f}s"
        self.events.appendleft(f"{stamp}  {text}")
        print(f"[brain {stamp}] {text}", flush=True)

    def snapshot(self) -> Dict[str, Any]:
        """Compact state dict for the agent's `get_state()` tool / LLM prompt."""
        return {
            "robot_room": self.robot_room,
            "robot_pose": [round(v, 2) for v in self.robot_pose],
            "guiding_active": self.guiding_active,
            "owner_believed_room": self.owner_room,
            "owner_visible_now": self.owner_visible,
            "owner_seen_last_5s": self.owner_recently_seen(5.0),
            "owner_distance_m": None if self.owner_pos is None else round(self.dist_to_owner(), 2),
            "sound": {"label": self.sound.get("label"), "confidence": self.sound.get("confidence")},
            "mission": None if self.mission is None else {
                "sound": self.mission["sound"], "deliver_owner_to": self.mission["target"]},
            "phase": self.phase,
            "searched_rooms": sorted(self.searched_rooms),
            "blocked_doors": sorted(self.blocked_doors),
        }
