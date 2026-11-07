#!/usr/bin/env python3
"""
Milestone 7b: Plain-code Navigator (`nudge-brain/navigator.py`).
  - Topological room graph (from `sim/legacy_box_house.py` until ported) searched with BFS, skipping
    doorways currently believed blocked.
  - Each doorway becomes three waypoints (approach -> door centre -> exit) so the
    robot always crosses walls square-on.
  - Pure-pursuit heading controller producing body-frame (vx, vy, wz).
  - LiDAR obstacle skirt: sidesteps toward the freer side when something is ahead.
  - LiDAR doorway check: if obstacle hits land inside an upcoming doorway the door is
    marked blocked and the route is re-planned around it (e.g. the chair, hotkey [C]).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sim.legacy_box_house import DOORWAYS, LANDMARKS, ROOM_GRAPH, ROOMS, get_room_at

Point = Tuple[float, float]

LIDAR_ANGLES = [-math.pi / 2 + i * math.pi / 35 for i in range(36)]
LIDAR_MAX = 4.0
DOOR_APPROACH = 0.6  # metres either side of the doorway centre
DOOR_BLOCK_RADIUS = 0.45  # LiDAR hits this close to a doorway centre = something in the gap
DOOR_BLOCK_TICKS = 4  # consecutive LiDAR scans before declaring a doorway blocked


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def room_path(start: str, goal: str, blocked: Set[str]) -> Optional[List[str]]:
    """BFS over rooms; returns [start, ..., goal] or None."""
    if start == goal:
        return [start]
    prev: Dict[str, str] = {}
    q = deque([start])
    seen = {start}
    while q:
        room = q.popleft()
        for (a, b), door in ROOM_GRAPH.items():
            if a != room or b in seen or door in blocked:
                continue
            prev[b] = room
            if b == goal:
                path = [b]
                while path[-1] != start:
                    path.append(prev[path[-1]])
                return path[::-1]
            seen.add(b)
            q.append(b)
    return None


class Navigator:
    """Waypoint planner + follower with LiDAR obstacle handling."""

    def __init__(self) -> None:
        self.goal: Optional[Point] = None
        self.final_tol: float = 0.25
        # Each waypoint: (x, y, door_name or None). door_name is set on the door-centre waypoint.
        self.waypoints: List[Tuple[float, float, Optional[str]]] = []
        self.blocked_doors: Set[str] = set()
        self._block_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------ planning
    def set_goal(self, goal: Point, robot_xy: Point, blocked: Set[str], final_tol: float = 0.25) -> bool:
        self.goal = goal
        self.final_tol = final_tol
        self.blocked_doors = set(blocked)
        self._block_counts.clear()
        return self._plan(robot_xy)

    def clear(self) -> None:
        self.goal = None
        self.waypoints = []

    def _plan(self, robot_xy: Point) -> bool:
        assert self.goal is not None
        start_room = get_room_at(*robot_xy)
        if start_room == "back_exit_outside":
            start_room = "bedroom"
        goal_room = get_room_at(*self.goal)
        exiting = goal_room == "back_exit_outside"
        if exiting:
            goal_room = "bedroom"

        rooms = room_path(start_room, goal_room, self.blocked_doors)
        if rooms is None:
            # Every route is blocked: ignore blocks rather than freeze.
            rooms = room_path(start_room, goal_room, set()) or [start_room]

        wps: List[Tuple[float, float, Optional[str]]] = []
        for a, b in zip(rooms, rooms[1:]):
            door = ROOM_GRAPH[(a, b)]
            dx, dy = DOORWAYS[door]
            ax, ay = ROOMS[a]["center"]
            bx, by = ROOMS[b]["center"]
            if abs(dx) < 1e-6:  # door in the vertical wall x = 0 -> cross along x
                s = 1.0 if bx > ax else -1.0
                wps += [(dx - s * DOOR_APPROACH, dy, None), (dx, dy, door), (dx + s * DOOR_APPROACH, dy, None)]
            else:  # door in the horizontal wall y = 0 -> cross along y
                s = 1.0 if by > ay else -1.0
                wps += [(dx, dy - s * DOOR_APPROACH, None), (dx, dy, door), (dx, dy + s * DOOR_APPROACH, None)]

        if exiting:
            bdx, bdy = LANDMARKS["back_door"]
            wps += [(bdx, bdy - DOOR_APPROACH, None), (bdx, bdy, "back_door")]
        wps.append((self.goal[0], self.goal[1], None))

        # Drop leading approach points we are already past / standing on.
        while len(wps) > 1 and wps[0][2] is None and math.hypot(wps[0][0] - robot_xy[0], wps[0][1] - robot_xy[1]) < 0.3:
            wps.pop(0)
        self.waypoints = wps
        return True

    def path_points(self, robot_xy: Point) -> List[Point]:
        return [robot_xy] + [(w[0], w[1]) for w in self.waypoints]

    # ------------------------------------------------------------------ perception
    def check_doorways(self, pose: Tuple[float, float, float], lidar: Sequence[float],
                       ignore: Optional[Point] = None) -> Optional[str]:
        """
        Project LiDAR hits into the world. If hits land in the gap of a doorway still ahead
        on the route for several scans, mark it blocked, re-plan, and return the door name.
        """
        upcoming = [w[2] for w in self.waypoints if w[2] in DOORWAYS]
        if not upcoming:
            return None
        x, y, yaw = pose
        hits: List[Point] = []
        for ang, r in zip(LIDAR_ANGLES, lidar):
            if r >= LIDAR_MAX - 0.05:
                continue
            hx, hy = x + r * math.cos(yaw + ang), y + r * math.sin(yaw + ang)
            if ignore is not None and math.hypot(hx - ignore[0], hy - ignore[1]) < 0.45:
                continue  # that's the owner, not furniture
            hits.append((hx, hy))

        for door in upcoming:
            dx, dy = DOORWAYS[door]
            if math.hypot(dx - x, dy - y) < 0.35:
                continue  # already in the doorway
            blocked_now = any(math.hypot(hx - dx, hy - dy) < DOOR_BLOCK_RADIUS for hx, hy in hits)
            self._block_counts[door] = self._block_counts.get(door, 0) + 1 if blocked_now else 0
            if self._block_counts[door] >= DOOR_BLOCK_TICKS:
                self._block_counts[door] = 0
                self.blocked_doors.add(door)
                self._plan((x, y))
                return door
        return None

    # ------------------------------------------------------------------ control
    def step(self, pose: Tuple[float, float, float], lidar: Sequence[float],
             max_speed: float) -> Tuple[float, float, float, str]:
        """Return (vx, vy, wz, status) with status in {idle, moving, arrived}."""
        if not self.waypoints:
            return 0.0, 0.0, 0.0, ("arrived" if self.goal is not None else "idle")
        x, y, yaw = pose
        wx, wy, _ = self.waypoints[0]
        final = len(self.waypoints) == 1
        dist = math.hypot(wx - x, wy - y)
        tol = self.final_tol if final else 0.22
        if dist < tol:
            self.waypoints.pop(0)
            if not self.waypoints:
                return 0.0, 0.0, 0.0, "arrived"
            return self.step(pose, lidar, max_speed)

        err = wrap(math.atan2(wy - y, wx - x) - yaw)
        wz = max(-1.8, min(1.8, 2.4 * err))
        vx = max_speed * max(0.0, math.cos(err)) ** 3 if abs(err) < 1.0 else 0.0
        if final:
            vx = min(vx, 0.7 * dist + 0.08)
        vy = 0.0

        # LiDAR skirt: obstacle in the front cone that is closer than the waypoint.
        front = min(r for a, r in zip(LIDAR_ANGLES, lidar) if abs(a) <= math.radians(22))
        if front < 0.55 and front < dist - 0.15:
            left = sum(r for a, r in zip(LIDAR_ANGLES, lidar) if math.radians(30) <= a <= math.radians(90))
            right = sum(r for a, r in zip(LIDAR_ANGLES, lidar) if -math.radians(90) <= a <= -math.radians(30))
            vy = 0.22 if left >= right else -0.22
            vx = min(vx, max(0.0, front - 0.32))
        return vx, vy, wz, "moving"
