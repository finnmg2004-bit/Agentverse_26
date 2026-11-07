#!/usr/bin/env python3
"""
Milestone 7b: Plain-code Navigator (`nudge-brain/navigator.py`).
  - Path planning on the house's static floor-plan map (sim/nav_map.py GridPlanner:
    shortest path with a clearance cost, so routes go around furniture and through the
    middle of doorways), skipping doorways currently believed blocked. Every doorway the
    route crosses becomes a tagged waypoint.
  - Pure-pursuit heading controller producing body-frame (vx, vy, wz).
  - LiDAR obstacle skirt: sidesteps toward the freer side when something is ahead.
  - LiDAR doorway check: if obstacle hits land inside an upcoming doorway the door is
    marked blocked and the route is re-planned around it (e.g. the chair, hotkey [C]).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sim.house_layout import DOORWAYS, OUTSIDE, ROOM_GRAPH
from sim.nav_map import GridPlanner, load_or_build

Point = Tuple[float, float]

LIDAR_ANGLES = [-math.pi / 2 + i * math.pi / 35 for i in range(36)]
LIDAR_MAX = 4.0
DOOR_APPROACH = 0.6  # metres either side of the doorway centre
DOOR_BLOCK_RADIUS = 0.45  # LiDAR hits this close to a doorway centre = something in the gap
DOOR_BLOCK_TICKS = 4  # consecutive LiDAR scans before declaring a doorway blocked


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def room_path(start: str, goal: str, blocked: Set[str]) -> Optional[List[str]]:
    """BFS over rooms; returns [start, ..., goal] or None. OUTSIDE is never a shortcut."""
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
            if b == OUTSIDE and goal != OUTSIDE:
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
        self.planner = GridPlanner(load_or_build(verbose=True))
        self.goal: Optional[Point] = None
        self.final_tol: float = 0.25
        # Each waypoint: (x, y, door_name or None). door_name is set on the door-centre waypoint.
        self.waypoints: List[Tuple[float, float, Optional[str]]] = []
        self.blocked_doors: Set[str] = set()
        self._block_counts: Dict[str, int] = {}
        # Owner's no-go areas (rules.py): () -> (rects, cost|None). Set by main.py.
        self.avoid_provider = None
        self.cross_areas = False      # the agent decided to break a place rule (emergency only)
        self.rule_note: str = ""      # how the current route relates to the owner's rules

    # ------------------------------------------------------------------ planning
    def set_goal(self, goal: Point, robot_xy: Point, blocked: Set[str], final_tol: float = 0.25,
                 cross_areas: bool = False) -> bool:
        self.goal = goal
        self.final_tol = final_tol
        self.blocked_doors = set(blocked)
        self.cross_areas = cross_areas
        self._block_counts.clear()
        return self._plan(robot_xy)

    def clear(self) -> None:
        self.goal = None
        self.waypoints = []

    def _plan(self, robot_xy: Point) -> bool:
        assert self.goal is not None
        avoid, cost = (self.avoid_provider() if self.avoid_provider else ([], None))
        self.rule_note = ""
        if avoid and self.cross_areas:
            avoid, self.rule_note = [], "crossing the owner's no-go area (emergency decision)"
        wps = self.planner.plan(robot_xy, self.goal, self.blocked_doors, avoid, cost)
        if wps is None and avoid and cost is None:
            # No way around the owner's no-go area: cross as little of it as possible.
            wps = self.planner.plan(robot_xy, self.goal, self.blocked_doors, avoid, 6.0)
            self.rule_note = "no route around the owner's no-go area: crossing as little as possible"
        elif avoid and wps is not None:
            self.rule_note = "route keeps off the owner's no-go area" if cost is None else \
                "fire: no-go area allowed only if much shorter"
        if wps is None:
            # Every route is blocked: ignore blocks rather than freeze.
            wps = self.planner.plan(robot_xy, self.goal, ())
        if wps is None:
            wps = [(self.goal[0], self.goal[1], None)]
        # Drop leading points we are already standing on.
        while len(wps) > 1 and wps[0][2] is None and math.hypot(wps[0][0] - robot_xy[0], wps[0][1] - robot_xy[1]) < 0.3:
            wps.pop(0)
        self.waypoints = list(wps)
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
