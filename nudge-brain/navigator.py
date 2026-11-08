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
  - Getting stuck (#19): if the route stops getting shorter while the robot is trying to move
    (something not on the floor plan in the way, e.g. a chair beside an archway), recover in steps:
      1. back off + turn toward the freer LiDAR side; an unknown obstacle in front (not on the
         floor-plan map, not the owner) is remembered as a temporary no-go box and the route is
         re-planned around it;
      2. stuck again: mark whatever is in front, back off, re-plan;
      3. stuck a third time on the same goal: report status "stuck" so the skill hands the
         decision back to the agent (another route / room / exit).
    The owner is never treated as an obstacle (back off and wait), and while guiding the window
    is longer and the back-off gentler: the owner sets the pace.
"""

from __future__ import annotations

import math
import os
import time
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

# Getting stuck (#19)
STUCK_WINDOW_S = 4.0           # route not shorter by STUCK_MIN_PROGRESS_M for this long = stuck
STUCK_WINDOW_GUIDING_S = 7.0   # leading the owner: they set the pace
STUCK_MIN_PROGRESS_M = 0.15
STUCK_GIVE_UP = 3              # third time stuck on the same goal -> status "stuck" (agent decides)
BACKOFF_S = 1.4
BACKOFF_SPEED = 0.22
BACKOFF_SPEED_GUIDING = 0.08   # the owner holds the handle behind us: barely step back
TEMP_OBSTACLE_S = 60.0         # remembered obstacles expire (someone may move the chair back)
TEMP_OBSTACLE_HALF = 0.30      # half-size of the no-go box around an obstacle hit (chair r = 0.22)
OBSTACLE_LOOK_DEG = 40.0       # LiDAR cone searched for the thing in the way
OBSTACLE_LOOK_M = 1.0
STATIC_HIT_CLEARANCE = 0.12    # a hit this close to the floor-plan map is a known wall/furniture
OWNER_HIT_RADIUS = 0.5


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
        self.static_map = load_or_build(verbose=True)
        # Option A (NUDGE_ROBOT_MAP=1, default off): plan on the robot's own explored map + live
        # changes (robot_map_nav.py). Off -> exactly the static floor plan, as before.
        self.robot_map = None
        if os.environ.get("NUDGE_ROBOT_MAP", "0") == "1":
            self.robot_map = _load_robot_map()
        if self.robot_map is not None:
            self.planner = GridPlanner(self.robot_map.navmap())
        else:
            self.planner = GridPlanner(self.static_map)
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
        # Getting stuck (#19): progress tracker, recovery manoeuvre, remembered obstacles.
        self.clock = time.monotonic   # tests may swap in sim time
        self.stuck_level = 0         # times stuck on the current goal
        self.stuck_total = 0          # times stuck since start (dashboard / tests)
        self._prog_best = math.inf    # shortest remaining route length seen in this window
        self._prog_t = 0.0            # when it last got STUCK_MIN_PROGRESS_M shorter
        self._last_step_t = 0.0
        self._recover_until = 0.0
        self._recover_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._replan_after_recover = False
        self.temp_obstacles: List[Tuple[float, float, float, float, float]] = []  # x0, y0, x1, y1, expires
        self._news: List[str] = []    # log lines for the skill (it owns the world / log)

    # ------------------------------------------------------------------ planning
    def set_goal(self, goal: Point, robot_xy: Point, blocked: Set[str], final_tol: float = 0.25,
                 cross_areas: bool = False) -> bool:
        self.goal = goal
        self.final_tol = final_tol
        self.blocked_doors = set(blocked)
        self.cross_areas = cross_areas
        self._block_counts.clear()
        self.stuck_level = 0
        self._recover_until = 0.0
        self._replan_after_recover = False
        return self._plan(robot_xy)

    def clear(self) -> None:
        self.goal = None
        self.waypoints = []

    def obstacle_rects(self) -> List[Tuple[float, float, float, float]]:
        """Obstacles found by getting stuck (not on the floor plan), until they expire."""
        now = self.clock()
        self.temp_obstacles = [o for o in self.temp_obstacles if o[4] > now]
        return [o[:4] for o in self.temp_obstacles]

    def pop_news(self) -> List[str]:
        news, self._news = self._news, []
        return news

    def _plan(self, robot_xy: Point) -> bool:
        assert self.goal is not None
        self._prog_best, self._prog_t = math.inf, self.clock()  # new route: new progress window
        avoid, cost = (self.avoid_provider() if self.avoid_provider else ([], None))
        avoid = list(avoid)
        self.rule_note = ""
        if avoid and self.cross_areas:
            avoid, self.rule_note = [], "crossing the owner's no-go area (emergency decision)"
        obstacles = self.obstacle_rects()
        wps = None
        if obstacles:
            # Something physical is in the way: always a hard no-go, even in an emergency (where the
            # owner's areas are only a cost). Planner takes one cost for all boxes, so: owner areas
            # hard too if a route exists, else ignore the owner's areas rather than the obstacle.
            wps = self.planner.plan(robot_xy, self.goal, self.blocked_doors, avoid + obstacles, None)
            if wps is None and avoid:
                wps = self.planner.plan(robot_xy, self.goal, self.blocked_doors, obstacles, None)
                self.rule_note = "obstacle in the way: route crosses the owner's no-go area"
            if wps is None:
                self._news.append("no route around the obstacle - forgetting it")
                self.temp_obstacles = []
        if wps is None:
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
        if self.robot_map is not None:
            self._robot_map_update(pose, lidar, ignore)
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

    def _robot_map_update(self, pose, lidar, owner_xy) -> None:
        """NUDGE_ROBOT_MAP=1 only: keep the robot's map live; on a new / removed obstacle rebuild the
        planner in place (the evacuation planner shares it) and re-plan the current route."""
        rm = self.robot_map
        if not rm.observe(pose, lidar, owner_xy) and not rm.pending:
            return
        rm.pending = True
        if time.monotonic() - rm.last_rebuild < 2.0:
            return
        rm.pending = False
        rm.last_rebuild = time.monotonic()
        self.planner.__dict__.update(GridPlanner(rm.navmap()).__dict__)
        desc = ", ".join(f"({c['x']:.1f}, {c['y']:.1f})" for c in rm.changes()) or "none"
        print(f"[nav] ROBOT MAP: obstacles not in the saved map: {desc}" +
              (" -> re-planned route" if self.goal is not None else ""), flush=True)
        self.map_note = f"robot map changes: {desc}"
        if self.goal is not None:
            self._plan((pose[0], pose[1]))

    # ------------------------------------------------------------------ control
    # ------------------------------------------------------------------ getting stuck (#19)
    def _remaining(self, x: float, y: float) -> float:
        pts = [(x, y)] + [(w[0], w[1]) for w in self.waypoints]
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    def _front_obstacle(self, pose: Tuple[float, float, float], lidar: Sequence[float],
                        ignore: Optional[Point]) -> Optional[Tuple[float, float, str]]:
        """What is in the way, from the LiDAR hits ahead: (x, y, kind), kind = unknown (not on the floor
        plan, e.g. a moved chair: the new thing is what blocks us, so it wins), else owner, else static
        (a known wall / furniture: the robot is wedged by the controller, not by something new)."""
        x, y, yaw = pose
        best: Dict[str, Tuple[float, float, float]] = {}
        for ang, r in zip(LIDAR_ANGLES, lidar):
            if abs(ang) > math.radians(OBSTACLE_LOOK_DEG) or r >= OBSTACLE_LOOK_M:
                continue
            hx, hy = x + r * math.cos(yaw + ang), y + r * math.sin(yaw + ang)
            if ignore is not None and math.hypot(hx - ignore[0], hy - ignore[1]) < OWNER_HIT_RADIUS:
                kind = "owner"
            elif self.static_map.clearance(hx, hy) < STATIC_HIT_CLEARANCE:
                kind = "static"  # a LiDAR hit sits ON the surface: a known wall there has ~0 clearance
            else:
                kind = "unknown"
            if kind not in best or r < best[kind][0]:
                best[kind] = (r, hx, hy)
        for kind in ("unknown", "owner", "static"):
            if kind in best:
                return best[kind][1], best[kind][2], kind
        return None

    def _on_stuck(self, pose: Tuple[float, float, float], lidar: Sequence[float], guiding: bool,
                  ignore: Optional[Point], now: float) -> bool:
        """Start a recovery manoeuvre. Returns True if it is time to give up on this goal."""
        x, y, _ = pose
        what = self._front_obstacle(pose, lidar, ignore)
        kind = what[2] if what else "nothing visible"
        if kind != "owner":  # waiting for the owner to step aside is not a failed route
            self.stuck_level += 1
            self.stuck_total += 1
        if self.stuck_level >= STUCK_GIVE_UP:
            self._news.append(f"STUCK x{self.stuck_level} near ({x:.1f}, {y:.1f}): giving up on this route")
            self._prog_best, self._prog_t = math.inf, now
            return True
        left = sum(r for a, r in zip(LIDAR_ANGLES, lidar) if a > math.radians(20))
        right = sum(r for a, r in zip(LIDAR_ANGLES, lidar) if a < -math.radians(20))
        turn = 0.6 if left >= right else -0.6
        back = BACKOFF_SPEED_GUIDING if guiding else BACKOFF_SPEED
        self._recover_cmd = (-back, 0.0, turn) if kind != "owner" else (-back, 0.0, 0.0)
        self._recover_until = now + BACKOFF_S
        note = "owner in the way: backing off and waiting" if kind == "owner" else "backing off"
        if what and (kind == "unknown" or (self.stuck_level >= 2 and kind == "static")):
            hx, hy, _ = what
            h = TEMP_OBSTACLE_HALF
            self.temp_obstacles.append((hx - h, hy - h, hx + h, hy + h, now + TEMP_OBSTACLE_S))
            note += (f", obstacle at ({hx:.1f}, {hy:.1f}) not on the map -> re-planning around it" if kind == "unknown"
                     else f", avoiding the tight spot at ({hx:.1f}, {hy:.1f}) -> re-planning")
            self._replan_after_recover = True
        elif self.stuck_level >= 2:
            note += ", re-planning"
            self._replan_after_recover = True
        self._news.append(f"STUCK near ({x:.1f}, {y:.1f}) [{kind}]: {note}")
        self._prog_best, self._prog_t = math.inf, self._recover_until
        return False

    def step(self, pose: Tuple[float, float, float], lidar: Sequence[float],
             max_speed: float, guiding: bool = False,
             ignore: Optional[Point] = None) -> Tuple[float, float, float, str]:
        """Return (vx, vy, wz, status) with status in {idle, moving, arrived, stuck}.
        `guiding`: the owner holds the handle (longer stuck window, gentle back-off).
        `ignore`: the owner's position (never an obstacle to mark or route around)."""
        if not self.waypoints:
            return 0.0, 0.0, 0.0, ("arrived" if self.goal is not None else "idle")
        x, y, yaw = pose
        now = self.clock()
        if now < self._recover_until:
            return (*self._recover_cmd, "moving")
        if self._replan_after_recover:
            self._replan_after_recover = False
            self._plan((x, y))
            if not self.waypoints:
                return 0.0, 0.0, 0.0, "arrived"
        if now - self._last_step_t > 0.5:  # the skill paused us (e.g. owner let go): new window
            self._prog_best, self._prog_t = math.inf, now
        self._last_step_t = now
        rem = self._remaining(x, y)
        if rem < self._prog_best - STUCK_MIN_PROGRESS_M or self._prog_best == math.inf:
            self._prog_best, self._prog_t = rem, now
        elif now - self._prog_t > (STUCK_WINDOW_GUIDING_S if guiding else STUCK_WINDOW_S):
            if self._on_stuck(pose, lidar, guiding, ignore, now):
                return 0.0, 0.0, 0.0, "stuck"
            return (*self._recover_cmd, "moving")
        return self._drive(pose, lidar, max_speed)

    def _drive(self, pose: Tuple[float, float, float], lidar: Sequence[float],
               max_speed: float) -> Tuple[float, float, float, str]:
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
            return self._drive(pose, lidar, max_speed)

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


def _load_robot_map():
    """NUDGE_ROBOT_MAP=1: the robot's own saved map, or None (falls back to the static floor plan)."""
    try:
        from robot_map_nav import MAP_FILE, RobotMapNav
        from sim.house_layout import LANDMARKS
        rm = RobotMapNav(MAP_FILE)
        rm.pending = False
        places = rm.register_places(LANDMARKS)
        print(f"[nav] NUDGE_ROBOT_MAP=1: planning on the robot's own map {MAP_FILE.name} "
              f"({rm.grid.known_frac():.0%} of the grid known); places from objects: {places or 'none'}", flush=True)
        return rm
    except Exception as e:  # missing / unreadable map: keep the demo path working
        print(f"[nav] NUDGE_ROBOT_MAP=1 but no usable robot map ({e}); using the static floor plan", flush=True)
        return None
