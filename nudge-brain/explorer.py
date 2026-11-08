"""Option A step 4: frontier exploration on the robot's own occupancy grid (`robot_map.OccupancyGrid`).

A frontier is a known-free cell next to an unknown one. The explorer repeatedly drives to the nearest
reachable frontier cluster (path cost on the robot's own map: unknown counts as an obstacle, so it only
drives through floor it has seen), turns to face the frontier there (the LiDAR only covers the front
180 deg; one full spin at the start), and stops
when no reachable frontier is left, when coverage stops growing, or at the caller's time cap.

The only outside knowledge is `geofence`, a rectangle the installer gives ("the ground floor is inside
this box"), so it does not wander into the garden through an open door. Ground truth is never read.

    ex = FrontierExplorer(geofence=(x0, y0, x1, y1))
    vx, vy, wz, status = ex.step(grid, pose, lidar)        # status: spin | goto | done
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from sim.nav_map import GridPlanner, NavMap

LIDAR_ANGLES = [-math.pi / 2 + i * math.pi / 35 for i in range(36)]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class FrontierExplorer:
    def __init__(self, geofence: Optional[Tuple[float, float, float, float]] = None, min_cluster: int = 10,
                 speed: float = 0.8, spin_rate: float = 1.2, replan_s: float = 4.0,
                 stall_s: float = 90.0, stall_gain: float = 0.004) -> None:
        self.geofence = geofence
        self.min_cluster = min_cluster
        self.speed = speed
        self.spin_rate = spin_rate
        self.replan_s = replan_s
        self.stall_s, self.stall_gain = stall_s, stall_gain
        self.mode = "spin"
        self.spin_left = 2 * math.pi
        self.last_yaw: Optional[float] = None
        self.waypoints: List[Tuple[float, float]] = []
        self.goal: Optional[Tuple[float, float]] = None
        self.look_at: Optional[Tuple[float, float]] = None   # centre of the goal's frontier
        self.blacklist: List[Tuple[float, float]] = []
        self.t_plan = 0.0
        self.progress = (0.0, None)          # (time, xy) of the last real movement
        self.cov_hist: List[Tuple[float, float]] = []
        self.goals_visited = 0
        self.reason = ""

    # ---------------------------------------------------------------- planning
    def _navmap(self, grid) -> NavMap:
        occ = grid.nav_occupancy(unknown_is_occupied=True)
        if self.geofence is not None:
            x0, y0, x1, y1 = self.geofence
            xs = grid.origin[0] + np.arange(grid.w) * grid.res
            ys = grid.origin[1] + np.arange(grid.h) * grid.res
            inside = ((ys[:, None] >= y0) & (ys[:, None] <= y1)) & ((xs[None, :] >= x0) & (xs[None, :] <= x1))
            occ = occ | ~inside
        return NavMap(occ, grid.res, grid.bounds())

    def frontiers(self, grid) -> List[Tuple[np.ndarray, int]]:
        """[(cells (N,2) as (row, col), size)] of frontier clusters inside the geofence."""
        free = grid.free()
        unknown = ~(free | grid.occupied())
        nb = ndimage.binary_dilation(unknown, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
        fr = free & nb
        if self.geofence is not None:
            x0, y0, x1, y1 = self.geofence
            c0, r0 = grid._cell(x0, y0)
            c1, r1 = grid._cell(x1, y1)
            m = np.zeros_like(fr)
            m[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = True
            fr &= m
        lab, n = ndimage.label(fr, structure=np.ones((3, 3)))
        out = []
        for k, sl in enumerate(ndimage.find_objects(lab), start=1):
            cells = np.argwhere(lab[sl] == k) + [sl[0].start, sl[1].start]
            if len(cells) >= self.min_cluster:
                out.append((cells, len(cells)))
        return out

    def _choose(self, grid, pose) -> Optional[List[Tuple[float, float]]]:
        from scipy.sparse.csgraph import dijkstra
        nav = self._navmap(grid)
        planner = GridPlanner(nav)
        graph, free, clear, near = planner._grid(frozenset())
        h, w = free.shape
        s = planner._snap(planner.cell(pose[0], pose[1]), free, near)
        dist = dijkstra(graph, directed=False, indices=s[0] * w + s[1])
        best, best_cost, best_c = None, np.inf, None
        for cells, size in self.frontiers(grid):
            # frontier cells -> planner cells -> nearest drivable cell
            wx = grid.origin[0] + cells[:, 1] * grid.res
            wy = grid.origin[1] + cells[:, 0] * grid.res
            pr = np.clip(((wy - planner.y0) / planner.res).astype(int), 0, h - 1)
            pc = np.clip(((wx - planner.x0) / planner.res).astype(int), 0, w - 1)
            sr, sc = near[0][pr, pc], near[1][pr, pc]
            # snapped cell must be close to the frontier (else it is behind furniture / a wall)
            gx, gy = planner.x0 + (sc + 0.5) * planner.res, planner.y0 + (sr + 0.5) * planner.res
            close = np.hypot(gx - wx, gy - wy) < 0.9
            if not close.any():
                continue
            d = dist[(sr * w + sc)[close]]
            i = int(np.argmin(d))
            if not np.isfinite(d[i]):
                continue
            goal = (float(gx[close][i]), float(gy[close][i]))
            if any(math.hypot(goal[0] - bx, goal[1] - by) < 1.0 for bx, by in self.blacklist):
                continue
            cost = float(d[i]) - 0.05 * min(size, 120)    # bigger frontiers (more to see) are more attractive
            if cost < best_cost:
                best, best_cost, best_c = goal, cost, (float(wx.mean()), float(wy.mean()))
        if best is None:
            return None
        wps = planner.plan((pose[0], pose[1]), best)
        if not wps:
            self.blacklist.append(best)
            return None
        self.goal, self.look_at = best, best_c
        return [(p[0], p[1]) for p in wps]

    # ---------------------------------------------------------------- control
    def step(self, grid, pose, lidar: Sequence[float], now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        x, y, yaw = pose
        cov = float(grid.known().mean())
        self.cov_hist.append((now, cov))
        old = [c for t, c in self.cov_hist if now - t >= self.stall_s]
        if old and cov - old[-1] < self.stall_gain:
            self.mode, self.reason = "done", f"coverage stopped growing ({cov:.1%})"
        self.cov_hist = [(t, c) for t, c in self.cov_hist if now - t < self.stall_s + 5]
        if self.mode == "done":
            return 0.0, 0.0, 0.0, "done"

        if self.mode == "look":                      # face the frontier we came to see, then re-plan
            err = wrap(math.atan2(self.look_at[1] - y, self.look_at[0] - x) - yaw)
            if abs(err) > 0.25 and math.hypot(self.look_at[0] - x, self.look_at[1] - y) > 0.3:
                return 0.0, 0.0, max(-1.4, min(1.4, 2.5 * err)), "spin"
            self.mode = "plan"
        if self.mode == "spin":
            if self.last_yaw is not None:
                self.spin_left -= abs(wrap(yaw - self.last_yaw))
            self.last_yaw = yaw
            if self.spin_left > 0:
                return 0.0, 0.0, self.spin_rate, "spin"
            self.last_yaw = None
            self.mode = "plan"

        if self.mode == "plan" or (self.mode == "goto" and now - self.t_plan > self.replan_s):
            prev_goal = self.goal
            wps = self._choose(grid, pose)
            self.t_plan = now
            if wps is None:
                self.mode, self.reason = "done", "no reachable frontier left"
                return 0.0, 0.0, 0.0, "done"
            if self.mode == "plan" or prev_goal is None or math.hypot(self.goal[0] - prev_goal[0],
                                                                      self.goal[1] - prev_goal[1]) > 0.3:
                self.progress = (now, (x, y))
            self.waypoints = wps
            self.mode = "goto"

        # follow waypoints
        while len(self.waypoints) > 1 and math.hypot(self.waypoints[0][0] - x, self.waypoints[0][1] - y) < 0.35:
            self.waypoints.pop(0)
        tx, ty = self.waypoints[0]
        d = math.hypot(tx - x, ty - y)
        if len(self.waypoints) == 1 and d < 0.3:
            self.goals_visited += 1
            self.blacklist.append(self.goal)        # been there: do not pick the same spot again
            self.mode = "look" if self.look_at is not None else "plan"
            return 0.0, 0.0, 0.0, "spin"
        t0, p0 = self.progress
        if p0 is not None and math.hypot(x - p0[0], y - p0[1]) > 0.3:
            self.progress = (now, (x, y))
        elif now - t0 > 8.0:                         # stuck: give up on this goal
            self.blacklist.append(self.goal)
            self.mode = "plan"
            return 0.0, 0.0, 0.0, "goto"
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        wz = max(-1.6, min(1.6, 2.2 * err))
        vx = self.speed * max(0.0, math.cos(err)) ** 3 if abs(err) < 1.0 else 0.0
        if len(self.waypoints) == 1:
            vx = min(vx, 0.8 * d + 0.1)
        front = min(r for a, r in zip(LIDAR_ANGLES, lidar) if abs(a) <= math.radians(20))
        if front < 0.45:
            vx = min(vx, max(0.0, front - 0.33))
        return vx, 0.0, wz, "goto"


class ViewpointScan:
    """After exploration: look around from the middle of every open space on the robot's own map, so the
    head camera sees the furniture from 2-5 m (YOLO needs that; up close a bed is just a beige wall).
    Viewpoints = cells with the most clearance, at least `spacing` m apart (greedy); visited nearest-first,
    with one slow full turn at each."""

    def __init__(self, grid, start_xy, spacing: float = 4.0, min_clear: float = 1.0, spin_rate: float = 0.8,
                 geofence=None) -> None:
        self.ex = FrontierExplorer(geofence=geofence)
        nav = self.ex._navmap(grid)
        self.planner = GridPlanner(nav)
        dist = nav.dist
        order = np.argsort(dist.ravel())[::-1]
        pts: List[Tuple[float, float]] = []
        for idx in order[:200000]:
            r, c = divmod(int(idx), dist.shape[1])
            if dist[r, c] < min_clear:
                break
            p = nav.world(r, c)
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= spacing for q in pts):
                pts.append(p)
        # nearest-neighbour tour from the start
        tour, cur = [], start_xy
        while pts:
            i = min(range(len(pts)), key=lambda k: math.hypot(pts[k][0] - cur[0], pts[k][1] - cur[1]))
            cur = pts.pop(i)
            tour.append(cur)
        self.viewpoints = tour
        self.total = len(tour)
        self.spin_rate = spin_rate
        self.mode, self.wps, self.spin_left, self.last_yaw, self.t_leg = "next", [], 0.0, None, 0.0
        self.reason = ""
        self.goals_visited = 0

    def step(self, grid, pose, lidar, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        x, y, yaw = pose
        if self.mode == "spin":
            if self.last_yaw is not None:
                self.spin_left -= abs(wrap(yaw - self.last_yaw))
            self.last_yaw = yaw
            if self.spin_left > 0:
                return 0.0, 0.0, self.spin_rate, "spin"
            self.mode = "next"
        if self.mode == "next":
            if not self.viewpoints:
                self.reason = f"all {self.total} viewpoints visited"
                return 0.0, 0.0, 0.0, "done"
            goal = self.viewpoints.pop(0)
            plan = self.planner.plan((x, y), goal)
            if not plan:
                return 0.0, 0.0, 0.0, "goto"
            self.wps, self.mode, self.t_leg = [(p[0], p[1]) for p in plan], "goto", now
        while len(self.wps) > 1 and math.hypot(self.wps[0][0] - x, self.wps[0][1] - y) < 0.35:
            self.wps.pop(0)
        tx, ty = self.wps[0]
        d = math.hypot(tx - x, ty - y)
        if (len(self.wps) == 1 and d < 0.3) or now - self.t_leg > 40:
            self.goals_visited += 1
            self.mode, self.spin_left, self.last_yaw = "spin", 2 * math.pi, None
            return 0.0, 0.0, 0.0, "spin"
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        vx = 0.8 * max(0.0, math.cos(err)) ** 3 if abs(err) < 1.0 else 0.0
        if len(self.wps) == 1:
            vx = min(vx, 0.8 * d + 0.1)
        front = min(r for a, r in zip(LIDAR_ANGLES, lidar) if abs(a) <= math.radians(20))
        if front < 0.45:
            vx = min(vx, max(0.0, front - 0.33))
        return vx, 0.0, max(-1.6, min(1.6, 2.2 * err)), "goto"
