"""
Static 2D navigation map of the nerv-world house ground floor.

Built once from the compiled MuJoCo scene: every collidable house geom whose bounding box
overlaps the robot's height band is rasterized into an occupancy grid. Dynamic obstacles
(owner, chair) and the robot are excluded — they are handled by LiDAR / circle push-out.

Used by:
  * sim/locomotion.py   -> collision clamping (clearance field, 5 cm cells)
  * nudge-brain/navigator.py -> path planning (GridPlanner, 10 cm cells) — the brain's
    prior floor-plan map, like a real robot's saved map.

The map is cached at assets/maps/nerv_house_nav.npz (git-ignored) and rebuilt
automatically when the upstream scene changes. Rebuild manually: `pixi run build-map`.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402

MAP_VERSION = 1
MAP_RES = 0.05                                   # metres per cell
MAP_BOUNDS = (-18.5, -7.5, 18.5, 19.5)           # x0, y0, x1, y1 (house + outside margin)
Z_BAND = (0.10, 0.55)                            # obstacle if a geom overlaps this height band
MAP_PATH = PROJECT_ROOT / H.NAV_MAP_FILE

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def _scene_stamp() -> str:
    from sim.world import SCENE_XML, PROXY_SIZES
    h = hashlib.sha1(SCENE_XML.read_bytes())
    decor = (SCENE_XML.parent.parent / "decor" / "assets").exists()
    h.update(f"v{MAP_VERSION}|res{MAP_RES}|{MAP_BOUNDS}|{Z_BAND}|decor{decor}|{sorted(PROXY_SIZES)}".encode())
    return h.hexdigest()


def build_occupancy(model, data, res: float = MAP_RES, bounds=MAP_BOUNDS, z_band=Z_BAND) -> np.ndarray:
    """Rasterize static house obstacles into a bool grid [row = y, col = x]."""
    import cv2
    import mujoco
    from sim.world import BEACON_PREFIX, NUDGE_BODIES, geom_world_zrange

    x0, y0, x1, y1 = bounds
    w, h = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    occ = np.zeros((h, w), np.uint8)

    mujoco.mj_forward(model, data)
    excluded_roots = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("base",) + NUDGE_BODIES}
    excluded_roots.discard(-1)

    def excluded_body(b: int) -> bool:
        while b > 0:
            if b in excluded_roots:
                return True
            b = model.body_parentid[b]
        return False

    zr = geom_world_zrange(model, data)
    c_loc, h_loc = model.geom_aabb[:, :3], model.geom_aabb[:, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    shift = 4
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            continue                                   # purely visual
        if model.geom_group[i] in (2, 3):
            continue                                   # robot
        name = model.geom(i).name
        if name.startswith(BEACON_PREFIX) or excluded_body(int(model.geom_bodyid[i])):
            continue
        if zr[i, 1] < z_band[0] or zr[i, 0] > z_band[1]:
            continue
        corners = c_loc[i] + signs * h_loc[i]
        world = corners @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
        px = ((world[:, 0] - x0) / res * (1 << shift)).astype(np.int32)
        py = ((world[:, 1] - y0) / res * (1 << shift)).astype(np.int32)
        hull = cv2.convexHull(np.stack([px, py], axis=1))
        cv2.fillConvexPoly(occ, hull, 1, lineType=cv2.LINE_8, shift=shift)
    return occ.astype(bool)


class NavMap:
    """Occupancy grid + clearance field (metres to the nearest static obstacle)."""

    def __init__(self, occ: np.ndarray, res: float = MAP_RES, bounds=MAP_BOUNDS) -> None:
        self.occ = occ
        self.res = float(res)
        self.x0, self.y0 = float(bounds[0]), float(bounds[1])
        self.bounds = tuple(bounds)
        self.dist = ndimage.distance_transform_edt(~occ).astype(np.float32) * self.res

    @property
    def shape(self) -> Tuple[int, int]:
        return self.occ.shape

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int((y - self.y0) / self.res), int((x - self.x0) / self.res)

    def world(self, r: int, c: int) -> Point:
        return self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res

    def inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.occ.shape[0] and 0 <= c < self.occ.shape[1]

    def clearance(self, x: float, y: float) -> float:
        r, c = self.cell(x, y)
        return float(self.dist[r, c]) if self.inside(r, c) else 0.0

    # ---------------------------------------------------------------- caching
    def save(self, path: Path, stamp: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, occ=np.packbits(self.occ, axis=1), shape=np.array(self.occ.shape),
                            res=self.res, bounds=np.array(self.bounds), stamp=stamp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, stamp: Optional[str] = None) -> Optional["NavMap"]:
        try:
            z = np.load(path)
            if stamp is not None and str(z["stamp"]) != stamp:
                return None
            shape = tuple(int(v) for v in z["shape"])
            occ = np.unpackbits(z["occ"], axis=1)[:, : shape[1]].astype(bool)
            return cls(occ, float(z["res"]), tuple(float(v) for v in z["bounds"]))
        except Exception:
            return None


def build_map(model=None, data=None, verbose: bool = True) -> NavMap:
    if model is None:
        from sim.world import load_world
        model, data = load_world(verbose=False)
    nav = NavMap(build_occupancy(model, data))
    nav.save(MAP_PATH, _scene_stamp())
    if verbose:
        print(f"[nav_map] built {nav.shape[1]}x{nav.shape[0]} @ {MAP_RES} m -> {MAP_PATH.relative_to(PROJECT_ROOT)}")
    return nav


def load_or_build(model=None, data=None, verbose: bool = True) -> NavMap:
    """Load the cached map if it matches the current scene, else build (and cache) it."""
    stamp = _scene_stamp()
    nav = NavMap.load(MAP_PATH, stamp)
    if nav is not None:
        return nav
    return build_map(model, data, verbose)


# --------------------------------------------------------------------------- #
# Planning (brain side)
# --------------------------------------------------------------------------- #
class GridPlanner:
    """
    Shortest paths on a coarse copy of the map (default 10 cm cells), with a clearance
    cost that keeps paths centred in doorways and away from furniture.
    Blocked doorways are masked out as thin strips across the gap.
    """

    def __init__(self, nav: NavMap, res: float = 0.10, robot_radius: float = 0.30,
                 pref_clear: float = 0.65, clear_weight: float = 3.0) -> None:
        from scipy.sparse import csr_matrix  # noqa: F401  (import check)
        k = max(1, int(round(res / nav.res)))
        h, w = nav.occ.shape[0] // k, nav.occ.shape[1] // k
        occ = nav.occ[: h * k, : w * k].reshape(h, k, w, k).any(axis=(1, 3))
        self.res = nav.res * k
        self.x0, self.y0 = nav.x0, nav.y0
        self.occ = occ
        self.robot_radius = robot_radius
        self.pref_clear = pref_clear
        self.clear_weight = clear_weight
        self._cache: Dict[frozenset, tuple] = {}

    # ---------------------------------------------------------------- geometry
    def cell(self, x: float, y: float) -> Tuple[int, int]:
        r = min(max(int((y - self.y0) / self.res), 0), self.occ.shape[0] - 1)
        c = min(max(int((x - self.x0) / self.res), 0), self.occ.shape[1] - 1)
        return r, c

    def world(self, r: int, c: int) -> Point:
        return self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res

    def _door_strip(self, door: str) -> Tuple[float, float, float, float]:
        info = H.DOORWAY_INFO.get(door) or H.EXITS.get(door)
        (px, py), w = info["pos"], info["width"]
        half_t = 0.35
        if info["axis"] == "x":   # vertical wall: strip spans y
            return px - half_t, py - w / 2, px + half_t, py + w / 2
        return px - w / 2, py - half_t, px + w / 2, py + half_t

    def _grid(self, blocked: frozenset, avoid: Tuple = (), avoid_cost: Optional[float] = None):
        """`avoid`: rectangles (x0, y0, x1, y1) the owner asked us to keep off (e.g. a new carpet).
        avoid_cost None -> hard no-go (like a wall); a number -> walkable but each metre inside
        costs (1 + avoid_cost) metres (used when an emergency may override the owner's rule)."""
        key = (blocked, tuple(avoid), avoid_cost)
        if key in self._cache:
            return self._cache[key]
        from scipy.sparse import csr_matrix
        occ = self.occ.copy()
        for door in blocked:
            if door in H.DOORWAY_INFO or door in H.EXITS:
                x0, y0, x1, y1 = self._door_strip(door)
                r0, c0 = self.cell(x0, y0)
                r1, c1 = self.cell(x1, y1)
                occ[r0:r1 + 1, c0:c1 + 1] = True
        avoid_mask = np.zeros_like(occ)
        for x0, y0, x1, y1 in avoid:
            r0, c0 = self.cell(min(x0, x1), min(y0, y1))
            r1, c1 = self.cell(max(x0, x1), max(y0, y1))
            avoid_mask[r0:r1 + 1, c0:c1 + 1] = True
        if avoid_cost is None:
            occ |= avoid_mask
        clear = ndimage.distance_transform_edt(~occ) * self.res
        free = clear >= self.robot_radius
        cost = 1.0 + self.clear_weight * np.clip((self.pref_clear - clear) / self.pref_clear, 0.0, None)
        if avoid_cost is not None:
            # Penalise the area plus the robot's radius around it (its body would be on it).
            near_area = ndimage.distance_transform_edt(~avoid_mask) * self.res < self.robot_radius
            cost = cost + float(avoid_cost) * near_area

        h, w = occ.shape
        idx = np.arange(h * w).reshape(h, w)
        rows, cols, vals = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            r0, r1 = max(0, -dr), h - max(0, dr)
            c0, c1 = max(0, -dc), w - max(0, dc)
            a = (slice(r0, r1), slice(c0, c1))
            b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
            ok = free[a] & free[b]
            step = math.hypot(dr, dc) * self.res
            wgt = step * 0.5 * (cost[a] + cost[b])
            ia, ib, wv = idx[a][ok], idx[b][ok], wgt[ok]
            rows += [ia, ib]
            cols += [ib, ia]
            vals += [wv, wv]
        graph = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w, h * w))
        # nearest free cell for snapping start/goal
        _, near = ndimage.distance_transform_edt(~free, return_indices=True)
        entry = (graph, free, clear, near)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = entry
        return entry

    def _snap(self, rc: Tuple[int, int], free, near) -> Tuple[int, int]:
        r, c = rc
        if free[r, c]:
            return r, c
        return int(near[0][r, c]), int(near[1][r, c])

    def _los(self, a: Point, b: Point, clear, margin: float) -> bool:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(2, int(d / (self.res * 0.5)))
        for i in range(n + 1):
            t = i / n
            r, c = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if clear[r, c] < margin:
                return False
        return True

    def _smooth(self, pts: List[Point], clear) -> List[Point]:
        if len(pts) <= 2:
            return pts
        margin = self.robot_radius + 0.08
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._los(pts[i], pts[j], clear, margin):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    # ---------------------------------------------------------------- doors
    @staticmethod
    def _door_crossings(pts: List[Point]) -> List[Tuple[int, str, Point]]:
        """(index into pts after which the crossing happens, door name, crossing point)."""
        doors = [(n, d) for n, d in H.DOORWAY_INFO.items()] + [(n, e) for n, e in H.EXITS.items()]
        out = []
        for i in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[i], pts[i + 1]
            for name, d in doors:
                (px, py), w = d["pos"], d["width"]
                if d["axis"] == "x":
                    if (ax - px) * (bx - px) < 0 or (bx == px and ax != px):
                        t = (px - ax) / (bx - ax)
                        y = ay + t * (by - ay)
                        if abs(y - py) <= w / 2 + 0.05:
                            out.append((i, name, (px, y)))
                else:
                    if (ay - py) * (by - py) < 0 or (by == py and ay != py):
                        t = (py - ay) / (by - ay)
                        x = ax + t * (bx - ax)
                        if abs(x - px) <= w / 2 + 0.05:
                            out.append((i, name, (x, py)))
        return out

    # ---------------------------------------------------------------- public
    def plan(self, start: Point, goal: Point, blocked: Iterable[str] = (),
             avoid: Iterable[Tuple[float, float, float, float]] = (),
             avoid_cost: Optional[float] = None) -> Optional[List[Tuple[float, float, Optional[str]]]]:
        """Waypoints [(x, y, door_name|None), ...] ending at (the nearest free cell to) goal, or None.
        Optional `avoid` / `avoid_cost`: see _grid (owner's no-go areas; additive, defaults unchanged)."""
        from scipy.sparse.csgraph import dijkstra
        graph, free, clear, near = self._grid(frozenset(blocked), tuple(avoid), avoid_cost)
        s = self._snap(self.cell(*start), free, near)
        g = self._snap(self.cell(*goal), free, near)
        h, w = free.shape
        si, gi = s[0] * w + s[1], g[0] * w + g[1]
        if si == gi:
            return [(goal[0], goal[1], None)]
        dist, pred = dijkstra(graph, directed=False, indices=si, return_predecessors=True, limit=np.inf)
        if not np.isfinite(dist[gi]):
            return None
        cells = [gi]
        while cells[-1] != si:
            cells.append(int(pred[cells[-1]]))
        cells.reverse()
        raw = [self.world(c // w, c % w) for c in cells]
        raw[0] = start if free[s] and s == self.cell(*start) else raw[0]
        goal_free = self.cell(*goal) == g
        raw[-1] = goal if goal_free else raw[-1]

        # Split at doorway crossings so every crossed door becomes a tagged waypoint.
        crossings = self._door_crossings(raw)
        wps: List[Tuple[float, float, Optional[str]]] = []
        seg_start = 0
        cur = raw[0]
        for i, name, pt in crossings:
            piece = [cur] + raw[seg_start + 1: i + 1] + [pt]
            for p in self._smooth(piece, clear)[1:]:
                wps.append((p[0], p[1], None))
            x, y, _ = wps.pop()
            wps.append((x, y, name))
            cur, seg_start = pt, i
        piece = [cur] + raw[seg_start + 1:]
        for p in self._smooth(piece, clear)[1:]:
            wps.append((p[0], p[1], None))
        if not wps:
            wps = [(raw[-1][0], raw[-1][1], None)]
        return wps


if __name__ == "__main__":
    build_map()
