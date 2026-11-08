"""Option A step 5: let the brain use the robot's own map. Only active with NUDGE_ROBOT_MAP=1 (default off).

With the flag on, `navigator.Navigator` plans on the map the robot explored itself (scripts/test_mapping.py
--mode explore -> assets/maps/robot_map.npz, or NUDGE_ROBOT_MAP_FILE) instead of the static floor plan
(sim/nav_map.py), kept live from the LiDAR: obstacles that appear in floor the saved map knows as free
(a chair pushed into a doorway) are added and the route is re-planned; base obstacles the LiDAR has since
seen clearly free (something moved away) are dropped. Appeared cells where the owner was recently seen are
the owner, not furniture, and are ignored.

Objects the robot found (object_memory.py) become places the owner can ask for: "take me to the sofa",
"... to my bed", "... to the TV / fridge / dining table" resolve to a free spot next to that object
(`robot_map_target(text)`, used by rules.py). house_layout names (rooms, landmarks) keep working as before.
If the flag is on but no saved map exists, the brain logs it and uses the static floor plan. The only outside
knowledge is the installer's ground-floor box (house_layout.HOUSE_BOUNDS, the explorer's geofence).
"""
from __future__ import annotations

import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENABLED = os.environ.get("NUDGE_ROBOT_MAP", "0") == "1"
_ROOT = Path(__file__).resolve().parent.parent
# Your own fresh map (assets/maps/, git-ignored) wins; else the explored map committed in maps/ (so no PC
# has to re-explore); NUDGE_ROBOT_MAP_FILE overrides both.
_LOCAL_MAP = _ROOT / "assets" / "maps" / "robot_map.npz"
_SHARED_MAP = _ROOT / "maps" / "robot_map.npz"
MAP_FILE = Path(os.environ.get("NUDGE_ROBOT_MAP_FILE", "") or (_LOCAL_MAP if _LOCAL_MAP.exists() else _SHARED_MAP))

# object class -> (place key, words the owner may use)
OBJECT_PLACES = {
    "couch": ("sofa", r"\b(sofa|couch|settee)\b"),
    "bed": ("bed", r"\bbed\b"),
    "tv": ("tv", r"\b(tv|telly|television)\b"),
    "refrigerator": ("fridge", r"\b(fridge|refrigerator)\b"),
    "dining table": ("dining_table", r"\b(dining )?table\b"),
}
PLACES: Dict[str, Tuple[str, Tuple[float, float]]] = {}    # key -> (pattern, approach xy), filled by register()
SCAN_EXITS: Dict[str, dict] = {}   # exits found in the robot's own map (register_exits); used by evacuation.py
OWNER_MEMORY_S = 60.0
OWNER_RADIUS = 0.7
REBUILD_MIN_S = 2.0


def robot_map_target(text: str) -> Optional[str]:
    """'take me to the sofa' -> 'sofa' if the robot found a sofa (flag on), else None."""
    if not PLACES or not text:
        return None
    t = text.lower()
    for key, (pat, _xy) in PLACES.items():
        if re.search(pat, t):
            return key
    return None


class RobotMapNav:
    def __init__(self, path: Path = MAP_FILE) -> None:
        from object_memory import ObjectMemory
        from robot_map import OccupancyGrid
        self.path = Path(path)
        self.grid, objs, self.rooms, self.meta = OccupancyGrid.load(self.path)
        self.omem = ObjectMemory()
        self.omem.grid = self.grid
        self.omem.load(objs)
        self.owner_seen: deque = deque()
        self._changes: List[dict] = []
        self._changes_key: tuple = ()
        self._last_check = 0.0
        self.last_rebuild = 0.0
        # The installer's "ground floor is inside this box" (same as the explorer's geofence); unseen cells
        # outside it are garden, so evacuation routes can end outside an exit the robot saw open.
        from sim.house_layout import HOUSE_BOUNDS as B
        self.geofence = (B["x_min"], B["y_min"], B["x_max"], B["y_max"])

    # ---------------------------------------------------------------- live updates
    def observe(self, pose, lidar, owner_xy=None) -> bool:
        """Integrate one LiDAR scan. True if the set of (non-owner) changes differs from last time."""
        now = time.monotonic()
        self.grid.integrate(pose, lidar)
        if owner_xy is not None:
            self.owner_seen.append((now, float(owner_xy[0]), float(owner_xy[1])))
        while self.owner_seen and now - self.owner_seen[0][0] > OWNER_MEMORY_S:
            self.owner_seen.popleft()
        if now - self._last_check < 0.5:
            return False
        self._last_check = now
        ch = [c for c in self.grid.current_changes() if not self._near_owner(c["x"], c["y"])]
        key = tuple(sorted((round(c["x"]), round(c["y"]), c["cells"] // 4) for c in ch))
        if key == self._changes_key:
            return False
        self._changes, self._changes_key = ch, key
        return True

    def _near_owner(self, x: float, y: float) -> bool:
        return any(math.hypot(x - ox, y - oy) < OWNER_RADIUS for _t, ox, oy in self.owner_seen)

    def changes(self) -> List[dict]:
        return list(self._changes)

    def navmap(self):
        """sim.nav_map.NavMap of the robot's map (+ live changes, minus the owner's recent spots)."""
        import numpy as np
        from sim.nav_map import NavMap
        g = self.grid
        occ = g.nav_occupancy(unknown_is_occupied=True, with_changes=False)
        if self.geofence is not None:   # outside the installer's ground-floor box, unseen = open garden
            x0, y0, x1, y1 = self.geofence
            xs = g.origin[0] + np.arange(g.w) * g.res
            ys = g.origin[1] + np.arange(g.h) * g.res
            outside = ~(((ys[:, None] >= y0) & (ys[:, None] <= y1)) & ((xs[None, :] >= x0) & (xs[None, :] <= x1)))
            occ = occ & ~(outside & ~g.occupied())
        app = g.appeared_mask()
        if self.owner_seen and app.any():
            r = int(OWNER_RADIUS / g.res)
            for _t, ox, oy in list(self.owner_seen)[::5]:
                cx, cy = g._cell(ox, oy)
                app[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] = False
        return NavMap(occ | app, g.res, g.bounds())

    # ---------------------------------------------------------------- objects as places
    def approach_point(self, obj: dict, nav=None, clear: float = 0.5) -> Optional[Tuple[float, float]]:
        """Nearest spot with `clear` m of free floor around it, within 2 m of the object."""
        import numpy as np
        nav = nav or self.navmap()
        r0, c0 = nav.cell(obj["x"], obj["y"])
        k = int(2.0 / nav.res)
        rs, cs = slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1)
        ok = nav.dist[rs, cs] >= clear
        if not ok.any():
            return None
        rr, cc = np.nonzero(ok)
        rr, cc = rr + rs.start, cc + cs.start
        d = np.hypot(rr - r0, cc - c0)
        i = int(np.argmin(d))
        return nav.world(int(rr[i]), int(cc[i]))

    def register_places(self, landmarks: dict) -> Dict[str, Tuple[float, float]]:
        """Add found objects to LANDMARKS (in this process only) and to PLACES for rules.py."""
        nav = self.navmap()
        out = {}
        for cls, (key, pat) in OBJECT_PLACES.items():
            if key in landmarks and key not in PLACES:
                continue                                   # never shadow a house_layout landmark
            obj = self.omem.find(cls)
            if obj is None:
                continue
            xy = self.approach_point(obj, nav)
            if xy is None:
                continue
            xy = (round(xy[0], 2), round(xy[1], 2))
            landmarks[key] = xy
            PLACES[key] = (pat, xy)
            out[key] = xy
        return out

    def register_exits(self, landmarks: dict) -> List[dict]:
        """Find the exits in the robot's map and offer them to the evacuation planner (SCAN_EXITS) with an
        outside target each (added to LANDMARKS in this process). The room label of each exit (for "is the
        fire in this room?") is house_layout's room name, the same names the fire estimate uses."""
        from sim.house_layout import get_room_at
        SCAN_EXITS.clear()
        found = find_exits(self.grid, self.geofence)
        used: Dict[str, int] = {}
        for e in found:
            used[e["side"]] = used.get(e["side"], 0) + 1
            base = f"{e['side']}_exit" + (f"_{used[e['side']]}" if used[e["side"]] > 1 else "")
            target = f"{base}_outside"
            landmarks[target] = e["outside"]
            SCAN_EXITS[base] = dict(e, name=base, target=target, room=get_room_at(*e["inside"]))
        return list(SCAN_EXITS.values())

    def status(self) -> dict:
        return {"file": str(self.path.name), "known_frac": round(self.grid.known_frac(), 3),
                "objects": [{k: o[k] for k in ("id", "cls", "x", "y", "room")} for o in self.omem.confirmed()],
                "rooms": self.rooms, "places": {k: v[1] for k, v in PLACES.items()}, "changes": self.changes()}


# ---------------------------------------------------------------------------- exits from the scan
EXIT_BAND_M = 0.6        # look this far either side of the outer wall line
EXIT_REACH_M = 0.35      # an opening must have known floor this far inside AND outside
EXIT_MIN_WIDTH_M = 0.6   # narrower gaps (LiDAR leaking past a door frame) are not exits
EXIT_OUT_M = 1.5         # target this far outside the door point


def find_exits(grid, geofence) -> List[dict]:
    """Openings in the explored outer wall: known-free floor that crosses the installer's ground-floor box
    edge (inside -> garden). Each: {name, pos (door point on the wall line), outside (1.5 m out), inside,
    width, axis ('x' = vertical wall), side}. Uses only the robot's map + the geofence box."""
    import numpy as np
    from scipy import ndimage
    x0, y0, x1, y1 = geofence
    xs = grid.origin[0] + np.arange(grid.w) * grid.res
    ys = grid.origin[1] + np.arange(grid.h) * grid.res
    X, Y = np.meshgrid(xs, ys)
    sides = np.stack([X - x0, x1 - X, Y - y0, y1 - Y])          # west, east, south, north
    d_in = sides.min(axis=0)                                     # >0 inside, <0 outside
    side = sides.argmin(axis=0)
    free = grid.free()
    band = np.abs(d_in) <= EXIT_BAND_M
    lab, n = ndimage.label(free & band, structure=np.ones((3, 3)))
    names = ("west", "east", "south", "north")
    normals = ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == k
        di, sd = d_in[sl][m], side[sl][m]
        if not ((di >= EXIT_REACH_M).any() and (di <= -EXIT_REACH_M).any()):
            continue
        cross = np.abs(di) <= 0.06          # free cells ON the wall centre line = the gap itself
        if not cross.any():
            continue
        s = int(np.bincount(sd[cross]).argmax())
        cx, cy = X[sl][m][cross], Y[sl][m][cross]
        along = cy if s in (0, 1) else cx
        width = float(along.max() - along.min()) + grid.res
        if width < EXIT_MIN_WIDTH_M:
            continue
        nx, ny = normals[s]
        if s in (0, 1):
            px, py = (x0 if s == 0 else x1), float(np.mean(cy))
        else:
            px, py = float(np.mean(cx)), (y0 if s == 2 else y1)
        out.append({"pos": (round(px, 2), round(py, 2)),
                    "outside": (round(px + EXIT_OUT_M * nx, 2), round(py + EXIT_OUT_M * ny, 2)),
                    "inside": (round(px - 0.8 * nx, 2), round(py - 0.8 * ny, 2)),
                    "width": round(width, 2), "axis": "x" if s in (0, 1) else "y", "side": names[s]})
    for i, e in enumerate(sorted(out, key=lambda e: (e["side"], e["pos"]))):
        e["name"] = f"scan_exit_{e['side']}_{i + 1}"
    return sorted(out, key=lambda e: e["name"])
