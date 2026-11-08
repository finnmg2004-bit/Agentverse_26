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
If the flag is on but no saved map exists, the brain logs it and uses the static floor plan.
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
MAP_FILE = Path(os.environ.get("NUDGE_ROBOT_MAP_FILE", str(_ROOT / "assets" / "maps" / "robot_map.npz")))

# object class -> (place key, words the owner may use)
OBJECT_PLACES = {
    "couch": ("sofa", r"\b(sofa|couch|settee)\b"),
    "bed": ("bed", r"\bbed\b"),
    "tv": ("tv", r"\b(tv|telly|television)\b"),
    "refrigerator": ("fridge", r"\b(fridge|refrigerator)\b"),
    "dining table": ("dining_table", r"\b(dining )?table\b"),
}
PLACES: Dict[str, Tuple[str, Tuple[float, float]]] = {}    # key -> (pattern, approach xy), filled by register()
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

    def status(self) -> dict:
        return {"file": str(self.path.name), "known_frac": round(self.grid.known_frac(), 3),
                "objects": [{k: o[k] for k in ("id", "cls", "x", "y", "room")} for o in self.omem.confirmed()],
                "rooms": self.rooms, "places": {k: v[1] for k, v in PLACES.items()}, "changes": self.changes()}
