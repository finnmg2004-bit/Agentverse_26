"""
EXPERIMENTAL second house for Nudge: the nerv-world "apt" scene (Manhattan duplex, floor 62),
selected with NUDGE_HOUSE=apt (start-up option `--house apt`). Default off: without the env var
sim/house_layout.py is the house as always.

Same public names as sim/house_layout.py (ROOMS, DOORWAYS, DOORWAY_INFO, EXITS, LANDMARKS,
SOUND_SOURCES, OWNER_SPOTS, ROBOT_START_POSE, WALL_SEGMENTS_2D, get_room_at, ...), so the sim and the
brain run unchanged; sim/house_layout.py copies this module's names over its own at import.
Room KEYS are Nudge's (entry, living_room, dining, ...) so code that names rooms keeps working;
labels say what the room is in the flat.

Values copied from worlds/nerv-world/scenes/apt/layout.py (ROOMS / DOORS / FRONT_DOOR / furniture),
floor 62 only (z = 0). Never edit the submodule.

         y=7.5 +-----------+-----------+--------------+-----------+
               |  kitchen  |  dining   |  living_room | family_rm |
               |  (OOB)    |           |  (great rm)  | (library) |
         y=1.4 +---[gap]---+----------=[ open ]=-----=+--[open]---+
               | stair hall|        gallery           =robot_home |
               |  (flights)=[gap]                     | (east hall)
        y=-1.4 +---[gap]---+-[door]---+---------=[open]=+---------+
               |  guest    | guest    | laundry =  entry  = media |
               |  (OOB)    | bath     =         | (foyer) |  (TV)  |
        y=-7.5 +-----------+----------+---------+-[FRONT]-+-------+
            x=-11.5     x=-5.12    x=-1.5    x=1.5  x=4  x=6.5  x=11.5
Known limits (day/reports/startup-options.md): kitchen + guest bedroom are only reachable across the
stair hall: the guest bedroom is unreachable (no asleep-in-bed here); the kitchen is reached across the hall;
the front door is the only exit (into a small lift-lobby stub Nudge adds at load time).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

from sim import house_layout as _HOUSE  # the house module (already loaded; used for the shared registry data)

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

# --------------------------------------------------------------------------- #
# Scene files (relative to the project root)
# --------------------------------------------------------------------------- #
WORLD_SUBMODULE = "worlds/nerv-world"
WORLD_SCENE_XML = f"{WORLD_SUBMODULE}/build/apt-go2.xml"   # generated upstream; never hand-edit
NAV_MAP_FILE = "assets/maps/nerv_apt_nav.npz"              # own cache: never mixes with the house map

FLOOR_Z = 0.0               # upstream FLOOR_Z(0): floor 62 is still z = 0 (the city is lowered instead)
WALL_HEIGHT = 3.60          # upstream WALL_HEIGHT (3.75 storey - slabs); the cutaway hides floor 63
WALL_THICK = 0.14

HOUSE_BOUNDS = {"x_min": -11.5, "x_max": 11.5, "y_min": -7.5, "y_max": 7.5}  # 23 x 15 m, floor 62
OUTSIDE = "outside"         # pseudo-room outside the flat (= the lift-lobby stub, ADD_GEOMS)

# --------------------------------------------------------------------------- #
# Rooms (floor 62). Nudge room key -> (label, rect). Upstream grid: X0=-11.5, XS=-5.12, XA=-0.5,
# XB=5.5, X4=11.5; Y0=-7.5, Y1=-1.4, Y2=1.4, Y3=7.5.
# --------------------------------------------------------------------------- #
_ROOM_DEFS: Dict[str, Tuple[str, Rect]] = {
    "entry":       ("Foyer (front door)",  (1.5, -7.5, 6.5, -1.4)),      # upstream foyer
    "gallery":     ("Gallery",             (-5.12, -1.4, 5.5, 1.4)),     # gallery (the flat's spine)
    "living_room": ("Great room",          (-0.5, 1.4, 5.5, 7.5)),       # great_room (double height)
    "dining":      ("Dining room",         (-5.12, 1.4, -0.5, 7.5)),     # dining
    "family_room": ("Library",             (5.5, 1.4, 11.5, 7.5)),       # library
    "robot_home":  ("East hall",           (5.5, -1.4, 11.5, 1.4)),      # east_hall (robot dock)
    "media":       ("Media room",          (6.5, -7.5, 11.5, -1.4)),     # media (TV)
    "laundry":     ("Laundry",             (-1.5, -7.5, 1.5, -1.4)),
    "guest_bath":  ("Guest bathroom",      (-5.12, -7.5, -1.5, -1.4)),
    # "guest" (guest_bed, -11.5..-5.12 x -7.5..-1.4) is left out: its only door opens onto the stair
    # flights, the nav map finds no path, so it is out of bounds (no asleep-in-bed in the apt).
    "kitchen":     ("Kitchen",             (-11.5, 1.4, -5.12, 7.5)),    # kitchen (only via the stair hall)
}
# Stair hall (stair_f62): the two flights fill it; kitchen + guest bedroom open only onto it. The nav
# map still finds a way into the kitchen (kettle / smoke alarm room); the guest bedroom is unreachable.
STAIR_CORE: Rect = (-11.5, -1.4, -5.12, 1.4)

# Clear floor point the robot drives to for "go to room" / look_around (furniture-free).
ROOM_SCAN_POINTS: Dict[str, Vec2] = {
    "entry":       (4.0, -4.6),
    "gallery":     (0.0, 0.0),
    "living_room": (4.6, 2.4),
    "dining":      (-1.4, 2.4),
    "family_room": (8.0, 3.0),
    "robot_home":  (8.5, 0.0),
    "media":       (7.4, -2.4),
    "laundry":     (0.0, -4.4),
    "guest_bath":  (-3.3, -4.4),
    "kitchen":     (-8.0, 2.4),
}

# Where the owner stands in each room (clear of furniture; check_world verifies them).
OWNER_SPOTS: Dict[str, Vec2] = {
    "living_room": (0.5, 3.2),
    "dining":      (-4.2, 2.4),
    "family_room": (8.5, 5.6),
    "entry":       (2.6, -4.6),
    "media":       (9.6, -2.4),
}
OWNER_START_ROOM = "living_room"

OWNER_ROOM_CYCLE: List[str] = ["living_room", "dining", "family_room", "media"]
ROOM_ORDER: List[str] = OWNER_ROOM_CYCLE

ROOMS: Dict[str, dict] = {}
for _key, (_label, _rect) in _ROOM_DEFS.items():
    _x0, _y0, _x1, _y1 = _rect
    _center = ROOM_SCAN_POINTS.get(_key, ((_x0 + _x1) / 2.0, (_y0 + _y1) / 2.0))
    ROOMS[_key] = {
        "label": _label,
        "rect": _rect,
        "bounds_x": (_x0, _x1),
        "bounds_y": (_y0, _y1),
        "center": _center,
        "geo_center": ((_x0 + _x1) / 2.0, (_y0 + _y1) / 2.0),
        "owner_spot": OWNER_SPOTS.get(_key, _center),
        "search": _key in ROOM_ORDER,
    }
ROOM_CENTERS: Dict[str, Vec2] = {k: r["center"] for k, r in ROOMS.items()}

# --------------------------------------------------------------------------- #
# Doorways: upstream DOORS, floor 0. orient "h" -> wall at y = coord (axis "y"); "v" -> x = coord (axis "x").
# --------------------------------------------------------------------------- #
DOORWAY_INFO: Dict[str, dict] = {
    "door_entry_gallery":     dict(a="entry",       b="gallery",     pos=(2.5, -1.4),  width=2.6,  axis="y", kind="open"),
    "door_gallery_living":    dict(a="gallery",     b="living_room", pos=(2.5, 1.4),   width=4.0,  axis="y", kind="open"),
    "door_dining_living":     dict(a="dining",      b="living_room", pos=(-0.5, 4.6),  width=1.8,  axis="x", kind="open"),
    "door_living_family":     dict(a="living_room", b="family_room", pos=(5.5, 4.6),   width=1.6,  axis="x", kind="open"),
    "door_gallery_robothome": dict(a="gallery",     b="robot_home",  pos=(5.5, 0.0),   width=1.4,  axis="x", kind="open"),
    "door_gallery_guestbath": dict(a="gallery",     b="guest_bath",  pos=(-3.4, -1.4), width=1.0,  axis="y", kind="door"),
    "door_guestbath_laundry": dict(a="guest_bath",  b="laundry",     pos=(-1.5, -4.4), width=0.95, axis="x", kind="door"),
    "door_laundry_entry":     dict(a="laundry",     b="entry",       pos=(1.5, -4.4),  width=1.0,  axis="x", kind="door"),
    "door_entry_media":       dict(a="entry",       b="media",       pos=(6.5, -4.4),  width=1.2,  axis="x", kind="door"),
}
# Gaps in the walls that are not navigation edges (the stair hall's doors).
WALL_GAPS_ONLY: List[dict] = [
    dict(pos=(-5.12, 0.0), width=1.3, axis="x"),   # gallery -> stair hall
    dict(pos=(-8.0, 1.4), width=1.6, axis="y"),    # stair hall -> kitchen
    dict(pos=(-8.6, -1.4), width=1.1, axis="y"),   # stair hall -> guest bedroom
]

# Load-time patches (sim/world.py, apt only). Upstream's front door is a decorative leaf in front of a
# SOLID foyer wall (foyer_ws0) above a 232 m drop, so the flat has no walkable exit. Nudge cuts a real
# 1.3 m doorway (between the existing casings, x 3.35..4.65) and adds a small lift-lobby stub outside
# (floor + walls) so "lead the owner out of the front door" means the same as in the house.
REMOVE_GEOMS: Tuple[str, ...] = ("front_door", "front_door_handle", "foyer_ws0")
# (name, half-size xyz, centre xyz, rgba) boxes, group 0, collide like walls.
_PLASTER = (0.90, 0.89, 0.86, 1.0)
ADD_GEOMS: Tuple[tuple, ...] = (
    ("nudge_foyer_wall_w", (0.925, 0.07, 1.8), (2.425, -7.43, 1.8), _PLASTER),   # foyer south wall, west of the door
    ("nudge_foyer_wall_e", (0.925, 0.07, 1.8), (5.575, -7.43, 1.8), _PLASTER),   # ... east of the door
    ("nudge_foyer_lintel", (0.65, 0.07, 0.575), (4.0, -7.43, 3.025), _PLASTER),  # above the 2.45 m door
    ("nudge_lobby_floor",  (1.5, 1.25, 0.05), (4.0, -8.75, -0.05), (0.55, 0.55, 0.57, 1.0)),
    ("nudge_lobby_wall_w", (0.07, 1.25, 1.4), (2.43, -8.75, 1.4), _PLASTER),
    ("nudge_lobby_wall_e", (0.07, 1.25, 1.4), (5.57, -8.75, 1.4), _PLASTER),
    ("nudge_lobby_wall_s", (1.57, 0.07, 1.4), (4.0, -10.07, 1.4), _PLASTER),
)

# The flat's only exterior door: upstream FRONT_DOOR, foyer south wall, centre x 4.0, 1.3 m wide.
EXITS: Dict[str, dict] = {
    "exit_front": dict(room="entry", pos=(4.0, -7.5), outside=(4.0, -9.0), width=1.3, axis="y"),  # -> lift lobby
}

DOORWAYS: Dict[str, Vec2] = {k: d["pos"] for k, d in DOORWAY_INFO.items()}
DOORWAYS.update({k: e["pos"] for k, e in EXITS.items()})
DOORWAY_WIDTH: Dict[str, float] = {k: d["width"] for k, d in DOORWAY_INFO.items()}
DOORWAY_WIDTH.update({k: e["width"] for k, e in EXITS.items()})

ROOM_GRAPH: Dict[Tuple[str, str], str] = {}
for _name, _d in DOORWAY_INFO.items():
    ROOM_GRAPH[(_d["a"], _d["b"])] = _name
    ROOM_GRAPH[(_d["b"], _d["a"])] = _name
for _name, _e in EXITS.items():
    ROOM_GRAPH[(_e["room"], OUTSIDE)] = _name
    ROOM_GRAPH[(OUTSIDE, _e["room"])] = _name

# --------------------------------------------------------------------------- #
# Named navigation targets. back_door / back_exit_outside / terrace_* are ALIASES of the front door,
# so code written for the house (safety.EVACUATION_TARGET, the sim judge, intents) still finds them.
# --------------------------------------------------------------------------- #
LANDMARKS: Dict[str, Vec2] = {
    "front_door":         (4.0, -6.8),
    "front_door_outside": EXITS["exit_front"]["outside"],
    "back_door":          (4.0, -6.8),                     # alias: the flat has one door
    "back_exit_outside":  EXITS["exit_front"]["outside"],  # alias
    "terrace_door":       (4.0, -6.8),                     # alias
    "terrace_outside":    EXITS["exit_front"]["outside"],  # alias
    "robot_dock":         (8.5, -0.6),                     # east hall (upstream ROBOT_HOME_XY is mid-gallery)
    "kettle":             (-9.6, 5.9),                     # kitchen counter run - out of bounds here
    "stove":              (-9.6, 5.9),
}
SMOKE_EVACUATION_TARGET = "front_door_outside"

# Robot start: upstream START_POS_XY (4.0, -6.6) stands in the front door; start a bit further in.
ROBOT_START_XY: Vec2 = (4.0, -5.4)
ROBOT_START_YAW: float = math.pi / 2
ROBOT_START_POSE: Vec3 = (ROBOT_START_XY[0], ROBOT_START_XY[1], ROBOT_START_YAW)
ROBOT_START_ROOM = "entry"

# Sound registry: the house's sounds (tiers, clips, YAMNet classes, messages), moved into the flat.
_SOUND_PLACES = {
    "doorbell":    dict(pos=(5.2, -7.35, 1.9), room="entry"),           # beside the front door
    "knock":       dict(pos=(4.0, -7.45, 1.0), room="entry"),           # on the door leaf
    "smoke_alarm": dict(pos=(-8.4, 4.2, 3.45), room="kitchen", target_landmark=SMOKE_EVACUATION_TARGET),
    "kettle":      dict(pos=(-9.6, 6.9, 1.0), room="kitchen"),
    "oven_timer":  dict(pos=(-8.0, 6.9, 1.05), room="kitchen"),
}
SOUND_SOURCES: Dict[str, Dict[str, object]] = {}
for _name, _spec in _HOUSE.SOUND_SOURCES.items():
    SOUND_SOURCES[_name] = dict(copy.deepcopy(_spec), **_SOUND_PLACES.get(_name, {}))
SOUND_TIERS: Dict[str, int] = {name: int(spec["tier"]) for name, spec in SOUND_SOURCES.items()}
DANGER_SOUNDS = tuple(n for n, t in SOUND_TIERS.items() if t == 1)
ALERT_SOUNDS = tuple(sorted((n for n, t in SOUND_TIERS.items() if t > 1), key=lambda n: SOUND_TIERS[n]))

OWNER_VOICE: Dict[str, object] = dict(_HOUSE.OWNER_VOICE)
TV_SOURCE: Dict[str, object] = dict(_HOUSE.TV_SOURCE, pos=(10.95, -4.4, 1.3), room="media")  # upstream md_screen

CHAIR_DOORWAY_POS: Vec3 = (DOORWAY_INFO["door_entry_media"]["pos"][0], DOORWAY_INFO["door_entry_media"]["pos"][1], 0.0)
CHAIR_STOWED_POS: Vec3 = (10.6, -6.9, 0.0)  # media-room corner
CHAIR_EXIT_POS: Vec3 = (EXITS["exit_front"]["pos"][0], EXITS["exit_front"]["pos"][1], 0.0)
EXIT_TARGETS: Dict[str, str] = {"exit_front": "front_door_outside"}
CHAIR_COLLISION_RADIUS: float = 0.22

AVOID_AREAS: Dict[str, Tuple[float, float, float, float]] = {
    "living_room_carpet": (-0.2, 5.4, 1.0, 7.0),
}
AVOID_AREA_LABELS: Dict[str, str] = {"living_room_carpet": "the new carpet in the great room"}
AVOID_AREA_MARGIN: float = 0.25


def avoid_area_at(x: float, y: float, margin: float = 0.0) -> Optional[str]:
    """Name of the avoid area containing (x, y) (rectangle grown by `margin`), else None."""
    for name, (x0, y0, x1, y1) in AVOID_AREAS.items():
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
            return name
    return None


# --------------------------------------------------------------------------- #
# Wall segments (2D) for acoustic occlusion and the dashboard map:
# all room edges merged per wall line, minus door/exit gaps.
# --------------------------------------------------------------------------- #
def _merge(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def _subtract(intervals: List[Tuple[float, float]], gaps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    res = intervals
    for glo, ghi in gaps:
        nxt = []
        for lo, hi in res:
            if ghi <= lo or glo >= hi:
                nxt.append((lo, hi))
                continue
            if glo > lo:
                nxt.append((lo, glo))
            if ghi < hi:
                nxt.append((ghi, hi))
        res = nxt
    return [(a, b) for a, b in res if b - a > 1e-3]


def _build_wall_segments() -> List[Tuple[Vec2, Vec2]]:
    horiz: Dict[float, List[Tuple[float, float]]] = {}
    vert: Dict[float, List[Tuple[float, float]]] = {}
    rects = [r for _, r in _ROOM_DEFS.values()] + [STAIR_CORE]
    for x0, y0, x1, y1 in rects:
        for y in (y0, y1):
            horiz.setdefault(round(y, 3), []).append((x0, x1))
        for x in (x0, x1):
            vert.setdefault(round(x, 3), []).append((y0, y1))
    gaps = [(d["pos"], d["width"], d["axis"]) for d in DOORWAY_INFO.values()]
    gaps += [(e["pos"], e["width"], e["axis"]) for e in EXITS.values()]
    gaps += [(g["pos"], g["width"], g["axis"]) for g in WALL_GAPS_ONLY]

    segs: List[Tuple[Vec2, Vec2]] = []
    for y, ivs in horiz.items():
        cut = [(p[0] - w / 2, p[0] + w / 2) for p, w, ax in gaps if ax == "y" and abs(p[1] - y) < 1e-3]
        segs += [((a, y), (b, y)) for a, b in _subtract(_merge(ivs), cut)]
    for x, ivs in vert.items():
        cut = [(p[1] - w / 2, p[1] + w / 2) for p, w, ax in gaps if ax == "x" and abs(p[0] - x) < 1e-3]
        segs += [((x, a), (x, b)) for a, b in _subtract(_merge(ivs), cut)]
    return segs


WALL_SEGMENTS_2D: List[Tuple[Vec2, Vec2]] = _build_wall_segments()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_room_at(x: float, y: float) -> str:
    """Ground-floor room key containing (x, y); 'stair_core' or OUTSIDE otherwise."""
    for key, (_, (x0, y0, x1, y1)) in _ROOM_DEFS.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return key
    x0, y0, x1, y1 = STAIR_CORE
    if x0 <= x <= x1 and y0 <= y <= y1:
        return "stair_core"
    return OUTSIDE


def room_at(x: float, y: float) -> Optional[str]:
    """Like get_room_at() but None outside the house."""
    r = get_room_at(x, y)
    return None if r == OUTSIDE else r


def get_landmark_room(target: str) -> str:
    """Room associated with a named room or landmark."""
    if target in ROOMS:
        return target
    if target in LANDMARKS:
        return get_room_at(*LANDMARKS[target])
    return ROBOT_START_ROOM


def room_graph() -> Dict[str, List[Tuple[str, str]]]:
    """Adjacency: room -> [(neighbor_room, doorway_name), ...] (includes OUTSIDE)."""
    g: Dict[str, List[Tuple[str, str]]] = {}
    for (a, b), door in ROOM_GRAPH.items():
        g.setdefault(a, []).append((b, door))
    return g


def nudge_rooms() -> List[str]:
    return list(ROOMS)


def count_walls_crossed(p1: Vec2, p2: Vec2) -> int:
    """Count how many 2D wall segments intersect the line segment p1 -> p2."""
    def ccw(a: Vec2, b: Vec2, c: Vec2) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool:
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    return sum(1 for sa, sb in WALL_SEGMENTS_2D if intersects(p1, p2, sa, sb))
