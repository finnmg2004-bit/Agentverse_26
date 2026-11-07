"""
Nudge house layout — single source of truth for every coordinate the project uses.

The world geometry comes from the nerv-world "house" scene (git submodule at
worlds/nerv-world, MIT, Yanshi-Robotics). Its own truth is
worlds/nerv-world/scenes/house/layout.py; the values below are copied from that
file (room rectangles) or derived from its door rules, and verified against the
compiled scene by `pixi run check-world` (sim/check_world.py).

Nudge only uses the GROUND FLOOR (z = 0). The Go2 in Plan B cannot climb stairs,
so floors 1-2 and the stair core are out of bounds.

World frame: +X = east, +Y = north, +Z = up, metres. Yaw 0 faces +X.

         y=17.5 +--------------------+-----------------+----------------+
                |                    |                 |                |
                |    family_room     |     gallery     |  garden_room   |
                |                    |                 |                |
         y=6.08 +====[ open ]=====+=[open]=+=[open]=+--+-----+--[door]--+
                |                 |                 |  stair |          |
                |   living_room   =[open] dining    |  core  | kitchen  =[door]-> pool terrace (E)
   west door <-=[door]            |                 | (no go)|          |
         y=0.6  +--[door]---------+=====[ open ]====+--------+--[door]--+
                |                 |                 | lobby0 |robot_home|laundry
                |   guest         =[door]  entry    =[door]  =[door]    |
                |   (bedroom)     |                 |        |          |
         y=-4.5 +-----------------+-----[FRONT]-----+--------+----------+
              x=-15             x=-6   x=-1.4     x=3.2    x=6   x=10  x=15
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Rect = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

# --------------------------------------------------------------------------- #
# Scene files (relative to the project root)
# --------------------------------------------------------------------------- #
WORLD_SUBMODULE = "worlds/nerv-world"
WORLD_SCENE_XML = f"{WORLD_SUBMODULE}/build/house-go2.xml"  # generated upstream; never hand-edit

FLOOR_Z = 0.0
WALL_HEIGHT = 2.73          # interior clear height (upstream STOREY_H - slabs)
WALL_THICK = 0.14

HOUSE_BOUNDS: Rect = (-15.0, -4.5, 15.0, 17.5)  # ground-floor footprint

# --------------------------------------------------------------------------- #
# Rooms (ground floor). Keys match upstream layout.py room keys.
# `nudge` = True for rooms the agent may search / navigate to.
# --------------------------------------------------------------------------- #
ROOMS: Dict[str, dict] = {
    "entry":       dict(label="Entry hall",        rect=(-6.0, -4.5, 3.2, 0.6),         nudge=True),
    "guest":       dict(label="Guest bedroom",     rect=(-15.0, -4.5, -6.0, 0.6),       nudge=True),
    "living_room": dict(label="Living room",       rect=(-15.0, 0.6, -6.0, 6.08),       nudge=True),
    "dining":      dict(label="Dining room",       rect=(-6.0, 0.6, 3.2, 6.08),         nudge=True),
    "family_room": dict(label="Garden family room", rect=(-15.0, 6.08, -3.0, 17.5),     nudge=True),
    "kitchen":     dict(label="Kitchen",           rect=(6.0, 0.6, 15.0, 6.08),         nudge=True),
    "lobby0":      dict(label="Stair lobby",       rect=(3.2, -4.5, 6.0, 0.6),          nudge=True),
    "robot_home":  dict(label="Robot dock",        rect=(6.0, -4.5, 10.0, 0.6),         nudge=True),
    "gallery":     dict(label="Garden gallery",    rect=(-3.0, 6.08, 6.0, 17.5),        nudge=False),
    "garden_room": dict(label="Garden lounge",     rect=(6.0, 6.08, 15.0, 17.5),        nudge=False),
    "laundry":     dict(label="Laundry",           rect=(10.0, -4.5, 15.0, 0.6),        nudge=False),
    "stair_f0":    dict(label="Stair core",        rect=(3.2, 0.6, 6.0, 6.08),          nudge=False),
}

# Room center (navigation goal when "go to room" has no better target)
ROOM_CENTERS: Dict[str, Vec2] = {
    k: ((r["rect"][0] + r["rect"][2]) / 2.0, (r["rect"][1] + r["rect"][3]) / 2.0)
    for k, r in ROOMS.items()
}
# Rooms with furniture in the middle get a clear "scan point" instead of the raw center.
ROOM_SCAN_POINTS: Dict[str, Vec2] = {
    "entry":       (-1.4, -2.0),
    "guest":       (-9.5, -2.0),
    "living_room": (-8.5, 4.6),
    "dining":      (-4.5, 1.6),
    "family_room": (-8.0, 9.0),
    "kitchen":     (8.0, 2.4),
    "lobby0":      (4.6, -2.0),
    "robot_home":  (8.0, -2.0),
}

# Where the owner stands in each room (clear of furniture, verified by check_world)
OWNER_SPOTS: Dict[str, Vec2] = {
    "living_room": (-11.0, 5.0),
    "guest":       (-10.0, -2.6),
    "dining":      (-4.6, 1.5),
    "family_room": (-8.0, 9.5),
    "entry":       (-3.6, -2.2),
    "kitchen":     (8.4, 3.2),
}
OWNER_START_ROOM = "living_room"
# Hotkey [O] cycles the owner through these rooms, in order.
OWNER_ROOM_CYCLE: List[str] = ["living_room", "guest", "dining", "family_room"]

# --------------------------------------------------------------------------- #
# Doorways (ground floor). Derived from upstream door rules:
#   public<->public rooms: wide open archways; others: framed doors <= 2.4 m.
# `a`/`b` are the rooms it connects; `pos` is the doorway center on the wall line.
# --------------------------------------------------------------------------- #
DOORWAYS: Dict[str, dict] = {
    "door_entry_dining":      dict(a="entry",       b="dining",      pos=(-1.4, 0.6),   width=8.6,  kind="open"),
    "door_dining_living":     dict(a="dining",      b="living_room", pos=(-6.0, 3.34),  width=4.88, kind="open"),
    "door_guest_entry":       dict(a="guest",       b="entry",       pos=(-6.0, -1.95), width=2.4,  kind="door"),
    "door_guest_living":      dict(a="guest",       b="living_room", pos=(-7.6, 0.6),   width=1.8,  kind="door"),
    "door_living_family":     dict(a="living_room", b="family_room", pos=(-10.5, 6.08), width=8.4,  kind="open"),
    "door_dining_family":     dict(a="dining",      b="family_room", pos=(-4.5, 6.08),  width=2.4,  kind="open"),
    "door_dining_gallery":    dict(a="dining",      b="gallery",     pos=(0.1, 6.08),   width=5.6,  kind="open"),
    "door_entry_lobby":       dict(a="entry",       b="lobby0",      pos=(3.2, -1.95),  width=2.4,  kind="door"),
    "door_lobby_robothome":   dict(a="lobby0",      b="robot_home",  pos=(6.0, -1.95),  width=2.4,  kind="door"),
    "door_robothome_kitchen": dict(a="robot_home",  b="kitchen",     pos=(8.0, 0.6),    width=2.4,  kind="door"),
    "door_robothome_laundry": dict(a="robot_home",  b="laundry",     pos=(10.0, -1.95), width=2.4,  kind="door"),
    # NOTE: upstream also cuts a kitchen<->laundry door at (13.4, 0.6), but the laundry
    # cabinet stands right behind it, so it is NOT traversable and is left out of the graph.
    "door_kitchen_garden":    dict(a="kitchen",     b="garden_room", pos=(13.4, 6.08),  width=1.8,  kind="door"),
    "door_gallery_garden":    dict(a="gallery",     b="garden_room", pos=(6.0, 11.79),  width=10.8, kind="open"),
    "door_family_gallery":    dict(a="family_room", b="gallery",     pos=(-3.0, 11.79), width=10.8, kind="open"),
}

# Exterior doors (wall gap center) and a point just outside each one.
EXITS: Dict[str, dict] = {
    "front_door":    dict(room="entry",       pos=(-1.4, -4.5),  outside=(-1.4, -6.0), width=2.0),
    "back_door":     dict(room="living_room", pos=(-15.0, 3.6),  outside=(-16.6, 3.6), width=2.0),
    "terrace_door":  dict(room="kitchen",     pos=(15.0, 2.6),   outside=(16.6, 2.6),  width=2.2),
}

# --------------------------------------------------------------------------- #
# Named navigation targets (what guide_owner_to(target) accepts)
# --------------------------------------------------------------------------- #
LANDMARKS: Dict[str, Vec2] = {
    "front_door":         (-1.4, -3.9),   # inside, facing the front door
    "front_door_outside": EXITS["front_door"]["outside"],
    "back_door":          (-14.2, 3.6),   # inside living room, at the west garden door
    "back_exit_outside":  EXITS["back_door"]["outside"],   # smoke-alarm evacuation target
    "terrace_door":       (14.2, 2.6),
    "robot_dock":         (8.0, -2.7),    # upstream ROBOT_HOME_XY
}
SMOKE_EVACUATION_TARGET = "back_exit_outside"

# --------------------------------------------------------------------------- #
# Robot start pose (upstream START_POS_XY / START_YAW: entry hall, facing north)
# --------------------------------------------------------------------------- #
ROBOT_START_XY: Vec2 = (-1.4, -2.5)
ROBOT_START_YAW: float = math.pi / 2
ROBOT_START_ROOM = "entry"

# --------------------------------------------------------------------------- #
# Sound sources (x, y, z) — where the stereo-ear synth places each emitter.
# --------------------------------------------------------------------------- #
SOUND_SOURCES: Dict[str, dict] = {
    "doorbell":    dict(pos=(-0.2, -4.35, 1.9), room="entry"),     # chime box beside the front door
    "knock":       dict(pos=(-1.4, -4.45, 1.0), room="entry"),     # on the front door leaf
    "smoke_alarm": dict(pos=(11.0, 4.2, 2.65), room="kitchen"),    # ceiling, over the stove side
}

# --------------------------------------------------------------------------- #
# Movable chair obstacle (hotkey [C]): blocks the guest <-> living_room door,
# forcing a reroute through the entry hall.
# --------------------------------------------------------------------------- #
CHAIR_BLOCK_POS: Vec2 = DOORWAYS["door_guest_living"]["pos"]
CHAIR_PARKED_POS: Vec2 = (-13.8, -3.6)  # guest-room corner, out of the way
CHAIR_SIZE: Vec3 = (0.5, 0.5, 0.9)      # full extents (m)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def room_at(x: float, y: float) -> Optional[str]:
    """Ground-floor room key containing (x, y); None if outside the house."""
    for key, r in ROOMS.items():
        x0, y0, x1, y1 = r["rect"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return key
    return None


def room_graph() -> Dict[str, List[Tuple[str, str]]]:
    """Adjacency: room -> [(neighbor_room, doorway_name), ...]."""
    g: Dict[str, List[Tuple[str, str]]] = {k: [] for k in ROOMS}
    for name, d in DOORWAYS.items():
        g[d["a"]].append((d["b"], name))
        g[d["b"]].append((d["a"], name))
    return g


def nudge_rooms() -> List[str]:
    return [k for k, r in ROOMS.items() if r["nudge"]]
