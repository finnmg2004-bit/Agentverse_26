"""
Nudge house layout — single source of truth for every coordinate the project uses.

The world geometry comes from the nerv-world "house" scene (git submodule at
worlds/nerv-world, MIT, Yanshi-Robotics). Its own truth is
worlds/nerv-world/scenes/house/layout.py; the values below are copied from that
file (room rectangles) or derived from its door rules, and verified against the
compiled scene by `pixi run check-world` (sim/check_world.py).

Nudge only uses the GROUND FLOOR (z = 0). The Go2 in Plan B cannot climb stairs,
so floors 1-2 and the stair core are out of bounds.

The public names/shapes (ROOMS[...]["center"/"owner_spot"/"bounds_x"/"bounds_y"],
DOORWAYS as (x, y), ROOM_GRAPH, ROOM_ORDER, LANDMARKS, SOUND_SOURCES, CHAIR_*,
ROBOT_START_POSE, WALL_SEGMENTS_2D, get_room_at, count_walls_crossed, ...) are the
same as the original 10x8 m box-house module, so sim and brain code is unchanged.

World frame: +X = east, +Y = north, +Z = up, metres. Yaw 0 faces +X.

         y=17.5 +--------------------+-----------------+----------------+
                |    family_room     |     gallery     |  garden_room   |
         y=6.08 +====[ open ]=====+=[open]=+=[open]=+--+-----+--[door]--+
                |   living_room   =[open] dining    |  stair | kitchen  =[door]-> pool terrace (E)
   back door <-=[door]            |                 |  core  |   (smoke)|
         y=0.6  +--[door]---------+=====[ open ]====+--------+--[door]--+
                |  guest (bed)    =[door]  entry    =[door]  =[door]    |
                |                 |  (robot start)  | lobby0 |robot_home| laundry
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
NAV_MAP_FILE = "assets/maps/nerv_house_nav.npz"            # static occupancy map (sim/nav_map.py)

FLOOR_Z = 0.0
WALL_HEIGHT = 2.73          # interior clear height (upstream STOREY_H - slabs)
WALL_THICK = 0.14

HOUSE_BOUNDS = {"x_min": -15.0, "x_max": 15.0, "y_min": -4.5, "y_max": 17.5}  # ground-floor footprint
OUTSIDE = "outside"         # pseudo-room returned by get_room_at() outside the house

# --------------------------------------------------------------------------- #
# Rooms (ground floor). Keys match upstream layout.py room keys.
# --------------------------------------------------------------------------- #
_ROOM_DEFS: Dict[str, Tuple[str, Rect]] = {
    "entry":       ("Entry hall",         (-6.0, -4.5, 3.2, 0.6)),
    "guest":       ("Guest bedroom",      (-15.0, -4.5, -6.0, 0.6)),
    "living_room": ("Living room",        (-15.0, 0.6, -6.0, 6.08)),
    "dining":      ("Dining room",        (-6.0, 0.6, 3.2, 6.08)),
    "family_room": ("Garden family room", (-15.0, 6.08, -3.0, 17.5)),
    "kitchen":     ("Kitchen",            (6.0, 0.6, 15.0, 6.08)),
    "lobby0":      ("Stair lobby",        (3.2, -4.5, 6.0, 0.6)),
    "robot_home":  ("Robot dock",         (6.0, -4.5, 10.0, 0.6)),
    "gallery":     ("Garden gallery",     (-3.0, 6.08, 6.0, 17.5)),
    "garden_room": ("Garden lounge",      (6.0, 6.08, 15.0, 17.5)),
    "laundry":     ("Laundry",            (10.0, -4.5, 15.0, 0.6)),
}
# Stair core: physically there (walls), but never a navigation target.
STAIR_CORE: Rect = (3.2, 0.6, 6.0, 6.08)

# Clear floor point the robot drives to for "go to room" / look_around (furniture-free).
ROOM_SCAN_POINTS: Dict[str, Vec2] = {
    "entry":       (-1.4, -2.0),
    "guest":       (-9.5, -2.0),
    "living_room": (-8.5, 4.6),
    "dining":      (-4.5, 1.6),
    "family_room": (-8.0, 9.0),
    "kitchen":     (8.0, 2.4),
    "lobby0":      (4.6, -2.0),
    "robot_home":  (8.0, -2.0),
    "gallery":     (1.5, 11.0),
    "garden_room": (12.5, 13.5),
    "laundry":     (12.5, -2.2),
}

# Where the owner stands in each room (clear of furniture, verified by check_world).
OWNER_SPOTS: Dict[str, Vec2] = {
    "living_room": (-11.0, 5.0),
    "guest":       (-10.0, -2.6),
    "dining":      (-4.6, 1.5),
    "family_room": (-8.0, 9.5),
    "entry":       (-3.6, -2.2),
    "kitchen":     (8.4, 3.2),
}
OWNER_START_ROOM = "living_room"

# Hotkey [O] cycles the owner through these rooms; the brain also searches them in this order.
OWNER_ROOM_CYCLE: List[str] = ["living_room", "guest", "dining", "family_room"]
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
        "center": _center,                       # navigation point (clear floor)
        "geo_center": ((_x0 + _x1) / 2.0, (_y0 + _y1) / 2.0),
        "owner_spot": OWNER_SPOTS.get(_key, _center),
        "search": _key in ROOM_ORDER,
    }
ROOM_CENTERS: Dict[str, Vec2] = {k: r["center"] for k, r in ROOMS.items()}

# --------------------------------------------------------------------------- #
# Doorways. Derived from upstream door rules (public<->public rooms: wide open archways;
# others: framed doors <= 2.4 m). `pos` = centre of the gap on the wall line,
# `axis` = "x" if the wall is vertical (cross along x), "y" if horizontal.
# --------------------------------------------------------------------------- #
DOORWAY_INFO: Dict[str, dict] = {
    "door_entry_dining":      dict(a="entry",       b="dining",      pos=(-1.4, 0.6),   width=8.6,  axis="y", kind="open"),
    "door_dining_living":     dict(a="dining",      b="living_room", pos=(-6.0, 3.34),  width=4.88, axis="x", kind="open"),
    "door_guest_entry":       dict(a="guest",       b="entry",       pos=(-6.0, -1.95), width=2.4,  axis="x", kind="door"),
    "door_guest_living":      dict(a="guest",       b="living_room", pos=(-7.6, 0.6),   width=1.8,  axis="y", kind="door"),
    "door_living_family":     dict(a="living_room", b="family_room", pos=(-10.5, 6.08), width=8.4,  axis="y", kind="open"),
    "door_dining_family":     dict(a="dining",      b="family_room", pos=(-4.5, 6.08),  width=2.4,  axis="y", kind="open"),
    "door_dining_gallery":    dict(a="dining",      b="gallery",     pos=(0.1, 6.08),   width=5.6,  axis="y", kind="open"),
    "door_entry_lobby":       dict(a="entry",       b="lobby0",      pos=(3.2, -1.95),  width=2.4,  axis="x", kind="door"),
    "door_lobby_robothome":   dict(a="lobby0",      b="robot_home",  pos=(6.0, -1.95),  width=2.4,  axis="x", kind="door"),
    "door_robothome_kitchen": dict(a="robot_home",  b="kitchen",     pos=(8.0, 0.6),    width=2.4,  axis="y", kind="door"),
    "door_robothome_laundry": dict(a="robot_home",  b="laundry",     pos=(10.0, -1.95), width=2.4,  axis="x", kind="door"),
    "door_kitchen_garden":    dict(a="kitchen",     b="garden_room", pos=(13.4, 6.08),  width=1.8,  axis="y", kind="door"),
    "door_gallery_garden":    dict(a="gallery",     b="garden_room", pos=(6.0, 11.79),  width=10.8, axis="x", kind="open"),
    "door_family_gallery":    dict(a="family_room", b="gallery",     pos=(-3.0, 11.79), width=10.8, axis="x", kind="open"),
}
# Gaps that exist in the walls but are not traversable / not navigation edges.
# (kitchen<->laundry: blocked by the laundry cabinet; stair core entrance: no stairs for us.)
WALL_GAPS_ONLY: List[dict] = [
    dict(pos=(13.4, 0.6), width=1.8, axis="y"),
    dict(pos=(4.6, 0.6), width=1.2, axis="y"),
]

# Exterior doors (wall gap centre) and a point just outside each one.
EXITS: Dict[str, dict] = {
    "exit_front":   dict(room="entry",       pos=(-1.4, -4.5), outside=(-1.4, -6.0), width=2.0, axis="y"),
    "exit_back":    dict(room="living_room", pos=(-15.0, 3.6), outside=(-16.6, 3.6), width=2.0, axis="x"),
    "exit_terrace": dict(room="kitchen",     pos=(15.0, 2.6),  outside=(16.6, 2.6),  width=2.2, axis="x"),
}

# Legacy-shaped views: name -> (x, y)  (interior doorways + exterior doors)
DOORWAYS: Dict[str, Vec2] = {k: d["pos"] for k, d in DOORWAY_INFO.items()}
DOORWAYS.update({k: e["pos"] for k, e in EXITS.items()})
DOORWAY_WIDTH: Dict[str, float] = {k: d["width"] for k, d in DOORWAY_INFO.items()}
DOORWAY_WIDTH.update({k: e["width"] for k, e in EXITS.items()})

# Room adjacency (both directions) -> doorway name. Exits connect a room to OUTSIDE.
ROOM_GRAPH: Dict[Tuple[str, str], str] = {}
for _name, _d in DOORWAY_INFO.items():
    ROOM_GRAPH[(_d["a"], _d["b"])] = _name
    ROOM_GRAPH[(_d["b"], _d["a"])] = _name
for _name, _e in EXITS.items():
    ROOM_GRAPH[(_e["room"], OUTSIDE)] = _name
    ROOM_GRAPH[(OUTSIDE, _e["room"])] = _name

# --------------------------------------------------------------------------- #
# Named navigation targets (what guide_owner_to(target) accepts)
# --------------------------------------------------------------------------- #
LANDMARKS: Dict[str, Vec2] = {
    "front_door":         (-1.4, -3.9),                    # inside the entry, facing the front door
    "front_door_outside": EXITS["exit_front"]["outside"],
    "back_door":          (-14.2, 3.6),                    # inside living room, at the west garden door
    "back_exit_outside":  EXITS["exit_back"]["outside"],   # smoke-alarm evacuation target
    "terrace_door":       (14.2, 2.6),
    "terrace_outside":    EXITS["exit_terrace"]["outside"],  # alternative evacuation target (kitchen exit)
    "robot_dock":         (8.0, -2.7),                     # upstream ROBOT_HOME_XY
    "kettle":             (13.2, 4.2),                     # kitchen, in front of the stove (kettle on the hob)
    "stove":              (13.2, 4.2),                     # same spot: oven timer / cooking jobs
}
SMOKE_EVACUATION_TARGET = "back_exit_outside"

# --------------------------------------------------------------------------- #
# Robot start pose (upstream START_POS_XY / START_YAW: entry hall, facing north)
# --------------------------------------------------------------------------- #
ROBOT_START_XY: Vec2 = (-1.4, -2.5)
ROBOT_START_YAW: float = math.pi / 2
ROBOT_START_POSE: Vec3 = (ROBOT_START_XY[0], ROBOT_START_XY[1], ROBOT_START_YAW)
ROBOT_START_ROOM = "entry"

# --------------------------------------------------------------------------- #
# Sound registry (AGENTS.md 2.1 / 9.4) — one entry per sound; adding a sound is a data change here
# plus a clip in assets/audio/ (sim/sensors.py synthesizes placeholders) and, if YAMNet can't
# recognise the clip, a spectral rule in nudge-brain/hearing.py.
#   pos              (x, y, z) where the stereo-ear synth places the emitter
#   room             room the source is in
#   target_landmark  where Nudge leads the owner (LANDMARKS key)
#   tier             1 = emergency (pre-empts everything), 2 = needs action, 3 = someone wants you
#   alert_mode       haptic handle code sent in `cmd` (legacy: knock keeps 2_pulse until the
#                    per-tier patterns of 2.1 land; the tier is the source of truth for priority)
#   level            loudness (1.0 = doorbell); wall occlusion 0.45x/wall, capped at 3 walls
#   file             clip in assets/audio/
#   yamnet           YAMNet (AudioSet) class names whose max score is this sound's YAMNet score
#   label / say      display name and the default "Nudge says" message (when Gemma writes none)
# --------------------------------------------------------------------------- #
SOUND_SOURCES: Dict[str, Dict[str, object]] = {
    "doorbell": {
        "pos": (-0.2, -4.35, 1.9),          # chime box beside the front door
        "room": "entry",
        "target_landmark": "front_door",
        "tier": 3,
        "alert_mode": "1_pulse",
        "level": 1.0,
        "file": "doorbell.wav",
        "yamnet": ("Doorbell", "Ding-dong"),
        "label": "Doorbell",
        # P8: a real visitor rings, waits, rings again, then gives up. The sim plays this pattern only with
        # NUDGE_INTERMITTENT=1; SoundEvents always holds a stop through the pause (one visitor = one event).
        "ring_pattern": {"ring_s": 3.0, "pause_s": 6.0, "give_up_s": 60.0},
        "event_quiet_below": 0.5,           # SoundEvents: below this counts as a pause (TV babble ~0.3 doesn't)
        "say": "Doorbell. Hold my handle. I will take you to the front door.",
        "rgba_active": (0.1, 0.85, 1.0, 0.95),
        "rgba_idle": (0.1, 0.4, 0.5, 0.35),
    },
    "knock": {
        "pos": (-1.4, -4.45, 1.0),          # on the front door leaf
        "room": "entry",
        "target_landmark": "front_door",
        "tier": 3,
        "alert_mode": "2_pulse",
        "level": 1.0,
        "file": "knock.wav",
        "yamnet": ("Knock",),
        "label": "Knock",
        "event_stop_windows": 8,            # SoundEvents: knocking comes in bursts -> 4 s quiet = stopped
        "ring_pattern": {"ring_s": 2.0, "pause_s": 5.0, "give_up_s": 50.0},  # P8, see doorbell
        "say": "Someone is knocking. Hold my handle. I will take you to the front door.",
        "rgba_active": (1.0, 0.75, 0.1, 0.95),
        "rgba_idle": (0.5, 0.38, 0.1, 0.35),
    },
    "smoke_alarm": {
        "pos": (11.0, 4.2, 2.55),           # kitchen ceiling
        "room": "kitchen",
        "target_landmark": SMOKE_EVACUATION_TARGET,
        "tier": 1,
        "alert_mode": "continuous",
        "level": 4.0,                     # loud: must carry through the whole house
        "file": "smoke_alarm.wav",
        "yamnet": ("Smoke detector, smoke alarm", "Fire alarm"),
        "label": "Smoke alarm",
        "event_stop_windows": 6,            # SoundEvents: real alarms pause ~1.5 s (temporal-3) -> 3 s quiet = stopped
        "say": "Smoke alarm! Hold my handle. We are going outside now.",
        "rgba_active": (1.0, 0.15, 0.15, 0.98),
        "rgba_idle": (0.5, 0.1, 0.1, 0.35),
    },
    "kettle": {                           # tier 2 "needs action": the kettle has boiled
        "pos": (12.55, 5.2, 1.0),           # on the stove hob (upstream h2_stove at 12.7, 5.25)
        "room": "kitchen",
        "target_landmark": "kettle",
        "tier": 2,
        "alert_mode": "2_pulse",
        "level": 2.0,                     # a whistling kettle is loud; heard from the entry / living room
        "file": "kettle.wav",
        "yamnet": ("Whistle", "Steam whistle"),
        "label": "Kettle",
        "say": "The kettle is boiling. Hold my handle. I will take you to the kitchen.",
        "rgba_active": (1.0, 0.55, 0.05, 0.95),
        "rgba_idle": (0.5, 0.28, 0.05, 0.35),
    },
    "oven_timer": {                       # tier 2 "needs action": the oven timer is beeping
        "pos": (12.85, 5.55, 1.05),         # stove control panel (upstream h2_stove at 12.7, 5.25)
        "room": "kitchen",
        "target_landmark": "stove",
        "tier": 2,
        "alert_mode": "2_pulse",
        "level": 1.5,
        "file": "oven_timer.wav",
        # No YAMNet class: "Beep, bleep" is generic (0.4-0.7 on the kettle whistle and the smoke
        # alarm) and doesn't fire on the timer clip itself (0.01). Detection = spectral rule only.
        "yamnet": (),
        "label": "Oven timer",
        "say": "The oven timer is beeping. Hold my handle. I will take you to the stove.",
        "rgba_active": (1.0, 0.85, 0.3, 0.95),
        "rgba_idle": (0.5, 0.42, 0.15, 0.35),
    },
}
SOUND_TIERS: Dict[str, int] = {name: int(spec["tier"]) for name, spec in SOUND_SOURCES.items()}
DANGER_SOUNDS = tuple(n for n, t in SOUND_TIERS.items() if t == 1)
# Sounds that start an ordinary (non-emergency) mission, highest priority first.
ALERT_SOUNDS = tuple(sorted((n for n, t in SOUND_TIERS.items() if t > 1), key=lambda n: SOUND_TIERS[n]))

# The owner's own voice. Not a registry sound (it never starts a mission): whenever the owner talks to
# Nudge (voice mic or GUI text box) the sim plays this clip from the owner's position into the robot's
# ears, and the brain localises it like any other sound to work out WHERE the owner is.
OWNER_VOICE: Dict[str, object] = {
    "file": "owner_voice.wav",      # scripts/make_owner_voice.py
    "level": 0.8,                   # normal speech, a bit quieter than the doorbell
    "yamnet": ("Speech", "Male speech, man speaking", "Female speech, woman speaking",
               "Narration, monologue", "Conversation", "Speech synthesizer"),
    "words_per_s": 2.5,             # how long the owner talks for a sentence of N words
    "min_s": 2.0,
    "max_s": 6.0,
}

# The living-room TV (AGENTS.md 6.2 "Is the sound real?"): a programme on it can play a doorbell or a
# knock. Not a registry sound: the sim plays the registry clip FROM THE TV (hotkey [Y] / gui action
# {"action": "tv", "play": "doorbell"|"knock"|"off"}), the brain hears "doorbell" and must work out from
# direction + loudness that it came from the TV, not the front door (nudge-brain/sound_check.py).
# Position = Sean's wall-mounted TV stand-in (sim/coco_standins.py `coco_tv`, north wall of the
# living room), 0.1 m in front of the screen so it is on the living-room side of the wall.
TV_SOURCE: Dict[str, object] = {
    "pos": (-9.0, 5.9, 1.35),
    "room": "living_room",
    "level": 0.8,                    # TV volume: a bit quieter than the real doorbell (1.0)
    "plays": ("doorbell", "knock"),  # sounds the TV can play (and the brain double-checks)
    "duration_s": 6.0,               # one TV doorbell lasts this long, then the programme moves on
}

# --------------------------------------------------------------------------- #
# Movable chair obstacle (hotkey [C]): blocks the guest <-> living_room door,
# forcing a reroute through the entry hall.
# --------------------------------------------------------------------------- #
CHAIR_DOORWAY_POS: Vec3 = (DOORWAY_INFO["door_guest_living"]["pos"][0], DOORWAY_INFO["door_guest_living"]["pos"][1], 0.0)
CHAIR_STOWED_POS: Vec3 = (-13.8, -3.6, 0.0)  # guest-room corner, out of the way
# Hotkey [E]: the chair falls into the back (garden) door gap -> back exit blocked, the
# evacuation planner must pick another exit.
CHAIR_EXIT_POS: Vec3 = (EXITS["exit_back"]["pos"][0], EXITS["exit_back"]["pos"][1], 0.0)

# Each exterior exit -> the LANDMARKS name just outside it (evacuation targets).
EXIT_TARGETS: Dict[str, str] = {
    "exit_front": "front_door_outside",
    "exit_back": "back_exit_outside",
    "exit_terrace": "terrace_outside",
}
CHAIR_COLLISION_RADIUS: float = 0.22

# --------------------------------------------------------------------------- #
# Avoid areas (AGENTS.md 9.4): named floor rectangles (x0, y0, x1, y1) the owner can
# tell the robot to keep off ("I've got a new carpet in the living room, please don't
# step on it"). Not obstacles: the static nav map ignores them; the brain's planner
# avoids an area only while a matching rule is active, and a danger sound (smoke) may
# override it. The sim draws each one as a flat rug and reports (ground truth) when the
# robot steps on it ("[sim] AVOID AREA ...", `in_avoid_area` in `state`).
#
# living_room_carpet sits on the direct diagonal from the guest-room door to the owner
# (-11, 5): with no rule the robot walks across it; with the rule it goes in through the
# dining-room door and along the north wall instead (~1.2 m longer). It keeps clear of
# the owner's spot, every living-room doorway and the back (evacuation) door.
# --------------------------------------------------------------------------- #
AVOID_AREAS: Dict[str, Tuple[float, float, float, float]] = {
    "living_room_carpet": (-10.0, 2.4, -7.6, 4.4),
}
AVOID_AREA_LABELS: Dict[str, str] = {
    "living_room_carpet": "the new carpet in the living room",
}
# Robot footprint half-size used when judging "stepped on": the Go2 body centre is ~0.3 m
# from its feet, so the robot counts as on the area once its centre is within this margin.
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
