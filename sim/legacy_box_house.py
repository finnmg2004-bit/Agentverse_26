#!/usr/bin/env python3
"""
Single source of truth for Nudge's 4-room house layout, coordinates, doorways,
landmarks, sound sources, wall segments, and MuJoCo XML scene construction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# 1. House Bounds & Room Coordinates (AGENTS.md Section 4)
# ---------------------------------------------------------------------------
HOUSE_BOUNDS = {
    "x_min": -5.0,
    "x_max": +5.0,
    "y_min": -4.0,
    "y_max": +4.0,
}

ROOMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "living_room": {
        "center": (-2.5, -2.0),
        "owner_spot": (-3.2, -2.2),
        "bounds_x": (-5.0, 0.0),
        "bounds_y": (-4.0, 0.0),
    },
    "hallway": {
        "center": (+2.5, -2.0),
        "owner_spot": (+3.2, -2.2),
        "bounds_x": (0.0, +5.0),
        "bounds_y": (-4.0, 0.0),
    },
    "kitchen": {
        "center": (-2.5, +2.0),
        "owner_spot": (-3.2, +2.2),
        "bounds_x": (-5.0, 0.0),
        "bounds_y": (0.0, +4.0),
    },
    "bedroom": {
        "center": (+2.5, +2.0),
        "owner_spot": (+3.2, +2.2),
        "bounds_x": (0.0, +5.0),
        "bounds_y": (0.0, +4.0),
    },
}

# Cycle order for the [O] / [4] hotkey
ROOM_ORDER: List[str] = ["living_room", "kitchen", "bedroom", "hallway"]

# Robot starting pose in Hallway (x, y, yaw)
ROBOT_START_POSE: Tuple[float, float, float] = (+2.5, -2.0, math.pi)

# Doorways (1.4 m wide gaps in interior walls)
DOORWAY_WIDTH: float = 1.4
DOORWAYS: Dict[str, Tuple[float, float]] = {
    "door_living_kitchen": (-2.5, 0.0),
    "door_hallway_bedroom": (+2.5, 0.0),
    "door_living_hallway": (0.0, -2.0),
    "door_kitchen_bedroom": (0.0, +2.0),
}

# Room adjacency -> doorway name
ROOM_GRAPH: Dict[Tuple[str, str], str] = {
    ("living_room", "kitchen"): "door_living_kitchen",
    ("kitchen", "living_room"): "door_living_kitchen",
    ("hallway", "bedroom"): "door_hallway_bedroom",
    ("bedroom", "hallway"): "door_hallway_bedroom",
    ("living_room", "hallway"): "door_living_hallway",
    ("hallway", "living_room"): "door_living_hallway",
    ("kitchen", "bedroom"): "door_kitchen_bedroom",
    ("bedroom", "kitchen"): "door_kitchen_bedroom",
}

# Exits / Landmarks
LANDMARKS: Dict[str, Tuple[float, float]] = {
    "front_door": (+2.5, -3.7),
    "back_door": (+2.5, +3.7),
    "back_exit_outside": (+2.5, +4.6),
}

# Sound Sources (x, y, z) and their home room / landmark
SOUND_SOURCES: Dict[str, Dict[str, object]] = {
    "doorbell": {
        "pos": (2.5, -3.9, 0.9),
        "room": "hallway",
        "target_landmark": "front_door",
        "alert_mode": "1_pulse",
        "rgba_active": (0.1, 0.85, 1.0, 0.95),
        "rgba_idle": (0.1, 0.4, 0.5, 0.35),
    },
    "knock": {
        "pos": (2.1, -3.9, 0.7),
        "room": "hallway",
        "target_landmark": "front_door",
        "alert_mode": "2_pulse",
        "rgba_active": (1.0, 0.75, 0.1, 0.95),
        "rgba_idle": (0.5, 0.38, 0.1, 0.35),
    },
    "smoke_alarm": {
        "pos": (-2.5, 3.8, 1.1),
        "room": "kitchen",
        "target_landmark": "back_exit_outside",
        "alert_mode": "continuous",
        "rgba_active": (1.0, 0.15, 0.15, 0.98),
        "rgba_idle": (0.5, 0.1, 0.1, 0.35),
    },
}

# Chair obstacle positions for [C] / [5] hotkey
# Placed in the living_room <-> kitchen doorway (-2.5, 0.0)
CHAIR_DOORWAY_POS: Tuple[float, float, float] = (-2.5, 0.0, 0.0)
CHAIR_STOWED_POS: Tuple[float, float, float] = (-4.5, +3.4, 0.0)
CHAIR_COLLISION_RADIUS: float = 0.22

# Wall geometry specification: (name, center_x, center_y, center_z, half_x, half_y, half_z)
WALL_HEIGHT: float = 1.2
WALL_HALF_Z: float = WALL_HEIGHT / 2.0
WALL_THICK_HALF: float = 0.07

WALL_BOXES: List[Tuple[str, float, float, float, float, float, float]] = [
    # Exterior South wall (y = -4.0, x in [-5, +5])
    ("wall_south", 0.0, -4.0, WALL_HALF_Z, 5.0, WALL_THICK_HALF, WALL_HALF_Z),
    # Exterior West wall (x = -5.0, y in [-4, +4])
    ("wall_west", -5.0, 0.0, WALL_HALF_Z, WALL_THICK_HALF, 4.0, WALL_HALF_Z),
    # Exterior East wall (x = +5.0, y in [-4, +4])
    ("wall_east", +5.0, 0.0, WALL_HALF_Z, WALL_THICK_HALF, 4.0, WALL_HALF_Z),
    # Exterior North wall (y = +4.0) with 1.4m Back Door opening at x = +2.5 (gap [1.8, 3.2])
    # Segment 1: x in [-5.0, +1.8] -> center -1.6, half_x 3.4
    ("wall_north_west", -1.6, +4.0, WALL_HALF_Z, 3.4, WALL_THICK_HALF, WALL_HALF_Z),
    # Segment 2: x in [+3.2, +5.0] -> center +4.1, half_x 0.9
    ("wall_north_east", +4.1, +4.0, WALL_HALF_Z, 0.9, WALL_THICK_HALF, WALL_HALF_Z),
    # Interior Horizontal Wall (y = 0.0) with gaps at (-2.5, 0) [-3.2, -1.8] and (+2.5, 0) [+1.8, +3.2]
    # Segment H1: x in [-5.0, -3.2] -> center -4.1, half_x 0.9
    ("wall_int_h_west", -4.1, 0.0, WALL_HALF_Z, 0.9, WALL_THICK_HALF, WALL_HALF_Z),
    # Segment H2: x in [-1.8, +1.8] -> center 0.0, half_x 1.8
    ("wall_int_h_mid", 0.0, 0.0, WALL_HALF_Z, 1.8, WALL_THICK_HALF, WALL_HALF_Z),
    # Segment H3: x in [+3.2, +5.0] -> center +4.1, half_x 0.9
    ("wall_int_h_east", +4.1, 0.0, WALL_HALF_Z, 0.9, WALL_THICK_HALF, WALL_HALF_Z),
    # Interior Vertical Wall (x = 0.0) with gaps at (0, -2.0) [-2.7, -1.3] and (0, +2.0) [+1.3, +2.7]
    # Segment V1: y in [-4.0, -2.7] -> center -3.35, half_y 0.65
    ("wall_int_v_south", 0.0, -3.35, WALL_HALF_Z, WALL_THICK_HALF, 0.65, WALL_HALF_Z),
    # Segment V2: y in [-1.3, +1.3] -> center 0.0, half_y 1.3
    ("wall_int_v_mid", 0.0, 0.0, WALL_HALF_Z, WALL_THICK_HALF, 1.3, WALL_HALF_Z),
    # Segment V3: y in [+2.7, +4.0] -> center +3.35, half_y 0.65
    ("wall_int_v_north", 0.0, +3.35, WALL_HALF_Z, WALL_THICK_HALF, 0.65, WALL_HALF_Z),
]

# 2D Line segments ((x1, y1), (x2, y2)) for fast acoustic occlusion & dashboard drawing
WALL_SEGMENTS_2D: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((-5.0, -4.0), (+5.0, -4.0)),
    ((-5.0, -4.0), (-5.0, +4.0)),
    ((+5.0, -4.0), (+5.0, +4.0)),
    ((-5.0, +4.0), (+1.8, +4.0)),
    ((+3.2, +4.0), (+5.0, +4.0)),
    ((-5.0, 0.0), (-3.2, 0.0)),
    ((-1.8, 0.0), (+1.8, 0.0)),
    ((+3.2, 0.0), (+5.0, 0.0)),
    ((0.0, -4.0), (0.0, -2.7)),
    ((0.0, -1.3), (0.0, +1.3)),
    ((0.0, +2.7), (0.0, +4.0)),
]


def get_room_at(x: float, y: float) -> str:
    """Return the canonical room name for world coordinates (x, y)."""
    if y > 4.05:
        return "back_exit_outside"
    if x < 0.0 and y < 0.0:
        return "living_room"
    if x >= 0.0 and y < 0.0:
        return "hallway"
    if x < 0.0 and y >= 0.0:
        return "kitchen"
    return "bedroom"


def get_landmark_room(target: str) -> str:
    """Return the room associated with a named room or landmark."""
    if target in ROOMS:
        return target
    if target == "front_door":
        return "hallway"
    if target in ("back_door", "back_exit_outside"):
        return "bedroom"
    return "hallway"


def count_walls_crossed(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    """Count how many 2D wall segments intersect the line segment p1 -> p2."""
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def intersects(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
        d: Tuple[float, float],
    ) -> bool:
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    count = 0
    for seg_a, seg_b in WALL_SEGMENTS_2D:
        if intersects(p1, p2, seg_a, seg_b):
            count += 1
    return count


def populate_house_xml(root: ET.Element) -> None:
    """
    Injects the 4-room house, floor carpets, doorway markers, head camera,
    Owner humanoid mocap body, movable Chair mocap body, and sound beacons into
    the Unitree Go2 MuJoCo XML tree.
    """
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.24 0.36 0.52",
        rgb2="0.06 0.08 0.12",
        width="512",
        height="3072",
    )
    ET.SubElement(
        asset,
        "texture",
        name="groundplane",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1="0.20 0.23 0.28",
        rgb2="0.15 0.18 0.22",
        markrgb="0.32 0.36 0.42",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="4 4",
        reflectance="0.08",
    )
    ET.SubElement(
        asset,
        "material",
        name="wall_mat",
        rgba="0.84 0.82 0.78 1.0",
    )
    ET.SubElement(
        asset,
        "material",
        name="door_mat",
        rgba="0.52 0.32 0.18 1.0",
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid go2.xml: missing <worldbody>")

    # 1. Attach head_cam and haptic alert halo to Go2 `base` body
    base_body = worldbody.find("body[@name='base']")
    if base_body is not None:
        # In MuJoCo cameras look along -Z by default, +X right, +Y up.
        # To look forward along robot +X with +Z up:
        # camera axes in base frame: cam_X = (0, -1, 0), cam_Y = (0, 0, 1), cam_Z = (-1, 0, 0)
        ET.SubElement(
            base_body,
            "camera",
            name="head_cam",
            pos="0.30 0.0 0.12",
            xyaxes="0 -1 0 0 0 1",
            fovy="60",
        )
        # Guide handle visual bar + haptic pulse indicator sphere on the back of Go2
        ET.SubElement(
            base_body,
            "geom",
            name="guide_handle_bar",
            type="capsule",
            fromto="-0.05 0.08 0.08 -0.22 0.14 0.24",
            size="0.014",
            rgba="0.9 0.85 0.2 1.0",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            base_body,
            "geom",
            name="alert_halo",
            type="sphere",
            pos="-0.15 0.0 0.26",
            size="0.065",
            rgba="0.2 0.9 0.4 0.0",
            contype="0",
            conaffinity="0",
        )

    # 2. Overhead lights over each room quadrant
    for light_idx, (lx, ly) in enumerate([(-2.5, -2.0), (2.5, -2.0), (-2.5, 2.0), (2.5, 2.0)]):
        ET.SubElement(
            worldbody,
            "light",
            name=f"room_light_{light_idx}",
            pos=f"{lx} {ly} 3.8",
            dir="0 0 -1",
            diffuse="0.55 0.55 0.52",
            specular="0.15 0.15 0.15",
            directional="false",
            castshadow="true",
        )

    # 3. Floor plane + subtle room-colored floor mats (non-colliding visual pads)
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="12 12 0.1",
        material="groundplane",
    )

    room_rugs = [
        ("rug_living_room", "-2.5 -2.0 0.002", "2.1 1.6 0.002", "0.24 0.34 0.46 0.55"),
        ("rug_hallway", "2.5 -2.0 0.002", "2.1 1.6 0.002", "0.36 0.32 0.26 0.55"),
        ("rug_kitchen", "-2.5 2.0 0.002", "2.1 1.6 0.002", "0.25 0.40 0.32 0.55"),
        ("rug_bedroom", "2.5 2.0 0.002", "2.1 1.6 0.002", "0.38 0.26 0.40 0.55"),
        ("rug_back_exit", "2.5 4.6 0.003", "0.9 0.5 0.002", "0.15 0.65 0.28 0.70"),
    ]
    for name, pos, size, rgba in room_rugs:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            rgba=rgba,
            contype="0",
            conaffinity="0",
        )

    # 4. House walls (4 rooms with 1.4m doorways)
    for name, cx, cy, cz, hx, hy, hz in WALL_BOXES:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=f"{cx} {cy} {cz}",
            size=f"{hx} {hy} {hz}",
            material="wall_mat",
        )

    # Front door visual panel on South wall at (+2.5, -3.91)
    ET.SubElement(
        worldbody,
        "geom",
        name="front_door_panel",
        type="box",
        pos="2.5 -3.91 0.58",
        size="0.55 0.02 0.56",
        material="door_mat",
        contype="0",
        conaffinity="0",
    )

    # 5. Sound Source Beacons (glowing spheres at exact sound source coordinates)
    for sound_name, spec in SOUND_SOURCES.items():
        sx, sy, sz = spec["pos"]
        r, g, b, a = spec["rgba_idle"]
        ET.SubElement(
            worldbody,
            "geom",
            name=f"beacon_{sound_name}",
            type="sphere",
            pos=f"{sx} {sy} {sz}",
            size="0.14",
            rgba=f"{r} {g} {b} {a}",
            contype="0",
            conaffinity="0",
        )

    # 6. Owner Humanoid Mocap Body (starts in Living Room at (-3.2, -2.2))
    # Designed with realistic human proportions + high-contrast teal jacket & skin head
    # so both YOLOv8n and depth/color detectors localize the owner reliably, and LiDAR sees legs/torso.
    ox, oy = ROOMS["living_room"]["owner_spot"]
    owner_body = ET.SubElement(
        worldbody,
        "body",
        name="owner_body",
        mocap="true",
        pos=f"{ox} {oy} 0.0",
    )
    # Legs (collidable for LiDAR at z=0.32m)
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_leg_left",
        type="capsule",
        fromto="0.0 0.11 0.06 0.0 0.11 0.72",
        size="0.075",
        rgba="0.16 0.20 0.30 1.0",
    )
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_leg_right",
        type="capsule",
        fromto="0.0 -0.11 0.06 0.0 -0.11 0.72",
        size="0.075",
        rgba="0.16 0.20 0.30 1.0",
    )
    # Lower torso / hips (collidable for LiDAR at z=0.32m)
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_hips",
        type="box",
        pos="0.0 0.0 0.42",
        size="0.12 0.19 0.26",
        rgba="0.05 0.68 0.85 1.0",
    )
    # Upper torso (distinctive bright cyan/teal jacket for visual detection)
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_torso",
        type="box",
        pos="0.0 0.0 1.02",
        size="0.14 0.22 0.30",
        rgba="0.05 0.72 0.88 1.0",
    )
    # Arms
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_arm_left",
        type="capsule",
        fromto="0.0 0.26 1.24 0.08 0.26 0.76",
        size="0.055",
        rgba="0.05 0.72 0.88 1.0",
    )
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_arm_right",
        type="capsule",
        fromto="0.0 -0.26 1.24 0.08 -0.26 0.76",
        size="0.055",
        rgba="0.05 0.72 0.88 1.0",
    )
    # Head (warm skin tone)
    ET.SubElement(
        owner_body,
        "geom",
        name="owner_head",
        type="sphere",
        pos="0.0 0.0 1.46",
        size="0.13",
        rgba="0.92 0.74 0.60 1.0",
    )

    # 7. Movable Chair Obstacle Mocap Body (starts stowed at CHAIR_STOWED_POS)
    cx, cy, cz = CHAIR_STOWED_POS
    chair_body = ET.SubElement(
        worldbody,
        "body",
        name="chair_body",
        mocap="true",
        pos=f"{cx} {cy} {cz}",
    )
    # Chair seat + solid lower base so 2D LiDAR at z=0.32m detects it clearly
    ET.SubElement(
        chair_body,
        "geom",
        name="chair_base",
        type="cylinder",
        pos="0.0 0.0 0.26",
        size="0.22 0.24",
        rgba="0.72 0.42 0.16 1.0",
    )
    ET.SubElement(
        chair_body,
        "geom",
        name="chair_back",
        type="box",
        pos="-0.16 0.0 0.62",
        size="0.04 0.20 0.22",
        rgba="0.65 0.36 0.12 1.0",
    )
