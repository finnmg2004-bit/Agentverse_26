"""
Owner poses beyond standing (#8): the owner lies on their back in the guest bed while `asleep`.

Behind NUDGE_OWNER_IN_BED=1 (default off; standing at the guest owner spot as before). The scanned owner
is one rigid posed mesh, so "lying" = the whole owner body rotated onto the mattress (face up, head on
the pillows at the north end), feet at the foot of the bed, where the robot can reach them for the wake-up
nudge (owner_behavior.NUDGE_CONTACT is measured to the body origin = the feet). On waking the owner stands
up beside the bed. The bed is read from the upstream scene (geom `furn_h2_guest_mattress`), not hard-coded.

A sitting pose (sofa) would need a seated scan: the free Renderpeople posed models are rigid, not rigged.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import mujoco
import numpy as np

OWNER_IN_BED = os.environ.get("NUDGE_OWNER_IN_BED", "0").lower() not in ("0", "", "false", "off")
MATTRESS_GEOM = "furn_h2_guest_mattress"
BODY_HALF_DEPTH = 0.25   # scanned owner: 0.51 m front-to-back -> back on the mattress when lying face up
BODY_LENGTH = 1.73       # feet to head
EDGE_GAP = 0.26          # body centre line this far inside the mattress's east edge (the side the robot comes from)
ARM_REACH = 0.12         # her arm lies at the mattress edge: a nudge against the bed side there is felt
FOOT_MARGIN = 0.02       # feet this far inside the foot end of the mattress

# Body frame of the owner: +Z up (head), +X forward (face). Lying face up with the head to world +Y:
# body Z -> world +Y, body X -> world +Z (face up), body Y -> world +X.
_LYING = np.column_stack([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
LYING_QUAT = np.zeros(4)
mujoco.mju_mat2Quat(LYING_QUAT, _LYING.reshape(-1))


def _mattress(model: mujoco.MjModel, data: mujoco.MjData) -> Optional[Tuple[float, float, float, float, float]]:
    """(x_centre, y_centre, half_x, half_y, top_z) of the guest mattress, or None if the scene lacks it."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, MATTRESS_GEOM)
    if gid < 0:
        return None
    x, y, z = data.geom_xpos[gid]
    hx, hy, hz = model.geom_size[gid]
    return float(x), float(y), float(hx), float(hy), float(z + hz)


def lying_pose(model: mujoco.MjModel, data: mujoco.MjData) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Mocap (pos, quat) for the owner lying near the east edge of the guest bed, head on the pillows (north)."""
    m = _mattress(model, data)
    if m is None:
        return None
    x, y, hx, hy, top = m
    feet = np.array([x + hx - EDGE_GAP, y - hy + FOOT_MARGIN, top + BODY_HALF_DEPTH])
    return feet, LYING_QUAT.copy()


def bedside_xy_yaw(model: mujoco.MjModel, data: mujoco.MjData) -> Optional[Tuple[float, float, float]]:
    """Where the owner stands after getting up: beside the bed's east edge, facing east (into the room)."""
    m = _mattress(model, data)
    if m is None:
        return None
    x, y, hx, hy, _top = m
    return x + hx + 0.45, y - 0.4 * hy, 0.0


def lie_in_bed(loco) -> bool:
    """Put the owner (mocap) in bed. Returns False if the bed isn't in the scene."""
    pose = lying_pose(loco.model, loco.data)
    mid = getattr(loco, "owner_mocap_id", -1)
    if pose is None or mid < 0:
        return False
    loco.data.mocap_pos[mid], loco.data.mocap_quat[mid] = pose
    print("[sim] owner lies down in the guest bed", flush=True)
    return True


def is_lying(loco) -> bool:
    mid = getattr(loco, "owner_mocap_id", -1)
    return mid >= 0 and abs(float(loco.data.mocap_quat[mid][0]) - LYING_QUAT[0]) < 1e-3 and \
        abs(float(loco.data.mocap_quat[mid][1]) - LYING_QUAT[1]) < 1e-3


def body_distance(loco, rx: float, ry: float) -> float:
    """Robot centre -> nearest point of the lying body (feet -> head line). Used for the wake-up nudge while
    lying: the robot can't reach the feet from the bed's side (it stops ~0.64 m from the body line), but a
    nudge against the bed side next to her arm counts."""
    mid = loco.owner_mocap_id
    fx, fy = float(loco.data.mocap_pos[mid][0]), float(loco.data.mocap_pos[mid][1])
    hx, hy = fx, fy + BODY_LENGTH  # head to +Y (LYING_QUAT)
    t = max(0.0, min(1.0, ((rx - fx) * (hx - fx) + (ry - fy) * (hy - fy)) / (BODY_LENGTH ** 2)))
    return math.hypot(rx - (fx + t * (hx - fx)), ry - (fy + t * (hy - fy))) - ARM_REACH
