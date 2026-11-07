#!/usr/bin/env python3
"""
Milestone 3: Locomotion hardware abstraction layer (`sim/locomotion.py`).
Accepts body-frame velocity commands `cmd_vel = (vx, vy, wz)` plus `guiding` and
`alert_mode` states, and steps the Unitree Go2 robot in MuJoCo.

Supports:
  - Plan B (Default): Kinematic base integration on `qpos[0:7]` with house (static nav-map
    clearance field: walls + furniture), chair and owner collision clamping, upright attitude stabilization, and speed-scaled sinusoidal
    trotting leg animation on `qpos[7:19]` (kinematic: no ctrl, no mj_step).
  - Owner guiding physics: the owner walks (<= 0.4 m/s) to the guide handle when `guiding=True`
    and within 1.55 m, then holds it rigidly on the robot's right (OWNER_OFFSET_BODY).
  - Pop-up guide handle (`sim/guide_handle.py`) whose grip glows in the haptic code, and the
    police-style light bar (`sim/lightbar.py`) on the Go2 back.
"""

from __future__ import annotations

import math
from typing import Tuple

import mujoco
import numpy as np

from sim.house_layout import (
    CHAIR_COLLISION_RADIUS,
    OWNER_SPOTS,
    OWNER_START_ROOM,
    ROBOT_START_POSE,
    get_room_at,
)
from sim.guide_handle import GRIP_BASE, GuideHandle
from sim.lightbar import LightBar
from sim.nav_map import NavMap


def yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert yaw angle (rad) about +Z to MuJoCo quaternion [w, x, y, z]."""
    half = 0.5 * yaw
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw angle (rad) about +Z from MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# Owner centre in the robot frame while holding the handle: left hand on the grip, hand 0.12 m ahead
# of and 0.22 m left of the owner's centre (owner faces the robot's heading).
OWNER_OFFSET_BODY = (float(GRIP_BASE[0]) - 0.12, float(GRIP_BASE[1]) - 0.22)


class Go2Locomotion:
    """Velocity controller (`vx, vy, wz`) and guiding physics for Unitree Go2."""

    ROBOT_RADIUS: float = 0.25
    STANDING_Z: float = 0.27   # nerv Go2 'home' keyframe base height
    HOME_JOINTS: np.ndarray = np.array(
        [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        dtype=np.float64,
    )

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, nav_map: NavMap) -> None:
        self.model = model
        self.data = data
        self.nav_map = nav_map

        # Look up mocap IDs for Owner and Chair
        owner_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_body")
        chair_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chair_body")
        self.owner_mocap_id: int = int(model.body_mocapid[owner_body_id]) if owner_body_id >= 0 else -1
        self.chair_mocap_id: int = int(model.body_mocapid[chair_body_id]) if chair_body_id >= 0 else -1

        self.handle = GuideHandle(model)
        self.lightbar = LightBar(model)

        # Internal state
        self.pose_x: float = ROBOT_START_POSE[0]
        self.pose_y: float = ROBOT_START_POSE[1]
        self.pose_yaw: float = ROBOT_START_POSE[2]
        self.gait_phase: float = 0.0
        self.guiding_active: bool = False
        self.owner_locked: bool = False

        self.reset_robot_pose(*ROBOT_START_POSE)

    def reset_robot_pose(self, x: float, y: float, yaw: float) -> None:
        """Place Go2 upright at (x, y, yaw) in its standing home configuration."""
        self.pose_x = float(x)
        self.pose_y = float(y)
        self.pose_yaw = float(yaw)

        self.data.qpos[0] = self.pose_x
        self.data.qpos[1] = self.pose_y
        self.data.qpos[2] = self.STANDING_Z
        self.data.qpos[3:7] = yaw_to_quat(self.pose_yaw)
        self.data.qpos[7:19] = self.HOME_JOINTS
        self.data.qvel[:] = 0.0
        # No ctrl writes: Plan B never calls mj_step, and the nerv Go2 actuators are torque motors.
        mujoco.mj_forward(self.model, self.data)

    def get_pose(self) -> Tuple[float, float, float]:
        """Return current (x, y, yaw) of the Go2 base."""
        return (self.pose_x, self.pose_y, self.pose_yaw)

    def get_owner_pos(self) -> Tuple[float, float]:
        """Return current (x, y) of the Owner mocap body."""
        if self.owner_mocap_id >= 0:
            pos = self.data.mocap_pos[self.owner_mocap_id]
            return (float(pos[0]), float(pos[1]))
        return OWNER_SPOTS[OWNER_START_ROOM]

    def set_owner_pos(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Teleport the Owner mocap body to (x, y)."""
        if self.owner_mocap_id >= 0:
            self.data.mocap_pos[self.owner_mocap_id] = [float(x), float(y), 0.0]
            self.data.mocap_quat[self.owner_mocap_id] = yaw_to_quat(yaw)
            mujoco.mj_forward(self.model, self.data)

    def set_chair_pos(self, x: float, y: float, z: float = 0.0) -> None:
        """Move the Chair obstacle mocap body to (x, y, z)."""
        if self.chair_mocap_id >= 0:
            self.data.mocap_pos[self.chair_mocap_id] = [float(x), float(y), float(z)]
            mujoco.mj_forward(self.model, self.data)

    def _clamp_against_obstacles(
        self, next_x: float, next_y: float, allow_owner_close: bool = False
    ) -> Tuple[float, float]:
        """
        Resolve house clearance (static nav map) and circle-vs-circle (chair & owner)
        collisions so the robot never penetrates walls or furniture.
        """
        r = self.ROBOT_RADIUS
        px, py = self.pose_x, self.pose_y

        # 1. House (walls + furniture): never end closer than r to a static obstacle,
        #    unless already closer and moving away. Slide along obstacles axis by axis.
        x, y = self._slide(px, py, next_x, next_y)

        # 2. Chair circle push-out
        if self.chair_mocap_id >= 0:
            chair_pos = self.data.mocap_pos[self.chair_mocap_id]
            dx = x - float(chair_pos[0])
            dy = y - float(chair_pos[1])
            min_d = r + CHAIR_COLLISION_RADIUS
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_d * min_d and dist_sq > 1e-8:
                dist = math.sqrt(dist_sq)
                push = min_d - dist
                x += (dx / dist) * push
                y += (dy / dist) * push

        # 3. Owner circle push-out (allow soft nudge contact up to 0.42m center-to-center)
        if self.owner_mocap_id >= 0:
            ox, oy = self.get_owner_pos()
            dx = x - ox
            dy = y - oy
            min_owner_d = 0.40 if allow_owner_close else 0.44
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_owner_d * min_owner_d and dist_sq > 1e-8:
                dist = math.sqrt(dist_sq)
                push = min_owner_d - dist
                x += (dx / dist) * push
                y += (dy / dist) * push

        # Chair / owner push-outs must not shove the robot into a wall.
        if not self._ok(px, py, x, y):
            x, y = self._slide(px, py, x, y)

        # Keep within the mapped area (house + outside margin)
        bx0, by0, bx1, by1 = self.nav_map.bounds
        x = max(bx0 + r, min(bx1 - r, x))
        y = max(by0 + r, min(by1 - r, y))
        return (x, y)

    def _ok(self, px: float, py: float, x: float, y: float) -> bool:
        c = self.nav_map.clearance(x, y)
        return c >= self.ROBOT_RADIUS or c >= self.nav_map.clearance(px, py) + 1e-4

    def _slide(self, px: float, py: float, x: float, y: float) -> Tuple[float, float]:
        if self._ok(px, py, x, y):
            return x, y
        if self._ok(px, py, x, py):
            return x, py
        if self._ok(px, py, px, y):
            return px, y
        return px, py

    def step(
        self,
        vx: float,
        vy: float,
        wz: float,
        guiding: bool = False,
        alert_mode: str = "none",
        dt: float = 0.002,
    ) -> None:
        """
        Advance robot kinematics, trotting animation, owner guiding physics,
        and visual alert indicators by one physics step `dt`.
        """
        # Clamp velocities to physical limits (and proximity speed cap <= 0.4 m/s near owner)
        ox, oy = self.get_owner_pos()
        dist_to_owner = math.hypot(self.pose_x - ox, self.pose_y - oy)

        max_lin = 0.40 if (dist_to_owner <= 1.0 or guiding) else 1.35
        vx = float(np.clip(vx, -max_lin, max_lin))
        vy = float(np.clip(vy, -max_lin, max_lin))
        wz = float(np.clip(wz, -2.4, 2.4))

        speed = math.hypot(vx, vy)
        if speed > max_lin and speed > 1e-6:
            scale = max_lin / speed
            vx *= scale
            vy *= scale
            speed = max_lin

        # Integrate yaw and body-frame (vx, vy) into world frame
        self.pose_yaw = math.atan2(
            math.sin(self.pose_yaw + wz * dt),
            math.cos(self.pose_yaw + wz * dt),
        )
        cos_y = math.cos(self.pose_yaw)
        sin_y = math.sin(self.pose_yaw)
        world_vx = vx * cos_y - vy * sin_y
        world_vy = vx * sin_y + vy * cos_y

        next_x = self.pose_x + world_vx * dt
        next_y = self.pose_y + world_vy * dt
        self.pose_x, self.pose_y = self._clamp_against_obstacles(
            next_x, next_y, allow_owner_close=True
        )

        # Diagonal trotting gait animation proportional to linear + angular speed
        effective_speed = speed + 0.25 * abs(wz)
        joints = self.HOME_JOINTS.copy()
        bob_z = 0.0
        if effective_speed > 0.03:
            self.gait_phase += dt * 11.5 * min(1.3, effective_speed / 0.55)
            swing = 0.24 * min(1.0, effective_speed / 0.65) * math.sin(self.gait_phase)
            knee_lift = 0.18 * min(1.0, effective_speed / 0.65)
            # FL (idx 0,1,2) & RR (idx 9,10,11) in phase; FR (idx 3,4,5) & RL (idx 6,7,8) anti-phase
            joints[1] += swing
            joints[2] += knee_lift * max(0.0, math.sin(self.gait_phase))
            joints[10] += swing
            joints[11] += knee_lift * max(0.0, math.sin(self.gait_phase))

            joints[4] -= swing
            joints[5] += knee_lift * max(0.0, -math.sin(self.gait_phase))
            joints[7] -= swing
            joints[8] += knee_lift * max(0.0, -math.sin(self.gait_phase))
            bob_z = 0.008 * abs(math.sin(self.gait_phase))

        # Write kinematic pose & joint angles directly to MjData
        self.data.qpos[0] = self.pose_x
        self.data.qpos[1] = self.pose_y
        self.data.qpos[2] = self.STANDING_Z + bob_z
        self.data.qpos[3:7] = yaw_to_quat(self.pose_yaw)
        self.data.qpos[7:19] = joints
        self.data.qvel[:] = 0.0

        # Owner guiding physics (rigid guide-dog handle). Like a real guide dog, the robot walks on the
        # owner's LEFT, so the owner is on the robot's RIGHT, left hand on the grip. While guiding and
        # within 1.55 m, the owner first walks (<= 0.4 m/s) to the handle; once the handle is fully out
        # and they reach it they are LOCKED: the left hand stays on the grip and they face the robot's
        # heading, so every stop or turn of the robot moves them (that is what the rigid handle conveys).
        self.guiding_active = False
        if guiding and self.owner_mocap_id >= 0 and dist_to_owner <= 1.55:
            self.guiding_active = True
            bx, by = OWNER_OFFSET_BODY  # owner centre in the robot frame (behind-right)
            target_ox = self.pose_x + bx * cos_y - by * sin_y
            target_oy = self.pose_y + bx * sin_y + by * cos_y
            dx = target_ox - ox
            dy = target_oy - oy
            d = math.hypot(dx, dy)
            if self.owner_locked or (d <= 0.03 and self.handle.deployed):
                self.owner_locked = True
                self.data.mocap_pos[self.owner_mocap_id] = [target_ox, target_oy, 0.0]
                self.data.mocap_quat[self.owner_mocap_id] = yaw_to_quat(self.pose_yaw)
            elif d > 0.005:
                step_d = min(d, 0.40 * dt)
                new_ox = ox + (dx / d) * step_d
                new_oy = oy + (dy / d) * step_d
                walk_yaw = math.atan2(dy, dx) if d > 0.15 else self.pose_yaw
                self.data.mocap_pos[self.owner_mocap_id] = [new_ox, new_oy, 0.0]
                self.data.mocap_quat[self.owner_mocap_id] = yaw_to_quat(walk_yaw)
        else:
            self.owner_locked = False

        # Pop-up guide handle: out while offering it / guiding, grip glows in the haptic code
        self.handle.update(self.model, dt, float(self.data.time),
                           extended=guiding or alert_mode in ("1_pulse", "2_pulse", "continuous"),
                           alert_mode=alert_mode)
        self.lightbar.update(self.model, float(self.data.time), alert_mode)

        self.data.time += dt
        mujoco.mj_forward(self.model, self.data)
