#!/usr/bin/env python3
"""
Milestone 3: Locomotion hardware abstraction layer (`sim/locomotion.py`).
Accepts body-frame velocity commands `cmd_vel = (vx, vy, wz)` plus `guiding` and
`alert_mode` states, and steps the Unitree Go2 robot in MuJoCo.

Supports:
  - Plan B (Default): Kinematic base integration on `qpos[0:7]` with wall/chair/owner
    collision clamping, upright attitude stabilization, and speed-scaled sinusoidal
    trotting leg animation on `qpos[7:19]` + `ctrl[0:12]`.
  - Owner guiding physics: Smoothly moves the Owner mocap body behind the robot's
    guide handle at <= 0.4 m/s when `guiding=True` and within 1.4 m.
  - Visual haptic handle pulse indicator (`alert_halo`) on the Go2 back.
"""

from __future__ import annotations

import math
from typing import Tuple

import mujoco
import numpy as np

from sim.legacy_box_house import (
    CHAIR_COLLISION_RADIUS,
    ROBOT_START_POSE,
    WALL_BOXES,
    get_room_at,
)


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


class Go2Locomotion:
    """Velocity controller (`vx, vy, wz`) and guiding physics for Unitree Go2."""

    ROBOT_RADIUS: float = 0.25
    STANDING_Z: float = 0.285
    HOME_JOINTS: np.ndarray = np.array(
        [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
        dtype=np.float64,
    )

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        # Look up mocap IDs for Owner and Chair
        owner_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "owner_body")
        chair_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chair_body")
        self.owner_mocap_id: int = int(model.body_mocapid[owner_body_id]) if owner_body_id >= 0 else -1
        self.chair_mocap_id: int = int(model.body_mocapid[chair_body_id]) if chair_body_id >= 0 else -1

        self.alert_halo_geom_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "alert_halo"
        )

        # Internal state
        self.pose_x: float = ROBOT_START_POSE[0]
        self.pose_y: float = ROBOT_START_POSE[1]
        self.pose_yaw: float = ROBOT_START_POSE[2]
        self.gait_phase: float = 0.0
        self.guiding_active: bool = False

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
        if self.model.nu >= 12:
            self.data.ctrl[:12] = self.HOME_JOINTS
        mujoco.mj_forward(self.model, self.data)

    def get_pose(self) -> Tuple[float, float, float]:
        """Return current (x, y, yaw) of the Go2 base."""
        return (self.pose_x, self.pose_y, self.pose_yaw)

    def get_owner_pos(self) -> Tuple[float, float]:
        """Return current (x, y) of the Owner mocap body."""
        if self.owner_mocap_id >= 0:
            pos = self.data.mocap_pos[self.owner_mocap_id]
            return (float(pos[0]), float(pos[1]))
        return (-3.2, -2.2)

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
        Resolve 2D circle-vs-AABB (walls) and circle-vs-circle (chair & owner)
        collisions so the robot never penetrates walls or furniture.
        """
        r = self.ROBOT_RADIUS
        x, y = next_x, next_y

        # 1. Wall AABB push-out
        for _name, cx, cy, _cz, hx, hy, _hz in WALL_BOXES:
            # Closest point on AABB to (x, y)
            closest_x = max(cx - hx, min(x, cx + hx))
            closest_y = max(cy - hy, min(y, cy + hy))
            dx = x - closest_x
            dy = y - closest_y
            dist_sq = dx * dx + dy * dy
            if dist_sq < r * r:
                dist = math.sqrt(dist_sq)
                if dist > 1e-6:
                    push = r - dist
                    x += (dx / dist) * push
                    y += (dy / dist) * push
                else:
                    # Inside box center fallback
                    if abs(x - cx) / hx > abs(y - cy) / hy:
                        x = (cx + hx + r) if x >= cx else (cx - hx - r)
                    else:
                        y = (cy + hy + r) if y >= cy else (cy - hy - r)

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

        # Keep within outer courtyard bounds (including back_exit_outside up to y=4.85)
        x = max(-4.72, min(4.72, x))
        y = max(-3.72, min(4.85, y))
        return (x, y)

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
        if self.model.nu >= 12:
            self.data.ctrl[:12] = joints

        # Owner guiding physics: if guiding=True and robot is within 1.45 m of Owner,
        # Owner holds the handle offset 0.52m behind-left of the robot and walks at <= 0.4 m/s
        self.guiding_active = False
        if guiding and self.owner_mocap_id >= 0 and dist_to_owner <= 1.55:
            self.guiding_active = True
            target_ox = self.pose_x - 0.50 * cos_y - 0.14 * sin_y
            target_oy = self.pose_y - 0.50 * sin_y + 0.14 * cos_y
            dx = target_ox - ox
            dy = target_oy - oy
            d = math.hypot(dx, dy)
            max_step = 0.40 * dt
            if d > 0.02:
                step_d = min(d, max_step)
                new_ox = ox + (dx / d) * step_d
                new_oy = oy + (dy / d) * step_d
                owner_yaw = math.atan2(self.pose_y - new_oy, self.pose_x - new_ox)
                self.data.mocap_pos[self.owner_mocap_id] = [new_ox, new_oy, 0.0]
                self.data.mocap_quat[self.owner_mocap_id] = yaw_to_quat(owner_yaw)

        # Update visual haptic pulse halo on the Go2 handle
        if self.alert_halo_geom_id >= 0:
            t = float(self.data.time)
            if alert_mode == "1_pulse":
                # Single cyan pulse every 1.0s
                pulse = 1.0 if (t % 1.0) < 0.28 else 0.15
                self.model.geom_rgba[self.alert_halo_geom_id] = [0.1, 0.9, 1.0, 0.88 * pulse]
            elif alert_mode == "2_pulse":
                # Double amber pulse every 1.2s
                phase = t % 1.2
                pulse = 1.0 if (0.0 <= phase < 0.18 or 0.30 <= phase < 0.48) else 0.15
                self.model.geom_rgba[self.alert_halo_geom_id] = [1.0, 0.78, 0.1, 0.90 * pulse]
            elif alert_mode == "continuous":
                # High-frequency red emergency strobe
                pulse = 0.55 + 0.45 * math.sin(t * 28.0)
                self.model.geom_rgba[self.alert_halo_geom_id] = [1.0, 0.12, 0.12, float(pulse)]
            elif alert_mode == "complete":
                # Steady green: mission finished, owner delivered (sim-side visual only)
                self.model.geom_rgba[self.alert_halo_geom_id] = [0.2, 0.95, 0.35, 0.85]
            else:
                self.model.geom_rgba[self.alert_halo_geom_id] = [0.2, 0.9, 0.4, 0.0]

        self.data.time += dt
        mujoco.mj_forward(self.model, self.data)
