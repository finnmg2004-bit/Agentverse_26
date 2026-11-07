#!/usr/bin/env python3
"""
Guide-dog style handle on the Go2 (`sim/guide_handle.py`) — the rigid "hard leash" a blind or
deafblind owner holds. Built from primitives (no third-party asset).

Like a real guide-dog harness handle, plus a telescope so it fits the owner and stows away:
  * Mounted on a harness saddle at the withers (front of the back), on a hinge across the spine.
  * STOWED: the U-shaped frame folds flat along the spine, pointing backwards; nothing sticks up.
  * OFFERED / GUIDING: it swings up on the hinge (~43 deg) toward the owner, who walks on the dog's
    RIGHT (real guide dogs walk on the handler's left), then telescopes out so the padded grip ends
    at hand height (~0.75 m above the floor). Retracting runs the same in reverse.
  * The grip glows softly in the haptic code (1 pulse / 2 pulses / continuous) so the vibration the
    owner feels is visible on camera.
  * `GRIP_BASE` (grip position in the base frame when fully out) is what `locomotion.py` uses to
    hold the owner's left hand rigidly on the grip while guiding.

Visual only: robot-visual group 2, no mass, no contacts (LiDAR via HOUSE_RAY_GROUPS and the nav map
ignore it). Animated at runtime through `body_quat`, `geom_pos` and `geom_size`.
Usage: `add_guide_handle(spec)` before compiling; `GuideHandle(model).update(model, dt, t, extended, alert_mode)`.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

HINGE_POS = np.array([0.05, 0.0, 0.071])        # base frame: withers, just above the back (top z 0.057)
FOLDED_LEN = 0.22                                # frame length lying flat along the spine
GRIP_BASE = np.array([-0.38, -0.12, 0.48])       # base frame: owner's hand (dog's right, ~0.75 m above floor)
_v = GRIP_BASE - HINGE_POS
FULL_LEN = float(np.linalg.norm(_v))
RAISE_PITCH = math.atan2(_v[2], math.hypot(_v[0], _v[1]))   # ~43 deg up
RAISE_YAW = math.atan2(-_v[1], -_v[0])                       # ~15 deg toward the right side
OUTER_LEN = 0.20                                 # fixed outer tubes; inner tubes slide out of them
HALF_W = 0.045                                   # half-width of the U frame
SWING_TIME, TELESCOPE_TIME = 0.45, 0.40
GRIP_RGBA = (0.07, 0.07, 0.08, 1.0)
GLOW_RGBA = (0.80, 0.95, 1.0, 1.0)
_Z2X = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]  # capsule axis z -> x
_Z2Y = [math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]  # capsule axis z -> y


def _frame_quat(swing: float) -> np.ndarray:
    """Handle frame: local +x along the bars (from hinge to grip), local +z = frame 'up'."""
    pitch, yaw = swing * RAISE_PITCH, swing * RAISE_YAW
    x = np.array([-math.cos(pitch) * math.cos(yaw), -math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    y = np.array([math.sin(yaw), -math.cos(yaw), 0.0])  # horizontal, perpendicular to the bars
    y = y - x * float(x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if z[2] < 0:
        y, z = -y, -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).reshape(-1))
    return q


def add_guide_handle(spec: mujoco.MjSpec, parent: str = "base") -> None:
    G = mujoco.mjtGeom
    vis = dict(contype=0, conaffinity=0, mass=0, group=2)
    spec.add_material(name="handle_frame_mat", rgba=[0.10, 0.10, 0.11, 1.0], specular=0.5, shininess=0.6)
    spec.add_material(name="handle_inner_mat", rgba=[0.66, 0.68, 0.72, 1.0], specular=0.9, shininess=0.9, reflectance=0.1)
    spec.add_material(name="handle_grip_mat", rgba=list(GRIP_RGBA), specular=0.15, shininess=0.15, emission=0.0)
    spec.add_material(name="harness_mat", rgba=[0.13, 0.13, 0.14, 1.0], specular=0.2, shininess=0.2)

    base = spec.body(parent)
    # Harness saddle + hinge brackets at the withers.
    base.add_geom(name="harness_saddle", type=G.mjGEOM_BOX, pos=[HINGE_POS[0], 0, 0.060], size=[0.045, 0.085, 0.004],
                  material="harness_mat", **vis)
    for side, y in (("l", HALF_W + 0.014), ("r", -HALF_W - 0.014)):
        base.add_geom(name=f"harness_bracket_{side}", type=G.mjGEOM_BOX, pos=[HINGE_POS[0], y, 0.066],
                      size=[0.014, 0.005, 0.011], material="harness_mat", **vis)

    h = base.add_body(name="guide_handle", pos=HINGE_POS.tolist(), quat=_frame_quat(0.0).tolist())
    h.add_geom(name="handle_hinge", type=G.mjGEOM_CAPSULE, pos=[0, 0, 0], size=[0.011, HALF_W + 0.006, 0],
               quat=_Z2Y, material="handle_frame_mat", **vis)
    for side, y in (("left", HALF_W), ("right", -HALF_W)):
        h.add_geom(name=f"handle_outer_{side}", type=G.mjGEOM_CAPSULE, pos=[OUTER_LEN / 2, y, 0],
                   size=[0.0095, OUTER_LEN / 2, 0], quat=_Z2X, material="handle_frame_mat", **vis)
        h.add_geom(name=f"handle_inner_{side}", type=G.mjGEOM_CAPSULE, pos=[OUTER_LEN, y, 0],
                   size=[0.0065, 0.001, 0], quat=_Z2X, material="handle_inner_mat", **vis)
        h.add_geom(name=f"handle_corner_{side}", type=G.mjGEOM_SPHERE, pos=[FOLDED_LEN, y, 0], size=[0.0105, 0, 0],
                   material="handle_frame_mat", **vis)
    # Padded crossbar grip closing the U.
    h.add_geom(name="handle_grip", type=G.mjGEOM_CAPSULE, pos=[FOLDED_LEN, 0, 0], size=[0.017, HALF_W, 0],
               quat=_Z2Y, material="handle_grip_mat", **vis)


class GuideHandle:
    def __init__(self, model: mujoco.MjModel) -> None:
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "guide_handle")
        self.inner = [gid("handle_inner_left"), gid("handle_inner_right")]
        self.corners = [gid("handle_corner_left"), gid("handle_corner_right")]
        self.grip = gid("handle_grip")
        self.grip_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "handle_grip_mat")
        ids = [self.body, self.grip, self.grip_mat] + self.inner + self.corners
        self.enabled = all(i >= 0 for i in ids)
        self.swing = 0.0      # 0 folded flat .. 1 raised
        self.extend = 0.0     # 0 short .. 1 telescoped to hand height

    @property
    def deployed(self) -> bool:
        """Fully raised and extended: the owner can hold it rigidly."""
        return self.swing >= 0.999 and self.extend >= 0.999

    def update(self, model: mujoco.MjModel, dt: float, t: float, extended: bool, alert_mode: str) -> None:
        if not self.enabled:
            return
        if extended:  # swing up first, then telescope out
            if self.swing < 1.0:
                self.swing = min(1.0, self.swing + dt / SWING_TIME)
            else:
                self.extend = min(1.0, self.extend + dt / TELESCOPE_TIME)
        else:         # telescope in first, then fold down
            if self.extend > 0.0:
                self.extend = max(0.0, self.extend - dt / TELESCOPE_TIME)
            else:
                self.swing = max(0.0, self.swing - dt / SWING_TIME)

        s = 0.5 - 0.5 * math.cos(math.pi * self.swing)  # ease in/out
        model.body_quat[self.body] = _frame_quat(s)
        length = FOLDED_LEN + (FULL_LEN - FOLDED_LEN) * (0.5 - 0.5 * math.cos(math.pi * self.extend))
        inner_half = max((length - OUTER_LEN) / 2.0, 0.001)
        for g in self.inner:
            model.geom_pos[g][0] = OUTER_LEN + inner_half - 0.01
            model.geom_size[g][1] = inner_half + 0.01
        for g in self.corners:
            model.geom_pos[g][0] = length
        model.geom_pos[self.grip][0] = length

        # Haptic code made visible: soft glow pulses on the grip.
        if alert_mode == "1_pulse":
            on = (t % 1.0) < 0.28
        elif alert_mode == "2_pulse":
            ph = t % 1.2
            on = ph < 0.18 or 0.30 <= ph < 0.48
        elif alert_mode == "continuous":
            on = (t % 0.25) < 0.15
        else:
            on = False
        model.mat_rgba[self.grip_mat] = GLOW_RGBA if on else GRIP_RGBA
        model.mat_emission[self.grip_mat] = 0.9 if on else 0.0
