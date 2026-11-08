#!/usr/bin/env python3
"""
Visible nudge (`sim/robot_nudge.py`) - TODO #4: the robot visibly touches the owner, the owner reacts.

Viewer / recording only. Navigation, the owner model and every sensor value the brain uses are
unchanged: the effects are applied AFTER `locomotion.step()` as a small pose offset on the robot's
rendered qpos (rewritten from the kinematic pose on the next step anyway) and as a tilt / hop of the
owner's mocap body (z + pitch only; x, y and yaw - what owner_behavior and locomotion read - stay).

  * Bump: while an alert vibration is on (1_pulse / 2_pulse / continuous) and the robot is within
    CONTACT_RANGE of the owner, the robot lunges toward the owner and dips its head, synced to the
    haptic code on the handle grip: 1 bump for 1_pulse, 2 for 2_pulse, repeated quickly for
    continuous. Each bump reaches the owner's legs (reach = distance - TOUCH_DIST).
  * Contact flash + ripple: a bright flash at the touch point and an expanding ring around the
    owner (alert colour) at the peak of every bump.
  * Owner reaction: when the owner FEELS the nudge (owner_behavior.felt_at) they startle - small hop
    and lean back - and a "!" appears above their head; owner_behavior already turns them toward
    the robot. While they tap the handle (acknowledge) their hand / the grip glows and an arm
    reaches to the grip.
  * Asleep owner: reclined by ASLEEP_RECLINE_DEG while asleep; when woken they straighten up
    (sit / stand up) over WAKE_RISE_S, then startle like above.

Works with the primitive owner and the scanned-mesh owner (both are geoms of the one mocap body).
`apply()` runs inside step_once (already under viewer.lock() in the viewer loop); `draw(scn)` appends
geoms to a viewer user scene or an offscreen Renderer scene and must be called after anything that
resets scn.ngeom (RobotScreen.draw_label).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from sim import owner_pose
from sim.locomotion import quat_to_yaw, yaw_to_quat

PULSE_MODES = ("1_pulse", "2_pulse", "continuous")
CONTACT_RANGE = 1.05      # robot centre to owner centre: bumps only this close (brain stands off at 0.75)
TOUCH_DIST = 0.46         # robot centre to owner centre when the nose touches the owner's legs
MAX_REACH = 0.32          # longest lunge (m)
MIN_REACH = 0.04
DIP_DEG = 9.0             # head dip at the peak of a bump
RIPPLE_S = 0.7
FLASH_S = 0.28
STARTLE_S = 0.6
BANG_S = 1.4              # "!" above the owner's head
ASLEEP_RECLINE_DEG = 16.0
WAKE_RISE_S = 1.3

# (period s, bump onsets within the period, bump length s): same timing as the grip glow
# (sim/guide_handle.py) so the bump, the vibration and the glow read as one signal.
PATTERNS = {
    "1_pulse": (1.0, (0.0,), 0.42),
    "2_pulse": (1.2, (0.0, 0.30), 0.26),
    "continuous": (0.5, (0.0,), 0.30),
}
COLOURS = {
    "1_pulse": (0.25, 1.0, 0.45),
    "2_pulse": (1.0, 0.85, 0.15),
    "continuous": (1.0, 0.35, 0.10),
}


def _pitch_quat(angle: float) -> np.ndarray:
    """Rotation about the local +y axis (positive = nose down for a +x-forward body)."""
    h = 0.5 * angle
    return np.array([math.cos(h), 0.0, math.sin(h), 0.0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return q


class NudgeFX:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.grip_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "handle_grip")
        self.ripples: List[Tuple[float, Tuple[float, float], Tuple[float, float, float]]] = []
        self.flashes: List[Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]] = []
        self._last_peak = -1.0
        self._last_felt: Optional[float] = None
        self._was_asleep = False
        self.wake_t: Optional[float] = None
        self.startle_t: Optional[float] = None
        self.owner_dirty = False
        self.t = 0.0
        self.tap = False
        self.owner_xy = (0.0, 0.0)
        self.owner_yaw = 0.0
        self.bumps = 0

    def reset(self) -> None:
        self.ripples.clear()
        self.flashes.clear()
        self._last_felt = None
        self.wake_t = self.startle_t = None
        self._was_asleep = False

    # ------------------------------------------------------------------ per-step pose effects
    def _bump_env(self, mode: str, t: float) -> Tuple[float, Optional[float]]:
        """(envelope 0..1 of the current bump, peak time of the current bump or None)."""
        period, onsets, length = PATTERNS[mode]
        ph = t % period
        for o in onsets:
            s = (ph - o) / length
            if 0.0 <= s <= 1.0:
                return math.sin(math.pi * s), t - ph + o + 0.5 * length
        return 0.0, None

    def apply(self, model, data, t: float, alert_mode: str, loco, owner) -> None:
        """Offset the rendered robot / owner pose for this step. Call right after locomotion.step()."""
        self.t = t
        rx, ry, ryaw = loco.get_pose()
        ox, oy = loco.get_owner_pos()
        self.owner_xy = (ox, oy)
        d = math.hypot(ox - rx, oy - ry)
        changed = False

        # ---- robot bump toward the owner, synced to the haptic code
        if alert_mode in PULSE_MODES and d < CONTACT_RANGE and not loco.guiding_active:
            env, peak = self._bump_env(alert_mode, t)
            if env > 0.0:
                ux, uy = (ox - rx) / max(d, 1e-6), (oy - ry) / max(d, 1e-6)
                reach = min(MAX_REACH, max(MIN_REACH, d - TOUCH_DIST)) * env
                data.qpos[0] += ux * reach
                data.qpos[1] += uy * reach
                data.qpos[3:7] = _qmul(data.qpos[3:7], _pitch_quat(math.radians(DIP_DEG) * env))
                changed = True
            if peak is not None and peak != self._last_peak and t >= peak:
                self._last_peak = peak
                self.bumps += 1
                col = COLOURS[alert_mode]
                self.ripples.append((t, (ox, oy), col))
                ux, uy = (rx - ox) / max(d, 1e-6), (ry - oy) / max(d, 1e-6)
                self.flashes.append((t, (ox + 0.20 * ux, oy + 0.20 * uy, 0.32), col))
        self.ripples = [r for r in self.ripples if t - r[0] < RIPPLE_S]
        self.flashes = [f for f in self.flashes if t - f[0] < FLASH_S]

        # ---- owner reaction
        mid = loco.owner_mocap_id
        if mid >= 0:
            asleep = getattr(owner, "owner_state", "awake") == "asleep"
            if self._was_asleep and not asleep:
                self.wake_t = t
            self._was_asleep = asleep
            felt = getattr(owner, "felt_at", None)
            if getattr(owner, "owner_state", "awake") == "unresponsive":
                # Feels nothing, reacts to nothing: only the robot's bump / contact flash show.
                felt, self.startle_t, self.wake_t = self._last_felt, None, None
            if felt is not None and felt != self._last_felt:
                self._last_felt = felt
                # A sleeper first straightens up, then startles.
                self.startle_t = felt + (WAKE_RISE_S if self.wake_t is not None and t - self.wake_t < 0.1 else 0.0)
            self.tap = bool(getattr(owner, "handle_tap", False))
            pitch, hop = 0.0, 0.0
            if asleep:
                pitch = -math.radians(ASLEEP_RECLINE_DEG)
            elif self.wake_t is not None and t - self.wake_t < WAKE_RISE_S:
                s = (t - self.wake_t) / WAKE_RISE_S
                pitch = -math.radians(ASLEEP_RECLINE_DEG) * (1.0 - (3 * s * s - 2 * s * s * s))
            if self.startle_t is not None and 0.0 <= t - self.startle_t < STARTLE_S:
                s = (t - self.startle_t) / STARTLE_S
                hop = 0.05 * math.sin(math.pi * s)
                pitch += -math.radians(8.0) * math.sin(math.pi * s)
            q = data.mocap_quat[mid]
            self.owner_yaw = quat_to_yaw(q)
            if owner_pose.is_lying(loco):  # #8 (Ricktho): lying in bed (NUDGE_OWNER_IN_BED=1) keeps its own pose
                pitch, hop = 0.0, 0.0
                self.owner_dirty = False
            if pitch != 0.0 or hop != 0.0:
                data.mocap_pos[mid][2] = hop
                data.mocap_quat[mid] = _qmul(yaw_to_quat(self.owner_yaw), _pitch_quat(pitch))
                self.owner_dirty = changed = True
            elif self.owner_dirty:
                data.mocap_pos[mid][2] = 0.0
                data.mocap_quat[mid] = yaw_to_quat(self.owner_yaw)
                self.owner_dirty, changed = False, True
        if changed:
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)

    # ------------------------------------------------------------------ drawing
    @staticmethod
    def _add(scn, gtype, size, pos, rgba, mat=None, label: str = "") -> bool:
        if scn.ngeom >= scn.maxgeom:
            return False
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64), np.asarray(pos, dtype=np.float64),
                            (np.eye(3) if mat is None else mat).flatten(), np.asarray(rgba, dtype=np.float32))
        g.emission = 0.8
        if label:
            g.label = label
        scn.ngeom += 1
        return True

    def draw(self, scn, data=None) -> None:
        """Append the nudge effects to a scene (viewer.user_scn under viewer.lock(), or Renderer.scene)."""
        if scn is None:
            return
        t = self.t
        ox, oy = self.owner_xy
        for t0, (cx, cy), col in self.ripples:
            s = (t - t0) / RIPPLE_S
            r = 0.30 + 0.75 * s
            a = 0.85 * (1.0 - s)
            for z in (0.30, 0.75):
                for k in range(18):
                    ang = 2 * math.pi * k / 18
                    self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.040, 0, 0],
                              [cx + r * math.cos(ang), cy + r * math.sin(ang), z], [*col, a])
        for t0, p, col in self.flashes:
            s = (t - t0) / FLASH_S
            self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.08 + 0.14 * s, 0, 0], p,
                      [1.0, 1.0, 0.85 * (1 - s) + 0.15 * col[2], 0.9 * (1.0 - s)])
        if self.startle_t is not None and 0.0 <= t - self.startle_t < BANG_S:
            self._add(scn, mujoco.mjtGeom.mjGEOM_LABEL, [0, 0, 0], [ox, oy, 2.05], [1, 1, 1, 1], label="! felt the nudge")
            self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.09, 0, 0], [ox, oy, 1.85], [1.0, 0.9, 0.2, 0.9])
        if self.tap and data is not None and self.grip_gid >= 0:
            gp = np.array(data.geom_xpos[self.grip_gid])
            self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.065, 0, 0], gp, [1.0, 0.95, 0.3, 0.85])
            # The owner's arm reaching to the grip (shoulder on the side facing the grip).
            c, s_ = math.cos(self.owner_yaw), math.sin(self.owner_yaw)
            best = None
            for side in (0.26, -0.26):
                sh = np.array([ox - side * s_, oy + side * c, 1.24])
                dist = float(np.linalg.norm(gp - sh))
                if best is None or dist < best[0]:
                    best = (dist, sh)
            if best is not None and best[0] < 1.1 and scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                                    np.array([1.0, 0.85, 0.35, 0.9], dtype=np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.04, best[1], gp)
                g.emission = 0.6
                scn.ngeom += 1
