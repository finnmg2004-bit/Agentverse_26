"""
Monte-Carlo localisation (particle filter) for Nudge: the brain works out WHERE THE ROBOT IS from its own
odometry + LiDAR against the saved floor-plan map, instead of reading `state.robot_pose` from the sim.

Off by default. `NUDGE_LOCALIZE=1` -> main.Brain replaces `world.robot_pose` (and `robot_room`) with this
estimate every state message, so navigation, vision projection, hearing and the GUI all run on it.
The sim's ground-truth pose is only used to MEASURE the error (`err_xy`, `err_yaw` in the status).

  predict : odometry. `NUDGE_LOCALIZE_ODOM=cmd` = the brain's own last velocity command
            (vx, vy, wz) integrated over the state interval, clipped like the robot's speed limits, with
            per-particle noise. A particle that would end up inside a wall stays put (the real robot is
            stopped by the wall too). `NUDGE_LOCALIZE_ODOM=delta` = leg odometry: the pose change between
            two state messages, re-expressed in the body frame, read by a
            deliberately bad odometry sensor (+4 % scale, 0.9 deg/m yaw drift, slip noise), then per-particle noise.
            DEFAULT: legged robots measure their real motion (stuck behind the owner = no motion); pure commands
            can't tell, and drift metres when the robot is held up in a long room (test_localize --odom cmd).
  correct : the 36-ray LiDAR (`state.lidar_ranges`, [-90, +90] deg, 4 m max) scored on a likelihood field =
            distance transform of the static nav map (sim/nav_map.py, same 0.10-0.55 m band the LiDAR sees).
            Mixture with a uniform term, so the owner / chair (not in the map) don't kill good particles.
  resample: low-variance resampling when the effective sample size drops below N/2.
  init    : Gaussian around the known start pose (house_layout.ROBOT_START_POSE). `global_init()` (experimental,
            NUDGE_LOCALIZE_GLOBAL=1) spreads 6000 particles over the whole floor (kidnapped robot): it does NOT
            converge reliably yet in this house (many look-alike rooms; scripts/test_localize.py --kidnap fails).

Env: NUDGE_LOCALIZE=1, NUDGE_LOCALIZE_N (400), NUDGE_LOCALIZE_ODOM (cmd|delta), NUDGE_LOCALIZE_GLOBAL=1.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.house_layout import HOUSE_BOUNDS, ROBOT_START_POSE, get_room_at  # noqa: E402

LIDAR_ANGLES = np.array([-math.pi / 2 + i * math.pi / 35 for i in range(36)])  # = navigator.LIDAR_ANGLES
LIDAR_MAX = 4.0
ROBOT_RADIUS = 0.22          # a particle closer than this to a wall is "inside" it (sim clamps at 0.25)
SIGMA_HIT = 0.12             # m, LiDAR endpoint vs map
Z_HIT, Z_RAND = 0.85, 0.15   # mixture weights (z_rand absorbs owner / chair / unmapped clutter)
RAY_TEMPER = 0.25            # 36 rays are not independent: temper the joint log-likelihood
MAX_SPEED, GUIDE_SPEED, MAX_WZ = 1.35, 0.40, 2.4   # = sim/locomotion.py limits
ODOM_SCALE_BIAS = 1.04      # simulated leg odometry over-reads distance and turn by 4 %...
ODOM_YAW_DRIFT = 0.015      # ...and drifts 0.015 rad (0.9 deg) per metre walked


def enabled() -> bool:
    return os.environ.get("NUDGE_LOCALIZE", "0") == "1"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ParticleLocalizer:
    def __init__(self, n: Optional[int] = None, start_pose: Sequence[float] = ROBOT_START_POSE,
                 odom: Optional[str] = None, seed: int = 0, nav=None) -> None:
        if nav is None:
            from sim.nav_map import load_or_build
            nav = load_or_build(verbose=False)
        self.nav = nav
        self.n = int(n or os.environ.get("NUDGE_LOCALIZE_N", 400))
        self.odom = (odom or os.environ.get("NUDGE_LOCALIZE_ODOM", "delta")).lower()
        self.rng = np.random.default_rng(seed)
        self.dist = nav.dist                        # metres to the nearest mapped obstacle, 5 cm cells
        self.res, self.x0, self.y0 = nav.res, nav.x0, nav.y0
        self.H, self.W = self.dist.shape
        self.cmd = (0.0, 0.0, 0.0)
        self.guiding = False
        self.reset(start_pose)
        if os.environ.get("NUDGE_LOCALIZE_GLOBAL") == "1":
            self.global_init()

    # ------------------------------------------------------------------ init
    def reset(self, pose: Sequence[float] = ROBOT_START_POSE, sxy: float = 0.15, syaw: float = 0.08) -> None:
        x, y, yaw = (float(v) for v in pose)
        self.p = np.column_stack([self.rng.normal(x, sxy, self.n), self.rng.normal(y, sxy, self.n),
                                  _wrap(self.rng.normal(yaw, syaw, self.n))])
        self.w = np.full(self.n, 1.0 / self.n)
        self.t_last: Optional[float] = None
        self.last_gt: Optional[Tuple[float, float, float]] = None
        self.n_eff = float(self.n)
        self.fit: Optional[float] = None
        self.updates = 0
        self.errs: list = []
        self.err_xy: Optional[float] = None
        self.err_yaw: Optional[float] = None
        self.dr_err_xy: Optional[float] = None
        self.estimate = (x, y, yaw)
        self.dr = (x, y, yaw)              # dead reckoning (odometry only), for comparison

    def global_init(self) -> None:
        """Kidnapped robot: particles uniformly on free floor inside the map (heading uniform)."""
        b = HOUSE_BOUNDS
        free = np.argwhere(self.dist > 0.35)
        fx, fy = self.x0 + (free[:, 1] + 0.5) * self.res, self.y0 + (free[:, 0] + 0.5) * self.res
        inside = (fx > b["x_min"]) & (fx < b["x_max"]) & (fy > b["y_min"]) & (fy < b["y_max"])
        fx, fy = fx[inside], fy[inside]
        m = max(15 * self.n, 6000)        # many particles until it has converged, then back to n
        pick = self.rng.integers(0, len(fx), m)
        self.p = np.column_stack([fx[pick] + self.rng.uniform(-0.025, 0.025, m),
                                  fy[pick] + self.rng.uniform(-0.025, 0.025, m), self.rng.uniform(-np.pi, np.pi, m)])
        self.w = np.full(m, 1.0 / m)

    # ------------------------------------------------------------------ helpers
    def _dist_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = ((y - self.y0) / self.res).astype(np.int64)
        c = ((x - self.x0) / self.res).astype(np.int64)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        out = np.full(x.shape, 2.0, np.float32)   # outside the map: "far from everything"
        out[ok] = self.dist[r[ok], c[ok]]
        return out

    def set_cmd(self, vx: float, vy: float, wz: float, guiding: bool = False) -> None:
        """The velocity command the brain just sent (odometry for the next predict)."""
        self.cmd = (float(vx), float(vy), float(wz))
        self.guiding = bool(guiding)

    def _dead_reckon(self, dx: float, dy: float, dth: float) -> None:
        """Odometry alone, no LiDAR (for comparison only: how far the robot would drift without the filter)."""
        x, y, yaw = self.dr
        m = yaw + 0.5 * dth
        self.dr = (x + math.cos(m) * dx - math.sin(m) * dy, y + math.sin(m) * dx + math.cos(m) * dy,
                   math.atan2(math.sin(yaw + dth), math.cos(yaw + dth)))

    # ------------------------------------------------------------------ filter
    def _predict(self, dt: float, gt: Optional[Tuple[float, float, float]]) -> None:
        n = len(self.p)
        if self.odom == "delta" and gt is not None and self.last_gt is not None:
            # leg odometry: measured body-frame motion, 5 % scale error + noise
            dxw, dyw = gt[0] - self.last_gt[0], gt[1] - self.last_gt[1]
            c, s = math.cos(self.last_gt[2]), math.sin(self.last_gt[2])
            dx, dy = c * dxw + s * dyw, -s * dxw + c * dyw
            dth = math.atan2(math.sin(gt[2] - self.last_gt[2]), math.cos(gt[2] - self.last_gt[2]))
            dist = math.hypot(dx, dy)
            # the odometry SENSOR is wrong, systematically and randomly (foot slip): this reading is all the
            # filter gets; the truth above never reaches the particles
            dx, dy = dx * ODOM_SCALE_BIAS + self.rng.normal(0, 0.01 * dist), dy * ODOM_SCALE_BIAS
            dth = dth * ODOM_SCALE_BIAS + ODOM_YAW_DRIFT * dist + self.rng.normal(0, 0.002)
            self._dead_reckon(dx, dy, dth)
            dxp = dx * self.rng.normal(1.0, 0.05, n) + self.rng.normal(0, 0.005 + 0.05 * dist, n)
            dyp = dy * self.rng.normal(1.0, 0.05, n) + self.rng.normal(0, 0.005 + 0.05 * dist, n)
            dthp = dth * self.rng.normal(1.0, 0.05, n) + self.rng.normal(0, 0.004 + 0.05 * abs(dth), n)
        else:
            vx, vy, wz = self.cmd
            lim = GUIDE_SPEED if self.guiding else MAX_SPEED
            vx, vy = float(np.clip(vx, -lim, lim)), float(np.clip(vy, -lim, lim))
            sp = math.hypot(vx, vy)
            if sp > lim:
                vx, vy = vx * lim / sp, vy * lim / sp
            wz = float(np.clip(wz, -MAX_WZ, MAX_WZ))
            # noise grows with speed; a little even at rest (slip, unmodelled clamps)
            vxp = vx + self.rng.normal(0, 0.02 + 0.10 * abs(vx), n)
            vyp = vy + self.rng.normal(0, 0.02 + 0.10 * abs(vy), n)
            wzp = wz + self.rng.normal(0, 0.02 + 0.10 * abs(wz), n)
            dthp = wzp * dt
            self._dead_reckon(vx * dt, vy * dt, wz * dt)
            dxp, dyp = vxp * dt, vyp * dt
        yaw_mid = self.p[:, 2] + 0.5 * dthp
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        nx = self.p[:, 0] + c * dxp - s * dyp
        ny = self.p[:, 1] + s * dxp + c * dyp
        ok = self._dist_at(nx, ny) >= ROBOT_RADIUS     # blocked by a wall -> stays (like the sim's clamp)
        self.p[ok, 0], self.p[ok, 1] = nx[ok], ny[ok]
        self.p[:, 2] = _wrap(self.p[:, 2] + dthp)

    def _correct(self, ranges: Sequence[float]) -> None:
        r = np.asarray(ranges, float)
        if r.shape[0] != LIDAR_ANGLES.shape[0]:
            return
        hit = r < LIDAR_MAX - 0.05
        if hit.sum() < 3:
            return
        ang, rr = LIDAR_ANGLES[hit], r[hit]
        a = self.p[:, 2:3] + ang[None, :]                                   # (n, k)
        ex = self.p[:, 0:1] + rr[None, :] * np.cos(a)
        ey = self.p[:, 1:2] + rr[None, :] * np.sin(a)
        d = self._dist_at(ex, ey)
        lik = Z_HIT * np.exp(-0.5 * (d / SIGMA_HIT) ** 2) + Z_RAND
        # map fit: share of LiDAR hits within 2 sigma of a mapped wall, for the estimate (1 = all match)
        self.fit = float(np.dot(self.w, (d < 2 * SIGMA_HIT).mean(axis=1)))
        logw = RAY_TEMPER * np.log(lik).sum(axis=1)
        # a particle standing inside a wall is impossible
        logw[self._dist_at(self.p[:, 0], self.p[:, 1]) < 0.12] -= 20.0
        logw += np.log(self.w + 1e-300)
        logw -= logw.max()
        w = np.exp(logw)
        self.w = w / w.sum()
        self.n_eff = float(1.0 / np.sum(self.w ** 2))
        if self.n_eff < len(self.p) / 2:
            self._resample()

    def _resample(self) -> None:
        # after a global init keep the big particle set until the cloud has collapsed to one place
        m = self.n if (len(self.p) <= self.n or self.std_xy() < 1.0) else len(self.p)
        pos = (self.rng.random() + np.arange(m)) / m
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), len(self.p) - 1)
        self.p = self.p[idx]
        self.p[:, 0:2] += self.rng.normal(0, 0.01, (m, 2))   # roughening against depletion
        self.p[:, 2] = _wrap(self.p[:, 2] + self.rng.normal(0, 0.005, m))
        self.w = np.full(m, 1.0 / m)

    def update(self, msg: Dict[str, Any]) -> Tuple[float, float, float]:
        """One sim `state` message -> new pose estimate. `msg['robot_pose']` is only used for the error
        (and for delta odometry when NUDGE_LOCALIZE_ODOM=delta)."""
        t = float(msg.get("t", 0.0))
        gt = msg.get("robot_pose")
        gt = (float(gt[0]), float(gt[1]), float(gt[2])) if gt else None
        if self.t_last is not None:
            dt = t - self.t_last
            if dt < 0 or dt > 2.0:      # sim restarted / long gap: re-anchor to what we last believed
                dt = 0.0
            if dt > 0:
                self._predict(dt, gt)
                self._correct(msg.get("lidar_ranges") or [])
                self.updates += 1
        self.t_last = t
        self.last_gt = gt
        self.estimate = self._mean()
        if gt is not None:
            self.err_xy = math.hypot(self.estimate[0] - gt[0], self.estimate[1] - gt[1])
            self.err_yaw = abs(math.atan2(math.sin(self.estimate[2] - gt[2]), math.cos(self.estimate[2] - gt[2])))
            self.dr_err_xy = math.hypot(self.dr[0] - gt[0], self.dr[1] - gt[1])
            self.errs.append((t, self.err_xy, self.err_yaw, self.dr_err_xy))
            if len(self.errs) > 20000:
                del self.errs[:5000]
        return self.estimate

    def _mean(self) -> Tuple[float, float, float]:
        w = self.w
        x, y = float(np.dot(w, self.p[:, 0])), float(np.dot(w, self.p[:, 1]))
        yaw = math.atan2(float(np.dot(w, np.sin(self.p[:, 2]))), float(np.dot(w, np.cos(self.p[:, 2]))))
        return x, y, yaw

    def std_xy(self) -> float:
        x, y, _ = self.estimate
        return float(math.sqrt(np.dot(self.w, (self.p[:, 0] - x) ** 2 + (self.p[:, 1] - y) ** 2)))

    def room(self) -> str:
        return get_room_at(self.estimate[0], self.estimate[1])

    def status(self) -> Dict[str, Any]:
        """Brain status `localization` (GUI / pitch). err_* = vs the sim's pose: measurement only."""
        e = np.array([r[1] for r in self.errs]) if self.errs else np.zeros(1)
        return {
            "pose": [round(v, 3) for v in self.estimate],
            "std_xy": round(self.std_xy(), 3),
            "err_xy": None if self.err_xy is None else round(self.err_xy, 3),
            "err_yaw_deg": None if self.err_yaw is None else round(math.degrees(self.err_yaw), 1),
            "err_xy_mean": round(float(e.mean()), 3),
            "err_xy_max": round(float(e.max()), 3),
            "odom_only_err_xy": None if self.dr_err_xy is None else round(self.dr_err_xy, 3),
            "n_eff": round(self.n_eff, 1),
            "fit": None if self.fit is None else round(self.fit, 2),
            "n": self.n,
            "odom": self.odom,
        }
