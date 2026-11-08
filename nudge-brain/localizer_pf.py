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
  init    : Gaussian around the known start pose (house_layout.ROBOT_START_POSE) = mode "tracking" (default).
  global  : NUDGE_LOCALIZE_GLOBAL=1 (agent N): unknown start. The robot turns once on the spot (`wants_spin()`,
            main.py hook), LiDAR points are buffered in the body frame (motion compensated by odometry), then an
            exhaustive scan match (every free 25 cm cell x 72 headings, padded likelihood field, ~0.5-1 s) seeds
            3600 particles from the 300 best hypotheses. Converged = std < 0.3 m + fit > 0.7 for 2 s AND the
            buffered scan fits >= UNIQUE_MARGIN better near the estimate than anywhere > 1.5 m away (look-alike
            rooms); else search again. Kidnap recovery (flag on, while tracking): fit < 0.3 for 2 s -> lost ->
            search on the scans since the drop. A pose jump > 0.5 m in one step is not read as leg odometry.
            Status adds mode, converged, spinning, converge_s, relocalizations, margin.

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
# --- global localisation / kidnapped robot (NUDGE_LOCALIZE_GLOBAL=1) ---
SPIN_WZ = 1.2                # rad/s, the start-up look-around turn (one full turn ~5.5 s)
SPIN_TURN = 2.15 * math.pi   # turn this far before the first global search
GRID_STEP = 0.25             # m, candidate positions of the global search
GRID_YAWS = 72               # 5 deg candidate headings
SIGMA_GLOBAL = 0.22          # m, wider likelihood for the coarse search grid
SEED_HYPS, SEED_PER_HYP = 300, 12   # top hypotheses kept, particles per hypothesis
BUF_AGE = 8.0                # s of motion-compensated LiDAR points kept for the search
CONV_STD, CONV_FIT, CONV_HOLD = 0.30, 0.70, 40   # converged = std < 0.3 m and fit > 0.7 for 2 s
LOST_FIT, LOST_HOLD = 0.30, 40                   # tracking + fit < 0.3 for 2 s -> lost (kidnapped)
RESEARCH_EVERY = 4.0         # s, re-run the global search while not converged
UNIQUE_MARGIN = 6.0         # log-likelihood lead of the estimate over the best place > 1.5 m away
TELEPORT_JUMP = 0.5          # m in one state step: no leg can do that -> the odometry reads nothing


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
        self.global_flag = os.environ.get("NUDGE_LOCALIZE_GLOBAL") == "1"
        self._build_global_fields()
        self.reset(start_pose)
        if self.global_flag:
            self.global_init()

    # ------------------------------------------------------------------ init
    def reset(self, pose: Optional[Sequence[float]] = None, sxy: float = 0.15, syaw: float = 0.08) -> None:
        """Known start pose (tracking). With NUDGE_LOCALIZE_GLOBAL=1 and no pose given: global init."""
        if pose is None and getattr(self, "global_flag", False):
            self.reset(ROBOT_START_POSE)
            self.global_init()
            return
        x, y, yaw = (float(v) for v in (ROBOT_START_POSE if pose is None else pose))
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
        self.mode = "tracking"             # "global" while the robot does not know where it is
        self.converged = True
        self.spin_left = 0.0               # rad still to turn on the spot (start-up look-around)
        self.conv_count = 0
        self.lost_count = 0
        self.fit_ema: Optional[float] = None
        self.t_global: Optional[float] = None    # sim time the global phase started
        self.t_converged: Optional[float] = None
        self.t_search: Optional[float] = None
        self.searches = 0
        self.relocalizations = 0
        self.buf_hit = np.zeros((0, 3))          # body-frame LiDAR points (x, y, t), motion compensated
        self.buf_free = np.zeros((0, 3))         # body-frame points the beams passed through (free)
        self.odom_step = (0.0, 0.0, 0.0)
        self.margin: Optional[float] = None

    def global_init(self, spin: bool = True) -> None:
        """Unknown start / kidnapped robot. The robot first turns once on the spot (`wants_spin()`, start-up
        only) while its LiDAR points are collected in the body frame (motion compensated by odometry); then
        `_global_search()` scores every free 25 cm cell x 72 headings of the house against that 360 deg scan
        and seeds the filter with the best few hundred hypotheses. Until the convergence test passes, `mode`
        stays "global" and `converged` False (the brain should not trust the estimate yet)."""
        self.mode, self.converged = "global", False
        self.conv_count = 0
        self.t_global, self.t_converged, self.t_search = None, None, None
        self.spin_left = SPIN_TURN if spin else 0.0
        # until the first search: particles spread over the free floor (the status shows a big cloud)
        m = 4 * self.n
        pick = self.rng.integers(0, len(self.cand_xy), m)
        self.p = np.column_stack([self.cand_xy[pick, 0], self.cand_xy[pick, 1], self.rng.uniform(-np.pi, np.pi, m)])
        self.w = np.full(m, 1.0 / m)
        self.estimate = self._mean()

    def wants_spin(self) -> bool:
        """True while the start-up look-around turn is not finished (brain hook: command (0, 0, SPIN_WZ))."""
        return self.mode == "global" and self.spin_left > 0.0

    def _build_global_fields(self) -> None:
        d = self.dist
        pad = self.pad = int(LIDAR_MAX / self.res) + 2      # padded fields: no bounds checks in the search
        gh = np.log(Z_HIT * np.exp(-0.5 * (d / SIGMA_GLOBAL) ** 2) + 0.05).astype(np.float32)
        self.g_hit = np.pad(gh, pad, constant_values=np.log(0.05)).ravel()
        gf = np.where(d < 0.06, -1.5, 0.0).astype(np.float32)   # a beam passing through a wall
        self.g_free = np.pad(gf, pad, constant_values=0.0).ravel()
        self.Wp = self.W + 2 * pad
        b = HOUSE_BOUNDS
        step = max(1, int(round(GRID_STEP / self.res)))
        rr, cc = np.mgrid[0:self.H:step, 0:self.W:step]
        rr, cc = rr.ravel(), cc.ravel()
        fx, fy = self.x0 + (cc + 0.5) * self.res, self.y0 + (rr + 0.5) * self.res
        ok = (d[rr, cc] > 0.28) & (fx > b["x_min"]) & (fx < b["x_max"]) & (fy > b["y_min"]) & (fy < b["y_max"])
        self.cand_rc = np.column_stack([rr[ok], cc[ok]])
        self.cand_xy = np.column_stack([fx[ok], fy[ok]])

    def _buffer_scan(self, ranges: Sequence[float], t: float) -> None:
        """Add this scan to the body-frame buffer: LiDAR hits + points the beams passed through (free)."""
        r = np.asarray(ranges, float)
        if r.shape[0] != LIDAR_ANGLES.shape[0]:
            return
        hit = r < LIDAR_MAX - 0.05
        ca, sa = np.cos(LIDAR_ANGLES), np.sin(LIDAR_ANGLES)
        h = np.column_stack([r[hit] * ca[hit], r[hit] * sa[hit], np.full(int(hit.sum()), t)])
        free = []
        for k in (0.6, 1.2, 1.8, 2.4, 3.0, 3.6):
            m = r - 0.3 > k
            free.append(np.column_stack([k * ca[m], k * sa[m], np.full(int(m.sum()), t)]))
        self.buf_hit = np.vstack([self.buf_hit[self.buf_hit[:, 2] > t - BUF_AGE], h])
        self.buf_free = np.vstack([self.buf_free[self.buf_free[:, 2] > t - BUF_AGE]] + free)

    def _move_buffer(self, dx: float, dy: float, dth: float) -> None:
        """The body moved by (dx, dy, dth) (odometry reading): re-express the buffered points in the new frame."""
        c, s = math.cos(-dth), math.sin(-dth)
        for b in (self.buf_hit, self.buf_free):
            if len(b):
                px, py = b[:, 0] - dx, b[:, 1] - dy
                b[:, 0], b[:, 1] = c * px - s * py, s * px + c * py

    @staticmethod
    def _bin(pts: np.ndarray, nb: int, rstep: float = 0.0) -> np.ndarray:
        """Newest point per bearing bin (and per range ring if rstep): a small, even point set."""
        if not len(pts):
            return pts
        key = ((np.arctan2(pts[:, 1], pts[:, 0]) + np.pi) / (2 * np.pi) * nb).astype(np.int64)
        if rstep:
            key = key * 100 + (np.hypot(pts[:, 0], pts[:, 1]) / rstep).astype(np.int64)
        order = np.argsort(-pts[:, 2], kind="stable")
        _, first = np.unique(key[order], return_index=True)
        return pts[order[first]]

    def _scores(self):
        """Exhaustive scan match: every candidate cell x heading vs the buffered (up to 360 deg) scan."""
        near = lambda b: b[np.hypot(b[:, 0], b[:, 1]) < LIDAR_MAX]   # noqa: E731 (inside the padding)
        hits = self._bin(near(self.buf_hit), 90)
        if len(hits) < 8:
            return None
        free = self._bin(near(self.buf_free), 45, 0.6)
        if len(free) > 90:
            free = free[self.rng.choice(len(free), 90, replace=False)]
        pts, nh, P = np.vstack([hits[:, :2], free[:, :2]]), len(hits), self.cand_rc
        base = ((P[:, 0] + self.pad) * self.Wp + P[:, 1] + self.pad).astype(np.int64)
        yaws = np.linspace(-np.pi, np.pi, GRID_YAWS, endpoint=False)
        scores = np.empty((GRID_YAWS, len(P)), np.float32)
        for k, yaw in enumerate(yaws):
            c, s = math.cos(yaw), math.sin(yaw)
            dc = np.rint((c * pts[:, 0] - s * pts[:, 1]) / self.res).astype(np.int64)
            dr = np.rint((s * pts[:, 0] + c * pts[:, 1]) / self.res).astype(np.int64)
            off = dr * self.Wp + dc
            scores[k] = (self.g_hit[base[:, None] + off[None, :nh]].sum(axis=1)
                         + self.g_free[base[:, None] + off[None, nh:]].sum(axis=1))
        return scores, yaws

    def _global_search(self, t: float, sy=None) -> None:
        sy = sy or self._scores()
        if sy is None:
            return
        scores, yaws = sy
        P = self.cand_rc
        flat = scores.ravel()
        top = np.argpartition(-flat, SEED_HYPS)[:SEED_HYPS]
        ky, kp = np.divmod(top, len(P))
        sc = flat[top].astype(float)
        m = SEED_HYPS * SEED_PER_HYP
        rep = np.repeat(np.arange(SEED_HYPS), SEED_PER_HYP)
        self.p = np.column_stack([self.cand_xy[kp[rep], 0] + self.rng.normal(0, 0.08, m),
                                  self.cand_xy[kp[rep], 1] + self.rng.normal(0, 0.08, m),
                                  _wrap(yaws[ky[rep]] + self.rng.normal(0, 0.03, m))])
        w = np.exp(0.3 * (sc[rep] - sc.max()))      # the 360 deg evidence as a tempered prior (no early collapse)
        self.w = w / w.sum()
        self.estimate = self._mean()
        self.searches += 1
        self.t_search = t
        self.conv_count = 0

    def _unique(self):
        """Ambiguity test before trusting a converged cloud: the buffered scan must fit clearly better
        within 0.75 m of the estimate than anywhere > 1.5 m away (look-alike rooms)."""
        sy = self._scores()
        if sy is None:
            return False, None
        best = sy[0].max(axis=0)
        d = np.hypot(self.cand_xy[:, 0] - self.estimate[0], self.cand_xy[:, 1] - self.estimate[1])
        near, far = best[d < 0.75], best[d > 1.5]
        if not len(near) or not len(far):
            return bool(len(near)), sy
        self.margin = float(near.max() - far.max())
        return self.margin >= UNIQUE_MARGIN, sy

    def _update_mode(self, t: float) -> None:
        """Convergence test (global -> tracking) and lost detection (tracking -> global: kidnapped)."""
        if self.fit is not None:
            self.fit_ema = self.fit if self.fit_ema is None else 0.9 * self.fit_ema + 0.1 * self.fit
        if self.mode == "global":
            if self.t_global is None:
                self.t_global = t
            if self.spin_left > 0.0:
                return
            if self.t_search is None:
                self._global_search(t)
                return
            ok = self.std_xy() < CONV_STD and (self.fit_ema or 0.0) > CONV_FIT
            self.conv_count = self.conv_count + 1 if ok else 0
            if self.conv_count >= CONV_HOLD:
                unique, sy = self._unique()
                if not unique:              # a look-alike place fits as well: start again from all modes
                    self._global_search(t, sy)
                    return
                self.mode, self.converged, self.t_converged = "tracking", True, t
                self.lost_count = 0
                self._resample(force_n=self.n)
            elif t - self.t_search > RESEARCH_EVERY and (self.std_xy() > 1.0 or (self.fit_ema or 0.0) < CONV_FIT):
                self._global_search(t)      # still ambiguous, or one compact cloud on the wrong walls
        elif self.global_flag:
            low = self.fit_ema is not None and self.fit_ema < LOST_FIT
            self.lost_count = self.lost_count + 1 if low else 0
            if self.lost_count >= LOST_HOLD:   # kidnapped: keep only the scans since the fit dropped
                t_cut = t - LOST_HOLD * 0.05 - 0.5
                self.buf_hit = self.buf_hit[self.buf_hit[:, 2] >= t_cut]
                self.buf_free = self.buf_free[self.buf_free[:, 2] >= t_cut]
                self.relocalizations += 1
                self.global_init(spin=False)
                self.t_global = t
                self._global_search(t)

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
            if dist > TELEPORT_JUMP:     # picked up and put down elsewhere: the legs felt nothing
                dx = dy = dth = dist = 0.0
            # the odometry SENSOR is wrong, systematically and randomly (foot slip): this reading is all the
            # filter gets; the truth above never reaches the particles
            dx, dy = dx * ODOM_SCALE_BIAS + self.rng.normal(0, 0.01 * dist), dy * ODOM_SCALE_BIAS
            dth = dth * ODOM_SCALE_BIAS + ODOM_YAW_DRIFT * dist + self.rng.normal(0, 0.002)
            self._dead_reckon(dx, dy, dth)
            self.odom_step = (dx, dy, dth)
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
            self.odom_step = (vx * dt, vy * dt, wz * dt)
            dxp, dyp = vxp * dt, vyp * dt
        yaw_mid = self.p[:, 2] + 0.5 * dthp
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        nx = self.p[:, 0] + c * dxp - s * dyp
        ny = self.p[:, 1] + s * dxp + c * dyp
        ok = self._dist_at(nx, ny) >= ROBOT_RADIUS     # blocked by a wall -> stays (like the sim's clamp)
        self.p[ok, 0], self.p[ok, 1] = nx[ok], ny[ok]
        self.p[:, 2] = _wrap(self.p[:, 2] + dthp)
        if self.global_flag or self.mode == "global":
            self._move_buffer(*self.odom_step)
            if self.spin_left > 0.0:
                self.spin_left -= abs(self.odom_step[2])

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

    def _resample(self, force_n: Optional[int] = None) -> None:
        # global mode keeps the big particle set until the convergence test passes
        m = force_n or (len(self.p) if self.mode == "global" else self.n)
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
                if self.mode != "global" or self.t_search is not None:
                    self._correct(msg.get("lidar_ranges") or [])
                self.updates += 1
        if self.global_flag or self.mode == "global":
            self._buffer_scan(msg.get("lidar_ranges") or [], t)
        self.estimate = self._mean()
        self._update_mode(t)
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
            "n": len(self.p),
            "odom": self.odom,
            "mode": self.mode,
            "converged": self.converged,
            "spinning": self.wants_spin(),
            "converge_s": (None if self.t_converged is None or self.t_global is None
                           else round(self.t_converged - self.t_global, 1)),
            "relocalizations": self.relocalizations,
            "margin": None if self.margin is None else round(self.margin, 1),
        }
