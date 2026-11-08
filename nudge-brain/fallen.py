#!/usr/bin/env python3
"""
Fallen owner (`nudge-brain/fallen.py`, Finn): find and help an owner LYING ON THE FLOOR.

YOLO does not detect a person lying on the floor (Teo, AGENTS.md 12.7); Gemma's camera look does.
So, while the robot is on a mission (searching, approaching, ...):

  1. Floor finder (plain geometry, every camera frame, ~2 ms): project the depth image into the
     world; points 5-55 cm above the floor that the static floor-plan map does not explain (walls,
     furniture) are a "something on the floor" cluster. Clusters with tall unexplained points next
     to them (a standing / sitting person, a chair back) are ignored.
  2. Is it a person? Either the person detector (YOLO / HSV) fires on the same spot - a person with
     nothing upright above them is lying down (no Gemma needed) - or, when YOLO sees nobody (it misses
     the scanned owner lying down), Gemma confirms (background thread, one call at a time, never while
     the agent waits for a decision): Teo's OwnerChecker.check() on that frame, cropped to the cluster.
     state = lying_down with confidence >= 0.5. Either way -> the owner has fallen, at the cluster's centre (owner source
     "gemma_floor"); the nearest cluster point sets a safe standoff (never drive onto them).
  3. Response (plain code, takes over from the agent until it is resolved):
       approach to ~0.9 m -> "I found you on the floor. Are you OK?" (C5 ask, yes/no, 10 s)
       yes -> "OK." and the mission carries on;  no / no answer -> 999 (C8 reason "fallen"),
       "I am calling 999. Help is coming. I will stay with you." and stays with them.
       In a fire: 999 at once (no question) and stays with them - you can't lead someone out who
       is lying on the floor.

On with NUDGE_FALLEN_CHECK=1 or Teo's NUDGE_OWNER_CHECK=1; otherwise FallenWatch.from_env() is None
and the brain behaves exactly as before. Brain status: `fallen` = {"stage", "xy", "confidence",
"description", "t"} | null.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from sim.house_layout import room_at
from skills import Skill, SkillOutput, _nav_step
from navigator import wrap

CAM_W, CAM_H, FOVY_DEG = 320, 240, 60.0
FOCAL = 0.5 * CAM_H / math.tan(math.radians(FOVY_DEG / 2))
CAM_FORWARD = 0.30                # head_cam ahead of the base centre (vision.py)
CAM_Z = 0.39                      # base height 0.27 + head_cam 0.12
STEP = 2                          # use every 2nd pixel (160 x 120)
LOW_Z = (0.05, 0.55)              # a person lying on the floor
TALL_Z = (0.75, 1.9)              # someone standing / sitting, chair backs...
MAX_RANGE = 4.5
MAP_CLEAR = 0.15                  # points this close to the static map are walls / furniture
CELL = 0.10
MIN_POINTS = 60                   # (at STEP 2) ~ a body at 3 m
MIN_EXTENT = 0.45                 # m, longest side of the cluster
MAX_TALL = 12                     # tall unexplained points allowed next to the cluster
MIN_CONF = 0.5
DETECT_MATCH_M = 1.0              # a person detection this close to the floor cluster = the same person
GAP_S = 2.5                       # between Gemma looks
REJECT_S = 25.0                   # don't re-ask Gemma about a spot it said "no person" for
STANDOFF = 0.9                    # m from the nearest body point
ASK_TIMEOUT_S = 10.0
OK_QUIET_S = 120.0                # after "yes, I'm OK": don't re-trigger for this long
APPROACH_TIMEOUT_S = 30.0

SAY_SEEN = "I can see you on the floor. I am coming."
SAY_ASK = "I found you on the floor. Are you OK?"
SAY_OK = "OK. I am glad you are OK. I am here if you need me."
SAY_CALLING = "I am calling 999. Help is coming. I will stay with you."
SAY_FIRE = "Smoke alarm! You are on the floor. I have called 999. Help is coming. I will stay with you."


def enabled() -> bool:
    return os.environ.get("NUDGE_FALLEN_CHECK", "0") == "1" or os.environ.get("NUDGE_OWNER_CHECK", "0") == "1"


class FloorFinder:
    """Depth image -> 'something lying on the floor that the map doesn't explain' (or None)."""

    def __init__(self, planner) -> None:
        self.x0, self.y0, self.res = planner.x0, planner.y0, planner.res
        self.dist = (ndimage.distance_transform_edt(~planner.occ) * planner.res).astype(np.float32)
        vs, us = np.mgrid[0:CAM_H:STEP, 0:CAM_W:STEP]
        self.us, self.vs = us.ravel(), vs.ravel()
        self.k_left = -(self.us - CAM_W / 2.0) / FOCAL
        self.k_up = -(self.vs - CAM_H / 2.0) / FOCAL

    def _map_dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.clip(((y - self.y0) / self.res).astype(int), 0, self.dist.shape[0] - 1)
        c = np.clip(((x - self.x0) / self.res).astype(int), 0, self.dist.shape[1] - 1)
        return self.dist[r, c]

    def find(self, depth: np.ndarray, pose: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        d = np.asarray(depth, dtype=np.float32)[self.vs, self.us]
        valid = np.isfinite(d) & (d > 0.3) & (d < MAX_RANGE)
        if not valid.any():
            return None
        d, left, up, us, vs = d[valid], self.k_left[valid] * d[valid], self.k_up[valid] * d[valid], \
            self.us[valid], self.vs[valid]
        rx, ry, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        cx, cy = rx + CAM_FORWARD * c, ry + CAM_FORWARD * s
        x = cx + d * c - left * s
        y = cy + d * s + left * c
        z = CAM_Z + up
        free = self._map_dist(x, y) > MAP_CLEAR
        low = free & (z > LOW_Z[0]) & (z < LOW_Z[1])
        if low.sum() < MIN_POINTS:
            return None
        tall = free & (z > TALL_Z[0]) & (z < TALL_Z[1])
        # cluster the low points on a 10 cm grid
        gx0, gy0 = float(x[low].min()) - 0.3, float(y[low].min()) - 0.3
        gi = ((y - gy0) / CELL).astype(int)
        gj = ((x - gx0) / CELL).astype(int)
        h, w = int(gi[low].max()) + 4, int(gj[low].max()) + 4
        grid = np.zeros((h, w), np.int32)
        np.add.at(grid, (gi[low], gj[low]), 1)
        labels, n = ndimage.label(ndimage.binary_dilation(grid > 0), structure=np.ones((3, 3)))
        if n == 0:
            return None
        lab_low = labels[gi[low], gj[low]]
        counts = np.bincount(lab_low, minlength=n + 1)
        counts[0] = 0
        best = int(np.argmax(counts))
        if counts[best] < MIN_POINTS:
            return None
        sel = np.zeros_like(low)
        sel[np.flatnonzero(low)[lab_low == best]] = True
        px, py = x[sel], y[sel]
        extent = max(float(px.max() - px.min()), float(py.max() - py.min()))
        if extent < MIN_EXTENT:
            return None
        inside = (gi >= 0) & (gi < h) & (gj >= 0) & (gj < w)
        near_cells = ndimage.binary_dilation(labels == best, iterations=3)
        tall_near = int((tall & inside & near_cells[np.clip(gi, 0, h - 1), np.clip(gj, 0, w - 1)]).sum())
        if tall_near > MAX_TALL:
            return None  # someone (or something) upright: not a person lying on the floor
        centre = (float(px.mean()), float(py.mean()))
        if room_at(*centre) is None:
            return None
        k = int(np.argmin(np.hypot(px - rx, py - ry)))
        su, sv = us[sel], vs[sel]
        return {"xy": centre, "near": (float(px[k]), float(py[k])), "points": int(sel.sum()),
                "extent": round(extent, 2), "tall_near": tall_near,
                "bbox": (int(su.min()), int(sv.min()), int(su.max()) + STEP, int(sv.max()) + STEP)}


class ApproachFallen(Skill):
    """Drive to a standoff point next to the fallen owner (never onto them), then face them."""

    name = "check_on_fallen_owner"

    def __init__(self, world, nav, standoff: Tuple[float, float], body: Tuple[float, float]) -> None:
        super().__init__()
        self.standoff, self.body, self.reached = standoff, body, False
        nav.set_goal(standoff, world.robot_pose[:2], world.active_blocked_doors(), final_tol=0.3)

    def label(self) -> str:
        return "check_on_fallen_owner" + (" [with them]" if self.reached else " [approach]")

    def step(self, world, nav, alerts) -> SkillOutput:
        rx, ry, yaw = world.robot_pose
        if not self.reached:
            vx, vy, wz, status = _nav_step(world, nav, 0.6)
            near = math.hypot(self.standoff[0] - rx, self.standoff[1] - ry) < 0.35
            if status in ("arrived", "stuck", "idle") or near or time.monotonic() - self.started > APPROACH_TIMEOUT_S:
                nav.clear()
                self.reached = True
            else:
                return SkillOutput(vx, vy, wz)
        err = wrap(math.atan2(self.body[1] - ry, self.body[0] - rx) - yaw)  # camera + screen to the owner
        return SkillOutput(wz=max(-1.0, min(1.0, 1.5 * err)) if abs(err) > 0.2 else 0.0)


class FallenWatch:
    """Finds a fallen owner and runs the response. Hooks (main.py): on_frame, tick, reset, status."""

    def __init__(self, planner, checker, busy) -> None:
        self.finder = FloorFinder(planner)
        self.checker = checker            # gemma_vision.OwnerChecker (only .check() is used)
        self.busy = busy
        self._lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None
        self._result: Optional[Dict[str, Any]] = None
        self._wake = threading.Event()
        self._last_call = -1e9
        self.calls = 0
        threading.Thread(target=self._worker, daemon=True, name="fallen-check").start()
        self.reset()

    @classmethod
    def from_env(cls, planner, owner_check=None, busy=lambda: False) -> Optional["FallenWatch"]:
        if not enabled():
            return None
        if owner_check is None:
            from gemma_vision import OwnerChecker
            owner_check = OwnerChecker(busy=busy, log=lambda *_: None)  # its own thread idles (no frames)
        watch = cls(planner, owner_check, busy)
        print(f"[fallen] ON: floor finder + Gemma look ({owner_check.model}) while on a mission", flush=True)
        return watch

    def reset(self) -> None:
        with self._lock:
            self._job, self._result = None, None
        self.stage: Optional[str] = None   # None | approach | asking | ok | staying
        self.info: Optional[Dict[str, Any]] = None
        self.skill: Optional[Skill] = None
        self.quiet_until = 0.0
        self.rejected: list = []           # (xy, wall time) Gemma said "no person lying here"
        self.ask_id = None
        self.frames = self.candidates = 0

    # ------------------------------------------------------------------ perception (main thread)
    def on_frame(self, rgb, depth, pose, world) -> None:
        if self.stage is not None or world.mission is None or world.guiding_active:
            return
        now = time.monotonic()
        if now < self.quiet_until or now - self._last_call < GAP_S or self.busy():
            return
        with self._lock:
            if self._job is not None:
                return
        cand = self.finder.find(depth, pose)
        self.frames += 1
        if cand is None:
            return
        self.candidates += 1
        if self.candidates == 1 or self.candidates % 20 == 0:
            world.log(f"FALLEN candidate #{self.candidates}: low cluster at ({cand['xy'][0]:.1f}, {cand['xy'][1]:.1f}) "
                      f"n={cand['points']} extent={cand['extent']} m")
        det = world.last_detection or {}
        if det.get("found") and det.get("world_xy") is not None:
            dx, dy = det["world_xy"]
            if min(math.hypot(dx - cand["xy"][0], dy - cand["xy"][1]),
                   math.hypot(dx - cand["near"][0], dy - cand["near"][1])) < DETECT_MATCH_M:
                # The person detector sees someone exactly where everything is below 55 cm: lying down.
                with self._lock:
                    self._result = {"rgb": rgb, "cand": cand, "t": world.t_sim, "pose": pose, "res": {
                        "state": "lying_down", "confidence": 0.8, "latency": 0.0, "source": det.get("backend"),
                        "description": "person detector + depth: lying on the floor"}}
                return
        self.rejected = [(p, t) for p, t in self.rejected if now - t < REJECT_S]
        if any(math.hypot(p[0] - cand["xy"][0], p[1] - cand["xy"][1]) < 0.8 for p, _ in self.rejected):
            return
        self._last_call = now
        with self._lock:
            self._job = {"rgb": rgb, "cand": cand, "t": world.t_sim, "pose": pose}
        self._wake.set()

    def _worker(self) -> None:
        while True:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            try:
                res = self.checker.check(job["rgb"], job["cand"]["bbox"])
            except Exception as exc:  # Ollama down etc.: fail soft
                res = {"state": "error", "confidence": 0.0, "description": str(exc)[:60], "latency": 0.0}
            self.calls += 1
            with self._lock:
                self._result = {**job, "res": res}
                self._job = None

    # ------------------------------------------------------------------ response (main thread)
    def tick(self, brain) -> None:
        w = brain.world
        with self._lock:
            done, self._result = self._result, None
        if done is not None and self.stage is None:
            res, cand = done["res"], done["cand"]
            if res.get("state") == "lying_down" and float(res.get("confidence", 0)) >= MIN_CONF:
                self._found(brain, cand, res)
            else:
                self.rejected.append((cand["xy"], time.monotonic()))
                w.log(f"FALLEN check: something on the floor at ({cand['xy'][0]:.1f}, {cand['xy'][1]:.1f}), "
                      f"Gemma: {res.get('state')} ({float(res.get('confidence', 0)):.2f}) - not a fallen person")
        if self.stage is None or self.stage == "ok":
            return
        if w.is_emergency() and self.stage == "asking":
            self._call(brain, by="rules", asked=False, answer=None, say=SAY_FIRE, detail="smoke alarm")
        # Keep control until this is resolved (a pending agent decision must not drive away).
        if brain.executor.skill is not self.skill:
            brain.agent.cancel()
            brain.executor.skill = self.skill
            w.phase = "fallen"
        if self.stage == "approach" and getattr(self.skill, "reached", False):
            if w.is_emergency():
                self.stage = "staying"
            else:
                self._ask(brain)

    def _found(self, brain, cand: Dict[str, Any], res: Dict[str, Any]) -> None:
        w = brain.world
        xy, near = cand["xy"], cand["near"]
        w._set_owner(xy, "gemma_floor")
        standoff = self._standoff(near, w.robot_pose[:2])
        self.info = {"stage": "approach", "xy": [round(xy[0], 2), round(xy[1], 2)],
                     "confidence": res.get("confidence"), "description": res.get("description", ""),
                     "t": round(w.t_sim, 1), "source": "gemma_floor"}
        w.fallen = self.info
        w.log(f"FALLEN: owner LYING ON THE FLOOR at ({xy[0]:.1f}, {xy[1]:.1f}) in {room_at(*xy)} - Gemma "
              f"{res.get('confidence')} ({res.get('latency', 0):.1f}s): {res.get('description', '')}")
        brain.agent.cancel()
        brain.nav.clear()
        self.skill = ApproachFallen(w, brain.nav, standoff, xy)
        brain.executor.skill = self.skill
        w.phase = "fallen"
        self.stage = "approach"
        w.log(f"SKILL {self.skill.label()} -> standoff ({standoff[0]:.1f}, {standoff[1]:.1f})")
        if w.is_emergency():
            self._call(brain, by="rules", asked=False, answer=None, say=SAY_FIRE, detail="smoke alarm")
        else:
            brain.tell(SAY_SEEN)

    def _standoff(self, near: Tuple[float, float], robot: Tuple[float, float]) -> Tuple[float, float]:
        """A free spot STANDOFF from the nearest body point, as close to the robot's side as possible."""
        base = math.atan2(robot[1] - near[1], robot[0] - near[0])
        for k in range(0, 13):
            for sgn in ((1,) if k == 0 else (1, -1)):
                a = base + sgn * k * math.radians(15)
                px, py = near[0] + STANDOFF * math.cos(a), near[1] + STANDOFF * math.sin(a)
                if float(self.finder._map_dist(np.array([px]), np.array([py]))[0]) >= 0.32:
                    return px, py
        return near[0] + STANDOFF * math.cos(base), near[1] + STANDOFF * math.sin(base)

    def _ask(self, brain) -> None:
        w = brain.world
        self.stage = self.info["stage"] = "asking"
        brain.tell(SAY_ASK)

        def on_answer(choice) -> None:
            if self.stage != "asking":
                return
            answer = {"yes": "yes", "no": "no"}.get(choice, "timeout")
            if answer == "yes":
                self.stage = self.info["stage"] = "ok"
                self.quiet_until = time.monotonic() + OK_QUIET_S
                brain.world.log("FALLEN: owner answered yes (OK) - carrying on")
                brain.tell(SAY_OK)
                brain.world.phase = "searching" if brain.world.mission else "idle"
                if brain.world.mission is not None:
                    brain.agent.submit("owner_seen", brain.world)
            else:
                self._call(brain, by="owner" if answer == "no" else "timeout", asked=True, answer=answer,
                           say=SAY_CALLING,
                           detail="owner said no" if answer == "no" else f"no answer in {ASK_TIMEOUT_S:.0f} s")

        self.ask_id = brain.ask.ask("Are you OK?", w.t_sim, on_answer, choices=["yes", "no"],
                                    timeout_s=ASK_TIMEOUT_S)
        w.log(f'ASK {self.ask_id}: "Are you OK?" (owner on the floor; no answer in {ASK_TIMEOUT_S:.0f} s = call 999)')

    def _call(self, brain, by: str, asked: bool, answer, say: str, detail: str) -> None:
        w = brain.world
        if self.ask_id is not None and brain.ask.current and brain.ask.current.get("id") == self.ask_id:
            brain.ask.clear()
        if not brain.escalate("fallen", "999", by=by, asked=asked, answer=answer,
                              detail=f"owner lying on the floor ({detail})", say=say):
            brain.tell(say)  # 999 already called by another path: still tell the owner
        self.stage = self.info["stage"] = "staying" if self.stage != "approach" else "approach"
        if self.stage == "staying":
            w.log("FALLEN: staying with the owner until help comes")

    def status(self) -> Optional[Dict[str, Any]]:
        return dict(self.info, calls=self.calls) if self.info else None
