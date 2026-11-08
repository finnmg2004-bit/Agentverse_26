#!/usr/bin/env python3
"""Option A: the robot's own occupancy map (E3.1, improved 2026-11-08).

`OccupancyGrid` integrates the 2D LiDAR + robot pose into a 5 cm log-odds grid. Ground truth
(house_layout, nav_map, owner_pos_gt, objects_gt) is never read. Pose comes from the sim (a real robot
would use SLAM).

Change detection: when a saved (explored) map is loaded it is the *base*; a LiDAR hit in a cell the base
knows as clearly free (nothing occupied within 10 cm) is a candidate "appeared" cell. Cells hit in at
least CHANGE_MIN_HITS scans are confirmed; a cell seen free again later is cleared. Connected confirmed
cells (>= CHANGE_MIN_CELLS) are reported as `changes`. Without a base map the grid's own history is the
reference (the old E3.5 behaviour).

The map can be saved / loaded (`.npz`: log-odds + objects + rooms) so a video can start from an explored
map (`assets/maps/robot_map.npz`, git-ignored). `nav_occupancy()` gives the boolean grid the planner
uses (sim.nav_map.NavMap-compatible).

As its own process it publishes topic `robot_map` on :5559 at 1 Hz:
    [b"robot_map", header_json, png_bytes]
header: {"t", "res", "origin": [x0, y0], "shape": [h, w], "known_frac", "pose", "encoding": "png-u8",
         "changes": [{kind, x, y, cells, t}], "objects"?, "rooms"?}
PNG pixel values: 127 = unknown, 0 = free, 255 = occupied (row 0 = y0, i.e. not flipped).

    pixi run python nudge-brain/robot_map.py [--objects] [--load assets/maps/robot_map.npz] [--save map.png]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import zmq
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "nudge-brain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RES = 0.05
ORIGIN = (-17.0, -7.0)          # covers the house footprint plus the outside exits
SIZE_M = (34.0, 26.0)
if os.environ.get("NUDGE_HOUSE", "").strip().lower() not in ("", "house"):   # other house (--house apt):
    from sim import house_layout as _H                                       # grid follows its footprint
    _b = _H.HOUSE_BOUNDS
    ORIGIN = (_b["x_min"] - 2.0, _b["y_min"] - 2.5)
    SIZE_M = (_b["x_max"] - _b["x_min"] + 4.0, _b["y_max"] - _b["y_min"] + 4.0)
MAX_RANGE = 4.0
FOV = np.pi                      # [-90, +90] deg, matches sim/sensors.py
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -4.0, 4.0
OCC_T, FREE_T = 0.5, -0.3        # log-odds thresholds: occupied / free (between = unknown)
CAM_W, CAM_H, CAM_FOVY = 320, 240, 60.0   # head_cam (sim/sensors.py)
CAM_FOCAL = 0.5 * CAM_H / np.tan(np.radians(CAM_FOVY / 2))
CAM_FWD, CAM_Z = 0.30, 0.39      # camera ahead of / above the base centre
CAM_BAND = (0.10, 0.55)          # obstacles in this height band block the robot (same band as sim/nav_map.py)
CAM_RANGE = 3.0
CAM_MIN_HITS = 2                 # depth frames that must see an obstacle in a cell
CHANGE_MIN_HITS = 3              # scans hitting a cell before it counts as appeared
CHANGE_MIN_CELLS = 2             # appeared cells (grouped within CHANGE_GROUP_M) before a change is reported
CHANGE_GROUP_M = 0.20            # LiDAR rays are 5 deg apart: hits on one object are ~10-30 cm apart
DEFAULT_MAP_FILE = _ROOT / "assets" / "maps" / "robot_map.npz"


class OccupancyGrid:
    def __init__(self, res: float = RES, origin=ORIGIN, size_m=SIZE_M) -> None:
        self.res = res
        self.origin = tuple(origin)
        self.w = int(round(size_m[0] / res))
        self.h = int(round(size_m[1] / res))
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self.base_free: Optional[np.ndarray] = None   # clearly-free cells of a loaded (explored) map
        self.appear = np.zeros((self.h, self.w), dtype=np.int16)   # hits in reference-free cells
        self.appear_t = np.zeros((self.h, self.w), dtype=np.float32)
        # head-camera depth: obstacles in the robot's height band the 2D LiDAR plane misses (table tops)
        self.cam_hits = np.zeros((self.h, self.w), dtype=np.int16)
        self.change_log: List[Dict] = []              # appeared / cleared events, newest last
        self._reported: List[Dict] = []
        self._t0 = time.time()
        self.version = 0                              # bumped when the confirmed changes change

    # ---------------------------------------------------------------- geometry
    def _cell(self, x: float, y: float):
        return int(round((x - self.origin[0]) / self.res)), int(round((y - self.origin[1]) / self.res))

    def world(self, cx: float, cy: float):
        return self.origin[0] + cx * self.res, self.origin[1] + cy * self.res

    def bounds(self):
        """(x0, y0, x1, y1) for sim.nav_map.NavMap (cell centres sit at origin + i*res)."""
        x0, y0 = self.origin[0] - self.res / 2, self.origin[1] - self.res / 2
        return (x0, y0, x0 + self.w * self.res, y0 + self.h * self.res)

    # ---------------------------------------------------------------- update
    def integrate(self, pose, ranges, t: Optional[float] = None) -> None:
        rx, ry, yaw = pose
        n = len(ranges)
        if n == 0:
            return
        t = (time.time() - self._t0) if t is None else t
        angles = yaw + np.linspace(-FOV / 2, FOV / 2, n)
        free = np.zeros_like(self.logodds, dtype=np.uint8)
        occ = []
        c0 = self._cell(rx, ry)
        rs = np.minimum(np.asarray(ranges, dtype=float), MAX_RANGE)
        for a, r in zip(angles, ranges):
            hit = r < MAX_RANGE - 1e-3
            rr = float(r) if hit else MAX_RANGE
            ex, ey = rx + rr * np.cos(a), ry + rr * np.sin(a)
            c1 = self._cell(ex, ey)
            cv2.line(free, c0, c1, 1, 1)
            if hit:
                occ.append(c1)
        # Fill the wedge between neighbouring rays up to the nearer of the two ranges (minus 10 cm),
        # so the floor between 5-degree-apart rays is known too, without freeing cells behind corners.
        tris = []
        for i in range(n - 1):
            rm = max(0.0, min(rs[i], rs[i + 1]) - 0.10)
            if rm < 0.2:
                continue
            p1 = self._cell(rx + rm * np.cos(angles[i]), ry + rm * np.sin(angles[i]))
            p2 = self._cell(rx + rm * np.cos(angles[i + 1]), ry + rm * np.sin(angles[i + 1]))
            tris.append(np.array([c0, p1, p2], dtype=np.int32))
        if tris:
            cv2.fillPoly(free, tris, 1)
        mask = free.astype(bool)
        occ = [(cx, cy) for cx, cy in occ if 0 <= cx < self.w and 0 <= cy < self.h]
        for cx, cy in occ:          # the end cell of a hit is applied as occupied, not free
            mask[cy, cx] = False
        # a cell some ray now passes through is no longer an appeared obstacle
        cleared = mask & (self.appear > 0)
        if cleared.any():
            self.appear[cleared] = np.maximum(self.appear[cleared] - 2, 0)
        self.logodds[mask] += L_FREE
        for cx, cy in occ:
            if self.appear[cy, cx] > 0 or self._ref_free(cx, cy):
                if self.appear[cy, cx] == 0:
                    self.appear_t[cy, cx] = t
                self.appear[cy, cx] = min(self.appear[cy, cx] + 1, 50)
            self.logodds[cy, cx] += L_OCC
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)

    def integrate_depth(self, pose, depth: np.ndarray, step: int = 4) -> int:
        """Head-camera depth image -> obstacle cells in the CAM_BAND height band. Returns #points used."""
        d = depth[::step, ::step].astype(np.float32)
        vv, uu = np.mgrid[0:CAM_H:step, 0:CAM_W:step]
        ok = np.isfinite(d) & (d > 0.2) & (d < CAM_RANGE)
        z = CAM_Z - (vv - CAM_H / 2.0) / CAM_FOCAL * d
        ok &= (z > CAM_BAND[0]) & (z < CAM_BAND[1])
        if not ok.any():
            return 0
        fwd, left = d[ok], -(uu[ok] - CAM_W / 2.0) / CAM_FOCAL * d[ok]
        rx, ry, yaw = pose
        c, s = np.cos(yaw), np.sin(yaw)
        wx = rx + (CAM_FWD + fwd) * c - left * s
        wy = ry + (CAM_FWD + fwd) * s + left * c
        cx = np.round((wx - self.origin[0]) / self.res).astype(int)
        cy = np.round((wy - self.origin[1]) / self.res).astype(int)
        inb = (cx >= 0) & (cx < self.w) & (cy >= 0) & (cy < self.h)
        cells = np.unique(cy[inb] * self.w + cx[inb])
        flat = self.cam_hits.reshape(-1)
        flat[cells] = np.minimum(flat[cells] + 1, 100)
        return int(ok.sum())

    def _ref_free(self, cx: int, cy: int) -> bool:
        """Reference says this cell was clearly free, nothing occupied within 10 cm (pose / ray
        quantisation at wall edges is not a change)."""
        y0, y1, x0, x1 = max(cy - 2, 0), cy + 3, max(cx - 2, 0), cx + 3
        if self.base_free is not None:
            return bool(self.base_free[y0:y1, x0:x1].all())
        return bool(self.logodds[cy, cx] < -1.2 and np.all(self.logodds[y0:y1, x0:x1] < 0.3))

    # ---------------------------------------------------------------- changes
    def appeared_mask(self) -> np.ndarray:
        return self.appear >= CHANGE_MIN_HITS

    def current_changes(self, t: Optional[float] = None) -> List[Dict]:
        """Connected appeared cells -> [{kind, x, y, cells, t}] (t = first seen). Logs appear/clear."""
        m = self.appeared_mask()
        out: List[Dict] = []
        if m.any():
            k_it = max(1, int(round(CHANGE_GROUP_M / self.res / 2)))
            lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=k_it), structure=np.ones((3, 3)))
            lab = lab * m                                  # labels of the grouped blobs, on real cells only
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                cells = np.argwhere(lab[sl] == k) + [sl[0].start, sl[1].start]
                if len(cells) < CHANGE_MIN_CELLS:
                    continue
                x, y = self.world(float(cells[:, 1].mean()), float(cells[:, 0].mean()))
                t0 = float(self.appear_t[cells[:, 0], cells[:, 1]].min())
                out.append({"kind": "appeared", "x": round(x, 2), "y": round(y, 2), "cells": int(len(cells)),
                            "t": round(t0, 1)})
        self._diff_log(out, t)
        return out

    def _diff_log(self, now: List[Dict], t: Optional[float]) -> None:
        t = round((time.time() - self._t0) if t is None else t, 1)
        def near(a, lst):
            return any(np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 0.6 for b in lst)
        changed = False
        for c in now:
            if not near(c, self._reported):
                self.change_log.append(dict(c, event="appeared", t_event=t))
                print(f"[robot_map] CHANGE: new obstacle near ({c['x']:.1f}, {c['y']:.1f}), {c['cells']} cells, t={t:.0f}s",
                      flush=True)
                changed = True
        for c in self._reported:
            if not near(c, now):
                self.change_log.append(dict(c, event="cleared", t_event=t))
                print(f"[robot_map] CHANGE: obstacle near ({c['x']:.1f}, {c['y']:.1f}) gone, t={t:.0f}s", flush=True)
                changed = True
        if changed or len(now) != len(self._reported):
            self.version += 1
        self._reported = now
        del self.change_log[:-50]

    def confirmed_changes(self, min_cells: int = CHANGE_MIN_CELLS) -> List[Dict]:   # old name, kept
        return [c for c in self.current_changes() if c["cells"] >= min_cells]

    # ---------------------------------------------------------------- views
    def occupied(self) -> np.ndarray:
        return (self.logodds > OCC_T) | (self.cam_hits >= CAM_MIN_HITS)

    def free(self) -> np.ndarray:
        return (self.logodds < FREE_T) & (self.cam_hits < CAM_MIN_HITS)

    def known(self) -> np.ndarray:
        return self.occupied() | self.free()

    def to_image(self) -> np.ndarray:
        img = np.full(self.logodds.shape, 127, dtype=np.uint8)
        img[self.free()] = 0
        img[self.occupied()] = 255
        return img

    def known_frac(self) -> float:
        return float(np.mean((np.abs(self.logodds) > 0.3) | (self.cam_hits >= CAM_MIN_HITS)))

    def nav_occupancy(self, unknown_is_occupied: bool = True, with_changes: bool = True) -> np.ndarray:
        """Boolean grid (row = y) for sim.nav_map.NavMap: occupied cells (+ unknown, by default).
        A loaded base map's obstacles stay; live hits only count through confirmed changes, so the
        owner walking past or LiDAR noise does not slowly fill doorways."""
        if self.base_free is not None:
            # base obstacles stay unless the live LiDAR has since seen the cell clearly free many times
            # (something was moved away); live hits only count through confirmed changes.
            live_free = (self.logodds < -1.0) & (self.cam_hits < CAM_MIN_HITS)
            occ = (~self.base_free if unknown_is_occupied else self._base_occ) & ~live_free
            occ = occ | (self.cam_hits >= CAM_MIN_HITS)
            if with_changes:
                occ = occ | self.appeared_mask()
            return occ
        occ = self.occupied()
        if unknown_is_occupied:
            occ = occ | ~self.free()
        return occ

    def header(self, pose) -> dict:
        return {"t": time.time(), "res": self.res, "origin": list(self.origin),
                "shape": [self.h, self.w], "known_frac": round(self.known_frac(), 4),
                "pose": [round(float(v), 3) for v in pose], "encoding": "png-u8"}

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path, objects: Optional[list] = None, rooms: Optional[list] = None,
             meta: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, logodds=self.logodds.astype(np.float16), cam_hits=self.cam_hits, res=self.res,
                            origin=np.array(self.origin), size=np.array([self.w * self.res, self.h * self.res]),
                            objects=json.dumps(objects or []), rooms=json.dumps(rooms or []),
                            meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path: Path, as_base: bool = True):
        """-> (grid, objects, rooms, meta). as_base: the loaded map is the change-detection reference."""
        z = np.load(Path(path), allow_pickle=False)
        g = cls(float(z["res"]), tuple(float(v) for v in z["origin"]), tuple(float(v) for v in z["size"]))
        g.logodds[:] = z["logodds"].astype(np.float32)
        if "cam_hits" in z.files:
            g.cam_hits[:] = z["cam_hits"]
        if as_base:
            g.set_base()
        return g, json.loads(str(z["objects"])), json.loads(str(z["rooms"])), json.loads(str(z["meta"]))

    def set_base(self) -> None:
        self.base_free = (self.logodds < -1.0) & (self.cam_hits < CAM_MIN_HITS)
        self._base_occ = self.occupied()
        self.appear[:] = 0


class MapBuilder:
    """Grid + optional object memory fed straight from the sim's ZMQ messages (used by this process,
    scripts/test_mapping.py and the brain's flagged robot-map mode)."""

    def __init__(self, objects: bool = False, grid: Optional[OccupancyGrid] = None) -> None:
        self.grid = grid or OccupancyGrid()
        self.omem = self.odet = None
        if objects:
            from object_memory import ObjectDetector, ObjectMemory
            self.odet, self.omem = ObjectDetector(), ObjectMemory()
            self.omem.grid = self.grid
        self.pose = (0.0, 0.0, 0.0)
        self.episode = None
        self.last_det = 0.0
        self.det_period = 0.5
        self.use_depth = True

    def on_state(self, st: dict) -> bool:
        """Returns True if the episode changed (map was reset)."""
        reset = False
        if self.episode is not None and st.get("episode") != self.episode:
            base = self.grid.base_free is not None
            if not base:                                  # R reset: start a fresh map (a loaded one stays)
                self.grid = OccupancyGrid()
                if self.omem is not None:
                    self.omem.objects.clear()
                    self.omem.grid = self.grid
            else:
                self.grid.appear[:] = 0
            reset = True
        self.episode = st.get("episode")
        self.pose = tuple(st["robot_pose"])
        self.grid.integrate(self.pose, st["lidar_ranges"])
        return reset

    def on_camera(self, meta: dict, rgb: np.ndarray, depth: np.ndarray, force: bool = False):
        if self.use_depth:
            self.grid.integrate_depth(tuple(meta.get("robot_pose", self.pose)), depth)
        if self.omem is None or (not force and time.time() - self.last_det < self.det_period):
            return None
        self.last_det = time.time()
        dets = self.odet.detect(rgb, depth, tuple(meta.get("robot_pose", self.pose)))
        self.omem.update(dets)
        return dets

    def save(self, path: Path, meta: Optional[dict] = None) -> None:
        objs = self.omem.confirmed() if self.omem is not None else []
        rooms = self.omem.name_rooms() if self.omem is not None else []
        self.grid.save(path, objs, rooms, meta)


def decode_camera(parts):
    meta = json.loads(parts[1])
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(parts[2], np.uint8), 1), cv2.COLOR_BGR2RGB)
    depth = np.frombuffer(parts[3], np.float16).astype(np.float32).reshape(rgb.shape[:2])
    return meta, rgb, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="tcp://127.0.0.1:5555")
    ap.add_argument("--pub", default="tcp://127.0.0.1:5559")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--objects", action="store_true", help="object memory from the head camera (YOLO)")
    ap.add_argument("--load", default=None, help="start from a saved map (.npz); it becomes the change reference")
    ap.add_argument("--save", default=None, help="write the map PNG on exit")
    ap.add_argument("--save-map", default=None, help="write the map .npz (log-odds + objects + rooms) on exit")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N s (testing)")
    ap.add_argument("--stop-file", default=None, help="exit (and save) when this file appears (testing)")
    args = ap.parse_args()
    t_end = time.time() + args.duration if args.duration else None

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.connect(args.sim)
    sub.setsockopt_string(zmq.SUBSCRIBE, "state")
    pub = None
    if not args.no_publish:
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(args.pub)

    grid = None
    loaded_objs: list = []
    if args.load and Path(args.load).exists():
        grid, loaded_objs, _rooms, _meta = OccupancyGrid.load(Path(args.load))
        print(f"[robot_map] loaded {args.load} ({grid.known_frac():.1%} known, {len(loaded_objs)} objects)")
    mb = MapBuilder(objects=args.objects, grid=grid)
    if mb.omem is not None and loaded_objs:
        mb.omem.load(loaded_objs)
    sub.setsockopt_string(zmq.SUBSCRIBE, "camera")      # depth obstacles (+ objects with --objects)
    last_pub = 0.0
    try:
        while t_end is None or time.time() < t_end:
            if args.stop_file and Path(args.stop_file).exists():
                break
            if sub.poll(200):
                parts = sub.recv_multipart()
                if parts[0] == b"camera":
                    mb.on_camera(*decode_camera(parts))
                    continue
                mb.on_state(json.loads(parts[1] if len(parts) > 1 else parts[0].split(b" ", 1)[1]))
            if pub is not None and time.time() - last_pub >= 1.0:
                last_pub = time.time()
                ok, png = cv2.imencode(".png", mb.grid.to_image())
                if ok:
                    hdr = mb.grid.header(mb.pose)
                    hdr["changes"] = mb.grid.current_changes()
                    if mb.omem is not None:
                        hdr["objects"], hdr["rooms"] = mb.omem.confirmed(), mb.omem.name_rooms()
                    pub.send_multipart([b"robot_map", json.dumps(hdr).encode(), png.tobytes()])
    except KeyboardInterrupt:
        pass
    finally:
        if mb.omem is not None:
            print("[robot_map] objects:", json.dumps(mb.omem.confirmed()))
            print("[robot_map] rooms:", json.dumps(mb.omem.name_rooms()))
        if args.save:
            cv2.imwrite(args.save, mb.grid.to_image())
        if args.save_map:
            mb.save(Path(args.save_map))


if __name__ == "__main__":
    main()
