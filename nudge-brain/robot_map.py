#!/usr/bin/env python3
"""Option A experiment (E3.1): the robot's own occupancy map, in its own process.

Subscribes to the sim's `state` (:5555: robot_pose + lidar_ranges), integrates the LiDAR into a
5 cm log-odds grid, and publishes topic `robot_map` on :5559 at 1 Hz:
    [b"robot_map", header_json, png_bytes]
header: {"t", "res", "origin": [x0, y0], "shape": [h, w], "known_frac", "pose", "encoding": "png-u8"}
PNG pixel values: 127 = unknown, 0 = free, 255 = occupied (row 0 = y0, i.e. not flipped).
Ground truth (house_layout, owner_pos_gt) is never read. Pose comes from the sim (a real robot
would use SLAM). Add-only: nothing else depends on this process.

    pixi run python nudge-brain/robot_map.py [--no-publish] [--save map.png]
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np
import zmq
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "nudge-brain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RES = 0.05
ORIGIN = (-17.0, -7.0)          # covers the house footprint plus the outside exits
SIZE_M = (34.0, 26.0)
MAX_RANGE = 4.0
FOV = np.pi                      # [-90, +90] deg, matches sim/sensors.py
L_OCC, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -4.0, 4.0
START_OFFSET = 0.27              # rays start at the robot's perimeter in the sim


class OccupancyGrid:
    def __init__(self, res: float = RES, origin=ORIGIN, size_m=SIZE_M) -> None:
        self.res = res
        self.origin = origin
        self.w = int(round(size_m[0] / res))
        self.h = int(round(size_m[1] / res))
        self.logodds = np.zeros((self.h, self.w), dtype=np.float32)
        self.changes: list[dict] = []     # E3.5: "something new appeared in known free space"
        self._t0 = time.time()

    def _cell(self, x: float, y: float):
        return int(round((x - self.origin[0]) / self.res)), int(round((y - self.origin[1]) / self.res))

    def integrate(self, pose, ranges) -> None:
        rx, ry, yaw = pose
        n = len(ranges)
        if n == 0:
            return
        angles = yaw + np.linspace(-FOV / 2, FOV / 2, n)
        free = np.zeros_like(self.logodds, dtype=np.uint8)
        occ = []
        c0 = self._cell(rx, ry)
        for a, r in zip(angles, ranges):
            hit = r < MAX_RANGE - 1e-3
            rr = float(r) if hit else MAX_RANGE
            ex, ey = rx + rr * np.cos(a), ry + rr * np.sin(a)
            c1 = self._cell(ex, ey)
            cv2.line(free, c0, c1, 1, 1)
            if hit:
                occ.append(c1)
        mask = free.astype(bool)
        # drop the end cell of hits from the free update (it is applied as occupied below)
        for cx, cy in occ:
            if 0 <= cx < self.w and 0 <= cy < self.h:
                mask[cy, cx] = False
        self.logodds[mask] += L_FREE
        new_cells = []
        for cx, cy in occ:
            if 0 <= cx < self.w and 0 <= cy < self.h:
                if self._was_free(cx, cy):
                    new_cells.append((cx, cy))
                self.logodds[cy, cx] += L_OCC
        if new_cells:
            self._record_change(new_cells)
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)

    def _was_free(self, cx: int, cy: int) -> bool:
        """True if the cell was confidently free and nothing occupied was next to it before this hit
        (so wall-edge pose jitter is not reported as a change)."""
        y0, y1, x0, x1 = max(cy - 2, 0), cy + 3, max(cx - 2, 0), cx + 3
        return bool(self.logodds[cy, cx] < -1.2 and np.all(self.logodds[y0:y1, x0:x1] < 0.3))

    def _record_change(self, cells) -> None:
        xs = [self.origin[0] + cx * self.res for cx, _ in cells]
        ys = [self.origin[1] + cy * self.res for _, cy in cells]
        x, y = float(np.mean(xs)), float(np.mean(ys))
        t = time.time() - self._t0
        for c in self.changes:                      # merge with a recent change nearby
            if np.hypot(c["x"] - x, c["y"] - y) < 0.8 and t - c["t"] < 20.0:
                c["cells"] += len(cells)
                c["t"] = round(t, 1)
                return
        self.changes.append({"kind": "appeared", "x": round(x, 2), "y": round(y, 2),
                             "cells": len(cells), "t": round(t, 1)})
        print(f"[robot_map] CHANGE: new obstacle near ({x:.1f}, {y:.1f}) at t={t:.0f}s", flush=True)
        del self.changes[:-20]

    def confirmed_changes(self, min_cells: int = 3) -> list[dict]:
        return [c for c in self.changes if c["cells"] >= min_cells]

    def to_image(self) -> np.ndarray:
        img = np.full(self.logodds.shape, 127, dtype=np.uint8)
        img[self.logodds < -0.3] = 0
        img[self.logodds > 0.5] = 255
        return img

    def known_frac(self) -> float:
        return float(np.mean(np.abs(self.logodds) > 0.3))

    def header(self, pose) -> dict:
        return {"t": time.time(), "res": self.res, "origin": list(self.origin),
                "shape": [self.h, self.w], "known_frac": round(self.known_frac(), 4),
                "pose": [round(float(v), 3) for v in pose], "encoding": "png-u8"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="tcp://127.0.0.1:5555")
    ap.add_argument("--pub", default="tcp://127.0.0.1:5559")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--objects", action="store_true", help="E3.2: object memory from the head camera (YOLO)")
    ap.add_argument("--save", default=None, help="write the map PNG on exit")
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

    omem = odet = None
    if args.objects:
        from object_memory import ObjectDetector, ObjectMemory
        sub.setsockopt_string(zmq.SUBSCRIBE, "camera")
        odet, omem = ObjectDetector(), ObjectMemory()
    last_det = 0.0
    grid = OccupancyGrid()
    episode, last_pub, pose = None, 0.0, (0.0, 0.0, 0.0)
    try:
        while t_end is None or time.time() < t_end:
            if args.stop_file and Path(args.stop_file).exists():
                break
            if sub.poll(200):
                parts = sub.recv_multipart()
                if parts[0] == b"camera":
                    if omem is not None and time.time() - last_det >= 0.5:   # ~2 Hz is plenty
                        last_det = time.time()
                        meta = json.loads(parts[1])
                        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(parts[2], np.uint8), 1), cv2.COLOR_BGR2RGB)
                        depth = np.frombuffer(parts[3], np.float16).astype(np.float32).reshape(rgb.shape[:2])
                        omem.update(odet.detect(rgb, depth, tuple(meta.get("robot_pose", pose))))
                    continue
                st = json.loads(parts[1] if len(parts) > 1 else parts[0].split(b" ", 1)[1])
                if episode is not None and st.get("episode") != episode:
                    grid = OccupancyGrid()           # R reset: start a fresh map
                    if omem is not None:
                        omem.objects.clear()
                episode = st.get("episode")
                pose = tuple(st["robot_pose"])
                grid.integrate(pose, st["lidar_ranges"])
            if pub is not None and time.time() - last_pub >= 1.0:
                last_pub = time.time()
                ok, png = cv2.imencode(".png", grid.to_image())
                if ok:
                    hdr = grid.header(pose)
                    hdr["changes"] = grid.confirmed_changes()
                    if omem is not None:
                        hdr["objects"], hdr["rooms"] = omem.snapshot(), omem.name_rooms()
                    pub.send_multipart([b"robot_map", json.dumps(hdr).encode(), png.tobytes()])
    except KeyboardInterrupt:
        pass
    finally:
        if omem is not None:
            print("[robot_map] objects:", json.dumps(omem.snapshot()))
            print("[robot_map] rooms:", json.dumps(omem.name_rooms()))
        if args.save:
            cv2.imwrite(args.save, grid.to_image())


if __name__ == "__main__":
    main()
