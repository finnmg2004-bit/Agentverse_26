#!/usr/bin/env python3
"""
Headless sanity check: every coordinate in sim/house_layout.py must agree with the
compiled nerv-world house. Uses mj_ray against house geoms only (robot excluded).

    pixi run check-world

Checks:
  * spots (owner spots, scan points, landmarks, robot start, exits outside):
    floor under them, and >= CLEARANCE m of free space around them;
  * doorways / exterior doors: a ray straight through the doorway center is unobstructed;
  * room_at() of every owner spot / scan point returns the expected room;
  * the robot start pose compiles and the Go2 is in the expected room.
Exit code 1 if anything fails.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402
from sim.world import HOUSE_RAY_GROUPS, load_world  # noqa: E402

CLEARANCE = 0.35        # free radius around a standing spot (m)
PROBE_HEIGHTS = (0.15, 0.30, 0.90)

failures: list[str] = []


def ray(model, data, start, direction) -> float:
    gid = np.zeros(1, dtype=np.int32)
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    dist = mujoco.mj_ray(model, data, np.asarray(start, float), d, HOUSE_RAY_GROUPS, 1, -1, gid)
    return dist if dist >= 0 else math.inf


def check_spot(model, data, name, xy, need_floor=True):
    x, y = xy
    if need_floor:
        down = ray(model, data, (x, y, 1.2), (0, 0, -1))
        floor_z = 1.2 - down
        if not math.isfinite(down) or abs(floor_z - H.FLOOR_Z) > 0.12:
            failures.append(f"{name} {xy}: no floor at z~{H.FLOOR_Z} (hit z={floor_z:.2f})")
            return
    nearest = math.inf
    for z in PROBE_HEIGHTS:
        for k in range(16):
            a = 2 * math.pi * k / 16
            nearest = min(nearest, ray(model, data, (x, y, z), (math.cos(a), math.sin(a), 0)))
    status = "ok " if nearest >= CLEARANCE else "BAD"
    if nearest < CLEARANCE:
        failures.append(f"{name} {xy}: only {nearest:.2f} m clearance")
    print(f"  [{status}] {name:<28} {str(xy):<16} clearance {min(nearest, 9.99):.2f} m")


def wall_axis(pos):
    """Return the unit normal of the wall a doorway sits in (axis-aligned)."""
    x, y = pos
    for r in H.ROOMS.values():
        x0, y0, x1, y1 = r["rect"]
        if abs(y - y0) < 1e-6 or abs(y - y1) < 1e-6:
            if x0 - 1e-6 <= x <= x1 + 1e-6:
                return (0.0, 1.0)
        if abs(x - x0) < 1e-6 or abs(x - x1) < 1e-6:
            if y0 - 1e-6 <= y <= y1 + 1e-6:
                return (1.0, 0.0)
    raise ValueError(f"doorway {pos} is not on any room edge")


def check_door(model, data, name, pos):
    nx, ny = wall_axis(pos)
    ok = True
    for z in (0.3, 1.0):
        start = (pos[0] - 0.6 * nx, pos[1] - 0.6 * ny, z)
        d = ray(model, data, start, (nx, ny, 0))
        if d < 1.2:
            ok = False
            failures.append(f"{name} {pos}: blocked at z={z} ({d:.2f} m into the 1.2 m crossing)")
    print(f"  [{'ok ' if ok else 'BAD'}] {name:<28} {pos}")


def main() -> int:
    model, data = load_world(verbose=True)
    print(f"[check] scene compiled: ngeom={model.ngeom} nbody={model.nbody} nq={model.nq} nu={model.nu}")

    print("\nOwner spots")
    for room, xy in H.OWNER_SPOTS.items():
        if H.room_at(*xy) != room:
            failures.append(f"owner spot {room} {xy} is in room {H.room_at(*xy)}")
        check_spot(model, data, f"owner@{room}", xy)
    print("\nRoom scan points")
    for room, xy in H.ROOM_SCAN_POINTS.items():
        if H.room_at(*xy) != room:
            failures.append(f"scan point {room} {xy} is in room {H.room_at(*xy)}")
        check_spot(model, data, f"scan@{room}", xy)
    print("\nLandmarks")
    for name, xy in H.LANDMARKS.items():
        check_spot(model, data, name, xy)
    print("\nChair parking spot")
    check_spot(model, data, "chair_parked", H.CHAIR_PARKED_POS)

    print("\nInterior doorways")
    for name, d in H.DOORWAYS.items():
        check_door(model, data, name, d["pos"])
    print("\nExterior doors")
    for name, e in H.EXITS.items():
        check_door(model, data, name, e["pos"])

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    bx, by = data.xpos[base][:2]
    room = H.room_at(bx, by)
    print(f"\nRobot start: base at ({bx:.2f}, {by:.2f}) in '{room}'")
    if room != H.ROBOT_START_ROOM:
        failures.append(f"robot starts in {room}, expected {H.ROBOT_START_ROOM}")
    check_spot(model, data, "robot_start", H.ROBOT_START_XY)

    print()
    if failures:
        print(f"[check] FAILED ({len(failures)}):")
        for f in failures:
            print("   -", f)
        return 1
    print("[check] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
