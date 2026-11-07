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


def check_door(model, data, name, pos, axis):
    nx, ny = (1.0, 0.0) if axis == "x" else (0.0, 1.0)
    ok = True
    for z in (0.3, 1.0):
        start = (pos[0] - 0.6 * nx, pos[1] - 0.6 * ny, z)
        d = ray(model, data, start, (nx, ny, 0))
        if d < 1.2:
            ok = False
            failures.append(f"{name} {pos}: blocked at z={z} ({d:.2f} m into the 1.2 m crossing)")
    print(f"  [{'ok ' if ok else 'BAD'}] {name:<28} {pos}")


def check_navigation(model, data) -> None:
    """Static nav map: every room / owner spot / landmark reachable from the robot start."""
    from sim.nav_map import GridPlanner, build_map
    print("\nNavigation (static map + planner)")
    nav = build_map(model, data, verbose=True)
    planner = GridPlanner(nav)
    targets = {f"room:{k}": r["center"] for k, r in H.ROOMS.items()}
    targets.update({f"owner:{k}": v for k, v in H.OWNER_SPOTS.items()})
    targets.update({f"landmark:{k}": v for k, v in H.LANDMARKS.items()})
    for name, goal in targets.items():
        wps = planner.plan(H.ROBOT_START_XY, goal)
        if wps is None:
            failures.append(f"no path from robot start to {name} {goal}")
            print(f"  [BAD] {name:<28} no path")
            continue
        end = wps[-1][:2]
        miss = math.hypot(end[0] - goal[0], end[1] - goal[1])
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([H.ROBOT_START_XY] + [w[:2] for w in wps], [w[:2] for w in wps]))
        ok = miss < 0.35
        if not ok:
            failures.append(f"path to {name} ends {miss:.2f} m from the goal")
        print(f"  [{'ok ' if ok else 'BAD'}] {name:<28} {length:5.1f} m, {len(wps)} waypoints")
    # Chair blocking door_guest_living must still leave a route living_room -> guest.
    wps = planner.plan(H.ROOMS["living_room"]["center"], H.OWNER_SPOTS["guest"], ["door_guest_living"])
    doors = [w[2] for w in (wps or []) if w[2]]
    reroute_ok = wps is not None and "door_guest_living" not in doors
    if not reroute_ok:
        failures.append("no reroute living_room -> guest with door_guest_living blocked")
    print(f"  [{'ok ' if reroute_ok else 'BAD'}] {'reroute (chair in door)':<28} via {doors}")


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
    print("\nRoom navigation points (ROOMS[...]['center'])")
    for room, r in H.ROOMS.items():
        if H.room_at(*r["center"]) != room:
            failures.append(f"room centre {room} {r['center']} is in room {H.room_at(*r['center'])}")
        check_spot(model, data, f"center@{room}", r["center"])
    print("\nChair parking spot")
    check_spot(model, data, "chair_parked", H.CHAIR_STOWED_POS[:2])

    print("\nInterior doorways")
    for name, d in H.DOORWAY_INFO.items():
        check_door(model, data, name, d["pos"], d["axis"])
    print("\nExterior doors")
    for name, e in H.EXITS.items():
        check_door(model, data, name, e["pos"], e["axis"])

    check_navigation(model, data)

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
