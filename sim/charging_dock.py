"""
Charging station in the ENTRY HALL: the robot starts the demo parked on it (team feedback, AGENTS.md 12.0 P10).

`house_layout.LANDMARKS["robot_dock"]` = `ROBOT_START_XY` (-3.7, -3.3, beside the front door, west side); the robot starts there facing north.
Added at load time with MjSpec (`sim.world.load_spec`, group 0), so the camera and the nav-map builder see it
like the rest of the house. Never written into worlds/nerv-world/.

It is a low charging PAD, deliberately with no upright panel:
    pad        centred on robot_dock, 0.76 x 1.10 m, 2.4 cm high, copper contacts, a light front stripe and
               a lightning bolt printed on it
    rear lip   0.08 m high along the pad's south edge, carrying the status LED bar
Everything stays below the nav map's 0.10-0.55 m obstacle band and the 0.32 m LiDAR plane, so the robot and
the owner walk over it and nothing blocks the front-door route (straight south of the start), the
guest/entry door, the entry/dining archway or the owner's path in the doorbell scenario.
Why no panel: in the entry, every upright version (dark or light, 0.54-0.8 m) raised YOLO scores next to the
open front door ("tv" 0.36-0.48 > YOLO_CONF 0.35, "person" up to 0.35); in robot_home a blue glowing sphere
was taken for the owner by the HSV fallback and a white-rimmed panel for a "potted plant" (agent-f report).

Status light: warm white = standby (dock empty), pulsing green = robot charging (base within 0.30 m of
robot_dock, so it is green at the start of every run). No teal/cyan anywhere (HSV owner fallback).
`python sim/charging_dock.py [out.png]` checks it and renders a picture.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402

DOCK_X, DOCK_Y = H.LANDMARKS["robot_dock"]
DOCK_BODY = "charging_dock"
STATUS_GEOM = "dock_status_light"
STATUS_MAT = "dock_status_mat"
PLATE_HALF = (0.38, 0.55, 0.012)
LIP_HALF = (0.38, 0.06, 0.04)    # rear lip: 0.08 m high (< nav band 0.10 m)

STANDBY_RGBA = (1.0, 0.86, 0.62, 1.0)  # warm white: low saturation, never teal for the HSV owner fallback
CHARGING_RGBA = (0.10, 1.0, 0.25, 1.0)
PARKED_RADIUS_M = 0.30           # robot base within this of robot_dock = "on the dock"


def stamp() -> str:
    """Changes whenever the dock's geometry changes -> sim.nav_map rebuilds its cache."""
    return f"dock3:{DOCK_X},{DOCK_Y},{PLATE_HALF},{LIP_HALF}"


def add_charging_dock(spec: mujoco.MjSpec) -> None:
    """Static body in the world: charging pad + contacts + bolt + stripe, rear lip with the status LED."""
    G = mujoco.mjtGeom.mjGEOM_BOX
    C = mujoco.mjtGeom.mjGEOM_CAPSULE
    dock = spec.worldbody.add_body(name=DOCK_BODY, pos=[DOCK_X, DOCK_Y, H.FLOOR_Z])
    dark, mid = [0.13, 0.14, 0.16, 1.0], [0.30, 0.32, 0.36, 1.0]
    top = 2 * PLATE_HALF[2]
    vis = dict(contype=0, conaffinity=0, group=0)
    # Pad (the robot stands on it, facing north): copper contacts under the feet, light stripe at the front edge.
    dock.add_geom(name="dock_plate", type=G, pos=[0, 0, PLATE_HALF[2]], size=list(PLATE_HALF), rgba=dark, group=0)
    for side, sx in (("l", -0.13), ("r", 0.13)):
        dock.add_geom(name=f"dock_contact_{side}", type=G, pos=[sx, -0.05, top + 0.003],
                      size=[0.035, 0.20, 0.003], rgba=[0.85, 0.55, 0.20, 1.0], **vis)
    dock.add_geom(name="dock_stripe", type=G, pos=[0, PLATE_HALF[1] - 0.04, top + 0.002],
                  size=[PLATE_HALF[0] - 0.02, 0.02, 0.002], rgba=[0.85, 0.86, 0.88, 1.0], **vis)
    # Lightning bolt printed on the pad in front of the robot (seen from the top / follow cameras).
    by, bz = PLATE_HALF[1] - 0.20, top + 0.004
    for k, (x0, y0, x1, y1) in enumerate(((0.06, by + 0.10, -0.02, by), (-0.02, by, 0.02, by),
                                          (0.02, by, -0.06, by - 0.10))):
        dock.add_geom(name=f"dock_bolt_{k}", type=C, fromto=[x0, y0, bz, x1, y1, bz], size=[0.012, 0, 0],
                      rgba=[1.0, 0.85, 0.15, 1.0], **vis)
    # Rear lip (south edge) with the status LED bar on its north face (visual only, emissive).
    ly = -PLATE_HALF[1] + LIP_HALF[1]
    dock.add_geom(name="dock_lip", type=G, pos=[0, ly, LIP_HALF[2]], size=list(LIP_HALF), rgba=mid, group=0)
    spec.add_material(name=STATUS_MAT, rgba=list(STANDBY_RGBA), emission=0.9, specular=0.5)
    dock.add_geom(name=STATUS_GEOM, type=G, pos=[0, ly + LIP_HALF[1] + 0.004, LIP_HALF[2] + 0.01],
                  size=[0.12, 0.004, 0.012], material=STATUS_MAT, **vis)


class DockStatus:
    """Colours the dock's status light from the robot pose (cheap: one material write at 10 Hz)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, STATUS_MAT)
        self.last_t = -1.0
        self.charging = False

    def update(self, model: mujoco.MjModel, sim_t: float, robot_xy: Tuple[float, float]) -> Optional[bool]:
        if self.mid < 0 or 0.0 <= sim_t - self.last_t < 0.1:
            return None
        self.last_t = sim_t
        self.charging = math.hypot(robot_xy[0] - DOCK_X, robot_xy[1] - DOCK_Y) <= PARKED_RADIUS_M
        if self.charging:
            k = 0.55 + 0.45 * math.sin(2 * math.pi * 0.5 * sim_t)   # slow "breathing" pulse
            model.mat_rgba[self.mid] = [CHARGING_RGBA[0] * k, CHARGING_RGBA[1] * k, CHARGING_RGBA[2] * k, 1.0]
        else:
            model.mat_rgba[self.mid] = STANDBY_RGBA
        return self.charging


if __name__ == "__main__":  # python sim/charging_dock.py [out.png]
    from sim.nav_map import NavMap, build_occupancy
    from sim.world import HOUSE_RAY_GROUPS, build_nudge_model
    problems = []
    if tuple(H.LANDMARKS["robot_dock"]) != tuple(H.ROBOT_START_XY):
        problems.append(f"robot_dock {H.LANDMARKS['robot_dock']} != ROBOT_START_XY {H.ROBOT_START_XY}")
    model, data = build_nudge_model(verbose=False)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dock_plate") < 0:
        problems.append("dock_plate not in the model")
    # Nothing of the dock may be an obstacle in the static nav map: check the pad footprint (0.76 x 1.10 m)
    # plus ~0.1 m (a larger box would catch the south wall's margin now that the dock stands beside the door).
    nav = NavMap(build_occupancy(model, data))
    for dx in np.arange(-0.5, 0.51, 0.1):
        for dy in np.arange(-0.6, 0.61, 0.1):
            if nav.clearance(DOCK_X + dx, DOCK_Y + dy) < 0.3:
                problems.append(f"nav map obstacle near the dock at ({DOCK_X + dx:.1f}, {DOCK_Y + dy:.1f})")
    # The LiDAR plane (0.32 m) must not hit any dock geom.
    hit = np.zeros(1, np.int32)
    for k in range(16):
        a = 2 * math.pi * k / 16
        mujoco.mj_ray(model, data, np.array([DOCK_X, DOCK_Y, 0.32]), np.array([math.cos(a), math.sin(a), 0.0]),
                      HOUSE_RAY_GROUPS, 1, -1, hit)
        if hit[0] >= 0 and model.geom(hit[0]).name.startswith("dock"):
            problems.append(f"LiDAR ray at {math.degrees(a):.0f} deg hits {model.geom(hit[0]).name}")
    st = DockStatus(model)
    if st.update(model, 1.0, H.ROBOT_START_XY) is not True:
        problems.append("robot at the start pose is not on the dock")
    print(f"  dock at {H.LANDMARKS['robot_dock']} (= robot start), robot on dock: {st.charging}")
    if len(sys.argv) > 1:
        import cv2
        r = mujoco.Renderer(model, 540, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [DOCK_X, DOCK_Y, 0.2]
        cam.distance, cam.azimuth, cam.elevation = 3.0, -60.0, -40.0
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
        r.update_scene(data, camera=cam, scene_option=opt)
        cv2.imwrite(sys.argv[1], cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        r.close()
    for p in problems[:10]:
        print("  FAIL", p)
    print("charging_dock", "OK" if not problems else "FAIL")
    sys.exit(1 if problems else 0)
