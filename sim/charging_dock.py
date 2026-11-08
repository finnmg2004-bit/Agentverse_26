"""
Charging station at `house_layout.LANDMARKS["robot_dock"]` (robot_home room), added at load time with MjSpec.

It is part of the house (added in `sim.world.load_spec`), so the static nav map, the brain's planner,
LiDAR and the head camera all see it like furniture (group 0, collidable). Never written into
worlds/nerv-world/.

Layout (robot parks on the plate facing north, yaw +pi/2, rear towards the back panel):

    y -2.7   robot_dock (base centre of the parked robot)
    y -2.85  base plate (flat, 2.4 cm: below the nav map's 0.10-0.55 m band, the robot drives onto it)
    y -3.40  back panel (0.60 m tall: an obstacle in the nav map / LiDAR, 0.66 m behind the dock spot)
    y -4.43  south wall (cable from the panel to a wall socket)

Status light (a flat LED bar on the panel front): warm white = standby (dock empty), pulsing green = robot charging.
No teal/cyan anywhere and no round glowing light: the HSV owner fallback (teal jacket) and YOLO (sports ball)
fired on an earlier blue sphere (checked with the head camera, see day/reports/agent-f.md).
`python sim/charging_dock.py` checks clearances to doorways / spots and renders a picture.
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
PANEL_Y = DOCK_Y - 0.70          # back panel centre (south of the dock spot)
PANEL_HALF = (0.34, 0.04, 0.40)  # x, y, z half sizes (0.8 m tall: lower, the garden tree above it read as a "potted plant")
PLATE_Y = DOCK_Y - 0.15
PLATE_HALF = (0.38, 0.55, 0.012)
WALL_Y = H.ROOMS["robot_home"]["rect"][1] + 0.07   # inner face of the south wall (wall is 0.14 m thick)

STANDBY_RGBA = (1.0, 0.86, 0.62, 1.0)  # warm white: low saturation, never teal for the HSV owner fallback
CHARGING_RGBA = (0.10, 1.0, 0.25, 1.0)
PARKED_RADIUS_M = 0.30           # robot base within this of robot_dock = "on the dock"

# Footprints that block the robot (x0, y0, x1, y1), used for the clearance check and the nav-map stamp.
BLOCKING_RECTS = [(DOCK_X - PANEL_HALF[0], PANEL_Y - PANEL_HALF[1], DOCK_X + PANEL_HALF[0], PANEL_Y + PANEL_HALF[1])]


def stamp() -> str:
    """Changes whenever the dock's collidable geometry changes -> sim.nav_map rebuilds its cache."""
    return f"dock{DOCK_X},{DOCK_Y},{PANEL_Y},{PANEL_HALF},{PLATE_Y},{PLATE_HALF}"


def add_charging_dock(spec: mujoco.MjSpec) -> None:
    """Static body in the world: base plate + contacts, back panel + cap, status light, wall cable."""
    G = mujoco.mjtGeom.mjGEOM_BOX
    dock = spec.worldbody.add_body(name=DOCK_BODY, pos=[DOCK_X, DOCK_Y, H.FLOOR_Z])
    # No white rim on the cap: seen with the garden tree behind it, a white-rimmed box read as "potted plant".
    dark, mid, accent = [0.13, 0.14, 0.16, 1.0], [0.24, 0.26, 0.30, 1.0], [0.32, 0.34, 0.38, 1.0]
    py = PANEL_Y - DOCK_Y
    # Base plate (flat; the robot stands on it) with two copper charging contacts and a guide stripe.
    dock.add_geom(name="dock_plate", type=G, pos=[0, PLATE_Y - DOCK_Y, PLATE_HALF[2]], size=list(PLATE_HALF),
                  rgba=dark, group=0)
    for side, sx in (("l", -0.13), ("r", 0.13)):
        dock.add_geom(name=f"dock_contact_{side}", type=G, pos=[sx, -0.05, 2 * PLATE_HALF[2] + 0.003],
                      size=[0.035, 0.20, 0.003], rgba=[0.85, 0.55, 0.20, 1.0], contype=0, conaffinity=0, group=0)
    dock.add_geom(name="dock_stripe", type=G, pos=[0, PLATE_Y - DOCK_Y + PLATE_HALF[1] - 0.04, 2 * PLATE_HALF[2] + 0.002],
                  size=[PLATE_HALF[0] - 0.02, 0.02, 0.002], rgba=[0.85, 0.86, 0.88, 1.0],
                  contype=0, conaffinity=0, group=0)
    # Back panel (collides like furniture) + rounded-looking cap + front fascia.
    dock.add_geom(name="dock_panel", type=G, pos=[0, py, PANEL_HALF[2]], size=list(PANEL_HALF), rgba=mid, group=0)
    dock.add_geom(name="dock_cap", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[-PANEL_HALF[0], py, 2 * PANEL_HALF[2], PANEL_HALF[0], py, 2 * PANEL_HALF[2]],
                  size=[PANEL_HALF[1] + 0.01, 0, 0], rgba=accent, contype=0, conaffinity=0, group=0)
    dock.add_geom(name="dock_fascia", type=G, pos=[0, py + PANEL_HALF[1] + 0.004, 0.36],
                  size=[0.20, 0.004, 0.14], rgba=dark, contype=0, conaffinity=0, group=0)
    # Lightning-bolt mark on the fascia: slanted bar, short cross bar, slanted bar.
    fy = py + PANEL_HALF[1] + 0.012
    for k, (x0, z0, x1, z1) in enumerate(((0.045, 0.465, -0.018, 0.365), (-0.018, 0.365, 0.018, 0.365),
                                          (0.018, 0.365, -0.045, 0.265))):
        dock.add_geom(name=f"dock_bolt_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[x0, fy, z0, x1, fy, z1],
                      size=[0.010, 0, 0], rgba=[1.0, 0.85, 0.15, 1.0], contype=0, conaffinity=0, group=0)
    # Status light (visual only, emissive LED bar): warm-white standby / green charging (set by DockStatus).
    spec.add_material(name=STATUS_MAT, rgba=list(STANDBY_RGBA), emission=0.9, specular=0.5)
    dock.add_geom(name=STATUS_GEOM, type=G,
                  pos=[0, py + PANEL_HALF[1] + 0.008, 0.19], size=[0.07, 0.006, 0.010],
                  material=STATUS_MAT, contype=0, conaffinity=0, group=0)
    # Cable from the panel back to a wall socket (flat on the floor, below LiDAR and the nav band).
    back = py - PANEL_HALF[1]
    wall = WALL_Y + 0.02 - DOCK_Y
    if wall < back - 0.05:
        dock.add_geom(name="dock_cable", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0.18, back, 0.012, 0.18, wall, 0.012], size=[0.012, 0, 0],
                      rgba=[0.05, 0.05, 0.05, 1.0], contype=0, conaffinity=0, group=0)
        dock.add_geom(name="dock_socket", type=G, pos=[0.18, wall + 0.005, 0.30], size=[0.05, 0.005, 0.05],
                      rgba=[0.95, 0.95, 0.93, 1.0], contype=0, conaffinity=0, group=0)
        dock.add_geom(name="dock_cable_up", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0.18, wall + 0.012, 0.012, 0.18, wall + 0.012, 0.26], size=[0.010, 0, 0],
                      rgba=[0.05, 0.05, 0.05, 1.0], contype=0, conaffinity=0, group=0)


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


def _clearance_problems(min_m: float = 0.6) -> list:
    """Every doorway / spot / landmark (except the dock itself) must stay >= min_m from the panel."""
    spots = {f"doorway:{k}": v for k, v in H.DOORWAYS.items()}
    spots.update({f"scan_point:{k}": v for k, v in H.ROOM_SCAN_POINTS.items()})
    spots.update({f"owner_spot:{k}": v for k, v in H.OWNER_SPOTS.items()})
    spots.update({f"landmark:{k}": v for k, v in H.LANDMARKS.items()})
    spots["robot_start"] = H.ROBOT_START_XY
    bad = []
    for x0, y0, x1, y1 in BLOCKING_RECTS:
        for name, (sx, sy) in spots.items():
            d = math.hypot(max(x0 - sx, 0.0, sx - x1), max(y0 - sy, 0.0, sy - y1))
            if d < min_m:
                bad.append(f"dock panel is {d:.2f} m from {name}")
    return bad


if __name__ == "__main__":  # python sim/charging_dock.py [out.png]
    from sim.world import build_nudge_model, HOUSE_RAY_GROUPS
    problems = _clearance_problems()
    model, data = build_nudge_model(verbose=False)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "dock_panel")
    if gid < 0:
        problems.append("dock_panel not in the model")
    # LiDAR-height ray from the dock spot going south must hit the panel (~0.66 m away).
    hit = np.zeros(1, np.int32)
    d = mujoco.mj_ray(model, data, np.array([DOCK_X, DOCK_Y, 0.32]), np.array([0.0, -1.0, 0.0]),
                      HOUSE_RAY_GROUPS, 1, -1, hit)
    print(f"  ray south from robot_dock at 0.32 m: {d:.2f} m -> {model.geom(hit[0]).name if hit[0] >= 0 else '-'}")
    if hit[0] < 0 or not model.geom(hit[0]).name.startswith("dock_") or d > 0.75:
        problems.append("LiDAR ray does not see the dock panel")
    if len(sys.argv) > 1:
        import cv2
        r = mujoco.Renderer(model, 540, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [DOCK_X, DOCK_Y - 0.6, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 3.2, -70.0, -28.0
        opt = mujoco.MjvOption()
        opt.geomgroup[1] = 0
        r.update_scene(data, camera=cam, scene_option=opt)
        cv2.imwrite(sys.argv[1], cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        r.close()
    for p in problems:
        print("  FAIL", p)
    print("charging_dock", "OK" if not problems else "FAIL")
    sys.exit(1 if problems else 0)
