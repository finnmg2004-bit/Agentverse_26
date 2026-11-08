"""
COCO-class stand-ins (overnight S7, experimental, `run_sim.py --coco-objects`).

The ground floor has a visible dining table and fridge, but no bed, couch or TV the head camera
could detect with a COCO detector (Option A: the robot's own map / object memory). This adds simple
box stand-ins at load time with MjSpec (group 0, visible; never edits worlds/nerv-world/) and lists
every COCO object for the judge as `state.objects_gt` (judge only, the brain must never read it).

Placement keeps clear of owner spots, scan points, landmarks and doorways (check:
`python sim/coco_standins.py`). LiDAR sees the bed and couch like any furniture; the static nav map
is NOT rebuilt with them (run_sim.py builds/loads it from the plain house), so the brain meets them
only through its sensors - that's why this is a flag.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402

# name -> (COCO class, centre (x, y, z), half sizes (x, y, z), rgba)
STANDINS: Dict[str, Tuple[str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]] = {
    # guest bedroom, head against the north wall, west corner
    "coco_bed":        ("bed",   (-14.05, -0.55, 0.26), (0.80, 1.00, 0.26), (0.85, 0.85, 0.92, 1.0)),
    "coco_bed_head":   ("bed",   (-14.05, 0.40, 0.55),  (0.80, 0.05, 0.55), (0.45, 0.33, 0.22, 1.0)),
    # living room, south-west corner, back to the south wall. Not the north-west corner: there it sat
    # next to the owner spot (-11, 5.0) and broke the smoke evacuation (owner pushed past the couch).
    # The back exit path is at y 2.6-4.6.
    "coco_couch":      ("couch", (-13.6, 1.17, 0.22),   (1.00, 0.40, 0.22), (0.30, 0.38, 0.55, 1.0)),
    "coco_couch_back": ("couch", (-13.6, 0.75, 0.45),   (1.00, 0.08, 0.45), (0.30, 0.38, 0.55, 1.0)),
    # living room, wall-mounted on the north wall, above LiDAR height
    "coco_tv":         ("tv",    (-9.0, 6.00, 1.35),    (0.60, 0.03, 0.36), (0.05, 0.05, 0.06, 1.0)),
}
# COCO objects the house already has (visible group-0 geoms): class -> geom name
HOUSE_OBJECTS: Dict[str, str] = {
    "dining table": "furn_h2_dining_table_top_finish",
    "refrigerator": "furn_h2_fridge",
}
CLEARANCE_M = 0.8  # min distance from any stand-in footprint to a spot / doorway


def add_standins(spec: mujoco.MjSpec) -> int:
    wb = spec.worldbody
    for name, (_cls, pos, half, rgba) in STANDINS.items():
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half),
                    rgba=list(rgba), contype=0, conaffinity=0, group=0)
    return len(STANDINS)


def build_model_with_objects(verbose: bool = True, cutaway: bool = True):
    """Same as sim.world.build_nudge_model, plus the stand-ins."""
    from sim.world import add_nudge_entities, apply_cutaway, load_spec, set_robot_pose, UPPER_STOREY_GROUP
    spec = load_spec(verbose)
    add_nudge_entities(spec)
    n = add_standins(spec)
    model = spec.compile()
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(model)
    set_robot_pose(model, data, *H.ROBOT_START_XY, H.ROBOT_START_YAW)
    if cutaway:
        k = apply_cutaway(model, data)
        if verbose:
            print(f"[world] cutaway: {k} upper-storey geoms hidden (group {UPPER_STOREY_GROUP})")
    if verbose:
        print(f"[world] COCO stand-ins: {n} geoms ({', '.join(sorted({v[0] for v in STANDINS.values()}))})")
    return model, data


def objects_gt(model: mujoco.MjModel, data: mujoco.MjData) -> List[dict]:
    """Judge-only list of COCO objects in the scene: [{"class", "name", "pos", "room"}]."""
    out = []
    seen = set()
    for name, (cls, _pos, _half, _rgba) in STANDINS.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0 or cls in seen:  # one entry per object (bed + headboard = one bed)
            continue
        seen.add(cls)
        p = data.geom_xpos[gid]
        out.append({"class": cls, "name": name, "pos": [round(float(v), 2) for v in p],
                    "room": H.get_room_at(float(p[0]), float(p[1]))})
    for cls, gname in HOUSE_OBJECTS.items():
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            p = data.geom_xpos[gid]
            out.append({"class": cls, "name": gname, "pos": [round(float(v), 2) for v in p],
                        "room": H.get_room_at(float(p[0]), float(p[1]))})
    return out


def _clearance_problems() -> List[str]:
    spots = {f"owner_spot:{r}": v["owner_spot"] for r, v in H.ROOMS.items() if "owner_spot" in v}
    spots.update({f"landmark:{k}": v for k, v in H.LANDMARKS.items()})
    spots.update({f"doorway:{k}": v for k, v in H.DOORWAYS.items()})
    spots.update({f"owner_spots:{k}": v for k, v in H.OWNER_SPOTS.items()})
    spots.update({f"scan_point:{k}": v for k, v in H.ROOM_SCAN_POINTS.items()})
    spots["robot_start"] = H.ROBOT_START_XY
    spots["chair_stowed"] = H.CHAIR_STOWED_POS[:2]
    bad = []
    for name, (_cls, (cx, cy, cz), (hx, hy, hz), _rgba) in STANDINS.items():
        if cz - hz > 0.8:  # wall-mounted, nothing walks into it
            continue
        for sname, (sx, sy) in spots.items():
            dx, dy = max(abs(sx - cx) - hx, 0.0), max(abs(sy - cy) - hy, 0.0)
            if math.hypot(dx, dy) < CLEARANCE_M:
                bad.append(f"{name} is {math.hypot(dx, dy):.2f} m from {sname}")
    return bad


if __name__ == "__main__":  # python sim/coco_standins.py : placement + load check
    problems = _clearance_problems()
    model, data = build_model_with_objects(verbose=False, cutaway=False)
    mujoco.mj_forward(model, data)
    objs = objects_gt(model, data)
    for o in objs:
        print(f"  {o['class']:13s} {o['room']:12s} {o['pos']}")
    want = {"bed": "guest", "couch": "living_room", "tv": "living_room",
            "dining table": "dining", "refrigerator": "kitchen"}
    for cls, room in want.items():
        if not any(o["class"] == cls and o["room"] == room for o in objs):
            problems.append(f"{cls} not found in {room}")
    for p in problems:
        print("  FAIL", p)
    print("coco_standins", "OK" if not problems else "FAIL")
    sys.exit(1 if problems else 0)
