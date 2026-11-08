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

# name -> (COCO class, centre (x, y, z), half sizes (x, y, z), rgba). The first entry of each class is the
# object's anchor for objects_gt (its footprint is what the clearance check uses); PROP_PARTS add the shape.
STANDINS: Dict[str, Tuple[str, Tuple[float, float, float], Tuple[float, float, float], Tuple[float, ...]]] = {
    # guest bedroom, head against the north wall, west corner: wooden frame (anchor) + parts below
    "coco_bed":        ("bed",   (-14.05, -0.55, 0.15), (0.80, 1.00, 0.12), (0.36, 0.23, 0.13, 1.0)),
    # (the TV is no longer a stand-in: sim/tv_model.py adds it always, see HOUSE_OBJECTS "tv")
}
# Shape parts so a COCO detector recognises the props from the robot's 0.39 m head camera (a plain box
# was never detected as a bed / couch). name -> (geom type, centre, size (MuJoCo convention), rgba).
_B, _E, _C = "box", "ellipsoid", "cylinder"
_SOFA, _CUSH, _ACCENT = (0.40, 0.25, 0.15, 1.0), (0.46, 0.29, 0.17, 1.0), (0.86, 0.62, 0.25, 1.0)
_DUVET, _SHEET, _HEAD = (0.27, 0.43, 0.68, 1.0), (0.94, 0.94, 0.96, 1.0), (0.58, 0.53, 0.50, 1.0)
PROP_PARTS: Dict[str, Tuple[str, Tuple[float, float, float], Tuple[float, ...], Tuple[float, ...]]] = {
    # bed: mattress, duvet (with drapes over the sides and the foot), two pillows, upholstered headboard
    "coco_bed_mattress":   (_B, (-14.05, -0.55, 0.36), (0.77, 0.97, 0.09), _SHEET),
    "coco_bed_duvet":      (_B, (-14.05, -0.85, 0.48), (0.81, 0.72, 0.04), _DUVET),
    "coco_bed_duvet_w":    (_B, (-14.87, -0.85, 0.36), (0.02, 0.72, 0.15), _DUVET),
    "coco_bed_duvet_e":    (_B, (-13.23, -0.85, 0.36), (0.02, 0.72, 0.15), _DUVET),
    "coco_bed_duvet_foot": (_B, (-14.05, -1.58, 0.36), (0.81, 0.02, 0.15), _DUVET),
    "coco_bed_pillow_w":   (_E, (-14.45, 0.15, 0.52), (0.30, 0.17, 0.08), _SHEET),
    "coco_bed_pillow_e":   (_E, (-13.65, 0.15, 0.52), (0.30, 0.17, 0.08), _SHEET),
    "coco_bed_head":       (_B, (-14.05, 0.47, 0.75), (0.86, 0.05, 0.50), _HEAD),
    # house fridge (furn_h2_fridge, front faces -x at x = 13.84): door gap + freezer gap + two handles
    "coco_fridge_gap":     (_B, (13.835, 5.10, 1.25), (0.006, 0.43, 0.006), (0.15, 0.15, 0.16, 1.0)),
    "coco_fridge_handle_t": (_B, (13.80, 4.80, 1.55), (0.02, 0.02, 0.22), (0.75, 0.76, 0.78, 1.0)),
    "coco_fridge_handle_b": (_B, (13.80, 4.80, 0.85), (0.02, 0.02, 0.30), (0.75, 0.76, 0.78, 1.0)),
}
# Sofa, designed around (-13.6, 1.15) with the seat facing +y, then placed by _place_sofa(). It lives in the
# garden family room, back to the north, facing the room: in the living-room corner (its first spot) YOLO never
# saw it (behind the armchair, next to a wall pier; max 0.11 over 9 robot-height views); here a brown, 0.8x-height
# sofa scores up to 0.76 (>= 0.15 in 5/9 views). This also clears the corner next to the back-exit path.
SOFA_AT = (-8.0, 13.0)       # centre; seat faces -y (south, into the family room)
SOFA_Z = 0.8                 # height scale (a lower sofa reads better from the 0.39 m head camera)
_SOFA_DESIGN = {
    "coco_couch":          (_B, (-13.6, 1.15, 0.22), (1.00, 0.42, 0.12), _SOFA),
    "coco_couch_back":     (_B, (-13.6, 0.80, 0.56), (1.00, 0.12, 0.34), _SOFA),
    "coco_couch_seat_w":   (_B, (-14.15, 1.22, 0.42), (0.27, 0.35, 0.08), _CUSH),
    "coco_couch_seat_c":   (_B, (-13.60, 1.22, 0.42), (0.27, 0.35, 0.08), _CUSH),
    "coco_couch_seat_e":   (_B, (-13.05, 1.22, 0.42), (0.27, 0.35, 0.08), _CUSH),
    "coco_couch_bcush_w":  (_E, (-14.15, 0.97, 0.68), (0.27, 0.09, 0.22), _CUSH),
    "coco_couch_bcush_c":  (_E, (-13.60, 0.97, 0.68), (0.27, 0.09, 0.22), _CUSH),
    "coco_couch_bcush_e":  (_E, (-13.05, 0.97, 0.68), (0.27, 0.09, 0.22), _CUSH),
    "coco_couch_arm_w":    (_B, (-14.50, 1.15, 0.44), (0.10, 0.42, 0.20), _SOFA),
    "coco_couch_arm_e":    (_B, (-12.70, 1.15, 0.44), (0.10, 0.42, 0.20), _SOFA),
    "coco_couch_throw_w":  (_E, (-14.25, 1.05, 0.60), (0.16, 0.07, 0.15), _ACCENT),
    "coco_couch_throw_e":  (_E, (-12.95, 1.05, 0.60), (0.16, 0.07, 0.15), _ACCENT),
    "coco_couch_leg_1":    (_C, (-14.52, 0.78, 0.05), (0.03, 0.05), (0.1, 0.07, 0.05, 1.0)),
    "coco_couch_leg_2":    (_C, (-12.68, 0.78, 0.05), (0.03, 0.05), (0.1, 0.07, 0.05, 1.0)),
    "coco_couch_leg_3":    (_C, (-14.52, 1.52, 0.05), (0.03, 0.05), (0.1, 0.07, 0.05, 1.0)),
    "coco_couch_leg_4":    (_C, (-12.68, 1.52, 0.05), (0.03, 0.05), (0.1, 0.07, 0.05, 1.0)),
}


def _place_sofa():
    """Rotate the design by 180 deg about its centre (seat -> south) to SOFA_AT, scale heights by SOFA_Z."""
    out = {}
    for name, (gt, (x, y, z), size, rgba) in _SOFA_DESIGN.items():
        nx, ny = SOFA_AT[0] - (x - (-13.6)), SOFA_AT[1] - (y - 1.15)
        if gt == _C:   # legs keep their size, stand on the floor
            out[name] = (gt, (nx, ny, z), size, rgba)
        else:
            out[name] = (gt, (nx, ny, z * SOFA_Z), tuple(size[:2]) + (size[2] * SOFA_Z,), rgba)
    return out


_SOFA_PARTS = _place_sofa()
_g, _pos, _size, _rgba = _SOFA_PARTS.pop("coco_couch")
STANDINS["coco_couch"] = ("couch", _pos, _size, _rgba)
PROP_PARTS.update(_SOFA_PARTS)
_TYPES = {_B: mujoco.mjtGeom.mjGEOM_BOX, _E: mujoco.mjtGeom.mjGEOM_ELLIPSOID, _C: mujoco.mjtGeom.mjGEOM_CYLINDER}
# COCO objects the house already has (visible group-0 geoms): class -> geom name
HOUSE_OBJECTS: Dict[str, str] = {
    "dining table": "furn_h2_dining_table_top_finish",
    "refrigerator": "furn_h2_fridge",
    "tv": "tv_frame",  # sim/tv_model.py (always loaded, was the `coco_tv` stand-in)
}
CLEARANCE_M = 0.8  # min distance from any stand-in footprint to a spot / doorway


def add_standins(spec: mujoco.MjSpec) -> int:
    wb = spec.worldbody
    for name, (_cls, pos, half, rgba) in STANDINS.items():
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(half),
                    rgba=list(rgba), contype=0, conaffinity=0, group=0)
    for name, (gtype, pos, size, rgba) in PROP_PARTS.items():
        size = list(size) + [0.0] * (3 - len(size))
        wb.add_geom(name=name, type=_TYPES[gtype], pos=list(pos), size=size,
                    rgba=list(rgba), contype=0, conaffinity=0, group=0)
    return len(STANDINS) + len(PROP_PARTS)


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
    want = {"bed": "guest", "couch": "family_room", "tv": "living_room",
            "dining table": "dining", "refrigerator": "kitchen"}
    for cls, room in want.items():
        if not any(o["class"] == cls and o["room"] == room for o in objs):
            problems.append(f"{cls} not found in {room}")
    for p in problems:
        print("  FAIL", p)
    print("coco_standins", "OK" if not problems else "FAIL")
    sys.exit(1 if problems else 0)
