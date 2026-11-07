"""
Load the nerv-world house + Unitree Go2 scene for Nudge.

Source: worlds/nerv-world/build/house-go2.xml (generated upstream by
tools/make_house.py from scenes/house/layout.py — treat it as read-only).

That XML references third-party furniture meshes (decor/assets/...) that are
never committed upstream. On a fresh clone they are missing, so this loader
edits the scene in memory with MjSpec:

  * drops mesh/texture assets whose files are missing, and every geom using them;
  * gives each affected furniture piece a simple stand-in so the camera still
    sees it and LiDAR (mj_ray) / collisions still hit it:
      - pieces that already carry an invisible collision box (fridge, stove, sink)
        get that box moved to visible group 0;
      - pieces whose collision was hull meshes only (armchair, coffee table,
        dining chairs) get a visible proxy box of the piece's nominal size.

If you later fetch the real meshes upstream (`python -m decor.fetch` and friends
inside worlds/nerv-world, see its README), nothing is missing and the scene
loads exactly as published.

Geom groups (shared contract with the upstream scene, keep it):
  0 house visible | 1 ceilings | 2/3 robot visual/collision | 4 hidden house collision | 5 reserved
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402

SCENE_XML = PROJECT_ROOT / H.WORLD_SCENE_XML

# Nominal full extents (x, y, z) of furniture whose only collision is hull meshes,
# copied from F.mesh_piece(..., size=...) in worlds/nerv-world/scenes/house/layout.py.
PROXY_SIZES: Dict[str, Tuple[float, float, float]] = {
    "coffee_table": (1.05, 1.05, 0.397),
    "armchair": (0.82, 0.987, 1.023),
    "dining_chair": (0.434, 0.576, 0.973),
}
PROXY_RGBA = {
    "coffee_table": (0.45, 0.33, 0.22, 1.0),
    "armchair": (0.78, 0.74, 0.66, 1.0),
    "dining_chair": (0.70, 0.63, 0.52, 1.0),
}

# mj_ray geom-group mask that sees the house but not the robot (groups 2/3).
HOUSE_RAY_GROUPS = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)

_DECOR_RE = re.compile(r"decor/assets/([^/]+)/")
_PIECE_RE = re.compile(r"^(?:dg_(?P<a>.+)_\d+|furn_(?P<b>.+)__h\d+)$")


def _piece_of(geom_name: str) -> str | None:
    m = _PIECE_RE.match(geom_name or "")
    return (m.group("a") or m.group("b")) if m else None


def _strip_missing_assets(spec: mujoco.MjSpec, verbose: bool) -> None:
    base = SCENE_XML.parent
    meshdir = base / spec.meshdir if spec.meshdir else base
    texdir = base / spec.texturedir if spec.texturedir else base

    missing_mesh: Dict[str, str] = {}  # mesh name -> decor asset key
    for m in spec.meshes:
        if m.file and not (meshdir / m.file).exists():
            hit = _DECOR_RE.search(m.file.replace("\\", "/"))
            missing_mesh[m.name] = hit.group(1) if hit else "unknown"
    missing_tex = {t.name for t in spec.textures if t.file and not (texdir / t.file).exists()}
    if not missing_mesh and not missing_tex:
        return

    # piece name -> (asset key, parent body, pos of first dropped geom)
    pieces: Dict[str, Tuple[str, mujoco.MjsBody, np.ndarray]] = {}
    dropped = 0
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in missing_mesh:
            piece = _piece_of(g.name)
            if piece and piece not in pieces:
                pieces[piece] = (missing_mesh[g.meshname], g.parent, np.array(g.pos, dtype=float))
            spec.delete(g)
            dropped += 1
    for m in list(spec.meshes):
        if m.name in missing_mesh:
            spec.delete(m)
    for mat in spec.materials:
        roles = list(mat.textures)
        if any(r in missing_tex for r in roles):
            mat.textures = ["" if r in missing_tex else r for r in roles]
    for t in list(spec.textures):
        if t.name in missing_tex:
            spec.delete(t)

    # Stand-ins so perception and collision still see the furniture.
    boxes = {g.name: g for g in spec.geoms if g.name.startswith("furn_") and g.type == mujoco.mjtGeom.mjGEOM_BOX}
    revealed, proxied = 0, 0
    for piece, (asset, body, pos) in pieces.items():
        box = boxes.get(f"furn_{piece}")
        if box is not None:
            box.group = 0
            revealed += 1
        elif asset in PROXY_SIZES:
            sx, sy, sz = PROXY_SIZES[asset]
            body.add_geom(
                name=f"proxy_{piece}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx / 2, sy / 2, sz / 2],
                pos=pos.tolist(),  # upstream places mesh pieces at their box center
                rgba=PROXY_RGBA.get(asset, (0.6, 0.6, 0.6, 1.0)),
                group=0,
            )
            proxied += 1
        elif verbose:
            print(f"[world] warning: no stand-in for missing furniture '{piece}' ({asset})")

    if verbose:
        print(
            f"[world] furniture meshes not fetched: dropped {len(missing_mesh)} meshes / {dropped} geoms; "
            f"{revealed} collision boxes made visible, {proxied} proxy boxes added"
        )


def load_spec(verbose: bool = True) -> mujoco.MjSpec:
    if not SCENE_XML.exists():
        raise FileNotFoundError(
            f"Missing {SCENE_XML}. Fetch the world submodule first:\n"
            "    git submodule update --init --depth 1 worlds/nerv-world"
        )
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    _strip_missing_assets(spec, verbose)
    return spec


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
    """Reset to the Go2 'home' keyframe, then place the base at (x, y, yaw) on the floor."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    adr = model.jnt_qposadr[model.body_jntadr[base]]  # Go2 freejoint
    data.qpos[adr + 0] = x
    data.qpos[adr + 1] = y
    data.qpos[adr + 2] += H.FLOOR_Z
    data.qpos[adr + 3: adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


class StandingPD:
    """
    Joint-space PD hold for the upstream Go2, whose actuators are torque <motor>s
    (explicit PD, zero model damping) — NOT position servos like the stock Menagerie
    model. Writing the 'home' joint angles into data.ctrl would apply them as torques.

    tau = kp * (q_target - q) - kd * qd, clipped to each actuator's ctrlrange.
    q_target defaults to the 'home' keyframe pose; Plan B locomotion can overwrite it.
    """

    def __init__(self, model: mujoco.MjModel, kp: float = 40.0, kd: float = 1.0):
        self.kp, self.kd = kp, kd
        jids = model.actuator_trnid[:, 0]
        self.qadr = model.jnt_qposadr[jids]
        self.vadr = model.jnt_dofadr[jids]
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_target = (model.key_qpos[key][self.qadr] if key >= 0 else np.zeros(model.nu)).copy()
        limited = model.actuator_ctrllimited.astype(bool)
        self.lo = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self.hi = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)

    def apply(self, data: mujoco.MjData) -> None:
        tau = self.kp * (self.q_target - data.qpos[self.qadr]) - self.kd * data.qvel[self.vadr]
        data.ctrl[:] = np.clip(tau, self.lo, self.hi)


def load_world(verbose: bool = True) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile the house + Go2 and place the robot at the Nudge start pose."""
    model = load_spec(verbose).compile()
    data = mujoco.MjData(model)
    set_robot_pose(model, data, *H.ROBOT_START_XY, H.ROBOT_START_YAW)
    return model, data
