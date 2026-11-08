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

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim import house_layout as H  # noqa: E402
from sim.guide_handle import add_guide_handle  # noqa: E402
from sim.lightbar import add_lightbar  # noqa: E402
from sim.robot_screen import add_robot_screen  # noqa: E402
from sim.charging_dock import add_charging_dock  # noqa: E402
from sim.night import add_night_lamps  # noqa: E402

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


def _remove_layout_geoms(spec: mujoco.MjSpec, verbose: bool) -> None:
    """Geoms a non-default layout (sim/layouts/<name>.py, NUDGE_HOUSE) drops / adds at load (the apt's
    front doorway + lift lobby). The house layout defines neither, so this does nothing by default."""
    names = set(getattr(H, "REMOVE_GEOMS", ()))
    if not names:
        return
    gone = [g for g in list(spec.geoms) if g.name in names]
    for g in gone:
        spec.delete(g)
    for name, half, pos, rgba in getattr(H, "ADD_GEOMS", ()):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos),
                                rgba=list(rgba), group=0)
    if verbose:
        print(f"[world] layout: removed {len(gone)} geoms ({', '.join(sorted(names))}), "
              f"added {len(getattr(H, 'ADD_GEOMS', ()))}")


def load_spec(verbose: bool = True) -> mujoco.MjSpec:
    if not SCENE_XML.exists():
        raise FileNotFoundError(
            f"Missing {SCENE_XML}. Fetch the world submodule first:\n"
            "    git submodule update --init --depth 1 worlds/nerv-world"
        )
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    _strip_missing_assets(spec, verbose)
    _remove_layout_geoms(spec, verbose)  # experimental other house (--house): no-op for the house
    add_charging_dock(spec)  # sim/charging_dock.py: furniture at robot_dock (nav map, LiDAR, camera see it)
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


# --------------------------------------------------------------------------- #
# Nudge entities (added at load time; never written into the upstream XML)
# --------------------------------------------------------------------------- #
NUDGE_BODIES = ("owner_body", "chair_body")   # dynamic obstacles: excluded from the static nav map
BEACON_PREFIX = "beacon_"
UPPER_STOREY_GROUP = 5                         # cutaway: geoms above the ground floor move here


def _xyaxes_quat(xaxis, yaxis) -> list:
    x = np.asarray(xaxis, float)
    y = np.asarray(yaxis, float)
    z = np.cross(x, y)
    mat = np.column_stack([x, y, z]).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q.tolist()


OWNER_MESH_DIR = Path(__file__).resolve().parent.parent / "assets" / "models" / "owner"
OWNER_DEFAULT_PACK = "rp_posed_00178_29_OBJ"  # same default as scripts/prepare_owner.py


def _owner_tools():
    """scripts/prepare_owner.py as a module (owner names, where the packs are, the converter)."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "scripts" / "prepare_owner.py"
    spec = importlib.util.spec_from_file_location("nudge_prepare_owner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def available_owners() -> list:
    """Owner names for --owner: downloaded packs (converted on first use) + already converted owners."""
    try:
        P = _owner_tools()
        names = set(P.find_packs())
        if P.OWNERS_DIR.is_dir():
            names |= {d.name for d in P.OWNERS_DIR.iterdir() if (d / "owner.obj").exists()}
        return sorted(names)
    except Exception:  # noqa: BLE001 - listing is informational only
        return []


def _chosen_owner_dir(default: Path) -> Path:
    """Start-up option `--owner <name>` / NUDGE_OWNER=<name>: which scanned person is the owner.

    Unset / "default" = assets/models/owner/ as before. Otherwise <models>/owners/<name>/ (short names:
    mei, dennis, posed_00178_29; see scripts/prepare_owner.py --list), converted on first use from the
    downloaded pack. <models> = NUDGE_MODELS_DIR, assets/models/, or the main checkout's (git worktree).
    Never in git (Renderpeople licence; assets/ is ignored).
    """
    name = os.environ.get("NUDGE_OWNER", "").strip()
    if not name or name.lower() == "default":
        return default
    try:
        P = _owner_tools()
        out = P.OWNERS_DIR / name
        if not (out / "owner.obj").exists():
            hit = P.resolve(name)
            if hit is not None:
                out = P.OWNERS_DIR / hit[0]
                if not (out / "owner.obj").exists() and os.environ.get("NUDGE_AUTO_OWNER", "1") != "0":
                    print(f"[world] converting owner {hit[0]!r} from {hit[1].name} (first run only)...", flush=True)
                    P.convert(hit[1], out)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - a bad pack must not stop the sim
        print(f"[world] owner {name!r} not available: {exc}")
        out = default
    if out != default and (out / "owner.obj").exists() and (out / "owner.png").exists():
        print(f"[world] owner: {out.name} ({out.as_posix()})")
        return out
    print(f"[world] owner {name!r} not found (have: {', '.join(available_owners()) or 'none'}), "
          f"using the default owner")
    return default


OWNER_MESH_DIR = _chosen_owner_dir(OWNER_MESH_DIR)


def _ensure_scanned_owner() -> None:
    """Run scripts/prepare_owner.py automatically when a downloaded Renderpeople pack
    (assets/models/rp_*_OBJ/) has no converted owner yet, or the pack is newer than the conversion.

    Nothing to do (no cost) when the owner is up to date or no pack was downloaded; a failed
    conversion logs one line and the sim keeps the primitive owner. NUDGE_AUTO_OWNER=0 turns it off.
    """
    if os.environ.get("NUDGE_AUTO_OWNER", "1") == "0" or OWNER_MESH_DIR.name != "owner":
        return  # NUDGE_OWNER=<name>: converted by _chosen_owner_dir
    models = OWNER_MESH_DIR.parent
    packs = [p for p in models.glob("rp_*_OBJ") if p.is_dir() and any(p.glob("*.obj"))]
    if not packs:
        return
    pack = next((p for p in packs if p.name == OWNER_DEFAULT_PACK), max(packs, key=lambda p: p.stat().st_mtime))
    obj, png = OWNER_MESH_DIR / "owner.obj", OWNER_MESH_DIR / "owner.png"
    newest_src = max(f.stat().st_mtime for f in pack.rglob("*") if f.is_file())
    if obj.exists() and png.exists() and min(obj.stat().st_mtime, png.stat().st_mtime) >= newest_src:
        return
    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_owner.py"
    print(f"[world] converting the scanned owner from {pack.name} (first run only, a few seconds)...", flush=True)
    r = subprocess.run([sys.executable, str(script), "--src", str(pack)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[world] owner conversion failed, using the primitive owner: {(r.stderr or r.stdout).strip()[-200:]}")


def _add_scanned_owner(spec: mujoco.MjSpec, owner) -> bool:
    """Swap the primitive owner's looks for the scanned human mesh (converted by `pixi run prepare-owner`,
    or automatically by _ensure_scanned_owner on sim start).

    The primitives become invisible proxies in group 4 (hidden house-collision group; rgba kept,
    since mj_ray skips alpha-0 geoms), so LiDAR and the nav map still see the owner's footprint.
    """
    _ensure_scanned_owner()
    obj, png = OWNER_MESH_DIR / "owner.obj", OWNER_MESH_DIR / "owner.png"
    if not (obj.exists() and png.exists()):
        return False
    spec.add_texture(name="owner_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=png.as_posix())
    mat = spec.add_material(name="owner_mat", specular=0.1, shininess=0.1)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "owner_tex"
    # Scans aren't watertight; "shell" inertia avoids volume errors (the body is mocap anyway).
    spec.add_mesh(name="owner_mesh", file=obj.as_posix(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    for g in owner.geoms:
        g.group = 4
    owner.add_geom(name="owner_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="owner_mesh",
                   material="owner_mat", contype=0, conaffinity=0, mass=0, group=0)
    return True


def add_nudge_entities(spec: mujoco.MjSpec) -> None:
    """
    Head camera, pop-up guide handle and light bar on the Go2; owner and chair mocap bodies;
    sound beacons. Same geometry as the original box-house scene.
    """
    G = mujoco.mjtGeom
    base = spec.body("base")
    # Forward-looking head camera (camera looks along -Z: x=(0,-1,0), y=(0,0,1) -> looks along +X)
    base.add_camera(name="head_cam", pos=[0.30, 0.0, 0.12], quat=_xyaxes_quat([0, -1, 0], [0, 0, 1]), fovy=60.0)
    # Pop-up guide handle (sim/guide_handle.py): retracts flush into a dock, telescopes out for the owner
    add_guide_handle(spec, "base")
    # (The old floating `alert_halo` sphere is gone: the light bar shows alerts, the grip shows the haptic code.)

    wb = spec.worldbody
    # Sound beacons (visual only)
    for name, src in H.SOUND_SOURCES.items():
        wb.add_geom(name=f"{BEACON_PREFIX}{name}", type=G.mjGEOM_SPHERE, pos=list(src["pos"]), size=[0.14, 0, 0],
                    rgba=list(src["rgba_idle"]), contype=0, conaffinity=0, group=0)
    # The living-room TV as a sound source (AGENTS.md 6.2): a beacon on the screen, lit while it plays
    wb.add_geom(name=f"{BEACON_PREFIX}tv", type=G.mjGEOM_SPHERE, pos=list(H.TV_SOURCE["pos"]), size=[0.10, 0, 0],
                rgba=[0.45, 0.3, 0.6, 0.3], contype=0, conaffinity=0, group=0)

    # Owner: humanoid mocap body (teal jacket for the HSV detector; legs/hips at LiDAR height)
    ox, oy = H.ROOMS[H.OWNER_START_ROOM]["owner_spot"]
    owner = wb.add_body(name="owner_body", mocap=True, pos=[ox, oy, 0.0])
    jacket, trousers, skin = [0.05, 0.72, 0.88, 1.0], [0.16, 0.20, 0.30, 1.0], [0.92, 0.74, 0.60, 1.0]
    owner.add_geom(name="owner_leg_left", type=G.mjGEOM_CAPSULE, fromto=[0, 0.11, 0.06, 0, 0.11, 0.72], size=[0.075, 0, 0], rgba=trousers)
    owner.add_geom(name="owner_leg_right", type=G.mjGEOM_CAPSULE, fromto=[0, -0.11, 0.06, 0, -0.11, 0.72], size=[0.075, 0, 0], rgba=trousers)
    owner.add_geom(name="owner_hips", type=G.mjGEOM_BOX, pos=[0, 0, 0.42], size=[0.12, 0.19, 0.26], rgba=[0.05, 0.68, 0.85, 1.0])
    owner.add_geom(name="owner_torso", type=G.mjGEOM_BOX, pos=[0, 0, 1.02], size=[0.14, 0.22, 0.30], rgba=jacket)
    owner.add_geom(name="owner_arm_left", type=G.mjGEOM_CAPSULE, fromto=[0, 0.26, 1.24, 0.08, 0.26, 0.76], size=[0.055, 0, 0], rgba=jacket)
    owner.add_geom(name="owner_arm_right", type=G.mjGEOM_CAPSULE, fromto=[0, -0.26, 1.24, 0.08, -0.26, 0.76], size=[0.055, 0, 0], rgba=jacket)
    owner.add_geom(name="owner_head", type=G.mjGEOM_SPHERE, pos=[0, 0, 1.46], size=[0.13, 0, 0], rgba=skin)
    _add_scanned_owner(spec, owner)
    add_lightbar(spec, "base")
    add_robot_screen(spec, "base")  # "Nudge says" flank screens (sim/robot_screen.py)
    add_night_lamps(spec)  # warm ceiling lamps, off by day (sim/night.py)

    # Avoid-area rugs (house_layout.AVOID_AREAS): flat, visual only (contype/conaffinity 0 ->
    # the static nav map, LiDAR rays at 0.32 m and physics all ignore them). A dark-red field
    # with a lighter border so it reads as a carpet in the viewer and the head camera.
    for name, (x0, y0, x1, y1) in H.AVOID_AREAS.items():
        cx_, cy_, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        wb.add_geom(name=f"rug_{name}_border", type=G.mjGEOM_BOX, pos=[cx_, cy_, 0.004],
                    size=[hx, hy, 0.004], rgba=[0.86, 0.72, 0.45, 1.0], contype=0, conaffinity=0, group=0)
        wb.add_geom(name=f"rug_{name}", type=G.mjGEOM_BOX, pos=[cx_, cy_, 0.010],
                    size=[max(hx - 0.12, 0.05), max(hy - 0.12, 0.05), 0.004],
                    rgba=[0.55, 0.10, 0.14, 1.0], contype=0, conaffinity=0, group=0)

    # Movable chair obstacle (starts stowed)
    cx, cy, cz = H.CHAIR_STOWED_POS
    chair = wb.add_body(name="chair_body", mocap=True, pos=[cx, cy, cz])
    chair.add_geom(name="chair_base", type=G.mjGEOM_CYLINDER, pos=[0, 0, 0.26], size=[0.22, 0.24, 0], rgba=[0.72, 0.42, 0.16, 1.0])
    chair.add_geom(name="chair_back", type=G.mjGEOM_BOX, pos=[-0.16, 0, 0.62], size=[0.04, 0.20, 0.22], rgba=[0.65, 0.36, 0.12, 1.0])


def geom_world_zrange(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """(ngeom, 2) world-frame z-min / z-max of every geom's bounding box."""
    c, h = model.geom_aabb[:, :3], model.geom_aabb[:, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    corners = c[:, None, :] + signs[None, :, :] * h[:, None, :]                      # (n, 8, 3) local
    rot = data.geom_xmat.reshape(-1, 3, 3)
    world = np.einsum("nij,nkj->nki", rot, corners) + data.geom_xpos[:, None, :]       # (n, 8, 3)
    return np.stack([world[:, :, 2].min(1), world[:, :, 2].max(1)], axis=1)


def apply_cutaway(model: mujoco.MjModel, data: mujoco.MjData, z_cut: float = H.WALL_HEIGHT - 0.1) -> int:
    """Move house geoms that sit entirely above the ground-floor rooms (upper storeys, roof)
    to group 5, which MuJoCo does not render by default -> dollhouse view of the ground floor.
    Ceilings stay in group 1 (hide them with the viewer's group-1 toggle)."""
    mujoco.mj_forward(model, data)
    zr = geom_world_zrange(model, data)
    # Never touch ceilings (group 1): the head camera must keep seeing them, otherwise it looks
    # straight into the (blue, teal-ish) sky. The viewer hides group 1 via its own option.
    mask = (zr[:, 0] > z_cut) & np.isin(model.geom_group, [0, 4])
    model.geom_group[mask] = UPPER_STOREY_GROUP
    return int(mask.sum())


def build_nudge_model(verbose: bool = True, cutaway: bool = True) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """Full Nudge scene: house + Go2 + head_cam/handle/light bar + owner + chair + beacons."""
    spec = load_spec(verbose)
    add_nudge_entities(spec)
    model = spec.compile()
    # Plan B is kinematic (qpos + mj_forward): skip contact detection over ~3.3k geoms.
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(model)
    set_robot_pose(model, data, *H.ROBOT_START_XY, H.ROBOT_START_YAW)
    if cutaway:
        n = apply_cutaway(model, data)
        if verbose:
            print(f"[world] cutaway: {n} upper-storey geoms hidden (group {UPPER_STOREY_GROUP})")
    return model, data
