#!/usr/bin/env python3
"""
Milestone 1: Load Unitree Go2 from MuJoCo Menagerie standing in a box-wall room
with a doorway opening on Linux.
"""

import os
import sys
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GO2_DIR = PROJECT_ROOT / "mujoco_menagerie" / "unitree_go2"
GO2_XML_PATH = GO2_DIR / "go2.xml"


def build_room_with_go2_xml() -> str:
    """
    Reads unitree_go2/go2.xml, fixes its meshdir to an absolute path so MuJoCo
    can resolve all .obj/.stl meshes from any working directory, and adds:
      - textured floor plane + skybox + directional light
      - a 6m x 6m room made of box geoms with a 1.2m doorway in the North wall
    """
    if not GO2_XML_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {GO2_XML_PATH}. "
            "Did you clone mujoco_menagerie/unitree_go2 into the project root?"
        )

    tree = ET.parse(GO2_XML_PATH)
    root = tree.getroot()

    # 1. Point compiler meshdir to the absolute assets directory of unitree_go2
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((GO2_DIR / "assets").resolve()))

    # 2. Ensure asset section has a checker floor texture/material
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        width="512",
        height="3072",
    )
    ET.SubElement(
        asset,
        "texture",
        name="groundplane",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1="0.22 0.26 0.32",
        rgb2="0.14 0.17 0.22",
        markrgb="0.4 0.45 0.5",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane",
        texture="groundplane",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.1",
    )
    ET.SubElement(
        asset,
        "material",
        name="wall_mat",
        rgba="0.82 0.80 0.76 1.0",
    )

    # 3. Add floor, light, and room walls with a doorway into worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid go2.xml: missing <worldbody>")

    # Directional overhead light
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 4.0",
        dir="0 0 -1",
        directional="true",
        castshadow="true",
    )

    # Ground plane
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="10 10 0.1",
        material="groundplane",
    )

    # Room walls: 6m x 6m room centered at (0, 0), height 1.2m, wall thickness 0.1m
    # Note: MuJoCo box sizes are HALF-extents (hx, hy, hz).
    walls = [
        # South wall (y = -3.0)
        ("wall_south", "0 -3.0 0.6", "3.0 0.08 0.6"),
        # West wall (x = -3.0)
        ("wall_west", "-3.0 0 0.6", "0.08 3.0 0.6"),
        # East wall (x = +3.0)
        ("wall_east", "3.0 0 0.6", "0.08 3.0 0.6"),
        # North wall Left segment (x from -3.0 to -0.6 -> center -1.8, half-width 1.2)
        ("wall_north_left", "-1.8 3.0 0.6", "1.2 0.08 0.6"),
        # North wall Right segment (x from +0.6 to +3.0 -> center +1.8, half-width 1.2)
        # Leaves a 1.2m doorway gap centered at (0.0, 3.0)
        ("wall_north_right", "1.8 3.0 0.6", "1.2 0.08 0.6"),
    ]

    for name, pos, size in walls:
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            material="wall_mat",
        )

    return ET.tostring(root, encoding="unicode")


def main() -> None:
    print("[sim] Building room + Unitree Go2 XML...")
    xml_string = build_room_with_go2_xml()

    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    # Reset Go2 to its "home" standing keyframe so PD actuators hold it upright
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
        print(f"[sim] Loaded 'home' keyframe (id={home_key_id}).")
    else:
        mujoco.mj_resetData(model, data)
        print("[sim] Warning: 'home' keyframe not found; using default qpos.")

    mujoco.mj_forward(model, data)
    base_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")]
    print(f"[sim] Go2 base standing at xyz = {np.round(base_pos, 3)}")
    print("[sim] Launching MuJoCo viewer... (Press ESC or close window to exit)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Frame the camera nicely on the room and robot
        viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
        viewer.cam.distance = 5.5
        viewer.cam.elevation = -28.0
        viewer.cam.azimuth = 135.0

        dt = model.opt.timestep
        while viewer.is_running():
            step_start = time.perf_counter()

            # Step physics (with "home" keyframe PD targets active in data.ctrl)
            mujoco.mj_step(model, data)
            viewer.sync()

            # Maintain real-time simulation rate
            elapsed = time.perf_counter() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)