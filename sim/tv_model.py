"""
The living-room TV (AGENTS.md 6.2 "Is the sound real?", 12.0 P6): always loaded, added at load time
with MjSpec (never written into worlds/nerv-world/).

A 1.24 m flat screen hanging from a ceiling mount in the wide opening between the living room and the
family room (the living room's north wall is open there, so a wall mount would float), facing south
into the living room, with a soundbar under it. Its sound comes from house_layout.TV_SOURCE (0.1 m in
front of the screen). Everything is above the LiDAR / nav-map height band (lowest part: soundbar at
~0.9 m), contype 0: it never blocks a route and the static nav map does not change. Group 0, so the head
camera sees it (a dark screen in a bezel, the shape YOLO calls "tv"; same footprint as the old
`coco_tv` stand-in, which this replaces).

The screen glows while the TV plays (sim/run_sim.py `TvScreen.update`): bright, slowly changing
"programme" colours while it plays a doorbell / knock, dim blue while the background noise is the
TV (`--noise tv`), dark otherwise. The brain never reads this: it only hears the sound.
"""

from __future__ import annotations

import math
from typing import Tuple

import mujoco

from sim import house_layout as H

TV_SCREEN_GEOM = "tv_screen"
TV_FRAME_GEOM = "tv_frame"          # COCO object anchor (sim/coco_standins.py HOUSE_OBJECTS "tv")
TV_SCREEN_MAT = "tv_screen_mat"

_SX, _SY = float(H.TV_SOURCE["pos"][0]), float(H.TV_SOURCE["pos"][1]) + 0.10  # screen plane centre
_SZ = float(H.TV_SOURCE["pos"][2])
FRAME_HALF = (0.62, 0.03, 0.38)
SCREEN_HALF = (0.575, 0.004, 0.335)
OFF_RGBA = (0.015, 0.017, 0.022, 1.0)


def add_tv(spec: mujoco.MjSpec) -> int:
    """Add the TV geoms to the world body. Returns the number of geoms added."""
    G = mujoco.mjtGeom
    wb = spec.worldbody
    spec.add_material(name=TV_SCREEN_MAT, rgba=list(OFF_RGBA), specular=0.6, shininess=0.8,
                            reflectance=0.05, emission=0.0)
    black, grey = [0.04, 0.04, 0.05, 1.0], [0.18, 0.18, 0.20, 1.0]
    ceiling = float(getattr(H, "WALL_HEIGHT", 2.73))
    n = 0

    def geom(**kw):
        nonlocal n
        wb.add_geom(contype=0, conaffinity=0, group=0, **kw)
        n += 1

    geom(name=TV_FRAME_GEOM, type=G.mjGEOM_BOX, pos=[_SX, _SY, _SZ], size=list(FRAME_HALF), rgba=black)
    geom(name=TV_SCREEN_GEOM, type=G.mjGEOM_BOX, material=TV_SCREEN_MAT, rgba=list(OFF_RGBA),
         pos=[_SX, _SY - FRAME_HALF[1] - SCREEN_HALF[1], _SZ], size=list(SCREEN_HALF))
    # small power LED on the bezel (bottom right), and the ceiling mount: plate + pole + ceiling cup
    geom(name="tv_led", type=G.mjGEOM_BOX, pos=[_SX + 0.52, _SY - FRAME_HALF[1] - 0.003, _SZ - 0.355],
         size=[0.012, 0.003, 0.006], rgba=[0.8, 0.1, 0.1, 1.0])
    geom(name="tv_mount_plate", type=G.mjGEOM_BOX, pos=[_SX, _SY + FRAME_HALF[1] + 0.02, _SZ + 0.1],
         size=[0.18, 0.02, 0.14], rgba=grey)
    geom(name="tv_mount_pole", type=G.mjGEOM_CYLINDER, size=[0.025, 0, 0],
         fromto=[_SX, _SY + FRAME_HALF[1] + 0.04, _SZ + 0.2, _SX, _SY + FRAME_HALF[1] + 0.04, ceiling - 0.02],
         rgba=grey)
    geom(name="tv_mount_cup", type=G.mjGEOM_CYLINDER, pos=[_SX, _SY + FRAME_HALF[1] + 0.04, ceiling - 0.03],
         size=[0.08, 0.015, 0], rgba=grey)
    # soundbar under the screen (where the TV's sound comes from), hung from the TV
    geom(name="tv_soundbar", type=G.mjGEOM_BOX, pos=[_SX, _SY - 0.01, _SZ - FRAME_HALF[2] - 0.06],
         size=[0.48, 0.045, 0.035], rgba=[0.08, 0.08, 0.09, 1.0])
    geom(name="tv_soundbar_grille", type=G.mjGEOM_BOX, pos=[_SX, _SY - 0.057, _SZ - FRAME_HALF[2] - 0.06],
         size=[0.44, 0.002, 0.025], rgba=[0.22, 0.22, 0.25, 1.0])
    return n


# Bright "programme" colours the playing screen drifts through (a TV show, not a single flat colour)
_PROGRAMME = ((0.20, 0.45, 1.00), (1.00, 0.55, 0.20), (0.25, 0.85, 0.45), (0.80, 0.30, 0.85))


def _mix(a: Tuple[float, ...], b: Tuple[float, ...], f: float) -> Tuple[float, ...]:
    return tuple(x + (y - x) * f for x, y in zip(a, b))


class TvScreen:
    """Runtime glow of the TV screen (call `update` every viewer frame / sensor step)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TV_SCREEN_GEOM)
        self.mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, TV_SCREEN_MAT)
        self.led = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tv_led")
        self.state = "off"

    def ok(self) -> bool:
        return self.gid >= 0

    def update(self, t: float, playing: bool, ambient_tv: bool = False) -> None:
        if self.gid < 0:
            return
        if playing:
            k = (t / 1.6) % len(_PROGRAMME)
            i = int(k)
            rgb = _mix(_PROGRAMME[i], _PROGRAMME[(i + 1) % len(_PROGRAMME)], k - i)
            flicker = 0.9 + 0.1 * math.sin(t * 9.0)
            rgb = tuple(min(1.0, c * flicker) for c in rgb)
            emission, state, led = 0.45, "playing", (0.1, 0.9, 0.2, 1.0)
        elif ambient_tv:
            rgb = (0.16, 0.22, 0.36)
            emission, state, led = 0.35, "on", (0.1, 0.9, 0.2, 1.0)
        else:
            rgb, emission, state, led = OFF_RGBA[:3], 0.0, "off", (0.8, 0.1, 0.1, 1.0)
        rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
        self.model.geom_rgba[self.gid] = rgba
        if self.mid >= 0:
            self.model.mat_rgba[self.mid] = rgba
            self.model.mat_emission[self.mid] = emission
        if self.led >= 0:
            self.model.geom_rgba[self.led] = list(led)
        self.state = state
