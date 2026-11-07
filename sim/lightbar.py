#!/usr/bin/env python3
"""
Police-style light bar on the Go2's back (`sim/lightbar.py`): the visual alarm for deaf owners.

Model: "Police Lights" by KIDosn (https://sketchfab.com/3d-models/police-lights-54321f506475441c8f92abdaa740b2d6),
CC BY 4.0, converted by `pixi run prepare-lightbar` into assets/lightbar/*.obj. The housing keeps the
original texture; the red lens, the centre lens and the blue lens are separate meshes with their own
materials, so their colour and glow change at runtime. Each lens also has a soft glow shell (an
emissive translucent ellipsoid, invisible when unlit) and a spotlight on the bar washes the floor and
walls around the robot in the alert colour, so a lit lens reads clearly even in the wide viewer shot.

Two jobs, two parts (agreed design):
  * RED + BLUE ENDS = DANGER. Only in an emergency: the red end flashes red, then the blue end
    flashes blue, back and forth (police double strobe), and the spotlight washes the floor in
    the same alternating colour. Otherwise the ends stay dark.
  * CENTRE LENS = INFORMATION for the owner: WHICH sound Nudge heard, as a colour plus a blink
    count that matches the haptic code (see SOUND_INFO; colours avoid red/blue so they never
    read as "danger"):
        doorbell     GREEN   1 pulse / s
        knock        YELLOW  2 pulses / 1.2 s
        smoke_alarm  ORANGE  fast pulse (with the danger strobe on the ends)
        complete     WHITE   steady (task done)       idle: faint white
Today the sim only receives `alert_mode` from the brain, so the sound is inferred from it
(ALERT_MODE_SOUND). Once the planned `alert` event lands in the cmd contract (AGENTS.md 5),
pass its `sound` to `update(..., sound=...)` and new sounds only need a SOUND_INFO entry.

Usage: `add_lightbar(spec)` before compiling, then `LightBar(model).update(model, t, alert_mode)` every
step. Both are no-ops if the meshes are missing.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

LIGHTBAR_DIR = Path(__file__).resolve().parent.parent / "assets" / "lightbar"
# Mounted across the front of the back on `base`, behind the head camera (x = 0.30), ahead of the handle hinge.
MOUNT_POS = [0.15, 0.0, 0.077]
LENSES = ("red", "center", "blue")
GLOW_SCALE = 1.45  # glow shell half-size relative to the lens bounding box

IDLE = {"red": (0.30, 0.02, 0.02, 0.95), "center": (0.88, 0.90, 0.94, 0.62), "blue": (0.02, 0.04, 0.32, 0.95)}
LIT = {"red": (1.0, 0.05, 0.03, 1.0), "blue": (0.10, 0.35, 1.0, 1.0), "green": (0.15, 1.0, 0.25, 1.0),
       "yellow": (1.0, 0.92, 0.08, 1.0), "orange": (1.0, 0.45, 0.02, 1.0), "white": (1.0, 1.0, 1.0, 1.0),
       "amber": (1.0, 0.55, 0.0, 1.0)}  # amber: between knock yellow and smoke orange, never teal
# Emission per lit colour (default 1.0). Bright emission pushes every warm colour to the same
# saturated yellow on screen; amber at 0.4 renders ~(255, 160, 0) vs knock yellow (255, 255, 31).
LIT_EMISSION = {"amber": 0.4}
GLOW_ALPHA = 0.35
WASH = {"red": (1.0, 0.06, 0.03), "blue": (0.08, 0.25, 1.0)}
WASH_LEVEL = 2.5

# Centre information light: sound -> (colour, blink pattern). Blink patterns: "1", "2", "fast", "steady".
SOUND_INFO = {
    "doorbell": ("green", "1"),
    "knock": ("yellow", "2"),
    "kettle": ("amber", "2"),      # tier 2 sounds: amber, 2 blinks (same as their haptic code)
    "oven_timer": ("amber", "2"),
    "smoke_alarm": ("orange", "fast"),
    "complete": ("white", "steady"),
}
ALERT_MODE_SOUND = {"1_pulse": "doorbell", "2_pulse": "knock", "continuous": "smoke_alarm", "complete": "complete"}
DANGER_SOUNDS = {"smoke_alarm"}


def _blink(pattern: str, t: float) -> bool:
    if pattern == "1":
        return (t % 1.0) < 0.30
    if pattern == "2":
        ph = t % 1.2
        return ph < 0.18 or 0.30 <= ph < 0.48
    if pattern == "fast":
        return (t % 0.5) < 0.30
    return True


def _obj_bbox(path: Path) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[float(t) for t in ln.split()[1:4]] for ln in path.read_text().splitlines() if ln.startswith("v ")])
    return v.min(0), v.max(0)


def add_lightbar(spec: mujoco.MjSpec, parent: str = "base") -> bool:
    """Attach the light bar (visual only: group 2, no mass, no collisions) to the Go2 `base` body."""
    files = {p: LIGHTBAR_DIR / f"lightbar_{p}.obj" for p in ("housing",) + LENSES}
    tex = LIGHTBAR_DIR / "lightbar.png"
    if not tex.exists() or not all(f.exists() for f in files.values()):
        return False
    G = mujoco.mjtGeom
    spec.add_texture(name="lightbar_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=tex.as_posix())
    housing_mat = spec.add_material(name="lightbar_housing_mat", specular=0.6, shininess=0.7)
    housing_mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "lightbar_tex"
    for lens in LENSES:
        spec.add_material(name=f"lightbar_{lens}_mat", rgba=list(IDLE[lens]), specular=0.9, shininess=0.9, emission=0.0)
        spec.add_material(name=f"lightbar_{lens}_glow_mat", rgba=[1, 1, 1, 0], emission=0.0, specular=0.0)
    for part, f in files.items():
        spec.add_mesh(name=f"lightbar_{part}", file=f.as_posix(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)

    bar = spec.body(parent).add_body(name="lightbar", pos=MOUNT_POS)
    vis = dict(contype=0, conaffinity=0, mass=0, group=2)
    for part in ("housing",) + LENSES:
        bar.add_geom(name=f"lightbar_{part}", type=G.mjGEOM_MESH, meshname=f"lightbar_{part}",
                     material=f"lightbar_{part}_mat", **vis)
    for lens in LENSES:  # glow shells: translucent emissive ellipsoids around each lens
        lo, hi = _obj_bbox(files[lens])
        half = np.maximum((hi - lo) / 2 * GLOW_SCALE, 0.008)
        bar.add_geom(name=f"lightbar_{lens}_glow", type=G.mjGEOM_ELLIPSOID, pos=((lo + hi) / 2).tolist(),
                     size=half.tolist(), material=f"lightbar_{lens}_glow_mat", **vis)
    # Spotlight pointing down: washes the floor/walls around the robot in the alert colour.
    bar.add_light(name="lightbar_light", pos=[0, 0, 0.45], dir=[0, 0, -1], cutoff=85, exponent=1,
                  diffuse=[0, 0, 0], specular=[0, 0, 0], castshadow=False)
    return True


class LightBar:
    def __init__(self, model: mujoco.MjModel) -> None:
        def mat(name: str) -> int:
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        self.mat = {lens: mat(f"lightbar_{lens}_mat") for lens in LENSES}
        self.glow = {lens: mat(f"lightbar_{lens}_glow_mat") for lens in LENSES}
        self.light = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "lightbar_light")
        self.enabled = all(i >= 0 for i in list(self.mat.values()) + list(self.glow.values()))

    def _lens(self, model: mujoco.MjModel, lens: str, colour: str | None, glow_level: float = 1.0) -> None:
        """colour=None -> unlit idle look; otherwise lit in LIT[colour] with a glow shell."""
        m, g = self.mat[lens], self.glow[lens]
        if colour is None:
            model.mat_rgba[m] = IDLE[lens]
            model.mat_emission[m] = 0.12 if lens == "center" else 0.0
            model.mat_rgba[g] = (1, 1, 1, 0)
            model.mat_emission[g] = 0.0
        else:
            r, gg, b, _ = LIT[colour]
            model.mat_rgba[m] = LIT[colour]
            model.mat_emission[m] = LIT_EMISSION.get(colour, 1.0)
            model.mat_rgba[g] = (r, gg, b, GLOW_ALPHA * glow_level)
            model.mat_emission[g] = 1.0

    def _wash(self, model: mujoco.MjModel, colour: str | None, level: float = WASH_LEVEL) -> None:
        if self.light >= 0:
            model.light_diffuse[self.light] = [0, 0, 0] if colour is None else [c * level for c in WASH[colour]]

    def update(self, model: mujoco.MjModel, t: float, alert_mode: str, sound: str | None = None) -> None:
        if not self.enabled:
            return
        sound = sound or ALERT_MODE_SOUND.get(alert_mode)

        # Ends: danger only — red end then blue end, back and forth (police double strobe).
        self._lens(model, "red", None)
        self._lens(model, "blue", None)
        if sound in DANGER_SOUNDS:
            ph = t % 0.8
            red_on = ph < 0.08 or 0.14 <= ph < 0.22
            blue_on = 0.40 <= ph < 0.48 or 0.54 <= ph < 0.62
            if red_on:
                self._lens(model, "red", "red")
            if blue_on:
                self._lens(model, "blue", "blue")
            self._wash(model, "red" if red_on else "blue" if blue_on else None)
        else:
            self._wash(model, None)

        # Centre: what Nudge heard (colour + blink count), faint white when idle.
        colour, pattern = SOUND_INFO.get(sound, (None, "steady"))
        if colour is None:
            self._lens(model, "center", None)
        else:
            self._lens(model, "center", colour if _blink(pattern, t) else None, glow_level=0.7)
