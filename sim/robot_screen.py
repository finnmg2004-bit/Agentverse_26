#!/usr/bin/env python3
"""
On-robot "Nudge says" screen (`sim/robot_screen.py`) - robot outputs, AGENTS.md 9.1 (Sean's lane).

What the owner (or a carer / the judges) sees on the robot:
  * Two flank panels on the Go2 trunk (left + right, visual only). Their colour follows the alert
    tier, re-using the light bar's palette (sim/lightbar.py) so screen and LEDs always agree:
        idle            dark screen
        doorbell        green   (info)       knock   yellow (info)
        smoke_alarm     red <-> orange flashing (danger)
        mission complete white
    No teal/cyan anywhere: the HSV owner detector looks for the owner's teal jacket.
  * The message itself: the brain already publishes what Nudge tells the owner as `mission.say`
    in its `brain` status topic (Gemma's message, else agent.DEFAULT_SAY per sound). The sim
    subscribes to that topic (no brain change needed) and shows it
      - as a floating label above the robot in the MuJoCo viewer (follows the robot), and
      - in the viewer's bottom-left text overlay ("Nudge says: ...").
    Every time the text changes the sim prints `[sim] NUDGE SAYS: ...` (scenario tests check it)
    and hands it to sim/robot_voice.py for speech.

Usage: `add_robot_screen(spec)` before compiling; then `RobotScreen(model)` and every step
`screen.update(model, t, tier_sound, mission_status)`; in the viewer loop
`screen.draw_label(viewer.user_scn, robot_pose, text)`.
"""
from __future__ import annotations

import mujoco
import numpy as np

from sim.lightbar import LIT

SCREEN_SIDES = {"left": 1.0, "right": -1.0}
# Base-frame placement: vertical panels on the trunk flanks, mid-body, just outside the shell.
PANEL_POS_X, PANEL_POS_Y, PANEL_POS_Z = -0.02, 0.098, 0.005
PANEL_HALF = (0.075, 0.003, 0.028)     # long, thin, low: reads as a side display
BEZEL_PAD = 0.008

IDLE_RGBA = (0.06, 0.06, 0.08, 1.0)
BEZEL_RGBA = (0.02, 0.02, 0.02, 1.0)
TIER_COLOUR = {"doorbell": "green", "knock": "yellow", "complete": "white"}  # smoke_alarm: flashes
LABEL_HEIGHT = 0.85                    # metres above the robot base for the floating viewer label
LABEL_MAX = 99                         # mjvGeom.label is a 100-char C string


def add_robot_screen(spec: mujoco.MjSpec, parent: str = "base") -> None:
    """Attach the two flank screens (visual only: group 2, no mass, no collisions)."""
    G = mujoco.mjtGeom
    body = spec.body(parent)
    vis = dict(contype=0, conaffinity=0, mass=0, group=2)
    hx, hy, hz = PANEL_HALF
    for side, s in SCREEN_SIDES.items():
        y = s * PANEL_POS_Y
        body.add_geom(name=f"screen_bezel_{side}", type=G.mjGEOM_BOX, pos=[PANEL_POS_X, y, PANEL_POS_Z],
                      size=[hx + BEZEL_PAD, hy, hz + BEZEL_PAD], rgba=list(BEZEL_RGBA), **vis)
        spec.add_material(name=f"screen_{side}_mat", rgba=list(IDLE_RGBA), emission=0.0, specular=0.3)
        body.add_geom(name=f"screen_{side}", type=G.mjGEOM_BOX, pos=[PANEL_POS_X, y + s * 0.002, PANEL_POS_Z],
                      size=[hx, hy, hz], material=f"screen_{side}_mat", **vis)


class RobotScreen:
    """Drives the flank panel colour (alert tier) and the floating 'Nudge says' label."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.mats = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, f"screen_{s}_mat") for s in SCREEN_SIDES]
        self.enabled = all(m >= 0 for m in self.mats)
        self.tier = "idle"

    @staticmethod
    def tier_for(sound: str | None, mission_status: str) -> str:
        if mission_status == "complete":
            return "complete"
        if mission_status == "active" and sound:
            return sound
        return "idle"

    def update(self, model: mujoco.MjModel, t: float, sound: str | None, mission_status: str) -> None:
        if not self.enabled:
            return
        self.tier = self.tier_for(sound, mission_status)
        if self.tier == "idle":
            rgba, emission = IDLE_RGBA, 0.0
        elif self.tier == "smoke_alarm":  # danger: red <-> orange, same rhythm family as the light bar
            rgba, emission = (LIT["red"] if (t % 0.5) < 0.25 else LIT["orange"]), 1.0
        else:
            rgba, emission = LIT[TIER_COLOUR.get(self.tier, "white")], 0.8
        for m in self.mats:
            model.mat_rgba[m] = rgba
            model.mat_emission[m] = emission

    @staticmethod
    def draw_label(scn: mujoco.MjvScene | None, pose, text: str) -> None:
        """Floating 'Nudge says' text above the robot (viewer user scene; call under viewer.lock())."""
        if scn is None:
            return
        scn.ngeom = 0
        if not text:
            return
        rx, ry, _ = pose
        g = scn.geoms[0]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3), np.array([rx, ry, LABEL_HEIGHT]),
                            np.eye(3).flatten(), np.array([1.0, 1.0, 1.0, 1.0]))
        g.label = f"Nudge says: {text}"[:LABEL_MAX]
        scn.ngeom = 1
