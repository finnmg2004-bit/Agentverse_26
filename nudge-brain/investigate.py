#!/usr/bin/env python3
"""
"Go and check" skill for the sound check (AGENTS.md 12.0 P6, nudge-brain/sound_check.py).

When a doorbell / knock might be the living-room TV and the ears aren't sure yet, Nudge doesn't just
stand and think: it walks a few metres toward the likely source (the TV, house_layout.TV_SOURCE - map
knowledge, not ground truth) and then turns its head camera toward it, so every next 0.5 s audio window
is heard from a new place. sound_check.py decides; this skill only moves the robot and ends when the
sound check gives its verdict (or after `timeout_s`, standing still).

Started / stopped only by SoundCheck.gate (never by the agent), and only while there is no mission.
"""

from __future__ import annotations

import math
from typing import Tuple

from navigator import Navigator, wrap
from skills import CRUISE_SPEED, Skill, SkillOutput, _nav_step

WALK_M = 2.4             # walk at most this far toward the sound, then stand and look
WALK_SPEED = 0.8 * CRUISE_SPEED
LOOK_TOL = 0.12          # rad: facing the source "enough"
NEAR_M = 3.0             # already this close to the source: just turn and look
STANDOFF_M = 1.3         # walking goal: this far in front of the source (into its room)


class InvestigateSound(Skill):
    name = "investigate_sound_check"
    timeout_s = 7.0

    def __init__(self, sound: str, source_xy: Tuple[float, float], facing_xy: Tuple[float, float],
                 world, nav: Navigator) -> None:
        super().__init__()
        self.sound = sound
        self.source = (float(source_xy[0]), float(source_xy[1]))
        self.start_xy = (float(world.robot_pose[0]), float(world.robot_pose[1]))
        self.walked = 0.0
        self.last_xy = self.start_xy
        d = math.hypot(self.source[0] - self.start_xy[0], self.source[1] - self.start_xy[1])
        self.stage = "look" if d <= NEAR_M else "walk"
        if self.stage == "walk":
            # goal in front of the source (e.g. under the TV, facing it), reached through the doorways
            fx, fy = facing_xy
            n = math.hypot(fx, fy) or 1.0
            goal = (self.source[0] + STANDOFF_M * fx / n, self.source[1] + STANDOFF_M * fy / n)
            if not nav.set_goal(goal, self.start_xy, world.active_blocked_doors(), final_tol=0.5):
                self.stage = "look"

    def label(self) -> str:
        return f"investigate({self.sound}: TV or door?, {self.stage})"

    def step(self, world, nav: Navigator, alerts) -> SkillOutput:
        x, y, yaw = world.robot_pose
        self.walked += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        if self.stage == "walk":
            vx, vy, wz, status = _nav_step(world, nav, WALK_SPEED)
            if status in ("arrived", "stuck", "idle") or self.walked >= WALK_M:
                nav.clear()
                self.stage = "look"
            else:
                return SkillOutput(vx, vy, wz)
        # look: turn the head camera toward the source and hold (until the verdict ends this skill)
        err = wrap(math.atan2(self.source[1] - y, self.source[0] - x) - yaw)
        if abs(err) <= LOOK_TOL:
            return SkillOutput()
        return SkillOutput(wz=max(-1.2, min(1.2, 2.0 * err)))
