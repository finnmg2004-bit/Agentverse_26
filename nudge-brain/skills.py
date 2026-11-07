#!/usr/bin/env python3
"""
Skill executor (`nudge-brain/skills.py`).
Turns one agent tool call into a running behaviour that outputs a velocity command every
tick, plus a discrete event when it finishes (which wakes the agent again):

    go_to_room(room)        -> GoToRoom        -> event "arrived_in_room"
    investigate_sound()     -> GoToRoom(sound room)
    look_around()           -> LookAround      -> event "owner_not_found" (owner_seen comes from vision)
    alert_owner(sound,place)-> ApproachAndAlert-> event "owner_alerted" | "owner_not_found"
    offer_handle()          -> OfferHandle     -> event "handle_taken"
    guide_owner_to(target)  -> GuideOwner      -> event "guide_complete"
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sim.house_layout import EXITS, LANDMARKS, OUTSIDE, ROOMS, SOUND_SOURCES, WALL_THICK, get_room_at

from alerts import HapticAlerts
from navigator import LIDAR_ANGLES, Navigator, wrap

CRUISE_SPEED = 0.9
GUIDE_SPEED = 0.35
STANDOFF = 0.75
NUDGE_CONTACT_M = 0.47    # robot centre to owner centre at the moment of the nudge
ACK_WAIT_S = 3.0         # wait this long for the owner's handle tap after the buzz
MAX_NUDGES = 3            # then nudge again, up to this many times
OWNER_AT_GOAL_TOL = 0.7   # owner within this distance of a landmark counts as delivered
FINAL_PUSH_SPEED = 0.25   # robot creeps on after reaching the goal so the owner (behind it) gets there
FINAL_PUSH_MAX_S = 8.0    # give up (event "timeout", not a false "guide_complete") after this
FINAL_PUSH_WALL_STOP = 0.3
# Evacuated = owner's whole body past the outer face of the exit wall, not just its centre.
OWNER_BODY_RADIUS = 0.2


def owner_clear_of_exit(ox: float, oy: float, goal: Tuple[float, float]) -> bool:
    """True if the owner is outside the house and fully past the wall of the exit leading to `goal`."""
    if get_room_at(ox, oy) != OUTSIDE:
        return False
    exit_ = min(EXITS.values(), key=lambda e: math.hypot(e["outside"][0] - goal[0], e["outside"][1] - goal[1]))
    (px, py), (gx, gy) = exit_["pos"], exit_["outside"]
    margin = WALL_THICK / 2 + OWNER_BODY_RADIUS
    if exit_["axis"] == "x":  # exit in a vertical wall
        return math.copysign(1.0, gx - px) * (ox - px) >= margin
    return math.copysign(1.0, gy - py) * (oy - py) >= margin


@dataclass
class SkillOutput:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    guiding: bool = False
    allow_contact: bool = False
    event: Optional[str] = None
    done: bool = False


class Skill:
    name = "idle"
    timeout_s = 1e9

    def __init__(self) -> None:
        self.started = time.monotonic()

    def label(self) -> str:
        return self.name

    def step(self, world, nav: Navigator, alerts: HapticAlerts) -> SkillOutput:
        return SkillOutput()


class GoToRoom(Skill):
    name = "go_to_room"
    timeout_s = 45.0

    def __init__(self, room: str, world, nav: Navigator) -> None:
        super().__init__()
        self.room = room
        nav.set_goal(ROOMS[room]["center"], world.robot_pose[:2], world.active_blocked_doors(), final_tol=0.35)

    def label(self) -> str:
        return f"go_to_room({self.room})"

    def step(self, world, nav, alerts) -> SkillOutput:
        vx, vy, wz, status = nav.step(world.robot_pose, world.lidar_ranges, CRUISE_SPEED)
        if status == "arrived":
            nav.clear()
            return SkillOutput(event="arrived_in_room", done=True)
        return SkillOutput(vx, vy, wz)


class LookAround(Skill):
    name = "look_around"
    timeout_s = 15.0
    RATE = 1.1

    def __init__(self, world) -> None:
        super().__init__()
        self.room = world.robot_room
        self.turned = 0.0
        self.last_yaw = world.robot_pose[2]

    def step(self, world, nav, alerts) -> SkillOutput:
        yaw = world.robot_pose[2]
        self.turned += abs(wrap(yaw - self.last_yaw))
        self.last_yaw = yaw
        if self.turned >= 2 * math.pi - 0.2:
            world.searched_rooms.add(self.room)
            if world.mission is not None and not world.owner_visible:
                return SkillOutput(event="owner_not_found", done=True)
            return SkillOutput(done=True)
        return SkillOutput(wz=self.RATE)


class ApproachAndAlert(Skill):
    """Walk up to the owner, nudge them (physical contact), then start the vibration code."""

    name = "alert_owner"
    timeout_s = 60.0

    def __init__(self, sound: str, world, nav: Navigator) -> None:
        super().__init__()
        self.sound = sound
        self.stage = "approach"
        self.stage_t = time.monotonic()
        self.attempt = 1
        self.nudge_t = time.monotonic()
        self.goal_owner: Optional[Tuple[float, float]] = None
        self._replan(world, nav)

    def label(self) -> str:
        return f"alert_owner({self.sound}) [{self.stage}]"

    def _replan(self, world, nav: Navigator) -> None:
        if world.owner_pos is None:
            return
        ox, oy = world.owner_pos
        rx, ry, _ = world.robot_pose
        d = math.hypot(rx - ox, ry - oy) or 1.0
        standoff = (ox + STANDOFF * (rx - ox) / d, oy + STANDOFF * (ry - oy) / d)
        nav.set_goal(standoff, (rx, ry), world.active_blocked_doors(), final_tol=0.2)
        self.goal_owner = world.owner_pos

    def _to(self, stage: str) -> None:
        self.stage = stage
        self.stage_t = time.monotonic()

    def step(self, world, nav, alerts) -> SkillOutput:
        dt_stage = time.monotonic() - self.stage_t
        if world.owner_pos is None:
            return SkillOutput(event="owner_not_found", done=True)

        # Owner gone from where we thought (e.g. moved mid-approach): close enough that the
        # camera should see them, yet nothing for 2 s -> give up and search. Not checked during
        # the nudge itself, where the owner is too close for the depth camera.
        if (self.stage in ("approach", "face") and world.dist_to_owner() < 2.0
                and not world.owner_recently_seen(2.0)):
            world.searched_rooms.add(world.robot_room)
            nav.clear()
            if not world.is_emergency():
                alerts.clear()
            return SkillOutput(event="owner_not_found", done=True)

        ox, oy = world.owner_pos
        rx, ry, yaw = world.robot_pose
        face_err = wrap(math.atan2(oy - ry, ox - rx) - yaw)

        if self.stage == "approach":
            if self.goal_owner and math.hypot(ox - self.goal_owner[0], oy - self.goal_owner[1]) > 0.4:
                self._replan(world, nav)
            vx, vy, wz, status = nav.step(world.robot_pose, world.lidar_ranges, CRUISE_SPEED)
            if status == "arrived" or world.dist_to_owner() < STANDOFF + 0.1:
                nav.clear()
                self._to("face")
            return SkillOutput(vx, vy, wz)

        if self.stage == "face":
            if abs(face_err) < 0.15 or dt_stage > 4.0:
                self._to("nudge")
                self.nudge_t = time.monotonic()
                self.attempt = world.nudge_attempts + 1
                world.owner_ack = f"waiting - nudge {self.attempt}"
                alerts.set_for_sound(self.sound)
            return SkillOutput(wz=max(-1.5, min(1.5, 2.5 * face_err)))

        if self.stage == "nudge":  # gentle forward bump into the owner's leg, until contact
            if world.dist_to_owner() < NUDGE_CONTACT_M or dt_stage > 3.0:
                self._to("retreat")
            return SkillOutput(vx=0.18, wz=1.5 * face_err, allow_contact=True)

        if self.stage == "retreat":  # step back far enough that a repeat nudge is a new touch
            if world.dist_to_owner() > STANDOFF - 0.05 or dt_stage > 2.5:
                self._to("buzz")
            return SkillOutput(vx=-0.15)

        if self.stage == "buzz":  # hold still while the handle vibrates the code
            if world.owner_tapped_since(self.nudge_t):
                return self._acknowledged(world)
            if dt_stage > 1.5:
                self._to("await_ack")
            return SkillOutput()

        if self.stage == "await_ack":  # owner taps the handle = "I felt it, I'm ready"
            if world.owner_tapped_since(self.nudge_t):
                return self._acknowledged(world)
            if dt_stage > ACK_WAIT_S:
                # No tap: the AGENT decides (nudge again / lead them anyway / call for help).
                world.nudge_attempts += 1
                world.owner_ack = f"no response after {world.nudge_attempts} nudge(s)"
                world.log(f"No handle tap after nudge {world.nudge_attempts}")
                return SkillOutput(event="owner_no_response", done=True)
            return SkillOutput()
        return SkillOutput(done=True)

    def _acknowledged(self, world) -> SkillOutput:
        world.owner_ack = "acknowledged"
        world.log(f"Owner ACKNOWLEDGED (handle tap) after nudge {self.attempt}")
        return SkillOutput(event="owner_alerted", done=True)


class OfferHandle(Skill):
    name = "offer_handle"
    timeout_s = 6.0

    def step(self, world, nav, alerts) -> SkillOutput:
        if world.owner_pos is None:
            return SkillOutput(event="owner_not_found", done=True)
        if world.guiding_active:
            return SkillOutput(guiding=True, event="handle_taken", done=True)
        # Turn the handle (robot's back) toward the owner.
        ox, oy = world.owner_pos
        rx, ry, yaw = world.robot_pose
        err = wrap(math.atan2(ry - oy, rx - ox) - yaw)
        return SkillOutput(wz=max(-1.5, min(1.5, 2.0 * err)), guiding=True)


class GuideOwner(Skill):
    """Lead the owner (holding the handle) to a room or landmark, waiting if they let go.

    Success is judged on where the OWNER is, not the robot: the owner walks ~0.5 m behind the
    robot on the handle, so when the robot reaches the goal the owner is still short of it
    (e.g. still inside the house during a smoke-alarm evacuation). After the robot arrives it
    keeps creeping on until the owner is delivered, and reports "timeout" if that fails.
    """

    name = "guide_owner_to"
    timeout_s = 120.0

    def __init__(self, target: str, world, nav: Navigator) -> None:
        super().__init__()
        self.target = target
        self.goal = LANDMARKS[target] if target in LANDMARKS else ROOMS[target]["center"]
        self.lost_since: Optional[float] = None
        self.push_started: Optional[float] = None
        self.regrabbing = False
        nav.set_goal(self.goal, world.robot_pose[:2], world.active_blocked_doors(), final_tol=0.3)

    def label(self) -> str:
        if self.regrabbing:
            return f"guide_owner_to({self.target}) [owner let go]"
        if self.push_started is not None:
            return f"guide_owner_to({self.target}) [bringing owner in]"
        return f"guide_owner_to({self.target})"

    def _owner_delivered(self, world) -> bool:
        if world.owner_pos is None:
            return False
        ox, oy = world.owner_pos
        if get_room_at(*self.goal) == OUTSIDE:
            return owner_clear_of_exit(ox, oy, self.goal)  # fully outside, not standing in the doorway
        if self.target in ROOMS:
            return get_room_at(ox, oy) == self.target
        return math.hypot(ox - self.goal[0], oy - self.goal[1]) <= OWNER_AT_GOAL_TOL

    def _final_push(self, world) -> SkillOutput:
        """Robot is at the goal but the owner is not: keep walking the way the owner must go."""
        if self._owner_delivered(world):
            return SkillOutput(guiding=True, event="guide_complete", done=True)
        if time.monotonic() - self.push_started > FINAL_PUSH_MAX_S:
            world.log(f"Guide to {self.target} FAILED: owner not delivered")
            return SkillOutput(guiding=True, event="timeout", done=True)
        if world.owner_pos is None:
            return SkillOutput(guiding=True)
        ox, oy = world.owner_pos
        rx, ry, yaw = world.robot_pose
        err = wrap(math.atan2(self.goal[1] - oy, self.goal[0] - ox) - yaw)
        wz = max(-1.5, min(1.5, 2.4 * err))
        vx = FINAL_PUSH_SPEED if abs(err) < 0.5 else 0.0
        front = min(r for a, r in zip(LIDAR_ANGLES, world.lidar_ranges) if abs(a) <= math.radians(22))
        if front < FINAL_PUSH_WALL_STOP:
            vx = 0.0
        return SkillOutput(vx=vx, wz=wz, guiding=True)

    def step(self, world, nav, alerts) -> SkillOutput:
        now = time.monotonic()
        grace = now - self.started < 2.5
        if world.guiding_active or grace:
            self.lost_since = None
            if self.regrabbing:
                self.regrabbing = False
                self.push_started = None
                nav.set_goal(self.goal, world.robot_pose[:2], world.active_blocked_doors(), final_tol=0.3)
            if self.push_started is not None:
                return self._final_push(world)
            vx, vy, wz, status = nav.step(world.robot_pose, world.lidar_ranges, GUIDE_SPEED)
            if status == "arrived":
                nav.clear()
                if self._owner_delivered(world):
                    return SkillOutput(guiding=True, event="guide_complete", done=True)
                self.push_started = now
                return self._final_push(world)
            return SkillOutput(vx, vy, wz, guiding=True)

        # Owner not on the handle: stop, then back up toward their last known position.
        self.lost_since = self.lost_since or now
        if now - self.lost_since < 1.0 or world.owner_pos is None:
            return SkillOutput(guiding=True)
        self.regrabbing = True
        ox, oy = world.owner_pos
        rx, ry, yaw = world.robot_pose
        d = math.hypot(ox - rx, oy - ry)
        # Keep facing the goal direction; reverse/strafe in body frame toward the owner.
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = (ox - rx) * c + (oy - ry) * s, -(ox - rx) * s + (oy - ry) * c
        k = 0.3 / max(d, 1e-3)
        return SkillOutput(vx=bx * k, vy=by * k, guiding=True)


class SkillExecutor:
    def __init__(self, nav: Navigator, alerts: HapticAlerts) -> None:
        self.nav = nav
        self.alerts = alerts
        self.skill: Skill = Skill()

    def execute(self, action: Dict[str, Any], world) -> None:
        tool = action.get("tool")
        args = action.get("args") or {}
        self.nav.clear()
        if tool == "go_to_room":
            self.skill = GoToRoom(args["room"], world, self.nav)
            world.phase = "searching" if world.mission else world.phase
        elif tool == "investigate_sound":
            label = world.mission["sound"] if world.mission else world.sound.get("label")
            # Where the ears localised it (sound_localizer.py); the registry room only as a fallback.
            est = world.sound_estimates.get(label) or {}
            room = est.get("room") or str(SOUND_SOURCES.get(label, {}).get("room", world.robot_room))
            self.skill = GoToRoom(room, world, self.nav)
        elif tool == "look_around":
            self.skill = LookAround(world)
        elif tool == "alert_owner":
            self.skill = ApproachAndAlert(args.get("sound", "doorbell"), world, self.nav)
            world.phase = "approaching"
        elif tool == "offer_handle":
            self.skill = OfferHandle()
        elif tool == "guide_owner_to":
            world.phase = "guiding"  # before planning, so evacuation hazard doors apply
            self.skill = GuideOwner(args["target"], world, self.nav)
        else:  # get_state / none
            self.skill = Skill()
        world.log(f"SKILL {self.skill.label()}")

    def step(self, world) -> SkillOutput:
        out = self.skill.step(world, self.nav, self.alerts)
        if not out.done and time.monotonic() - self.skill.started > self.skill.timeout_s:
            out = SkillOutput(event="timeout", done=True)
        if out.done:
            self.skill = Skill()
        return out
