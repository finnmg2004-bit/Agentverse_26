#!/usr/bin/env python3
"""
Owner behaviour model (`sim/owner_behavior.py`): the owner acts like a person, not a prop.

States
  idle      Standing at a spot, shifting weight / turning occasionally. After a dwell time
            (UNPREDICTABLE: 35-90 s) they walk to another room on their own (if wandering is on).
  walking   Following a floor-plan path (same static map the robot uses) at ~0.45 m/s,
            pausing briefly at doorways (feeling for the frame - they are deafblind).
  guided    Holding the robot's handle. The owner's walking pace varies (0.30-0.40 m/s), and now
            and then they LET GO (stumble / adjust grip) and stand still until the robot comes
            back within reach.
  alerted   Felt the robot's nudge: stop, turn toward the robot, and after a human reaction
            delay TAP THE HANDLE to acknowledge. They don't always notice the first nudge.

Owner state (overnight S4; `state.owner_state_gt`, judge only, the brain must never read it)
  awake     normal behaviour above (default; [R] reset returns to it)
  asleep    in the guest bed: no wandering, no fidgeting. Only a nudge with alert mode `1_pulse`
            or stronger (WAKE_ALERT_MODES) wakes them - speech alone doesn't; waking is slow
            (WAKE_REACTION_S), then they are awake and react like above.
  idle      awake but staying put (no wandering), e.g. to trigger a taught "idle" task.
  unresponsive  doesn't react to anything (no tap, won't take the handle): tests the brain's
            no-response escalation (#20). Added by Finn's lane; [R] / "awake" undo it.
  fallen    lying on the floor where they are (the scan tipped over, as in scripts/test_owner_check.py),
            reacts to nothing: tests the brain's fallen-owner check (YOLO misses a person lying down).
            Optional "yaw" in the gui action = the direction the head points. Finn's lane; [R] / "awake"
            stand them up again.
  Set by the `gui` action {"action": "owner_state", "state": "asleep"} (sim/run_sim.py).

Sensors exposed to the brain (published in `state`):
  handle_tap  true while the owner taps the handle (a touch sensor on the handle)
The owner's position itself is never given to the brain (only `owner_pos_gt` for debugging).

Hotkey [W] toggles wandering. [O] (teleport) and [R] (reset) reset the behaviour.
Everything uses its own seeded RNG so a run can be reproduced with --owner-seed.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

from sim.house_layout import DOORWAYS, OWNER_SPOTS, get_room_at

WALK_SPEED = (0.38, 0.52)          # m/s, chosen per walk
GUIDED_SPEED = (0.30, 0.40)        # m/s pace on the handle, re-drawn every few seconds
DWELL_S = (35.0, 90.0)             # how long they stay put before wandering
FIRST_WANDER_MIN_S = 45.0          # never wander off in the first 45 s after start / reset
DOOR_PAUSE_S = (0.4, 1.0)          # tactile pause at a doorway
NOTICE_FIRST_NUDGE = 0.7           # chance the very first nudge is noticed
REACTION_S = (0.6, 1.6)            # nudge felt -> handle tap
TAP_S = 0.8                        # how long the tap sensor reads true
LET_GO_RATE = 0.025                # per second while guided (after the first few seconds)
LET_GO_HOLD_S = (2.0, 3.5)         # stands still at least this long after letting go
LET_GO_COOLDOWN_S = 30.0
REGRAB_REACH = 1.1                 # robot handle within this distance -> owner can regrab
NUDGE_CONTACT = 0.60               # robot centre this close + alert on = nudge felt
OWNER_STATES = ("awake", "asleep", "idle", "unresponsive", "fallen")
FALLEN_Z = 0.14                    # body origin height while lying (scan tipped over 90 deg)
WAKE_ALERT_MODES = ("1_pulse", "2_pulse", "continuous")  # what can wake a sleeping owner (vibration)
WAKE_REACTION_S = (1.5, 3.0)       # asleep: nudge felt -> awake + handle tap


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class OwnerBehavior:
    def __init__(self, locomotion, planner_factory, seed: Optional[int] = None, wander: bool = True) -> None:
        self.loco = locomotion
        self._planner_factory = planner_factory
        self._planner = None
        self.rng = random.Random(seed)
        self.wander = wander
        self.reset(0.0)

    # ------------------------------------------------------------------ control
    def reset(self, t: float) -> None:
        self._now = t
        self.owner_state = "awake"
        self._t_prev = t
        self.state = "idle"
        self.t_state = t
        self.dwell_until = t + max(FIRST_WANDER_MIN_S, self.rng.uniform(*DWELL_S))
        self.path: List[Tuple[float, float]] = []
        self.walk_speed = 0.45
        self.pause_until = 0.0
        self.yaw = 0.0
        self.next_fidget = t + self.rng.uniform(4.0, 10.0)
        self.target_yaw: Optional[float] = None
        # alert / acknowledgement
        self.nudges_felt = 0
        self.alert_session = False
        self.ack_at: Optional[float] = None
        self.tap_until = 0.0
        self.ignored_this_contact = False
        self.ignored_at = t
        self.felt_at: Optional[float] = None  # sim time the owner last FELT a nudge (sim/robot_nudge.py reacts)
        # guiding
        self.loco.owner_grip = True
        self.loco.owner_speed = 0.38
        self.next_pace = t + 3.0
        self.guided_since: Optional[float] = None
        self.let_go_until = 0.0
        self.last_let_go = -1e9

    def on_teleport(self, t: float) -> None:
        self.reset(t)
        self.dwell_until = t + self.rng.uniform(*DWELL_S)

    def toggle_wander(self) -> bool:
        self.wander = not self.wander
        if self.state == "walking" and not self.wander:
            self.state, self.path = "idle", []
        return self.wander

    @property
    def handle_tap(self) -> bool:
        return self._now < self.tap_until

    def status(self) -> str:
        """Activity: idle | walking | guided | alerted (published as `owner_activity_gt`)."""
        return self.state

    def set_owner_state(self, state: str, t: float, yaw: Optional[float] = None) -> bool:
        """awake | asleep | idle | unresponsive | fallen (published as `owner_state_gt`).
        Returns False for an unknown state. `yaw` (fallen only): where the head points."""
        if state not in OWNER_STATES:
            return False
        was_fallen = self.owner_state == "fallen"
        self.owner_state = state
        self.loco.owner_grip = state not in ("unresponsive", "fallen")  # neither ever takes the handle
        if state == "fallen":
            self._lie_down(self.yaw if yaw is None else float(yaw))
        elif was_fallen:
            ox, oy = self.loco.get_owner_pos()
            self._set_pose(ox, oy, self.yaw)  # stand up again
        if state != "awake":
            self.state, self.path, self.target_yaw = "idle", [], None
        else:
            self.dwell_until = t + max(FIRST_WANDER_MIN_S, self.rng.uniform(*DWELL_S))
        return True

    # ------------------------------------------------------------------ helpers
    def _planner_get(self):
        if self._planner is None:
            self._planner = self._planner_factory()
        return self._planner

    def _start_walk(self, t: float) -> None:
        ox, oy = self.loco.get_owner_pos()
        here = get_room_at(ox, oy)
        options = [r for r in OWNER_SPOTS if r != here]
        if not options:
            return
        dest = self.rng.choice(options)
        wps = self._planner_get().plan((ox, oy), OWNER_SPOTS[dest], ())
        if not wps:
            self.dwell_until = t + self.rng.uniform(*DWELL_S)
            return
        self.path = [(w[0], w[1]) for w in wps]
        self.walk_speed = self.rng.uniform(*WALK_SPEED)
        self.state, self.t_state = "walking", t
        print(f"[sim] owner decides to walk from {here} to {dest} ({self.walk_speed:.2f} m/s)", flush=True)

    def _set_pose(self, x: float, y: float, yaw: float) -> None:
        mid = self.loco.owner_mocap_id
        if mid < 0:
            return
        from sim.locomotion import yaw_to_quat
        self.loco.data.mocap_pos[mid] = [x, y, 0.0]
        self.loco.data.mocap_quat[mid] = yaw_to_quat(yaw)
        self.yaw = yaw

    def _lie_down(self, yaw: float) -> None:
        """Fallen: tip the owner 90 deg about the body's y axis (head along `yaw`, back on the floor)."""
        mid = self.loco.owner_mocap_id
        if mid < 0:
            return
        ox, oy = self.loco.get_owner_pos()
        a, b = math.cos(yaw / 2), math.sin(yaw / 2)       # q_yaw = (a, 0, 0, b)
        c = math.cos(math.pi / 4)                           # q_tip = (c, 0, c, 0)
        self.loco.data.mocap_pos[mid] = [ox, oy, FALLEN_Z]
        self.loco.data.mocap_quat[mid] = [a * c, -b * c, a * c, b * c]  # q_yaw * q_tip
        self.yaw = yaw

    def _turn_toward(self, target_yaw: float, dt: float, rate: float = 1.5) -> None:
        ox, oy = self.loco.get_owner_pos()
        err = _wrap(target_yaw - self.yaw)
        step = max(-rate * dt, min(rate * dt, err))
        self._set_pose(ox, oy, self.yaw + step)

    # ------------------------------------------------------------------ main update
    def step(self, t: float, dt: float, robot_pose, alert_mode: str, cmd_guiding: bool) -> None:
        self._now = t
        rx, ry, ryaw = robot_pose
        ox, oy = self.loco.get_owner_pos()
        d_robot = math.hypot(rx - ox, ry - oy)
        if self.owner_state in ("unresponsive", "fallen"):  # #20 / fallen test hooks: feel, tap, hold nothing
            self.loco.owner_grip = False
            return

        # ---- being guided (locomotion moves the owner while owner_grip is True)
        if self.loco.guiding_active or (cmd_guiding and self.state == "guided"):
            if self.state != "guided":
                self.state, self.t_state, self.guided_since = "guided", t, t
                self.path = []
            self._step_guided(t, d_robot, robot_pose)
            return
        if self.state == "guided" and not cmd_guiding:
            # Robot finished guiding: stand where we were delivered.
            self.state, self.t_state = "idle", t
            self.loco.owner_grip = True
            self.dwell_until = t + self.rng.uniform(*DWELL_S)

        # ---- nudge / acknowledgement
        alert_on = alert_mode not in ("none", "complete")
        if not alert_on:
            self.alert_session = False
            self.nudges_felt = 0
            self.ack_at = None
        if self.owner_state == "asleep":
            # Only a real vibration nudge wakes them; anything else (speech, lights) does nothing.
            if alert_mode in WAKE_ALERT_MODES and d_robot < NUDGE_CONTACT:
                self.owner_state = "awake"
                self.alert_session = True
                self.nudges_felt += 1
                self.state, self.t_state, self.path = "alerted", t, []
                self.ack_at = t + self.rng.uniform(*WAKE_REACTION_S)
                self.felt_at = t
                self.dwell_until = t + max(FIRST_WANDER_MIN_S, self.rng.uniform(*DWELL_S))
                print(f"[sim] owner WAKES UP (nudge {alert_mode})", flush=True)
            return
        if alert_on and d_robot < NUDGE_CONTACT:
            if not self.ignored_this_contact and self.ack_at is None and self.state != "alerted":
                first = self.nudges_felt == 0 and not self.alert_session
                noticed = self.rng.random() < (NOTICE_FIRST_NUDGE if first else 0.95)
                self.alert_session = True
                if noticed:
                    self.nudges_felt += 1
                    self.state, self.t_state, self.path = "alerted", t, []
                    self.ack_at = t + self.rng.uniform(*REACTION_S)
                    self.felt_at = t
                    print("[sim] owner FEELS the nudge", flush=True)
                else:
                    self.ignored_this_contact = True
                    self.ignored_at = t
                    print("[sim] owner does not notice the nudge (distracted)", flush=True)
        # A missed touch is forgotten once the dog steps back, or after a few seconds anyway,
        # so the next nudge is a fresh chance to be noticed.
        if self.ignored_this_contact and (d_robot > NUDGE_CONTACT + 0.03 or t - self.ignored_at > 2.5):
            self.ignored_this_contact = False

        if self.state == "alerted":
            self._turn_toward(math.atan2(ry - oy, rx - ox), dt, rate=2.5)
            if self.ack_at is not None and t >= self.ack_at:
                self.tap_until = t + TAP_S
                self.ack_at = None
                print("[sim] owner TAPS the handle (acknowledged)", flush=True)
            if t > self.tap_until and self.ack_at is None and t - self.t_state > 1.0:
                self.state, self.t_state = "idle", t
                self.dwell_until = t + self.rng.uniform(*DWELL_S)
            return

        # ---- walking on their own
        if self.state == "walking":
            # Bumping into the dog stops them (it is in the way / it is trying to get attention).
            if d_robot < 0.7:
                return
            if t < self.pause_until:
                return
            if not self.path:
                self.state, self.t_state = "idle", t
                self.dwell_until = t + self.rng.uniform(*DWELL_S)
                return
            wx, wy = self.path[0]
            dx, dy = wx - ox, wy - oy
            dist = math.hypot(dx, dy)
            stepd = self.walk_speed * dt
            if dist <= stepd:
                self._set_pose(wx, wy, self.yaw)
                self.path.pop(0)
                if any(math.hypot(wx - px, wy - py) < 0.6 for px, py in DOORWAYS.values()):
                    self.pause_until = t + self.rng.uniform(*DOOR_PAUSE_S)
                return
            heading = math.atan2(dy, dx)
            self._set_pose(ox + dx / dist * stepd, oy + dy / dist * stepd,
                           self.yaw + _wrap(heading - self.yaw) * min(1.0, 6.0 * dt))
            return

        # ---- idle: fidget, maybe wander off
        if t >= self.next_fidget:
            self.target_yaw = self.yaw + self.rng.uniform(-1.2, 1.2)
            self.next_fidget = t + self.rng.uniform(5.0, 14.0)
        if self.target_yaw is not None:
            self._turn_toward(self.target_yaw, dt, rate=0.8)
            if abs(_wrap(self.target_yaw - self.yaw)) < 0.02:
                self.target_yaw = None
        if (self.wander and self.owner_state == "awake" and t >= self.dwell_until and d_robot > 1.5
                and not alert_on):
            self._start_walk(t)

    def _step_guided(self, t: float, d_robot: float, robot_pose) -> None:
        loco = self.loco
        if t >= self.next_pace:
            loco.owner_speed = self.rng.uniform(*GUIDED_SPEED)
            self.next_pace = t + self.rng.uniform(2.0, 5.0)
        if loco.owner_grip:
            held = t - (self.guided_since or t)
            if (held > 4.0 and t - self.last_let_go > LET_GO_COOLDOWN_S
                    and self.rng.random() < LET_GO_RATE * (t - getattr(self, "_t_prev", t))):
                loco.owner_grip = False
                self.let_go_until = t + self.rng.uniform(*LET_GO_HOLD_S)
                self.last_let_go = t
                print("[sim] owner LETS GO of the handle", flush=True)
        else:
            # Re-grab once they've steadied themselves and the handle is within reach.
            from sim.locomotion import OWNER_OFFSET_BODY  # where the owner stands on the handle
            rx, ry, ryaw = robot_pose
            bx, by = OWNER_OFFSET_BODY
            hx = rx + bx * math.cos(ryaw) - by * math.sin(ryaw)
            hy = ry + bx * math.sin(ryaw) + by * math.cos(ryaw)
            ox, oy = loco.get_owner_pos()
            if t >= self.let_go_until and math.hypot(hx - ox, hy - oy) < REGRAB_REACH:
                loco.owner_grip = True
                print("[sim] owner takes the handle again", flush=True)
        self._t_prev = t
