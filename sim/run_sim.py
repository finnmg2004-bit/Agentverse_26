#!/usr/bin/env python3
"""
Milestones 1–5: Main MuJoCo Simulator & ZMQ Telemetry/Command Server (`sim/run_sim.py`).
Loads the nerv-world house (ground floor) + Unitree Go2 with Owner body, movable Chair,
and sound beacons (sim/world.py; coordinates in sim/house_layout.py).
Supports:
  - Interactive MuJoCo viewer with keyboard hotkeys:
      [B / 1] Ring Doorbell (Entry hall, front door)
      [K / 2] Knock on Front Door (Entry hall)
      [M / 3] Trigger Smoke Alarm (Kitchen ceiling — Emergency)
      [O / 4] Cycle Owner room (living_room -> guest -> dining -> family_room)
      [C / 5] Toggle Chair obstacle in the guest <-> living_room door (-7.6, 0.6)
      [X]     Silence all active sounds
      [R]     Reset the scene (robot, owner, chair, sounds) and tell the brain to reset
      [N / 6] Cycle background noise (quiet -> fan -> tv -> vacuum -> party)   also: --noise NAME
      [V]     Cycle viewer camera (top -> follow robot -> dog's head camera)   also: --view NAME
  - Scenario end conditions judged from ground truth: owner at the front door ends a doorbell /
    knock (sound stops); owner at the back door or outside ends a smoke alarm (alarm clears).
  - ZeroMQ PUB on tcp://127.0.0.1:5555 (`state` @ 20 Hz, `camera` @ 10 Hz, `audio` @ 2 Hz)
  - ZeroMQ SUB on tcp://127.0.0.1:5556 (`cmd` @ 20 Hz + optional `sim_event` for scripted tests)
  - Mission status overlay in the viewer (from the brain's `mission_status` in `cmd`); the
    robot's halo turns steady green when a mission is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional


def _prefer_wsl_nvidia_gpu() -> None:
    """Opt-in (NUDGE_GPU=1): on WSL2, render through Mesa's d3d12 driver on the NVIDIA GPU
    instead of llvmpipe (CPU). Mesa needs both GALLIUM_DRIVER=d3d12 and, on hybrid laptops,
    the NVIDIA adapter name. OFF by default: measured on an RTX 5050 laptop it was 2-3x SLOWER
    than llvmpipe (every frame is translated to DirectX and read back across the WSL boundary).
    Native Windows (`pixi run sim` on win-64) uses the NVIDIA OpenGL driver directly instead.
    Must run before MuJoCo creates a GL context (Mesa reads these at driver load)."""
    if not sys.platform.startswith("linux") or os.environ.get("NUDGE_GPU", "0") != "1":
        return
    if not (os.path.exists("/dev/dxg") and os.path.exists("/usr/lib/wsl/lib/libcuda.so.1")):
        return
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")


_prefer_wsl_nvidia_gpu()

import mujoco  # noqa: E402
import mujoco.viewer
import cv2
import numpy as np
import zmq

# Ensure project root is on sys.path when invoked as `python sim/run_sim.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.house_layout import (
    AVOID_AREA_MARGIN,
    AVOID_AREAS,
    CHAIR_DOORWAY_POS,
    CHAIR_STOWED_POS,
    LANDMARKS,
    EXITS,
    OUTSIDE,
    ROBOT_START_POSE,
    ROOMS,
    ROOM_ORDER,
    SOUND_SOURCES,
    avoid_area_at,
    get_room_at,
)
from sim.locomotion import Go2Locomotion
from sim.nav_map import load_or_build
from sim.lightbar import ALERT_MODE_SOUND, SOUND_INFO
from sim.robot_screen import RobotScreen
from sim.robot_voice import RobotVoice
from sim.sensors import NOISE_NAMES, SimSensors
from sim.world import build_nudge_model
from sim.owner_behavior import OWNER_STATES, OwnerBehavior
from sim.robot_nudge import NudgeFX
from sim.sim_clock import SimClock
from sim.house_layout import CHAIR_EXIT_POS

VIEW_NAMES = ["top", "follow", "dog"]

# Scenario end conditions (ground truth, judged by the sim, not the brain):
#   doorbell / knock -> owner stays at the front door        => visitor answered, sound stops
#   smoke_alarm      -> owner at the back door, or outside    => evacuated, every alarm clears
# A sound still unresolved after SCENARIO_TIMEOUT_S of sim time is reported as FAILED (keeps going).
# The owner walks behind the handle and lags further than the brain's handle-offset estimate: at
# delivery they stop ~1.0-1.3 m from the goal (kettle approached from the south: ~1.2 m), which a
# 1.15 m radius sometimes missed (flaky doorbell, TODO.md Verify). Same radius for every
# non-evacuation target (front door, kettle, ...).
GOAL_RADIUS_M = 1.4
GOAL_DWELL_S = 1.0    # must stay inside the radius this long, so walking past/through doesn't count
SCENARIO_TIMEOUT_S = 120.0
FIRE_TOO_CLOSE_M = 3.0  # judge only: the robot should never route this close to the fire (smoke alarm)


class NudgeSimulator:
    """Manages MuJoCo state, hotkeys, sensors, and ZeroMQ PUB/SUB sockets."""

    def __init__(
        self,
        pub_endpoint: str = "tcp://127.0.0.1:5555",
        sub_endpoint: str = "tcp://127.0.0.1:5556",
        enable_camera: bool = True,
        cutaway: bool = True,
        gui_endpoint: str = "tcp://127.0.0.1:5557",
        owner_wander: bool = True,
        owner_seed=None,
        clock: str = "08:00",
        coco_objects: bool = False,
    ) -> None:
        self.cutaway = cutaway
        if coco_objects:  # S7 (experimental): COCO-class stand-ins + state.objects_gt
            from sim.coco_standins import build_model_with_objects
            self.model, self.data = build_model_with_objects(verbose=True, cutaway=cutaway)
            # The static nav map stays the plain house: never cache a map with the stand-ins in it.
            from sim.nav_map import MAP_PATH, NavMap, _scene_stamp
            self.nav_map = NavMap.load(MAP_PATH, _scene_stamp()) or load_or_build(
                *build_nudge_model(verbose=False, cutaway=cutaway))
        else:
            self.model, self.data = build_nudge_model(verbose=True, cutaway=cutaway)
            self.nav_map = load_or_build(self.model, self.data)
        self.coco_objects = coco_objects
        self._objects_gt = None  # computed once (all static) on the first state publish

        self.locomotion = Go2Locomotion(self.model, self.data, self.nav_map)
        self.sensors = SimSensors(self.model, self.data, enable_camera=enable_camera)

        # The owner behaves like a person: wanders, reacts to nudges, taps the handle, may let go.
        from sim.nav_map import GridPlanner
        self.owner = OwnerBehavior(self.locomotion, lambda: GridPlanner(self.nav_map),
                                   seed=owner_seed, wander=owner_wander)
        # Sim clock (contract C2): time of day in the house, advancing with sim time.
        self.clock = SimClock(clock, float(self.data.time))
        self.chair_at_exit = False

        # Scene state controlled by hotkeys
        self.owner_room_idx: int = 0  # Starts in living_room
        self.chair_blocking: bool = False
        self.active_sounds: Dict[str, bool] = {name: False for name in SOUND_SOURCES}  # registry-driven
        self.beacon_geom_ids: Dict[str, int] = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"beacon_{name}")
            for name in SOUND_SOURCES
        }

        # Latest command from nudge-brain
        self.cmd_vx: float = 0.0
        self.cmd_vy: float = 0.0
        self.cmd_wz: float = 0.0
        self.cmd_alert_mode: str = "none"
        self.cmd_guiding: bool = False
        self.last_cmd_time: float = time.monotonic()

        # Episode counter: bumped by [R]. Published in `state` so the brain resets too, and
        # used to ignore stale `cmd` packets the brain sent for the previous episode.
        self.episode: int = 0
        # Mission status reported by the brain: "idle" | "active" | "complete"
        self.mission_status: str = "idle"
        self.mission_text: str = ""

        # ZeroMQ setup
        self.zmq_ctx = zmq.Context.instance()
        self.pub_sock = self.zmq_ctx.socket(zmq.PUB)
        self.pub_sock.setsockopt(zmq.SNDHWM, 10)
        self.pub_sock.setsockopt(zmq.LINGER, 0)
        self.pub_sock.bind(pub_endpoint)

        self.sub_sock = self.zmq_ctx.socket(zmq.SUB)
        self.sub_sock.setsockopt(zmq.RCVHWM, 20)
        self.sub_sock.setsockopt(zmq.LINGER, 0)
        self.sub_sock.connect(sub_endpoint)
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "cmd")
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "sim_event")
        # Brain status (`brain`, ~10 Hz, same :5556 socket): read only for the robot outputs -
        # mission.say ("Nudge says") and mission.sound (screen tier). Older brains just don't send it.
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "brain")
        # Control GUI (gui/nudge_control.py) binds :5557 and sends `gui` messages: hotkeys,
        # "move owner to room", and 3D-view camera requests. Connecting before it exists is fine.
        self.sub_sock.connect(gui_endpoint)
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "gui")

        # Publication timers
        self.last_state_pub: float = 0.0
        self.last_cam_pub: float = 0.0
        self.last_audio_pub: float = 0.0

        # 3D view stream for the control GUI: rendered only while a GUI keeps requesting it.
        self.view_req: Optional[dict] = None
        self.view_req_wall: float = 0.0
        self.view_renderer: Optional[mujoco.Renderer] = None
        self.view_cam = mujoco.MjvCamera()
        self.view_opt = mujoco.MjvOption()
        if cutaway:
            self.view_opt.geomgroup[1] = 0  # hide ceilings, same as the viewer's dollhouse view
        self.last_view_pub: float = 0.0

        # Viewer-window key presses arrive on the viewer's thread; they are queued here and
        # applied by the main loop under viewer.lock() so they never race the renderer.
        self.pending_keys: Deque[str] = deque()
        self.view_idx: int = 0
        self.view_changed: bool = False

        # Scenario tracking: sound -> sim time it started (cleared when resolved/silenced)
        self.sound_started: Dict[str, float] = {}
        # Judge-only fire check: closest the robot came to the smoke alarm (the fire) while it rang.
        self.fire_min_dist: float = float("inf")
        self.fire_too_close_logged: bool = False
        self.sound_timed_out: Dict[str, bool] = {}
        self.goal_since: Dict[str, float] = {}  # sound -> sim time the owner entered the goal radius
        self.last_scenario_check: float = 0.0

        # Avoid areas (house_layout.AVOID_AREAS, e.g. the living-room carpet): ground-truth
        # "did the robot step on it". Rules live in the brain; the sim only reports.
        self.in_avoid_area: str | None = None
        self.avoid_entered_at: float = 0.0
        self.avoid_steps: int = 0  # entries this episode

        # Robot outputs (AGENTS.md 9.1): flank screens coloured by alert tier, the "Nudge says"
        # message (viewer label + overlay + log) and speech (on by default, NUDGE_SPEECH=0 = off, sim/robot_voice.py).
        self.screen = RobotScreen(self.model)
        self.voice = RobotVoice()
        # Visible nudge (TODO #4): robot bump + contact ripple + owner reaction, render-only.
        self.nudge_fx = NudgeFX(self.model)
        self.nudge_says: str = ""
        self.mission_sound: str | None = None

    def _start_sound(self, name: str) -> None:
        if not self.active_sounds.get(name):
            self.sound_started[name] = float(self.data.time)
            self.sound_timed_out[name] = False
            if name == "smoke_alarm":
                self.fire_min_dist = float("inf")
                self.fire_too_close_logged = False
        self.active_sounds[name] = True

    def _stop_sound(self, name: str) -> None:
        self.active_sounds[name] = False
        self.sound_started.pop(name, None)
        self.sound_timed_out.pop(name, None)
        self.goal_since.pop(name, None)

    def check_scenario_end(self) -> None:
        """Resolve active sounds once the owner has actually reached (and stays at) the target."""
        t = float(self.data.time)
        if t - self.last_scenario_check < 0.1:
            return
        self.last_scenario_check = t
        ox, oy = self.locomotion.get_owner_pos()
        if "smoke_alarm" in self.sound_started:
            rx, ry, _ = self.locomotion.get_pose()
            fx, fy = SOUND_SOURCES["smoke_alarm"]["pos"][:2]
            fd = float(np.hypot(rx - fx, ry - fy))
            self.fire_min_dist = min(self.fire_min_dist, fd)
            if fd < FIRE_TOO_CLOSE_M and not self.fire_too_close_logged:
                self.fire_too_close_logged = True
                print(f"[sim] FIRE PROXIMITY: robot {fd:.1f} m from the fire at ({fx}, {fy}) "
                      f"(limit {FIRE_TOO_CLOSE_M} m), robot at ({rx:.2f}, {ry:.2f})")
        for name, started in list(self.sound_started.items()):
            target = str(SOUND_SOURCES[name]["target_landmark"])
            evac = name == "smoke_alarm"
            # Evacuation: the owner follows ~1 m behind the robot's handle, so when the robot is
            # out on the exit point the owner is in the back doorway. Count "at the back door or outside".
            tx, ty = LANDMARKS["back_door"] if evac else LANDMARKS[target]
            dist = float(np.hypot(ox - tx, oy - ty))
            outside = evac and get_room_at(ox, oy) == OUTSIDE
            # Back exit blocked by the chair ([E]): standing at it is not an evacuation, only OUTSIDE counts.
            at_goal = dist <= GOAL_RADIUS_M and not (evac and self.chair_at_exit)
            if at_goal or outside:
                self.goal_since.setdefault(name, t)
            else:
                self.goal_since.pop(name, None)
            if name in self.goal_since and t - self.goal_since[name] >= GOAL_DWELL_S:
                elapsed = t - started
                if evac:
                    for s in list(self.sound_started):
                        self._stop_sound(s)  # owner is leaving the house: nothing left to alert to
                    if outside:
                        via = min(EXITS, key=lambda n: float(np.hypot(ox - EXITS[n]["outside"][0],
                                                                      oy - EXITS[n]["outside"][1])))
                        where = f"OUTSIDE via {via}"
                    else:
                        where = "at the back door exit"
                    print(f"[sim] SCENARIO COMPLETE: smoke_alarm -> owner EVACUATED ({where}) "
                          f"after {elapsed:.1f}s (owner at {ox:.2f}, {oy:.2f}; robot came within "
                          f"{self.fire_min_dist:.1f} m of the fire). Alarm cleared.")
                else:
                    self._stop_sound(name)
                    print(f"[sim] SCENARIO COMPLETE: {name} -> owner reached the {target} after {elapsed:.1f}s "
                          f"(owner at {ox:.2f}, {oy:.2f}). Visitor answered, {name} stopped.")
                return
            if t - started > SCENARIO_TIMEOUT_S and not self.sound_timed_out.get(name):
                self.sound_timed_out[name] = True
                print(f"[sim] SCENARIO FAILED: {name} unresolved after {SCENARIO_TIMEOUT_S:.0f}s "
                      f"(owner at {ox:.2f}, {oy:.2f}, {dist:.1f} m from {target}).")

    def handle_brain_status(self, status: dict) -> None:
        """Robot outputs from the brain status: what Nudge says to the owner + which sound."""
        if int(status.get("episode", self.episode) or 0) != self.episode:
            return  # stale status from before an [R] reset
        mission = status.get("mission") or {}
        self.mission_sound = mission.get("sound")
        say = str(mission.get("say") or "").strip() if mission.get("status") == "active" else ""
        if not say:
            # Proposed add-only field (overnight C1/C3): a reply to the owner outside a mission, e.g. a
            # taught "just tell me" task or "Sorry, I can't do that yet...". Older brains don't send it.
            say = str(status.get("say") or "").strip()
        if say != self.nudge_says:
            self.nudge_says = say
            if say:
                print(f"[sim] NUDGE SAYS: {say}")
            self.voice.say_if_changed(say, urgent=self.mission_sound == "smoke_alarm")

    def check_avoid_areas(self) -> None:
        """Print once on entering / leaving an avoid area (robot centre within the footprint margin)."""
        rx, ry, _ = self.locomotion.get_pose()
        area = avoid_area_at(rx, ry, AVOID_AREA_MARGIN)
        t = float(self.data.time)
        if area and area != self.in_avoid_area:
            self.avoid_steps += 1
            self.avoid_entered_at = t
            sounds = [s for s, on in self.active_sounds.items() if on] or ["none"]
            print(f"[sim] AVOID AREA: robot stepped ON {area} at ({rx:.2f}, {ry:.2f}) "
                  f"[active sounds: {', '.join(sounds)}]")
        elif not area and self.in_avoid_area:
            print(f"[sim] AVOID AREA: robot left {self.in_avoid_area} after {t - self.avoid_entered_at:.1f}s")
        self.in_avoid_area = area

    def handle_hotkey(self, key_name: str) -> None:
        """Process a single-character hotkey or scripted event command."""
        k = key_name.upper()
        if k in ("B", "1"):
            self._start_sound("doorbell")
            self._stop_sound("knock")
            print(f"[sim] HOTKEY [B]: Doorbell ringing at the front door {SOUND_SOURCES['doorbell']['pos'][:2]}!")
        elif k in ("K", "2"):
            self._start_sound("knock")
            self._stop_sound("doorbell")
            print(f"[sim] HOTKEY [K]: Knocking on the front door {SOUND_SOURCES['knock']['pos'][:2]}!")
        elif k in ("M", "3"):
            self._start_sound("smoke_alarm")
            print(f"[sim] HOTKEY [M]: EMERGENCY Smoke Alarm triggered in the kitchen {SOUND_SOURCES['smoke_alarm']['pos'][:2]}!")
        elif k in ("J", "7"):
            self._start_sound("kettle")
            print(f"[sim] HOTKEY [J]: Kettle whistling on the kitchen stove {SOUND_SOURCES['kettle']['pos'][:2]}!")
        elif k in ("T", "8"):
            self._start_sound("oven_timer")
            print(f"[sim] HOTKEY [T]: Oven timer beeping on the kitchen stove {SOUND_SOURCES['oven_timer']['pos'][:2]}!")
        elif k in ("O", "4"):
            self.owner_room_idx = (self.owner_room_idx + 1) % len(ROOM_ORDER)
            new_room = ROOM_ORDER[self.owner_room_idx]
            ox, oy = ROOMS[new_room]["owner_spot"]
            self.locomotion.set_owner_pos(ox, oy)
            self.owner.on_teleport(float(self.data.time))
            print(f"[sim] HOTKEY [O]: Moved Owner to {new_room} at ({ox}, {oy})")
        elif k in ("C", "5"):
            self.chair_blocking = not self.chair_blocking
            self.chair_at_exit = False
            pos = CHAIR_DOORWAY_POS if self.chair_blocking else CHAIR_STOWED_POS
            self.locomotion.set_chair_pos(*pos)
            state_str = f"BLOCKING doorway {CHAIR_DOORWAY_POS[:2]}" if self.chair_blocking else f"STOWED at {pos[:2]}"
            print(f"[sim] HOTKEY [C]: Chair obstacle {state_str}")
        elif k == "X":
            for s in list(self.active_sounds):
                self._stop_sound(s)
            print("[sim] HOTKEY [X]: Silenced all active sounds.")
        elif k == "R":
            self.reset_scene()
        elif k == "W":
            on = self.owner.toggle_wander()
            print(f"[sim] HOTKEY [W]: Owner wandering {'ON' if on else 'OFF'}")
        elif k == "E":
            # The chair falls into the back (garden) door -> evacuation must use another exit.
            self.chair_at_exit = not self.chair_at_exit
            self.chair_blocking = False
            pos = CHAIR_EXIT_POS if self.chair_at_exit else CHAIR_STOWED_POS
            self.locomotion.set_chair_pos(*pos)
            print(f"[sim] HOTKEY [E]: Chair {'BLOCKING the back exit' if self.chair_at_exit else 'stowed'} at {pos[:2]}")
        elif k in ("N", "6"):
            p = self.sensors.noise.cycle()
            print(f"[sim] HOTKEY [N]: Background noise -> {p['name']} ({p['label']})")
        elif k == "V":
            self.view_idx = (self.view_idx + 1) % len(VIEW_NAMES)
            self.view_changed = True
            print(f"[sim] HOTKEY [V]: Viewer camera -> {VIEW_NAMES[self.view_idx]}")

    def set_owner_room(self, room: str) -> None:
        """Move the owner straight to `room` (control GUI dropdown), instead of cycling with [O]."""
        if room not in ROOMS:
            print(f"[sim] GUI: unknown room {room!r}")
            return
        ox, oy = ROOMS[room]["owner_spot"]
        self.locomotion.set_owner_pos(ox, oy)
        if room in ROOM_ORDER:
            self.owner_room_idx = ROOM_ORDER.index(room)
        print(f"[sim] GUI: Moved Owner to {room} at ({ox}, {oy})")

    def handle_gui(self, payload: dict) -> None:
        """`gui` topic from the control GUI: {"action": "key"|"owner_room"|"view", ...}."""
        action = payload.get("action")
        if action == "key":
            self.handle_hotkey(str(payload.get("key", "")))
        elif action == "owner_room":
            self.set_owner_room(str(payload.get("room", "")))
        elif action == "sound":  # any registry sound by name (GUI buttons for sounds without a hotkey)
            name = str(payload.get("name", ""))
            if name in SOUND_SOURCES:
                self._start_sound(name)
                print(f"[sim] GUI: {SOUND_SOURCES[name].get('label', name)} started at {SOUND_SOURCES[name]['pos'][:2]}")
        elif action == "clock":  # contract C2: {"action": "clock", "time": "21:59"}
            try:
                self.clock.set(str(payload.get("time", "")), float(self.data.time))
                print(f"[sim] GUI: clock set to {self.clock.hhmm(float(self.data.time))} "
                      f"(day {self.clock.day(float(self.data.time))})")
            except ValueError:
                print(f"[sim] GUI: bad clock time {payload.get('time')!r} (want HH:MM)")
        elif action == "owner_state":  # overnight S4: {"action": "owner_state", "state": "asleep"}
            st = str(payload.get("state", ""))
            if st == "asleep":
                self.set_owner_room("guest")  # in bed = guest bedroom
            if self.owner.set_owner_state(st, float(self.data.time)):
                print(f"[sim] GUI: owner is now {st.upper()}")
            else:
                print(f"[sim] GUI: unknown owner state {st!r} ({'|'.join(OWNER_STATES)})")
        elif action == "chair":  # #19 repro: {"pos": [x, y]} / {"x": .., "y": ..} / {"ahead": m} (in front of the robot)
            try:
                if "ahead" in payload:  # e.g. dropped mid-guide right in front of the robot
                    rx, ry, ryaw = self.locomotion.get_pose()
                    d = float(payload.get("ahead", 0.8))
                    x, y = rx + d * math.cos(ryaw), ry + d * math.sin(ryaw)
                elif "pos" in payload:
                    x, y = (float(v) for v in payload.get("pos", ())[:2])
                else:
                    x, y = float(payload.get("x", CHAIR_STOWED_POS[0])), float(payload.get("y", CHAIR_STOWED_POS[1]))
            except (TypeError, ValueError):
                print(f"[sim] GUI: bad chair payload {payload!r} (want pos [x, y], x/y or ahead)")
            else:
                self.chair_blocking = self.chair_at_exit = False
                self.locomotion.set_chair_pos(x, y)
                print(f"[sim] GUI: Chair placed at ({x:.2f}, {y:.2f})")
        elif action == "owner_pos":  # test repro (#19): {"action": "owner_pos", "x": .., "y": .., "yaw": ..}
            x, y = float(payload.get("x", 0.0)), float(payload.get("y", 0.0))
            self.locomotion.set_owner_pos(x, y, float(payload.get("yaw", 0.0)))
            self.owner.on_teleport(float(self.data.time))
            print(f"[sim] GUI: Moved Owner to ({x:.2f}, {y:.2f}) in {get_room_at(x, y)}")
        elif action == "view":
            self.view_req = payload
            self.view_req_wall = time.monotonic()

    def publish_view(self, now: float) -> None:
        """Render the GUI's 3D view (free or robot-follow camera) and publish it as JPEG on `view`.
        Only while a GUI has asked for it in the last 2 s, so it costs nothing without a GUI."""
        req = self.view_req
        if req is None or time.monotonic() - self.view_req_wall > 2.0:
            return
        fps = float(req.get("fps", 25.0))
        if now - self.last_view_pub < 1.0 / max(1.0, fps):
            return
        self.last_view_pub = now
        w = int(min(1600, max(160, req.get("width", 960))))
        h = int(min(1200, max(120, req.get("height", 540))))
        r = self.view_renderer
        if r is None or r.width != w or r.height != h:
            if r is not None:
                r.close()
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            r = self.view_renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = self.view_cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mode = req.get("mode", "free")
        rx, ry, ryaw = self.locomotion.get_pose()
        if mode in ("follow", "chase"):
            cam.lookat[:] = [rx, ry, 0.3]
        else:
            cam.lookat[:] = req.get("lookat", [-6.5, 1.0, 0.0])
        cam.distance = float(req.get("distance", 21.0))
        # chase: camera sits behind the robot and turns with it (azimuth offset from the heading)
        cam.azimuth = (math.degrees(ryaw) + float(req.get("azimuth", 0.0))) if mode == "chase" \
            else float(req.get("azimuth", 90.0))
        cam.elevation = float(req.get("elevation", -62.0))
        r.update_scene(self.data, camera=cam, scene_option=self.view_opt)
        self.nudge_fx.draw(r.scene, self.data)
        rgb = r.render()
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(req.get("quality", 85))])
        if ok:
            meta = {"t": round(float(self.data.time), 3), "width": w, "height": h,
                    "episode": self.episode, "mission_status": self.mission_status,
                    "mission_text": self.mission_text}
            self.pub_sock.send_multipart([b"view", json.dumps(meta).encode("utf-8"), jpg.tobytes()])

    def reset_scene(self) -> None:
        """Put robot, owner, chair and sounds back to the demo start state; start a new episode."""
        self.episode += 1
        for s in list(self.active_sounds):
            self._stop_sound(s)
        self.owner_room_idx = 0
        ox, oy = ROOMS[ROOM_ORDER[0]]["owner_spot"]
        self.locomotion.set_owner_pos(ox, oy)
        self.owner.reset(float(self.data.time))
        self.nudge_fx.reset()
        self.clock.reset(float(self.data.time))
        self.chair_blocking = False
        self.chair_at_exit = False
        self.locomotion.set_chair_pos(*CHAIR_STOWED_POS)
        self.locomotion.gait_phase = 0.0
        self.locomotion.reset_robot_pose(*ROBOT_START_POSE)
        self.cmd_vx = self.cmd_vy = self.cmd_wz = 0.0
        self.cmd_alert_mode = "none"
        self.cmd_guiding = False
        self.mission_status = "idle"
        self.mission_text = ""
        self.in_avoid_area = None
        self.avoid_steps = 0
        self.nudge_says = ""
        self.mission_sound = None
        self.voice.last_text = ""
        print(f"[sim] HOTKEY [R]: Scene RESET (episode {self.episode}): robot in "
              f"{get_room_at(*ROBOT_START_POSE[:2])}, owner in {ROOM_ORDER[0]}, chair stowed, sounds off")

    def status_overlay(self) -> tuple:
        """(title, detail) for the viewer's on-screen mission status (title starts with the sim clock)."""
        now = self.clock.hhmm(float(self.data.time))
        if self.mission_status == "complete":
            return (f"{now}  MISSION COMPLETE", self.mission_text)
        if self.mission_status == "active":
            return (f"{now}  MISSION ACTIVE", self.mission_text)
        return (f"{now}  READY (episode {self.episode})",
                "B/K/M = sound   O = move owner   W = owner wander   C/E = chair   X = silence   R = reset"
                "   N = noise   V = view")

    def glfw_key_callback(self, keycode: int) -> None:
        """Callback invoked by `mujoco.viewer.launch_passive` on key press (viewer thread)."""
        if 32 <= keycode <= 126:
            ch = chr(keycode).upper()
            if ch in ("B", "1", "K", "2", "M", "3", "O", "4", "C", "5", "X", "R", "W", "E", "N", "6", "V", "J", "7", "T", "8"):
                self.pending_keys.append(ch)  # applied in step_once() on the main thread

    def apply_view(self, cam) -> None:
        """Point the viewer camera: dollhouse overview, chase the robot, or the dog's head camera."""
        name = VIEW_NAMES[self.view_idx]
        if name == "top":
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [-6.5, 1.0, 0.0]
            cam.distance, cam.elevation, cam.azimuth = 21.0, -62.0, 90.0
        elif name == "follow":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.sensors.base_body_id
            cam.distance, cam.elevation, cam.azimuth = 3.5, -30.0, 180.0
        else:  # dog: exactly what the head RGB-D camera (and YOLO) sees
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        self.view_changed = False

    def close_renderer(self) -> None:
        """Free the head-cam GL context (idempotent)."""
        renderer = getattr(self.sensors, "renderer", None)
        if renderer is not None:
            self.sensors.renderer = None
            try:
                renderer.close()
            except Exception:
                pass

    def _update_sound_beacons(self) -> None:
        """Pulse active sound source beacon geoms in the 3D scene."""
        t = float(self.data.time)
        for name, is_on in self.active_sounds.items():
            gid = self.beacon_geom_ids.get(name, -1)
            if gid < 0:
                continue
            spec = SOUND_SOURCES[name]
            if is_on:
                r, g, b, a = spec["rgba_active"]
                pulse = 0.65 + 0.35 * np.sin(t * 14.0)
                self.model.geom_rgba[gid] = [r, g, b, float(a * pulse)]
                self.model.geom_size[gid][0] = 0.16 + 0.04 * float(pulse)
            else:
                r, g, b, a = spec["rgba_idle"]
                self.model.geom_rgba[gid] = [r, g, b, a]
                self.model.geom_size[gid][0] = 0.12

    def poll_zmq_commands(self) -> None:
        """Read non-blocking velocity/guiding commands (`cmd`) or test events (`sim_event`)."""
        while True:
            try:
                parts = self.sub_sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            if not parts:
                continue

            if len(parts) == 1:
                raw = parts[0].decode("utf-8", errors="ignore")
                if " " in raw:
                    topic, payload_str = raw.split(" ", 1)
                else:
                    continue
            else:
                topic = parts[0].decode("utf-8", errors="ignore")
                payload_str = parts[1].decode("utf-8", errors="ignore")

            try:
                payload = json.loads(payload_str)
            except Exception:
                continue

            if topic == "cmd":
                # Drop packets the brain sent before it noticed a reset (older brains omit the field).
                if int(payload.get("episode", self.episode)) != self.episode:
                    continue
                status = str(payload.get("mission_status", self.mission_status))
                if status == "complete" and self.mission_status != "complete":
                    print(f"[sim] MISSION COMPLETE: {payload.get('mission_text', '')}")
                self.mission_status = status
                self.mission_text = str(payload.get("mission_text", ""))
                self.cmd_vx = float(payload.get("vx", 0.0))
                self.cmd_vy = float(payload.get("vy", 0.0))
                self.cmd_wz = float(payload.get("wz", 0.0))
                self.cmd_alert_mode = str(payload.get("alert_mode", "none"))
                self.cmd_guiding = bool(payload.get("guiding", False))
                self.last_cmd_time = time.monotonic()
            elif topic == "sim_event":
                key = str(payload.get("key", ""))
                if key:
                    self.handle_hotkey(key)
            elif topic == "gui":
                self.handle_gui(payload)
            elif topic == "brain":
                self.handle_brain_status(payload)

        # Safety timeout if brain disconnects for > 0.6s
        if time.monotonic() - self.last_cmd_time > 0.6:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0

    def objects_gt(self) -> list:
        if not self.coco_objects:
            return []
        if self._objects_gt is None:
            from sim.coco_standins import objects_gt
            self._objects_gt = objects_gt(self.model, self.data)
        return self._objects_gt

    def publish_telemetry(self, now: float) -> None:
        """Publish `state` (20 Hz), `camera` (10 Hz), and `audio` (2 Hz) over ZMQ."""
        pose = self.locomotion.get_pose()
        rx, ry, ryaw = pose
        owner_pos = self.locomotion.get_owner_pos()

        # 1. `state` @ 20 Hz (every 0.05s)
        if now - self.last_state_pub >= 0.05:
            self.last_state_pub = now
            lidar_ranges = self.sensors.compute_lidar(pose)
            state_msg = {
                "t": round(float(self.data.time), 3),
                "robot_pose": [round(rx, 4), round(ry, 4), round(ryaw, 4)],
                "robot_room": get_room_at(rx, ry),
                "lidar_ranges": lidar_ranges,
                "guiding_active": self.locomotion.guiding_active,
                "owner_pos_gt": [round(owner_pos[0], 3), round(owner_pos[1], 3)],
                "chair_blocking": self.chair_blocking,
                "episode": self.episode,
                "active_sounds": [s for s, on in self.active_sounds.items() if on],
                # Touch sensor on the guide handle: the owner taps it to acknowledge an alert.
                "handle_tap": self.owner.handle_tap,
                "owner_state_gt": self.owner.owner_state,  # OWNER_STATES (awake|asleep|idle|unresponsive) - judge only, like owner_pos_gt
                "owner_activity_gt": self.owner.status(),  # idle|walking|guided|alerted - judge only
                # Ground truth for avoid-area tests / GUI shading (rules themselves live in the brain).
                "in_avoid_area": self.in_avoid_area,
                "avoid_steps": self.avoid_steps,
                "avoid_areas": {k: list(v) for k, v in AVOID_AREAS.items()},
                "nudge_says": self.nudge_says,  # what the robot screen shows / speaks
                "sim_clock": self.clock.hhmm(float(self.data.time)),  # contract C2
                "sim_day": self.clock.day(float(self.data.time)),
                "objects_gt": self.objects_gt(),  # S7: COCO objects, judge only ([] without --coco-objects)
            }
            self.pub_sock.send_multipart(
                [b"state", json.dumps(state_msg).encode("utf-8")]
            )

        # 2. `camera` @ 10 Hz (every 0.10s)
        if now - self.last_cam_pub >= 0.10:
            self.last_cam_pub = now
            jpeg_bytes, depth_bytes, _rgb, _depth = self.sensors.render_head_camera(
                pose, owner_pos
            )
            cam_meta = {
                "t": round(float(self.data.time), 3),
                "width": 320,
                "height": 240,
                "fovy": 60.0,
                "robot_pose": [round(rx, 4), round(ry, 4), round(ryaw, 4)],
            }
            self.pub_sock.send_multipart(
                [
                    b"camera",
                    json.dumps(cam_meta).encode("utf-8"),
                    jpeg_bytes,
                    depth_bytes,
                ]
            )

        # 3. `audio` @ 2 Hz (every 0.50s)
        if now - self.last_audio_pub >= 0.50:
            self.last_audio_pub = now
            pcm_bytes, audio_meta = self.sensors.synthesize_stereo_audio(
                pose, self.active_sounds
            )
            audio_meta["t"] = round(float(self.data.time), 3)
            audio_meta["robot_pose"] = [round(rx, 4), round(ry, 4), round(ryaw, 4)]
            self.pub_sock.send_multipart(
                [
                    b"audio",
                    json.dumps(audio_meta).encode("utf-8"),
                    pcm_bytes,
                ]
            )

    def step_once(self, dt: float) -> None:
        """Advance one simulation step."""
        while self.pending_keys:
            self.handle_hotkey(self.pending_keys.popleft())
        self.poll_zmq_commands()
        self._update_sound_beacons()
        halo_mode = self.cmd_alert_mode
        if halo_mode == "none" and self.mission_status == "complete":
            halo_mode = "complete"
        # Light bar centre lens: the brain's mission sound while alerting (so e.g. the kettle shows
        # its own colour instead of being inferred as "knock" from 2_pulse); else inferred as before.
        bar_sound = (self.mission_sound
                     if halo_mode in ("1_pulse", "2_pulse", "continuous") and self.mission_sound in SOUND_INFO
                     else None)
        self.owner.step(float(self.data.time), dt, self.locomotion.get_pose(),
                        self.cmd_alert_mode, self.cmd_guiding)
        self.locomotion.step(
            self.cmd_vx,
            self.cmd_vy,
            self.cmd_wz,
            guiding=self.cmd_guiding,
            alert_mode=halo_mode,
            dt=dt,
            sound=bar_sound,
        )
        self.check_scenario_end()
        self.check_avoid_areas()
        # Screen tier: the brain's mission sound if it sent one, else inferred from the alert mode.
        self.screen.update(self.model, float(self.data.time),
                           self.mission_sound or ALERT_MODE_SOUND.get(self.cmd_alert_mode),
                           self.mission_status)
        self.publish_telemetry(float(self.data.time))
        # After the sensors: the bump offset is for the viewer / recording only (head camera and
        # LiDAR above always see the true kinematic pose; locomotion rewrites it next step).
        self.nudge_fx.apply(self.model, self.data, float(self.data.time), self.cmd_alert_mode,
                            self.locomotion, self.owner)
        self.publish_view(float(self.data.time))

    def close(self) -> None:
        if self.view_renderer is not None:
            self.view_renderer.close()
        self.pub_sock.close()
        self.sub_sock.close()


class RealTimePacer:
    """Keeps simulated time locked to wall-clock time.

    Sleeping after every 2 ms step loses time: each short sleep overshoots (~1.5 ms asked,
    ~1.8 ms taken on Windows), which capped the sim at ~0.8x real time even though one step
    only costs ~0.14 ms. Instead track a running deadline and sleep only when ahead of it,
    so an overshoot on one step is made up on the next. If the sim falls far behind (e.g.
    the window is being dragged), resync rather than fast-forwarding to catch up.
    """

    MAX_LAG_S = 0.25

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.t0 = time.perf_counter()
        self.steps = 0

    def wait(self) -> None:
        self.steps += 1
        now = time.perf_counter()
        ahead = self.t0 + self.steps * self.dt - now
        if ahead > 0.002:
            time.sleep(ahead - 0.001)
        elif ahead < -self.MAX_LAG_S:
            self.t0, self.steps = now, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Nudge MuJoCo Simulator & ZMQ Server")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = infinite)")
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555", help="ZMQ telemetry PUB address")
    parser.add_argument("--sub", default="tcp://127.0.0.1:5556", help="ZMQ command SUB address")
    parser.add_argument("--render-camera", action="store_true",
                        help="Headless only: render the real head camera offscreen (EGL) instead of the synthetic fallback")
    parser.add_argument("--owner-static", action="store_true",
                        help="Owner never wanders on their own (still reacts to nudges / handle); toggle live with W")
    parser.add_argument("--owner-seed", type=int, default=None, help="Seed for the owner's random behaviour")
    parser.add_argument("--no-cutaway", action="store_true",
                        help="Show the whole house (upper storeys + roof) instead of the ground-floor cutaway view")
    parser.add_argument("--no-viewer", action="store_true",
                        help="No MuJoCo window, but keep the real head camera and 3D-view rendering "
                             "(use with the control GUI: `pixi run gui`)")
    parser.add_argument("--gui-sub", default="tcp://127.0.0.1:5557", help="Control GUI command address")
    parser.add_argument("--noise", choices=NOISE_NAMES, default="quiet",
                        help="Background noise profile at start (cycle live with [N])")
    parser.add_argument("--coco-objects", action="store_true",
                        help="Experimental: add COCO-class stand-ins (bed, couch, tv) + state.objects_gt")
    parser.add_argument("--clock", default="08:00",
                        help="Sim clock start time HH:MM (state.sim_clock; [R] resets to it; GUI action 'clock' sets it)")
    parser.add_argument("--view", choices=VIEW_NAMES, default="top",
                        help="Viewer camera at start: top, follow (chase robot), dog (head camera); cycle with [V]")
    args = parser.parse_args()

    print("[sim] Building nerv-world house + Unitree Go2 + Owner + Chair + Sound Beacons...")
    sim = NudgeSimulator(
        pub_endpoint=args.pub,
        sub_endpoint=args.sub,
        enable_camera=(not args.headless) or args.render_camera,
        cutaway=not args.no_cutaway,
        gui_endpoint=args.gui_sub,
        owner_wander=not args.owner_static,
        owner_seed=args.owner_seed,
        clock=args.clock,
        coco_objects=args.coco_objects,
    )
    dt = float(sim.model.opt.timestep)
    rx, ry, ryaw = sim.locomotion.get_pose()
    ox, oy = sim.locomotion.get_owner_pos()
    print(f"[sim] Go2 started in {get_room_at(rx, ry)} at ({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f})")
    print(f"[sim] Owner started in {get_room_at(ox, oy)} at ({ox:.2f}, {oy:.2f})")
    print("[sim] Hotkeys: [B/1] Doorbell | [K/2] Knock | [M/3] Smoke Alarm | [J/7] Kettle | [T/8] Oven timer | [O/4] Move Owner | [C/5] Chair | [X] Silence | [R] Reset")
    print("[sim]          [N/6] Background noise | [V] Camera view (top / follow / dog)")
    p = sim.sensors.noise.set(args.noise)
    sim.view_idx = VIEW_NAMES.index(args.view)
    print(f"[sim] Background noise: {p['name']} ({p['label']}) | view: {args.view}")

    if args.no_viewer:
        print("[sim] No MuJoCo window: open the control GUI with `pixi run gui` (Ctrl+C here to stop)")

    try:
        if args.headless or args.no_viewer:
            pacer = RealTimePacer(dt)
            start_wall = time.perf_counter()
            while True:
                sim.step_once(dt)
                if args.duration > 0.0 and (time.perf_counter() - start_wall) >= args.duration:
                    break
                pacer.wait()
        else:
            threads_before = set(threading.enumerate())
            with mujoco.viewer.launch_passive(
                sim.model, sim.data, key_callback=sim.glfw_key_callback
            ) as viewer:
                with viewer.lock():
                    # Dollhouse view of the west wing (entry, guest, living, dining) with ceilings hidden
                    if not args.no_cutaway:
                        viewer.opt.geomgroup[1] = 0
                    sim.apply_view(viewer.cam)

                start_wall = time.perf_counter()
                last_sync = 0.0
                shown_overlay = None
                pacer = RealTimePacer(dt)
                while viewer.is_running():
                    # The passive viewer renders model/data on its own thread; mutating them
                    # (physics, owner mocap, beacon geom_rgba, head-cam render) without the lock
                    # races that thread -> native access violation on Windows during guiding.
                    with viewer.lock():
                        sim.step_once(dt)
                        if sim.view_changed:
                            sim.apply_view(viewer.cam)
                    now_wall = time.perf_counter()
                    if now_wall - last_sync >= (1.0 / 60.0):
                        overlay = sim.status_overlay() + (sim.nudge_says,)
                        if overlay != shown_overlay:
                            texts = [(mujoco.mjtFontScale.mjFONTSCALE_200,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT, overlay[0], overlay[1])]
                            if sim.nudge_says:  # robot output: what Nudge tells the owner
                                texts.append((mujoco.mjtFontScale.mjFONTSCALE_200,
                                              mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "Nudge says:", sim.nudge_says))
                            viewer.set_texts(texts)
                            shown_overlay = overlay
                        with viewer.lock():  # floating "Nudge says" label above the robot
                            sim.screen.draw_label(viewer.user_scn, sim.locomotion.get_pose(), sim.nudge_says)
                            sim.nudge_fx.draw(viewer.user_scn, sim.data)  # visible nudge (after the label reset)
                        viewer.sync()
                        last_sync = now_wall
                    if args.duration > 0.0 and (now_wall - start_wall) >= args.duration:
                        break
                    pacer.wait()
                # The head-cam offscreen Renderer shares the GLFW library with the viewer; release
                # it while GLFW is still alive, before the viewer tears GLFW down.
                with viewer.lock():
                    sim.close_renderer()
            # Leaving the `with` only *signals* the (daemon) viewer thread to close. If we return
            # now, interpreter shutdown runs glfw.terminate() (atexit) while that thread is still
            # destroying its window -> access violation on exit on Windows. Wait for it first.
            for t in set(threading.enumerate()) - threads_before:
                t.join(timeout=5.0)
    finally:
        sim.close_renderer()
        sim.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
