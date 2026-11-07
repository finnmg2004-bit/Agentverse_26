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
  - ZeroMQ PUB on tcp://127.0.0.1:5555 (`state` @ 20 Hz, `camera` @ 10 Hz, `audio` @ 2 Hz)
  - ZeroMQ SUB on tcp://127.0.0.1:5556 (`cmd` @ 20 Hz + optional `sim_event` for scripted tests)
  - Mission status overlay in the viewer (from the brain's `mission_status` in `cmd`); the
    robot's halo turns steady green when a mission is complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict


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
import numpy as np
import zmq

# Ensure project root is on sys.path when invoked as `python sim/run_sim.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.house_layout import (
    CHAIR_DOORWAY_POS,
    CHAIR_STOWED_POS,
    ROBOT_START_POSE,
    ROOMS,
    ROOM_ORDER,
    SOUND_SOURCES,
    get_room_at,
)
from sim.locomotion import Go2Locomotion
from sim.nav_map import load_or_build
from sim.sensors import SimSensors
from sim.world import build_nudge_model


class NudgeSimulator:
    """Manages MuJoCo state, hotkeys, sensors, and ZeroMQ PUB/SUB sockets."""

    def __init__(
        self,
        pub_endpoint: str = "tcp://127.0.0.1:5555",
        sub_endpoint: str = "tcp://127.0.0.1:5556",
        enable_camera: bool = True,
        cutaway: bool = True,
    ) -> None:
        self.model, self.data = build_nudge_model(verbose=True, cutaway=cutaway)
        self.nav_map = load_or_build(self.model, self.data)

        self.locomotion = Go2Locomotion(self.model, self.data, self.nav_map)
        self.sensors = SimSensors(self.model, self.data, enable_camera=enable_camera)

        # Scene state controlled by hotkeys
        self.owner_room_idx: int = 0  # Starts in living_room
        self.chair_blocking: bool = False
        self.active_sounds: Dict[str, bool] = {
            "doorbell": False,
            "knock": False,
            "smoke_alarm": False,
        }
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

        # Publication timers
        self.last_state_pub: float = 0.0
        self.last_cam_pub: float = 0.0
        self.last_audio_pub: float = 0.0

    def handle_hotkey(self, key_name: str) -> None:
        """Process a single-character hotkey or scripted event command."""
        k = key_name.upper()
        if k in ("B", "1"):
            self.active_sounds["doorbell"] = True
            self.active_sounds["knock"] = False
            print(f"[sim] HOTKEY [B]: Doorbell ringing at the front door {SOUND_SOURCES['doorbell']['pos'][:2]}!")
        elif k in ("K", "2"):
            self.active_sounds["knock"] = True
            self.active_sounds["doorbell"] = False
            print(f"[sim] HOTKEY [K]: Knocking on the front door {SOUND_SOURCES['knock']['pos'][:2]}!")
        elif k in ("M", "3"):
            self.active_sounds["smoke_alarm"] = True
            print(f"[sim] HOTKEY [M]: EMERGENCY Smoke Alarm triggered in the kitchen {SOUND_SOURCES['smoke_alarm']['pos'][:2]}!")
        elif k in ("O", "4"):
            self.owner_room_idx = (self.owner_room_idx + 1) % len(ROOM_ORDER)
            new_room = ROOM_ORDER[self.owner_room_idx]
            ox, oy = ROOMS[new_room]["owner_spot"]
            self.locomotion.set_owner_pos(ox, oy)
            print(f"[sim] HOTKEY [O]: Moved Owner to {new_room} at ({ox}, {oy})")
        elif k in ("C", "5"):
            self.chair_blocking = not self.chair_blocking
            pos = CHAIR_DOORWAY_POS if self.chair_blocking else CHAIR_STOWED_POS
            self.locomotion.set_chair_pos(*pos)
            state_str = f"BLOCKING doorway {CHAIR_DOORWAY_POS[:2]}" if self.chair_blocking else f"STOWED at {pos[:2]}"
            print(f"[sim] HOTKEY [C]: Chair obstacle {state_str}")
        elif k == "X":
            for s in self.active_sounds:
                self.active_sounds[s] = False
            print("[sim] HOTKEY [X]: Silenced all active sounds.")
        elif k == "R":
            self.reset_scene()

    def reset_scene(self) -> None:
        """Put robot, owner, chair and sounds back to the demo start state; start a new episode."""
        self.episode += 1
        for s in self.active_sounds:
            self.active_sounds[s] = False
        self.owner_room_idx = 0
        ox, oy = ROOMS[ROOM_ORDER[0]]["owner_spot"]
        self.locomotion.set_owner_pos(ox, oy)
        self.chair_blocking = False
        self.locomotion.set_chair_pos(*CHAIR_STOWED_POS)
        self.locomotion.gait_phase = 0.0
        self.locomotion.reset_robot_pose(*ROBOT_START_POSE)
        self.cmd_vx = self.cmd_vy = self.cmd_wz = 0.0
        self.cmd_alert_mode = "none"
        self.cmd_guiding = False
        self.mission_status = "idle"
        self.mission_text = ""
        print(f"[sim] HOTKEY [R]: Scene RESET (episode {self.episode}): robot in "
              f"{get_room_at(*ROBOT_START_POSE[:2])}, owner in {ROOM_ORDER[0]}, chair stowed, sounds off")

    def status_overlay(self) -> tuple:
        """(title, detail) for the viewer's on-screen mission status."""
        if self.mission_status == "complete":
            return ("MISSION COMPLETE", self.mission_text)
        if self.mission_status == "active":
            return ("MISSION ACTIVE", self.mission_text)
        return (f"READY (episode {self.episode})", "B/K/M = sound   O = move owner   C = chair   X = silence   R = reset")

    def glfw_key_callback(self, keycode: int) -> None:
        """Callback invoked by `mujoco.viewer.launch_passive` on key press."""
        if 32 <= keycode <= 126:
            ch = chr(keycode).upper()
            if ch in ("B", "1", "K", "2", "M", "3", "O", "4", "C", "5", "X", "R"):
                self.handle_hotkey(ch)

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

        # Safety timeout if brain disconnects for > 0.6s
        if time.monotonic() - self.last_cmd_time > 0.6:
            self.cmd_vx = 0.0
            self.cmd_vy = 0.0
            self.cmd_wz = 0.0

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
        self.poll_zmq_commands()
        self._update_sound_beacons()
        halo_mode = self.cmd_alert_mode
        if halo_mode == "none" and self.mission_status == "complete":
            halo_mode = "complete"
        self.locomotion.step(
            self.cmd_vx,
            self.cmd_vy,
            self.cmd_wz,
            guiding=self.cmd_guiding,
            alert_mode=halo_mode,
            dt=dt,
        )
        self.publish_telemetry(float(self.data.time))

    def close(self) -> None:
        self.pub_sock.close()
        self.sub_sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nudge MuJoCo Simulator & ZMQ Server")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = infinite)")
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555", help="ZMQ telemetry PUB address")
    parser.add_argument("--sub", default="tcp://127.0.0.1:5556", help="ZMQ command SUB address")
    parser.add_argument("--render-camera", action="store_true",
                        help="Headless only: render the real head camera offscreen (EGL) instead of the synthetic fallback")
    parser.add_argument("--no-cutaway", action="store_true",
                        help="Show the whole house (upper storeys + roof) instead of the ground-floor cutaway view")
    args = parser.parse_args()

    print("[sim] Building nerv-world house + Unitree Go2 + Owner + Chair + Sound Beacons...")
    sim = NudgeSimulator(
        pub_endpoint=args.pub,
        sub_endpoint=args.sub,
        enable_camera=(not args.headless) or args.render_camera,
        cutaway=not args.no_cutaway,
    )
    dt = float(sim.model.opt.timestep)
    rx, ry, ryaw = sim.locomotion.get_pose()
    ox, oy = sim.locomotion.get_owner_pos()
    print(f"[sim] Go2 started in {get_room_at(rx, ry)} at ({rx:.2f}, {ry:.2f}, yaw={ryaw:.2f})")
    print(f"[sim] Owner started in {get_room_at(ox, oy)} at ({ox:.2f}, {oy:.2f})")
    print("[sim] Hotkeys: [B/1] Doorbell | [K/2] Knock | [M/3] Smoke Alarm | [O/4] Move Owner | [C/5] Chair | [X] Silence | [R] Reset")

    try:
        if args.headless:
            start_wall = time.perf_counter()
            while True:
                step_start = time.perf_counter()
                sim.step_once(dt)
                if args.duration > 0.0 and (time.perf_counter() - start_wall) >= args.duration:
                    break
                elapsed = time.perf_counter() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        else:
            with mujoco.viewer.launch_passive(
                sim.model, sim.data, key_callback=sim.glfw_key_callback
            ) as viewer:
                # Dollhouse view of the west wing (entry, guest, living, dining) with ceilings hidden
                if not args.no_cutaway:
                    viewer.opt.geomgroup[1] = 0
                viewer.cam.lookat[:] = [-6.5, 1.0, 0.0]
                viewer.cam.distance = 21.0
                viewer.cam.elevation = -62.0
                viewer.cam.azimuth = 90.0

                start_wall = time.perf_counter()
                last_sync = 0.0
                shown_overlay = None
                while viewer.is_running():
                    step_start = time.perf_counter()
                    sim.step_once(dt)
                    now_wall = time.perf_counter()
                    if now_wall - last_sync >= (1.0 / 60.0):
                        overlay = sim.status_overlay()
                        if overlay != shown_overlay:
                            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_200,
                                              mujoco.mjtGridPos.mjGRID_TOPLEFT, overlay[0], overlay[1]))
                            shown_overlay = overlay
                        viewer.sync()
                        last_sync = now_wall
                    if args.duration > 0.0 and (now_wall - start_wall) >= args.duration:
                        break
                    elapsed = time.perf_counter() - step_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
    finally:
        sim.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
